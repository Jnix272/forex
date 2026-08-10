"""
Async Event Bus for Training Observability.

Priority-queued, async event processing with deduplication, persistence, and backpressure.
"""

from __future__ import annotations

import asyncio
import aiosqlite
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from monitoring.events import TrainingEvent, EventType, Severity, validate_payload


class HandlerPriority(int, Enum):
    """Handler execution priority - lower runs first."""
    CRITICAL = 0    # Alert dispatch, crash handlers
    HIGH = 10       # Checkpoint saves, critical checks
    NORMAL = 50     # Standard logging, metrics
    LOW = 100       # Debug, heartbeats, progress


@dataclass
class EventHandler:
    """Registered event handler with metadata."""
    func: Callable[[TrainingEvent], Any]
    event_types: set[EventType] = field(default_factory=set)
    min_severity: Severity = Severity.DEBUG
    priority: HandlerPriority = HandlerPriority.NORMAL
    tags: set[str] = field(default_factory=set)
    async_fn: bool = False
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = self.func.__name__
    
    def matches(self, event: TrainingEvent) -> bool:
        """Check if handler should process this event."""
        if self.event_types and event.event_type not in self.event_types:
            return False
        if event.severity.value < self.min_severity.value:
            return False
        if self.tags and not self.tags.intersection(event.tags):
            return False
        return True


@dataclass
class DeduplicationEntry:
    """Deduplication tracking for repeated events."""
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_emitted: float = 0


class EventBus:
    """
    Async priority event bus with deduplication, persistence, and backpressure.
    
    Features:
    - Priority queue (CRITICAL > HIGH > NORMAL > LOW)
    - Per-event-type deduplication with configurable window
    - SQLite persistence for replay/debugging
    - Backpressure handling with queue size limits
    - Handler registration with filters
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        persistence_path: str = "logs/events.db",
        dedup_window_sec: float = 60.0,
        dedup_max_count: int = 100,
        persistence_enabled: bool = True,
    ):
        self.max_queue_size = max_queue_size
        self.persistence_path = Path(persistence_path)
        self.dedup_window_sec = dedup_window_sec
        self.dedup_max_count = dedup_max_count
        self.persistence_enabled = persistence_enabled
        
        # Priority queue: (priority, timestamp, event)
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        
        # Handlers: event_type -> list of handlers
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._all_handlers: list[EventHandler] = []
        
        # Deduplication
        self._dedup: dict[str, DeduplicationEntry] = {}
        self._dedup_lock = asyncio.Lock()
        
        # Persistence
        self._db: Optional[aiosqlite.Connection] = None
        self._persist_task: Optional[asyncio.Task] = None
        self._persist_batch: list[TrainingEvent] = []
        self._persist_batch_size = 100
        
        # Metrics
        self._stats = {
            "enqueued": 0,
            "processed": 0,
            "dropped": 0,
            "deduplicated": 0,
            "handler_errors": 0,
        }
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the event bus worker and persistence."""
        self._running = True
        
        # Initialize database
        if self.persistence_enabled:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self.persistence_path)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT,
                    run_id TEXT,
                    session_id TEXT,
                    epoch INTEGER,
                    batch INTEGER,
                    model_name TEXT,
                    payload TEXT,
                    tags TEXT,
                    parent_event_id TEXT,
                    correlation_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)
            """)
            await self._db.commit()
            self._persist_task = asyncio.create_task(self._persist_loop())
        
        # Start worker
        self._worker_task = asyncio.create_task(self._worker_loop())
    
    async def stop(self):
        """Stop the event bus gracefully."""
        self._running = False
        
        # Wait for queue to drain
        while not self._queue.empty():
            await asyncio.sleep(0.1)
        
        # Cancel tasks
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        if self._persist_task:
            self._persist_task.cancel()
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining persist batch
        if self._persist_batch and self._db:
            await self._flush_persist_batch()
        
        if self._db:
            await self._db.close()
    
    def register_handler(
        self,
        func: Callable[[TrainingEvent], Any],
        event_types: Optional[set[EventType]] = None,
        min_severity: Severity = Severity.DEBUG,
        priority: HandlerPriority = HandlerPriority.NORMAL,
        tags: Optional[set[str]] = None,
    ) -> EventHandler:
        """Register an event handler."""
        handler = EventHandler(
            func=func,
            event_types=event_types or set(),
            min_severity=min_severity,
            priority=priority,
            tags=tags or set(),
            async_fn=asyncio.iscoroutinefunction(func),
        )
        self._all_handlers.append(handler)
        if event_types:
            for et in event_types:
                self._handlers[et].append(handler)
        else:
            for et in EventType:
                self._handlers[et].append(handler)
        # Sort by priority
        for et in self._handlers:
            self._handlers[et].sort(key=lambda h: h.priority)
        return handler
    
    def unregister_handler(self, handler: EventHandler):
        """Unregister an event handler."""
        if handler in self._all_handlers:
            self._all_handlers.remove(handler)
        for et, handlers in self._handlers.items():
            if handler in handlers:
                handlers.remove(handler)
    
    async def emit(self, event: TrainingEvent, force: bool = False) -> bool:
        """
        Emit an event to the bus.
        
        Returns True if enqueued, False if dropped (queue full or deduplicated).
        """
        # Validate payload
        valid, errors = validate_payload(event.event_type, event.payload)
        if not valid:
            # Log validation error but still emit
            pass
        
        # Deduplication check
        if not force:
            dedup_key = f"{event.source}:{event.event_type.value}:{event.payload.get('name', event.payload.get('message', ''))}"
            async with self._dedup_lock:
                now = time.time()
                entry = self._dedup.get(dedup_key)
                if entry:
                    # Clean old entries
                    if now - entry.first_seen > self.dedup_window_sec:
                        self._dedup.pop(dedup_key, None)
                    else:
                        entry.count += 1
                        entry.last_seen = now
                        if entry.count <= self.dedup_max_count:
                            self._stats["deduplicated"] += 1
                            return False  # Deduplicated
                        # Allow through after max count
        
        # Enqueue with priority
        priority = self._get_event_priority(event)
        timestamp = time.time()
        try:
            self._queue.put_nowait((priority, timestamp, event))
            self._stats["enqueued"] += 1
            return True
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
            return False
    
    def _get_event_priority(self, event: TrainingEvent) -> int:
        """Determine priority from event type and severity."""
        base = {
            EventType.ALERT: HandlerPriority.CRITICAL,
            EventType.CHECKPOINT: HandlerPriority.HIGH,
            EventType.CHECK: HandlerPriority.NORMAL,
            EventType.METRIC: HandlerPriority.NORMAL,
            EventType.LOG: HandlerPriority.NORMAL,
            EventType.PROGRESS: HandlerPriority.LOW,
            EventType.HEARTBEAT: HandlerPriority.LOW,
        }.get(event.event_type, HandlerPriority.NORMAL)
        
        # Boost for higher severity
        severity_boost = {
            Severity.CRITICAL: -20,
            Severity.ERROR: -10,
            Severity.WARNING: -5,
            Severity.INFO: 0,
            Severity.DEBUG: 5,
        }.get(event.severity, 0)
        
        return base + severity_boost
    
    async def _worker_loop(self):
        """Main event processing loop."""
        while self._running:
            try:
                # Get next event with timeout
                try:
                    priority, timestamp, event = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process event
                await self._process_event(event)
                self._stats["processed"] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats["handler_errors"] += 1
                # Log error but continue
                print(f"[EventBus] Worker error: {e}")
    
    async def _process_event(self, event: TrainingEvent):
        """Process event through all matching handlers."""
        # Determine handlers to call
        handlers = self._handlers.get(event.event_type, [])
        
        for handler in handlers:
            if not handler.matches(event):
                continue
            try:
                if handler.async_fn:
                    await handler.func(event)
                else:
                    handler.func(event)
            except Exception as e:
                self._stats["handler_errors"] += 1
                print(f"[EventBus] Handler {handler.name} error: {e}")
        
        # Add to persistence batch after processing
        if self.persistence_enabled:
            self._persist_batch.append(event)
    
    async def _persist_loop(self):
        """Periodic persistence to SQLite."""
        while self._running:
            await asyncio.sleep(5.0)  # Flush every 5 seconds
            if self._persist_batch:
                await self._flush_persist_batch()
    
    async def _flush_persist_batch(self):
        """Flush pending events to database."""
        if not self._db or not self._persist_batch:
            return
        
        batch = self._persist_batch[:self._persist_batch_size]
        self._persist_batch = self._persist_batch[self._persist_batch_size:]
        
        try:
            await self._db.executemany("""
                INSERT OR REPLACE INTO events 
                (event_id, timestamp, event_type, severity, source, run_id, session_id,
                 epoch, batch, model_name, payload, tags, parent_event_id, correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    e.event_id, e.timestamp, e.event_type.value, e.severity.value,
                    e.source, e.run_id, e.session_id, e.epoch, e.batch, e.model_name,
                    json.dumps(e.payload), json.dumps(e.tags),
                    e.parent_event_id, e.correlation_id
                )
                for e in batch
            ])
            await self._db.commit()
        except Exception as e:
            print(f"[EventBus] Persist error: {e}")
            # Re-add to batch for retry
            self._persist_batch = batch + self._persist_batch
    
    def get_stats(self) -> dict:
        """Get event bus statistics."""
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "handlers_registered": len(self._all_handlers),
            "dedup_entries": len(self._dedup),
        }
    
    async def query_events(
        self,
        run_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        severity: Optional[Severity] = None,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> list[TrainingEvent]:
        """Query persisted events."""
        if not self._db:
            return []
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        
        events = []
        for row in rows:
            event = TrainingEvent(
                event_id=row[0],
                timestamp=row[1],
                event_type=EventType(row[2]),
                severity=Severity(row[3]),
                source=row[4] or "",
                run_id=row[5] or "",
                session_id=row[6] or "",
                epoch=row[7],
                batch=row[8],
                model_name=row[9] or "",
                payload=json.loads(row[10]) if row[10] else {},
                tags=json.loads(row[11]) if row[11] else [],
                parent_event_id=row[12],
                correlation_id=row[13],
            )
            events.append(event)
        return events


# Global event bus instance
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create global event bus."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


async def init_event_bus(**kwargs) -> EventBus:
    """Initialize and start global event bus."""
    global _global_bus
    _global_bus = EventBus(**kwargs)
    await _global_bus.start()
    return _global_bus


async def shutdown_event_bus():
    """Shutdown global event bus."""
    global _global_bus
    if _global_bus:
        await _global_bus.stop()
        _global_bus = None