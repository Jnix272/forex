"""
Unified Logger - Single entry point for ALL logging.

Replaces: train_logger.py, sidecar.py, logging_utils.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from monitoring.events import TrainingEvent, EventType, Severity
from monitoring.event_bus import EventBus, get_event_bus, init_event_bus, shutdown_event_bus


@dataclass
class UnifiedLoggerConfig:
    """Configuration for UnifiedLogger."""
    log_dir: str = "logs"
    jsonl_rotation_mb: int = 100
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    jsonl_enabled: bool = True
    console_enabled: bool = True
    structured_console: bool = False  # JSON lines to console
    event_bus_queue_size: int = 10000
    dedup_window_sec: float = 60.0
    persistence_enabled: bool = True
    run_id: str = ""
    session_id: str = ""


class UnifiedLogger:
    """
    Single entry point for ALL logging/checking/alerting/metrics.
    
    Replaces: TrainingLogger, Sidecar, logging_utils
    
    Usage:
        logger = UnifiedLogger.from_config(args)
        
        # Simple logging
        logger.info("Training started", model_name="haelt")
        
        # Metrics
        logger.metric("train_loss", 0.123, epoch=1, model_name="haelt")
        
        # Checks
        logger.check("grad_norm", passed=True, value=5.2, threshold=10.0, 
                     epoch=1, model_name="haelt")
        
        # Alerts
        logger.alert("nan_storm", "10 NaN gradients in 60s", severity="critical")
        
        # Checkpoints
        logger.checkpoint("checkpoints/haelt_best.pt", {"val_sharpe": 1.8}, epoch=10)
    """
    
    _instance: Optional["UnifiedLogger"] = None
    _lock = threading.Lock()
    
    def __init__(self, config: UnifiedLoggerConfig):
        self.config = config
        self._bus: Optional[EventBus] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        
        # File handles for backward compatibility
        self._jsonl_file = None
        self._log_file = None
        
        # Run ID
        self.run_id = config.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.session_id = config.session_id or f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Setup paths
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console logger (standard library)
        self._console_logger = logging.getLogger(f"unified.{self.run_id}")
        self._console_logger.setLevel(getattr(logging, config.console_level.upper()))
        self._console_logger.propagate = False
        
        # Metrics cache for web dashboard
        self._metrics_cache: dict[str, list[dict]] = defaultdict(list)
        self._max_metrics_cache = 10000
    
    @classmethod
    def from_config(cls, args_or_config: Any) -> "UnifiedLogger":
        """Create logger from argparse.Namespace or UnifiedLoggerConfig."""
        if isinstance(args_or_config, UnifiedLoggerConfig):
            config = args_or_config
        else:
            # Build from argparse.Namespace
            config = UnifiedLoggerConfig(
                log_dir=getattr(args_or_config, "log_dir", "logs"),
                jsonl_rotation_mb=getattr(args_or_config, "jsonl_rotation_mb", 100),
                console_level=getattr(args_or_config, "console_log_level", "INFO"),
                file_level=getattr(args_or_config, "file_log_level", "DEBUG"),
                jsonl_enabled=getattr(args_or_config, "jsonl_enabled", True),
                console_enabled=getattr(args_or_config, "console_enabled", True),
                event_bus_queue_size=getattr(args_or_config, "event_bus_queue_size", 10000),
                dedup_window_sec=getattr(args_or_config, "dedup_window_sec", 60.0),
                persistence_enabled=getattr(args_or_config, "persistence_enabled", True),
                run_id=getattr(args_or_config, "run_name", "") or getattr(args_or_config, "run_id", ""),
                session_id=getattr(args_or_config, "session_id", ""),
            )
        
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            else:
                # Update existing instance config
                cls._instance.config = config
            return cls._instance
    
    @classmethod
    def get_instance(cls) -> Optional["UnifiedLogger"]:
        """Get existing instance."""
        return cls._instance
    
    async def start(self):
        """Start the logger (async initialization)."""
        if self._started:
            return
        
        # Create event loop in background thread
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True, name="UnifiedLoggerLoop")
        self._thread.start()
        
        # Wait for loop to be ready
        while self._loop is None:
            await asyncio.sleep(0.01)
        
        # Initialize event bus in the background loop
        future = asyncio.run_coroutine_threadsafe(
            init_event_bus(
                max_queue_size=self.config.event_bus_queue_size,
                persistence_path=str(self.log_dir / "events.db"),
                dedup_window_sec=self.config.dedup_window_sec,
                persistence_enabled=self.config.persistence_enabled,
            ),
            self._loop
        )
        self._bus = future.result(timeout=10.0)
        
        # Setup file handles
        if self.config.jsonl_enabled:
            self._setup_files()
        
        # Register default handlers
        self._register_default_handlers()
        
        self._started = True
        self.info("UnifiedLogger started", run_id=self.run_id, session_id=self.session_id)
    
    def _setup_files(self):
        """Setup file handles for backward compatibility."""
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        base_name = f"train_{self.run_id}_{timestamp}"
        
        # JSONL file (structured events)
        jsonl_path = self.log_dir / f"{base_name}.jsonl"
        self._jsonl_file = open(jsonl_path, "a", encoding="utf-8")
        
        # Human-readable log file
        log_path = self.log_dir / f"{base_name}.log"
        self._log_file = open(log_path, "a", encoding="utf-8")
        
        # Setup console handler if not already
        if self.config.console_enabled and not self._console_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            if self.config.structured_console:
                handler.setFormatter(logging.Formatter("%(message)s"))
            else:
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S"
                ))
            self._console_logger.addHandler(handler)
    
    def _register_default_handlers(self):
        """Register default event handlers."""
        # JSONL file writer
        self._bus.register_handler(
            self._write_jsonl,
            min_severity=Severity.DEBUG,
            priority=HandlerPriority.NORMAL,
        )
        
        # Human-readable log file
        self._bus.register_handler(
            self._write_log,
            min_severity=Severity.INFO,
            priority=HandlerPriority.LOW,
        )
        
        # Console output
        self._bus.register_handler(
            self._write_console,
            event_types={EventType.LOG, EventType.ALERT, EventType.CHECKPOINT, EventType.PROGRESS},
            min_severity=Severity.INFO,
            priority=HandlerPriority.NORMAL,
        )
        
        # Metrics cache for web dashboard
        self._bus.register_handler(
            self._cache_metric,
            event_types={EventType.METRIC},
            priority=HandlerPriority.NORMAL,
        )
    
    def _write_jsonl(self, event: TrainingEvent):
        """Write event to JSONL file."""
        if self._jsonl_file:
            try:
                import json
                self._jsonl_file.write(json.dumps(event.to_dict()) + "\n")
                self._jsonl_file.flush()
            except Exception:
                pass
    
    def _write_log(self, event: TrainingEvent):
        """Write human-readable log entry."""
        if self._log_file:
            try:
                # Format: [timestamp] [severity] [source] message
                ts = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')).strftime("%H:%M:%S")
                msg = event.payload.get("message", "")
                if not msg and event.event_type == EventType.CHECK:
                    name = event.payload.get("name", "check")
                    passed = event.payload.get("passed", False)
                    msg = f"CHECK {name}: {'PASS' if passed else 'FAIL'}"
                elif not msg and event.event_type == EventType.METRIC:
                    name = event.payload.get("name", "")
                    value = event.payload.get("value", 0)
                    msg = f"METRIC {name}={value}"
                
                line = f"[{ts}] [{event.severity.value.upper():7s}] [{event.source}] {msg}\n"
                self._log_file.write(line)
                self._log_file.flush()
            except Exception:
                pass
    
    def _write_console(self, event: TrainingEvent):
        """Write to console via standard logger."""
        level = getattr(logging, event.severity.value.upper())
        
        if event.event_type == EventType.LOG:
            msg = event.payload.get("message", "")
        elif event.event_type == EventType.ALERT:
            msg = f"🚨 ALERT [{event.payload.get('rule', '')}]: {event.payload.get('message', '')}"
        elif event.event_type == EventType.CHECKPOINT:
            msg = f"💾 Checkpoint saved: {event.payload.get('path', '')}"
        elif event.event_type == EventType.PROGRESS:
            msg = f"📊 {event.payload.get('phase', '')}: {event.payload.get('current', 0)}/{event.payload.get('total', 0)} ({event.payload.get('pct', 0):.1f}%)"
        else:
            msg = str(event.payload)
        
        # Add context
        ctx = []
        if event.epoch is not None:
            ctx.append(f"ep={event.epoch}")
        if event.batch is not None:
            ctx.append(f"batch={event.batch}")
        if event.model_name:
            ctx.append(f"model={event.model_name}")
        if ctx:
            msg = f"[{' '.join(ctx)}] {msg}"
        
        self._console_logger.log(level, msg)
    
    def _cache_metric(self, event: TrainingEvent):
        """Cache metric for web dashboard."""
        if event.event_type != EventType.METRIC:
            return
        
        name = event.payload.get("name", "")
        if not name:
            return
        
        entry = {
            "timestamp": event.timestamp,
            "value": event.payload.get("value"),
            "epoch": event.epoch,
            "batch": event.batch,
            "model": event.model_name,
            "unit": event.payload.get("unit", ""),
        }
        
        self._metrics_cache[name].append(entry)
        # Trim cache
        if len(self._metrics_cache[name]) > self._max_metrics_cache:
            self._metrics_cache[name] = self._metrics_cache[name][-self._max_metrics_cache:]
    
    # Sync logging methods (for non-async code)
    def _emit_sync(self, event: TrainingEvent):
        """Emit event synchronously (runs in background thread)."""
        if not self._bus or not self._loop:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._bus.emit(event), self._loop)
            future.result(timeout=1.0)
        except Exception:
            pass
    
    # Public logging API
    def log(self, message: str, severity: str = "info", source: str = "", 
            tags: list[str] = None, **kwargs) -> TrainingEvent:
        """Log a message."""
        event = TrainingEvent.log(
            message=message,
            source=source or "unified_logger",
            severity=Severity(severity.lower()),
            run_id=kwargs.pop("run_id", self.run_id),
            tags=tags or [],
            **kwargs
        )
        self._emit_sync(event)
        return event
    
    def debug(self, message: str, source: str = "", **kwargs):
        return self.log(message, "debug", source, **kwargs)
    
    def info(self, message: str, source: str = "", **kwargs):
        return self.log(message, "info", source, **kwargs)
    
    def warning(self, message: str, source: str = "", **kwargs):
        return self.log(message, "warning", source, **kwargs)
    
    def error(self, message: str, source: str = "", **kwargs):
        return self.log(message, "error", source, **kwargs)
    
    def critical(self, message: str, source: str = "", **kwargs):
        return self.log(message, "critical", source, **kwargs)
    
    def metric(self, name: str, value: float, unit: str = "", 
               source: str = "", tags: list[str] = None, **kwargs) -> TrainingEvent:
        """Log a scalar metric."""
        event = TrainingEvent.metric(
            name=name,
            value=value,
            unit=unit,
            source=source or "unified_logger",
            run_id=self.run_id,
            tags=tags or [],
            **kwargs
        )
        self._emit_sync(event)
        return event
    
    def check(self, name: str, passed: bool, value: Any = None, 
              threshold: float = None, details: dict = None,
              source: str = "", tags: list[str] = None, **kwargs) -> TrainingEvent:
        """Log a validation check result."""
        event = TrainingEvent.check(
            name=name,
            passed=passed,
            source=source or "unified_logger",
            run_id=self.run_id,
            tags=tags or [],
            threshold=threshold,
            details=details or {"value": value},
            **kwargs
        )
        self._emit_sync(event)
        return event
    
    def alert(self, rule: str, message: str, severity: str = "warning",
              context: dict = None, source: str = "", tags: list[str] = None, **kwargs) -> TrainingEvent:
        """Emit an alert."""
        event = TrainingEvent.alert(
            rule=rule,
            message=message,
            severity=Severity(severity.lower()),
            source=source or "unified_logger",
            run_id=self.run_id,
            tags=tags or [],
            context=context or {},
            **kwargs
        )
        self._emit_sync(event)
        return event
    
    def checkpoint(self, path: str, metrics: dict, epoch: int,
                   source: str = "", tags: list[str] = None, **kwargs) -> TrainingEvent:
        """Log a model checkpoint."""
        event = TrainingEvent.checkpoint(
            path=path,
            metrics=metrics,
            epoch=epoch,
            source=source or "unified_logger",
            run_id=self.run_id,
            tags=tags or [],
            **kwargs
        )
        self._emit_sync(event)
        return event
    
    def heartbeat(self, component: str, status: str = "alive", source: str = "", **kwargs):
        """Emit a heartbeat."""
        event = TrainingEvent.heartbeat(
            component=component,
            status=status,
            source=source or "unified_logger",
            run_id=self.run_id,
            **kwargs
        )
        self._emit_sync(event)
    
    def progress(self, phase: str, current: int, total: int, eta_sec: float = 0,
                 source: str = "", **kwargs):
        """Log progress update."""
        event = TrainingEvent.progress(
            phase=phase,
            current=current,
            total=total,
            eta_sec=eta_sec,
            source=source or "unified_logger",
            run_id=self.run_id,
            **kwargs
        )
        self._emit_sync(event)
    
    # Context managers for structured logging
    @contextmanager
    def epoch_context(self, epoch: int, model_name: str = ""):
        """Context manager for epoch-scoped logging."""
        old_epoch = getattr(self, '_current_epoch', None)
        old_model = getattr(self, '_current_model', None)
        self._current_epoch = epoch
        self._current_model = model_name
        try:
            yield self
        finally:
            self._current_epoch = old_epoch
            self._current_model = old_model
    
    @contextmanager
    def batch_context(self, batch: int):
        """Context manager for batch-scoped logging."""
        old_batch = getattr(self, '_current_batch', None)
        self._current_batch = batch
        try:
            yield self
        finally:
            self._current_batch = old_batch
    
    def get_metrics(self, name: str = None) -> list[dict] | dict[str, list[dict]]:
        """Get cached metrics for web dashboard."""
        if name:
            return self._metrics_cache.get(name, [])
        return dict(self._metrics_cache)
    
    def get_stats(self) -> dict:
        """Get logger statistics."""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "started": self._started,
            "bus_stats": self._bus.get_stats() if self._bus else {},
            "metrics_cached": {k: len(v) for k, v in self._metrics_cache.items()},
        }
    
    async def stop(self):
        """Stop the logger."""
        if self._bus and self._loop:
            # Shutdown event bus in its own loop
            future = asyncio.run_coroutine_threadsafe(shutdown_event_bus(), self._loop)
            try:
                future.result(timeout=10.0)
            except Exception:
                pass
        
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread:
            self._thread.join(timeout=5.0)
        
        # Close files
        if self._jsonl_file:
            self._jsonl_file.close()
        if self._log_file:
            self._log_file.close()
        
        self._started = False
    
    # Compatibility aliases for existing code
    def log_batch(self, batch_idx: int, metrics: dict, **kwargs):
        """Log batch metrics (compatibility)."""
        for name, value in metrics.items():
            self.metric(f"batch_{name}", value, batch=batch_idx, **kwargs)
    
    def log_epoch(self, epoch: int, metrics: dict, **kwargs):
        """Log epoch metrics (compatibility)."""
        for name, value in metrics.items():
            self.metric(f"epoch_{name}", value, epoch=epoch, **kwargs)


# Backward compatibility: global instance accessor
def get_logger() -> Optional[UnifiedLogger]:
    """Get global logger instance."""
    return UnifiedLogger.get_instance()


def create_logger(config: UnifiedLoggerConfig) -> UnifiedLogger:
    """Create and start logger (async)."""
    logger = UnifiedLogger(config)
    return logger


from monitoring.event_bus import HandlerPriority