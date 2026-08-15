"""
Lineage Store
=============
Persistence layer for lineage records.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from lineage.tracker import LineageRecord, LineageEventType


class LineageStore(ABC):
    """Abstract base class for lineage stores"""
    
    @abstractmethod
    def save(self, record: LineageRecord):
        """Save a lineage record"""
        pass
    
    @abstractmethod
    def query(
        self,
        run_id: str | None = None,
        stage: str | None = None,
        pair: str | None = None,
        event_type: LineageEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[LineageRecord]:
        """Query lineage records"""
        pass
    
    @abstractmethod
    def get_by_hash(self, output_hash: str) -> LineageRecord | None:
        """Get record by output hash"""
        pass
    
    @abstractmethod
    def get_lineage_chain(self, output_hash: str) -> list[LineageRecord]:
        """Get full lineage chain for an output"""
        pass


class FileLineageStore(LineageStore):
    """File-based lineage store (JSON lines)"""
    
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def save(self, record: LineageRecord):
        with self._lock:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
    
    def _load_all(self) -> list[LineageRecord]:
        if not self.filepath.exists():
            return []
        
        records = []
        with open(self.filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(LineageRecord.from_dict(json.loads(line)))
                    except Exception:
                        continue
        return records
    
    def query(
        self,
        run_id: str | None = None,
        stage: str | None = None,
        pair: str | None = None,
        event_type: LineageEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[LineageRecord]:
        records = self._load_all()
        
        if run_id:
            records = [r for r in records if r.record_id.startswith(run_id)]
        if stage:
            records = [r for r in records if r.stage == stage]
        if pair:
            records = [r for r in records if r.pair == pair]
        if event_type:
            records = [r for r in records if r.event_type == event_type]
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]
        
        return records[:limit]
    
    def get_by_hash(self, output_hash: str) -> LineageRecord | None:
        records = self._load_all()
        for r in records:
            if r.output_hash == output_hash:
                return r
        return None
    
    def get_lineage_chain(self, output_hash: str) -> list[LineageRecord]:
        records = self._load_all()
        chain = []
        
        # Find the target record
        target = None
        for r in records:
            if r.output_hash == output_hash:
                target = r
                break
        
        if not target:
            return []
        
        chain.append(target)
        
        # Walk backwards through input hashes
        visited = {output_hash}
        to_process = target.input_hashes.copy()
        
        while to_process:
            current_hash = to_process.pop(0)
            if current_hash in visited:
                continue
            visited.add(current_hash)
            
            # Find record that produced this hash
            for r in records:
                if r.output_hash == current_hash:
                    chain.append(r)
                    for inp in r.input_hashes:
                        if inp not in visited:
                            to_process.append(inp)
                    break
        
        return list(reversed(chain))


class SQLiteLineageStore(LineageStore):
    """SQLite-based lineage store for larger scale"""
    
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage_records (
                    record_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    pair TEXT,
                    input_hashes TEXT NOT NULL,  -- JSON array
                    output_hash TEXT,
                    output_path TEXT,
                    metadata TEXT NOT NULL,  -- JSON
                    git_commit TEXT,
                    git_branch TEXT,
                    code_version TEXT,
                    config_hash TEXT,
                    duration_ms REAL,
                    records_in INTEGER,
                    records_out INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineage_run_id 
                ON lineage_records(record_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineage_stage 
                ON lineage_records(stage)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineage_pair 
                ON lineage_records(pair)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineage_output_hash 
                ON lineage_records(output_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineage_timestamp 
                ON lineage_records(timestamp)
            """)
    
    def save(self, record: LineageRecord):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO lineage_records 
                    (record_id, event_type, timestamp, stage, pair, input_hashes, 
                     output_hash, output_path, metadata, git_commit, git_branch,
                     code_version, config_hash, duration_ms, records_in, records_out)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.record_id,
                    record.event_type.value,
                    record.timestamp.isoformat(),
                    record.stage,
                    record.pair,
                    json.dumps(record.input_hashes),
                    record.output_hash,
                    record.output_path,
                    json.dumps(record.metadata),
                    record.git_commit,
                    record.git_branch,
                    record.code_version,
                    record.config_hash,
                    record.duration_ms,
                    record.records_in,
                    record.records_out,
                ))
    
    def query(
        self,
        run_id: str | None = None,
        stage: str | None = None,
        pair: str | None = None,
        event_type: LineageEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[LineageRecord]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                where_clauses = []
                params = []
                
                if run_id:
                    where_clauses.append("record_id LIKE ?")
                    params.append(f"{run_id}%")
                if stage:
                    where_clauses.append("stage = ?")
                    params.append(stage)
                if pair:
                    where_clauses.append("pair = ?")
                    params.append(pair)
                if event_type:
                    where_clauses.append("event_type = ?")
                    params.append(event_type.value)
                if start_time:
                    where_clauses.append("timestamp >= ?")
                    params.append(start_time.isoformat())
                if end_time:
                    where_clauses.append("timestamp <= ?")
                    params.append(end_time.isoformat())
                
                where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                
                query = f"""
                    SELECT * FROM lineage_records
                    {where_sql}
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                records = []
                for row in rows:
                    records.append(LineageRecord(
                        record_id=row["record_id"],
                        event_type=LineageEventType(row["event_type"]),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        stage=row["stage"],
                        pair=row["pair"],
                        input_hashes=json.loads(row["input_hashes"]),
                        output_hash=row["output_hash"],
                        output_path=row["output_path"],
                        metadata=json.loads(row["metadata"]),
                        git_commit=row["git_commit"],
                        git_branch=row["git_branch"],
                        code_version=row["code_version"],
                        config_hash=row["config_hash"],
                        duration_ms=row["duration_ms"],
                        records_in=row["records_in"],
                        records_out=row["records_out"],
                    ))
                
                return records
    
    def get_by_hash(self, output_hash: str) -> LineageRecord | None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM lineage_records WHERE output_hash = ? LIMIT 1",
                    (output_hash,)
                )
                row = cursor.fetchone()
                
                if row:
                    return LineageRecord(
                        record_id=row["record_id"],
                        event_type=LineageEventType(row["event_type"]),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        stage=row["stage"],
                        pair=row["pair"],
                        input_hashes=json.loads(row["input_hashes"]),
                        output_hash=row["output_hash"],
                        output_path=row["output_path"],
                        metadata=json.loads(row["metadata"]),
                        git_commit=row["git_commit"],
                        git_branch=row["git_branch"],
                        code_version=row["code_version"],
                        config_hash=row["config_hash"],
                        duration_ms=row["duration_ms"],
                        records_in=row["records_in"],
                        records_out=row["records_out"],
                    )
                return None
    
    def get_lineage_chain(self, output_hash: str) -> list[LineageRecord]:
        chain = []
        visited = {output_hash}
        to_process = [output_hash]
        
        while to_process:
            current_hash = to_process.pop(0)
            record = self.get_by_hash(current_hash)
            
            if not record:
                continue
            
            chain.append(record)
            
            for inp in record.input_hashes:
                if inp not in visited:
                    visited.add(inp)
                    to_process.append(inp)
        
        return list(reversed(chain))
    
    def get_stats(self) -> dict[str, Any]:
        """Get store statistics"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                cursor = conn.execute("SELECT COUNT(*) as count FROM lineage_records")
                stats["total_records"] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT stage) as count FROM lineage_records")
                stats["unique_stages"] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT pair) as count FROM lineage_records WHERE pair IS NOT NULL")
                stats["unique_pairs"] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM lineage_records")
                min_ts, max_ts = cursor.fetchone()
                stats["time_range"] = {"start": min_ts, "end": max_ts}
                
                return stats