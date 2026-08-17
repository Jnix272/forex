"""
Data Lineage Tracker
====================
Tracks data provenance through the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl

from lineage.store import LineageStore


class LineageEventType(StrEnum):
    """Types of lineage events"""

    SOURCE_LOAD = "source_load"  # Loading raw data
    TRANSFORM = "transform"  # Data transformation
    VALIDATION = "validation"  # Data validation
    JOIN = "join"  # Data join
    FEATURE_COMPUTE = "feature_compute"  # Feature computation
    LABEL_COMPUTE = "label_compute"  # Label computation
    DATASET_BUILD = "dataset_build"  # Dataset construction
    MODEL_TRAIN = "model_train"  # Model training
    MODEL_EVAL = "model_eval"  # Model evaluation


@dataclass
class LineageRecord:
    """Single lineage record"""

    record_id: str
    event_type: LineageEventType
    timestamp: datetime
    stage: str
    pair: str | None
    input_hashes: list[str] = field(default_factory=list)  # Hashes of input data
    output_hash: str | None = None
    output_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    git_branch: str | None = None
    code_version: str | None = None
    config_hash: str | None = None
    duration_ms: float = 0.0
    records_in: int = 0
    records_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageRecord:
        data = data.copy()
        data["event_type"] = LineageEventType(data["event_type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class LineageEvent:
    """Event context for lineage tracking"""

    event_type: LineageEventType
    stage: str
    pair: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LineageTracker:
    """Tracks data lineage through the pipeline"""

    def __init__(
        self,
        run_id: str | None = None,
        store: LineageStore | None = None,
        auto_git: bool = True,
        config: dict | None = None,
    ):
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.store = store
        self.auto_git = auto_git
        self.config = config or {}
        self.records: list[LineageRecord] = []
        self._lock = threading.Lock()
        self._current_inputs: list[str] = []
        self._git_info = self._get_git_info() if auto_git else {"commit": None, "branch": None}
        self._config_hash = self._hash_config(config) if config else None

    def _get_git_info(self) -> dict[str, str | None]:
        """Get current git commit and branch"""
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            return {"commit": commit, "branch": branch}
        except Exception:
            return {"commit": None, "branch": None}

    def _hash_config(self, config: dict | None) -> str | None:
        """Hash configuration for reproducibility"""
        if not config:
            return None
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _compute_data_hash(self, df: pl.DataFrame, sample_size: int = 1000) -> str:
        """Compute hash of DataFrame for lineage"""
        sample = df.head(min(sample_size, len(df)))
        data_str = sample.write_csv()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    @contextmanager
    def track(
        self,
        event: LineageEvent,
        input_data: pl.DataFrame | list[pl.DataFrame] | None = None,
    ):
        """Context manager for tracking a pipeline operation"""
        start_time = datetime.now()
        f"{self.run_id}_{event.event_type.value}_{uuid.uuid4().hex[:8]}"

        # Compute input hashes
        input_hashes = []
        records_in = 0

        if input_data is not None:
            if isinstance(input_data, list):
                for df in input_data:
                    if df is not None and len(df) > 0:
                        input_hashes.append(self._compute_data_hash(df))
                        records_in += len(df)
            elif len(input_data) > 0:
                input_hashes.append(self._compute_data_hash(input_data))
                records_in = len(input_data)

        # Also include previously tracked inputs
        input_hashes.extend(self._current_inputs)

        yield  # Execute the operation

        # After operation, caller should set output via set_output()
        (datetime.now() - start_time).total_seconds() * 1000

        # Store for chaining
        self._current_inputs = input_hashes.copy()

    def record_output(
        self,
        output_data: pl.DataFrame | None,
        output_path: str | None = None,
        records_out: int | None = None,
        metadata: dict | None = None,
    ) -> LineageRecord:
        """Record the output of an operation"""
        if not self._current_inputs and output_data is None:
            return None

        output_hash = self._compute_data_hash(output_data) if output_data is not None and len(output_data) > 0 else None

        record = LineageRecord(
            record_id=f"{self.run_id}_output_{uuid.uuid4().hex[:8]}",
            event_type=LineageEventType.TRANSFORM,  # Will be updated by caller
            timestamp=datetime.now(),
            stage="unknown",
            pair=None,
            input_hashes=self._current_inputs.copy(),
            output_hash=output_hash,
            output_path=output_path,
            metadata=metadata or {},
            git_commit=self._git_info["commit"],
            git_branch=self._git_info["branch"],
            config_hash=self._config_hash,
            records_in=sum(1 for _ in self._current_inputs),  # Approximate
            records_out=records_out or (len(output_data) if output_data is not None else 0),
        )

        with self._lock:
            self.records.append(record)
            if self.store:
                self.store.save(record)

        # Update current inputs for chaining
        if output_hash:
            self._current_inputs = [output_hash]

        return record

    def record_event(
        self,
        event: LineageEvent,
        input_data: pl.DataFrame | list[pl.DataFrame] | None = None,
        output_data: pl.DataFrame | None = None,
        output_path: str | None = None,
        metadata: dict | None = None,
    ) -> LineageRecord:
        """Record a complete lineage event"""
        start_time = datetime.now()
        record_id = f"{self.run_id}_{event.event_type.value}_{uuid.uuid4().hex[:8]}"

        # Compute input hashes
        input_hashes = []
        records_in = 0

        if input_data is not None:
            if isinstance(input_data, list):
                for df in input_data:
                    if df is not None and len(df) > 0:
                        input_hashes.append(self._compute_data_hash(df))
                        records_in += len(df)
            elif len(input_data) > 0:
                input_hashes.append(self._compute_data_hash(input_data))
                records_in = len(input_data)

        # Compute output hash
        output_hash = None
        records_out = 0
        if output_data is not None and len(output_data) > 0:
            output_hash = self._compute_data_hash(output_data)
            records_out = len(output_data)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        record = LineageRecord(
            record_id=record_id,
            event_type=event.event_type,
            timestamp=datetime.now(),
            stage=event.stage,
            pair=event.pair,
            input_hashes=input_hashes,
            output_hash=output_hash,
            output_path=output_path,
            metadata={**event.metadata, **(metadata or {})},
            git_commit=self._git_info["commit"],
            git_branch=self._git_info["branch"],
            code_version=self._git_info["commit"][:8] if self._git_info["commit"] else None,
            config_hash=self._config_hash,
            duration_ms=duration_ms,
            records_in=records_in,
            records_out=records_out,
        )

        with self._lock:
            self.records.append(record)
            if self.store:
                self.store.save(record)

        # Update current inputs for chaining
        if output_hash:
            self._current_inputs = [output_hash]

        return record

    def get_lineage_graph(self) -> dict[str, Any]:
        """Get lineage as a graph structure"""
        nodes = {}
        edges = []

        for record in self.records:
            # Output node
            if record.output_hash:
                nodes[record.output_hash] = {
                    "hash": record.output_hash,
                    "type": "output",
                    "stage": record.stage,
                    "event_type": record.event_type.value,
                    "timestamp": record.timestamp.isoformat(),
                    "records": record.records_out,
                    "path": record.output_path,
                }

            # Input nodes
            for inp_hash in record.input_hashes:
                if inp_hash not in nodes:
                    nodes[inp_hash] = {
                        "hash": inp_hash,
                        "type": "input",
                    }
                edges.append(
                    {
                        "from": inp_hash,
                        "to": record.output_hash,
                        "stage": record.stage,
                        "event_type": record.event_type.value,
                    }
                )

        return {"nodes": list(nodes.values()), "edges": edges}

    def get_stage_summary(self) -> dict[str, dict]:
        """Get summary by stage"""
        summary = {}
        for record in self.records:
            if record.stage not in summary:
                summary[record.stage] = {
                    "events": 0,
                    "records_in": 0,
                    "records_out": 0,
                    "duration_ms": 0,
                    "event_types": set(),
                }
            s = summary[record.stage]
            s["events"] += 1
            s["records_in"] += record.records_in
            s["records_out"] += record.records_out
            s["duration_ms"] += record.duration_ms
            s["event_types"].add(record.event_type.value)

        # Convert sets to lists for JSON serialization
        for s in summary.values():
            s["event_types"] = list(s["event_types"])

        return summary

    def export_json(self, filepath: str | Path):
        """Export lineage to JSON file"""
        data = {
            "run_id": self.run_id,
            "git_info": self._git_info,
            "config_hash": self._config_hash,
            "generated_at": datetime.now().isoformat(),
            "records": [r.to_dict() for r in self.records],
            "graph": self.get_lineage_graph(),
            "stage_summary": self.get_stage_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def clear(self):
        """Clear current lineage records"""
        with self._lock:
            self.records.clear()
            self._current_inputs.clear()


# Global tracker instance
_global_tracker: LineageTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> LineageTracker | None:
    """Get global tracker instance"""
    return _global_tracker


def set_tracker(tracker: LineageTracker | None):
    """Set global tracker instance"""
    global _global_tracker
    with _tracker_lock:
        _global_tracker = tracker


@contextmanager
def track_operation(
    event_type: LineageEventType,
    stage: str,
    pair: str | None = None,
    input_data: pl.DataFrame | list[pl.DataFrame] | None = None,
    metadata: dict | None = None,
):
    """Convenience context manager for tracking operations"""
    tracker = get_tracker()
    if tracker is None:
        yield None
        return

    event = LineageEvent(
        event_type=event_type,
        stage=stage,
        pair=pair,
        metadata=metadata or {},
    )

    with tracker.track(event, input_data) as ctx:
        yield ctx

    # Caller must call tracker.record_output() after the operation
