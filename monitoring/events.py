"""
Unified Event Schema for Training Observability.

Single structured event format for ALL logging/checking/alerting/metrics.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class EventType(StrEnum):
    """Type of event - determines payload structure and handling."""

    LOG = "log"  # General log message
    CHECK = "check"  # Validation check result
    ALERT = "alert"  # Alert triggered
    METRIC = "metric"  # Scalar metric value
    CHECKPOINT = "checkpoint"  # Model checkpoint saved
    HEARTBEAT = "heartbeat"  # Liveness signal
    PROGRESS = "progress"  # Training progress update


class Severity(StrEnum):
    """Event severity - maps to standard logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CheckPhase(StrEnum):
    """Training phase when check runs."""

    PREFLIGHT = "preflight"
    BATCH = "batch"
    EPOCH = "epoch"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    PRETRAIN = "pretrain"
    RL_EPISODE = "rl_episode"
    POST_TRAIN = "post_train"


@dataclass
class TrainingEvent:
    """
    Unified training event - all logging/checking/alerting goes through this.

    Payload structure varies by event_type:
    - LOG: {"message": str, "logger": str}
    - CHECK: {"name": str, "passed": bool, "details": dict, "threshold": float}
    - ALERT: {"rule": str, "message": str, "context": dict}
    - METRIC: {"name": str, "value": float, "unit": str}
    - CHECKPOINT: {"path": str, "metrics": dict, "epoch": int}
    - HEARTBEAT: {"component": str, "status": str}
    - PROGRESS: {"phase": str, "current": int, "total": int, "eta_sec": float}
    """

    # Core identifiers
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_type: EventType = EventType.LOG
    severity: Severity = Severity.INFO

    # Source tracking
    source: str = ""  # module:function (e.g., "supervised_loop:train_epoch")
    run_id: str = ""  # unique run identifier
    session_id: str = ""  # session identifier for live trading

    # Training context
    epoch: int | None = None
    batch: int | None = None
    model_name: str | None = None
    fold: int | None = None

    # Payload - typed per event_type
    payload: dict = field(default_factory=dict)

    # Filtering
    tags: list[str] = field(default_factory=list)

    # Relationships
    parent_event_id: str | None = None
    correlation_id: str | None = None

    def __post_init__(self):
        if isinstance(self.event_type, str):
            self.event_type = EventType(self.event_type)
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity)
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSONL output."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "source": self.source,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "epoch": self.epoch,
            "batch": self.batch,
            "model_name": self.model_name,
            "fold": self.fold,
            "payload": self.payload,
            "tags": self.tags,
            "parent_event_id": self.parent_event_id,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrainingEvent:
        """Deserialize from dictionary."""
        # Convert string enums
        if "event_type" in data and isinstance(data["event_type"], str):
            data["event_type"] = EventType(data["event_type"])
        if "severity" in data and isinstance(data["severity"], str):
            data["severity"] = Severity(data["severity"])
        return cls(**data)

    # Factory methods for common event types
    @classmethod
    def log(
        cls, message: str, source: str, severity: Severity = Severity.INFO, run_id: str = "", **kwargs
    ) -> TrainingEvent:
        return cls(
            event_type=EventType.LOG,
            severity=severity,
            source=source,
            run_id=run_id,
            payload={"message": message, "logger": kwargs.get("logger", "")},
            tags=kwargs.get("tags", []),
            epoch=kwargs.get("epoch"),
            batch=kwargs.get("batch"),
            model_name=kwargs.get("model_name"),
        )

    @classmethod
    def check(
        cls,
        name: str,
        passed: bool,
        source: str,
        details: dict | None = None,
        threshold: float | None = None,
        run_id: str = "",
        **kwargs,
    ) -> TrainingEvent:
        return cls(
            event_type=EventType.CHECK,
            severity=Severity.INFO if passed else Severity.WARNING,
            source=source,
            run_id=run_id,
            payload={
                "name": name,
                "passed": passed,
                "details": details or {},
                "threshold": threshold,
            },
            tags=[*kwargs.get("tags", []), "check", name],
            epoch=kwargs.get("epoch"),
            batch=kwargs.get("batch"),
            model_name=kwargs.get("model_name"),
        )

    @classmethod
    def alert(
        cls, rule: str, message: str, severity: Severity, source: str, context: dict | None = None, run_id: str = "", **kwargs
    ) -> TrainingEvent:
        return cls(
            event_type=EventType.ALERT,
            severity=severity,
            source=source,
            run_id=run_id,
            payload={
                "rule": rule,
                "message": message,
                "context": context or {},
            },
            tags=[*kwargs.get("tags", []), "alert", rule],
            epoch=kwargs.get("epoch"),
            batch=kwargs.get("batch"),
            model_name=kwargs.get("model_name"),
        )

    @classmethod
    def metric(cls, name: str, value: float, source: str, unit: str = "", run_id: str = "", **kwargs) -> TrainingEvent:
        return cls(
            event_type=EventType.METRIC,
            severity=Severity.INFO,
            source=source,
            run_id=run_id,
            payload={
                "name": name,
                "value": value,
                "unit": unit,
            },
            tags=[*kwargs.get("tags", []), "metric", name],
            epoch=kwargs.get("epoch"),
            batch=kwargs.get("batch"),
            model_name=kwargs.get("model_name"),
        )

    @classmethod
    def checkpoint(cls, path: str, metrics: dict, epoch: int, source: str, run_id: str = "", **kwargs) -> TrainingEvent:
        return cls(
            event_type=EventType.CHECKPOINT,
            severity=Severity.INFO,
            source=source,
            run_id=run_id,
            epoch=epoch,
            payload={
                "path": path,
                "metrics": metrics,
                "epoch": epoch,
            },
            tags=[*kwargs.get("tags", []), "checkpoint"],
            model_name=kwargs.get("model_name"),
        )

    @classmethod
    def heartbeat(
        cls, component: str, status: str = "alive", source: str = "", run_id: str = "", **kwargs
    ) -> TrainingEvent:
        return cls(
            event_type=EventType.HEARTBEAT,
            severity=Severity.DEBUG,
            source=source,
            run_id=run_id,
            payload={
                "component": component,
                "status": status,
            },
            tags=[*kwargs.get("tags", []), "heartbeat", component],
        )

    @classmethod
    def progress(
        cls, phase: str, current: int, total: int, eta_sec: float = 0, source: str = "", run_id: str = "", **kwargs
    ) -> TrainingEvent:
        return cls(
            event_type=EventType.PROGRESS,
            severity=Severity.INFO,
            source=source,
            run_id=run_id,
            payload={
                "phase": phase,
                "current": current,
                "total": total,
                "eta_sec": eta_sec,
                "pct": (current / total * 100) if total > 0 else 0,
            },
            tags=[*kwargs.get("tags", []), "progress", phase],
        )


# Type-specific payload schemas for validation
LOG_PAYLOAD_KEYS = {"message", "logger"}
CHECK_PAYLOAD_KEYS = {"name", "passed", "details", "threshold"}
ALERT_PAYLOAD_KEYS = {"rule", "message", "context"}
METRIC_PAYLOAD_KEYS = {"name", "value", "unit"}
CHECKPOINT_PAYLOAD_KEYS = {"path", "metrics", "epoch"}
HEARTBEAT_PAYLOAD_KEYS = {"component", "status"}
PROGRESS_PAYLOAD_KEYS = {"phase", "current", "total", "eta_sec", "pct"}

PAYLOAD_SCHEMA = {
    EventType.LOG: LOG_PAYLOAD_KEYS,
    EventType.CHECK: CHECK_PAYLOAD_KEYS,
    EventType.ALERT: ALERT_PAYLOAD_KEYS,
    EventType.METRIC: METRIC_PAYLOAD_KEYS,
    EventType.CHECKPOINT: CHECKPOINT_PAYLOAD_KEYS,
    EventType.HEARTBEAT: HEARTBEAT_PAYLOAD_KEYS,
    EventType.PROGRESS: PROGRESS_PAYLOAD_KEYS,
}


def validate_payload(event_type: EventType, payload: dict) -> tuple[bool, list[str]]:
    """Validate payload has required keys for event type."""
    required = PAYLOAD_SCHEMA.get(event_type, set())
    missing = required - set(payload.keys())
    extra = set(payload.keys()) - required
    return len(missing) == 0, list(missing) + [f"unexpected: {e}" for e in extra]
