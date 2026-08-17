"""
Monitoring Package - Unified Observability for Training.

Provides unified logging, checking, alerting, and dashboard capabilities.
"""

import monitoring.checks.checkpoint_load
import monitoring.checks.data_drift
import monitoring.checks.gradient_norm
import monitoring.checks.loss_plateau

# Import all checks to register them
import monitoring.checks.nan_detection
import monitoring.checks.representation_collapse
import monitoring.checks.resource_monitor  # noqa: F401
from monitoring.alerts.engine import (
    AlertChannel,
    AlertRule,
    AlertRuleEngine,
    AlertState,
    get_alert_engine,
    init_alert_engine,
)
from monitoring.checks import (
    CheckContext,
    CheckEngine,
    CheckMetadata,
    CheckRegistry,
    CheckResult,
    CheckStatus,
    get_engine,
    get_registry,
    register_check,
)
from monitoring.dashboard.app import run_dashboard
from monitoring.event_bus import (
    EventBus,
    HandlerPriority,
    get_event_bus,
    init_event_bus,
    shutdown_event_bus,
)
from monitoring.events import (
    CheckPhase,
    EventType,
    Severity,
    TrainingEvent,
    validate_payload,
)
from monitoring.unified_logger import (
    UnifiedLogger,
    UnifiedLoggerConfig,
    create_logger,
    get_logger,
)

__all__ = [
    "AlertChannel",
    "AlertRule",
    # Alerts
    "AlertRuleEngine",
    "AlertState",
    "CheckContext",
    "CheckEngine",
    "CheckMetadata",
    "CheckPhase",
    # Checks
    "CheckRegistry",
    "CheckResult",
    "CheckStatus",
    # Event Bus
    "EventBus",
    "EventType",
    "HandlerPriority",
    "Severity",
    # Events
    "TrainingEvent",
    # Unified Logger
    "UnifiedLogger",
    "UnifiedLoggerConfig",
    "create_logger",
    "get_alert_engine",
    "get_engine",
    "get_event_bus",
    "get_logger",
    "get_registry",
    "init_alert_engine",
    "init_event_bus",
    "register_check",
    # Dashboard
    "run_dashboard",
    "shutdown_event_bus",
    "validate_payload",
]

__version__ = "1.0.0"
