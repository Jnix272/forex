"""
Monitoring Package - Unified Observability for Training.

Provides unified logging, checking, alerting, and dashboard capabilities.
"""

from monitoring.events import (
    TrainingEvent,
    EventType,
    Severity,
    CheckPhase,
    validate_payload,
)

from monitoring.event_bus import (
    EventBus,
    get_event_bus,
    init_event_bus,
    shutdown_event_bus,
    HandlerPriority,
)

from monitoring.unified_logger import (
    UnifiedLogger,
    UnifiedLoggerConfig,
    get_logger,
    create_logger,
)

from monitoring.checks import (
    CheckRegistry,
    CheckEngine,
    CheckContext,
    CheckResult,
    CheckStatus,
    CheckMetadata,
    get_registry,
    get_engine,
    register_check,
)

from monitoring.alerts.engine import (
    AlertRuleEngine,
    AlertRule,
    AlertChannel,
    AlertState,
    get_alert_engine,
    init_alert_engine,
)

from monitoring.dashboard.app import run_dashboard

# Import all checks to register them
import monitoring.checks.nan_detection
import monitoring.checks.gradient_norm
import monitoring.checks.loss_plateau
import monitoring.checks.representation_collapse
import monitoring.checks.checkpoint_load
import monitoring.checks.data_drift
import monitoring.checks.resource_monitor

__all__ = [
    # Events
    "TrainingEvent",
    "EventType",
    "Severity",
    "CheckPhase",
    "validate_payload",
    
    # Event Bus
    "EventBus",
    "get_event_bus",
    "init_event_bus",
    "shutdown_event_bus",
    "HandlerPriority",
    
    # Unified Logger
    "UnifiedLogger",
    "UnifiedLoggerConfig",
    "get_logger",
    "create_logger",
    
    # Checks
    "CheckRegistry",
    "CheckEngine",
    "CheckContext",
    "CheckResult",
    "CheckStatus",
    "CheckMetadata",
    "get_registry",
    "get_engine",
    "register_check",
    
    # Alerts
    "AlertRuleEngine",
    "AlertRule",
    "AlertChannel",
    "AlertState",
    "get_alert_engine",
    "init_alert_engine",
    
    # Dashboard
    "run_dashboard",
]

__version__ = "1.0.0"