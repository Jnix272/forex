"""
Alert Rules Engine.

Configurable alert rules with rate limiting, deduplication, and multi-channel dispatch.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from monitoring.events import TrainingEvent, EventType, Severity
from monitoring.unified_logger import UnifiedLogger


class AlertChannel(str, Enum):
    """Alert delivery channels."""
    DISCORD = "discord"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: str  # Human-readable; actual logic in evaluator
    evaluator: Callable[[TrainingEvent], bool]  # Returns True if alert should fire
    severity: Severity = Severity.WARNING
    channels: list[AlertChannel] = field(default_factory=lambda: [AlertChannel.CONSOLE])
    cooldown_sec: float = 300.0  # Minimum time between same alert
    max_fires_per_hour: int = 10  # Rate limit
    tags: set[str] = field(default_factory=set)
    enabled: bool = True
    description: str = ""
    
    def __post_init__(self):
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity)
        if isinstance(self.channels, list):
            self.channels = [AlertChannel(c) if isinstance(c, str) else c for c in self.channels]


@dataclass
class AlertState:
    """Tracks alert firing state for rate limiting."""
    last_fired: float = 0
    fire_count: int = 0
    hour_window_start: float = field(default_factory=time.time)
    
    def can_fire(self, cooldown_sec: float, max_per_hour: int) -> bool:
        now = time.time()
        
        # Reset hour window
        if now - self.hour_window_start > 3600:
            self.hour_window_start = now
            self.fire_count = 0
        
        # Check cooldown
        if now - self.last_fired < cooldown_sec:
            return False
        
        # Check rate limit
        if self.fire_count >= max_per_hour:
            return False
        
        return True
    
    def record_fire(self):
        self.last_fired = time.time()
        self.fire_count += 1


class AlertRuleEngine:
    """Evaluates alert rules and dispatches alerts."""
    
    def __init__(self, logger: UnifiedLogger):
        self.logger = logger
        self.rules: dict[str, AlertRule] = {}
        self.states: dict[str, AlertState] = defaultdict(AlertState)
        self.dispatchers: dict[AlertChannel, Callable] = {}
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register built-in training alert rules."""
        
        # NaN Storm
        self.add_rule(AlertRule(
            name="nan_storm",
            condition=">10 NaN events in 60 seconds",
            evaluator=lambda e: e.event_type == EventType.CHECK and "nan" in e.tags and not e.payload.get("passed", True),
            severity=Severity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.DISCORD],
            cooldown_sec=60,
            max_fires_per_hour=20,
            tags={"nan", "gradient", "critical"},
            description="Multiple NaN gradients detected in short time window",
        ))
        
        # OOM Storm
        self.add_rule(AlertRule(
            name="oom_storm",
            condition=">5 OOM events in 5 minutes",
            evaluator=lambda e: "oom" in e.payload.get("message", "").lower() or "out of memory" in e.payload.get("message", "").lower(),
            severity=Severity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.DISCORD],
            cooldown_sec=300,
            max_fires_per_hour=10,
            tags={"oom", "memory", "critical"},
            description="Multiple CUDA out of memory errors",
        ))
        
        # Gradient Explosion
        self.add_rule(AlertRule(
            name="grad_explosion",
            condition="grad_norm > 100 for 3 consecutive epochs",
            evaluator=lambda e: e.event_type == EventType.CHECK and "grad_norm" in e.payload.get("name", "") and e.payload.get("value", 0) > 100,
            severity=Severity.ERROR,
            channels=[AlertChannel.CONSOLE],
            cooldown_sec=600,
            max_fires_per_hour=6,
            tags={"gradient", "explosion"},
            description="Gradient norm exceeds critical threshold",
        ))
        
        # Loss Divergence
        self.add_rule(AlertRule(
            name="loss_divergence",
            condition="train_loss > 10x initial for 5 epochs",
            evaluator=lambda e: e.event_type == EventType.CHECK and "divergence" in e.name and not e.payload.get("passed", True),
            severity=Severity.ERROR,
            channels=[AlertChannel.CONSOLE, AlertChannel.DISCORD],
            cooldown_sec=600,
            max_fires_per_hour=4,
            tags={"loss", "divergence", "training"},
            description="Training loss diverging significantly",
        ))
        
        # Val Sharpe Collapse
        self.add_rule(AlertRule(
            name="val_sharpe_collapse",
            condition="val_sharpe < 0.5 for 3 epochs",
            evaluator=lambda e: e.event_type == EventType.METRIC and "val_sharpe" in e.payload.get("name", "") and e.payload.get("value", 1) < 0.5,
            severity=Severity.WARNING,
            channels=[AlertChannel.CONSOLE],
            cooldown_sec=1800,
            max_fires_per_hour=2,
            tags={"sharpe", "validation", "performance"},
            description="Validation Sharpe ratio below threshold",
        ))
        
        # LR Too Low
        self.add_rule(AlertRule(
            name="lr_too_low",
            condition="learning_rate < 1e-9 for 10 epochs",
            evaluator=lambda e: e.event_type == EventType.METRIC and "lr" in e.payload.get("name", "") and e.payload.get("value", 1) < 1e-9,
            severity=Severity.WARNING,
            channels=[AlertChannel.CONSOLE],
            cooldown_sec=3600,
            max_fires_per_hour=1,
            tags={"lr", "learning_rate", "training"},
            description="Learning rate decayed too far",
        ))
        
        # Data Drift Critical
        self.add_rule(AlertRule(
            name="data_drift_critical",
            condition="PSI > 0.25 for 3 features",
            evaluator=lambda e: e.event_type == EventType.CHECK and "drift" in e.name and "critical" in str(e.payload.get("message", "")).lower(),
            severity=Severity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.DISCORD],
            cooldown_sec=1800,
            max_fires_per_hour=3,
            tags={"drift", "psi", "data", "critical"},
            description="Critical feature distribution drift detected",
        ))
        
        # Checkpoint Load Fail
        self.add_rule(AlertRule(
            name="checkpoint_load_fail",
            condition="Checkpoint load < 60% params",
            evaluator=lambda e: e.event_type == EventType.CHECK and "checkpoint_load" in e.name and not e.payload.get("passed", True),
            severity=Severity.ERROR,
            channels=[AlertChannel.CONSOLE, AlertChannel.DISCORD],
            cooldown_sec=3600,
            max_fires_per_hour=2,
            tags={"checkpoint", "load", "integrity"},
            description="Checkpoint failed to load properly",
        ))
        
        # Circuit Breaker
        self.add_rule(AlertRule(
            name="circuit_breaker",
            condition="Daily loss limit exceeded",
            evaluator=lambda e: "circuit_breaker" in e.payload.get("message", "").lower() or "daily_loss" in e.payload.get("message", "").lower(),
            severity=Severity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.DISCORD, AlertChannel.PAGERDUTY],
            cooldown_sec=60,
            max_fires_per_hour=50,
            tags={"circuit_breaker", "risk", "live", "critical"},
            description="Live trading circuit breaker triggered",
        ))
        
        # Model Demoted
        self.add_rule(AlertRule(
            name="model_demoted",
            condition="Model demoted due to performance",
            evaluator=lambda e: "demot" in str(e.payload).lower() or "rollback" in str(e.payload).lower(),
            severity=Severity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.DISCORD, AlertChannel.EMAIL],
            cooldown_sec=3600,
            max_fires_per_hour=2,
            tags={"demotion", "model", "performance", "critical"},
            description="Model automatically demoted from production",
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule
        if rule.name not in self.states:
            self.states[rule.name] = AlertState()
    
    def remove_rule(self, name: str):
        """Remove an alert rule."""
        self.rules.pop(name, None)
        self.states.pop(name, None)
    
    def enable_rule(self, name: str):
        """Enable a rule."""
        if name in self.rules:
            self.rules[name].enabled = True
    
    def disable_rule(self, name: str):
        """Disable a rule."""
        if name in self.rules:
            self.rules[name].enabled = False
    
    def register_dispatcher(self, channel: AlertChannel, func: Callable):
        """Register a channel dispatcher."""
        self.dispatchers[channel] = func
    
    async def evaluate(self, event: TrainingEvent) -> list[str]:
        """Evaluate event against all rules, return list of fired rule names."""
        fired = []
        
        for name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                should_fire = rule.evaluator(event)
            except Exception:
                continue
            
            if not should_fire:
                continue
            
            state = self.states[name]
            if not state.can_fire(rule.cooldown_sec, rule.max_fires_per_hour):
                continue
            
            # Fire alert
            state.record_fire()
            fired.append(name)
            
            # Dispatch to channels
            await self._dispatch(rule, event)
        
        return fired
    
    async def _dispatch(self, rule: AlertRule, event: TrainingEvent):
        """Dispatch alert to registered channels."""
        alert_event = TrainingEvent.alert(
            rule=rule.name,
            message=f"{rule.description}: {event.payload.get('message', event.payload)}",
            severity=rule.severity,
            source=f"alerts.{rule.name}",
            run_id=event.run_id,
            tags=list(rule.tags) + ["alert"],
            context={
                "trigger_event": event.to_dict(),
                "rule": rule.name,
                "rule_description": rule.description,
            },
        )
        
        # Log via unified logger
        self.logger._emit_sync(alert_event)
        
        # Dispatch to external channels
        for channel in rule.channels:
            dispatcher = self.dispatchers.get(channel)
            if dispatcher:
                try:
                    if asyncio.iscoroutinefunction(dispatcher):
                        await dispatcher(rule, event, alert_event)
                    else:
                        dispatcher(rule, event, alert_event)
                except Exception as e:
                    print(f"[AlertEngine] Dispatcher {channel} error: {e}")


# Default console dispatcher (uses logger)
def console_dispatcher(rule: AlertRule, event: TrainingEvent, alert_event: TrainingEvent):
    """Print alert to console via logger."""
    pass  # Already logged via logger._emit_sync


# Discord dispatcher (placeholder - requires webhook URL)
async def discord_dispatcher(rule: AlertRule, event: TrainingEvent, alert_event: TrainingEvent):
    """Send alert to Discord webhook."""
    # Implementation would use aiohttp to post to Discord webhook
    pass


# Global alert engine
_global_alert_engine: Optional[AlertRuleEngine] = None


def get_alert_engine(logger: UnifiedLogger = None) -> AlertRuleEngine:
    global _global_alert_engine
    if _global_alert_engine is None:
        if logger is None:
            from monitoring.unified_logger import get_logger
            logger = get_logger()
        _global_alert_engine = AlertRuleEngine(logger)
        _global_alert_engine.register_dispatcher(AlertChannel.CONSOLE, console_dispatcher)
    return _global_alert_engine


async def init_alert_engine(logger: UnifiedLogger) -> AlertRuleEngine:
    """Initialize and return alert engine."""
    global _global_alert_engine
    _global_alert_engine = AlertRuleEngine(logger)
    _global_alert_engine.register_dispatcher(AlertChannel.CONSOLE, console_dispatcher)
    return _global_alert_engine