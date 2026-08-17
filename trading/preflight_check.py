"""
trading/preflight_check.py
============================
INF-010: Live trading pre-flight readiness check.

Must pass before live_engine.py begins trading. Verifies data feed,
broker connection, model warmup, risk limits, and calibration state.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReadinessReport:
    """Structured pre-flight report. Trading is blocked unless is_ready == True."""

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    feed_connected: bool = False
    feed_latency_ms: float = float("inf")
    last_tick_age_s: float = float("inf")
    broker_connected: bool = False
    broker_authenticated: bool = False
    positions_reconciled: bool = False
    risk_limits_loaded: bool = False
    risk_limits_valid: bool = False
    model_loaded: bool = False
    model_warmed_up: bool = False
    calibration_loaded: bool = False
    schema_matched: bool = False
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return all(
            [
                self.feed_connected,
                self.last_tick_age_s < 30.0,
                self.broker_connected,
                self.broker_authenticated,
                self.risk_limits_loaded,
                self.risk_limits_valid,
                self.model_loaded,
                self.model_warmed_up,
                self.schema_matched,
            ]
        )

    def save(self, path: str = "logs/live_readiness_report.json"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "is_ready": self.is_ready,
            "feed_connected": self.feed_connected,
            "feed_latency_ms": self.feed_latency_ms,
            "last_tick_age_s": self.last_tick_age_s,
            "broker_connected": self.broker_connected,
            "broker_authenticated": self.broker_authenticated,
            "positions_reconciled": self.positions_reconciled,
            "risk_limits_loaded": self.risk_limits_loaded,
            "risk_limits_valid": self.risk_limits_valid,
            "model_loaded": self.model_loaded,
            "model_warmed_up": self.model_warmed_up,
            "calibration_loaded": self.calibration_loaded,
            "schema_matched": self.schema_matched,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def check_feed(report: ReadinessReport, feed_client: Any) -> None:
    """Verify data feed is connected and recent."""
    try:
        if (hasattr(feed_client, "is_connected") and feed_client.is_connected()) or (
            hasattr(feed_client, "connected") and feed_client.connected
        ):
            report.feed_connected = True
        else:
            report.errors.append("Data feed not connected")
            return

        if hasattr(feed_client, "last_tick_time"):
            last_tick = feed_client.last_tick_time
            if last_tick is not None:
                age = (datetime.now(UTC) - last_tick).total_seconds()
                report.last_tick_age_s = age
                if age > 30.0:
                    report.errors.append(f"Last tick is {age:.1f}s old (stale)")
            else:
                report.last_tick_age_s = 0.0
                report.warnings.append("No tick received yet (first start?)")

        if hasattr(feed_client, "ping"):
            t0 = time.perf_counter()
            feed_client.ping()
            report.feed_latency_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        report.errors.append(f"Feed check failed: {e}")
        report.feed_connected = False


def check_broker(report: ReadinessReport, broker_client: Any) -> None:
    """Verify broker connection and authentication."""
    try:
        if hasattr(broker_client, "is_connected"):
            report.broker_connected = broker_client.is_connected()
        elif hasattr(broker_client, "connected"):
            report.broker_connected = bool(broker_client.connected)

        if not report.broker_connected:
            report.errors.append("Broker not connected")
            return

        if hasattr(broker_client, "is_authenticated"):
            report.broker_authenticated = broker_client.is_authenticated()
        elif hasattr(broker_client, "account_id"):
            report.broker_authenticated = bool(broker_client.account_id)
        else:
            report.broker_authenticated = report.broker_connected

        if hasattr(broker_client, "get_positions"):
            try:
                broker_client.get_positions()
                report.positions_reconciled = True
            except Exception as e:
                report.warnings.append(f"Position reconciliation failed: {e}")
    except Exception as e:
        report.errors.append(f"Broker check failed: {e}")


def check_risk_limits(report: ReadinessReport, config: dict) -> None:
    """Verify risk parameters are loaded and within sane bounds."""
    try:
        risk_cfg = config.get("risk", config.get("live_risk", {}))
        if not risk_cfg:
            report.errors.append("No risk configuration found")
            return

        report.risk_limits_loaded = True

        max_dd = risk_cfg.get("max_drawdown_pct", 0)
        kelly_frac = risk_cfg.get("kelly_fraction", 0)
        max_pos = risk_cfg.get("max_position_size", 0)
        risk_pct = risk_cfg.get("risk_pct", 0)

        issues = []
        if max_dd <= 0 or max_dd > 50:
            issues.append(f"max_drawdown_pct={max_dd} out of sane range (0,50]")
        if kelly_frac <= 0 or kelly_frac > 1.0:
            issues.append(f"kelly_fraction={kelly_frac} out of range (0,1]")
        if max_pos <= 0:
            issues.append(f"max_position_size={max_pos} must be positive")
        if risk_pct <= 0 or risk_pct > 10:
            issues.append(f"risk_pct={risk_pct} out of sane range (0,10]")

        if issues:
            for issue in issues:
                report.errors.append(f"Risk limit invalid: {issue}")
        else:
            report.risk_limits_valid = True
    except Exception as e:
        report.errors.append(f"Risk check failed: {e}")


def check_model(report: ReadinessReport, model: Any, warmup_input: Any | None = None) -> None:
    """Verify model is loaded and can produce inference."""
    try:
        if model is None:
            report.errors.append("Model is None")
            return

        report.model_loaded = True

        if warmup_input is not None:
            import torch

            model.eval()
            with torch.no_grad():
                out = model(warmup_input)
            if out is not None:
                report.model_warmed_up = True
            else:
                report.errors.append("Model warmup returned None")
        else:
            report.model_warmed_up = True
            report.warnings.append("No warmup input provided - model not tested")
    except Exception as e:
        report.errors.append(f"Model warmup failed: {e}")


def check_calibration(report: ReadinessReport, checkpoint_dir: str) -> None:
    """Verify calibration artifacts exist."""
    cal_path = Path(checkpoint_dir) / "calibration_report.json"
    if cal_path.exists():
        report.calibration_loaded = True
    else:
        report.warnings.append("No calibration_report.json found (optional)")


def check_schema(report: ReadinessReport, model_schema_hash: str, live_schema_hash: str) -> None:
    """Verify live feature schema matches model's trained schema."""
    if model_schema_hash == live_schema_hash:
        report.schema_matched = True
    else:
        report.errors.append(f"Schema mismatch: model={model_schema_hash[:12]} live={live_schema_hash[:12]}")


def run_preflight(
    feed_client: Any = None,
    broker_client: Any = None,
    config: dict | None = None,
    model: Any = None,
    warmup_input: Any = None,
    checkpoint_dir: str = "",
    model_schema_hash: str = "",
    live_schema_hash: str = "",
    save_report: bool = True,
) -> ReadinessReport:
    """Run all pre-flight checks and return a structured report.

    Raises RuntimeError if critical checks fail.
    """
    report = ReadinessReport()

    if feed_client is not None:
        check_feed(report, feed_client)
    else:
        report.errors.append("No feed client provided")

    if broker_client is not None:
        check_broker(report, broker_client)
    else:
        report.errors.append("No broker client provided")

    if config is not None:
        check_risk_limits(report, config)
    else:
        report.errors.append("No config provided for risk limit check")

    if model is not None:
        check_model(report, model, warmup_input)
    else:
        report.errors.append("No model provided")

    if checkpoint_dir:
        check_calibration(report, checkpoint_dir)

    if model_schema_hash and live_schema_hash:
        check_schema(report, model_schema_hash, live_schema_hash)
    elif model_schema_hash or live_schema_hash:
        report.warnings.append("Only one schema hash provided - cannot compare")
    else:
        report.schema_matched = True
        report.warnings.append("Schema check skipped (no hashes provided)")

    if save_report:
        try:
            report.save()
        except Exception as e:
            logger.warning(f"[Preflight] Could not save report: {e}")

    logger.info(
        f"[Preflight] {'READY' if report.is_ready else 'BLOCKED'} | "
        f"errors={len(report.errors)} warnings={len(report.warnings)}"
    )

    if not report.is_ready:
        raise RuntimeError(f"Pre-flight check FAILED ({len(report.errors)} errors): " + "; ".join(report.errors[:5]))

    return report
