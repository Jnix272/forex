"""
monitoring/alerting.py - multi-channel alerting + escalation + runbooks
(Improvement #7)

Sinks:
  * ConsoleChannel        - stdout/console
  * FileChannel           - append to a log file
  * InMemoryChannel       - test mode (records alerts, sends nothing real)
  * DiscordWebhookChannel - reuses monitoring.discord_alerts.DiscordAlerter
  * SlackWebhookChannel   - POST to a Slack incoming webhook (requests/urllib)
  * SMTPChannel           - email via smtplib (SMTP_HOST/port/user/pass env)

Behaviour:
  * Severity levels: info < warning < critical; each channel declares the
    minimum severity it accepts, so routing is per-channel.
  * Deduplication: identical (alert_type, message) within a window is dropped
    unless ``force=True``.
  * Rate limiting: per-alert-type cap on messages per minute.
  * Escalation: critical alerts not acknowledged within ``escalation_minutes``
    are re-emitted at higher severity to all channels (acked via ``ack``).
  * Runbooks: per-alert-type remediation steps stored in RUNBOOKS; included in
    the alert payload so responders know how to react.
"""

from __future__ import annotations

import hashlib
import json
import os
import smtplib
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

INFO = "info"
WARNING = "warning"
CRITICAL = "critical"
_SEVERITY_RANK = {INFO: 0, WARNING: 1, CRITICAL: 2}

try:
    import requests

    _REQUESTS = True
except ImportError:  # pragma: no cover
    import urllib.error
    import urllib.request

    _REQUESTS = False


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp() -> float:
    return time.time()


# ═════════════════════════════════════════════════════════════════════════════
# Runbooks
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class Runbook:
    alert_type: str
    title: str
    severity: str = INFO
    channels: list[str] = field(default_factory=list)
    remediation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "title": self.title,
            "severity": self.severity,
            "channels": list(self.channels),
            "remediation": list(self.remediation),
        }


RUNBOOKS: dict[str, Runbook] = {
    "risk_violation": Runbook(
        alert_type="risk_violation",
        title="Risk limit breached",
        severity=CRITICAL,
        channels=["discord", "slack", "email", "console", "file"],
        remediation=[
            "Verify the breached rule and value in the risk audit log.",
            "Confirm the engine flattened positions / entered standby if required.",
            "Check feed gaps that could produce a false gap-risk flag.",
        ],
    ),
    "drift_detected": Runbook(
        alert_type="drift_detected",
        title="Data / model drift detected",
        severity=WARNING,
        channels=["discord", "console", "file"],
        remediation=[
            "Inspect drifted features / SHAP attribution vs reference window.",
            "Trigger a retrain if streaming drift score persists above threshold.",
            "Escalate to rollback if model drift breaches the rollback monitor.",
        ],
    ),
    "model_drift": Runbook(
        alert_type="model_drift",
        title="Challenger underperforming champion",
        severity=WARNING,
        channels=["discord", "console", "file"],
        remediation=[
            "Halt canary escalation for the underperforming challenger.",
            "Review the champion-challenger comparison report.",
            "Deescalate canary fraction to minimum if the gap widens.",
        ],
    ),
    "circuit_breaker": Runbook(
        alert_type="circuit_breaker",
        title="Circuit breaker triggered",
        severity=CRITICAL,
        channels=["discord", "slack", "email", "console", "file"],
        remediation=[
            "Confirm all positions were flattened.",
            "Check the triggering drawdown / VaR limit and equity.",
            "Do not resume trading until the RiskEngine resumes and breach is reviewed.",
        ],
    ),
    "demotion": Runbook(
        alert_type="demotion",
        title="Model demoted / rollback",
        severity=CRITICAL,
        channels=["discord", "slack", "console", "file"],
        remediation=[
            "Verify production_prev.pt was restored.",
            "Check the retrain flag was set for train_gpu.py.",
            "Review the demotion triggers and rolling stats.",
        ],
    ),
    "promotion": Runbook(
        alert_type="promotion",
        title="Model promoted",
        severity=INFO,
        channels=["console", "file"],
        remediation=["Record the promotion in the audit trail."],
    ),
    "system": Runbook(
        alert_type="system",
        title="System alert",
        severity=WARNING,
        channels=["console", "file"],
        remediation=["Check the system log and component health."],
    ),
}


def register_runbook(runbook: Runbook) -> None:
    RUNBOOKS[runbook.alert_type] = runbook


# ═════════════════════════════════════════════════════════════════════════════
# Channels
# ═════════════════════════════════════════════════════════════════════════════


class Channel:
    name = "base"
    severity_min: str = INFO

    def send(self, alert: dict[str, Any]) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def accepts(self, severity: str) -> bool:
        return _SEVERITY_RANK[severity] >= _SEVERITY_RANK[self.severity_min]


class ConsoleChannel(Channel):
    name = "console"
    severity_min = INFO

    def send(self, alert: dict[str, Any]) -> bool:
        print(f"[alert:{alert['severity'].upper()}] {alert['title']}: {alert['message']}")
        if alert.get("runbook", {}).get("remediation"):
            for step in alert["runbook"]["remediation"]:
                print(f"    - {step}")
        return True


class FileChannel(Channel):
    name = "file"
    severity_min = INFO

    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("ALERT_LOG_FILE", "logs/alerts.log")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: dict[str, Any]) -> bool:
        import json

        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert, default=str, sort_keys=True) + "\n")
            return True
        except OSError:
            return False


class InMemoryChannel(Channel):
    """Test-mode channel: records alerts, never sends anything real."""

    name = "in_memory"
    severity_min = INFO

    def __init__(self) -> None:
        self.alerts: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def send(self, alert: dict[str, Any]) -> bool:
        with self._lock:
            self.alerts.append(dict(alert))
        return True

    def last(self) -> dict[str, Any] | None:
        with self._lock:
            return self.alerts[-1] if self.alerts else None

    def clear(self) -> None:
        with self._lock:
            self.alerts.clear()


class DiscordWebhookChannel(Channel):
    name = "discord"
    severity_min = WARNING

    def __init__(self) -> None:
        try:
            from monitoring.discord_alerts import DiscordAlerter

            self._alerter = DiscordAlerter()
        except Exception:  # pragma: no cover
            self._alerter = None

    def send(self, alert: dict[str, Any]) -> bool:
        if self._alerter is None:
            return False
        try:
            self._alerter.send(alert.get("alert_type", "system"), alert.get("details", {}))
            return True
        except Exception:
            return False


class SlackWebhookChannel(Channel):
    name = "slack"
    severity_min = WARNING

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")

    def send(self, alert: dict[str, Any]) -> bool:
        if not self.webhook_url:
            return False
        payload = {"text": f"[{alert['severity'].upper()}] {alert['title']}: {alert['message']}"}
        try:
            if _REQUESTS:
                resp = requests.post(self.webhook_url, json=payload, timeout=5)
                return resp.status_code < 300
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5):  # type: ignore[attr-defined]
                return True
        except Exception:
            return False


class SMTPChannel(Channel):
    name = "email"
    severity_min = CRITICAL

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        sender: str | None = None,
        recipients: Sequence[str] | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        self.host = host or os.getenv("SMTP_HOST", "")
        self.port = int(port or os.getenv("SMTP_PORT", "587"))
        self.sender = sender or os.getenv("SMTP_SENDER", "")
        self.recipients = (
            list(recipients) if recipients else [r for r in os.getenv("SMTP_RECIPIENTS", "").split(",") if r]
        )
        self.username = username if username is not None else os.getenv("SMTP_USERNAME", "")
        self.password = password if password is not None else os.getenv("SMTP_PASSWORD", "")

    def send(self, alert: dict[str, Any]) -> bool:
        if not self.host or not self.recipients:
            return False
        try:
            msg = MIMEText(
                f"{alert['title']}: {alert['message']}\n\n" + "\n".join(alert.get("runbook", {}).get("remediation", []))
            )
            msg["Subject"] = f"[{alert['severity'].upper()}] {alert['title']}"
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)
            with smtplib.SMTP(self.host, self.port, timeout=10) as server:
                server.starttls()
                if self.username:
                    server.login(self.username, self.password)
                server.sendmail(self.sender, self.recipients, msg.as_string())
            return True
        except Exception:
            return False


# ═════════════════════════════════════════════════════════════════════════════
# Alert manager
# ═════════════════════════════════════════════════════════════════════════════


class AlertManager:
    def __init__(
        self,
        channels: Sequence[Channel] | None = None,
        dedup_window_seconds: float = 60.0,
        rate_limit_per_min: int = 30,
        escalation_minutes: float = 5.0,
    ):
        self.channels: list[Channel] = list(channels) if channels is not None else [ConsoleChannel(), FileChannel()]
        self.dedup_window_seconds = dedup_window_seconds
        self.rate_limit_per_min = rate_limit_per_min
        self.escalation_minutes = escalation_minutes
        self._dedup: dict[str, float] = {}
        self._rate: dict[str, list[float]] = {}
        self._pending_escalation: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ── routing ────────────────────────────────────────────────────────────

    def add_channel(self, channel: Channel) -> None:
        self.channels.append(channel)

    def clear_channels(self) -> None:
        self.channels = []

    def _allowed(self, alert_type: str) -> bool:
        with self._lock:
            now = _timestamp()
            stamp = self._rate.get(alert_type, [])
            stamp = [t for t in stamp if now - t < 60.0]
            if len(stamp) >= self.rate_limit_per_min:
                self._rate[alert_type] = stamp
                return False
            stamp.append(now)
            self._rate[alert_type] = stamp
            return True

    def _dedup_key(self, alert_type: str, message: str) -> str:
        return hashlib.sha256(f"{alert_type}\0{message}".encode()).hexdigest()

    def _is_duplicate(self, alert_type: str, message: str, force: bool) -> bool:
        if force:
            return False
        key = self._dedup_key(alert_type, message)
        now = _timestamp()
        with self._lock:
            last = self._dedup.get(key)
            if last is not None and now - last < self.dedup_window_seconds:
                return True
            self._dedup[key] = now
        return False

    # ── alert entry point ──────────────────────────────────────────────────

    def alert(
        self,
        alert_type: str,
        message: str,
        severity: str | None = None,
        details: dict[str, Any] | None = None,
        force: bool = False,
        alert_id: str | None = None,
    ) -> dict[str, Any]:
        runbook = RUNBOOKS.get(alert_type, RUNBOOKS["system"])
        severity = severity or runbook.severity
        if not self._allowed(alert_type):
            return {"alert_id": alert_id or "", "sent": False, "reason": "rate_limited"}
        if self._is_duplicate(alert_type, message, force):
            return {"alert_id": alert_id or "", "sent": False, "reason": "duplicate"}

        alert = {
            "alert_id": alert_id or self._dedup_key(alert_type, message),
            "alert_type": alert_type,
            "title": runbook.title,
            "message": message,
            "severity": severity,
            "details": details or {},
            "runbook": runbook.to_dict(),
            "timestamp": _now_iso(),
        }

        sent_channels = []
        for channel in self.channels:
            if channel.accepts(severity):
                try:
                    ok = channel.send(alert)
                except Exception:
                    ok = False
                sent_channels.append({"channel": channel.name, "sent": ok})

        if severity == CRITICAL:
            with self._lock:
                self._pending_escalation[alert["alert_id"]] = {
                    "alert": alert,
                    "sent_at": _timestamp(),
                    "escalated": False,
                }

        alert["channels"] = sent_channels
        alert["sent"] = any(c["sent"] for c in sent_channels)
        return alert

    # ── escalation ─────────────────────────────────────────────────────────

    def ack(self, alert_id: str) -> bool:
        with self._lock:
            if alert_id in self._pending_escalation:
                del self._pending_escalation[alert_id]
                return True
        return False

    def pending_critical(self) -> list[dict[str, Any]]:
        with self._lock:
            return [e["alert"] for e in self._pending_escalation.values()]

    def escalate_pending(self) -> list[dict[str, Any]]:
        """Re-emit critical alerts that were not acked within escalation_minutes."""
        escalated = []
        now = _timestamp()
        with self._lock:
            for _alert_id, entry in list(self._pending_escalation.items()):
                age_min = (now - entry["sent_at"]) / 60.0
                if age_min < self.escalation_minutes:
                    continue
                if entry["escalated"]:
                    continue
                entry["escalated"] = True
                alert = dict(entry["alert"])
                alert["message"] = f"ESCALATED (unacked {age_min:.1f} min): {alert['message']}"
                alert["severity"] = CRITICAL
                for channel in self.channels:
                    if channel.accepts(CRITICAL):
                        try:
                            channel.send(alert)
                        except Exception:
                            pass
                escalated.append(alert)
        return escalated


# ═════════════════════════════════════════════════════════════════════════════
# Integration helpers - wire drift events (item 4/5) + risk violations (item 1)
# ═════════════════════════════════════════════════════════════════════════════


def notify_drift_event(manager: AlertManager, event: dict[str, Any]) -> dict[str, Any]:
    """Map a structured drift event (drift.data_drift / drift.model_drift) to an alert."""
    etype = event.get("type", "drift_detected")
    alert_type = "model_drift" if etype == "model_drift" else "drift_detected"
    if etype in ("feature_drift", "shap_drift"):
        message = f"{event.get('feature', 'feature')} drifted (severity {event.get('drift')})"
    elif etype == "adversarial_drift":
        message = f"Adversarial AUC {event.get('auc')} > threshold {event.get('auc_threshold')}"
    elif etype == "concept_drift":
        message = f"Concept drift {event.get('detector')} state={event.get('state')}"
    elif etype == "model_drift":
        message = f"{event.get('event', 'challenger_losing')}: {event.get('challenger')}"
    else:
        message = f"Drift event {etype}"
    return manager.alert(alert_type, message, severity=WARNING, details=event)


def notify_risk_violation(manager: AlertManager, violation: dict[str, Any]) -> dict[str, Any]:
    """Map a RiskEngine decision/violation dict to a critical alert."""
    rule = violation.get("rule", "unknown")
    value = violation.get("value")
    action = violation.get("action", "reject")
    message = f"Risk rule '{rule}' = {value} → action: {action}"
    return manager.alert(
        "risk_violation",
        message,
        severity=CRITICAL,
        details=violation,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Default manager
# ═════════════════════════════════════════════════════════════════════════════

_default_manager: AlertManager | None = None


def default_manager() -> AlertManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = AlertManager()
    return _default_manager


def publish_alert(alert_type: str, message: str, **kwargs: Any) -> dict[str, Any]:
    return default_manager().alert(alert_type, message, **kwargs)
