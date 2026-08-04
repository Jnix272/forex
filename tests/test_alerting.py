"""
Tests for monitoring.alerting (Improvement #7): channels, severity routing,
dedup, rate limiting, escalation, runbooks, and drift/risk integration.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from monitoring.alerting import (
    AlertManager,
    InMemoryChannel,
    FileChannel,
    ConsoleChannel,
    SMTPChannel,
    SlackWebhookChannel,
    Runbook,
    register_runbook,
    notify_drift_event,
    notify_risk_violation,
    INFO,
    WARNING,
    CRITICAL,
)


@pytest.fixture
def mem():
    return InMemoryChannel()


@pytest.fixture
def manager(mem):
    m = AlertManager(channels=[], dedup_window_seconds=60, rate_limit_per_min=100)
    m.add_channel(mem)
    return m


# ═════════════════════════════════════════════════════════════════════════════
# Channels
# ═════════════════════════════════════════════════════════════════════════════

def test_in_memory_channel_records(mem):
    ok = mem.send({"severity": "info", "title": "t", "message": "m"})
    assert ok is True
    assert mem.last()["message"] == "m"
    assert len(mem.alerts) == 1


def test_console_channel_sends(capsys):
    ch = ConsoleChannel()
    assert ch.send({"severity": "info", "title": "T", "message": "hello",
                    "runbook": {"remediation": []}}) is True
    captured = capsys.readouterr().out
    assert "hello" in captured


def test_file_channel_appends(tmp_path):
    path = tmp_path / "alerts.log"
    ch = FileChannel(str(path))
    assert ch.send({"severity": "warning", "title": "t", "message": "m",
                    "runbook": {"remediation": []}, "x": 1}) is True
    assert ch.send({"severity": "info", "title": "t2", "message": "m2",
                    "runbook": {"remediation": []}}) is True
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["message"] == "m"


def test_slack_channel_no_webhook_fails():
    ch = SlackWebhookChannel("")
    assert ch.send({"severity": "info", "title": "t", "message": "m",
                    "runbook": {"remediation": []}}) is False


def test_smtp_channel_no_host_fails():
    ch = SMTPChannel("", recipients=["a@b.com"])
    assert ch.send({"severity": "info", "title": "t", "message": "m",
                    "runbook": {"remediation": []}}) is False


def test_channel_accepts_severity():
    from monitoring.alerting import Channel, WARNING, INFO, CRITICAL
    ch = Channel()
    ch.severity_min = WARNING
    assert ch.accepts(CRITICAL)
    assert ch.accepts(WARNING)
    assert not ch.accepts(INFO)


# ═════════════════════════════════════════════════════════════════════════════
# AlertManager — routing & dedup
# ═════════════════════════════════════════════════════════════════════════════

def test_alert_sent_to_in_memory(mem, manager):
    res = manager.alert("system", "disk full", severity=WARNING)
    assert res["sent"] is True or "sent" in res
    assert len(mem.alerts) == 1
    assert mem.last()["message"] == "disk full"


def test_alert_duplicate_dropped(manager):
    manager.alert("system", "disk full", severity=WARNING)
    res = manager.alert("system", "disk full", severity=WARNING)
    assert res["reason"] == "duplicate"
    assert len(manager.channels[0].alerts) == 1  # type: ignore[attr-defined]


def test_alert_force_overrides_duplicate(manager):
    manager.alert("system", "disk full", severity=WARNING)
    res = manager.alert("system", "disk full", severity=WARNING, force=True)
    assert "reason" not in res or res.get("reason") != "duplicate"
    assert len(manager.channels[0].alerts) == 2  # type: ignore[attr-defined]


def test_alert_rate_limited(manager):
    manager.rate_limit_per_min = 2
    manager.alert("system", "a", severity=WARNING)
    manager.alert("system", "b", severity=WARNING)
    res = manager.alert("system", "c", severity=WARNING)
    assert res["reason"] == "rate_limited"


def test_runbook_defaults():
    res = AlertManager(channels=[InMemoryChannel()]).alert("risk_violation", "x", severity=CRITICAL)
    assert res["runbook"]["title"] == "Risk limit breached"
    assert "remediation" in res["runbook"]


def test_unknown_alert_type_uses_system_runbook():
    res = AlertManager(channels=[InMemoryChannel()]).alert("nope", "x", severity=INFO)
    assert res["runbook"]["alert_type"] == "system"


def test_register_runbook():
    register_runbook(Runbook(alert_type="test_alert", title="Test", severity=INFO,
                             channels=["console"], remediation=["do x"]))
    res = AlertManager(channels=[InMemoryChannel()]).alert("test_alert", "m")
    assert res["runbook"]["title"] == "Test"
    assert res["runbook"]["remediation"] == ["do x"]


def test_alert_id_present():
    res = AlertManager(channels=[InMemoryChannel()]).alert("system", "m", severity=INFO)
    assert res["alert_id"]


# ═════════════════════════════════════════════════════════════════════════════
# Escalation
# ═════════════════════════════════════════════════════════════════════════════

def test_critical_alert_tracks_pending(manager):
    res = manager.alert("circuit_breaker", "dd", severity=CRITICAL)
    pending = manager.pending_critical()
    assert any(p["alert_id"] == res["alert_id"] for p in pending)


def test_ack_removes_pending(manager):
    res = manager.alert("circuit_breaker", "dd", severity=CRITICAL)
    assert manager.ack(res["alert_id"]) is True
    assert manager.ack(res["alert_id"]) is False
    assert res["alert_id"] not in {p["alert_id"] for p in manager.pending_critical()}


def test_escalate_after_timeout(manager):
    manager.escalation_minutes = 0.0
    mem2 = InMemoryChannel()
    manager.add_channel(mem2)
    manager.alert("circuit_breaker", "dd", severity=CRITICAL)
    escalated = manager.escalate_pending()
    assert len(escalated) >= 1
    assert "ESCALATED" in escalated[0]["message"]
    # sent to a second channel (the one the original alert also reached) — at least one delivery
    assert any("ESCALATED" in a["message"] for a in mem2.alerts)


def test_escalate_only_once(manager):
    manager.escalation_minutes = 0.0
    manager.alert("circuit_breaker", "dd", severity=CRITICAL)
    manager.escalate_pending()
    second = manager.escalate_pending()
    assert second == []


# ═════════════════════════════════════════════════════════════════════════════
# Integration helpers
# ═════════════════════════════════════════════════════════════════════════════

def test_notify_drift_feature_event(manager):
    res = notify_drift_event(manager, {"type": "feature_drift", "feature": "rsi_14",
                                       "drift": True})
    assert res["alert_type"] == "drift_detected"
    assert "rsi_14" in res["message"]
    assert manager.channels[0].alerts  # type: ignore[attr-defined]


def test_notify_drift_model_event(manager):
    res = notify_drift_event(manager, {"type": "model_drift", "event": "challenger_losing",
                                       "challenger": "mamba_v2"})
    assert res["alert_type"] == "model_drift"
    assert "mamba_v2" in res["message"]


def test_notify_risk_violation(manager):
    res = notify_risk_violation(manager, {"rule": "max_position_pct", "value": 0.12,
                                          "action": "reject"})
    assert res["alert_type"] == "risk_violation"
    assert res["severity"] == CRITICAL
    assert "max_position_pct" in res["message"]
