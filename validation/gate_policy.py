"""
validation/gate_policy.py
=========================
Single source of truth for deployment-certification rules.

``scripts/auto_optimal_roadmap.py`` writes certifications with these rules and
stamps ``gate_version``; ``trading/live_engine.py`` refuses any artifact that
was produced under an older version, so a PASS issued under looser rules can
never authorise live capital after the rules are tightened.

Bump ``PROMOTION_GATE_VERSION`` whenever a threshold below is loosened or
tightened - every existing certification then becomes invalid.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# v1: n_trades>=10, sharpe>0.5, return>0 (consensus only; per-agent losses ignored)
# v2: >=1000 trades, DD<=15%, every agent profitable, fold reasons block certification
PROMOTION_GATE_VERSION = 2

CERTIFIED_STATUS = "CERTIFIED_READY_FOR_DEPLOYMENT"

MIN_TRADES = 1000
MIN_SHARPE = 0.5
MIN_RETURN_PCT = 0.0
MAX_DRAWDOWN_PCT = 15.0
MAX_CONFLICT_RATE = 0.50
MIN_AGREEMENT_SCORE = 0.50
MAX_AGENT_DRAWDOWN_PCT = 20.0


def agent_rejection_reasons(agents: list[Mapping[str, Any]] | None) -> list[str]:
    """Reject the ensemble if any individual agent lost money or blew through DD."""
    reasons: list[str] = []
    for agent in agents or []:
        aid = agent.get("agent_id", "?")
        ret = float(agent.get("eval_return_pct", float("-inf")))
        dd = float(agent.get("max_drawdown_pct", float("inf")))
        if ret <= MIN_RETURN_PCT:
            reasons.append(f"Agent {aid}: non-positive evaluation return {ret:+.2f}%")
        if dd > MAX_AGENT_DRAWDOWN_PCT:
            reasons.append(f"Agent {aid}: drawdown {dd:.2f}% > {MAX_AGENT_DRAWDOWN_PCT:.2f}%")
    return reasons


def check_gate_artifact(doc: Mapping[str, Any]) -> tuple[bool, str]:
    """Return (ok, reason) for a promotion/certification artifact loaded from JSON."""
    version = int(doc.get("gate_version", 0) or 0)
    if version < PROMOTION_GATE_VERSION:
        return False, f"gate_version={version} < required {PROMOTION_GATE_VERSION} (issued under older rules)"
    if doc.get("status") != CERTIFIED_STATUS:
        return False, f"status={doc.get('status')!r}"
    if doc.get("quality_gate_passed") is not True:
        return False, "quality_gate_passed is not true"
    if doc.get("rejection_reasons"):
        return False, f"has rejection_reasons: {doc.get('rejection_reasons')}"
    return True, "ok"
