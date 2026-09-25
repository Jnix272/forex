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

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# v1: n_trades>=10, sharpe>0.5, return>0 (consensus only; per-agent losses ignored)
# v2: >=1000 trades, DD<=15%, every agent profitable, fold reasons block certification
# v3: certificates carry SHA-256 of every certified artifact and are verified on
#     load; supervised/ensemble certificates come from PromotionGate on the
#     cached holdout (per-observation PSR/DSR).
PROMOTION_GATE_VERSION = 3

CERTIFIED_STATUS = "CERTIFIED_READY_FOR_DEPLOYMENT"

# Thresholds shared with PromotionGate (one source of truth). The RL
# certification applies the same Sharpe / drawdown floors as the supervised gate.
from validation.promotion_gate import GateConfig as _GateConfig  # noqa: E402

_GC = _GateConfig()
MIN_TRADES = max(1000, int(_GC.min_trades))
MIN_SHARPE = float(_GC.min_sharpe)
MIN_RETURN_PCT = 0.0
MAX_DRAWDOWN_PCT = min(15.0, float(_GC.max_drawdown_pct) * 100.0)
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


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_artifacts(paths: Iterable[str | Path]) -> dict[str, str]:
    """{resolved path: sha256} for the files that exist."""
    out: dict[str, str] = {}
    for p in paths:
        pp = Path(p)
        if pp.is_file():
            out[str(pp.resolve())] = sha256_file(pp)
    return out


def certificate_from_gate(gate_result: Mapping[str, Any], artifacts: Iterable[str | Path], **extra) -> dict:
    """Wrap a PromotionGate.evaluate() result as a verifiable certificate.

    The certificate is bound to the exact files it certifies (weights, scaler,
    feature sidecar): editing it or placing it next to a different checkpoint
    fails verification.
    """
    promoted = bool(gate_result.get("promoted"))
    failed = [k for k, v in (gate_result.get("gates") or {}).items() if not v]
    hashes = hash_artifacts(artifacts)
    reasons = [] if promoted else (failed or list(gate_result.get("reasons") or ["not promoted"]))
    if promoted and not hashes:
        reasons = ["no artifacts to certify"]
    return {
        "status": CERTIFIED_STATUS if promoted and hashes else "REJECTED",
        "quality_gate_passed": bool(promoted and hashes),
        "rejection_reasons": reasons,
        "gate_version": PROMOTION_GATE_VERSION,
        "artifact_hashes": hashes,
        "promoted": bool(promoted and hashes),
        "gate": dict(gate_result),
        **extra,
    }


def check_gate_artifact(doc: Mapping[str, Any], verify_files: bool = True) -> tuple[bool, str]:
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
    hashes = doc.get("artifact_hashes") or {}
    if not hashes:
        return False, "no artifact_hashes (certificate not bound to any checkpoint)"
    if verify_files:
        for path, digest in hashes.items():
            if not Path(path).is_file():
                return False, f"certified artifact missing: {path}"
            if sha256_file(path) != digest:
                return False, f"artifact changed since certification: {path}"
    return True, "ok"
