from validation.gate_policy import (
    CERTIFIED_STATUS,
    PROMOTION_GATE_VERSION,
    agent_rejection_reasons,
    check_gate_artifact,
)


def _cert(**over):
    doc = {
        "gate_version": PROMOTION_GATE_VERSION,
        "status": CERTIFIED_STATUS,
        "quality_gate_passed": True,
        "rejection_reasons": [],
    }
    doc.update(over)
    return doc


def test_current_certificate_passes():
    assert check_gate_artifact(_cert()) == (True, "ok")


def test_legacy_certificate_without_version_is_rejected():
    # Shape of the 2026-09-21 artifact that certified two agents losing >100%.
    legacy = {"promoted": True, "details": {"status": CERTIFIED_STATUS}, "summary": "PASS"}
    ok, why = check_gate_artifact(legacy)
    assert not ok and "gate_version" in why


def test_old_version_rejected():
    ok, _ = check_gate_artifact(_cert(gate_version=PROMOTION_GATE_VERSION - 1))
    assert not ok


def test_rejection_reasons_block_certificate():
    ok, _ = check_gate_artifact(_cert(rejection_reasons=["fold variance"]))
    assert not ok


def test_promoted_flag_alone_is_not_enough():
    ok, _ = check_gate_artifact({"gate_version": PROMOTION_GATE_VERSION, "promoted": True})
    assert not ok


def test_losing_agent_blocks_ensemble():
    agents = [
        {"agent_id": 0, "eval_return_pct": -122.5, "max_drawdown_pct": 262.0},
        {"agent_id": 1, "eval_return_pct": 3.0, "max_drawdown_pct": 3.3},
    ]
    reasons = agent_rejection_reasons(agents)
    assert len(reasons) == 2 and all("Agent 0" in r for r in reasons)
    assert agent_rejection_reasons(agents[1:]) == []
