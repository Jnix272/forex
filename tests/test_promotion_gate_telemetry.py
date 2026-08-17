"""Tests for the F1 challenger gate clarity fix + promotion telemetry.

Audit finding F1 (training/post_train.py:627-628):
  The challenger vs production gate used `min_delta = -0.001` for the loss
  case combined with comparator `> prod_metric + min_delta`. The negative
  min_delta cancels with the `+` so the logic was functionally equivalent
  to a correct gate with `+0.001` and `- min_delta`, BUT relied on a fragile
  sign-cancellation trick that any future edit could silently break.

Fix:
  - Use positive min_delta=0.001 for both sharpe and loss directions.
  - For loss case: reject if `metric_val > prod_metric - min_delta`
    (challenger must be lower than prod by at least min_delta).
  - Wire on_promotion_decision JSONL + CSV telemetry.

Tests use pure-python reimplementations to verify the contract without
importing the torch-heavy training module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Comparator semantics (the contract)
# ---------------------------------------------------------------------------


def old_challenger_accepts(metric_val, prod_metric, use_sharpe) -> bool:
    """The OLD logic. Used for equivalence verification."""
    min_delta_old = 0.001 if use_sharpe else -0.001
    if (use_sharpe and metric_val < prod_metric + min_delta_old) or (  # noqa: SIM103
        not use_sharpe and metric_val > prod_metric + min_delta_old
    ):
        return False  # rejected
    return True  # accepted


def new_challenger_accepts(metric_val, prod_metric, use_sharpe) -> bool:
    """The fixed logic. Positive min_delta for both directions."""
    min_delta_new = 0.001
    if use_sharpe:
        return not (metric_val < prod_metric + min_delta_new)
    else:
        return not (metric_val > prod_metric - min_delta_new)


# ---------------------------------------------------------------------------
# Equivalence: old and new logic produce identical decisions
# ---------------------------------------------------------------------------


def test_old_and_new_produce_identical_decisions_on_loss_boundary():
    """The negative min_delta in the old logic cancels with `+ min_delta`,
    making old ≡ new. Verify the contract holds so that future edits
    don't accidentally break the sign cancellation in either direction.
    """
    test_cases = [
        # (use_sharpe, prod, challenger)
        (False, 0.6, 0.59),
        (False, 0.6, 0.595),
        (False, 0.6, 0.599),
        (False, 0.6, 0.5995),
        (False, 0.6, 0.6),
        (False, 0.6, 0.6005),
        (False, 0.6, 0.601),
        (False, 0.6, 0.605),
        # audit edge case: prod=1.0
        (False, 1.0, 0.99),
        (False, 1.0, 0.998),
        (False, 1.0, 0.999),
        (False, 1.0, 0.9995),
        (False, 1.0, 1.0),
        (False, 1.0, 1.0005),
        # Sharpe direction
        (True, 1.5, 1.4),
        (True, 1.5, 1.499),
        (True, 1.5, 1.5),
        (True, 1.5, 1.5005),
        (True, 1.5, 1.599),
        (True, 1.5, 1.6),
    ]
    for use_sharpe, prod, ch in test_cases:
        old_ok = old_challenger_accepts(ch, prod, use_sharpe)
        new_ok = new_challenger_accepts(ch, prod, use_sharpe)
        assert old_ok == new_ok, (
            f" Old({ch}, {prod}, us={use_sharpe})={old_ok} but new({ch}, {prod}, us={use_sharpe})={new_ok}"
        )


def test_fix_accepts_better_sharpe_challenger():
    """Higher sharpe than prod by >= min_delta should be accepted."""
    assert new_challenger_accepts(1.5, 1.4, True)  # +0.1, big win
    assert new_challenger_accepts(1.401, 1.4, True)  # +0.001 boundary
    assert not new_challenger_accepts(1.4, 1.4, True)
    assert not new_challenger_accepts(1.0, 1.4, True)


def test_fix_accepts_better_loss_challenger():
    """Lower loss than prod by >= min_delta should be accepted."""
    assert new_challenger_accepts(0.500, 0.600, False)
    assert new_challenger_accepts(0.599, 0.600, False)
    assert not new_challenger_accepts(0.600, 0.600, False)
    assert not new_challenger_accepts(0.700, 0.600, False)


def test_rejects_slight_regression_for_loss():
    """Challenger with loss slightly higher than prod is rejected."""
    prod_loss = 1.0
    assert not new_challenger_accepts(1.0008, prod_loss, False)  # regression
    assert not new_challenger_accepts(1.005, prod_loss, False)


# ---------------------------------------------------------------------------
# Verify the fix is in the source
# ---------------------------------------------------------------------------


def test_post_train_uses_positive_min_delta():
    """Ensure the buggy `min_delta = -0.001` is gone from post_train.py."""
    post_train_path = _ROOT / "training" / "post_train.py"
    if not post_train_path.exists():
        pytest.skip("training/post_train.py not found")

    src = post_train_path.read_text(encoding="utf-8")
    assert "min_delta = 0.001 if use_sharpe else -0.001" not in src, (
        "F1 regression: min_delta must be +0.001 (positive) for both directions"
    )
    assert "min_delta = 0.001" in src


def test_post_train_loss_comparator_uses_minus():
    """The fixed loss comparator must subtract min_delta (lower is better)."""
    post_train_path = _ROOT / "training" / "post_train.py"
    if not post_train_path.exists():
        pytest.skip("training/post_train.py not found")

    src = post_train_path.read_text(encoding="utf-8")
    assert "metric_val > prod_metric - min_delta" in src, "Loss comparator must subtract min_delta (lower is better)"


# ---------------------------------------------------------------------------
# Telemetry: on_promotion_decision method exists and emits CSV
# ---------------------------------------------------------------------------


def test_on_promotion_decision_method_exists():
    try:
        from monitoring.train_logger import TrainingLogger
    except Exception as e:
        pytest.skip(f"could not import TrainingLogger: {e}")
    assert hasattr(TrainingLogger, "on_promotion_decision")
    assert callable(TrainingLogger.on_promotion_decision)


def test_on_promotion_decision_emits_csv(tmp_path):
    """Verify on_promotion_decision writes to promotion_decisions.csv."""
    try:
        from monitoring.train_logger import TrainingLogger
    except Exception as e:
        pytest.skip(f"could not import TrainingLogger: {e}")

    logger = TrainingLogger.__new__(TrainingLogger)
    logger.log_dir = tmp_path
    logger.model_name = "test_model"
    logger.run_name = "test_run"
    logger.verbose = False
    logger.sidecar = None
    logger._log = None
    logger._jlog = None
    logger._log_path = None
    logger._discord = None
    logger._discord_ready = False
    logger._watchdog = None
    logger._heartbeat_enabled = False
    logger._ep_oom_count = 0
    logger._ep_nan_count = 0
    logger._ep_nan_grads = []
    logger._epoch_history = []
    logger._errors = []
    logger._warnings = []
    logger._start_ts = 0.0
    logger._current_epoch = 0
    logger._total_epochs = 0

    captured_events: list = []
    logger._write_event = lambda evt, data: captured_events.append((evt, data))
    logger._safe_log = lambda lvl, msg: None
    logger.info = lambda msg: None
    logger._discord_send = lambda evt, fields, **kw: None

    logger.on_promotion_decision(
        model_name="haelt_alpha",
        promoted=True,
        metric_name="val_sharpe",
        metric_value=1.85,
        gate_summary="PROMOTE ✅",
        gate_reasons=["sharpe_ok: ✓", "pf_ok: ✓"],
        gate_details={"sharpe": 1.85, "psr": 0.97},
        challenger_vs_prod={"prod_metric": 1.7, "challenger_metric": 1.85, "direction": "sharpe", "accepted": True},
    )

    # JSONL event emitted
    assert len(captured_events) == 1
    evt_name, evt_data = captured_events[0]
    assert evt_name == "promotion_decision"
    assert evt_data["model"] == "haelt_alpha"
    assert evt_data["promoted"] is True
    assert evt_data["metric_name"] == "val_sharpe"
    assert evt_data["metric_value"] == 1.85
    assert "challenger_vs_prod" in evt_data

    # CSV row written
    csv_path = tmp_path / "promotion_decisions.csv"
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "haelt_alpha" in csv_text
    assert "ts,model,promoted" in csv_text.splitlines()[0]


def test_on_promotion_decision_handles_reject(tmp_path):
    """Verify rejected decisions are also logged."""
    try:
        from monitoring.train_logger import TrainingLogger
    except Exception as e:
        pytest.skip(f"could not import TrainingLogger: {e}")

    logger = TrainingLogger.__new__(TrainingLogger)
    logger.log_dir = tmp_path
    logger.model_name = "test_model"
    logger.run_name = "test_run"
    logger.verbose = False
    logger.sidecar = None
    logger._log = None
    logger._jlog = None
    logger._log_path = None
    logger._discord = None
    logger._discord_ready = False
    logger._watchdog = None
    logger._heartbeat_enabled = False
    logger._ep_oom_count = 0
    logger._ep_nan_count = 0
    logger._ep_nan_grads = []
    logger._epoch_history = []
    logger._errors = []
    logger._warnings = []
    logger._start_ts = 0.0
    logger._current_epoch = 0
    logger._total_epochs = 0

    captured_events: list = []
    logger._write_event = lambda evt, data: captured_events.append((evt, data))
    logger._safe_log = lambda lvl, msg: None
    logger.info = lambda msg: None
    logger._discord_send = lambda evt, fields, **kw: None

    logger.on_promotion_decision(
        model_name="haelt_alpha",
        promoted=False,
        metric_name="val_loss",
        metric_value=0.78,
        gate_summary="REJECT ❌ Failed: sharpe, pf",
        gate_reasons=["sharpe: ✗", "pf: ✗"],
        gate_details={"sharpe": 0.8, "pf": 1.0},
        challenger_vs_prod={
            "prod_metric": 0.6,
            "challenger_metric": 0.78,
            "direction": "loss",
            "rejected": True,
            "reason": "loss higher than prod",
        },
    )

    assert len(captured_events) == 1
    evt_name, evt_data = captured_events[0]
    assert evt_name == "promotion_decision"
    assert evt_data["promoted"] is False
    assert evt_data["metric_name"] == "val_loss"

    csv_path = tmp_path / "promotion_decisions.csv"
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "haelt_alpha" in csv_text


# ---------------------------------------------------------------------------
# Source-level: telemetry wiring in post_train.py
# ---------------------------------------------------------------------------


def test_post_train_emits_on_promotion_decision_after_gate():
    """Verify post_train.py calls on_promotion_decision after the gate runs."""
    post_train_path = _ROOT / "training" / "post_train.py"
    if not post_train_path.exists():
        pytest.skip("training/post_train.py not found")

    src = post_train_path.read_text(encoding="utf-8")
    assert "on_promotion_decision" in src, "post_train.py should emit on_promotion_decision telemetry"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
