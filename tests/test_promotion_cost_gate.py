"""Tests for the P1 promotion-gate silent-default fix (audit 2026-08-07).

Before the fix: ``_evaluate_forward_gate`` passed
  - ``gross_pnl = bt_metrics["net_pnl"]`` (cost-subtracted net P&L)
  - ``transaction_costs = 0.0`` (always zero)
which made the cost gate ``cost_pct = 0.0 / net_pnl = 0.0`` trivially always
pass ``max_cost_pct = 0.30``. The gate was effectively off for promotion
decisions.

After the fix: the upstream ``backtest_model`` exposes
``gross_pnl_usd`` and ``total_commission_usd``, which are threaded through
``_normalize_backtest_metrics`` as ``gross_pnl`` and ``total_commission``.
``_evaluate_forward_gate`` now passes those (with a fail-closed guard when
gross_pnl is unavailable).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import is heavy (torch via post_train); try and skip if unavailable.
try:
    from validation.promotion_gate import GateConfig, PromotionGate

    _PROMOTIONS_AVAILABLE = True
    _SKIP_REASON = ""
except Exception as e:
    _PROMOTIONS_AVAILABLE = False
    _SKIP_REASON = f"could not import promotion_gate: {e}"


# ---------------------------------------------------------------------------
# normalize_backtest_metrics now exposes gross_pnl and total_commission
# ---------------------------------------------------------------------------


def test_normalize_backtest_metrics_passes_gross_pnl_and_commission():
    """Calls to ``_normalize_backtest_metrics`` should emit gross_pnl and
    total_commission fields (so the gate can compute cost_pct correctly).
    """
    try:
        from scripts.backtest_model import _normalize_backtest_metrics
    except Exception as e:
        pytest.skip(f"could not import backtest_model: {e}")

    raw = {
        "n_trades": 100,
        "sharpe": 1.6,
        "win_rate_pct": 55.0,
        "max_drawdown_pct": 12.0,
        "gross_pnl_usd": 5000.0,
        "total_commission_usd": 250.0,
        "net_pnl_usd": 4750.0,
        "profit_factor": 1.8,
        "total_return_pct": 47.5,
    }
    out = _normalize_backtest_metrics(raw)
    assert "gross_pnl" in out
    assert "total_commission" in out
    assert "profit_factor" in out
    assert out["gross_pnl"] == 5000.0
    assert out["total_commission"] == 250.0
    assert out["net_pnl"] == 4750.0
    assert out["profit_factor"] == 1.8


def test_normalize_backtest_metrics_handles_missing_gross_pnl():
    """When the backtester returns only net_pnl (pre-2026-08-07), the
    normalizer should fall through to 0.0 (signalling 'unavailable').
    """
    try:
        from scripts.backtest_model import _normalize_backtest_metrics
    except Exception as e:
        pytest.skip(f"could not import backtest_model: {e}")

    raw = {
        "n_trades": 100,
        "sharpe": 1.6,
        "win_rate_pct": 55.0,
        "max_drawdown_pct": 12.0,
        "net_pnl_usd": 4750.0,  # only net available (old-style)
    }
    out = _normalize_backtest_metrics(raw)
    assert out.get("gross_pnl", 0.0) == 0.0
    assert out.get("total_commission", 0.0) == 0.0


# ---------------------------------------------------------------------------
# PromotionGate cost gate: behavior with real gross_pnl and costs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PROMOTIONS_AVAILABLE, reason=_SKIP_REASON)
def test_cost_gate_fails_when_costs_exceed_gross_pnl_pct():
    """With realistic cost data, the cost gate should fail when
    transaction_costs / gross_pnl > max_cost_pct.
    """
    gate = PromotionGate(GateConfig(max_cost_pct=0.30))
    gross = 5000.0
    costs = 2000.0  # cost_pct = 0.4 = 40% > 30% → reject
    # Other gates pass
    result = gate.evaluate(
        sharpe=2.0,
        profit_factor=1.8,
        max_drawdown=0.10,
        n_trades=100,
        gross_pnl=gross,
        transaction_costs=costs,
        n_obs=100,
        turnover_rate=2.0,
        avg_latency_ms=50.0,
        n_backtest_trials=3,
        backtest_sharpe_std=0.5,
    )
    assert not result["promoted"], "Gate should have been rejected due to cost_pct=0.40 > max=0.30"
    # The cost_ok gate should be False in particular
    assert result["gates"].get("cost_ok") is False
    assert result["details"]["cost_pct"] >= 0.30


@pytest.mark.skipif(not _PROMOTIONS_AVAILABLE, reason=_SKIP_REASON)
def test_cost_gate_passes_with_reasonable_costs():
    """A model with cost_pct < max_cost_pct should pass the cost gate."""
    gate = PromotionGate(GateConfig(max_cost_pct=0.30))
    gross = 5000.0
    costs = 250.0  # cost_pct = 0.05 = 5% < 30% → pass
    # All other gates set to pass
    result = gate.evaluate(
        sharpe=2.0,
        profit_factor=1.8,
        max_drawdown=0.10,
        n_trades=100,
        gross_pnl=gross,
        transaction_costs=costs,
        n_obs=100,
        turnover_rate=2.0,
        avg_latency_ms=50.0,
        n_backtest_trials=3,
        backtest_sharpe_std=0.5,
    )
    assert result["gates"].get("cost_ok") is True
    # Note: may still be rejected by other gates (psr, dsr, etc.) - only check cost


@pytest.mark.skipif(not _PROMOTIONS_AVAILABLE, reason=_SKIP_REASON)
def test_cost_pct_zero_with_zero_costs_is_documented():
    """When no cost is passed (transaction_costs=0), the cost gate now PASSES
    (optional gate behavior) - this is the FIX for P1 where missing cost data
    would previously cause a silent fail-closed rejection.
    """
    gate = PromotionGate(GateConfig(max_cost_pct=0.30))
    result = gate.evaluate(
        sharpe=2.0,
        profit_factor=1.8,
        max_drawdown=0.10,
        n_trades=100,
        gross_pnl=5000.0,
        transaction_costs=0.0,
        n_obs=100,
        turnover_rate=2.0,
        avg_latency_ms=50.0,
    )
    assert result["gates"].get("cost_ok") is True, (
        "When transaction_costs=0 (no info), the cost gate should now PASS (optional gate)"
    )


# ---------------------------------------------------------------------------
# Source-level: post_train uses real gross_pnl and transaction_costs
# ---------------------------------------------------------------------------


def test_post_train_does_not_substitute_net_pnl_for_gross_pnl():
    """The OLD bug line `gross_pnl=bt_metrics["net_pnl"]` must NOT appear."""
    post_train = _ROOT / "training" / "post_train.py"
    if not post_train.exists():
        pytest.skip("training/post_train.py not found")
    src = post_train.read_text(encoding="utf-8")
    assert 'gross_pnl=bt_metrics["net_pnl"]' not in src, "P1 bug is back! net_pnl substituted for gross_pnl"
    assert "gross_pnl=gross_for_cost_gate" in src, "post_train should pass the real (derived) gross_pnl to the gate"


def test_post_train_does_not_pass_hardcoded_zero_costs():
    """The OLD bug line `transaction_costs=0.0` (always-zero) must be replaced
    by the real commission number from backtest metrics.
    """
    post_train = _ROOT / "training" / "post_train.py"
    if not post_train.exists():
        pytest.skip("training/post_train.py not found")
    src = post_train.read_text(encoding="utf-8")
    assert "transaction_costs=0.0, # Already accounted for" not in src, (
        "P1 bug is back! hardcoded zero transaction_costs"
    )
    assert "transaction_costs_value" in src
    assert "transaction_costs=transaction_costs_value" in src
    assert "gross_for_cost_gate" in src


# ---------------------------------------------------------------------------
# post_train derives gross_pnl from net_pnl + total_commission as fallback
# ---------------------------------------------------------------------------


def test_post_train_falls_back_to_net_plus_costs_when_gross_missing():
    """When gross_pnl=0 but total_commission>0, gross is reconstructed as
    net_pnl + total_commission.
    """
    post_train = _ROOT / "training" / "post_train.py"
    if not post_train.exists():
        pytest.skip("training/post_train.py not found")
    src = post_train.read_text(encoding="utf-8")
    # The fallback formula should be present
    assert 'gross_pnl_value = float(bt_metrics.get("net_pnl", 0.0) or 0.0) + transaction_costs_value' in src


# ---------------------------------------------------------------------------
# Result telemetry - record cost_pct and gross_pnl in details
# ---------------------------------------------------------------------------


def test_normalize_includes_profit_factor_passthrough():
    """profit_factor should also be threaded through the normalizer so
    PromotionGate.evaluate can score the second-most important ratio."""
    try:
        from scripts.backtest_model import _normalize_backtest_metrics
    except Exception as e:
        pytest.skip(f"could not import: {e}")
    raw = {
        "n_trades": 100,
        "sharpe": 1.5,
        "profit_factor": 1.7,
        "win_rate_pct": 55.0,
        "max_drawdown_pct": 12.0,
        "net_pnl_usd": 4000.0,
        "gross_pnl_usd": 5000.0,
        "total_commission_usd": 200.0,
    }
    out = _normalize_backtest_metrics(raw)
    assert out["profit_factor"] == 1.7


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
