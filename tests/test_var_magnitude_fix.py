"""Tests for the parametric VaR cov-scaling fix (audit R-1/R-2, 2026-08-07).

Before the fix: ``parametric_var`` mixed pip-dollar weights
(``positions x pip_value`` ≈ 10) with a covariance matrix of price-fraction
returns (std ≈ 3e-4), producing ``var_usd ≈ $0.0096`` for a standard EURUSD
lot at 99% confidence - wrong by ~4 orders of magnitude vs the correct
~$96 (and ~1% of equity).

After the fix: weights are dollar-notional (£100k x lots) and returns are
price-fraction scale; units are consistent ($/lot x lots → $; cov in
(price-fraction)² → final variance in $²). VaR now comes out in dollars
with the correct magnitude.

Sanity-check math (closed-form, no simulation):
  σ_ret  = 3e-4 (daily FX return std, price-fraction scale)
  σ_ret² = 9e-8
  N      = 1.0 lot,   notional = $100,000
  Var(P&L) = N² x σ_ret² = $10¹⁰ x 9e-8 = $900
  σ_PnL = $30
  z_99% = 2.326
  VaR_1day = z x σ_PnL ≈ $69.78
"""  # noqa: RUF002

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from risk.execution import PortfolioVaR


def _build_with_returns(returns_by_pair: dict, n_obs: int = 100, confidence: float = 0.99):
    """Helper: build a PortfolioVaR with synthetic returns."""
    rng = np.random.default_rng(42)
    pv = PortfolioVaR(confidence=confidence)
    for pair, std in returns_by_pair.items():
        for _ in range(n_obs):
            pv.update_returns(pair, float(rng.normal(0, std)))
    return pv


# ---------------------------------------------------------------------------
# Magnitude sanity tests (the bug was a wrong-magnitude bug)
# ---------------------------------------------------------------------------


def test_var_usd_is_in_dollars_not_cents():
    """A standard EURUSD lot against a 99% confidence FX daily return std
    of 3e-4 should produce var_usd in the tens of dollars, not milli-cents.
    """
    pv = _build_with_returns({"EURUSD": 0.0003}, n_obs=200)
    result = pv.parametric_var({"EURUSD": 1.0}, equity=10_000.0)
    var_usd = result["var_usd"]
    # The OLD bug produced var_usd ≈ 0.0096 (literally under a cent).
    # The fix should produce var_usd ≈ $69 (z=2.326 x $100k x 3e-4 ≈ $69.78)
    assert var_usd > 1.0, (
        f"var_usd={var_usd} is too small - likely still using pip-dollar "
        "x price-fraction-cov (old bug) or notional x pips-cov (off-by-pip_size)"
    )
    assert var_usd < 1000.0, f"var_usd={var_usd} is too large - likely double-scaled"


def test_var_pct_is_a_nontrivial_fraction_of_equity():
    """var_pct = var_usd / equity should be a meaningful percentage (e.g. 0.5%)
    - the OLD bug produced var_pct ≈ 1e-6, which never tripped the 2% gate.
    """
    pv = _build_with_returns({"EURUSD": 0.0003}, n_obs=200)
    result = pv.parametric_var({"EURUSD": 1.0}, equity=10_000.0)
    var_pct = result["var_pct"]
    assert var_pct > 1e-4, f"var_pct={var_pct} is too small to ever trip a 2% daily VaR gate"
    # And realistically should be close to the closed-form ~0.0070
    expected_pct = (2.326 * 0.0003 * 100_000) / 10_000  # ≈ 0.00698
    assert abs(var_pct - expected_pct) / expected_pct < 0.30, (
        f"var_pct={var_pct} not close to closed-form {expected_pct}"
    )


def test_var_usd_scales_with_notional():
    """Double the lots → VaR should roughly double (linear in position size,
    ignoring correlation effects for a single-pair book).
    """
    pv = _build_with_returns({"EURUSD": 0.0003}, n_obs=200)
    v1 = pv.parametric_var({"EURUSD": 1.0}, equity=100_000.0)
    v2 = pv.parametric_var({"EURUSD": 2.0}, equity=100_000.0)
    assert v2["var_usd"] > 1.9 * v1["var_usd"], "Doubling lots should approximately double VaR"
    assert v2["var_usd"] < 2.1 * v1["var_usd"], "VaR scaling with lot count should be near linear (single pair)"


def test_var_usd_scales_with_return_volatility():
    """10x the return std → VaR should scale 10x (linear in σ, since z is const)."""  # noqa: RUF002
    pv_low = _build_with_returns({"EURUSD": 0.0001}, n_obs=200)
    pv_high = _build_with_returns({"EURUSD": 0.0010}, n_obs=200)
    v_low = pv_low.parametric_var({"EURUSD": 1.0}, equity=100_000.0)
    v_high = pv_high.parametric_var({"EURUSD": 1.0}, equity=100_000.0)
    assert v_high["var_usd"] > 9.5 * v_low["var_usd"], (
        f"10Ï returns should produce ~10Ã VaR (got {v_high['var_usd']} vs {v_low['var_usd']})"
    )
    assert v_high["var_usd"] < 10.5 * v_low["var_usd"]


def test_cvar_exceeds_var_for_known_distribution():
    """CVaR must be > VaR for a parametric (Gaussian) distribution by
    construction. The code uses `cvar = σ x φ(z) / (1 - conf)` (Tail Conditional
    Expectation form), which gives CVaR/VaR ratio of φ(z) / ((1-conf)·z).
    For conf=0.99, z=2.326: φ(z)=0.0264, (1-conf)·z=0.02326 → ratio ≈ 1.135.
    """  # noqa: RUF002
    pv = _build_with_returns({"EURUSD": 0.0003}, n_obs=200)
    result = pv.parametric_var({"EURUSD": 1.0}, equity=10_000.0)
    assert result["cvar_usd"] > result["var_usd"]
    ratio = result["cvar_usd"] / max(result["var_usd"], 1e-9)
    # Expected ratio ≈ 1.135; allow generous tolerance for empirical std
    assert 1.0 < ratio < 1.3, f"CVaR/VaR ratio = {ratio}, expected ~1.13"


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


def test_zero_positions_returns_zero():
    pv = _build_with_returns({"EURUSD": 0.0003}, n_obs=200)
    result = pv.parametric_var({}, equity=10_000.0)
    assert result["var_usd"] == 0.0
    assert result["var_pct"] == 0.0
    assert result["cvar_usd"] == 0.0


def test_insufficient_history_returns_zero():
    """Pairs with <20 obs are rejected → returns zero (fail-safe)."""
    pv = PortfolioVaR(confidence=0.99)
    for _ in range(5):  # too few
        pv.update_returns("EURUSD", 0.0003)
    result = pv.parametric_var({"EURUSD": 1.0}, equity=10_000.0)
    assert result["var_usd"] == 0.0
    assert result["var_pct"] == 0.0


def test_pip_scaled_returns_are_normalised_to_price_fraction():
    """The update_returns contract now accepts price-fraction returns. If a
    caller accidentally passes pip-scaled returns (e.g. 30 instead of 0.003),
    they should be normalised to keep VaR magnitude correct.
    """
    pv = PortfolioVaR(confidence=0.99)
    rng = np.random.default_rng(123)
    # "Wrong" caller passing pip-scaled returns with noise:
    for _ in range(100):
        pv.update_returns("EURUSD", 3.0 + float(rng.normal(0, 0.5)))
    result = pv.parametric_var({"EURUSD": 1.0}, equity=10_000.0)
    # After auto-normalization: magnitude should be small (≈$30 across 100k notional)
    var_usd = result["var_usd"]
    assert var_usd < 200.0, (
        f"After pip-size normalization, var_usd={var_usd} should be small; if large, the normalizer is broken"
    )
    assert var_usd > 1.0, "After normalization, var_usd should be nonzero"


def test_set_notional_overrides_default():
    """The notional per lot can be overridden per-pair for non-FX instruments."""
    pv = PortfolioVaR(confidence=0.99)
    pv.set_notional("XAUUSD", 100.0)  # 1 lot = 100 oz gold
    rng = np.random.default_rng(456)
    for _ in range(100):
        pv.update_returns("XAUUSD", float(rng.normal(0, 0.001)))  # 10 bps gold
    result = pv.parametric_var({"XAUUSD": 1.0}, equity=10_000.0)
    # $100 notional x ~1e-3 std x 2.326 z ≈ $0.23
    assert 0.05 < result["var_usd"] < 1.5, f"var_usd={result['var_usd']} out of expected range for overridden notional"


# ---------------------------------------------------------------------------
# max_allowed_lots
# ---------------------------------------------------------------------------


def test_max_allowed_lots_positive_and_bounded():
    pv = _build_with_returns({"EURUSD": 0.0003}, n_obs=200)
    lots = pv.max_allowed_lots("EURUSD", 10_000.0, {"EURUSD": 0.0})
    assert 0 < lots <= 10.0


def test_max_allowed_lots_falls_to_min_on_no_history():
    pv = PortfolioVaR(confidence=0.99)
    # No history at all (default-fill)
    lots = pv.max_allowed_lots("EURUSD", 10_000.0, {"EURUSD": 0.0})
    assert lots == 0.01


# ---------------------------------------------------------------------------
# Closed-form number verification (z·σ·notional)  # noqa: RUF003
# ---------------------------------------------------------------------------


def test_var_usd_matches_closed_form_calculation():
    """For a single pair, VaR = z x σ x N (no correlation complexity).
    Deterministic when we control the input distribution.
    """  # noqa: RUF002
    pv = PortfolioVaR(confidence=0.99)
    # Deterministic returns: 0.0 constant → std=0, so use a sine wave
    sigma_target = 0.0003
    # 200 deterministic returns with exactly that std (continuous)
    returns = [sigma_target * np.sin(i / 10.0) for i in range(200)]
    # Actual std of a sine wave with amplitude A is A / √2
    # So to get std=3e-4, set amplitude = 3e-4 x √2
    amplitude = sigma_target * np.sqrt(2.0)
    returns = [amplitude * np.sin(i / 7.5) for i in range(200)]
    for r in returns:
        pv.update_returns("EURUSD", float(r))
    result = pv.parametric_var({"EURUSD": 1.0}, equity=1_000_000.0)
    std_actual = np.std(returns, ddof=0)
    expected_var_usd = 2.326 * std_actual * 100_000 * 1.0  # z x σ x N x lots  # noqa: RUF003
    assert abs(result["var_usd"] - expected_var_usd) / expected_var_usd < 0.05, (
        f"var_usd={result['var_usd']} vs expected {expected_var_usd:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
