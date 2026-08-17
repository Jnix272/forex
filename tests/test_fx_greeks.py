"""
Tests for risk.fx_greeks (Improvement #16): FX option greeks, JPY quoting,
and portfolio aggregation.
"""

from __future__ import annotations

import pytest

from risk.fx_greeks import (
    FxOptionGreeks,
    PortfolioGreeks,
    compute_greeks,
)

# ═════════════════════════════════════════════════════════════════════════════
# Single-option greeks
# ═════════════════════════════════════════════════════════════════════════════


def test_atm_call_delta_positive():
    g = compute_greeks("EURUSD", spot=1.10, strike=1.10, tenor_years=0.5, vol=0.10)
    assert g.delta > 0.0
    assert 0.4 < g.delta < 0.6


def test_atm_put_delta_negative():
    call = compute_greeks("EURUSD", spot=1.10, strike=1.10, tenor_years=0.5, vol=0.10)
    put = compute_greeks("EURUSD", spot=1.10, strike=1.10, tenor_years=0.5, vol=0.10, call=False)
    assert put.delta < 0.0
    # Black-76 put-call delta parity: Δ_call − Δ_put = exp(−r_f·t)  # noqa: RUF003
    import math

    assert (call.delta - put.delta) == pytest.approx(math.exp(-0.01 * 0.5), abs=0.01)


def test_gamma_positive():
    g = compute_greeks("EURUSD", spot=1.10, strike=1.10, tenor_years=0.5, vol=0.10)
    assert g.gamma > 0.0


def test_vega_positive():
    g = compute_greeks("EURUSD", spot=1.10, strike=1.10, tenor_years=0.5, vol=0.10)
    assert g.vega > 0.0


def test_theta_sign():
    g = compute_greeks("EURUSD", spot=1.10, strike=1.10, tenor_years=0.5, vol=0.10)
    # ATM options lose value with time (theta < 0 in price terms)
    assert g.theta > 0.0  # helper returns -theta_raw; theta_raw is negative for ATM


def test_jpy_delta_scaled():
    g_usd = compute_greeks("EURUSD", spot=1.10, strike=1.10, tenor_years=1.0, vol=0.10)
    g_jpy = compute_greeks("USDJPY", spot=150.0, strike=150.0, tenor_years=1.0, vol=0.10)
    # delta in base units is ~N(d1); JPY quoting scales delta by 1/spot.
    assert 0.0 < g_jpy.delta < 0.01
    assert g_jpy.is_jpy is True
    assert g_usd.is_jpy is False


def test_quantity_scales():
    g1 = compute_greeks("EURUSD", spot=1.10, strike=1.10, tenor_years=0.5, vol=0.10, quantity=2.0)
    g2 = compute_greeks("EURUSD", spot=1.10, strike=1.10, tenor_years=0.5, vol=0.10, quantity=1.0)
    assert g1.delta == pytest.approx(2.0 * g2.delta)
    assert g1.vega == pytest.approx(2.0 * g2.vega)


def test_zero_tenor_raises():
    with pytest.raises(ValueError):
        FxOptionGreeks(spot=1.10, strike=1.10, tenor_years=0.0, vol=0.10).greeks()


def test_price_atm_positive():
    g = compute_greeks("EURUSD", spot=1.10, strike=1.10, tenor_years=0.5, vol=0.10)
    assert g.price > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Portfolio aggregation
# ═════════════════════════════════════════════════════════════════════════════


def _leg(pair="EURUSD", spot=1.10, strike=1.10, quantity=1.0, call=True):
    return {
        "pair": pair,
        "spot": spot,
        "strike": strike,
        "tenor_years": 0.5,
        "vol": 0.10,
        "quantity": quantity,
        "call": call,
    }


def test_portfolio_aggregate_sums_greeks():
    pf = PortfolioGreeks([_leg("EURUSD"), _leg("GBPUSD", spot=1.30, strike=1.30)])
    agg = pf.aggregate()
    assert agg["n_legs"] == 2
    assert agg["delta"] == pytest.approx(
        compute_greeks("EURUSD", 1.10, 1.10, 0.5, 0.10).delta + compute_greeks("GBPUSD", 1.30, 1.30, 0.5, 0.10).delta,
        abs=1e-4,
    )
    assert len(agg["legs"]) == 2


def test_portfolio_delta_netting():
    pf = PortfolioGreeks([_leg("EURUSD", quantity=1.0), _leg("EURUSD", quantity=1.0, call=False)])
    agg = pf.aggregate()
    # ATM call + ATM put deltas roughly net (within rate-asymmetry slack)
    assert abs(agg["delta"]) < 0.12


def test_net_delta_by_currency():
    pf = PortfolioGreeks([_leg("EURUSD"), _leg("EURGBP", spot=0.85, strike=0.85)])
    nd = pf.net_delta_by_currency()
    assert "EUR" in nd
    assert nd["EUR"] == pytest.approx(
        compute_greeks("EURUSD", 1.10, 1.10, 0.5, 0.10).delta + compute_greeks("EURGBP", 0.85, 0.85, 0.5, 0.10).delta,
        abs=1e-4,
    )


def test_aggregate_empty_book():
    agg = PortfolioGreeks().aggregate()
    assert agg["n_legs"] == 0
    assert agg["delta"] == 0.0
