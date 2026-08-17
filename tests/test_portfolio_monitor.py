"""
Tests for risk.portfolio_monitor (Improvement #16): aggregate exposure,
net currency exposure, correlation-aware exposure, liquidity tiering.
"""

from __future__ import annotations

import numpy as np
import pytest

from risk.portfolio_monitor import PortfolioMonitor


@pytest.fixture
def positions():
    return {
        "EURUSD": {"lots": 1.0, "entry_price": 1.10, "direction": "long"},
        "USDJPY": {"lots": 0.5, "entry_price": 150.0, "direction": "short"},
    }


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.001, 500)
    return {
        "EURUSD": base,
        "USDJPY": 0.9 * base + rng.normal(0.0, 0.0002, 500),
        "AUDUSD": rng.normal(0.0, 0.001, 500),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Exposure aggregation
# ═════════════════════════════════════════════════════════════════════════════


def test_aggregate_exposure_total_lots(positions):
    mon = PortfolioMonitor()
    agg = mon.aggregate_exposure(positions)
    assert agg["total_lots"] == pytest.approx(1.5)
    assert agg["n_pairs"] == 2


def test_aggregate_exposure_notional(positions):
    mon = PortfolioMonitor()
    agg = mon.aggregate_exposure(positions)
    expected = 1.0 * 100_000 * 1.10 + 0.5 * 100_000 * 150.0
    assert agg["notional_usd"] == pytest.approx(expected)


def test_net_currency_exposure(positions):
    mon = PortfolioMonitor()
    agg = mon.aggregate_exposure(positions)
    net = agg["net_currency_notional"]
    # EURUSD long 1.0 -> +EUR100k, -USD100k
    # USDJPY short 0.5 -> -USD50k, +JPY50k
    assert net["EUR"] == pytest.approx(100_000.0)
    assert net["JPY"] == pytest.approx(50_000.0)
    assert net["USD"] == pytest.approx(-150_000.0)


def test_aggregate_empty():
    mon = PortfolioMonitor()
    agg = mon.aggregate_exposure({})
    assert agg["total_lots"] == 0.0
    assert agg["n_pairs"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# Liquidity tiering
# ═════════════════════════════════════════════════════════════════════════════


def test_liquidity_exposure_tiers(positions):
    mon = PortfolioMonitor()
    liq = mon.liquidity_exposure(positions)
    # EURUSD tier 1, USDJPY tier 1 -> no illiquid exposure
    assert liq["tier_lots"].get(1) == pytest.approx(1.5)
    assert liq["illiquid_lots"] == 0.0


def test_liquidity_exposure_illiquid():
    mon = PortfolioMonitor()
    pos = {"EURJPY": {"lots": 0.4, "entry_price": 160.0, "direction": "long"}}
    liq = mon.liquidity_exposure(pos)
    assert liq["tier_lots"].get(3) == pytest.approx(0.4)
    assert liq["illiquid_lots"] == pytest.approx(0.4)
    assert liq["liquidity_adjusted_lots"] == pytest.approx(0.4 - 0.5 * 0.4)


# ═════════════════════════════════════════════════════════════════════════════
# Correlation-aware exposure
# ═════════════════════════════════════════════════════════════════════════════


def test_correlation_exposure_high_corr(positions, returns):
    mon = PortfolioMonitor(corr_threshold=0.60)
    corr = mon.correlation_exposure(positions, returns)
    assert corr["n_high_corr_edges"] >= 1
    assert any("EURUSD" in cl["pairs"] and "USDJPY" in cl["pairs"] for cl in corr["high_corr_clusters"])
    assert corr["correlation_avg"] > 0.5


def test_correlation_exposure_low_corr():
    mon = PortfolioMonitor(corr_threshold=0.95)
    rng = np.random.default_rng(1)
    returns = {
        "EURUSD": rng.normal(0.0, 0.001, 300),
        "AUDUSD": rng.normal(0.0, 0.001, 300),
    }
    positions = {
        "EURUSD": {"lots": 1.0, "entry_price": 1.1, "direction": "long"},
        "AUDUSD": {"lots": 1.0, "entry_price": 0.66, "direction": "long"},
    }
    corr = mon.correlation_exposure(positions, returns)
    assert corr["n_high_corr_edges"] == 0
    assert corr["high_corr_clusters"] == []


def test_correlation_exposure_insufficient_data(positions):
    mon = PortfolioMonitor()
    short = {p: np.array([0.0, 0.1]) for p in positions}
    corr = mon.correlation_exposure(positions, short)
    assert corr["correlation_avg"] == 0.0
    assert corr["high_corr_clusters"] == []


# ═════════════════════════════════════════════════════════════════════════════
# One-call report
# ═════════════════════════════════════════════════════════════════════════════


def test_report_combines_sections(positions, returns):
    mon = PortfolioMonitor()
    report = mon.report(positions, returns)
    assert set(report.keys()) == {"exposure", "liquidity", "correlation"}
    assert "total_lots" in report["exposure"]
    assert "tier_lots" in report["liquidity"]
    assert "max_pair_corr" in report["correlation"]


def test_report_without_returns(positions):
    mon = PortfolioMonitor()
    report = mon.report(positions)
    assert set(report.keys()) == {"exposure", "liquidity"}
