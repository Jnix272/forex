"""
Tests for evaluation.monte_carlo (Improvement #3):
block bootstrap, stationary bootstrap, PathMonteCarlo, TradeSequenceMonteCarlo.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.monte_carlo import (
    block_bootstrap,
    block_bootstrap_indices,
    stationary_bootstrap,
    stationary_bootstrap_indices,
    pl_block_bootstrap,
    PathMonteCarlo,
    TradeSequenceMonteCarlo,
    Trade,
    summarize_simulation,
    monte_carlo_backtest,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    return rng.normal(0.0005, 0.01, 500)


@pytest.fixture
def pos_trend_returns():
    """Small positive-drift series so Sharpe > 0 is detectable."""
    rng = np.random.default_rng(1)
    return rng.normal(0.001, 0.005, 1000)


@pytest.fixture
def trades():
    return [Trade(entry=i * 5, exit=i * 5 + 5, size=1.0) for i in range(40)]


# ═════════════════════════════════════════════════════════════════════════════
# Resampling primitives
# ═════════════════════════════════════════════════════════════════════════════

def test_block_bootstrap_indices_shape_and_contiguity():
    idx = block_bootstrap_indices(n=100, block_length=10, n_bootstraps=50, seed=7)
    assert idx.shape == (50, 100)
    assert idx.dtype == np.int64
    assert idx.min() >= 0 and idx.max() < 100
    # First block of each row must be contiguous run of length 10
    assert (idx[0, :10] == idx[0, 0] + np.arange(10)).all()


def test_block_bootstrap_indices_reproducible_with_seed():
    a = block_bootstrap_indices(80, 8, 10, seed=42)
    b = block_bootstrap_indices(80, 8, 10, seed=42)
    np.testing.assert_array_equal(a, b)


def test_block_bootstrap_indices_validation():
    with pytest.raises(ValueError):
        block_bootstrap_indices(n=0, block_length=5, n_bootstraps=10)
    with pytest.raises(ValueError):
        block_bootstrap_indices(n=100, block_length=0, n_bootstraps=10)
    with pytest.raises(ValueError):
        block_bootstrap_indices(n=100, block_length=5, n_bootstraps=0)


def test_block_bootstrap_values_preserve_set():
    data = np.arange(10.0)
    resampled = block_bootstrap(data, block_length=3, n_bootstraps=20, seed=3)
    assert resampled.shape == (20, 10)
    # Values are a permutation-with-replacement of the original set
    assert set(np.unique(resampled)) <= set(data)


def test_stationary_bootstrap_indices_shape():
    idx = stationary_bootstrap_indices(n=100, avg_block_length=10, n_bootstraps=30, seed=5)
    assert idx.shape == (30, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_stationary_bootstrap_reproducible_with_seed():
    a = stationary_bootstrap_indices(80, 12, 10, seed=99)
    b = stationary_bootstrap_indices(80, 12, 10, seed=99)
    np.testing.assert_array_equal(a, b)


def test_stationary_bootstrap_validation():
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(n=10, avg_block_length=0, n_bootstraps=5)
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(n=10, avg_block_length=3, n_bootstraps=0)


# ═════════════════════════════════════════════════════════════════════════════
# Polars-native bootstrap
# ═════════════════════════════════════════════════════════════════════════════

def test_pl_block_bootstrap(pytestconfig):
    pl = pytest.importorskip("polars")
    s = pl.Series("ret", np.arange(50.0))
    df = pl_block_bootstrap(s, block_length=5, n_bootstraps=4, seed=1)
    assert df.shape == (50, 4)
    assert df.columns == [f"boot_{i}" for i in range(4)]
    assert df["boot_0"].to_numpy().min() >= 0.0
    assert df["boot_0"].to_numpy().max() < 50.0


# ═════════════════════════════════════════════════════════════════════════════
# summarize_simulation
# ═════════════════════════════════════════════════════════════════════════════

def test_summarize_simulation_bounds(returns):
    sim = PathMonteCarlo(n_simulations=50, seed=1)
    res = sim.run(returns)
    assert res["sharpe_5th"] <= res["sharpe_median"] <= res["sharpe_95th"]
    assert res["max_drawdown_5th"] <= res["max_drawdown_median"] <= res["max_drawdown_95th"]
    assert 0.0 <= res["prob_sharpe_negative"] <= 1.0
    assert res["n_simulations"] == 50
    assert res["confidence"] == 0.95


def test_summarize_simulation_empty():
    res = summarize_simulation([])
    assert res == {"n_simulations": 0}


# ═════════════════════════════════════════════════════════════════════════════
# PathMonteCarlo
# ═════════════════════════════════════════════════════════════════════════════

def test_path_mc_default_buy_and_hold(pos_trend_returns):
    sim = PathMonteCarlo(n_simulations=100, seed=1)
    res = sim.run(pos_trend_returns)
    assert res["strategy"] == "buy_and_hold"
    assert res["total_return_mean"] > 0.0
    assert res["prob_sharpe_negative"] < 0.5


def test_path_mc_invalid_bootstrap():
    with pytest.raises(ValueError):
        PathMonteCarlo(bootstrap="circular")


def test_path_mc_strategy_shape_mismatch(returns):
    sim = PathMonteCarlo(strategy=lambda r: np.ones(len(r) + 5), n_simulations=10, seed=1)
    with pytest.raises(ValueError):
        sim.run(returns)


def test_path_mc_stationary_variant(pos_trend_returns):
    sim = PathMonteCarlo(
        n_simulations=50, bootstrap="stationary", block_length=10, seed=2
    )
    res = sim.run(pos_trend_returns)
    assert res["method"] == "path_stationary_bootstrap"


def test_path_mc_reproducible_with_seed(pos_trend_returns):
    a = PathMonteCarlo(n_simulations=30, seed=123).run(pos_trend_returns)
    b = PathMonteCarlo(n_simulations=30, seed=123).run(pos_trend_returns)
    assert a["sharpe_mean"] == b["sharpe_mean"]
    assert a["sharpe_ci"] == b["sharpe_ci"]


def test_path_mc_requires_min_length():
    sim = PathMonteCarlo(n_simulations=5, seed=1)
    with pytest.raises(ValueError):
        sim.run([0.0])


def test_monte_carlo_backtest_convenience(pos_trend_returns):
    res = monte_carlo_backtest(pos_trend_returns, n_simulations=30, seed=4)
    assert "sharpe_ci" in res
    assert "max_drawdown_ci" in res
    assert res["method"] == "path_block_bootstrap"


# ═════════════════════════════════════════════════════════════════════════════
# TradeSequenceMonteCarlo
# ═════════════════════════════════════════════════════════════════════════════

def test_trade_sequence_mc_runs(returns, trades):
    sim = TradeSequenceMonteCarlo(n_simulations=50, seed=1)
    res = sim.run(returns, trades)
    assert res["n_trades"] == len(trades)
    assert res["method"] == "trade_sequence_block_bootstrap"
    assert 0.0 <= res["prob_sharpe_negative"] <= 1.0


def test_trade_sequence_mc_stationary(returns, trades):
    sim = TradeSequenceMonteCarlo(
        n_simulations=30, bootstrap="stationary", block_length=5, seed=2
    )
    res = sim.run(returns, trades)
    assert res["method"] == "trade_sequence_stationary_bootstrap"


def test_trade_sequence_mc_empty_trades(returns):
    sim = TradeSequenceMonteCarlo(n_simulations=10, seed=1)
    with pytest.raises(ValueError):
        sim.run(returns, [])


def test_trade_sequence_mc_clamps_out_of_bounds(returns):
    trades = [Trade(entry=10_000, exit=10_005, size=1.0)]
    sim = TradeSequenceMonteCarlo(n_simulations=10, seed=1)
    res = sim.run(returns, trades)
    assert res["n_trades"] == 1


def test_trade_sequence_mc_from_pnls(returns):
    pnls = np.random.default_rng(0).normal(0.5, 1.0, 20)
    trades = TradeSequenceMonteCarlo.from_pnls(pnls, returns)
    assert len(trades) == 20
    sim = TradeSequenceMonteCarlo(n_simulations=20, seed=1)
    res = sim.run(returns, trades)
    assert res["n_trades"] == 20


# ═════════════════════════════════════════════════════════════════════════════
# Legacy facade shims (D2): evaluation.monte_carlo delegates preserve the
# legacy MonteCarloBacktest result schemas so existing callers keep working.
# ═════════════════════════════════════════════════════════════════════════════

def test_improvements_legacy_run_keys():
    from backtesting.improvements import MonteCarloBacktest
    pnls = np.random.default_rng(1).normal(50.0, 80.0, 40)
    mc = MonteCarloBacktest(n_simulations=50, random_seed=7)
    res = mc.run(pnls, annual_factor=252)
    for key in ["n_trades", "n_simulations", "original_sharpe", "original_max_dd",
                "sharpe_5th", "sharpe_median", "sharpe_95th", "sharpe_percentile",
                "max_dd_5th", "max_dd_median", "max_dd_95th", "prob_sharpe_above_1",
                "prob_sharpe_above_0", "robust", "method", "warning"]:
        assert key in res
    assert res["n_trades"] == 40
    assert res["method"] == "bootstrap_with_replacement"


def test_improvements_legacy_empty_result():
    from backtesting.improvements import MonteCarloBacktest
    mc = MonteCarloBacktest(n_simulations=10, random_seed=7)
    res = mc.run(np.array([10.0]))
    assert res["robust"] is False
    assert res["warning"] == "Need at least 2 closed trades for Monte Carlo"
    assert res["n_trades"] == 1


def test_improvements_legacy_run_from_backtest():
    from backtesting.improvements import MonteCarloBacktest

    class FakeBT:
        _trade_pnls = np.random.default_rng(2).normal(20.0, 40.0, 30)

    mc = MonteCarloBacktest(n_simulations=30, random_seed=7)
    res = mc.run_from_backtest(FakeBT())
    assert res["n_trades"] == 30


def test_pipeline_legacy_run_schemas():
    from monitoring.pipeline import MonteCarloBacktest
    rets = np.random.default_rng(3).normal(0.0005, 0.005, 200)
    mc = MonteCarloBacktest(n_simulations=40, seed=7)
    for method in ("shuffle", "bootstrap"):
        res = mc.run(rets, method=method)
        for key in ["method", "n_simulations", "sharpe_mean", "sharpe_ci",
                    "drawdown_mean", "drawdown_ci", "total_return_mean",
                    "total_return_ci", "pct_positive_sharpe", "confidence"]:
            assert key in res
        assert res["method"] == method
        assert res["n_simulations"] == 40
        assert isinstance(res["sharpe_ci"], list) and len(res["sharpe_ci"]) == 2
        assert isinstance(res["drawdown_ci"], list) and len(res["drawdown_ci"]) == 2
