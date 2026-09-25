"""
Comprehensive verification test suite for Quantitative Backtesting Audit fixes (BUG-01 to BUG-10).
"""

from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import ForexScalingBacktest, ScalingAction, _NUMBA_OK
from backtesting.execution import AdvancedBacktestEngine, LatencyModel
from backtesting.gpu_backtester import GPUBacktester
from evaluation.metrics import (
    _expected_max_of_normals,
    backtest_metrics,
    calmar_ratio,
    minimum_backtest_length,
)
from evaluation.monte_carlo import PathMonteCarlo, _equity_path_from_returns, summarize_simulation
from visualize_backtest import load_real_backtest_logs


def _make_bars(n: int = 500, start_px: float = 1.1000) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = np.full(n, start_px)
    return pd.DataFrame(
        {
            "open": close.copy(),
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
            "volume": 100.0,
            "bid_close": close - 0.00005,
            "ask_close": close + 0.00005,
            "spread_avg": np.full(n, 0.0001),
        },
        index=idx,
    )


# -----------------------------------------------------------------------------
# BUG-01: Numba path datetime object type crash fixed
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not _NUMBA_OK, reason="numba unavailable")
def test_bug01_numba_path_runs_without_pyobject_crash():
    bars = _make_bars(n=1200)
    sig = pd.DataFrame(index=bars.index)
    sig["action"] = 0
    sig["lots"] = 0.1
    sig.loc[bars.index[100], "action"] = int(ScalingAction.OPEN_LONG)
    sig.loc[bars.index[200], "action"] = int(ScalingAction.CLOSE_ALL)

    bt = ForexScalingBacktest(bars=bars, signals=sig, initial_equity=10_000, execution_delay_bars=0)
    bt.numba_min_bars = 1000
    res = bt.run(use_numba=True, return_trades=False)
    assert res is not None
    assert len(res) == len(bars)
    assert bt.equity > 0


# -----------------------------------------------------------------------------
# BUG-02: GPUBacktester commission scaling and ruin clamping
# -----------------------------------------------------------------------------
def test_bug02_gpu_backtester_commission_and_ruin():
    bt = GPUBacktester(use_gpu=False)
    prices = np.array([1.1000, 1.1000, 1.1000, 1.1000, 1.1000])
    signals = np.array([1.0, 1.0, 0.0, 0.0, 0.0])

    res = bt.run_vectorized_backtest(
        prices=prices,
        signals=signals,
        spread=0.0,
        commission_per_lot=3.5,
        slippage_pips=0.0,
        lot_size=0.1,
    )
    # Commission per trade should be ~ 3.5 / (100_000 * 1.1) ~= 3.18e-5 (0.00318%)
    # Not 0.35 / 1.1 ~= 31.8%!
    # Total return with zero price movement should be -0.003% to -0.01%, NOT -60%!
    assert -0.05 < res["total_return"] <= 0.0, f"Commission return penalty too extreme: {res['total_return']}%"

    # Test ruin clamping (no zombie compounding)
    prices_crash = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    signals_crash = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    res_crash = bt.run_vectorized_backtest(
        prices=prices_crash,
        signals=signals_crash,
        spread=0.0,
        commission_per_lot=0.0,
        slippage_pips=15000.0,
        pip_size=0.0001,
    )
    # Without clamping: cumprod([-0.5, -0.5]) = +0.25 (zombie resurrection)
    # With clamping: equity must stay 0.0
    eq = res_crash["equity_curve"]
    assert eq[-1] == 0.0


# -----------------------------------------------------------------------------
# BUG-03: LatencyModel lognormal parameter conversion
# -----------------------------------------------------------------------------
def test_bug03_latency_model_lognormal_parameters():
    lat = LatencyModel(network_mean_us=500.0, network_std_us=100.0, gateway_fixed_us=0.0, gateway_jitter_us=0.0)
    samples = [lat.sample_submission_latency() for _ in range(2000)]
    mean_val = float(np.mean(samples))
    # Target is ~500 us. Under the bug, it was ~0.0 us.
    assert 400.0 <= mean_val <= 600.0, f"Mean latency was {mean_val} us, expected ~500 us"


# -----------------------------------------------------------------------------
# BUG-04: Position flipping (Long -> Short, Short -> Long)
# -----------------------------------------------------------------------------
def test_bug04_position_flipping():
    bars = _make_bars(n=300)
    sig = pd.DataFrame(index=bars.index)
    sig["action"] = 0
    sig["lots"] = 0.1
    # Open long at bar 10, then directly open short at bar 20 without explicit CLOSE_ALL
    sig.loc[bars.index[10], "action"] = int(ScalingAction.OPEN_LONG)
    sig.loc[bars.index[20], "action"] = int(ScalingAction.OPEN_SHORT)
    sig.loc[bars.index[30], "action"] = int(ScalingAction.CLOSE_ALL)

    bt = ForexScalingBacktest(bars=bars, signals=sig, initial_equity=10_000, execution_delay_bars=0)
    bt.run(use_numba=False)
    # Should have 2 closed trades: the long trade closed by the flip, and the short trade closed by CLOSE_ALL
    assert len(bt.trades) == 2
    assert bt.trades[0].direction == 1
    assert bt.trades[0].exit_reason == "flip_to_short"
    assert bt.trades[1].direction == -1
    assert bt.trades[1].exit_reason == "signal_exit"


# -----------------------------------------------------------------------------
# BUG-05: Gap-bar stop loss fill clamping
# -----------------------------------------------------------------------------
def test_bug05_gap_bar_stop_loss_clamping():
    bars = _make_bars(n=50, start_px=1.1000)
    # At bar 15, bar gaps down by 50 pips (open = 1.0950, high = 1.0955, low = 1.0940)
    bars.loc[bars.index[15], "open"] = 1.0950
    bars.loc[bars.index[15], "high"] = 1.0955
    bars.loc[bars.index[15], "low"] = 1.0940
    bars.loc[bars.index[15], "close"] = 1.0945

    sig = pd.DataFrame(index=bars.index)
    sig["action"] = 0
    sig["lots"] = 0.1
    # Open long at bar 5 with stop loss at 1.0980 (20 pips risk)
    sig.loc[bars.index[5], ["action", "stop_loss", "take_profit"]] = [int(ScalingAction.OPEN_LONG), 1.0980, 1.1100]

    bt = ForexScalingBacktest(bars=bars, signals=sig, initial_equity=10_000, execution_delay_bars=0, slippage_pips=0.0)
    bt.run(use_numba=False)

    assert len(bt.trades) == 1
    t = bt.trades[0]
    assert t.exit_reason == "stop_loss"
    # Fills at bar open (1.0950) on gap bar, not at phantom 1.0980, minus half the
    # spread: a long stop sells at the bid (2026-09-25 audit B2).
    _half_spread = 0.5 * float(bt._arr_spread[15]) if bt._arr_spread is not None else 0.0
    assert t.exit_price == pytest.approx(1.0950 - _half_spread, abs=1e-5)


# -----------------------------------------------------------------------------
# BUG-06: Partial scale out weighted average exit price and pnl pips
# -----------------------------------------------------------------------------
def test_bug06_partial_scale_out_weighted_averaging():
    bars = _make_bars(n=100, start_px=1.1000)
    bars.loc[bars.index[20], ["open", "high", "low", "close"]] = 1.1050
    bars.loc[bars.index[30], ["open", "high", "low", "close"]] = 1.1100

    sig = pd.DataFrame(index=bars.index)
    sig["action"] = 0
    sig["lots"] = 1.0
    sig.loc[bars.index[10], "action"] = int(ScalingAction.OPEN_LONG)
    sig.loc[bars.index[20], "action"] = int(ScalingAction.SCALE_OUT_50)  # exit 0.5 lot @ 1.1050
    sig.loc[bars.index[30], "action"] = int(ScalingAction.CLOSE_ALL)     # exit remaining 0.5 lot @ 1.1100

    bars["spread_avg"] = 0.0
    bt = ForexScalingBacktest(
        bars=bars,
        signals=sig,
        initial_equity=10_000,
        execution_delay_bars=0,
        slippage_pips=0.0,
        use_bid_ask=False,
        apply_market_impact=False,
    )
    bt.run(use_numba=False)

    assert len(bt.trades) == 1
    t = bt.trades[0]
    assert t.entry_lots == pytest.approx(1.0)
    assert t.exit_lots == pytest.approx(1.0)
    # Weighted exit price: (0.5 * 1.1050 + 0.5 * 1.1100) / 1.0 = 1.1075
    assert t.exit_price == pytest.approx(1.1075, abs=1e-5)
    # PnL pips = (1.1075 - 1.1000) / 0.0001 = 75 pips
    assert t.pnl_pips == pytest.approx(75.0, abs=1e-3)


# -----------------------------------------------------------------------------
# BUG-07: AdvancedBacktestEngine commission deduction & mark-to-market unrealized PnL
# -----------------------------------------------------------------------------
def test_bug07_advanced_engine_commission_and_mtm():
    bars = _make_bars(n=20, start_px=1.1000)
    # Price rises to 1.1020 (+20 pips)
    for i in range(5, 15):
        bars.loc[bars.index[i], ["close", "bid_close", "ask_close"]] = 1.1020

    signals = pd.DataFrame(index=bars.index)
    signals["direction"] = 0.0
    signals.loc[bars.index[3:14], "direction"] = 1.0   # Hold Buy across bars 3-13
    signals.loc[bars.index[14], "direction"] = 0.0     # Close

    engine = AdvancedBacktestEngine(
        bars=bars,
        signals=signals,
        config={
            "initial_equity": 10_000.0,
            "lot_size": 1.0,
            "commission_per_lot": 5.0,
            "pip_value_per_lot": 10.0,
        },
    )
    res = engine.run()
    # While in position at bar 8, unrealised PnL must be reflected in equity
    eq_rows = res.to_dicts() if hasattr(res, "to_dicts") else res.to_dict(orient="records")
    bar8 = eq_rows[8]
    assert bar8["unrealised_pnl"] > 0.0
    assert bar8["equity"] > bar8["cash"]
    # Trade commission must be deducted
    assert len(engine.trades) >= 1
    assert engine.trades[0].commission >= 5.0


# -----------------------------------------------------------------------------
# BUG-08: Marcos López de Prado expected max of normals & guarded calmar
# -----------------------------------------------------------------------------
def test_bug08_expected_max_and_calmar_guards():
    # Expected max of 2 Gaussians: exact is 1/sqrt(pi) ~= 0.564
    em2 = _expected_max_of_normals(2)
    assert 0.50 <= em2 <= 0.60, f"Expected max for N=2 was {em2}, expected ~0.564"

    # Expected max of 10 Gaussians: exact is ~1.538
    em10 = _expected_max_of_normals(10)
    assert 1.45 <= em10 <= 1.65, f"Expected max for N=10 was {em10}, expected ~1.538"

    # Calmar guard on negative wealth
    ruined_returns = np.array([-0.90, -0.50])
    c = calmar_ratio(ruined_returns)
    assert np.isfinite(c)
    assert c < 0.0


# -----------------------------------------------------------------------------
# BUG-09: annual_factor forwarding & Monte Carlo prob_ruin
# -----------------------------------------------------------------------------
def test_bug09_min_bt_length_and_monte_carlo_ruin():
    # Min backtest length with high annual factor
    m1 = minimum_backtest_length(1.5, 0.0, 3.0, annual_factor=252)
    m2 = minimum_backtest_length(1.5, 0.0, 3.0, annual_factor=252 * 24)
    assert m2 > m1  # Higher frequency requires more bars for same annualized Sharpe

    # Monte Carlo bankruptcy clamping
    returns_ruin = np.array([-1.05, 0.50, 0.50])
    eq_path = _equity_path_from_returns(returns_ruin, 10_000.0)
    # Once bankrupt, stays bankrupt
    assert eq_path[-1] == 0.0

    # PathMonteCarlo summary includes prob_ruin
    sim = PathMonteCarlo(n_simulations=10, seed=42)
    res = sim.run(returns_ruin)
    assert "prob_ruin" in res
    assert res["prob_ruin"] == 1.0


# -----------------------------------------------------------------------------
# BUG-10: visualize_backtest timestamp parsing
# -----------------------------------------------------------------------------
def test_bug10_visualize_backtest_entry_time_parsing(tmp_path, monkeypatch):
    test_dir = tmp_path / "logs" / "backtests"
    test_dir.mkdir(parents=True)
    trades_csv = test_dir / "model_20260101_trades.csv"
    equity_csv = test_dir / "model_20260101_equity.csv"

    trades_df = pd.DataFrame({
        "trade_id": [1],
        "entry_time": ["2026-01-01 10:00:00"],
        "exit_time": ["2026-01-01 10:15:00"],
        "pnl_usd": [50.0],
    })
    equity_df = pd.DataFrame({
        "timestamp": ["2026-01-01 10:00:00", "2026-01-01 10:15:00"],
        "equity": [10000.0, 10050.0],
    })
    trades_df.to_csv(trades_csv, index=False)
    equity_df.to_csv(equity_csv, index=False)

    monkeypatch.chdir(tmp_path)
    res = load_real_backtest_logs("model")
    assert res is not None
    loaded_trades, loaded_equity, _ = res
    assert pd.api.types.is_datetime64_any_dtype(loaded_trades["entry_time"])
    assert pd.api.types.is_datetime64_any_dtype(loaded_trades["exit_time"])
