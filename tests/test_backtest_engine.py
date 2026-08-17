"""
Regression tests for the production backtesting engine.

Covers the audit findings:
  1. SL/TP values survive when a strategy supplies them only on signal rows.
  2. Missing / zero-valued stop_loss / take_profit columns are treated as
     "not configured" - no instance of the catastrophic partial-TP at a
     negative price and no phantom stop-outs.
  3. performance_metrics falls back to equity-curve stats (no RuntimeError)
     when Trade records are unavailable (the Numba path).
  4. run() is idempotent.
  5. Python and Numba execution paths agree on the same data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.backtest import _NUMBA_OK, ForexScalingBacktest, ScalingAction


def _make_bars(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0002, n))
    return pd.DataFrame(
        {
            "open": close.copy(),
            "high": close + 0.0004,
            "low": close - 0.0004,
            "close": close,
            "volume": 100.0,
            "bid_close": close - 0.00005,
            "ask_close": close + 0.00005,
            "spread_avg": np.full(n, 0.0001),
        },
        index=idx,
    )


def _base_signals(idx: pd.DatetimeIndex) -> pd.DataFrame:
    sig = pd.DataFrame(columns=["action", "lots", "stop_loss", "take_profit"], index=idx)
    sig["action"] = 0
    sig["lots"] = 0.1
    sig.loc[idx[100], ["action", "stop_loss", "take_profit"]] = [int(ScalingAction.OPEN_LONG), 1.0980, 1.1030]
    sig.loc[idx[150], "action"] = int(ScalingAction.CLOSE_ALL)
    return sig


def test_sl_tp_survive_when_only_on_signal_rows():
    """Regression: PIPE-009 masking used to erase SL/TP exactly at the signal bar."""
    bars = _make_bars()
    bt = ForexScalingBacktest(bars=bars, signals=_base_signals(bars.index), execution_delay_bars=0)
    bt.run()
    assert len(bt.trades) == 1
    t = bt.trades[0]
    assert t.stop_loss == pytest.approx(1.0980, abs=1e-12), t.stop_loss
    assert t.take_profit == pytest.approx(1.1030, abs=1e-12), t.take_profit
    assert t.exit_reason in ("signal_exit", "stop_loss", "scale_out_tp")


def test_missing_sl_tp_columns_do_not_cause_catastrophic_fills():
    """No stop_loss/take_profit columns -> disabled, not a -hh price TP fill."""
    bars = _make_bars()
    sig = pd.DataFrame(index=bars.index)
    sig["action"] = 0
    sig["lots"] = 0.1
    sig.loc[bars.index[100], "action"] = int(ScalingAction.OPEN_LONG)
    bt = ForexScalingBacktest(bars=bars, signals=sig, initial_equity=10_000, execution_delay_bars=0)
    bt.run()
    assert len(bt.trades) == 1
    t = bt.trades[0]
    assert t.entry_price > 0.0
    assert t.exit_reason in ("end_of_data", "signal_exit")
    assert abs(t.gross_pnl_usd) < 50.0, f"phantom TP/stop caused PnL {t.gross_pnl_usd}"


def test_zero_stop_tp_are_ignored():
    """Explicit 0.0 stop/TP (missing config) must not be treated as reachable prices."""
    bars = _make_bars()
    sig = pd.DataFrame(columns=["action", "lots", "stop_loss", "take_profit"], index=bars.index)
    sig["action"] = 0
    sig["lots"] = 0.1
    sig.loc[bars.index[100], ["action", "stop_loss", "take_profit"]] = [int(ScalingAction.OPEN_LONG), 0.0, 0.0]
    bt = ForexScalingBacktest(bars=bars, signals=sig, initial_equity=10_000, execution_delay_bars=0)
    bt.run()
    assert len(bt.trades) == 1
    assert bt.trades[0].exit_reason == "end_of_data"


def test_performance_metrics_falls_back_without_trade_records():
    """Numba-path state (empty trades + populated results_df) must not raise."""
    bars = _make_bars()
    bt = ForexScalingBacktest(bars=bars, signals=_base_signals(bars.index), execution_delay_bars=0)
    bt.run()
    bt.trades = []
    m = bt.performance_metrics()
    assert m.get("metrics_source") == "equity_curve"
    assert "n_trades" in m and float(m["n_trades"]) == 0.0
    assert m.get("warning") and "equity-curve fallback" in m["warning"]


def test_run_is_idempotent():
    bars = _make_bars()
    bt = ForexScalingBacktest(
        bars=bars, signals=_base_signals(bars.index), initial_equity=10_000, execution_delay_bars=0
    )
    bt.run()
    first_equity, first_trades = bt.equity, len(bt.trades)
    bt.run()
    assert len(bt.trades) == first_trades
    assert bt.equity == pytest.approx(first_equity, abs=1e-9)


@pytest.mark.skipif(not _NUMBA_OK, reason="numba unavailable")
def test_python_and_numba_paths_agree():
    bars = _make_bars(n=2000)
    sig = _base_signals(bars.index)
    py = ForexScalingBacktest(bars=bars, signals=sig, initial_equity=10_000, execution_delay_bars=0)
    py_res = py.run(use_numba=False)

    # Force the JIT core for the same book by lowering the numba threshold.
    nb = ForexScalingBacktest(bars=bars, signals=sig, initial_equity=10_000, execution_delay_bars=0)
    nb.numba_min_bars = 1000
    nb_res = nb.run(use_numba=True, return_trades=False)

    np.testing.assert_allclose(
        nb_res["total_value"].dropna().to_numpy(),
        py_res["total_value"].dropna().to_numpy(),
        rtol=1e-9,
        atol=1e-6,
    )
