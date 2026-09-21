"""
backtesting/gpu_backtester.py
=============================
A blazing-fast vectorized backtesting engine leveraging CuPy/JAX to simulate
trading strategies across millions of ticks instantly on the GPU.
"""

import logging
from typing import Any

import numpy as np

try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    cp = None


def _to_numpy(array: Any) -> np.ndarray:
    """Convert a NumPy or CuPy array back to a NumPy array when available."""
    if cp is not None and hasattr(array, "get"):
        return np.asarray(array.get())
    return np.asarray(array)


class GPUBacktester:
    def __init__(self, use_gpu: bool = True):
        self.logger = logging.getLogger(__name__)
        if use_gpu and cp is not None:
            self.xp = cp
            self.logger.info("GPU Backtester initialized with CuPy.")
        else:
            self.xp = np
            if use_gpu:
                self.logger.warning("CuPy not found or use_gpu=False. Falling back to NumPy (CPU).")

    def run_vectorized_backtest(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        spread: float = 0.0001,
        commission_per_lot: float = 3.5,
        slippage_pips: float = 0.7,
        pip_size: float = 0.0001,
        lot_size: float = 0.1,
    ):
        """
        Run a massive vectorized backtest.
        `prices` and `signals` should be 1D arrays of equal length.
        """
        # Transfer data to GPU
        d_prices = self.xp.array(prices)
        d_signals = self.xp.array(signals)

        # Calculate returns safely without zero division: return[i] = (price[i+1] - price[i]) / price[i]
        denom_prev = self.xp.maximum(d_prices[:-1], 1e-12)
        d_returns = self.xp.diff(d_prices) / denom_prev

        # BUG-007: Proper 1-bar lag - signal[i] trades return[i+1], not return[i].
        # d_returns[i] is the return from bar i to i+1. signal[i] is the signal from bar i.
        # To execute at bar i+1 open, we need return from i+1 to i+2 = d_returns[i+1]
        d_positions = d_signals[:-2]
        d_returns = d_returns[1:]

        # Strategy returns
        d_strat_returns = d_positions * d_returns

        # Denominator for transaction cost fractions
        denom_mid = self.xp.maximum(d_prices[1:-1], 1e-12)

        # Incorporate spread costs whenever position changes
        d_trades = self.xp.abs(self.xp.diff(d_positions, prepend=0))
        d_spread_costs = d_trades * ((spread * 0.5) / denom_mid)

        # Commission: per lot per trade (scaled by standard lot notional 100,000 * price)
        d_commission = d_trades * (commission_per_lot / (100_000.0 * denom_mid))

        # Slippage: fixed pips per trade
        d_slippage = d_trades * (slippage_pips * pip_size / denom_mid)

        # Net returns
        d_net_returns = d_strat_returns - d_spread_costs - d_commission - d_slippage

        # Calculate equity curve with bankruptcy clamping (no zombie compounding)
        d_equity = self.xp.cumprod(1 + d_net_returns)
        neg_mask = (d_equity <= 0) | self.xp.isnan(d_equity)
        if self.xp.any(neg_mask):
            first_neg = int(self.xp.argmax(neg_mask))
            d_equity[first_neg:] = 0.0

        # Transfer results back to CPU
        return {
            "total_return": float((d_equity[-1] - 1) * 100) if len(d_equity) > 0 else 0.0,
            "equity_curve": _to_numpy(d_equity),
            "num_trades": int(self.xp.sum(d_trades)),
        }
