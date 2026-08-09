"""
backtesting/gpu_backtester.py
=============================
A blazing-fast vectorized backtesting engine leveraging CuPy/JAX to simulate
trading strategies across millions of ticks instantly on the GPU.
"""

import logging

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

class GPUBacktester:
    def __init__(self, use_gpu: bool = True):
        self.logger = logging.getLogger(__name__)
        if use_gpu and cp is not None:
            self.xp = cp
            self.logger.info("GPU Backtester initialized with CuPy.")
        else:
            self.xp = np
            self.logger.warning("CuPy not found or use_gpu=False. Falling back to NumPy (CPU).")

    def run_vectorized_backtest(self, prices: np.ndarray, signals: np.ndarray, spread: float = 0.0001):
        """
        Run a massive vectorized backtest. 
        `prices` and `signals` should be 1D arrays of equal length.
        """
        # Transfer data to GPU
        d_prices = self.xp.array(prices)
        d_signals = self.xp.array(signals)

        # Calculate returns: return[i] = (price[i+1] - price[i]) / price[i]
        d_returns = self.xp.diff(d_prices) / d_prices[:-1]

        # BUG-007: Proper 1-bar lag — signal[i] trades return[i+1], not return[i].
        # d_returns[i] is the return from bar i to i+1. signal[i] is the signal from bar i.
        # To execute at bar i+1 open, we need return from i+1 to i+2 = d_returns[i+1]
        d_positions = d_signals[:-2]
        d_returns = d_returns[1:]

        # Strategy returns
        d_strat_returns = d_positions * d_returns

        # Incorporate spread costs whenever position changes
        d_trades = self.xp.abs(self.xp.diff(d_positions, prepend=0))
        d_costs = d_trades * ((spread * 0.5) / d_prices[1:-1])

        # Net returns
        d_net_returns = d_strat_returns - d_costs

        # Calculate equity curve
        d_equity = self.xp.cumprod(1 + d_net_returns)

        # Transfer results back to CPU
        return {
            "total_return": float((d_equity[-1] - 1) * 100) if len(d_equity) > 0 else 0.0,
            "equity_curve": self.xp.asnumpy(d_equity) if hasattr(d_equity, 'get') else d_equity,
            "num_trades": int(self.xp.sum(d_trades))
        }
