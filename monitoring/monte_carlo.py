"""Monte Carlo backtest — randomise trade order for Sharpe CI."""

from __future__ import annotations

import numpy as np

__all__ = ["MonteCarloBacktest"]


class MonteCarloBacktest:
    """
    Randomise trade order 1000x to get confidence intervals on Sharpe,
    max drawdown, and total return.

    Two simulation approaches:
      1. shuffle: Randomly permute the trade returns list.
      2. bootstrap: Sample trades with replacement (block_length=1).
    """

    def __init__(
        self,
        n_simulations: int = 1_000,
        confidence: float = 0.95,
        bars_per_year: int = 252 * 1440,
        seed: int = 42,
    ):
        self.n_sim = n_simulations
        self.conf = confidence
        self.bpy = bars_per_year
        self.rng = np.random.default_rng(seed)

    def run(self, trade_returns: np.ndarray, method: str = "shuffle") -> dict:
        """Run Monte Carlo simulation. Returns statistics including confidence intervals."""
        from evaluation.monte_carlo import block_bootstrap_indices

        arr = np.asarray(trade_returns, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        n = len(arr)
        sharpes, drawdowns, tot_returns = [], [], []

        if method == "bootstrap":
            idx = block_bootstrap_indices(
                n,
                block_length=1,
                n_bootstraps=self.n_sim,
                seed=int(self.rng.integers(0, 2**31 - 1)),
            )
            sims = (arr[row] for row in idx)
        else:
            sims = (self.rng.permutation(arr) for _ in range(self.n_sim))

        for sim in sims:
            cum = np.cumprod(1 + sim)
            total_r = float(cum[-1] - 1)
            sr = float(sim.mean() / sim.std() * np.sqrt(self.bpy / n * n)) if sim.std() > 0 else 0.0
            peak = np.maximum.accumulate(cum)
            mdd = float((cum / peak - 1).min())
            sharpes.append(sr)
            drawdowns.append(mdd)
            tot_returns.append(total_r)

        a = (1 - self.conf) / 2
        sharpes_arr = np.array(sharpes)
        drawdowns_arr = np.array(drawdowns)
        tot_r_arr = np.array(tot_returns)

        return {
            "method": method,
            "n_simulations": self.n_sim,
            "sharpe_mean": round(float(sharpes_arr.mean()), 4),
            "sharpe_ci": [
                round(float(np.percentile(sharpes_arr, a * 100)), 4),
                round(float(np.percentile(sharpes_arr, (1 - a) * 100)), 4),
            ],
            "drawdown_mean": round(float(drawdowns_arr.mean()), 4),
            "drawdown_ci": [
                round(float(np.percentile(drawdowns_arr, a * 100)), 4),
                round(float(np.percentile(drawdowns_arr, (1 - a) * 100)), 4),
            ],
            "total_return_mean": round(float(tot_r_arr.mean()), 4),
            "total_return_ci": [
                round(float(np.percentile(tot_r_arr, a * 100)), 4),
                round(float(np.percentile(tot_r_arr, (1 - a) * 100)), 4),
            ],
            "pct_positive_sharpe": round(float((sharpes_arr > 0).mean()), 4),
            "confidence": self.conf,
        }
