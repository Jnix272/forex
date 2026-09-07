"""Power-law slippage calibrator fitted to real fill data."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["SlippageCalibrator"]


class SlippageCalibrator:
    """
    Fit a power-law slippage model to actual LMAX fill data.

    Model: slippage_pips = α × (size / ADV)^β

    Parameters: α (coefficient), β (exponent, typically 0.5 for square-root)
    """

    def __init__(self):
        self.alpha: float | None = None
        self.beta: float | None = None
        self._fitted = False

    def fit(self, fill_data: pd.DataFrame, adv_lots: float = 1000.0) -> dict:
        """Fit power-law model. fill_data must have: order_size_lots, actual_slippage_pips."""
        size_pct = fill_data["order_size_lots"] / adv_lots
        slip = fill_data["actual_slippage_pips"].clip(lower=0)

        mask = (size_pct > 0) & (slip > 0)
        if mask.sum() < 5:
            print("[Slippage] Insufficient data - using defaults α=0.1, β=0.5")
            self.alpha = 0.1
            self.beta = 0.5
            return {"alpha": 0.1, "beta": 0.5, "r_squared": None}

        log_size = np.log(size_pct[mask].values)
        log_slip = np.log(slip[mask].values)
        coeffs = np.polyfit(log_size, log_slip, 1)
        self.beta = float(coeffs[0])
        self.alpha = float(np.exp(coeffs[1]))

        pred_log = np.polyval(coeffs, log_size)
        ss_res = ((log_slip - pred_log) ** 2).sum()
        ss_tot = ((log_slip - log_slip.mean()) ** 2).sum()
        r2 = float(1 - ss_res / (ss_tot + 1e-9))

        self._fitted = True
        print(f"[Slippage] Calibrated: α={self.alpha:.4f} β={self.beta:.4f} R²={r2:.3f}")
        return {"alpha": self.alpha, "beta": self.beta, "r_squared": round(r2, 4)}

    def predict(self, size_lots: float, adv_lots: float = 1000.0) -> float:
        alpha = self.alpha or 0.1
        beta = self.beta or 0.5
        pct = max(size_lots / adv_lots, 1e-9)
        return float(alpha * (pct**beta))

    def fit_synthetic(self) -> dict:
        rng = np.random.default_rng(42)
        n = 500
        sizes = rng.exponential(0.1, n).clip(0.01, 2.0)
        slip = 0.1 * (sizes / 1000) ** 0.5 + rng.normal(0, 0.002, n)
        df = pd.DataFrame({"order_size_lots": sizes, "actual_slippage_pips": np.clip(slip, 0, None)})
        return self.fit(df)
