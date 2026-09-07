"""Numerically-safe math helpers shared across the codebase."""

from __future__ import annotations

import numpy as np


def safe_corrcoef(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Correlation matrix that returns 0.0 instead of NaN when any input is constant.

    Drop-in replacement for np.corrcoef — suppresses the 'invalid value in divide'
    RuntimeWarning that fires when stddev == 0, and converts the resulting NaNs to 0.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.corrcoef(a, b)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
