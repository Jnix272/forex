"""
features/engineering/session.py
Session clock and time-of-day features.
These are typically computed inline in FeatureEngineer.build() via numpy/pandas,
but this module provides reusable helpers for standalone use.
"""

import numpy as np
import polars as pl


def tod_sin_cos(timestamps_pd) -> tuple:
    """Return (time_sin, time_cos) numpy arrays for a pandas DatetimeIndex/Series."""
    h = timestamps_pd.dt.hour
    m = timestamps_pd.dt.minute
    tm = h * 60 + m
    return np.sin(2 * np.pi * tm / 1440), np.cos(2 * np.pi * tm / 1440)


def dow_sin_cos(timestamps_pd) -> tuple:
    """Return (day_sin, day_cos) numpy arrays for a pandas DatetimeIndex/Series."""
    dow = timestamps_pd.dt.dayofweek
    return np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5)


def london_ny_session(timestamps_pd) -> np.ndarray:
    """Fixed-UTC London/NY overlap approximation (13:00-17:00 UTC)."""
    h = timestamps_pd.dt.hour
    return ((h >= 13) & (h <= 17)).astype(float).to_numpy()
