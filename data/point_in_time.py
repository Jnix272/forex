"""Leakage-resistant alignment for macro, news, COT and cross-asset observations."""

from __future__ import annotations

import pandas as pd


def align_asof_available(
    features_index,
    observations: pd.DataFrame,
    *,
    value_columns=None,
    event_time_col: str = "event_time",
    available_time_col: str = "available_time",
    default_delay: str | pd.Timedelta = "0s",
) -> pd.DataFrame:
    """Backward as-of join using when information became knowable, never event date."""
    idx = pd.DatetimeIndex(features_index)
    obs = observations.copy()
    if available_time_col not in obs:
        if event_time_col not in obs:
            raise ValueError("observations require event_time or available_time")
        obs[available_time_col] = pd.to_datetime(obs[event_time_col], utc=True) + pd.Timedelta(default_delay)
    obs[available_time_col] = pd.to_datetime(obs[available_time_col], utc=True)
    left = pd.DataFrame({"__time": pd.to_datetime(idx, utc=True), "__order": range(len(idx))})
    values = value_columns or [c for c in obs.columns if c not in {event_time_col, available_time_col}]
    right = obs[[available_time_col, *values]].sort_values(available_time_col)
    merged = pd.merge_asof(
        left.sort_values("__time"),
        right,
        left_on="__time",
        right_on=available_time_col,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("__order").set_index(pd.Index(idx))[values]


def assert_point_in_time(observations: pd.DataFrame, available_time_col="available_time") -> None:
    if available_time_col not in observations:
        raise ValueError(f"missing {available_time_col}; revised/calendar-dated data are unsafe")
    available = pd.to_datetime(observations[available_time_col], utc=True)
    if available.isna().any():
        raise ValueError("available_time contains missing or invalid timestamps")
