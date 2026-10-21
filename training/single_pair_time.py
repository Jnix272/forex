"""Timestamp alignment at the single-pair cache write boundary."""
from __future__ import annotations

import numpy as np
import pandas as pd


def timestamp_rows(time_idx, n_rows: int, last_t_ns: int = -1):
    times = pd.DatetimeIndex(pd.to_datetime(time_idx, utc=True)).as_unit('ns').asi8
    if len(times) != n_rows or np.any(times == np.iinfo(np.int64).min):
        raise ValueError('Single-pair timestamps must be finite and row-aligned')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Single-pair chunk timestamps must increase strictly')
    keep = times > last_t_ns
    return times[keep], keep
