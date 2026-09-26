"""Store coverage check: a tick store with missing session hours must not be used as-is."""

import pandas as pd
import polars as pl

from data.sources import ForexDataManager


def _ticks(hours):
    ts = [pd.Timestamp(h, tz="UTC") for h in hours]
    return pl.DataFrame({"timestamp_utc": ts, "bid": [1.0] * len(ts), "ask": [1.0001] * len(ts)})


def test_store_covers_full_and_partial_day():
    m = ForexDataManager(verbose=False)
    full = _ticks([f"2019-03-11 {h:02d}:30" for h in range(7, 18)])  # Monday, all 11 session hours
    assert m._store_covers(full, "2019-03-11", "2019-03-11", session_only=True)
    partial = _ticks([f"2019-03-11 {h:02d}:30" for h in range(7, 12)])
    assert not m._store_covers(partial, "2019-03-11", "2019-03-11", session_only=True)
    assert not m._store_covers(pl.DataFrame(), "2019-03-11", "2019-03-11", session_only=True)
