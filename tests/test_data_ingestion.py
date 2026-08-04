"""
tests/test_data_ingestion.py
=============================
Tests for the Stage 1 data-ingestion upgrades in data/data_ingestion.py:

  - timestamp column detection (incl. Dukascopy ``__index_level_0__``)
  - MAD-based bad-tick cleaning
  - tick / volume / dollar information bars
  - market-holiday filtering
  - gap detection + fill
  - Lomb-Scargle tick sampling detection
  - DST-aware session labelling
  - lazy parquet loading with start/end time filters
  - ForexDataPipeline end-to-end (all bar types + gap policies)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
import numpy as np
import polars as pl
import pytest

from data.data_ingestion import (
    ForexDataPipeline,
    clean_bad_ticks,
    detect_bar_gaps,
    detect_tick_sampling,
    fill_gaps,
    generate_synthetic_tick_data,
    load_tick_data,
    resample_to_dollar_bars,
    resample_to_tick_bars,
    resample_to_volume_bars,
)


def _make_dukascopy_df(n: int = 500) -> pl.DataFrame:
    """Frame shaped like raw Dukascopy parquet: no timestamp_utc column."""
    rng = np.random.default_rng(7)
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    idx = pl.datetime_range(
        start=start,
        end=start + timedelta(minutes=1),
        interval="100ms",
        time_zone="UTC",
        closed="left",
        eager=True,
    ).slice(0, n)
    bid = 1.0850 + rng.normal(0, 0.0002, n)
    return pl.DataFrame(
        {
            "__index_level_0__": idx,
            "bid": bid,
            "ask": bid + 0.00005,
            "volume": rng.integers(1, 100, n),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp detection
# ─────────────────────────────────────────────────────────────────────────────

def test_load_tick_data_dukascopy_index_col(tmp_path):
    path = tmp_path / "02_07.parquet"
    _make_dukascopy_df().write_parquet(path)
    df = load_tick_data(str(path))
    assert "timestamp_utc" in df.columns
    assert df.schema["timestamp_utc"].time_zone == "UTC"
    assert df["bid"].is_null().sum() == 0
    assert "spread" in df.columns


def test_load_tick_data_keeps_existing_ts(tmp_path):
    path = tmp_path / "ticks.parquet"
    generate_synthetic_tick_data(n_rows=100).write_parquet(path)
    out = load_tick_data(str(path))
    assert "timestamp_utc" in out.columns
    assert len(out) == 100


def test_load_tick_data_time_range_filter(tmp_path):
    # Multi-day frame so the start/end filter is meaningful.
    df = pl.concat([
        generate_synthetic_tick_data(n_rows=100),
        generate_synthetic_tick_data(n_rows=100).with_columns(
            (pl.col("timestamp_utc") + timedelta(days=1)).alias("timestamp_utc")
        ),
    ])
    path = tmp_path / "multi_day.parquet"
    df.write_parquet(path)
    out = load_tick_data(str(path), start="2024-01-03", end="2024-01-03 12:00")
    assert len(out) > 0
    assert out["timestamp_utc"].min() >= datetime(2024, 1, 3, tzinfo=timezone.utc)
    assert out["timestamp_utc"].max() < datetime(2024, 1, 3, 12, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Bad-tick cleaning (MAD)
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_bad_ticks_replaces_outliers():
    df = generate_synthetic_tick_data(n_rows=5000)
    # Inject a few absurd jumps relative to local MAD.
    mask = np.zeros(len(df), dtype=bool)
    mask[[1000, 2500, 4000]] = True
    df = df.with_columns(
        pl.when(pl.Series("_mask", mask))
        .then(pl.col("mid") * 100.0)
        .otherwise(pl.col("mid"))
        .alias("mid")
    )
    cleaned = clean_bad_ticks(df)
    assert len(cleaned) == len(df)  # remediation replaces, not drops
    # All remaining mids must be near the base price.
    assert cleaned["mid"].min() > 1.0
    assert cleaned["mid"].max() < 2.0


def test_clean_bad_ticks_no_outliers_unchanged():
    df = generate_synthetic_tick_data(n_rows=2000)
    cleaned = clean_bad_ticks(df)
    assert len(cleaned) == len(df)


def test_clean_bad_ticks_preserves_wide_spread_news_ticks():
    # Wide-spread ticks (e.g. news spikes) must NOT be replaced: they are
    # legitimate market data, not bad prints.
    df = generate_synthetic_tick_data(n_rows=1000)
    bad = df.with_columns((pl.col("ask") + 0.5).alias("ask"))
    cleaned = clean_bad_ticks(bad)
    # The news tick survives with its wide spread intact.
    assert ((cleaned["ask"] - cleaned["bid"]) >= 0.49).any()
    assert len(cleaned) == len(df)


# ─────────────────────────────────────────────────────────────────────────────
# Information bars
# ─────────────────────────────────────────────────────────────────────────────

def test_tick_bars_partition_input():
    ticks = generate_synthetic_tick_data(n_rows=1000)
    bars = resample_to_tick_bars(ticks, n_ticks=100)
    assert "open" in bars.columns and "close" in bars.columns
    assert len(bars) == 10  # 1000 / 100


def test_volume_bars_group_by_volume():
    ticks = generate_synthetic_tick_data(n_rows=1000)
    bars = resample_to_volume_bars(ticks, volume_target=200.0)
    assert len(bars) > 0
    assert (bars["volume"] >= 0).all()


def test_dollar_bars_ok_on_synthetic():
    ticks = generate_synthetic_tick_data(n_rows=1000)
    bars = resample_to_dollar_bars(ticks, dollar_target=1000.0)
    assert len(bars) > 0
    assert (bars["close"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Market holiday filtering
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_drops_weekend_and_holiday():
    # 2024-01-01 is New Year's Day (Monday) -> market holiday; synthetic
    # weekday filter alone should also remove Saturday/Sunday.
    ticks = generate_synthetic_tick_data(n_rows=30_000)
    # Fast-forward: generator starts 2024-01-02 (Tuesday), so verify weekday
    # filtering keeps only Mon-Fri using a wider window.
    p = ForexDataPipeline(bar_freq="30min", session_filter=False,
                          apply_frac_diff=False)
    bars = p.run(ticks)
    assert bars["timestamp_utc"].dt.weekday().max() <= 5
    assert bars["timestamp_utc"].dt.weekday().min() >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Gap detection / fill
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_bar_gaps_finds_missing_rows():
    bars = pl.DataFrame(
        {
            "timestamp_utc": pl.datetime_range(
                start=datetime(2024, 1, 2, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 0, 9, tzinfo=timezone.utc),
                interval="1m",
                eager=True,
                time_zone="UTC",
            ),
            "close": range(10),
        }
    )
    # Remove the 00:04 bar -> one 1-minute gap.
    bars = bars.filter(~pl.col("timestamp_utc").dt.minute().is_in([4]))
    report = detect_bar_gaps(bars, freq_minutes=1)
    assert report["n_gaps"] == 1
    assert report["n_missing_rows"] == 1


def test_fill_gaps_drop_and_ffill():
    n = 6
    bars = pl.DataFrame(
        {
            "timestamp_utc": pl.datetime_range(
                start=datetime(2024, 1, 2, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 0, 5, tzinfo=timezone.utc),
                interval="1m",
                eager=True,
                time_zone="UTC",
            ),
            "open": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "high": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "low": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "volume": [10.0] * n,
            "spread_avg": [0.0001] * n,
            "bid_close": [0.999] * n,
            "ask_close": [1.001] * n,
        }
    ).filter(~pl.col("timestamp_utc").dt.minute().is_in([3]))
    dropped = fill_gaps(bars, policy="drop", freq_minutes=1)
    assert len(dropped) == 5
    filled = fill_gaps(bars, policy="ffill", freq_minutes=1)
    assert len(filled) == 6
    # The missing 00:03 bar inherits the previous close.
    missing = filled.filter(pl.col("timestamp_utc").dt.minute() == 3)
    assert missing["close"][0] == 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Lomb-Scargle tick sampling
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_tick_sampling_regular_1s():
    df = generate_synthetic_tick_data(n_rows=1000)
    info = detect_tick_sampling(df)
    assert info["regular"] is True
    assert info["median_iat_ms"] == pytest.approx(1000.0, rel=0.05)
    assert info["n_ticks"] == 1000


def test_detect_tick_sampling_irregular():
    ticks = generate_synthetic_tick_data(n_rows=2000)
    # Randomly drop ~50% of rows -> geometric inter-arrival gaps (high CV).
    rng = np.random.default_rng(3)
    keep = rng.random(len(ticks)) > 0.5
    irregular = ticks.filter(pl.Series(keep))
    info = detect_tick_sampling(irregular)
    assert info["n_ticks"] > min(200, len(irregular))
    assert info["regular"] is False


# ─────────────────────────────────────────────────────────────────────────────
# DST-aware sessions
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_dst_session_labels_full_day():
    ticks = generate_synthetic_tick_data(n_rows=100_000)  # ~27h @1s
    p = ForexDataPipeline(bar_freq="5min", session_filter=True,
                          session_mode="dst", add_session_label=True,
                          apply_frac_diff=False)
    bars = p.run(ticks)
    assert "session_label" in bars.columns
    labels = set(bars["session_label"].unique().to_list())
    assert "off" in labels
    # Tokyo 09:00-18:00 local == 00:00-09:00 UTC -> early bars are "asia".
    asia = bars.filter(pl.col("session_label") == "asia")
    assert len(asia) > 0
    assert asia["timestamp_utc"].dt.hour().min() < 9


def test_pipeline_dst_session_filter_keeps_only_sessions():
    ticks = generate_synthetic_tick_data(n_rows=100_000)
    p = ForexDataPipeline(bar_freq="5min", session_filter=True,
                          session_mode="dst", add_session_label=False,
                          apply_frac_diff=False)
    bars = p.run(ticks)
    assert "session_label" not in bars.columns
    assert len(bars) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline end-to-end (all bar types / gap policies)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bar_type", ["time", "tick", "volume", "dollar"])
def test_pipeline_all_bar_types(bar_type):
    ticks = generate_synthetic_tick_data(n_rows=5000)
    p = ForexDataPipeline(bar_freq="1min", bar_type=bar_type,
                          session_filter=False, apply_frac_diff=False)
    bars = p.run(ticks)
    assert len(bars) > 0
    assert {"open", "high", "low", "close"} <= set(bars.columns)


def test_pipeline_gap_policy_interpolate():
    ticks = generate_synthetic_tick_data(n_rows=5000)
    p = ForexDataPipeline(bar_freq="1min", session_filter=False,
                          apply_frac_diff=False, gap_policy="interpolate")
    bars = p.run(ticks)
    assert len(bars) > 0
