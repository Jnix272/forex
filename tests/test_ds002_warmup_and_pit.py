"""Regression tests for DS-002 feature warmup and news PIT joins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest


def test_warmup_load_start_subtracts_days():
    from training.dataset_builder import _FEATURE_WARMUP_DAYS, _warmup_load_start

    assert _FEATURE_WARMUP_DAYS == 14
    assert _warmup_load_start("2024-02-15") == "2024-02-01"
    assert _warmup_load_start("2024-01-10", warmup_days=7) == "2024-01-03"


def test_join_asof_available_delays_same_bar_sentiment():
    from features.feature_engineering_pl import _join_asof_available

    bars = (
        pl.DataFrame({"close": [1.0, 1.1, 1.2]})
        .with_columns(
            pl.datetime_range(
                pl.datetime(2024, 1, 1, 12, 0, time_zone="UTC"),
                pl.datetime(2024, 1, 1, 12, 2, time_zone="UTC"),
                interval="1m",
                eager=True,
            ).alias("timestamp_utc")
        )
        .with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
    )

    # Sentiment published at 12:00 - with 1m delay becomes available at 12:01.
    sent = pl.from_pandas(
        pd.DataFrame(
            {
                "timestamp_utc": [pd.Timestamp("2024-01-01 12:00:00", tz="UTC")],
                "sentiment": [0.9],
            }
        )
    ).with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))

    joined = _join_asof_available(bars, sent)
    vals = joined["sentiment"].to_list()
    assert vals[0] is None or (isinstance(vals[0], float) and np.isnan(vals[0]))
    assert vals[1] == pytest.approx(0.9)
    assert vals[2] == pytest.approx(0.9)


def test_news_ok_kill_zone_is_post_release_only():
    from features.feature_engineering_pl import FeatureEngineer

    ts = pl.datetime_range(
        pl.datetime(2024, 1, 1, 12, 0, time_zone="UTC"),
        pl.datetime(2024, 1, 1, 12, 10, time_zone="UTC"),
        interval="1m",
        eager=True,
    )
    bars = pl.DataFrame(
        {
            "timestamp_utc": ts,
            "open": [1.0] * len(ts),
            "high": [1.01] * len(ts),
            "low": [0.99] * len(ts),
            "close": [1.0] * len(ts),
            "volume": [100.0] * len(ts),
            "bid_close": [0.999] * len(ts),
            "ask_close": [1.001] * len(ts),
        }
    ).with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))

    ev = pd.Timestamp("2024-01-01 12:05:00", tz="UTC")
    fe = FeatureEngineer(news_buf=2)
    # Minimal build path via public build with news_events only.
    out = fe.build(bars, news_events=[ev])
    pdf = out.select(["timestamp_utc", "news_ok", "pre_news", "post_news"]).to_pandas()
    # Bar at 12:04 (pre) should still be tradeable under post-only news_ok.
    pre_row = pdf[pdf["timestamp_utc"] == pd.Timestamp("2024-01-01 12:04:00", tz="UTC")].iloc[0]
    post_row = pdf[pdf["timestamp_utc"] == pd.Timestamp("2024-01-01 12:05:00", tz="UTC")].iloc[0]
    assert float(pre_row["news_ok"]) == 1.0
    assert float(pre_row["pre_news"]) == 1.0
    assert float(post_row["news_ok"]) == 0.0
    assert float(post_row["post_news"]) == 1.0
