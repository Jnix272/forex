"""Pytest-native smoke/regression coverage for the end-to-end stack.

This file replaces the old print-based mega-runner with focused pytest tests.
Run directly with:

    python -m pytest tests/test_all.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from data.data_ingestion import (
    ForexDataPipeline,
    clean_bad_ticks,
    generate_synthetic_tick_data,
    resample_to_bars,
)
from data.economic_calendar import EcoCalendarFeatureBuilder
from data.sources import _enforce_schema
from features.advanced_features import AdvancedFeatureBuilder, hurst_exponent, session_clock_features
from features.feature_engineering import FeatureEngineer
from features.finbert_sentiment import SentimentPipeline
from features.macro_features import MacroYieldFeatureBuilder
from labeling.rl_reward_labeling import align_labels_with_features, compute_rl_reward_labels
from labeling.triple_barrier_labeling import _NUMBA_IMPORT_OK, _run_barrier_scan, _scan_outcomes_sequential


def null_count(df) -> int:
    if isinstance(df, pl.DataFrame):
        return int(df.select(pl.all().null_count()).to_numpy().sum())
    return int(df.isna().sum().sum())


def make_bars(n: int = 500) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.085 + np.cumsum(rng.normal(0, 0.0001, n))
    df = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.0002, n),
            "high": close + rng.uniform(0, 0.0003, n),
            "low": close - rng.uniform(0, 0.0003, n),
            "close": close,
            "volume": rng.integers(10, 100, n).astype(float),
            "bid_close": close - 0.00003,
            "ask_close": close + 0.00003,
            "spread_avg": np.full(n, 0.00005),
        },
        index=idx,
    )
    return pl.from_pandas(df.reset_index(names="timestamp_utc"))


def to_pandas_indexed(df: pl.DataFrame) -> pd.DataFrame:
    out = df.to_pandas()
    if "timestamp_utc" in out.columns:
        out = out.set_index("timestamp_utc")
    return out


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return make_bars(400)


@pytest.fixture(scope="module")
def features(bars: pl.DataFrame) -> pl.DataFrame:
    return FeatureEngineer().build(bars)


class TestDataPipelineSmoke:
    def test_synthetic_ticks_have_expected_schema(self):
        ticks = generate_synthetic_tick_data(n_rows=10_000)
        assert len(ticks) == 10_000
        assert {"bid", "ask", "mid", "volume", "spread"}.issubset(ticks.columns)
        assert (ticks["ask"] > ticks["bid"]).all()
        assert (ticks["spread"] > 0).all()

    def test_tick_to_bar_pipeline_runs(self):
        ticks = generate_synthetic_tick_data(n_rows=50_000)
        bars = ForexDataPipeline(bar_freq="1min", session_filter=False, apply_frac_diff=False).run(ticks)
        assert len(bars) > 100
        assert "close" in bars.columns
        assert bars.schema["timestamp_utc"].time_zone is not None

    def test_schema_enforcement_drops_bad_rows(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="1s", tz="UTC")
        df = pd.DataFrame(
            {
                "bid": [1.085, 1.086, 1.085, 1.085, 1.085],
                "ask": [1.086, 1.085, 1.086, 1.086, 1.086],
            },
            index=idx,
        )
        out = _enforce_schema(df, "EURUSD", "test")
        assert list(out.columns) == ["bid", "ask", "mid", "spread", "volume", "pair", "source"]
        assert len(out) == 4

    def test_bad_tick_cleaning_caps_spike(self):
        ticks = generate_synthetic_tick_data(n_rows=200)
        ticks = (
            ticks.with_row_index("_row")
            .with_columns(
                [
                    pl.when(pl.col("_row") == 100).then(pl.col(col) + 0.5).otherwise(pl.col(col)).alias(col)
                    for col in ("mid", "bid", "ask")
                ]
            )
            .drop("_row")
        )
        out = clean_bad_ticks(ticks, z_thresh=5.0)
        assert abs(out["mid"][100] - out["mid"][99]) < 0.01

    def test_economic_calendar_features_align_to_bars(self):
        ticks = generate_synthetic_tick_data(n_rows=5_000)
        bars = resample_to_bars(ticks, freq="1min")
        feats = EcoCalendarFeatureBuilder(use_synthetic=True).build(to_pandas_indexed(bars))
        assert "eco_release_flag" in feats.columns
        assert feats.shape[0] == bars.shape[0]


class TestFeatureEngineeringSmoke:
    def test_core_feature_matrix_is_dense(self, features: pl.DataFrame):
        assert features.shape[1] >= 40
        assert features.shape[0] > 0
        assert {"atr_6", "ofi", "rsi_14", "macd", "bb_pct"}.issubset(features.columns)
        assert null_count(features) == 0

    def test_advanced_features_are_present(self, bars: pl.DataFrame):
        advanced = AdvancedFeatureBuilder(hurst_windows=[30]).build(bars)
        assert "sess_london" in advanced.columns
        assert advanced.shape[1] >= 30
        assert null_count(advanced) == 0

    def test_session_clock_flags(self):
        idx = pd.date_range("2024-01-01 08:00", periods=60, freq="1min", tz="UTC")
        flags = session_clock_features(idx)
        assert flags["sess_london"].iloc[0] == 1.0
        assert flags["sess_ny"].iloc[0] == 0.0

    def test_hurst_bounds(self):
        series = pd.Series(np.cumsum(np.random.randn(200)))
        value = hurst_exponent(series)
        assert 0.0 <= value <= 1.0

    def test_macro_features_and_sentiment_pipeline(self, bars: pl.DataFrame):
        macro = MacroYieldFeatureBuilder().build(bars)
        assert "spread_us_de" in macro.columns
        assert "carry_eur" in macro.columns
        assert null_count(macro) == 0

        sentiment = SentimentPipeline(prefer_backend="vader", use_cache=False)
        score = sentiment.score_headlines(["EUR/USD expected to rally on positive news"])
        assert score > 0
        assert sentiment.active_backend() == "vader"


class TestLabelingSmoke:
    def test_rl_labels_align_with_features(self, bars: pl.DataFrame, features: pl.DataFrame):
        bars_pd = to_pandas_indexed(bars)
        features_pd = to_pandas_indexed(features)
        labels = compute_rl_reward_labels(bars_pd, features_pd)
        assert {"reward", "label"}.issubset(labels.columns)
        assert set(labels["label"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

        x_aligned, y_aligned, _ = align_labels_with_features(labels, features_pd)
        assert len(x_aligned) == len(y_aligned)
        assert x_aligned.shape[1] == features_pd.shape[1]
        assert null_count(x_aligned) == 0

    def test_triple_barrier_numba_matches_reference(self):
        rng = np.random.default_rng(42)
        n = 1_500
        vertical_barrier = 12
        close = np.cumsum(rng.standard_normal(n) * 0.00005).astype(np.float64) + 1.085
        entry_long = close + 0.00002
        entry_short = close - 0.00002
        atr = np.full(n, 0.00045, dtype=np.float64)
        profit_mult, stop_mult = 1.5, 1.0

        lo_seq, tl_seq, so_seq, ts_seq = _scan_outcomes_sequential(
            close, close, entry_long, entry_short, atr, profit_mult, stop_mult, vertical_barrier
        )
        lo_numba, tl_numba, so_numba, ts_numba, tag = _run_barrier_scan(
            close,
            entry_long,
            entry_short,
            atr,
            profit_mult,
            stop_mult,
            vertical_barrier,
            use_numba=True,
            parallel=True,
        )
        np.testing.assert_array_equal(lo_seq, lo_numba)
        np.testing.assert_array_equal(tl_seq, tl_numba)
        np.testing.assert_array_equal(so_seq, so_numba)
        np.testing.assert_array_equal(ts_seq, ts_numba)
        if _NUMBA_IMPORT_OK:
            assert tag == "numba_parallel"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
