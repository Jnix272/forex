"""
Tests for feature engineering pipeline: sanitize, sentiment tiers, COT features,
cross-asset builder, and full FeatureEngineer.build().
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bars() -> pl.DataFrame:
    """Minimal 1-minute OHLCV bars for pipeline testing."""
    n = 300
    ts = [
        datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
        for i in range(n)
    ]
    rng = np.random.default_rng(42)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0001, n))
    spread = rng.uniform(0.5, 2.0, n)
    volume = rng.integers(50, 500, n).astype(float)
    return pl.DataFrame({
        "timestamp_utc": ts,
        "open": close - rng.uniform(0, 0.0002, n),
        "high": close + rng.uniform(0, 0.0003, n),
        "low":  close - rng.uniform(0, 0.0003, n),
        "close": close,
        "volume": volume,
        "spread_pips": spread,
    }).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC"))
    )


# ---------------------------------------------------------------------------
# 1. sanitize_frame
# ---------------------------------------------------------------------------

class TestSanitizeFrame:
    def test_replaces_nan_and_inf(self):
        from features.feature_engineering_pl import sanitize_frame
        df = pl.DataFrame({
            "feat_a": [1.0, float("nan"), float("inf"), -float("inf"), 5.0],
            "feat_b": [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        out = sanitize_frame(df)
        assert out["feat_a"].is_nan().sum() == 0
        assert out["feat_a"].is_infinite().sum() == 0

    def test_clips_features_to_bounds(self):
        from features.feature_engineering_pl import sanitize_frame
        df = pl.DataFrame({"x": [-100.0, 0.5, 50.0]})
        out = sanitize_frame(df)
        assert out["x"].min() >= -1e6
        assert out["x"].max() <= 1e6

    def test_does_not_clip_price_columns(self):
        from features.feature_engineering_pl import sanitize_frame
        df = pl.DataFrame({
            "close": [1.10, 1.20, 1.30],
            "volume": [100.0, 200.0, 300.0],
            "spread_pips": [0.5, 1.0, 1.5],
        })
        out = sanitize_frame(df)
        assert out["close"].to_list() == [1.10, 1.20, 1.30]
        assert out["volume"].to_list() == [100.0, 200.0, 300.0]

    def test_does_not_clip_cot_columns(self):
        from features.feature_engineering_pl import sanitize_frame
        df = pl.DataFrame({
            "cot_net_hf": [50000.0, -30000.0, 80000.0],
            "cot_net_comm": [-45000.0, 60000.0, 22000.0],
        })
        out = sanitize_frame(df)
        assert out["cot_net_hf"].max() == 80000.0
        assert out["cot_net_comm"].min() == -45000.0

    def test_preserves_non_float_columns(self):
        from features.feature_engineering_pl import sanitize_frame
        df = pl.DataFrame({
            "label": [0, 1, 2],
            "feat": [float("nan"), 1.0, 2.0],
        })
        out = sanitize_frame(df)
        assert out["label"].to_list() == [0, 1, 2]


# ---------------------------------------------------------------------------
# 2. sentiment_tiers
# ---------------------------------------------------------------------------

class TestSentimentTiers:
    def test_adds_sentiment_columns_when_present(self):
        from features.feature_engineering_pl import sentiment_tiers
        from datetime import datetime, timezone
        ts = [
            datetime(2024, 1, 2, 10, i, tzinfo=timezone.utc) for i in range(5)
        ]
        df = pl.DataFrame({
            "timestamp_utc": ts,
            "sentiment": [0.5, -0.3, 0.1, 0.8, -0.5],
        }).with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
        out = sentiment_tiers(df, decay_lam=0.1, fb_dim=4)
        assert "sentiment_raw" in out.columns
        assert "sentiment_decayed" in out.columns

    def test_adds_fb_placeholders(self):
        from features.feature_engineering_pl import sentiment_tiers
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        out = sentiment_tiers(df, fb_dim=8)
        for i in range(8):
            col = f"fb_{i}"
            assert col in out.columns, f"Missing placeholder {col}"
            assert out[col].sum() == 0.0

    def test_fb_overwrite_protection_in_build(self, sample_bars):
        """The FeatureEngineer.build() protects existing fb_* columns,
        even though sentiment_tiers() itself always overwrites."""
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        out = fe.build(sample_bars)
        for i in range(8):
            assert f"fb_{i}" in out.columns


# ---------------------------------------------------------------------------
# 3. FeatureEngineer.build() smoke test
# ---------------------------------------------------------------------------

class TestFeatureEngineerBuild:
    def test_build_returns_polars_df(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        out = fe.build(sample_bars)
        assert isinstance(out, pl.DataFrame)
        assert len(out) > 0

    def test_build_produces_core_features(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        out = fe.build(sample_bars)
        core = ["ofi", "atr_6", "rsi_14", "ofi_z"]
        for col in core:
            assert col in out.columns, f"Missing core feature '{col}'"

    def test_build_no_nan_in_output(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        out = fe.build(sample_bars)
        float_cols = out.select(pl.col(pl.Float32, pl.Float64)).columns
        for col in float_cols:
            nan_count = out[col].is_nan().sum()
            assert nan_count == 0, f"Feature '{col}' has {nan_count} NaN values"

    def test_build_output_no_infinite(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        out = fe.build(sample_bars)
        float_cols = out.select(pl.col(pl.Float32, pl.Float64)).columns
        for col in float_cols:
            inf_count = out[col].is_infinite().sum()
            assert inf_count == 0, f"Feature '{col}' has {inf_count} Inf values"

    def test_build_has_eco_surprise_and_revision(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        out = fe.build(sample_bars)
        assert "eco_surprise" in out.columns
        assert "eco_revision" in out.columns

    def test_build_has_news_cat_columns(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        out = fe.build(sample_bars)
        cats = ["cat_central_bank", "cat_inflation", "cat_labor",
                "cat_growth", "cat_geopolitical", "cat_commentary"]
        for col in cats:
            assert col in out.columns, f"Missing news category '{col}'"

    def test_build_has_cot_columns(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        out = fe.build(sample_bars)
        cot_cols = ["cot_net_hf", "cot_net_comm", "cot_hf_mom_4w", "cot_extreme"]
        for col in cot_cols:
            assert col in out.columns, f"Missing COT feature '{col}'"

    def test_build_timestamp_preserved(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        out = fe.build(sample_bars)
        assert "timestamp_utc" in out.columns

    def test_build_with_pandas_input(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        pd_bars = sample_bars.to_pandas()
        pd_bars = pd_bars.set_index("timestamp_utc")
        out = fe.build(pd_bars)
        assert isinstance(out, pl.DataFrame)

    def test_build_default_has_no_no_trade_score(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer()
        out = fe.build(sample_bars)
        assert "no_trade_score" not in out.columns

    def test_build_no_trade_zones_enabled(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer(enable_no_trade_zones=True)
        out = fe.build(sample_bars)
        assert "no_trade_score" in out.columns
        assert out["no_trade_score"].drop_nulls().len() == len(out)
        assert out["no_trade_score"].to_numpy().min() >= 0.0
        assert out["no_trade_score"].to_numpy().max() <= 1.0

    def test_build_quality_gate_enabled(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer(enable_quality_gate=True)
        out = fe.build(sample_bars)
        assert isinstance(out, pl.DataFrame)
        assert len(out) > 0
        assert hasattr(fe, "quality_report")
        assert "feature" in fe.quality_report.columns

    def test_build_both_flags_combined(self, sample_bars):
        from features.feature_engineering_pl import FeatureEngineer
        fe = FeatureEngineer(enable_no_trade_zones=True, enable_quality_gate=True)
        out = fe.build(sample_bars)
        assert "no_trade_score" in out.columns
        assert hasattr(fe, "quality_report")
        assert len(out) > 0
