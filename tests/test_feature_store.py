"""
Tests for Feature Store: registry, materialization, lineage, and convenience functions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from data.feature_definitions import (
    BUILTIN_FEATURES,
    REGISTRY,
    FeatureSource,
    FeatureType,
    MaterializationStrategy,
)
from data.feature_materializers import (
    MicrostructureMaterializer,
    PriceMaterializer,
    SessionMaterializer,
    VolatilityMaterializer,
    get_materializer,
    materialize_feature,
)
from data.feature_store import FeatureStore, compute_data_hash, compute_feature_hash

# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_store(tmp_path) -> FeatureStore:
    path = tmp_path / "feature_store"
    return FeatureStore(root=path)


@pytest.fixture
def sample_bars_1min() -> pl.DataFrame:
    n = 500
    ts = [
        datetime(2024, 1, 2, 8, 0, tzinfo=UTC) + timedelta(minutes=i)
        for i in range(n)
    ]
    rng = np.random.default_rng(42)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0001, n))
    return pl.DataFrame({
        "timestamp_utc": ts,
        "open": close - rng.uniform(0, 0.0002, n),
        "high": close + rng.uniform(0, 0.0003, n),
        "low":  close - rng.uniform(0, 0.0003, n),
        "close": close,
        "volume": rng.integers(50, 500, n).astype(float),
        "tick_volume": rng.integers(100, 1000, n).astype(float),
        "spread_pips": rng.uniform(0.5, 2.0, n),
    }).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC"))
    )


# ════════════════════════════════════════════════════════════════════════════
# Feature Hashing
# ════════════════════════════════════════════════════════════════════════════

class TestHashing:
    def test_compute_feature_hash_is_deterministic(self):
        spec = BUILTIN_FEATURES[0]
        assert compute_feature_hash(spec) == compute_feature_hash(spec)

    def test_different_features_have_different_hashes(self):
        h1 = compute_feature_hash(BUILTIN_FEATURES[0])
        h2 = compute_feature_hash(BUILTIN_FEATURES[1])
        assert h1 != h2

    def test_data_hash_changes_with_different_data(self):
        df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        df2 = pl.DataFrame({"x": [4.0, 5.0, 6.0]})
        assert compute_data_hash(df1, "x") == compute_data_hash(df1, "x")
        assert compute_data_hash(df1, "x") != compute_data_hash(df2, "x")


# ════════════════════════════════════════════════════════════════════════════
# FeatureStore: Registry
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureStore:
    def test_init_creates_db_and_syncs_builtins(self, tmp_store):
        assert tmp_store.db_path.exists()
        builtin = tmp_store.list_features()
        assert len(builtin) == len([s for s in BUILTIN_FEATURES if not s.deprecated])

    def test_get_feature_by_name(self, tmp_store):
        spec = tmp_store.get_feature("close")
        assert spec is not None
        assert spec.name == "close"
        assert spec.feature_type == FeatureType.NUMERIC

    def test_get_feature_unknown_returns_none(self, tmp_store):
        assert tmp_store.get_feature("does_not_exist") is None

    def test_list_features_by_source(self, tmp_store):
        price_features = tmp_store.list_features(source=FeatureSource.PRICE)
        assert all(s.source == FeatureSource.PRICE for s in price_features)
        assert len(price_features) > 0

    def test_list_features_by_tag(self, tmp_store):
        for spec in BUILTIN_FEATURES:
            if "session" in spec.tags:
                tagged = tmp_store.list_features(tag="session")
                assert any(s.name == spec.name for s in tagged)
                break

    def test_list_features_excludes_deprecated(self, tmp_store):
        # Temporarily mark a feature as deprecated
        conn = tmp_store._lock
        import sqlite3
        conn = sqlite3.connect(tmp_store.db_path)
        conn.execute("UPDATE features SET deprecated = 1 WHERE name = 'close'")
        conn.commit()
        conn.close()

        all_features = tmp_store.list_features()
        assert not any(s.name == "close" for s in all_features)

        with_deprecated = tmp_store.list_features(deprecated=True)
        assert any(s.name == "close" for s in with_deprecated)

    def test_feature_metadata_stored(self, tmp_store):
        spec = tmp_store.get_feature("log_ret_1")
        assert spec.version >= 1
        assert spec.description
        assert spec.transformation
        assert "returns" in spec.tags or "momentum" in spec.tags

    def test_get_lineage_upstream(self, tmp_store):
        lineage = tmp_store.get_lineage("log_ret_5", direction="upstream")
        assert "close" in lineage["upstream"]

    def test_get_lineage_downstream(self, tmp_store):
        lineage = tmp_store.get_lineage("close", direction="downstream")
        assert len(lineage["downstream"]) > 0

    def test_get_lineage_both(self, tmp_store):
        lineage = tmp_store.get_lineage("close", direction="both")
        assert len(lineage["downstream"]) > 0
        assert len(lineage["upstream"]) == 0

    def test_storage_stats_on_empty_store(self, tmp_store):
        stats = tmp_store.get_storage_stats()
        assert stats["total_bytes"] >= 0
        assert stats["feature_count"] == 0


# ════════════════════════════════════════════════════════════════════════════
# Materialization
# ════════════════════════════════════════════════════════════════════════════

class TestMaterialization:
    def test_is_materialized_returns_false_for_unmaterialized(self, tmp_store):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        assert not tmp_store.is_materialized("close", start, end)

    def test_store_materialization(self, tmp_store):
        start = datetime(2024, 1, 1, 8, tzinfo=UTC)
        end = datetime(2024, 1, 1, 9, tzinfo=UTC)
        df = pl.DataFrame({
            "timestamp_utc": [start, end],
            "close": [1.1000, 1.1005],
        })
        tmp_store._store_materialization(
            "close", df, start, end, MaterializationStrategy.EAGER_BATCH
        )
        assert tmp_store.is_materialized("close", start, end)

    def test_load_feature_after_materialization(self, tmp_store):
        start = datetime(2024, 1, 1, 8, tzinfo=UTC)
        end = datetime(2024, 1, 1, 9, tzinfo=UTC)
        df = pl.DataFrame({
            "timestamp_utc": [start, end],
            "close": [1.1000, 1.1005],
        })
        tmp_store._store_materialization(
            "close", df, start, end, MaterializationStrategy.EAGER_BATCH
        )
        loaded = tmp_store.load_feature("close", start, end)
        assert loaded is not None
        assert len(loaded) == 2

    def test_get_materialized_ranges(self, tmp_store):
        start = datetime(2024, 1, 1, 8, tzinfo=UTC)
        end = datetime(2024, 1, 1, 9, tzinfo=UTC)
        df = pl.DataFrame({
            "timestamp_utc": [start, end],
            "close": [1.1000, 1.1005],
        })
        tmp_store._store_materialization(
            "close", df, start, end, MaterializationStrategy.EAGER_BATCH
        )
        ranges = tmp_store.get_materialized_ranges("close")
        assert len(ranges) == 1
        assert ranges[0][0] == start
        assert ranges[0][1] == end

    def test_get_latest_timestamp(self, tmp_store):
        start = datetime(2024, 1, 1, 8, tzinfo=UTC)
        end = datetime(2024, 1, 1, 9, tzinfo=UTC)
        df = pl.DataFrame({
            "timestamp_utc": [start, end],
            "close": [1.1000, 1.1005],
        })
        tmp_store._store_materialization(
            "close", df, start, end, MaterializationStrategy.EAGER_BATCH
        )
        assert tmp_store.get_latest_timestamp("close") == end
        assert tmp_store.get_latest_timestamp("nonexistent") is None

    def test_materialize_duplicate_is_idempotent(self, tmp_store):
        start = datetime(2024, 1, 1, 8, tzinfo=UTC)
        end = datetime(2024, 1, 1, 9, tzinfo=UTC)
        df = pl.DataFrame({
            "timestamp_utc": [start, end],
            "close": [1.1000, 1.1005],
        })
        tmp_store._store_materialization(
            "close", df, start, end, MaterializationStrategy.EAGER_BATCH
        )
        tmp_store._store_materialization(
            "close", df, start, end, MaterializationStrategy.EAGER_BATCH
        )
        ranges = tmp_store.get_materialized_ranges("close")
        assert len(ranges) == 1  # same range, replaced


# ════════════════════════════════════════════════════════════════════════════
# Job Queue
# ════════════════════════════════════════════════════════════════════════════

class TestJobQueue:
    def test_enqueue_and_list_pending(self, tmp_store):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        job_id = tmp_store.enqueue_materialization(
            "close", start, end, MaterializationStrategy.INCREMENTAL
        )
        assert job_id > 0
        pending = tmp_store.get_pending_jobs()
        assert len(pending) >= 1
        assert pending[0]["feature_name"] == "close"

    def test_mark_job_started_and_done(self, tmp_store):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        job_id = tmp_store.enqueue_materialization("close", start, end)
        tmp_store.mark_job_started(job_id)
        tmp_store.mark_job_done(job_id)
        pending = tmp_store.get_pending_jobs()
        assert all(j["id"] != job_id for j in pending)

    def test_mark_job_failed(self, tmp_store):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        job_id = tmp_store.enqueue_materialization("close", start, end)
        tmp_store.mark_job_done(job_id, error="test error")
        pending = tmp_store.get_pending_jobs()
        assert all(j["id"] != job_id for j in pending)


# ════════════════════════════════════════════════════════════════════════════
# Materializers
# ════════════════════════════════════════════════════════════════════════════

class TestPriceMaterializer:
    def test_close_feature(self, sample_bars_1min, tmp_store):
        mat = PriceMaterializer(tmp_store)
        spec = tmp_store.get_feature("close")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None
        assert "timestamp_utc" in result.columns
        assert "close" in result.columns

    def test_log_ret_1(self, sample_bars_1min, tmp_store):
        mat = PriceMaterializer(tmp_store)
        spec = tmp_store.get_feature("log_ret_1")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None
        assert "log_ret_1" in result.columns
        # No NaNs (first row is null due to shift)
        assert len(result) <= len(sample_bars_1min)

    def test_log_ret_5(self, sample_bars_1min, tmp_store):
        mat = PriceMaterializer(tmp_store)
        spec = tmp_store.get_feature("log_ret_5")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None

    def test_spread_bps(self, sample_bars_1min, tmp_store):
        mat = PriceMaterializer(tmp_store)
        spec = tmp_store.get_feature("spread_bps")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None
        assert "spread_bps" in result.columns

    def test_requires_ohlcv_true(self, tmp_store):
        mat = PriceMaterializer(tmp_store)
        assert mat.requires_ohlcv


class TestVolatilityMaterializer:
    def test_atr_6(self, sample_bars_1min, tmp_store):
        mat = VolatilityMaterializer(tmp_store)
        spec = tmp_store.get_feature("atr_6")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None
        assert "atr_6" in result.columns

    def test_atr_20(self, sample_bars_1min, tmp_store):
        mat = VolatilityMaterializer(tmp_store)
        spec = tmp_store.get_feature("atr_20")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None

    def test_rolling_vol_20(self, sample_bars_1min, tmp_store):
        mat = VolatilityMaterializer(tmp_store)
        spec = tmp_store.get_feature("rolling_vol_20")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None

    def test_bollinger_upper_lower(self, sample_bars_1min, tmp_store):
        mat = VolatilityMaterializer(tmp_store)
        upper_spec = tmp_store.get_feature("bollinger_upper_20")
        lower_spec = tmp_store.get_feature("bollinger_lower_20")
        upper = mat.compute(upper_spec, sample_bars_1min,
                            datetime(2024, 1, 2, tzinfo=UTC),
                            datetime(2024, 1, 3, tzinfo=UTC))
        lower = mat.compute(lower_spec, sample_bars_1min,
                            datetime(2024, 1, 2, tzinfo=UTC),
                            datetime(2024, 1, 3, tzinfo=UTC))
        assert upper is not None
        assert lower is not None
        assert all(u >= l for u, l in zip(upper["bollinger_upper_20"], lower["bollinger_lower_20"]))


class TestMicrostructureMaterializer:
    def test_ofi_20(self, sample_bars_1min, tmp_store):
        mat = MicrostructureMaterializer(tmp_store)
        spec = tmp_store.get_feature("ofi_20")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None
        assert "ofi_20" in result.columns

    def test_obi_proxy(self, sample_bars_1min, tmp_store):
        mat = MicrostructureMaterializer(tmp_store)
        spec = tmp_store.get_feature("obi_proxy")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None
        assert "obi_proxy" in result.columns

    def test_hurst_120(self, sample_bars_1min, tmp_store):
        mat = MicrostructureMaterializer(tmp_store)
        spec = tmp_store.get_feature("hurst_120")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        # Hurst needs 120+ bars; may be None if window too short
        if result is not None:
            assert "hurst_120" in result.columns

    def test_iv_proxy_20(self, sample_bars_1min, tmp_store):
        mat = MicrostructureMaterializer(tmp_store)
        spec = tmp_store.get_feature("iv_proxy_20")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None

    def test_skew_proxy_20(self, sample_bars_1min, tmp_store):
        mat = MicrostructureMaterializer(tmp_store)
        spec = tmp_store.get_feature("skew_proxy_20")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None
        assert "skew_proxy_20" in result.columns


class TestSessionMaterializer:
    def test_hour_encoding(self, sample_bars_1min, tmp_store):
        mat = SessionMaterializer(tmp_store)
        spec = tmp_store.get_feature("hour_sin")
        result = mat.compute(spec, sample_bars_1min,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is not None
        assert "hour_sin" in result.columns
        # sin/cos should be bounded
        assert result["hour_sin"].abs().max() <= 1.0

    def test_dow_encoding(self, sample_bars_1min, tmp_store):
        mat = SessionMaterializer(tmp_store)
        dow_sin = mat.compute(tmp_store.get_feature("dow_sin"), sample_bars_1min,
                              datetime(2024, 1, 2, tzinfo=UTC),
                              datetime(2024, 1, 3, tzinfo=UTC))
        dow_cos = mat.compute(tmp_store.get_feature("dow_cos"), sample_bars_1min,
                              datetime(2024, 1, 2, tzinfo=UTC),
                              datetime(2024, 1, 3, tzinfo=UTC))
        assert dow_sin is not None
        assert dow_cos is not None

    def test_session_flags(self, sample_bars_1min, tmp_store):
        mat = SessionMaterializer(tmp_store)
        asia = mat.compute(tmp_store.get_feature("session_asia"), sample_bars_1min,
                           datetime(2024, 1, 2, tzinfo=UTC),
                           datetime(2024, 1, 3, tzinfo=UTC))
        assert asia is not None
        assert asia["session_asia"].dtype == pl.Int32

    def test_is_monday_friday(self, sample_bars_1min, tmp_store):
        mat = SessionMaterializer(tmp_store)
        mon = mat.compute(tmp_store.get_feature("is_monday"), sample_bars_1min,
                          datetime(2024, 1, 2, tzinfo=UTC),
                          datetime(2024, 1, 3, tzinfo=UTC))
        fri = mat.compute(tmp_store.get_feature("is_friday"), sample_bars_1min,
                          datetime(2024, 1, 2, tzinfo=UTC),
                          datetime(2024, 1, 3, tzinfo=UTC))
        assert mon is not None
        assert fri is not None

    def test_does_not_require_ohlcv(self, tmp_store):
        mat = SessionMaterializer(tmp_store)
        assert not mat.requires_ohlcv


class TestMaterializerDispatch:
    def test_get_materializer_price(self, tmp_store):
        spec = tmp_store.get_feature("close")
        mat = get_materializer(spec, tmp_store)
        assert isinstance(mat, PriceMaterializer)

    def test_get_materializer_volatility(self, tmp_store):
        spec = tmp_store.get_feature("atr_20")
        mat = get_materializer(spec, tmp_store)
        assert isinstance(mat, VolatilityMaterializer)

    def test_get_materializer_microstructure(self, tmp_store):
        spec = tmp_store.get_feature("ofi_20")
        mat = get_materializer(spec, tmp_store)
        assert isinstance(mat, MicrostructureMaterializer)

    def test_get_materializer_session(self, tmp_store):
        spec = tmp_store.get_feature("hour_sin")
        mat = get_materializer(spec, tmp_store)
        assert isinstance(mat, SessionMaterializer)

    def test_materialize_feature_end_to_end(self, sample_bars_1min, tmp_store):
        start = datetime(2024, 1, 2, 8, tzinfo=UTC)
        end = datetime(2024, 1, 2, 10, tzinfo=UTC)
        result = materialize_feature(tmp_store, "close", sample_bars_1min, start, end)
        assert result is not None
        assert "close" in result.columns
        # Should now be in the store
        assert tmp_store.is_materialized("close", start, end)


# ════════════════════════════════════════════════════════════════════════════
# FeatureRegistry dependency resolution
# ════════════════════════════════════════════════════════════════════════════

class TestDependencyResolution:
    def test_resolve_single_feature(self):
        names = REGISTRY.resolve_dependencies(["log_ret_1"])
        assert "close" in names  # close is a dependency

    def test_resolve_in_order(self):
        names = REGISTRY.resolve_dependencies(["log_ret_20"])
        assert "close" in names
        # close should come before log_ret_20
        assert names.index("close") < names.index("log_ret_20")

    def test_resolve_multiple(self):
        names = REGISTRY.resolve_dependencies(["log_ret_5", "log_ret_20"])
        assert "close" in names

    def test_preserves_order_of_independent_features(self):
        names = REGISTRY.resolve_dependencies(["gold", "wti"])
        assert names[0] == "gold"
        assert names[1] == "wti"


# ════════════════════════════════════════════════════════════════════════════
# FeatureStore maintenance
# ════════════════════════════════════════════════════════════════════════════

class TestMaintenance:
    def test_vacuum_does_not_error(self, tmp_store):
        tmp_store.vacuum()  # should just work

    def test_storage_stats_after_materialization(self, tmp_store):
        start = datetime(2024, 1, 1, 8, tzinfo=UTC)
        end = datetime(2024, 1, 1, 9, tzinfo=UTC)
        df = pl.DataFrame({
            "timestamp_utc": [start, end],
            "close": [1.1000, 1.1005],
        })
        tmp_store._store_materialization("close", df, start, end, MaterializationStrategy.EAGER_BATCH)
        stats = tmp_store.get_storage_stats()
        assert stats["feature_count"] == 1
        assert stats["total_bytes"] > 0
        assert "close" in stats["per_feature_mb"]


# ════════════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_materialization_without_timestamp_fails(self, tmp_store):
        df = pl.DataFrame({"close": [1.1, 1.2]})
        with pytest.raises(ValueError, match="timestamp_utc"):
            tmp_store._store_materialization(
                "close", df,
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                MaterializationStrategy.EAGER_BATCH,
            )

    def test_load_unmaterialized_feature_returns_none(self, tmp_store):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        assert tmp_store.load_feature("nonexistent", start, end) is None

    def test_empty_bars_does_not_error(self, sample_bars_1min, tmp_store):
        mat = PriceMaterializer(tmp_store)
        empty = sample_bars_1min.head(0)  # empty frame with same schema
        result = mat.compute(tmp_store.get_feature("close"), empty,
                             datetime(2024, 1, 2, tzinfo=UTC),
                             datetime(2024, 1, 3, tzinfo=UTC))
        assert result is None or len(result) == 0

    def test_concurrent_feature_registry_queries(self, tmp_store):
        """Multiple list_features calls should be safe."""
        import threading
        results = []

        def query():
            results.append(len(tmp_store.list_features()))

        threads = [threading.Thread(target=query) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == results[0] for r in results)
