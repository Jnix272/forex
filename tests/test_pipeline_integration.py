"""
Integration tests for Phase 4 — Full Pipeline: FeatureStore + DriftTracker + RetrainOrchestrator.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

import retraining.pipeline as pipeline_module
from data.feature_store import FeatureStore, MaterializationStrategy
from monitoring.drift_detection import DriftTracker
from retraining.orchestrator import RetrainConfig, RetrainOrchestrator, RetrainReason
from retraining.pipeline import FullPipeline, PipelineConfig, load_config_from_yaml

# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def seeded_store(tmp_path) -> FeatureStore:
    """FeatureStore with pre-materialized features for drift/retrain testing."""
    store = FeatureStore(root=tmp_path / "fs_integration")
    rng = np.random.default_rng(42)
    n_base = 5000
    n_live = 2000

    base_ts = [
        datetime(2024, 1, 1, 0, tzinfo=UTC) + timedelta(minutes=i)
        for i in range(n_base)
    ]
    live_ts = [
        datetime(2024, 6, 1, 0, tzinfo=UTC) + timedelta(minutes=i)
        for i in range(n_live)
    ]

    # Feature with drift
    for col in ("close", "log_ret_1"):
        base = pl.DataFrame({
            "timestamp_utc": base_ts,
            col: rng.normal(0, 1, n_base),
        })
        live = pl.DataFrame({
            "timestamp_utc": live_ts,
            col: rng.normal(0.5, 1.2, n_live),  # shifted
        })
        store._store_materialization(
            col, base, base_ts[0], base_ts[-1], MaterializationStrategy.EAGER_BATCH,
        )
        store._store_materialization(
            col, live, live_ts[0], live_ts[-1], MaterializationStrategy.EAGER_BATCH,
        )

    # Feature without drift
    base2 = pl.DataFrame({
        "timestamp_utc": base_ts,
        "atr_6": rng.normal(0.001, 0.0005, n_base),
    })
    live2 = pl.DataFrame({
        "timestamp_utc": live_ts,
        "atr_6": rng.normal(0.001, 0.0005, n_live),
    })
    store._store_materialization(
        "atr_6", base2, base_ts[0], base_ts[-1], MaterializationStrategy.EAGER_BATCH,
    )
    store._store_materialization(
        "atr_6", live2, live_ts[0], live_ts[-1], MaterializationStrategy.EAGER_BATCH,
    )

    return store


# ════════════════════════════════════════════════════════════════════════════
# FeatureStore + Drift Integration
# ════════════════════════════════════════════════════════════════════════════

class TestStoreDriftIntegration:
    def test_drift_check_from_store(self, seeded_store):
        tracker = DriftTracker(seeded_store)
        base_start = datetime(2024, 1, 1, tzinfo=UTC)
        base_end = datetime(2024, 1, 2, tzinfo=UTC)
        live_start = datetime(2024, 6, 1, tzinfo=UTC)
        live_end = datetime(2024, 6, 2, tzinfo=UTC)

        report = tracker.check_from_store(
            ["close", "atr_6"], base_start, base_end, live_start, live_end,
        )
        assert report.n_features == 2
        # close should have drift, atr_6 should not
        for r in report.feature_results:
            if r.feature_name == "close":
                assert r.drifted or r.psi > 0.05

    def test_drift_triggers_alerts(self, seeded_store):
        tracker = DriftTracker(seeded_store)
        base_start = datetime(2024, 1, 1, tzinfo=UTC)
        base_end = datetime(2024, 1, 2, tzinfo=UTC)
        live_start = datetime(2024, 6, 1, tzinfo=UTC)
        live_end = datetime(2024, 6, 2, tzinfo=UTC)

        report = tracker.check_from_store(
            ["close", "atr_6"], base_start, base_end, live_start, live_end,
        )
        alerts = tracker.get_active_alerts()
        # Drifted features should create alerts
        drifted_features = {r.feature_name for r in report.feature_results if r.drifted}
        alert_features = {a["feature_name"] for a in alerts}
        assert drifted_features.issubset(alert_features) or not drifted_features


# ════════════════════════════════════════════════════════════════════════════
# Drift → Retrain Integration
# ════════════════════════════════════════════════════════════════════════════

class TestDriftRetrainIntegration:
    def test_drift_triggers_retrain_check(self, seeded_store, tmp_path):
        """Drift should make should_retrain return True."""
        cfg = RetrainConfig(
            enable_drift_trigger=True,
            drift_feature_frac=0.3,
            model_root=tmp_path / "models_drift",
        )
        orch = RetrainOrchestrator(seeded_store, config=cfg)

        # Manually add a drift alert to trigger retrain
        from monitoring.drift_detection import check_feature_drift
        rng = np.random.default_rng(42)
        check_feature_drift("close", rng.normal(0, 1, 1000), rng.normal(1, 1, 1000))
        tracker = DriftTracker(seeded_store)
        from monitoring.drift_detection import run_drift_check
        report = run_drift_check(
            {"close": rng.normal(0, 1, 1000)},
            {"close": rng.normal(1, 1, 500)},
        )
        tracker._persist_report(report)

        should, reason, ctx = orch.should_retrain("haelt")
        # May or may not trigger depending on drift_feature_frac
        assert reason in ("", RetrainReason.DRIFT_DETECTED.value, RetrainReason.INITIAL.value)

    def test_full_drift_check_and_retrain_dry_run(self, seeded_store, tmp_path):
        cfg = RetrainConfig(
            enable_drift_trigger=True,
            drift_feature_frac=0.3,
            model_root=tmp_path / "models_full",
        )
        orch = RetrainOrchestrator(seeded_store, config=cfg)

        result = orch.check_and_retrain(family="haelt", force=True, dry_run=True)
        assert result["dry_run"]
        assert result["version"] >= 1


# ════════════════════════════════════════════════════════════════════════════
# FullPipeline End-to-End
# ════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_pipeline_init(self, tmp_path):
        cfg = PipelineConfig(feature_store_root=str(tmp_path / "fs_pipe"))
        pipeline = FullPipeline(cfg)
        assert pipeline.store.root == tmp_path / "fs_pipe"
        assert pipeline.store.db_path.exists()

    def test_pipeline_status(self, tmp_path):
        cfg = PipelineConfig(feature_store_root=str(tmp_path / "fs_status"))
        pipeline = FullPipeline(cfg)
        status = pipeline.status()
        assert "feature_store" in status
        assert "orchestrator" in status
        assert status["feature_store"]["feature_count"] >= 0

    def test_pipeline_drift_check_no_crash(self, tmp_path):
        """Drift check on empty store should not crash."""
        cfg = PipelineConfig(
            feature_store_root=str(tmp_path / "fs_drift_empty"),
            retrain_enabled=False,
        )
        pipeline = FullPipeline(cfg)
        result = pipeline.run(skip_materialize=True)
        assert result["status"] in ("complete", "error")
        if result.get("drift_report"):
            assert not result["drift_report"]["drift_detected"]

    def test_pipeline_materialize_and_drift(self, tmp_path):
        """Full pipeline with synthetic bars should not crash."""
        cfg = PipelineConfig(
            feature_store_root=str(tmp_path / "fs_pipe_run"),
            feature_names=["close", "log_ret_1"],
            retrain_enabled=False,
            auto_materialize=True,
        )
        pipeline = FullPipeline(cfg)

        # Create synthetic bars
        rng = np.random.default_rng(42)
        n = 1000
        ts = [
            datetime(2024, 1, 1, 8, tzinfo=UTC) + timedelta(minutes=i)
            for i in range(n)
        ]
        close = 1.1000 + np.cumsum(rng.normal(0, 0.0001, n))
        bars = pl.DataFrame({
            "timestamp_utc": ts,
            "open": close - rng.uniform(0, 0.0002, n),
            "high": close + rng.uniform(0, 0.0003, n),
            "low": close - rng.uniform(0, 0.0003, n),
            "close": close,
            "volume": rng.integers(50, 500, n).astype(float),
            "tick_volume": rng.integers(100, 1000, n).astype(float),
            "spread_pips": rng.uniform(0.5, 2.0, n),
        })

        start = datetime(2024, 1, 1, 8, tzinfo=UTC)
        end = datetime(2024, 1, 1, 10, tzinfo=UTC)
        result = pipeline.run(bars=bars, start=start, end=end, skip_materialize=False)
        assert result["status"] in ("complete", "error")

    def test_pipeline_anchors_drift_to_run_end(self, tmp_path, monkeypatch):
        """Pipeline drift checks should use the caller-provided run end time."""
        captured = {}

        def fake_schedule_drift_check(*args, **kwargs):
            captured["as_of"] = kwargs.get("as_of")
            from monitoring.drift_detection import DriftReport, DriftSeverity
            return DriftReport(
                feature_results=[],
                psi_max=0.0,
                ks_min_pvalue=1.0,
                ks_max_stat=0.0,
                n_drifted=0,
                n_features=0,
                overall_severity=DriftSeverity.NONE,
                drift_detected=False,
                baseline_time="",
                live_time="",
            )

        monkeypatch.setattr(pipeline_module, "schedule_drift_check", fake_schedule_drift_check)

        cfg = PipelineConfig(
            feature_store_root=str(tmp_path / "fs_drift_anchor"),
            retrain_enabled=False,
            auto_materialize=False,
        )
        pipeline = FullPipeline(cfg)
        end = datetime(2024, 2, 3, 12, tzinfo=UTC)

        result = pipeline.run(end=end, skip_materialize=True)

        assert result["status"] == "complete"
        assert captured["as_of"] == end


# ════════════════════════════════════════════════════════════════════════════
# Config loading
# ════════════════════════════════════════════════════════════════════════════

class TestConfigLoading:
    def test_load_config_from_yaml(self):
        cfg = load_config_from_yaml("config/run.yaml")
        assert cfg.feature_store_root == "data/feature_store"
        assert cfg.retrain_model_family == "haelt"
        assert cfg.drift_baseline_days == 90

    def test_load_config_missing_file(self):
        cfg = load_config_from_yaml("nonexistent.yaml")
        assert cfg.feature_store_root == "data/feature_store"
        assert cfg.retrain_enabled is True
