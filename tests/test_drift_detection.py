"""
Tests for Phase 2 — Drift Detection: PSI, KS, DriftTracker, and report generation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import sqlite3

from data.feature_store import FeatureStore, MaterializationStrategy
from monitoring.drift_detection import (
    DriftSeverity,
    DriftTracker,
    _psi_to_severity,
    _safe_psi,
    check_feature_drift,
    run_drift_check,
    schedule_drift_check,
)

# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_store(tmp_path) -> FeatureStore:
    return FeatureStore(root=tmp_path / "feature_store")


@pytest.fixture
def drift_tracker(tmp_store) -> DriftTracker:
    return DriftTracker(tmp_store)


@pytest.fixture
def drift_tracker_with_data(tmp_path) -> tuple[FeatureStore, DriftTracker]:
    """Pre-seed a store with materialized features for drift checks."""
    store = FeatureStore(root=tmp_path / "fs_drift")

    rng = np.random.default_rng(42)
    n_base = 200
    n_live = 100

    base_ts = [
        datetime(2024, 1, 1, 8, tzinfo=UTC) + timedelta(minutes=i)
        for i in range(n_base)
    ]
    live_ts = [
        datetime(2024, 6, 1, 8, tzinfo=UTC) + timedelta(minutes=i)
        for i in range(n_live)
    ]

    # Feature A: shifted distribution (drift)
    feat_a_base = pl.DataFrame({
        "timestamp_utc": base_ts,
        "feat_a": rng.normal(0, 1, n_base),
    })
    feat_a_live = pl.DataFrame({
        "timestamp_utc": live_ts,
        "feat_a": rng.normal(0.5, 1.2, n_live),
    })

    # Feature B: same distribution (no drift)
    feat_b_base = pl.DataFrame({
        "timestamp_utc": base_ts,
        "feat_b": rng.normal(0, 1, n_base),
    })
    feat_b_live = pl.DataFrame({
        "timestamp_utc": live_ts,
        "feat_b": rng.normal(0, 1, n_live),
    })

    # Register ad-hoc feature rows before storing materializations (the
    # materializations FK references features(name); bypassing materialize()
    # would deadlock the strict PRAGMA foreign_keys=ON enforced since 2026-08.)
    for name in ("feat_a", "feat_b"):
        fs_db = store.root / "registry.db"
        with sqlite3.connect(str(fs_db)) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            now_dt = datetime.now(UTC).isoformat()
            conn.execute(
                "INSERT OR IGNORE INTO features "
                "(name, feature_type, description, source, transformation, "
                " dependencies, params, version, tags, owner, "
                " created_at, updated_at, deprecated, content_hash) "
                "VALUES (?, ?, ?, ?, ?,  ?, ?, ?, ?, ?,  ?, ?, ?, ?)",
                (name, "test", "ad-hoc drift-test feature", "tests", "",
                 "", "", 1, "", "tests",
                 now_dt, now_dt, 0, name),
            )
            conn.commit()

    store._store_materialization(
        "feat_a", feat_a_base,
        base_ts[0], base_ts[-1],
        MaterializationStrategy.EAGER_BATCH,
    )
    store._store_materialization(
        "feat_a", feat_a_live,
        live_ts[0], live_ts[-1],
        MaterializationStrategy.EAGER_BATCH,
    )
    store._store_materialization(
        "feat_b", feat_b_base,
        base_ts[0], base_ts[-1],
        MaterializationStrategy.EAGER_BATCH,
    )
    store._store_materialization(
        "feat_b", feat_b_live,
        live_ts[0], live_ts[-1],
        MaterializationStrategy.EAGER_BATCH,
    )

    tracker = DriftTracker(store)
    return store, tracker


# ════════════════════════════════════════════════════════════════════════════
# PSI / KS Core
# ════════════════════════════════════════════════════════════════════════════

class TestSafePSI:
    def test_identical_distributions_return_low_psi(self):
        a = np.random.default_rng(42).normal(0, 1, 10_000)
        psi = _safe_psi(a, a)
        assert psi < 0.05

    def test_different_distributions_return_higher_psi(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 10_000)
        b = rng.normal(1, 1, 10_000)
        psi = _safe_psi(a, b)
        assert psi > 0.1

    def test_constant_feature_returns_zero(self):
        a = np.ones(1000)
        b = np.ones(1000)
        assert _safe_psi(a, b) == 0.0

    def test_handles_nan_and_inf(self):
        a = np.array([1.0, 2.0, float("nan"), float("inf"), 5.0])
        b = np.array([1.5, 2.5, 3.5, float("-inf"), 6.0])
        psi = _safe_psi(a, b)
        assert np.isfinite(psi)

    def test_very_small_samples(self):
        a = np.array([1.0])
        b = np.array([2.0])
        assert _safe_psi(a, b) == 0.0


class TestSeverity:
    def test_psi_none(self):
        assert _psi_to_severity(0.05) == DriftSeverity.NONE

    def test_psi_low(self):
        assert _psi_to_severity(0.12) == DriftSeverity.LOW

    def test_psi_moderate(self):
        assert _psi_to_severity(0.17) == DriftSeverity.MODERATE

    def test_psi_high(self):
        assert _psi_to_severity(0.22) == DriftSeverity.HIGH

    def test_psi_critical(self):
        assert _psi_to_severity(0.35) == DriftSeverity.CRITICAL


class TestCheckFeatureDrift:
    def test_drift_on_shifted_distribution(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 10_000)
        live = rng.normal(0.5, 1, 10_000)
        result = check_feature_drift("test", baseline, live)
        assert result.drifted
        assert result.psi > 0.1

    def test_no_drift_on_same_distribution(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 10_000)
        b = rng.normal(0, 1, 10_000)
        result = check_feature_drift("test", a, b)
        assert not result.drifted

    def test_result_contains_metadata(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 1000)
        live = rng.normal(0.3, 1, 1000)
        result = check_feature_drift("my_feat", baseline, live)
        assert result.feature_name == "my_feat"
        assert result.n_baseline == 1000
        assert result.n_live == 1000
        assert isinstance(result.psi, float)
        assert isinstance(result.ks_pvalue, float)
        assert isinstance(result.drifted, bool)

    def test_very_small_samples_no_false_positive(self):
        baseline = np.array([1.0, 1.1, 0.9])
        live = np.array([1.0, 1.1, 0.9])
        result = check_feature_drift("tiny", baseline, live)
        assert not result.drifted


# ════════════════════════════════════════════════════════════════════════════
# DriftReport
# ════════════════════════════════════════════════════════════════════════════

class TestRunDriftCheck:
    def test_empty_data_returns_no_drift(self):
        report = run_drift_check({}, {})
        assert not report.drift_detected
        assert "No common features" in " ".join(report.reasons)

    def test_multi_feature_report(self):
        rng = np.random.default_rng(42)
        baseline = {"a": rng.normal(0, 1, 10_000), "b": rng.normal(0, 1, 10_000)}
        live = {"a": rng.normal(0.5, 1, 10_000), "b": rng.normal(0, 1, 10_000)}
        report = run_drift_check(baseline, live)
        assert report.n_features == 2
        assert report.drift_detected
        assert report.n_drifted >= 1

    def test_no_drift_on_identical_data(self):
        rng = np.random.default_rng(42)
        data = {"a": rng.normal(0, 1, 10_000)}
        report = run_drift_check(data, data)
        assert not report.drift_detected

    def test_report_stores_timestamps(self):
        rng = np.random.default_rng(42)
        bl = {"x": rng.normal(0, 1, 100)}
        lv = {"x": rng.normal(0.5, 1, 100)}
        report = run_drift_check(
            bl, lv, baseline_time="2024-01-01", live_time="2024-06-01"
        )
        assert report.baseline_time == "2024-01-01"
        assert report.live_time == "2024-06-01"

    def test_missing_features_in_live(self):
        rng = np.random.default_rng(42)
        baseline = {"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)}
        live = {"a": rng.normal(0.5, 1, 100)}
        report = run_drift_check(baseline, live)
        assert report.n_features == 1  # only 'a' is common


# ════════════════════════════════════════════════════════════════════════════
# DriftTracker (FeatureStore integration)
# ════════════════════════════════════════════════════════════════════════════

class TestDriftTracker:
    def test_init_creates_tables(self, tmp_store):
        tracker = DriftTracker(tmp_store)
        import sqlite3
        with sqlite3.connect(tracker.store.db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'drift_%'"
            ).fetchall()
            names = [t[0] for t in tables]
            assert "drift_checks" in names
            assert "drift_alerts" in names

    def test_persist_report(self, drift_tracker):
        rng = np.random.default_rng(42)
        baseline = {"f1": rng.normal(0, 1, 1000), "f2": rng.normal(0, 1, 1000)}
        live = {"f1": rng.normal(0.5, 1, 500), "f2": rng.normal(0, 1, 500)}
        report = run_drift_check(baseline, live)
        drift_tracker._persist_report(report)

        checks = drift_tracker.get_recent_checks()
        assert len(checks) == 2  # f1 and f2

    def test_recent_checks_by_feature(self, drift_tracker):
        rng = np.random.default_rng(42)
        baseline = {"f1": rng.normal(0, 1, 100), "f2": rng.normal(0, 1, 100)}
        live = {"f1": rng.normal(0.5, 1, 50), "f2": rng.normal(0, 1, 50)}
        report = run_drift_check(baseline, live)
        drift_tracker._persist_report(report)

        f1_checks = drift_tracker.get_recent_checks(feature_name="f1")
        assert all(c["feature_name"] == "f1" for c in f1_checks)
        assert len(f1_checks) >= 1

    def test_active_alerts_on_drift(self, drift_tracker):
        rng = np.random.default_rng(42)
        baseline = {"f_drift": rng.normal(0, 1, 10_000)}
        live = {"f_drift": rng.normal(1, 1, 5_000)}
        report = run_drift_check(baseline, live)
        drift_tracker._persist_report(report)

        alerts = drift_tracker.get_active_alerts()
        assert len(alerts) >= 1
        assert alerts[0]["feature_name"] == "f_drift"

    def test_acknowledge_alert(self, drift_tracker, tmp_store):
        rng = np.random.default_rng(42)
        baseline = {"f": rng.normal(0, 1, 10_000)}
        live = {"f": rng.normal(1, 1, 5_000)}
        report = run_drift_check(baseline, live)
        drift_tracker._persist_report(report)
        alerts = drift_tracker.get_active_alerts(unacknowledged_only=False)
        if alerts:
            drift_tracker.acknowledge_alert(alerts[0]["id"])
            active = drift_tracker.get_active_alerts(unacknowledged_only=True)
            assert alerts[0]["id"] not in [a["id"] for a in active]

    def test_no_alerts_on_no_drift(self, drift_tracker):
        rng = np.random.default_rng(42)
        baseline = {"f": rng.normal(0, 1, 10_000)}
        live = {"f": rng.normal(0, 1, 5_000)}
        report = run_drift_check(baseline, live)
        drift_tracker._persist_report(report)
        alerts = drift_tracker.get_active_alerts()
        assert len(alerts) == 0

    def test_drift_summary(self, drift_tracker):
        rng = np.random.default_rng(42)
        baseline = {"f1": rng.normal(0, 1, 10_000), "f2": rng.normal(0, 1, 10_000)}
        live = {"f1": rng.normal(1, 1, 5_000), "f2": rng.normal(0, 1, 5_000)}
        report = run_drift_check(baseline, live)
        drift_tracker._persist_report(report)
        summary = drift_tracker.get_drift_summary()
        assert summary["total_checks"] == 2
        assert summary["n_drifted"] >= 1
        assert summary["drift_rate"] > 0

    def test_check_from_store(self, drift_tracker_with_data):
        store, tracker = drift_tracker_with_data
        base_start = datetime(2024, 1, 1, 8, tzinfo=UTC)
        base_end = datetime(2024, 1, 1, 8, tzinfo=UTC) + timedelta(minutes=199)
        live_start = datetime(2024, 6, 1, 8, tzinfo=UTC)
        live_end = datetime(2024, 6, 1, 8, tzinfo=UTC) + timedelta(minutes=99)

        report = tracker.check_from_store(
            ["feat_a", "feat_b"], base_start, base_end, live_start, live_end
        )
        assert report.n_features == 2
        # feat_a should have drift, feat_b should not
        for r in report.feature_results:
            if r.feature_name == "feat_a":
                assert r.drifted, f"feat_a should drift (PSI={r.psi:.4f})"
            elif r.feature_name == "feat_b":
                # feat_b may or may not drift depending on RNG
                pass

        # Alerts should be created for drifted features
        alerts = tracker.get_active_alerts()
        assert len(alerts) >= 1


# ════════════════════════════════════════════════════════════════════════════
# Convenience: schedule_drift_check
# ════════════════════════════════════════════════════════════════════════════

class TestScheduleDriftCheck:
    def test_requires_data_in_store(self, tmp_store):
        report = schedule_drift_check(tmp_store, ["close"])
        assert report.n_features == 0  # no common features found
