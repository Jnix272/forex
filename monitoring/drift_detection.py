"""
Drift Detection Module (Phase 2)
=================================
Multi-method drift detection integrated with the Feature Store:
- PSI (Population Stability Index)
- KS (Kolmogorov-Smirnov) with effect-size guard
- Windowed/rolling drift tracking over time
- Per-feature drift persistence in SQLite
- Drift summary with alert generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

import numpy as np

from data.feature_store import FeatureStore

try:
    from scipy import stats as _sp_stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════

# PSI thresholds (common convention)
PSI_LOW    = 0.1   # No significant shift
PSI_MEDIUM = 0.2   # Moderate shift — monitor
# Above 0.2 = significant shift → alert / retrain

# KS default thresholds
KS_PVALUE_THRESHOLD = 0.05
KS_STAT_THRESHOLD   = 0.05  # Minimum effect size (D-stat) to flag

DEFAULT_WINDOW_BASELINE = 20_000  # Rows for reference distribution
DEFAULT_WINDOW_LIVE     = 5_000   # Rows for recent distribution


class DriftSeverity(Enum):
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other):
        if not isinstance(other, DriftSeverity):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other):
        if not isinstance(other, DriftSeverity):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other):
        if not isinstance(other, DriftSeverity):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other):
        if not isinstance(other, DriftSeverity):
            return NotImplemented
        return self.value >= other.value


@dataclass
class DriftResult:
    """Result of drift check for a single feature."""
    feature_name: str
    psi: float
    ks_pvalue: float
    ks_stat: float
    severity: DriftSeverity
    drifted: bool
    baseline_mean: float
    live_mean: float
    baseline_std: float
    live_std: float
    n_baseline: int
    n_live: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class DriftReport:
    """Aggregate drift report across features."""
    feature_results: list[DriftResult]
    psi_max: float
    ks_min_pvalue: float
    ks_max_stat: float
    n_drifted: int
    n_features: int
    overall_severity: DriftSeverity
    drift_detected: bool
    baseline_time: str
    live_time: str
    reasons: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════════
# Core: Per-feature drift metrics
# ════════════════════════════════════════════════════════════════════════════

def _safe_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with degenerate-distribution hardening."""
    eps = 1e-9
    expected = expected[np.isfinite(expected)]
    actual   = actual[np.isfinite(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return 0.0
    lo = min(float(expected.min()), float(actual.min()))
    hi = max(float(expected.max()), float(actual.max()))
    if hi - lo < eps:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    exp_hist, _ = np.histogram(expected, bins=edges)
    act_hist, _ = np.histogram(actual, bins=edges)
    exp_hist = exp_hist.astype(np.float64)
    act_hist = act_hist.astype(np.float64)
    exp_sum = exp_hist.sum()
    act_sum = act_hist.sum()
    if exp_sum == 0 or act_sum == 0:
        return 0.0
    exp_p = (exp_hist + eps) / (exp_sum + eps * bins)
    act_p = (act_hist + eps) / (act_sum + eps * bins)
    psi = float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))
    return psi if np.isfinite(psi) else 0.0


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test with guard for small samples."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 1.0, 0.0
    if _HAS_SCIPY:
        stat, p = _sp_stats.ks_2samp(a, b)
        return float(stat), float(p)
    return 1.0, 1.0  # Can't compute — no drift detected


def _psi_to_severity(psi: float) -> DriftSeverity:
    if psi > 0.3:
        return DriftSeverity.CRITICAL
    elif psi > 0.2:
        return DriftSeverity.HIGH
    elif psi > 0.15:
        return DriftSeverity.MODERATE
    elif psi > 0.1:
        return DriftSeverity.LOW
    return DriftSeverity.NONE


def _ks_to_severity(ks_pvalue: float, ks_stat: float) -> DriftSeverity:
    if not _HAS_SCIPY:
        return DriftSeverity.NONE
    if ks_pvalue < 0.01 and ks_stat > 0.1:
        return DriftSeverity.CRITICAL
    elif ks_pvalue < 0.05 and ks_stat > 0.05:
        return DriftSeverity.HIGH
    elif ks_pvalue < 0.1 and ks_stat > 0.05:
        return DriftSeverity.MODERATE
    return DriftSeverity.NONE


# ════════════════════════════════════════════════════════════════════════════
# Drift Checker (single-shot)
# ════════════════════════════════════════════════════════════════════════════

def check_feature_drift(
    feature_name: str,
    baseline: np.ndarray,
    live: np.ndarray,
    psi_bins: int = 10,
    ks_pvalue_threshold: float = KS_PVALUE_THRESHOLD,
    ks_stat_threshold: float = KS_STAT_THRESHOLD,
) -> DriftResult:
    """Run PSI + KS on a single feature."""
    psi  = _safe_psi(baseline, live, bins=psi_bins)
    ks_stat, ks_p = _ks_2samp(baseline, live)

    psi_sev = _psi_to_severity(psi)
    ks_sev  = _ks_to_severity(ks_p, ks_stat) if _HAS_SCIPY else DriftSeverity.NONE
    severity = max(psi_sev, ks_sev, key=lambda s: [
        DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MODERATE,
        DriftSeverity.HIGH, DriftSeverity.CRITICAL,
    ].index(s))

    # Drift if PSI > 0.2 OR (KS p < threshold AND stat >= threshold)
    ks_drift = ks_p < ks_pvalue_threshold and ks_stat >= ks_stat_threshold
    drifted = bool(psi > 0.2 or ks_drift)

    return DriftResult(
        feature_name=feature_name,
        psi=psi,
        ks_pvalue=ks_p,
        ks_stat=ks_stat,
        severity=severity,
        drifted=drifted,
        baseline_mean=float(np.nanmean(baseline)) if len(baseline) > 0 else 0.0,
        live_mean=float(np.nanmean(live)) if len(live) > 0 else 0.0,
        baseline_std=float(np.nanstd(baseline)) if len(baseline) > 0 else 0.0,
        live_std=float(np.nanstd(live)) if len(live) > 0 else 0.0,
        n_baseline=len(baseline),
        n_live=len(live),
    )


def run_drift_check(
    baseline_data: dict[str, np.ndarray],
    live_data: dict[str, np.ndarray],
    baseline_time: str = "",
    live_time: str = "",
    psi_bins: int = 10,
    ks_pvalue_threshold: float = KS_PVALUE_THRESHOLD,
    ks_stat_threshold: float = KS_STAT_THRESHOLD,
) -> DriftReport:
    """Run multi-feature drift check from dicts of feature_name -> array."""
    common = set(baseline_data) & set(live_data)
    if not common:
        return DriftReport(
            feature_results=[], psi_max=0.0, ks_min_pvalue=1.0, ks_max_stat=0.0,
            n_drifted=0, n_features=0, overall_severity=DriftSeverity.NONE,
            drift_detected=False, baseline_time=baseline_time, live_time=live_time,
            reasons=["No common features between baseline and live"],
        )

    results = []
    for name in sorted(common):
        bl = baseline_data[name]
        lv = live_data[name]
        result = check_feature_drift(
            name, bl, lv, psi_bins, ks_pvalue_threshold, ks_stat_threshold,
        )
        results.append(result)

    psi_max = max(r.psi for r in results) if results else 0.0
    ks_min_p = min(r.ks_pvalue for r in results) if results else 1.0
    ks_max_s = max(r.ks_stat for r in results) if results else 0.0
    n_drifted = sum(1 for r in results if r.drifted)

    max_sev = max(r.severity for r in results) if results else DriftSeverity.NONE
    drift_detected = n_drifted > 0

    reasons = []
    if drift_detected:
        drifted_names = sorted(r.feature_name for r in results if r.drifted)
        reasons.append(f"Drift detected in {n_drifted}/{len(results)} features: {drifted_names}")
    if psi_max > 0.2:
        reasons.append(f"PSI max {psi_max:.4f} > 0.2")

    return DriftReport(
        feature_results=results,
        psi_max=psi_max,
        ks_min_pvalue=ks_min_p,
        ks_max_stat=ks_max_s,
        n_drifted=n_drifted,
        n_features=len(common),
        overall_severity=max_sev,
        drift_detected=drift_detected,
        baseline_time=baseline_time,
        live_time=live_time,
        reasons=reasons,
    )


# ════════════════════════════════════════════════════════════════════════════
# FeatureStore-Integrated Drift Checker
# ════════════════════════════════════════════════════════════════════════════

class DriftTracker:
    """
    Persistent drift monitoring integrated with the Feature Store.
    Stores drift check results in SQLite for trend analysis and alerting.
    """

    def __init__(self, store: FeatureStore):
        self.store = store
        self._init_db()

    def _init_db(self) -> None:
        """Create drift tracking tables in FeatureStore's registry DB."""
        import sqlite3
        with sqlite3.connect(self.store.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    check_time TEXT NOT NULL,
                    baseline_time TEXT,
                    live_time TEXT,
                    n_baseline INTEGER,
                    n_live INTEGER,
                    psi REAL,
                    ks_pvalue REAL,
                    ks_stat REAL,
                    severity TEXT NOT NULL,
                    drifted INTEGER NOT NULL,
                    baseline_mean REAL,
                    live_mean REAL,
                    meta TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_drift_feature_time
                ON drift_checks (feature_name, check_time)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_id INTEGER,
                    feature_name TEXT NOT NULL,
                    alert_time TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    FOREIGN KEY (check_id) REFERENCES drift_checks(id)
                )
            """)

    def check_from_store(
        self,
        feature_names: list[str],
        baseline_start: datetime,
        baseline_end: datetime,
        live_start: datetime,
        live_end: datetime,
        psi_bins: int = 10,
    ) -> DriftReport:
        """
        Load baseline and live data from FeatureStore and run drift check.
        """
        baseline_data = {}
        live_data = {}
        missing = []

        for name in feature_names:
            bl = self.store.load_feature(name, baseline_start, baseline_end)
            lv = self.store.load_feature(name, live_start, live_end)
            if bl is None or len(bl) == 0:
                missing.append(name)
                continue
            if lv is None or len(lv) == 0:
                missing.append(name)
                continue
            feat_col = [c for c in bl.columns if c != "timestamp_utc"][0]
            baseline_data[name] = bl[feat_col].to_numpy()
            live_data[name] = lv[feat_col].to_numpy()

        report = run_drift_check(
            baseline_data, live_data,
            baseline_time=baseline_start.isoformat(),
            live_time=live_start.isoformat(),
            psi_bins=psi_bins,
        )

        # Persist results
        self._persist_report(report)

        if missing:
            report.reasons.append(f"Features not found in store: {missing}")

        return report

    def _persist_report(self, report: DriftReport) -> None:
        """Write drift results to SQLite."""
        import sqlite3
        with sqlite3.connect(self.store.db_path) as conn:
            now = datetime.now(UTC).isoformat()
            for r in report.feature_results:
                conn.execute("""
                    INSERT INTO drift_checks
                    (feature_name, check_time, baseline_time, live_time,
                     n_baseline, n_live, psi, ks_pvalue, ks_stat,
                     severity, drifted, baseline_mean, live_mean, meta)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r.feature_name, now, report.baseline_time, report.live_time,
                    r.n_baseline, r.n_live, r.psi, r.ks_pvalue, r.ks_stat,
                    r.severity.value, int(r.drifted),
                    r.baseline_mean, r.live_mean, "{}",
                ))

                if r.drifted:
                    check_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    conn.execute("""
                        INSERT INTO drift_alerts
                        (check_id, feature_name, alert_time, severity, message)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        check_id, r.feature_name, now, r.severity.value,
                        f"PSI={r.psi:.4f} KS_p={r.ks_pvalue:.6f} "
                        f"baseline_mean={r.baseline_mean:.6f} live_mean={r.live_mean:.6f}"
                    ))

    def get_recent_checks(
        self, feature_name: str = None, limit: int = 20
    ) -> list[dict]:
        """Get recent drift check history for a feature."""
        import sqlite3
        with sqlite3.connect(self.store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if feature_name:
                rows = conn.execute("""
                    SELECT * FROM drift_checks
                    WHERE feature_name = ? ORDER BY check_time DESC LIMIT ?
                """, (feature_name, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM drift_checks
                    ORDER BY check_time DESC LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_active_alerts(self, unacknowledged_only: bool = True) -> list[dict]:
        """Get active drift alerts."""
        import sqlite3
        with sqlite3.connect(self.store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if unacknowledged_only:
                rows = conn.execute("""
                    SELECT a.*, d.psi, d.ks_pvalue, d.ks_stat, d.severity as check_severity
                    FROM drift_alerts a
                    JOIN drift_checks d ON a.check_id = d.id
                    WHERE a.acknowledged = 0
                    ORDER BY a.alert_time DESC
                """).fetchall()
            else:
                rows = conn.execute("""
                    SELECT a.*, d.psi, d.ks_pvalue, d.ks_stat, d.severity as check_severity
                    FROM drift_alerts a
                    JOIN drift_checks d ON a.check_id = d.id
                    ORDER BY a.alert_time DESC
                """).fetchall()
            return [dict(r) for r in rows]

    def acknowledge_alert(self, alert_id: int) -> None:
        import sqlite3
        with sqlite3.connect(self.store.db_path) as conn:
            conn.execute(
                "UPDATE drift_alerts SET acknowledged = 1 WHERE id = ?",
                (alert_id,)
            )

    def get_drift_summary(self, since: datetime = None) -> dict:
        """Get summary statistics of recent drift activity."""
        import sqlite3
        with sqlite3.connect(self.store.db_path) as conn:
            if since:
                rows = conn.execute("""
                    SELECT severity, COUNT(*) as cnt, MAX(psi) as max_psi
                    FROM drift_checks
                    WHERE check_time >= ? AND drifted = 1
                    GROUP BY severity
                """, (since.isoformat(),)).fetchall()
                total = conn.execute("""
                    SELECT COUNT(*) as cnt FROM drift_checks WHERE check_time >= ?
                """, (since.isoformat(),)).fetchone()[0]
                n_drifted = conn.execute("""
                    SELECT COUNT(*) as cnt FROM drift_checks
                    WHERE check_time >= ? AND drifted = 1
                """, (since.isoformat(),)).fetchone()[0]
            else:
                rows = conn.execute("""
                    SELECT severity, COUNT(*) as cnt, MAX(psi) as max_psi
                    FROM drift_checks WHERE drifted = 1
                    GROUP BY severity
                """).fetchall()
                total = conn.execute("SELECT COUNT(*) as cnt FROM drift_checks").fetchone()[0]
                n_drifted = conn.execute(
                    "SELECT COUNT(*) as cnt FROM drift_checks WHERE drifted = 1"
                ).fetchone()[0]

            return {
                "total_checks": total,
                "n_drifted": n_drifted,
                "drift_rate": n_drifted / max(total, 1),
                "by_severity": [
                    {"severity": r[0], "count": r[1], "max_psi": r[2]} for r in rows
                ],
                "active_alerts": len(self.get_active_alerts()),
            }


# ════════════════════════════════════════════════════════════════════════════
# Scheduled drift checks
# ════════════════════════════════════════════════════════════════════════════

def schedule_drift_check(
    store: FeatureStore,
    feature_names: list[str],
    baseline_window_days: int = 90,
    live_window_days: int = 7,
    as_of: datetime | None = None,
) -> DriftReport:
    """
    Convenience: run drift check using time-windowed data from FeatureStore.
    Baseline = last N days, Live = last M days (with gap from baseline).
    """
    now = as_of or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    live_end = now
    live_start = now - timedelta(days=live_window_days)
    baseline_end = live_start - timedelta(days=1)  # gap to avoid overlap
    baseline_start = baseline_end - timedelta(days=baseline_window_days)

    tracker = DriftTracker(store)
    return tracker.check_from_store(
        feature_names, baseline_start, baseline_end, live_start, live_end,
    )


# ════════════════════════════════════════════════════════════════════════════
# Convenience: run drift check on raw arrays from NPY/Zarr cache
# (wraps the existing monitoring/drift_gate.py logic with named features)
# ════════════════════════════════════════════════════════════════════════════

def drift_check_from_cache(
    cache_path: str,
    feature_names: list[str],
    baseline_samples: int = DEFAULT_WINDOW_BASELINE,
    live_samples: int = DEFAULT_WINDOW_LIVE,
    psi_bins: int = 10,
) -> DriftReport:
    """
    Load data slices from cache (Zarr or NPY) and run named-feature drift check.
    Feature ordering must match the cache schema.
    """
    try:
        from monitoring.drift_gate import _dataset_length, _read_feature_slice
    except ImportError:
        raise ImportError("Need monitoring.drift_gate for cache loading")

    n_total = _dataset_length(cache_path)
    if n_total < 100:
        raise ValueError(f"Dataset too small for drift check: n_total={n_total}")

    b = max(50, min(baseline_samples, n_total // 2))
    l = max(50, min(live_samples, n_total - b))
    live_start = max(b, n_total - l)

    x_base = _read_feature_slice(cache_path, 0, b)
    x_live = _read_feature_slice(cache_path, live_start, n_total)

    n_feats = min(x_base.shape[1], len(feature_names))
    baseline_data = {feature_names[i]: x_base[:, i] for i in range(n_feats)}
    live_data = {feature_names[i]: x_live[:, i] for i in range(n_feats)}

    return run_drift_check(
        baseline_data, live_data,
        baseline_time=f"rows 0..{b}",
        live_time=f"rows {live_start}..{n_total}",
        psi_bins=psi_bins,
    )


if __name__ == "__main__":
    # Quick demo
    rng = np.random.default_rng(42)
    baseline = {"feat_a": rng.normal(0, 1, 10_000), "feat_b": rng.normal(0, 1, 10_000)}
    live = {"feat_a": rng.normal(0.3, 1.1, 5_000), "feat_b": rng.normal(0, 1, 5_000)}
    report = run_drift_check(baseline, live)
    print(f"Drift detected: {report.drift_detected}")
    print(f"PSI max: {report.psi_max:.4f}")
    print(f"Drifted features: {report.n_drifted}/{report.n_features}")
    for r in report.feature_results:
        print(f"  {r.feature_name}: PSI={r.psi:.4f} KS={r.ks_pvalue:.6f} drifted={r.drifted}")
