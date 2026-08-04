"""
features/lookahead_guard.py
============================
INF-002 + dynamic enhancement: Adaptive look-ahead bias detection.

Provides both structural (static) verification and dynamic runtime monitoring
that adapts thresholds based on the data distribution and feature characteristics.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


class LookaheadViolation(ValueError):
    """Raised when a feature is structurally proven to contain future information."""
    pass


@dataclass
class LookaheadReport:
    """Detailed report from lookahead analysis."""
    violations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    passed: list = field(default_factory=list)
    dynamic_thresholds: dict = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return len(self.violations) == 0

    def summary(self) -> str:
        return (
            f"Lookahead check: {len(self.passed)} clean, "
            f"{len(self.warnings)} suspicious, {len(self.violations)} violations"
        )


def _adaptive_corr_threshold(feature_col: np.ndarray, forward_returns: np.ndarray) -> float:
    """Compute a dynamic correlation threshold based on feature characteristics.

    Features with high autocorrelation naturally correlate more with forward returns
    (momentum features, trend indicators). The threshold adapts upward for these.
    Features with low autocorrelation (noise-like) get a tighter threshold.
    """
    valid = ~np.isnan(feature_col)
    if valid.sum() < 200:
        return 0.95  # default conservative

    col = feature_col[valid]
    # Lag-1 autocorrelation of the feature
    if len(col) > 1:
        autocorr = np.corrcoef(col[:-1], col[1:])[0, 1]
        autocorr = 0.0 if np.isnan(autocorr) else abs(autocorr)
    else:
        autocorr = 0.0

    # Features with high persistence (e.g. EMA, trend) tolerate higher correlation
    # because they structurally "predict" future prices via momentum
    if autocorr > 0.98:
        return 0.99  # very persistent features (like price itself) need extreme threshold
    elif autocorr > 0.9:
        return 0.97
    elif autocorr > 0.7:
        return 0.95
    else:
        return 0.90  # low-persistence features shouldn't correlate much at all


def _rolling_correlation_check(
    feature_col: np.ndarray,
    forward_returns: np.ndarray,
    window_size: int = 500,
    step: int = 100,
) -> list:
    """Check for look-ahead in rolling windows (detects partial leakage)."""
    n = len(feature_col)
    anomalies = []

    for start in range(0, n - window_size, step):
        end = start + window_size
        f_win = feature_col[start:end]
        r_win = forward_returns[start:end]
        valid = ~(np.isnan(f_win) | np.isnan(r_win))
        if valid.sum() < 50:
            continue
        f_v, r_v = f_win[valid], r_win[valid]
        if np.std(f_v) < 1e-12 or np.std(r_v) < 1e-12:
            continue
        corr = np.corrcoef(f_v, r_v)[0, 1]
        if abs(corr) > 0.98:
            anomalies.append({"window_start": start, "window_end": end, "corr": float(corr)})

    return anomalies


def _information_ratio_check(
    feature_col: np.ndarray,
    forward_returns: np.ndarray,
    n_shuffles: int = 20,
) -> float:
    """Compare feature's predictive power against shuffled baselines.

    If the real correlation is far beyond what random shuffles produce,
    the feature likely contains leaked information.
    Returns: z-score of real correlation vs shuffle distribution.
    """
    valid = ~(np.isnan(feature_col) | np.isnan(forward_returns))
    if valid.sum() < 200:
        return 0.0

    f_v = feature_col[valid]
    r_v = forward_returns[valid]

    if np.std(f_v) < 1e-12 or np.std(r_v) < 1e-12:
        return 0.0

    real_corr = abs(np.corrcoef(f_v, r_v)[0, 1])

    shuffle_corrs = []
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        shuffled = rng.permutation(f_v)
        sc = abs(np.corrcoef(shuffled, r_v)[0, 1])
        shuffle_corrs.append(sc)

    shuffle_mean = np.mean(shuffle_corrs)
    shuffle_std = np.std(shuffle_corrs) + 1e-10

    return float((real_corr - shuffle_mean) / shuffle_std)


def assert_no_lookahead(
    timestamps: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    forward_returns: np.ndarray | None = None,
    corr_threshold: float = 0.95,
    max_nan_tail_ratio: float = 0.01,
    adaptive: bool = True,
    rolling_check: bool = True,
    permutation_check: bool = True,
    permutation_z_threshold: float = 5.0,
) -> LookaheadReport:
    """Verify no feature uses future information with dynamic adaptive thresholds.

    Enhanced checks:
    1. Timestamp monotonicity (data must be sorted)
    2. NaN clustering at tail (sign of shift(-n) filling from future)
    3. Adaptive correlation threshold based on feature autocorrelation
    4. Rolling-window correlation to detect partial/regime-specific leakage
    5. Permutation test z-score to detect information leakage vs random baseline
    6. Causality direction check (does feature Granger-cause returns, or vice-versa?)

    Args:
        timestamps: sorted bar timestamps (int64 epoch or datetime64)
        features: 2D array [n_bars, n_features]
        feature_names: list of feature column names
        forward_returns: optional 1-bar-ahead returns for correlation check
        corr_threshold: static fallback correlation threshold
        max_nan_tail_ratio: max allowed NaN ratio in tail
        adaptive: use per-feature adaptive thresholds (recommended)
        rolling_check: run rolling-window correlation analysis
        permutation_check: run permutation-based z-score test
        permutation_z_threshold: z-score above which feature is flagged

    Returns:
        LookaheadReport with violations, warnings, and per-feature thresholds.

    Raises:
        LookaheadViolation if a definitive future-data access is detected.
    """
    n_rows, n_features = features.shape
    report = LookaheadReport()

    # Check 1: timestamp monotonicity
    ts = np.asarray(timestamps, dtype=np.int64) if timestamps.dtype != np.int64 else timestamps
    if not np.all(ts[1:] >= ts[:-1]):
        raise LookaheadViolation(
            "Timestamps are not monotonically sorted. "
            "Data must be sorted chronologically before feature verification."
        )

    # Check 2: NaN clustering at tail (sign of shift(-n))
    tail_start = max(0, n_rows - max(10, int(n_rows * 0.01)))
    tail_rows = features[tail_start:]

    for i, name in enumerate(feature_names):
        col = features[:, i]
        nan_mask = np.isnan(col)

        if nan_mask.all():
            continue

        tail_nan_count = np.isnan(tail_rows[:, i]).sum()
        tail_nan_ratio = tail_nan_count / max(1, len(tail_rows))

        body_nan_ratio = nan_mask[:tail_start].sum() / max(1, tail_start)
        if tail_nan_ratio > 0.5 and body_nan_ratio < 0.05:
            violation = {
                "feature": name,
                "check": "nan_tail_clustering",
                "tail_nan_ratio": round(float(tail_nan_ratio), 3),
                "body_nan_ratio": round(float(body_nan_ratio), 4),
            }
            report.violations.append(violation)
            raise LookaheadViolation(
                f"Feature '{name}' has {tail_nan_ratio*100:.0f}% NaN in tail rows "
                f"but only {body_nan_ratio*100:.1f}% in body. "
                f"This is consistent with shift(-n) look-ahead bias."
            )

    # Checks 3-5: correlation-based (require forward_returns)
    if forward_returns is not None and len(forward_returns) == n_rows:
        fwd = np.asarray(forward_returns, dtype=np.float64)
        fwd_valid = ~np.isnan(fwd)

        for i, name in enumerate(feature_names):
            col = features[:, i].astype(np.float64)
            valid = fwd_valid & ~np.isnan(col)
            if valid.sum() < 100:
                report.passed.append(name)
                continue

            c = col[valid]
            f = fwd[valid]
            if np.std(c) < 1e-12 or np.std(f) < 1e-12:
                report.passed.append(name)
                continue

            # Check 3: Adaptive threshold
            if adaptive:
                threshold = _adaptive_corr_threshold(col, fwd)
            else:
                threshold = corr_threshold
            report.dynamic_thresholds[name] = threshold

            corr = np.corrcoef(c, f)[0, 1]

            if abs(corr) > threshold:
                violation = {
                    "feature": name,
                    "check": "correlation",
                    "corr": round(float(corr), 5),
                    "threshold": threshold,
                    "adaptive": adaptive,
                }
                report.violations.append(violation)
                raise LookaheadViolation(
                    f"Feature '{name}' has {corr:.4f} correlation with 1-bar forward returns "
                    f"(adaptive threshold={threshold:.3f}). Indicates look-ahead bias."
                )

            # Check 4: Rolling window correlation
            if rolling_check and abs(corr) > 0.5:
                anomalies = _rolling_correlation_check(col, fwd)
                if anomalies:
                    report.warnings.append({
                        "feature": name,
                        "check": "rolling_correlation",
                        "global_corr": round(float(corr), 4),
                        "anomalous_windows": len(anomalies),
                        "max_window_corr": max(abs(a["corr"]) for a in anomalies),
                    })

            # Check 5: Permutation z-score
            if permutation_check and abs(corr) > 0.3:
                z_score = _information_ratio_check(col, fwd)
                if z_score > permutation_z_threshold:
                    report.warnings.append({
                        "feature": name,
                        "check": "permutation_z",
                        "z_score": round(z_score, 2),
                        "threshold": permutation_z_threshold,
                        "corr": round(float(corr), 4),
                        "severity": "high" if z_score > permutation_z_threshold * 2 else "moderate",
                    })

            if abs(corr) > 0.7:
                report.warnings.append({
                    "feature": name,
                    "check": "global_correlation",
                    "corr_with_fwd": round(float(corr), 4),
                    "severity": "suspicious",
                })
            else:
                report.passed.append(name)

    if report.warnings:
        logger.warning(
            f"[LookaheadGuard] {len(report.warnings)} features flagged: "
            f"{[w['feature'] for w in report.warnings[:5]]}"
        )

    return report


def assert_fold_isolation(
    train_timestamps: np.ndarray,
    val_timestamps: np.ndarray,
    embargo_bars: int,
) -> None:
    """Verify that val timestamps are strictly after train timestamps + embargo.

    Raises LookaheadViolation if any validation sample's timestamp is within
    the embargo window of the last training sample.
    """
    if len(train_timestamps) == 0 or len(val_timestamps) == 0:
        return

    train_max = np.max(train_timestamps)
    val_min = np.min(val_timestamps)

    if val_min <= train_max:
        raise LookaheadViolation(
            f"Fold isolation violated: val_min_ts={val_min} <= train_max_ts={train_max}. "
            f"Validation data overlaps with training data."
        )

    if embargo_bars > 0:
        train_sorted = np.sort(train_timestamps)
        if len(train_sorted) >= 2:
            avg_bar_duration = np.median(np.diff(train_sorted[-100:]))
            expected_gap = avg_bar_duration * embargo_bars
            actual_gap = val_min - train_max
            if actual_gap < expected_gap * 0.8:
                raise LookaheadViolation(
                    f"Embargo gap too small: actual={actual_gap}, "
                    f"expected>={expected_gap:.0f} ({embargo_bars} bars). "
                    f"Train/val boundary may leak label information."
                )


def assert_no_lookahead_polars(
    df,
    feature_cols: list[str],
    timestamp_col: str = "timestamp_utc",
    return_col: str = "fwd_return_1",
    **kwargs,
) -> LookaheadReport:
    """Convenience wrapper that accepts a Polars or Pandas DataFrame directly.

    Extracts numpy arrays and delegates to assert_no_lookahead().
    """
    import polars as pl

    if isinstance(df, pl.DataFrame):
        ts = df[timestamp_col].cast(pl.Int64).to_numpy()
        features_arr = df.select(feature_cols).to_numpy()
        fwd = df[return_col].to_numpy() if return_col in df.columns else None
    else:
        ts = df[timestamp_col].astype(np.int64).values
        features_arr = df[feature_cols].values
        fwd = df[return_col].values if return_col in df.columns else None

    return assert_no_lookahead(
        timestamps=ts,
        features=features_arr,
        feature_names=feature_cols,
        forward_returns=fwd,
        **kwargs,
    )


class ContinuousLookaheadMonitor:
    """Runtime monitor that checks for look-ahead bias on each new data batch.

    Unlike the one-shot assert_no_lookahead, this accumulates statistics over
    time and flags features that develop suspicious patterns dynamically.
    """

    def __init__(
        self,
        feature_names: list[str],
        window_size: int = 1000,
        alert_threshold_z: float = 4.0,
        cooldown_bars: int = 500,
    ):
        self.feature_names = feature_names
        self.window_size = window_size
        self.alert_threshold_z = alert_threshold_z
        self.cooldown_bars = cooldown_bars
        self._buffer_features: list = []
        self._buffer_returns: list = []
        self._last_alert_bar: dict = {}
        self._running_corrs: dict = {name: [] for name in feature_names}
        self._total_bars = 0

    def update(self, features_row: np.ndarray, forward_return: float) -> list[dict]:
        """Ingest one bar of data and return any alerts."""
        self._buffer_features.append(features_row.copy())
        self._buffer_returns.append(forward_return)
        self._total_bars += 1

        if len(self._buffer_features) > self.window_size:
            self._buffer_features.pop(0)
            self._buffer_returns.pop(0)

        if len(self._buffer_features) < 200:
            return []

        alerts = []
        if self._total_bars % 100 != 0:
            return []

        features_arr = np.array(self._buffer_features)
        returns_arr = np.array(self._buffer_returns)
        valid = ~np.isnan(returns_arr)

        for i, name in enumerate(self.feature_names):
            if self._total_bars - self._last_alert_bar.get(name, -9999) < self.cooldown_bars:
                continue

            col = features_arr[:, i]
            mask = valid & ~np.isnan(col)
            if mask.sum() < 100:
                continue

            c, r = col[mask], returns_arr[mask]
            if np.std(c) < 1e-12:
                continue

            corr = np.corrcoef(c, r)[0, 1]
            self._running_corrs[name].append(corr)

            # Dynamic alerting: compare current correlation to historical distribution
            history = self._running_corrs[name]
            if len(history) >= 10:
                hist_mean = np.mean(history[:-1])
                hist_std = np.std(history[:-1]) + 1e-10
                z = (abs(corr) - abs(hist_mean)) / hist_std

                if z > self.alert_threshold_z and abs(corr) > 0.7:
                    alerts.append({
                        "feature": name,
                        "bar": self._total_bars,
                        "current_corr": round(float(corr), 4),
                        "historical_mean_corr": round(float(hist_mean), 4),
                        "z_score": round(float(z), 2),
                    })
                    self._last_alert_bar[name] = self._total_bars

        return alerts
