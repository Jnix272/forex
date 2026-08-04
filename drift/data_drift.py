"""
drift/data_drift.py — Data drift detection (Improvement #4)

Multi-signal drift detection pipeline:

  * Feature distribution drift — KS test / Wasserstein distance + PSI (reusing
    the PSI helper from features.feature_quality_monitor where available).
  * SHAP-based feature-attribution drift — compare train-time SHAP importances
    vs live importances; flag features whose relative importance shifts.
  * Concept drift detectors — ADWIN, Page-Hinkley, DDM/EDDM on model error
    with a streaming drift score.
  * Adversarial validation — train a classifier to distinguish train vs live
    samples; AUC above threshold ⇒ drift alert.

Emits structured drift events consumable by monitoring.alerting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as _sp_stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D arrays."""
    try:
        from features.feature_quality_monitor import _safe_psi
        return float(_safe_psi(np.asarray(expected, dtype=np.float64),
                               np.asarray(actual, dtype=np.float64), bins))
    except Exception:
        pass
    e = np.asarray(expected, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    e = e[np.isfinite(e)]; a = a[np.isfinite(a)]
    if e.size == 0 or a.size == 0:
        return 0.0
    lo, hi = float(np.quantile(np.concatenate([e, a]), 0.01)), float(np.quantile(np.concatenate([e, a]), 0.99))
    edges = np.linspace(lo, hi, bins + 1) if hi > lo else np.array([lo, lo + 1e-9])
    e_hist = np.histogram(e, bins=edges)[0].astype(np.float64)
    a_hist = np.histogram(a, bins=edges)[0].astype(np.float64)
    e_p = e_hist / max(e_hist.sum(), 1e-9) + 1e-6
    a_p = a_hist / max(a_hist.sum(), 1e-9) + 1e-6
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    if _HAS_SCIPY:
        return float(_sp_stats.wasserstein_distance(np.asarray(a), np.asarray(b)))
    a = np.sort(np.asarray(a, dtype=np.float64)); b = np.sort(np.asarray(b, dtype=np.float64))
    n = min(a.size, b.size)
    return float(np.mean(np.abs(a[:n] - b[:n]))) if n else 0.0


def _ks(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if _HAS_SCIPY:
        stat, p = _sp_stats.ks_2samp(np.asarray(a), np.asarray(b))
        return float(stat), float(p)
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    all_vals = np.unique(np.concatenate([a, b]))
    if all_vals.size == 0:
        return 0.0, 1.0
    ecdf_a = np.searchsorted(a, all_vals, side="right") / a.size
    ecdf_b = np.searchsorted(b, all_vals, side="right") / b.size
    return float(np.max(np.abs(ecdf_a - ecdf_b))), 0.5


# ═════════════════════════════════════════════════════════════════════════════
# Feature distribution drift
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    wasserstein: float
    drift: bool

    def to_event(self) -> Dict:
        return {
            "type": "feature_drift",
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "ks_stat": round(self.ks_stat, 4),
            "ks_pvalue": round(self.ks_pvalue, 4),
            "wasserstein": round(self.wasserstein, 6),
            "drift": self.drift,
        }


def check_feature_distribution_drift(
    reference: Dict[str, np.ndarray],
    live: Dict[str, np.ndarray],
    psi_threshold: float = 0.2,
    ks_pvalue_threshold: float = 0.05,
    wasserstein_threshold: float = 0.05,
) -> Dict:
    """Compare each feature's reference vs live distribution."""
    results: List[FeatureDriftResult] = []
    for name in reference:
        ref = np.asarray(reference[name], dtype=np.float64)
        lv = np.asarray(live.get(name, np.array([])), dtype=np.float64)
        ref = ref[np.isfinite(ref)]; lv = lv[np.isfinite(lv)]
        if ref.size < 5 or lv.size < 5:
            continue
        psi = _psi(ref, lv)
        ks_stat, ks_p = _ks(ref, lv)
        w = _wasserstein(ref, lv)
        drift = psi > psi_threshold or (ks_p < ks_pvalue_threshold and ks_stat > 0.05) or w > wasserstein_threshold
        results.append(FeatureDriftResult(name, psi, ks_stat, ks_p, w, drift))
    drifted = [r for r in results if r.drift]
    return {
        "feature_count": len(results),
        "drifted_count": len(drifted),
        "drifted_features": [r.feature for r in drifted],
        "max_psi": round(max((r.psi for r in results), default=0.0), 4),
        "results": [r.to_event() for r in results],
    }


# ═════════════════════════════════════════════════════════════════════════════
# SHAP-attribution drift
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ShapDriftResult:
    feature: str
    ref_importance: float
    live_importance: float
    shift: float
    drift: bool

    def to_event(self) -> Dict:
        return {
            "type": "shap_drift",
            "feature": self.feature,
            "ref_importance": round(self.ref_importance, 6),
            "live_importance": round(self.live_importance, 6),
            "shift": round(self.shift, 6),
            "drift": self.drift,
        }


def check_shap_attribution_drift(
    ref_importance: Dict[str, float],
    live_importance: Dict[str, float],
    shift_threshold: float = 0.25,
) -> Dict:
    """Compare SHAP importance distributions (absolute mean |SHAP| per feature).

    ``ref_importance`` / ``live_importance`` map feature name → importance.
    """
    results: List[ShapDriftResult] = []
    all_feats = set(ref_importance) | set(live_importance)
    ref_total = sum(abs(v) for v in ref_importance.values()) or 1e-9
    live_total = sum(abs(v) for v in live_importance.values()) or 1e-9
    for feat in sorted(all_feats):
        ref_w = abs(ref_importance.get(feat, 0.0)) / ref_total
        live_w = abs(live_importance.get(feat, 0.0)) / live_total
        shift = abs(live_w - ref_w)
        results.append(ShapDriftResult(feat, ref_w, live_w, shift, shift > shift_threshold))
    drifted = [r for r in results if r.drift]
    return {
        "drifted_count": len(drifted),
        "drifted_features": [r.feature for r in drifted],
        "max_shift": round(max((r.shift for r in results), default=0.0), 6),
        "results": [r.to_event() for r in results],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Concept-drift streaming detectors
# ═════════════════════════════════════════════════════════════════════════════

class ADWIN:
    """ADWIN (Adaptive Windowing, Bifet & Gavaldà) on streaming errors."""

    def __init__(self, delta: float = 0.002, max_bucket_rows: int = 5, min_window: int = 30):
        self.delta = delta
        self.min_window = min_window
        self._values: List[float] = []
        self._changed = False

    def add(self, value: float) -> bool:
        """Add an error observation; returns True if drift was detected."""
        self._values.append(float(value))
        if len(self._values) < self.min_window:
            return False
        self._changed = False
        n = len(self._values)
        total_mean = float(np.mean(self._values))
        eps = math_cutoff(self.delta, n)
        # two-window split test
        for cut in range(max(1, n // 10), n):
            left = self._values[:cut]; right = self._values[cut:]
            if len(left) < 5 or len(right) < 5:
                continue
            diff = abs(float(np.mean(left)) - float(np.mean(right)))
            if diff > eps:
                self._values = self._values[cut:]
                self._changed = True
                return True
        return False

    def get_estimation(self) -> float:
        return float(np.mean(self._values)) if self._values else 0.0

    def width(self) -> int:
        return len(self._values)


def math_cutoff(delta: float, n: int) -> float:
    """ADWIN cut-off: sqrt( (2 sigma^2 / n) * ln(1/delta) ) * (1 + sqrt(2/n))"""
    sigma2 = 0.25
    return math.sqrt(2.0 * sigma2 * math.log(1.0 / max(delta, 1e-12)) / max(n, 1)) * (1.0 + math.sqrt(2.0 / max(n, 1)))


class PageHinkley:
    """Page-Hinkley change detector on streaming error values."""

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 1.0):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self._sum = 0.0
        self._min = 0.0
        self._changed = False

    def add(self, value: float) -> bool:
        self._sum += float(value) - self.delta
        self._min = min(self._min, self._sum)
        self._changed = (self._sum - self._min) > self.lambda_
        return self._changed

    def reset(self) -> None:
        self._sum = 0.0
        self._min = 0.0
        self._changed = False


class DDM:
    """DDM (Drift Detection Method, Gama et al.) on error rate."""

    def __init__(self, min_instances: int = 30, warning_level: float = 2.0, drift_level: float = 3.0):
        self.min_instances = min_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self._n = 0
        self._p = 0.0
        self._s = 0.0
        self._p_min = 0.0
        self._s_min = 0.0
        self.state = "in_concept"

    def add(self, error: bool) -> str:
        """error=True on a misprediction. Returns state: in_concept/warning/drift."""
        self._n += 1
        self._p += (int(error) - self._p) / self._n
        self._s = math.sqrt(self._p * (1.0 - self._p) / max(self._n, 1))
        if self._n < self.min_instances:
            return self.state
        if self._p + self._s < self._p_min + self._s_min:
            self._p_min, self._s_min = self._p, self._s
            self.state = "in_concept"
        if self._p + self._s > self._p_min + self.warning_level * self._s_min:
            self.state = "warning"
            return self.state
        if self._p + self._s > self._p_min + self.drift_level * self._s_min:
            self.state = "drift"
            return self.state
        self.state = "in_concept"
        return self.state


class EDDM:
    """EDDM (Early Drift Detection Method) on error-distance distribution."""

    def __init__(self, min_instances: int = 30, warning_level: float = 0.95, drift_level: float = 0.90):
        self.min_instances = min_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self._n = 0
        self._last_error_dist = 0
        self._mean_dist = 0.0
        self._std_dist = 0.0
        self._max_dist = 0.0
        self._max_ratio = 0.0
        self.state = "in_concept"

    def add(self, error: bool) -> str:
        self._n += 1
        self._last_error_dist += 1
        if error:
            dist = self._last_error_dist
            self._last_error_dist = 0
            # EWMA of error distance
            self._mean_dist += (dist - self._mean_dist) / max(self._n, 1)
            self._std_dist = math.sqrt(max(self._std_dist ** 2 + (dist - self._mean_dist) ** 2 / max(self._n, 1), 0.0))
            if self._mean_dist + self._std_dist > self._max_dist:
                self._max_dist = self._mean_dist + self._std_dist
        if self._n < self.min_instances:
            return self.state
        if self._max_dist <= 0:
            return self.state
        ratio = (self._mean_dist + self._std_dist) / self._max_dist
        if ratio < self.drift_level:
            self.state = "drift"
        elif ratio < self.warning_level:
            self.state = "warning"
        else:
            self.state = "in_concept"
        return self.state


@dataclass
class StreamingDriftEvent:
    detector: str
    state: str
    value: float
    ts: str = field(default_factory=_now_iso)

    def to_event(self) -> Dict:
        return {"type": "concept_drift", "detector": self.detector,
                "state": self.state, "value": round(self.value, 6), "ts": self.ts}


class ConceptDriftTracker:
    """Runs ADWIN / PHT / DDM / EDDM in parallel over streaming errors."""

    def __init__(
        self,
        use_adwin: bool = True,
        use_page_hinkley: bool = True,
        use_ddm: bool = True,
        use_eddm: bool = True,
        severity_state: str = "drift",
    ):
        self.adwin = ADWIN() if use_adwin else None
        self.pht = PageHinkley() if use_page_hinkley else None
        self.ddm = DDM() if use_ddm else None
        self.eddm = EDDM() if use_eddm else None
        self.severity_state = severity_state
        self._events: List[StreamingDriftEvent] = []

    def add_error(self, error: bool, magnitude: float = 1.0) -> List[StreamingDriftEvent]:
        events: List[StreamingDriftEvent] = []
        val = magnitude if error else 0.0
        if self.adwin is not None and self.adwin.add(val):
            events.append(StreamingDriftEvent("ADWIN", "drift", self.adwin.get_estimation()))
        if self.pht is not None and self.pht.add(val):
            events.append(StreamingDriftEvent("PageHinkley", "drift", val))
        if self.ddm is not None:
            st = self.ddm.add(error)
            if st == self.severity_state:
                events.append(StreamingDriftEvent("DDM", st, self.ddm._p))
        if self.eddm is not None:
            st = self.eddm.add(error)
            if st == self.severity_state:
                events.append(StreamingDriftEvent("EDDM", st, self.eddm._mean_dist))
        self._events.extend(events)
        return events

    def streaming_score(self) -> float:
        """Fraction of active detectors currently in drift state (0..1)."""
        states = []
        for det in (self.adwin, self.pht, self.ddm, self.eddm):
            if det is None:
                continue
            if isinstance(det, ADWIN):
                states.append(1.0 if det._changed else 0.0)
            elif isinstance(det, PageHinkley):
                states.append(1.0 if det._changed else 0.0)
            elif isinstance(det, DDM):
                states.append(1.0 if det.state == "drift" else 0.0)
            else:
                states.append(1.0 if det.state == "drift" else 0.0)
        return float(np.mean(states)) if states else 0.0

    def events(self) -> List[Dict]:
        return [e.to_event() for e in self._events]


# ═════════════════════════════════════════════════════════════════════════════
# Adversarial validation
# ═════════════════════════════════════════════════════════════════════════════

def adversarial_validation(
    train: np.ndarray,
    live: np.ndarray,
    auc_threshold: float = 0.7,
    n_estimators: int = 100,
    max_depth: int = 3,
) -> Dict:
    """Train a classifier to distinguish train vs live samples. AUC near 0.5 ⇒
    indistinguishable; AUC above threshold ⇒ drift.

    Uses a histogram-features reduction so it works without sklearn if needed
    (exact AUC computed from ranked predictions)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        _HAS_SKLEARN = True
    except Exception:
        _HAS_SKLEARN = False

    train = np.asarray(train, dtype=np.float64)
    live = np.asarray(live, dtype=np.float64)
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if live.ndim == 1:
        live = live.reshape(-1, 1)
    if train.shape[0] < 10 or live.shape[0] < 10:
        return {"auc": 0.5, "drift": False, "n_train": train.shape[0], "n_live": live.shape[0],
                "method": "insufficient_data", "events": []}

    # align feature dims
    nf = min(train.shape[1], live.shape[1])
    X = np.vstack([train[:, :nf], live[:, :nf]])
    y = np.concatenate([np.zeros(len(train)), np.ones(len(live))])

    if _HAS_SKLEARN:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
        try:
            pred = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(y, pred))
        except Exception:
            clf.fit(X, y)
            pred = clf.predict_proba(X)[:, 1]
            auc = _rank_auc(y, pred)
    else:
        auc = _histogram_auc(X, y, nf)

    drift = auc > auc_threshold
    return {
        "auc": round(auc, 4),
        "drift": drift,
        "n_train": len(train),
        "n_live": len(live),
        "method": "random_forest" if _HAS_SKLEARN else "histogram",
        "auc_threshold": auc_threshold,
        "events": [{
            "type": "adversarial_drift",
            "auc": round(auc, 4),
            "drift": drift,
            "auc_threshold": auc_threshold,
        }],
    }


def _rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    """AUC from ranks (equivalent to Mann-Whitney U)."""
    order = np.argsort(-score, kind="mergesort")
    ranks = np.empty(len(y), dtype=np.float64)
    ranks[order] = np.arange(1, len(y) + 1)
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    u = float(ranks[y == 1].sum()) - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def _histogram_auc(X: np.ndarray, y: np.ndarray, nf: int) -> float:
    """Fallback AUC: reduce each sample to feature histograms, rank by the
    fraction of bins where the sample is above the joint median."""
    medians = np.median(X[:, :nf], axis=0)
    above = (X[:, :nf] > medians).mean(axis=1)
    return _rank_auc(y, above)


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_data_drift_check(
    reference_features: Dict[str, np.ndarray],
    live_features: Dict[str, np.ndarray],
    ref_shap_importance: Optional[Dict[str, float]] = None,
    live_shap_importance: Optional[Dict[str, float]] = None,
    train_matrix: Optional[np.ndarray] = None,
    live_matrix: Optional[np.ndarray] = None,
    error_stream: Optional[Sequence[bool]] = None,
    error_magnitudes: Optional[Sequence[float]] = None,
    psi_threshold: float = 0.2,
    ks_pvalue_threshold: float = 0.05,
    wasserstein_threshold: float = 0.05,
    shap_shift_threshold: float = 0.25,
    auc_threshold: float = 0.7,
) -> Dict:
    """One-call data-drift audit emitting structured events for alerting."""
    events: List[Dict] = []

    dist = check_feature_distribution_drift(
        reference_features, live_features, psi_threshold, ks_pvalue_threshold, wasserstein_threshold
    )
    events.extend(dist["results"])

    shap = None
    if ref_shap_importance is not None and live_shap_importance is not None:
        shap = check_shap_attribution_drift(ref_shap_importance, live_shap_importance, shap_shift_threshold)
        events.extend(shap["results"])

    adv = None
    if train_matrix is not None and live_matrix is not None:
        adv = adversarial_validation(train_matrix, live_matrix, auc_threshold)
        events.extend(adv["events"])

    concept = None
    if error_stream is not None:
        tracker = ConceptDriftTracker()
        for i, err in enumerate(error_stream):
            mag = error_magnitudes[i] if error_magnitudes is not None and i < len(error_magnitudes) else 1.0
            tracker.add_error(bool(err), mag)
        concept = {"streaming_score": tracker.streaming_score(), "events": tracker.events()}
        events.extend(concept["events"])

    drifted_count = dist["drifted_count"]
    if shap:
        drifted_count += shap["drifted_count"]
    if adv and adv["drift"]:
        drifted_count += 1

    return {
        "ts": _now_iso(),
        "feature_distribution": dist,
        "shap_attribution": shap,
        "adversarial": adv,
        "concept_drift": concept,
        "drifted_count": drifted_count,
        "alert": drifted_count > 0,
        "events": events,
    }
