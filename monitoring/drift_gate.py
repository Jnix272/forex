"""
Drift gate and reporting helpers.

Compares an early-training baseline feature window vs a recent window from
cached model inputs (Zarr or NPY sidecars).

Two metrics:
  PSI - Population Stability Index per feature.
        Hardened to handle constant / degenerate distributions (returns 0.0).
  KS  - Kolmogorov-Smirnov test per feature.
        Only fails when BOTH p-value < threshold AND D-statistic >= min_effect_size.
        This prevents false alarms on very large samples (1M+) where even
        micro-differences produce p≈0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from config.settings import MONITORING

try:
    import zarr

    _HAS_ZARR = True
except Exception:
    _HAS_ZARR = False

try:
    from scipy import stats

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Default KS effect-size floor: only flag if D-statistic (max CDF gap) is this
# large or bigger.  0.05 ≈ 5 % CDF gap - meaningful distributional difference.
_DEFAULT_KS_STATISTIC_THRESHOLD = 0.05


def _resolve_npy_x(cache_path: str) -> Path:
    p = Path(cache_path)
    if p.is_file() and p.name.endswith("_X.npy"):
        return p
    if p.suffix == ".npy":
        return p
    return Path(str(p) + "_X.npy")


def _open_zarr_x(cache_path: str):
    if not _HAS_ZARR:
        return None
    p = Path(cache_path)
    if not (p.is_dir() and str(p).endswith(".zarr")):
        return None
    try:
        g = zarr.open_group(str(p), mode="r")
        if "X" in g:
            return g["X"]
    except Exception:
        return None
    return None


def _dataset_length(cache_path: str) -> int:
    z_x = _open_zarr_x(cache_path)
    if z_x is not None:
        return int(z_x.shape[0])
    x_path = _resolve_npy_x(cache_path)
    if x_path.exists():
        return int(np.load(str(x_path), mmap_mode="r").shape[0])
    raise FileNotFoundError(f"Could not locate cache features for: {cache_path}")


def _read_feature_slice(cache_path: str, start: int, end: int) -> np.ndarray:
    z_x = _open_zarr_x(cache_path)
    if z_x is not None:
        x = np.asarray(z_x[start:end, -1, :], dtype=np.float32)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_path = _resolve_npy_x(cache_path)
    if not x_path.exists():
        raise FileNotFoundError(f"Could not locate NPY cache: {x_path}")
    x = np.asarray(np.load(str(x_path), mmap_mode="r")[start:end, -1, :], dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index, hardened for degenerate distributions.

    Returns 0.0 (no detected shift) instead of NaN/Inf for:
      - constant or near-constant features
      - empty bins after histogram
      - non-finite values in either array
    """
    eps = 1e-9
    # Drop non-finite values before computing
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return 0.0
    # Build bin edges on combined range
    lo = min(float(expected.min()), float(actual.min()))
    hi = max(float(expected.max()), float(actual.max()))
    if hi - lo < eps:
        # Constant or near-constant feature - no distributional difference
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    exp_hist, _ = np.histogram(expected, bins=edges)
    act_hist, _ = np.histogram(actual, bins=edges)
    exp_sum = exp_hist.sum()
    act_sum = act_hist.sum()
    if exp_sum == 0 or act_sum == 0:
        return 0.0
    exp_p = (exp_hist.astype(np.float64) + eps) / (exp_sum + eps * bins)
    act_p = (act_hist.astype(np.float64) + eps) / (act_sum + eps * bins)
    psi = float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))
    return psi if np.isfinite(psi) else 0.0


def compute_drift_report(
    cache_path: str,
    baseline_samples: int = 20_000,
    live_samples: int = 5_000,
    psi_threshold: float = float(MONITORING.get("psi_threshold", 0.2)),
    ks_pvalue_threshold: float = float(MONITORING.get("ks_pvalue_threshold", 0.05)),
    ks_statistic_threshold: float = _DEFAULT_KS_STATISTIC_THRESHOLD,
    top_k_features: int = 20,
) -> dict[str, Any]:
    """
    Compute a drift report comparing baseline vs recent feature windows.

    KS gate requires BOTH:
      p-value < ks_pvalue_threshold
      AND D-statistic >= ks_statistic_threshold   (effect-size guard)
    This prevents false alarms on large datasets where p≈0 even for tiny shifts.
    """
    n_total = _dataset_length(cache_path)
    if n_total < 100:
        raise ValueError(f"Dataset too small for drift check: n_total={n_total}")

    b = max(50, min(int(baseline_samples), n_total // 2))
    l = max(50, min(int(live_samples), n_total - b))  # noqa: E741
    baseline_end = b
    live_start = max(b, n_total - l)
    live_end = n_total

    x_base = _read_feature_slice(cache_path, 0, baseline_end)
    x_live = _read_feature_slice(cache_path, live_start, live_end)

    n_feats = int(min(x_base.shape[1], x_live.shape[1]))

    psi_vals = np.zeros(n_feats, dtype=np.float64)
    for i in range(n_feats):
        psi_vals[i] = _safe_psi(x_base[:, i], x_live[:, i])
    psi_vals = np.nan_to_num(psi_vals, nan=0.0, posinf=0.0, neginf=0.0)

    ks_pvals = np.ones(n_feats, dtype=np.float64)
    ks_stats = np.zeros(n_feats, dtype=np.float64)
    if _HAS_SCIPY:
        for i in range(n_feats):
            a = x_base[:, i][np.isfinite(x_base[:, i])]
            b_ = x_live[:, i][np.isfinite(x_live[:, i])]
            if len(a) >= 2 and len(b_) >= 2:
                stat, p = stats.ks_2samp(a, b_)
                ks_pvals[i] = float(p)
                ks_stats[i] = float(stat)

    psi_max = float(np.max(psi_vals)) if n_feats > 0 else 0.0
    ks_min_pvalue = float(np.nanmin(ks_pvals)) if _HAS_SCIPY else 1.0
    ks_max_stat = float(np.nanmax(ks_stats)) if _HAS_SCIPY else 0.0

    psi_rank = np.argsort(-psi_vals)
    top_features: list[dict[str, float]] = []
    for idx in psi_rank[: max(1, int(top_k_features))]:
        top_features.append(
            {
                "feature_idx": int(idx),
                "psi": float(psi_vals[idx]),
                "ks_pvalue": float(ks_pvals[idx]) if _HAS_SCIPY else 1.0,
                "ks_stat": float(ks_stats[idx]) if _HAS_SCIPY else 0.0,
            }
        )

    reasons: list[str] = []
    if not np.isfinite(psi_max):
        psi_max = 0.0  # safety - should not happen after _safe_psi hardening
    psi_drift = psi_max > float(psi_threshold)
    if psi_drift:
        reasons.append(f"PSI {psi_max:.4f} > {float(psi_threshold):.4f}")
    # KS only fails when p is small AND the effect size (D-stat) is meaningful
    ks_drift = False
    if _HAS_SCIPY:
        ks_pvalue_fail = ks_min_pvalue < float(ks_pvalue_threshold)
        ks_effect_fail = ks_max_stat >= float(ks_statistic_threshold)
        ks_drift = ks_pvalue_fail and ks_effect_fail
        if ks_drift:
            reasons.append(
                f"KS p-value {ks_min_pvalue:.6f} < {float(ks_pvalue_threshold):.6f}"
                f" AND D-stat {ks_max_stat:.4f} >= {float(ks_statistic_threshold):.4f}"
            )

    if not _HAS_SCIPY:
        reasons.append("SciPy unavailable; KS checks skipped")

    drift_detected = bool(psi_drift or ks_drift)

    print(f"[DriftGate] psi_max={psi_max:.4f}  ks_max_stat={ks_max_stat:.4f}  ks_min_pvalue={ks_min_pvalue:.6f}")

    return {
        "drift_detected": drift_detected,
        "cache_path": str(cache_path),
        "n_total": int(n_total),
        "baseline_rows": len(x_base),
        "live_rows": len(x_live),
        "n_features_checked": int(n_feats),
        "psi_threshold": float(psi_threshold),
        "ks_pvalue_threshold": float(ks_pvalue_threshold),
        "ks_statistic_threshold": float(ks_statistic_threshold),
        "psi_max": psi_max,
        "ks_min_pvalue": ks_min_pvalue,
        "ks_max_stat": ks_max_stat,
        "top_features": top_features,
        "reasons": reasons,
    }


def run_drift_gate(
    cache_path: str,
    baseline_samples: int,
    live_samples: int,
    psi_threshold: float,
    ks_pvalue_threshold: float,
    ks_statistic_threshold: float = _DEFAULT_KS_STATISTIC_THRESHOLD,
) -> dict[str, Any]:
    """Run drift check and return report dict."""
    return compute_drift_report(
        cache_path=cache_path,
        baseline_samples=baseline_samples,
        live_samples=live_samples,
        psi_threshold=psi_threshold,
        ks_pvalue_threshold=ks_pvalue_threshold,
        ks_statistic_threshold=ks_statistic_threshold,
    )
