"""
Feature Quality Monitor (Improvement #4)
========================================
Distribution-drift (PSI), predictive power (IV/WOE), temporal stability, and
target-leakage detection for feature columns.

All metrics are deterministic and causal:
  - PSI       : population stability index of current vs reference distribution
  - IV/WOE    : information value & weight-of-evidence of a feature vs a binary target
  - stability : rolling PSI + Kolmogorov-Smirnov of trailing window vs a baseline
  - leakage   : flag features with implausibly high predictive power on a forward
                target (IV/AUC) or that are a transform of the target itself

Threshold conventions (used for the drift/quality flags):
  PSI  < 0.10 stable | 0.10-0.25 moderate | > 0.25 severe
  IV   < 0.02 useless | 0.02-0.10 weak | 0.10-0.30 medium | 0.30-0.50 strong | > 0.50 suspicious
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl


# ════════════════════════════════════════════════════════════════════════════
# 1. Population Stability Index (PSI)
# ════════════════════════════════════════════════════════════════════════════

def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index of ``current`` vs ``reference`` distributions.

    Buckets are derived from ``reference`` quantiles (with fixed 0th/100th
    percentiles as bin edges); the same bins are applied to ``current``.
    PSI = sum((cur_share - ref_share) * ln(cur_share / ref_share)) over bins.
    Returns 0.0 when either input is empty or effectively constant.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0
    n_bins = max(2, int(n_bins))
    if np.unique(ref).size < 2:
        return 0.0

    edges = np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return 0.0

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_share = ref_hist / ref.size + eps
    cur_share = cur_hist / cur.size + eps
    psi = float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
    return psi


# ════════════════════════════════════════════════════════════════════════════
# 2. Information Value (WOE + IV) vs a binary target
# ════════════════════════════════════════════════════════════════════════════

def woe_iv(
    feature: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
) -> tuple:
    """Weight-of-Evidence and Information Value of ``feature`` for ``target``.

    ``target`` must be binary (0/1). Returns (woe [n_bins], iv, bin_edges).
    With a degenerate target or constant feature, returns empty WOE, iv=0.0.
    """
    f = np.asarray(feature, dtype=float)
    t = np.asarray(target, dtype=float).ravel()
    if f.size != t.size:
        return np.array([]), 0.0, np.array([])
    ok = np.isfinite(f) & np.isfinite(t)
    f, t = f[ok], t[ok]
    if f.size < 20 or np.unique(t).size < 2 or np.unique(f).size < 2:
        return np.array([]), 0.0, np.array([])

    n_bins = max(2, int(n_bins))
    edges = np.quantile(f, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([]), 0.0, edges

    bins = np.digitize(f, edges[1:-1])
    good = np.sum(t == 1)
    bad = np.sum(t == 0)
    woe = np.zeros(len(edges) - 1)
    iv = 0.0
    for b in range(len(edges) - 1):
        mask = bins == b
        g = np.sum(t[mask] == 1)
        bg = np.sum(t[mask] == 0)
        g_share = (g + 0.5) / (good + 0.5)
        b_share = (bg + 0.5) / (bad + 0.5)
        woe[b] = np.log(g_share / b_share)
        iv += (g_share - b_share) * woe[b]
    return woe, float(iv), edges


def information_value(feature: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """Information Value of ``feature`` for a binary ``target``."""
    _, iv, _ = woe_iv(feature, target, n_bins)
    return iv


def _roc_auc(feature: np.ndarray, target: np.ndarray) -> Optional[float]:
    """AUC of a single feature as a classifier of a binary target."""
    from sklearn.metrics import roc_auc_score
    f = np.asarray(feature, dtype=float)
    t = np.asarray(target, dtype=float).ravel()
    ok = np.isfinite(f) & np.isfinite(t)
    f, t = f[ok], t[ok]
    if f.size < 20 or np.unique(t).size < 2 or np.unique(f).size < 2:
        return None
    try:
        return float(roc_auc_score(t, f))
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# 3. Stability: rolling PSI + Kolmogorov-Smirnov vs a baseline window
# ════════════════════════════════════════════════════════════════════════════

def _ffill_steps(n: int, positions: List[int], values: List[float]) -> np.ndarray:
    """Forward-fill refit values: row ``i`` uses the last refit at or before ``i``."""
    out = np.zeros(n)
    if not positions:
        return out
    idx = 0
    for i in range(n):
        if idx + 1 < len(positions) and i >= positions[idx + 1]:
            idx += 1
        if i >= positions[0]:
            out[i] = values[idx]
    return out


def stability_index_series(
    series: np.ndarray,
    window: int = 500,
    step: int = 50,
    n_bins: int = 10,
) -> np.ndarray:
    """Rolling PSI of a trailing ``window`` vs the initial baseline window.

    Returns an array aligned to ``series``; rows before the first complete
    baseline+current pair are 0.
    """
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n < 2 * window:
        return np.zeros(n)
    baseline = s[:window]
    positions = list(range(window, n, step))
    if n - 1 > positions[-1]:
        positions.append(n - 1)
    values = [population_stability_index(baseline, s[max(0, t - window + 1): t + 1], n_bins)
              for t in positions]
    return _ffill_steps(n, positions, values)


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (0..1) between two samples."""
    from scipy import stats
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0
    return float(stats.ks_2samp(ref, cur).statistic)


# ════════════════════════════════════════════════════════════════════════════
# 4. Leakage detection
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class LeakageResult:
    """Per-feature leakage assessment."""
    feature: str
    iv: float
    auc: Optional[float]
    is_target_shift: bool
    near_perfect: bool
    leak_flag: bool


def _is_target_shift(feature: np.ndarray, target: np.ndarray) -> bool:
    f = np.asarray(feature, dtype=float)
    t = np.asarray(target, dtype=float).ravel()
    ok = np.isfinite(f) & np.isfinite(t)
    if np.sum(ok) < 50:
        return False
    f, t = f[ok], t[ok]
    return bool(np.corrcoef(f, t)[0, 1] > 0.999)


def leakage_scan(
    X: pl.DataFrame,
    target: np.ndarray,
    exclude_cols: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    auc_threshold: float = 0.85,
    iv_threshold: float = 0.5,
) -> pl.DataFrame:
    """Scan features for potential target leakage.

    Flags a feature when its IV > ``iv_threshold`` or AUC > ``auc_threshold``
    against the (forward) binary ``target``, or when it is a near-perfect
    transform of the target (|corr| > 0.999).

    Returns a Polars DataFrame: feature, iv, auc, is_target_shift, near_perfect,
    leak_flag.
    """
    exclude = set(exclude_cols or [])
    rows = []
    for col in X.columns:
        if col in exclude or not X[col].dtype.is_numeric():
            continue
        feat = X[col].to_numpy().astype(float)
        iv = information_value(feat, target, n_bins)
        auc = _roc_auc(feat, target)
        shift = _is_target_shift(feat, target)
        near_perfect = (auc is not None and auc > auc_threshold) or iv > iv_threshold
        rows.append({
            "feature": col,
            "iv": iv,
            "auc": auc,
            "is_target_shift": shift,
            "near_perfect": near_perfect,
            "leak_flag": shift or near_perfect,
        })
    out = pl.DataFrame(rows, schema={
        "feature": pl.Utf8, "iv": pl.Float64, "auc": pl.Float64,
        "is_target_shift": pl.Boolean, "near_perfect": pl.Boolean, "leak_flag": pl.Boolean,
    })
    return out.sort("leak_flag", descending=True)


# ════════════════════════════════════════════════════════════════════════════
# 5. Master per-feature quality monitor
# ════════════════════════════════════════════════════════════════════════════

def feature_quality_monitor(
    df: pl.DataFrame,
    target_col: Optional[str] = None,
    reference_df: Optional[pl.DataFrame] = None,
    exclude_cols: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    stability_window: int = 500,
    stability_step: int = 50,
    psi_threshold_moderate: float = 0.10,
    psi_threshold_severe: float = 0.25,
) -> pl.DataFrame:
    """Per-feature quality monitor for a feature DataFrame.

    Combines static quality (from ``features.quality``), distribution drift
    (PSI vs ``reference_df``), temporal stability (rolling PSI + KS), predictive
    power (IV/AUC vs a binary ``target_col``), and a leakage flag.

    Returns a Polars DataFrame with one row per numeric feature:
      feature, dtype, null_pct, std, unique_count, constant, near_constant,
      psi, psi_level, stability (mean rolling PSI), ks (baseline-vs-last),
      iv, auc, leak_flag, quality_flag.

    ``quality_flag`` is True when the feature is clean: no nulls/inf/constant/
    near-constant and no severe drift and no leakage.
    """
    from features.quality import compute_quality_report

    exclude = set(exclude_cols or [])
    exclude.add("timestamp_utc")
    qr = compute_quality_report(df, exclude_cols=list(exclude))
    qmap = {f.name: f for f in qr.features}

    target = None
    if target_col:
        if target_col not in df.columns:
            raise ValueError(f"target_col {target_col!r} not in df")
        target = df[target_col].to_numpy().astype(float)

    n = len(df)

    rows = []
    for fq in qr.features:
        col = fq.name
        s = df[col].drop_nulls().to_numpy().astype(float)
        s = s[np.isfinite(s)]
        if reference_df is not None and col in reference_df.columns:
            ref_s = reference_df[col].drop_nulls().to_numpy().astype(float)
            ref_s = ref_s[np.isfinite(ref_s)]
        else:
            ref_s = s[:stability_window]

        psi = population_stability_index(ref_s, s, n_bins) if len(s) > 0 else 0.0
        if psi < psi_threshold_moderate:
            psi_level = "stable"
        elif psi <= psi_threshold_severe:
            psi_level = "moderate"
        else:
            psi_level = "severe"

        stab = 0.0
        if n >= 2 * stability_window:
            stab = float(np.mean(stability_index_series(s, stability_window, stability_step, n_bins)))
        ks = ks_statistic(ref_s, s) if len(s) > 0 and len(ref_s) > 0 else 0.0

        iv = auc = None
        if target is not None:
            iv = information_value(s, target, n_bins)
            auc = _roc_auc(s, target)
        shift = _is_target_shift(s, target) if target is not None else False

        leak = shift or (auc is not None and auc > 0.85) or (iv is not None and iv > 0.5)
        clean = not (fq.has_nulls or fq.has_inf or fq.is_constant or fq.near_constant)
        clean = clean and psi_level != "severe" and not leak

        rows.append({
            "feature": col,
            "dtype": fq.dtype,
            "null_pct": fq.null_pct,
            "std": fq.std,
            "unique_count": fq.unique_count,
            "constant": fq.is_constant,
            "near_constant": fq.near_constant,
            "psi": psi,
            "psi_level": psi_level,
            "stability": stab,
            "ks": ks,
            "iv": iv,
            "auc": auc,
            "leak_flag": leak,
            "quality_flag": clean,
        })

    out = pl.DataFrame(rows).sort("leak_flag", descending=True)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Convenience: drift alarm helper
# ════════════════════════════════════════════════════════════════════════════

def drift_level(psi: float, threshold_moderate: float = 0.10, threshold_severe: float = 0.25) -> str:
    """Human-readable drift level for a PSI value."""
    if psi < threshold_moderate:
        return "stable"
    if psi <= threshold_severe:
        return "moderate"
    return "severe"


# ════════════════════════════════════════════════════════════════════════════
# 6. Convenience: quality gate for feature selection
# ════════════════════════════════════════════════════════════════════════════

def filter_features(
    df: pl.DataFrame,
    target_col: Optional[str] = None,
    reference_df: Optional[pl.DataFrame] = None,
    exclude_cols: Optional[Sequence[str]] = None,
    drop_leaky: bool = True,
    drop_severe_drift: bool = True,
    **kwargs,
) -> tuple:
    """Quality gate: return (kept_df, report, dropped_columns).

    Drops features flagged for leakage (and optionally severe drift or static
    quality issues), keeping the target and timestamp columns intact.
    """
    report = feature_quality_monitor(
        df, target_col=target_col, reference_df=reference_df,
        exclude_cols=exclude_cols, **kwargs,
    )
    drop = set()
    for r in report.to_dicts():
        if r["feature"] == target_col:
            continue
        if r["constant"] or r["near_constant"]:
            drop.add(r["feature"])
        if drop_leaky and r["leak_flag"]:
            drop.add(r["feature"])
        if drop_severe_drift and r["psi_level"] == "severe":
            drop.add(r["feature"])
    kept = [c for c in df.columns if c not in drop]
    return df.select(kept), report, sorted(drop)
