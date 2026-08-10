"""
Data Drift Detection Checks.

PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) tests for feature drift.
"""

import numpy as np
import torch
from typing import Any

from monitoring.checks import CheckContext, CheckResult, CheckStatus, register_check


def psi_score(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index."""
    # Use quantile-based binning on expected
    try:
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.quantile(expected, quantiles)
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
    except Exception:
        return 0.0
    
    # Histogram both distributions
    expected_hist, _ = np.histogram(expected, bins=bin_edges, density=True)
    actual_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    expected_hist = expected_hist + eps
    actual_hist = actual_hist + eps
    
    # Normalize
    expected_hist = expected_hist / expected_hist.sum()
    actual_hist = actual_hist / actual_hist.sum()
    
    # PSI = sum((actual - expected) * log(actual / expected))
    psi = np.sum((actual_hist - expected_hist) * np.log(actual_hist / expected_hist))
    return float(psi)


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Calculate KS statistic and p-value approximation."""
    from scipy import stats
    try:
        stat, p_value = stats.ks_2samp(expected, actual)
        return float(stat), float(p_value)
    except Exception:
        return 0.0, 1.0


def check_feature_drift_psi(context: CheckContext) -> CheckResult:
    """Check feature drift using PSI."""
    config = context.config
    psi_threshold = config.get("psi_threshold", 0.2)
    psi_critical = config.get("psi_critical", 0.3)
    n_features = config.get("drift_n_features", 10)  # Max features to check
    
    if context.batch_data is None:
        return CheckResult(
            name="feature_drift_psi",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No batch data",
        )
    
    data = context.batch_data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # Get reference distribution from context (should be set during preflight)
    reference = context.extra.get("drift_reference")
    if reference is None:
        return CheckResult(
            name="feature_drift_psi",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No reference distribution for PSI",
        )
    
    # Flatten if needed
    if data.ndim == 3:
        # (B, T, F) -> (B*T, F)
        data = data.reshape(-1, data.shape[-1])
    if reference.ndim == 3:
        reference = reference.reshape(-1, reference.shape[-1])
    
    n_feat = min(data.shape[-1], reference.shape[-1], n_features)
    psi_scores = []
    critical_features = []
    warning_features = []
    
    for i in range(n_feat):
        try:
            psi = psi_score(reference[:, i], data[:, i])
            psi_scores.append(psi)
            if psi > psi_critical:
                critical_features.append((i, psi))
            elif psi > psi_threshold:
                warning_features.append((i, psi))
        except Exception:
            psi_scores.append(0.0)
    
    max_psi = max(psi_scores) if psi_scores else 0.0
    mean_psi = np.mean(psi_scores) if psi_scores else 0.0
    
    if critical_features:
        return CheckResult(
            name="feature_drift_psi",
            status=CheckStatus.FAILED,
            passed=False,
            value=max_psi,
            threshold=psi_critical,
            message=f"CRITICAL drift: {len(critical_features)} features PSI > {psi_critical} (max: {max_psi:.4f})",
            details={
                "max_psi": max_psi,
                "mean_psi": mean_psi,
                "critical_features": critical_features,
                "warning_features": warning_features,
                "all_scores": psi_scores,
            },
        )
    elif warning_features:
        return CheckResult(
            name="feature_drift_psi",
            status=CheckStatus.FAILED,
            passed=False,
            value=max_psi,
            threshold=psi_threshold,
            message=f"WARNING drift: {len(warning_features)} features PSI > {psi_threshold} (max: {max_psi:.4f})",
            details={
                "max_psi": max_psi,
                "mean_psi": mean_psi,
                "critical_features": critical_features,
                "warning_features": warning_features,
                "all_scores": psi_scores,
            },
        )
    
    return CheckResult(
        name="feature_drift_psi",
        status=CheckStatus.PASSED,
        passed=True,
        value=max_psi,
        message=f"PSI OK: max={max_psi:.4f}, mean={mean_psi:.4f}",
        details={
            "max_psi": max_psi,
            "mean_psi": mean_psi,
            "all_scores": psi_scores,
        },
    )


def check_feature_drift_ks(context: CheckContext) -> CheckResult:
    """Check feature drift using KS test."""
    config = context.config
    ks_threshold = config.get("ks_threshold", 0.05)
    ks_effect_size = config.get("ks_effect_size", 0.05)
    n_features = config.get("drift_n_features", 10)
    
    if context.batch_data is None:
        return CheckResult(
            name="feature_drift_ks",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No batch data",
        )
    
    data = context.batch_data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    reference = context.extra.get("drift_reference")
    if reference is None:
        return CheckResult(
            name="feature_drift_ks",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No reference distribution for KS",
        )
    
    if data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
    if reference.ndim == 3:
        reference = reference.reshape(-1, reference.shape[-1])
    
    n_feat = min(data.shape[-1], reference.shape[-1], n_features)
    ks_results = []
    critical_features = []
    warning_features = []
    
    for i in range(n_feat):
        try:
            stat, p_val = ks_statistic(reference[:, i], data[:, i])
            ks_results.append((i, stat, p_val))
            if p_val < ks_threshold and stat > ks_effect_size:
                critical_features.append((i, stat, p_val))
            elif p_val < ks_threshold:
                warning_features.append((i, stat, p_val))
        except Exception:
            ks_results.append((i, 0.0, 1.0))
    
    if critical_features:
        max_stat = max(r[1] for r in ks_results)
        return CheckResult(
            name="feature_drift_ks",
            status=CheckStatus.FAILED,
            passed=False,
            value=max_stat,
            threshold=ks_threshold,
            message=f"CRITICAL KS drift: {len(critical_features)} features p<{ks_threshold} & D>{ks_effect_size}",
            details={
                "critical_features": critical_features,
                "warning_features": warning_features,
                "all_results": ks_results,
            },
        )
    elif warning_features:
        max_stat = max(r[1] for r in ks_results)
        return CheckResult(
            name="feature_drift_ks",
            status=CheckStatus.FAILED,
            passed=False,
            value=max_stat,
            threshold=ks_threshold,
            message=f"WARNING KS drift: {len(warning_features)} features p<{ks_threshold}",
            details={
                "critical_features": critical_features,
                "warning_features": warning_features,
                "all_results": ks_results,
            },
        )
    
    max_stat = max(r[1] for r in ks_results) if ks_results else 0.0
    return CheckResult(
        name="feature_drift_ks",
        status=CheckStatus.PASSED,
        passed=True,
        value=max_stat,
        message=f"KS OK: max D={max_stat:.4f}",
        details={"all_results": ks_results},
    )


def check_label_drift(context: CheckContext) -> CheckResult:
    """Check label/target distribution drift."""
    config = context.config
    psi_threshold = config.get("label_psi_threshold", 0.2)
    
    if context.batch_targets is None:
        return CheckResult(
            name="label_drift",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No batch targets",
        )
    
    targets = context.batch_targets
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    reference = context.extra.get("label_reference")
    if reference is None:
        return CheckResult(
            name="label_drift",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No label reference",
        )
    
    # For classification, compare class distributions
    unique_ref = np.unique(reference)
    unique_curr = np.unique(targets)
    all_classes = np.union1d(unique_ref, unique_curr)
    
    ref_dist = np.array([np.mean(reference == c) for c in all_classes])
    curr_dist = np.array([np.mean(targets == c) for c in all_classes])
    
    # PSI on class distributions
    eps = 1e-10
    ref_dist = ref_dist + eps
    curr_dist = curr_dist + eps
    ref_dist = ref_dist / ref_dist.sum()
    curr_dist = curr_dist / curr_dist.sum()
    
    psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
    
    passed = psi <= psi_threshold
    
    return CheckResult(
        name="label_drift",
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        passed=passed,
        value=psi,
        threshold=psi_threshold,
        message=f"Label PSI: {psi:.4f}" + (" (DRIFT)" if psi > psi_threshold else " OK"),
        details={
            "psi": psi,
            "reference_dist": ref_dist.tolist(),
            "current_dist": curr_dist.tolist(),
            "classes": all_classes.tolist(),
        },
    )


# Register drift checks
from monitoring.checks import register_check

register_check(
    name="feature_drift_psi",
    phase="batch",
    func=check_feature_drift_psi,
    description="Feature drift detection using Population Stability Index",
    severity="warning",
    tags={"drift", "psi", "feature", "distribution"},
    threshold={
        "psi_threshold": 0.2,
        "psi_critical": 0.3,
        "drift_n_features": 10,
    },
)

register_check(
    name="feature_drift_ks",
    phase="batch",
    func=check_feature_drift_ks,
    description="Feature drift detection using KS test",
    severity="warning",
    tags={"drift", "ks", "feature", "distribution"},
    threshold={
        "ks_threshold": 0.05,
        "ks_effect_size": 0.05,
        "drift_n_features": 10,
    },
)

register_check(
    name="label_drift",
    phase="batch",
    func=check_label_drift,
    description="Label/target distribution drift detection",
    severity="warning",
    tags={"drift", "label", "classification"},
    threshold={
        "label_psi_threshold": 0.2,
    },
)