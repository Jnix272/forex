"""Feature selection via LASSO / Mutual Information / VIF.

Provides pre-training filtering to reduce 584-dim input (185k samples → ~317 samples/feature)
and mitigate overfitting on HAELT/TFT. Used offline before dataset build or as audit.

All functions operate on 2D array (n_samples, n_features) or flattened sequences.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_vif(X: np.ndarray, feature_names: Sequence[str] | None = None, threshold: float = 10.0) -> dict:
    """Variance Inflation Factor per feature. VIF > threshold indicates high multicollinearity."""
    n_feat = X.shape[1]
    vif = np.zeros(n_feat)
    # Use correlation matrix inversion: VIF = diag(inv(corr))
    # For large 584-dim, use iterative regression to avoid singular matrix.
    from sklearn.linear_model import LinearRegression

    for i in range(n_feat):
        y = X[:, i]
        X_rest = np.delete(X, i, axis=1)
        # Quick check: constant feature → high VIF
        if np.std(y) < 1e-12:
            vif[i] = 999.0
            continue
        try:
            reg = LinearRegression()
            reg.fit(X_rest, y)
            r2 = reg.score(X_rest, y)
            r2 = np.clip(r2, 0, 0.999)
            vif[i] = 1.0 / (1.0 - r2 + 1e-9)
        except Exception:
            vif[i] = 999.0
    result = {}
    for i, v in enumerate(vif):
        name = feature_names[i] if feature_names else f"f{i}"
        result[name] = float(v)
    # Dropped = VIF > threshold
    dropped = [k for k, v in result.items() if v > threshold]
    return {"vif": result, "dropped": dropped, "threshold": threshold}


def compute_mi_scores(X: np.ndarray, y: np.ndarray, feature_names: Sequence[str] | None = None) -> dict:
    """Mutual Information regression scores (non-linear relevance to CPAR target)."""
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not available"}
    # Standardize for MI stability
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Clip y for MI
    y_clean = np.nan_to_num(y, nan=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_regression(Xs, y_clean, random_state=0)
    result = {}
    for i, s in enumerate(mi):
        name = feature_names[i] if feature_names else f"f{i}"
        result[name] = float(s)
    # Rank
    ranked = sorted(result.items(), key=lambda x: x[1], reverse=True)
    return {"mi": result, "ranked": ranked}


def lasso_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str] | None = None,
    cv: int = 5,
    max_features: int | None = None,
) -> dict:
    """LASSO CV selection. Returns mask of non-zero coefficients."""
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not available"}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    y_clean = np.nan_to_num(y, nan=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = LassoCV(cv=cv, random_state=0, max_iter=5000, n_jobs=-1).fit(Xs, y_clean)
    coef = np.abs(lasso.coef_)
    # Non-zero = selected
    selected = coef > 1e-6
    result = {}
    for i, c in enumerate(coef):
        name = feature_names[i] if feature_names else f"f{i}"
        result[name] = float(c)
    ranked = sorted(result.items(), key=lambda x: x[1], reverse=True)
    selected_names = [k for k, v in result.items() if v > 1e-6]
    if max_features and len(selected_names) > max_features:
        # Keep top max_features by coef magnitude
        top = [k for k, _ in ranked[:max_features]]
        selected_names = top
    return {
        "coef": result,
        "ranked": ranked,
        "selected": selected_names,
        "dropped": [k for k in result if k not in selected_names],
        "alpha": float(lasso.alpha_),
    }


def audit_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    vif_threshold: float = 10.0,
    lasso_max_features: int | None = 350,
) -> dict:
    """Full audit: VIF + MI + LASSO, returns recommended drop list (union)."""
    print(f"[FeatureSelect] Auditing {X.shape[1]} features on {X.shape[0]} samples...")
    vif_res = compute_vif(X, feature_names, threshold=vif_threshold)
    mi_res = compute_mi_scores(X, y, feature_names)
    lasso_res = lasso_select(X, y, feature_names, max_features=lasso_max_features)

    # Union of dropped: high VIF AND low MI (bottom 25%) AND LASSO zero
    mi_ranked = mi_res.get("ranked", [])
    n_mi = len(mi_ranked)
    low_mi = set(k for k, _ in mi_ranked[int(n_mi * 0.75) :]) if n_mi else set()
    vif_dropped = set(vif_res.get("dropped", []))
    lasso_dropped = set(lasso_res.get("dropped", []))

    # Conservative: drop only if VIF>threshold AND (low MI OR LASSO zero)
    # Prevents dropping high-VIF but high-importance features (e.g. atr_6 vs atr_20)
    consensus_drop = (vif_dropped & low_mi) | (vif_dropped & lasso_dropped) | (low_mi & lasso_dropped)
    # Also drop if all three agree
    strict_drop = vif_dropped & low_mi & lasso_dropped

    print(f"[FeatureSelect] VIF>{vif_threshold}: {len(vif_dropped)} | LASSO dropped: {len(lasso_dropped)} | low-MI: {len(low_mi)}")
    print(f"[FeatureSelect] Consensus drop (2/3 agree): {len(consensus_drop)} | strict (3/3): {len(strict_drop)}")
    if consensus_drop:
        print(f"  Sample drops: {sorted(list(consensus_drop))[:10]}")
    return {
        "vif": vif_res,
        "mi": mi_res,
        "lasso": lasso_res,
        "consensus_drop": sorted(consensus_drop),
        "strict_drop": sorted(strict_drop),
        "recommended_keep": [f for f in feature_names if f not in consensus_drop],
    }


if __name__ == "__main__":
    # Smoke test with synthetic data mimicking 584 features
    n, f = 2000, 20
    X = np.random.randn(n, f)
    # Make f0 and f1 collinear (high VIF)
    X[:, 1] = X[:, 0] * 0.95 + np.random.randn(n) * 0.05
    y = X[:, 0] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n) * 0.1
    names = [f"feat_{i}" for i in range(f)]
    res = audit_features(X, y, names, vif_threshold=5.0)
    print("Recommended keep:", len(res["recommended_keep"]))
