"""
Cross-Asset Factor Model (Improvement #2)
==========================================
PCA/ICA common-factor model, pairwise Granger causality, and lead-lag networks
over a panel of aligned cross-asset log-returns.

All estimators are causal: each row only uses data up to that timestamp.
Rolling windows are refit every ``step`` bars for speed, then forward-filled.

Outputs (one row per input bar, forward-filled):
  factor_{k}_score            : projection of standardized returns onto factor k
  factor_{k}_vev              : variance explained by factor k (within window)
  factor_total_vev            : cumulative VEV across factors
  factor_load_{k}_{asset}     : loading of asset on factor k (step-refreshed)
  granger_lead_{target}       : asset that best Granger-causes target (name)
  granger_p_{target}          : best (minimum) Granger p-value
  granger_score_{target}      : -log10(p) capped at 5, 0 for no signal
  leadlag_lead_corr_{i}       : max |lagged corr| of another asset leading i
  leadlag_lead_lag_{i}        : lag of that best leading relationship
  leadlag_follow_corr_{i}     : max |lagged corr| for assets i leads
  leadlag_follow_lag_{i}      : lag of that best following relationship
  leadlag_indegree_{i}        : number of assets leading i (network edge)
  leadlag_outdegree_{i}       : number of assets i leads
  leadlag_density             : edge density of the lead-lag network
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════


def _refit_indices(n: int, window: int, step: int) -> list[int]:
    """Refit row indices: rows ``window..n-1`` every ``step`` (incl. last)."""
    if n <= window:
        return []
    idx = list(range(window, n, step))
    last = n - 1
    if last > window and last != idx[-1]:
        idx.append(last)
    return idx


def _stdize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Column-wise z-score; zero-variance columns are zeroed (constant-col safe)."""
    sd = mat.std(axis=0)
    m = mat - mat.mean(axis=0)
    keep = np.argwhere(sd > eps).ravel()
    out = np.zeros_like(mat)
    if keep.size:
        out[:, keep] = m[:, keep] / sd[keep]
    return out


def _ffill_refits(n: int, positions: list[int], rows: list[np.ndarray]) -> np.ndarray:
    """Broadcast refit rows across full length: row ``i`` uses last refit <= i."""
    out = np.zeros((n, rows[0].shape[0]), dtype=float)
    if not positions:
        return out
    idx = 0
    for i in range(n):
        if idx + 1 < len(positions) and i >= positions[idx + 1]:
            idx += 1
        if i >= positions[0]:
            out[i] = rows[idx]
    return out


# ════════════════════════════════════════════════════════════════════════════
# 1. PCA / ICA common-factor model
# ════════════════════════════════════════════════════════════════════════════


def rolling_factor_scores(
    returns: pd.DataFrame,
    n_factors: int = 3,
    method: str = "pca",
    window: int = 120,
    step: int = 20,
) -> pd.DataFrame:
    """Rolling PCA/ICA common-factor scores and loadings from a returns panel.

    ``returns`` : DataFrame of aligned log-returns, index = bar index, columns =
    asset names (may contain NaNs in early rows).

    Returns a DataFrame aligned to ``returns.index`` with columns
    ``factor_<k>_score``, ``factor_<k>_vev``, ``factor_load_<k>_<asset>`` and
    ``factor_total_vev``. Leading rows (before the first full window) are 0.
    """
    assets = list(returns.columns)
    n, p = returns.shape
    n_comp = max(1, min(n_factors, max(p - 1, 1)))
    load_cols = [f"factor_load_{k}_{a}" for k in range(1, n_comp + 1) for a in assets]
    cols = (
        [f"factor_{k}_score" for k in range(1, n_comp + 1)]
        + [f"factor_{k}_vev" for k in range(1, n_comp + 1)]
        + load_cols
        + ["factor_total_vev"]
    )

    if p < 2:
        return pd.DataFrame(np.zeros((n, len(cols))), columns=cols, index=returns.index)

    positions = _refit_indices(n, window, step)
    rows: list[np.ndarray] = []
    for t in positions:
        W = returns.iloc[max(0, t - window + 1) : t + 1].fillna(0.0).to_numpy(dtype=float)
        X = _stdize(W)
        if method == "ica":
            try:
                from sklearn.decomposition import FastICA

                ica = FastICA(n_components=n_comp, max_iter=2000, tol=1e-8, random_state=0)
                scores = ica.fit_transform(X)[:, :n_comp]
                loads = ica.mixing_.T
            except Exception:
                scores, loads = _pca_fit(X, n_comp)
        else:
            scores, loads = _pca_fit(X, n_comp)
        vev = np.array([_shared_var(scores[:, k], X) for k in range(n_comp)])
        row = np.concatenate([scores[-1], vev, loads.reshape(-1), [float(vev.sum())]]).astype(float)
        rows.append(row)

    out = np.zeros((n, len(cols)), dtype=float)
    if rows:
        out = _ffill_refits(n, positions, rows)
    return pd.DataFrame(out, columns=cols, index=returns.index)


def _pca_fit(X: np.ndarray, n_comp: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA via eigendecomposition of the (possibly rank-deficient) covariance."""
    cov = np.cov(X, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    return X @ evecs[:, :n_comp], evecs[:, :n_comp].T


def _shared_var(scores: np.ndarray, X: np.ndarray) -> float:
    """Variance explained: mean squared Pearson corr of factor with each asset."""
    c = np.array([np.corrcoef(scores, X[:, j])[0, 1] for j in range(X.shape[1])])
    c = np.nan_to_num(c)
    return float(np.mean(c**2))


# ════════════════════════════════════════════════════════════════════════════
# 2. Granger causality (manual joint F-test, validated against statsmodels)
# ════════════════════════════════════════════════════════════════════════════


def _ols_rss(y: np.ndarray, X: np.ndarray) -> float:
    """Residual sum of squares of y on design X (intercept appended)."""
    Xd = np.concatenate([np.ones((len(y), 1)), X], axis=1)
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    return float(resid @ resid)


def granger_f_test(y: np.ndarray, x: np.ndarray, maxlag: int = 1) -> float:
    """Joint Granger F-test: does ``x`` (past) help forecast ``y``?

    Restricted:   y_t = c + sum_l phi_l  y_{t-l}
    Unrestricted: y_t = c + sum_l phi_l  y_{t-l} + sum_l psi_l x_{t-l}
    Returns the p-value (small = strong evidence x Granger-causes y).
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    T = len(y)
    p = int(maxlag)
    if 2 * p + 2 >= T or np.std(y) < 1e-12 or np.std(x) < 1e-12:
        return 1.0
    Y = np.stack([y[p - 1 - l : T - 1 - l] for l in range(p)], axis=1)  # y_{t-l}  # noqa: E741
    Xl = np.stack([x[p - 1 - l : T - 1 - l] for l in range(p)], axis=1)  # x_{t-l}  # noqa: E741
    y_obs = y[p:]
    rss_r = _ols_rss(y_obs, Y)
    rss_u = _ols_rss(y_obs, np.concatenate([Y, Xl], axis=1))
    df1 = p
    df2 = T - 2 * p - 1
    f = ((rss_r - rss_u) / df1) / (rss_u / df2 + 1e-15)
    return float(stats.f.sf(max(f, 0.0), df1, df2))


def granger_lead_scores(
    returns: pd.DataFrame,
    maxlag: int = 1,
    window: int = 120,
    step: int = 20,
) -> pd.DataFrame:
    """Rolling pairwise Granger causality: best predictor of each target asset.

    Returns a DataFrame aligned to ``returns.index`` with, per target asset:
    ``granger_lead_<target>`` (best predictor name, "" if none),
    ``granger_p_<target>`` and ``granger_score_<target>`` = -log10(p) capped at 5.
    """
    assets = list(returns.columns)
    n = len(returns)
    n_assets = len(assets)
    lead_cols = [f"granger_lead_{a}" for a in assets]
    p_cols = [f"granger_p_{a}" for a in assets]
    s_cols = [f"granger_score_{a}" for a in assets]
    cols = lead_cols + p_cols + s_cols
    out = pd.DataFrame(np.zeros((n, len(cols))), columns=cols, index=returns.index)

    if n_assets < 2:
        out[p_cols] = 1.0
        return out

    positions = _refit_indices(n, window, step)
    rows: list[tuple[np.ndarray, list[str]]] = []
    for t in positions:
        W = returns.iloc[max(0, t - window + 1) : t + 1].fillna(0.0)
        pvec = np.ones(n_assets)
        names: list[str] = [""] * n_assets
        for i, target in enumerate(assets):
            y = W[target].to_numpy(dtype=float)
            best_p = 1.0
            best_name = ""
            for j, pred in enumerate(assets):
                if i == j:
                    continue
                pval = granger_f_test(y, W[pred].to_numpy(dtype=float), maxlag)
                if pval < best_p:
                    best_p = pval
                    best_name = pred
            pvec[i] = best_p
            names[i] = best_name
        rows.append((pvec, names))

    if rows:
        pmat = _ffill_refits(n, positions, [r[0] for r in rows])
        for i, a in enumerate(assets):
            out[f"granger_p_{a}"] = pmat[:, i]
            sc = -np.log10(np.clip(pmat[:, i], 1e-9, 1.0))
            out[f"granger_score_{a}"] = np.clip(sc, 0.0, 5.0)
        # best-predictor names: string forward-fill from refit rows
        name_pos: list[list[str]] = [[""] * n_assets for _ in range(n)]
        idx = 0
        for i in range(n):
            if idx + 1 < len(positions) and i >= positions[idx + 1]:
                idx += 1
            if i >= positions[0]:
                name_pos[i] = list(rows[idx][1])
        for i, a in enumerate(assets):
            out[f"granger_lead_{a}"] = np.array([name_pos[k][i] for k in range(n)])
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. Lead-lag network (vectorized lagged cross-correlation)
# ════════════════════════════════════════════════════════════════════════════


def lead_lag_network(
    returns: pd.DataFrame,
    max_lag: int = 5,
    window: int = 120,
    step: int = 20,
    min_abs_corr: float = 0.05,
) -> pd.DataFrame:
    """Rolling lead-lag network from lagged cross-correlations.

    For each ordered pair (j, i), cross-correlation corr(x_j[t-l], x_i[t]) for
    l = 1..max_lag captures "asset j leads asset i by l bars". Best per-asset
    incoming/outgoing relationships and network degrees are emitted.

    Returns a DataFrame aligned to ``returns.index``.
    """
    assets = list(returns.columns)
    n = len(returns)
    n_assets = len(assets)
    cols = (
        [f"leadlag_lead_corr_{a}" for a in assets]
        + [f"leadlag_lead_lag_{a}" for a in assets]
        + [f"leadlag_follow_corr_{a}" for a in assets]
        + [f"leadlag_follow_lag_{a}" for a in assets]
        + [f"leadlag_indegree_{a}" for a in assets]
        + [f"leadlag_outdegree_{a}" for a in assets]
        + ["leadlag_density"]
    )
    out = np.zeros((n, len(cols)), dtype=float)
    if n_assets < 2:
        return pd.DataFrame(out, columns=cols, index=returns.index)

    positions = _refit_indices(n, window, step)
    rows = []
    for t in positions:
        rows.append(_refit_leadlag(returns, assets, t, max_lag, window, min_abs_corr))
    if rows:
        out = _ffill_refits(n, positions, rows)
    return pd.DataFrame(out, columns=cols, index=returns.index)


def _refit_leadlag(returns: pd.DataFrame, assets, t, max_lag, window, min_abs_corr) -> np.ndarray:
    """Refit lead-lag network on trailing window ending at row t; returns flat row."""
    n_assets = len(assets)
    W = returns.iloc[max(0, t - window + 1) : t + 1].fillna(0.0).to_numpy(dtype=float)
    X = _stdize(W)
    T = X.shape[0]
    L = int(max_lag)

    in_corr = np.zeros(n_assets)
    in_lag = np.zeros(n_assets, dtype=float)
    out_corr = np.zeros(n_assets)
    out_lag = np.zeros(n_assets, dtype=float)

    for l in range(1, L + 1):  # noqa: E741
        if T - l < 2:
            continue
        corr = (X[: T - l].T @ X[l:]) / (T - l)  # corr[j,i] = j leads i at lag l
        for i in range(n_assets):
            for j in range(n_assets):
                if j == i:
                    continue
                c = corr[j, i]
                if abs(c) > abs(in_corr[i]):
                    in_corr[i] = c
                    in_lag[i] = l
                if abs(c) > abs(out_corr[j]):
                    out_corr[j] = c
                    out_lag[j] = l

    indeg = np.zeros(n_assets)
    outdeg = np.zeros(n_assets)
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                continue
            best = _best_dir_corr(X, i, j, L)
            if abs(best) >= min_abs_corr:
                outdeg[i] += 1
                indeg[j] += 1

    n_possible = n_assets * (n_assets - 1)
    density = (indeg.sum() / n_possible) if n_possible else 0.0

    return np.concatenate(
        [
            np.abs(in_corr),
            in_lag,
            np.abs(out_corr),
            out_lag,
            indeg,
            outdeg,
            [density],
        ]
    ).astype(float)


def _best_dir_corr(X: np.ndarray, i: int, j: int, max_lag: int) -> float:
    """Best (signed) lagged corr that asset i leads asset j across lags."""
    best = 0.0
    T = X.shape[0]
    for l in range(1, max_lag + 1):  # noqa: E741
        if T - l < 2:
            continue
        c = float((X[: T - l, i] @ X[l:, j]) / (T - l))
        if abs(c) > abs(best):
            best = c
    return best


# ════════════════════════════════════════════════════════════════════════════
# 4. Orchestrator
# ════════════════════════════════════════════════════════════════════════════


def build_cross_asset_factors(
    returns: pd.DataFrame,
    n_factors: int = 3,
    method: str = "pca",
    factor_window: int = 120,
    factor_step: int = 20,
    maxlag: int = 1,
    granger_window: int = 120,
    granger_step: int = 20,
    max_lag: int = 5,
    leadlag_window: int = 120,
    leadlag_step: int = 20,
    min_abs_corr: float = 0.05,
) -> pl.DataFrame:
    """Compute all cross-asset factor/granger/lead-lag features at once.

    ``returns`` : DataFrame of aligned log-returns (index = bar index, columns =
    asset names). The primary traded pair may be included as one of the columns.

    Returns a Polars DataFrame aligned to ``returns.index`` containing every
    column described in the module docstring.
    """
    dfs = [
        rolling_factor_scores(returns, n_factors, method, factor_window, factor_step),
        granger_lead_scores(returns, maxlag, granger_window, granger_step),
        lead_lag_network(returns, max_lag, leadlag_window, leadlag_step, min_abs_corr),
    ]
    merged = pd.concat(dfs, axis=1)
    return pl.from_pandas(merged.reset_index(drop=True))
