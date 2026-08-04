"""
Tests for cross-asset factor model (Improvement #2):
PCA/ICA factors, Granger causality, lead-lag network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.stattools import grangercausalitytests

from features.cross_asset_factors import (
    build_cross_asset_factors,
    granger_f_test,
    granger_lead_scores,
    lead_lag_network,
    rolling_factor_scores,
)


def _common_factor_panel(n=600, seed=7, n_assets=6):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(0, 1, n))
    d = {f"A{k}": (0.5 + 0.3 * k) * f + rng.normal(0, 1, n) for k in range(n_assets)}
    return pd.DataFrame(d)


def _lead_lag_panel(n=600, seed=7, lag=2):
    """Panel where A leads B by ``lag`` bars; C is noise."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, n)
    B = np.zeros(n)
    src = np.concatenate([[0.0] * lag, A[:-lag]])
    B = 0.8 * src + rng.normal(0, 0.5, n)
    return pd.DataFrame({"A": A, "B": B, "C": rng.normal(0, 1, n)})


# ---------------------------------------------------------------------------
# Granger causality (vs statsmodels reference)
# ---------------------------------------------------------------------------

def test_granger_f_test_matches_statsmodels_significant():
    rng = np.random.default_rng(0)
    T = 400
    x = rng.normal(size=T)
    y = 0.8 * np.concatenate([[0.0], x[:-1]]) + rng.normal(size=T)
    manual = granger_f_test(y, x, maxlag=1)
    sm = grangercausalitytests(pd.DataFrame({"y": y, "x": x}), 1, verbose=False)[1][0]["ssr_ftest"][1]
    assert manual < 0.05
    assert abs(manual - float(sm)) < 0.02


def test_granger_f_test_matches_statsmodels_insignificant():
    rng = np.random.default_rng(1)
    T = 400
    x = rng.normal(size=T)
    y = rng.normal(size=T)  # independent
    manual = granger_f_test(y, x, maxlag=1)
    sm = grangercausalitytests(pd.DataFrame({"y": y, "x": x}), 1, verbose=False)[1][0]["ssr_ftest"][1]
    assert manual > 0.05
    assert abs(manual - float(sm)) < 0.02


def test_granger_reverse_direction_insignificant():
    rng = np.random.default_rng(2)
    T = 400
    x = rng.normal(size=T)
    y = 0.8 * np.concatenate([[0.0], x[:-1]]) + rng.normal(size=T)
    assert granger_f_test(x, y, maxlag=1) > 0.05


def test_granger_f_test_constant_inputs_return_one():
    y = np.ones(200)
    x = np.random.default_rng(3).normal(size=200)
    assert granger_f_test(y, x, 1) == 1.0
    assert granger_f_test(x, y, 1) == 1.0


# ---------------------------------------------------------------------------
# Rolling factor scores (PCA)
# ---------------------------------------------------------------------------

def test_pca_first_factor_dominates_common_factor_panel():
    F = rolling_factor_scores(_common_factor_panel(), n_factors=3, method="pca",
                              window=120, step=10)
    tail = F.tail(100)
    assert tail["factor_1_vev"].mean() > 0.9
    assert tail["factor_1_vev"].mean() > tail["factor_2_vev"].mean()
    assert tail["factor_2_vev"].mean() > tail["factor_3_vev"].mean()


def test_pca_output_shape_and_alignment():
    rets = _common_factor_panel()
    F = rolling_factor_scores(rets, n_factors=3, window=120, step=10)
    assert len(F) == len(rets)
    assert {"factor_1_score", "factor_2_score", "factor_3_score",
            "factor_1_vev", "factor_total_vev"}.issubset(F.columns)
    assert "factor_load_1_A0" in F.columns
    assert F.tail(100).isna().sum().sum() == 0
    # leading rows before first full window are zero
    assert (F["factor_1_score"].iloc[:50] == 0).all()


def test_pca_ica_both_run():
    rets = _common_factor_panel()
    for method in ("pca", "ica"):
        F = rolling_factor_scores(rets, n_factors=2, method=method, window=120, step=10)
        assert {"factor_1_score", "factor_2_score"}.issubset(F.columns)
        assert F["factor_1_score"].tail(10).notna().all()


def test_pca_single_asset_returns_zeros():
    rets = pd.DataFrame({"A": np.random.default_rng(4).normal(size=300)})
    F = rolling_factor_scores(rets, n_factors=3, window=100, step=20)
    assert (F["factor_1_score"].tail(100).abs() == 0).all()


# ---------------------------------------------------------------------------
# Lead-lag network
# ---------------------------------------------------------------------------

def test_leadlag_detects_known_lag():
    LL = lead_lag_network(_lead_lag_panel(lag=2), max_lag=5, window=120, step=10,
                          min_abs_corr=0.05)
    tail = LL.tail(50)
    assert tail["leadlag_lead_lag_B"].max() == 2.0
    assert tail["leadlag_lead_corr_B"].max() > 0.5
    assert tail["leadlag_indegree_B"].max() >= 1
    assert tail["leadlag_density"].max() > 0.0


def test_leadlag_output_shape():
    rets = _lead_lag_panel()
    LL = lead_lag_network(rets, max_lag=3, window=120, step=10)
    assert len(LL) == len(rets)
    assert {"leadlag_lead_corr_A", "leadlag_lead_lag_A", "leadlag_outdegree_A",
            "leadlag_indegree_B", "leadlag_density"}.issubset(LL.columns)
    assert LL.tail(100).isna().sum().sum() == 0


def test_leadlag_single_asset():
    rets = pd.DataFrame({"A": np.random.default_rng(5).normal(size=300)})
    LL = lead_lag_network(rets, window=100, step=20)
    assert (LL["leadlag_density"] == 0).all()


# ---------------------------------------------------------------------------
# Granger lead scores (rolling)
# ---------------------------------------------------------------------------

def test_granger_scores_detect_best_predictor():
    rets = _lead_lag_panel(lag=2)
    G = granger_lead_scores(rets, maxlag=2, window=120, step=10)
    tail = G.tail(50)
    assert (tail["granger_lead_B"] == "A").all()
    assert (tail["granger_p_B"] < 0.05).all()
    assert (tail["granger_score_B"] > 1.0).all()
    # reverse direction should not pick B as leading A
    assert (tail["granger_p_A"] > 0.05).all()


def test_granger_scores_shape_and_no_nan():
    rets = _lead_lag_panel()
    G = granger_lead_scores(rets, maxlag=1, window=120, step=10)
    assert len(G) == len(rets)
    assert {"granger_lead_A", "granger_lead_B", "granger_p_A", "granger_score_A"}.issubset(G.columns)
    assert G["granger_p_A"].tail(100).notna().all()
    assert G["granger_lead_A"].tail(100).notna().all()


def test_granger_scores_single_asset():
    rets = pd.DataFrame({"A": np.random.default_rng(6).normal(size=300)})
    G = granger_lead_scores(rets, window=100, step=20)
    assert (G["granger_p_A"] == 1.0).all()
    assert (G["granger_score_A"] == 0.0).all()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_build_cross_asset_factors_all_columns():
    rets = pd.concat([_common_factor_panel(), _lead_lag_panel().drop(columns=["A", "C"])], axis=1)
    R = build_cross_asset_factors(rets, factor_step=10, granger_step=10, leadlag_step=10)
    assert isinstance(R, pl.DataFrame)
    assert len(R) == len(rets)
    assert "factor_1_score" in R.columns
    assert "granger_lead_B" in R.columns
    assert "leadlag_lead_lag_B" in R.columns
    assert R.tail(100).to_pandas().isna().sum().sum() == 0


def test_build_cross_asset_factors_deterministic():
    rets = _common_factor_panel()
    a = build_cross_asset_factors(rets, factor_step=10)
    b = build_cross_asset_factors(rets, factor_step=10)
    num = [c for c in a.columns if a[c].dtype != pl.Utf8]
    assert np.allclose(a.select(num).to_numpy(), b.select(num).to_numpy(), equal_nan=True)
