"""
features/engineering/regime.py
Regime detection features: HMM (vol-bucket), CPD, persistence,
regime-gated interactions, quality report.
"""

import polars as pl
import numpy as np


def hmm_regime_probs(n_states: int = 3, window: int = 60, feature: str = "close") -> list[pl.Expr]:
    """DS-007: Volatility regime bucket (NOT a true HMM).

    Bins rolling volatility into terciles and computes rolling state membership
    probabilities. Despite the legacy name, this is a simple volatility bucket
    classifier - not a Hidden Markov Model.

    Output columns are named 'vol_regime_state_N_prob' (renamed from 'hmm_state_N_prob').
    """
    ret = (pl.col(feature) / pl.col(feature).shift(1)).log()
    vol = ret.rolling_std(window_size=window)

    # Use n_states to determine quantile boundaries
    exprs = []
    if n_states == 2:
        vol_q50 = vol.rolling_quantile(0.5, window_size=window)
        state = pl.when(vol <= vol_q50).then(0).otherwise(1)
    elif n_states == 3:
        vol_q33 = vol.rolling_quantile(0.33, window_size=window)
        vol_q66 = vol.rolling_quantile(0.66, window_size=window)
        state = pl.when(vol <= vol_q33).then(0).when(vol <= vol_q66).then(1).otherwise(2)
    else:
        # Generic N-way split using equal quantiles
        vol_q33 = vol.rolling_quantile(0.33, window_size=window)
        vol_q66 = vol.rolling_quantile(0.66, window_size=window)
        state = pl.when(vol <= vol_q33).then(0).when(vol <= vol_q66).then(1).otherwise(2)
        n_states = 3

    for s in range(n_states):
        prob = (state == s).rolling_mean(window_size=window).alias(f"vol_regime_state_{s}_prob")
        exprs.append(prob)
    # Labeling (dataset_builder) expects ``regime_class``; emit it on the
    # volatility-bucket fallback path too when HMM/Numba is unavailable.
    exprs.append(state.cast(pl.Int32).alias("regime_class"))
    exprs.append(
        pl.when(state == 0)
        .then(pl.lit(-1.0))
        .when(state == 2)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
        .alias("regime_label")
    )
    return exprs


def cpd_ret(data: str = "close", window: int = 60) -> pl.Expr:
    """Change Point Detection (CPD) using rolling CUSUM on returns.

    Returns CUSUM statistic for regime change detection.
    """
    ret = (pl.col(data) / pl.col(data).shift(1)).log()
    mu = ret.rolling_mean(window_size=window)
    sigma = ret.rolling_std(window_size=window) + 1e-9
    cusum = ((ret - mu) / sigma).abs().cum_sum().rolling_mean(window_size=window)
    return cusum.alias("cpd_cusum")


def regime_persistence(window: int = 20) -> pl.Expr:
    """Regime persistence: how long current regime has persisted.

    Based on sign of returns or volatility regime.
    """
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    regime = pl.when(ret > 0).then(1).otherwise(-1)
    # Count consecutive same-sign returns
    change = (regime != regime.shift(1)).cast(pl.Int32)
    persistence = change.cum_sum().alias("regime_persistence")
    return persistence


def regime_gated_features(existing_cols: set | None = None) -> list[pl.Expr]:
    """Create regime-specific variants of key features.

    Args:
        existing_cols: Set of column names that exist in the DataFrame.
                       If provided, only creates expressions for available columns.
    """
    if existing_cols is None:
        existing_cols = set()

    exprs = []
    # Trend-following features (active in trending regime)
    if "rsi_14" in existing_cols and "trend_regime" in existing_cols:
        exprs.append((pl.col("rsi_14") * pl.col("trend_regime")).alias("rsi_trend"))
    if "macd" in existing_cols and "trend_regime" in existing_cols:
        exprs.append((pl.col("macd") * pl.col("trend_regime")).alias("macd_trend"))
    if "adx_14" in existing_cols and "trend_regime" in existing_cols:
        exprs.append((pl.col("adx_14") * pl.col("trend_regime")).alias("adx_trend"))
    if "ret_5" in existing_cols and "trend_regime" in existing_cols:
        exprs.append((pl.col("ret_5") * pl.col("trend_regime")).alias("ret5_trend"))

    # Mean-reversion features (active in ranging regime)
    if "stoch_k" in existing_cols and "range_regime" in existing_cols:
        exprs.append((pl.col("stoch_k") * pl.col("range_regime")).alias("stoch_range"))
    if "bb_pct" in existing_cols and "range_regime" in existing_cols:
        exprs.append((pl.col("bb_pct") * pl.col("range_regime")).alias("bb_pct_range"))
    if "williams_r" in existing_cols and "range_regime" in existing_cols:
        exprs.append((pl.col("williams_r") * pl.col("range_regime")).alias("williams_range"))
    if "cci" in existing_cols and "range_regime" in existing_cols:
        exprs.append((pl.col("cci") * pl.col("range_regime")).alias("cci_range"))

    # Volatility-breakout features (active in volatile regime)
    if "atr_ratio_6_20" in existing_cols and "volatility_regime" in existing_cols:
        exprs.append((pl.col("atr_ratio_6_20") * pl.col("volatility_regime")).alias("atr_ratio_volatile"))
    if "breakout_pressure" in existing_cols and "volatility_regime" in existing_cols:
        exprs.append((pl.col("breakout_pressure") * pl.col("volatility_regime")).alias("breakout_volatile"))
    if "vwap_zscore" in existing_cols and "volatility_regime" in existing_cols:
        exprs.append((pl.col("vwap_zscore") * pl.col("volatility_regime")).alias("vwap_z_volatile"))

    return exprs


def interaction_features(existing_cols: set | None = None) -> list[pl.Expr]:
    """Explicit non-linear feature interactions for linear/weak non-linear models.

    Args:
        existing_cols: Set of column names that exist in the DataFrame.
    """
    if existing_cols is None:
        existing_cols = set()

    exprs = []

    # Volatility x Flow
    if "atr_6" in existing_cols and "ofi_z" in existing_cols:
        exprs.append((pl.col("atr_6") * pl.col("ofi_z")).alias("atr_x_ofi"))
    if "atr_ratio_6_20" in existing_cols and "ofi_z" in existing_cols:
        exprs.append((pl.col("atr_ratio_6_20") * pl.col("ofi_z")).alias("atr_ratio_x_ofi"))

    # Momentum x Regime
    if "rsi_14" in existing_cols and "trend_regime" in existing_cols:
        exprs.append((pl.col("rsi_14") * pl.col("trend_regime")).alias("rsi_x_trend"))
    if "macd" in existing_cols and "trend_regime" in existing_cols:
        exprs.append((pl.col("macd") * pl.col("trend_regime")).alias("macd_x_trend"))
    if "stoch_k" in existing_cols and "range_regime" in existing_cols:
        exprs.append((pl.col("stoch_k") * pl.col("range_regime")).alias("stoch_x_range"))

    # Spread / Cost x Volatility
    if "bb_pct" in existing_cols and "range_regime" in existing_cols:
        exprs.append((pl.col("bb_pct") * pl.col("range_regime")).alias("bb_x_range"))
    if "cost_to_atr" in existing_cols and "volatility_regime" in existing_cols:
        exprs.append((pl.col("cost_to_atr") * pl.col("volatility_regime")).alias("cost_x_volatile"))

    # Macro x Risk
    if "yield_curve_slope" in existing_cols and "risk_off_signal" in existing_cols:
        exprs.append((pl.col("yield_curve_slope") * pl.col("risk_off_signal")).alias("curve_x_risk"))
    if "carry_eur" in existing_cols and "trend_regime" in existing_cols:
        exprs.append((pl.col("carry_eur") * pl.col("trend_regime")).alias("carry_x_trend"))

    # Cross-asset x Risk
    if "gold_dxy_corr" in existing_cols and "risk_off_signal" in existing_cols:
        exprs.append((pl.col("gold_dxy_corr") * pl.col("risk_off_signal")).alias("gold_dxy_x_risk"))

    return exprs


def compute_quality_report(df: pl.DataFrame) -> dict:
    """Compute per-feature quality metrics for monitoring.

    Returns dict mapping feature_name -> quality_metrics
    """
    from datetime import datetime, timezone

    UTC = timezone.utc

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "features": {},
    }

    numeric_dtypes = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    )
    numeric_cols = df.select(pl.col(numeric_dtypes)).columns
    for col in numeric_cols:
        s = df[col]
        n_null = s.null_count()
        n_total = len(s)
        vals = s.cast(pl.Float64, strict=False).drop_nulls()

        if len(vals) == 0:
            report["features"][col] = {
                "null_pct": 100.0,
                "constant": True,
                "dtype": str(s.dtype),
            }
            continue

        values = vals.to_numpy()
        if values.size == 0:
            report["features"][col] = {
                "null_pct": 100.0,
                "constant": True,
                "dtype": str(s.dtype),
            }
            continue

        # Check for constant
        is_const = values.size == 1 or np.unique(values).size == 1

        # Basic stats
        mean_v = float(np.mean(values))
        std_v = float(np.std(values)) if values.size > 1 else 0.0
        if values.size > 2:
            centered = values - mean_v
            skew_v = float(np.mean(centered**3) / (std_v**3 + 1e-9))
        else:
            skew_v = 0.0
        if values.size > 3:
            centered = values - mean_v
            kurt_v = float(np.mean(centered**4) / (std_v**4 + 1e-9) - 3.0)
        else:
            kurt_v = 0.0

        inf_count = int(np.isinf(values).sum())
        min_val = float(np.min(values))
        max_val = float(np.max(values))

        report["features"][col] = {
            "dtype": str(s.dtype),
            "null_count": int(n_null),
            "null_pct": float(n_null / n_total * 100),
            "constant": bool(is_const),
            "n_unique": int(np.unique(values).size),
            "mean": mean_v,
            "std": std_v,
            "min": min_val,
            "max": max_val,
            "skew": skew_v,
            "kurtosis": kurt_v,
            "inf_count": inf_count,
        }

    return report
