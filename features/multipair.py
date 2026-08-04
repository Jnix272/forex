"""
Multi-Pair Interaction Features (Improvement #6)
================================================
Cross-pair features B7–B11 and VPIN, realized moments, Asia-London gap.
Extracted from advanced_features.py for modular use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

# ════════════════════════════════════════════════════════════════════════════
# VPIN (Volume-Synchronized Probability of Informed Trading)
# ════════════════════════════════════════════════════════════════════════════

def compute_vpin(bars: pl.DataFrame, bucket_size: int = 50, n_buckets: int = 50) -> pl.Series:
    """Compute VPIN from OHLCV bars using buy/sell volume classification."""
    buy_vol = pl.when(pl.col("close") > pl.col("open")).then(pl.col("volume")).otherwise(0.0)
    sell_vol = pl.when(pl.col("close") <= pl.col("open")).then(pl.col("volume")).otherwise(0.0)

    df = bars.with_columns([
        buy_vol.alias("buy_vol"),
        sell_vol.alias("sell_vol")
    ])

    df = df.with_columns([
        pl.col("buy_vol").rolling_sum(window_size=bucket_size).alias("buy_bucket"),
        pl.col("sell_vol").rolling_sum(window_size=bucket_size).alias("sell_bucket"),
        pl.col("volume").rolling_sum(window_size=bucket_size).alias("total_bucket")
    ])

    vpin = (
        (df["buy_bucket"] - df["sell_bucket"]).abs() / (df["total_bucket"] + 1e-9)
    ).rolling_mean(window_size=n_buckets)

    return vpin.fill_null(0.0).fill_nan(0.0).alias("vpin")


# ════════════════════════════════════════════════════════════════════════════
# Realized Moments (Skewness, Kurtosis, Up/Down Vol Ratio)
# ════════════════════════════════════════════════════════════════════════════

def compute_realized_moments(close: pl.Series, window: int = 20) -> pl.DataFrame:
    """Compute realized higher moments and up/down volatility ratio."""
    ret = (close / close.shift(1)).log().alias("ret")

    up_ret = pl.when(ret > 0).then(ret).otherwise(0.0)
    dn_ret = pl.when(ret < 0).then(ret).otherwise(0.0)

    df = pl.DataFrame({"ret": ret, "up_ret": up_ret, "dn_ret": dn_ret})

    df = df.with_columns([
        pl.col("ret").rolling_skew(window_size=window).alias(f"rolling_skew_{window}"),
        pl.col("ret").rolling_mean(window_size=window).alias("mu"),
        (pl.col("ret")**2).rolling_mean(window_size=window).alias("mu2"),
        (pl.col("ret")**3).rolling_mean(window_size=window).alias("mu3"),
        (pl.col("ret")**4).rolling_mean(window_size=window).alias("mu4"),
        pl.col("up_ret").rolling_std(window_size=window).alias("up_std"),
        pl.col("dn_ret").rolling_std(window_size=window).alias("dn_std"),
    ])

    df = df.with_columns([
        (pl.col("mu2") - pl.col("mu")**2).alias("var")
    ])

    df = df.with_columns([
        (
            (pl.col("mu4") - 4*pl.col("mu3")*pl.col("mu") + 6*pl.col("mu2")*(pl.col("mu")**2) - 3*(pl.col("mu")**4))
            / (pl.col("var")**2 + 1e-9)
        ).alias(f"rolling_kurt_{window}"),
        (pl.col("up_std") / (pl.col("dn_std") + 1e-9)).alias("rvol_ratio")
    ])

    res = df.select([
        pl.col(f"rolling_skew_{window}").fill_nan(0.0).fill_null(0.0),
        pl.col(f"rolling_kurt_{window}").fill_nan(0.0).fill_null(0.0),
        pl.col("rvol_ratio").fill_nan(0.0).fill_null(0.0)
    ])
    return res


# ════════════════════════════════════════════════════════════════════════════
# Asia-London Gap
# ════════════════════════════════════════════════════════════════════════════

def compute_asia_london_gap(bars: pl.DataFrame, atr: pl.Series = None) -> pl.Series:
    """Compute Asia session close to London open gap, normalized by ATR."""
    time_col = 'timestamp_utc' if 'timestamp_utc' in bars.columns else 'timestamp' if 'timestamp' in bars.columns else 'datetime'

    df = bars.select([pl.col(time_col), pl.col('close')])
    df = df.with_columns([
        pl.col(time_col).dt.date().alias('date'),
        pl.col(time_col).dt.time().alias('time')
    ])

    df = df.with_columns([
        (pl.col('time') < pl.time(7, 0)).alias('is_asia'),
        (pl.col('time') >= pl.time(7, 0)).alias('is_london')
    ])

    london_open_times = (
        df.filter(pl.col('is_london'))
          .group_by('date')
          .agg(pl.col(time_col).first().alias('london_open_time'))
    )

    asia_close_vals = (
        df.filter(pl.col('is_asia'))
          .group_by('date')
          .agg(pl.col('close').last().alias('asia_close'))
    )

    daily_gaps = london_open_times.join(asia_close_vals, on='date', how='inner')

    df = df.join(daily_gaps, left_on=time_col, right_on='london_open_time', how='left')

    df = df.with_columns([
        (pl.col('close') - pl.col('asia_close')).alias('gap')
    ])

    gap_series = df['gap'].forward_fill()

    if atr is not None:
        gap_series = gap_series / (atr + 1e-9)

    return gap_series.fill_null(0.0).fill_nan(0.0).alias('asia_london_gap')


# ════════════════════════════════════════════════════════════════════════════
# Multi-Pair Features (B7–B11)
# ════════════════════════════════════════════════════════════════════════════

def compute_multipair_features(
    pair_bars: dict[str, pl.DataFrame | pd.DataFrame],
    momentum_window: int = 20,
    atr_col: str = "atr_6",
    dispersion_window: int = 5,
    atr_window: int = 6,
    _ofi_z_fast: int = 20,
    _ofi_z_slow: int = 120,
    _tbm_default_horizon: int = 10,
) -> pl.DataFrame:
    """
    Compute cross-pair features B7–B11 from a dict of {pair: OHLCV DataFrame}.

    Returns a DataFrame indexed to the primary pair with columns:
      B7  rel_mom_{i}_{j}     : r_i(20) - r_j(20) for economically linked pairs
      B8  vol_share_{pair}    : ATR_i / sum(ATR basket)
      B9  cross_dispersion    : StdDev of 5-bar returns across all pairs
      B10 time_to_barrier_est : ATR_20 / |ΔP_5| proxy for momentum vs noise
      B11 no_trade_score      : 1 if low-vol + neutral OFI + choppy trend

    All features respect temporal causality: only past data is used.
    """
    if not pair_bars:
        return pl.DataFrame()

    pairs = list(pair_bars.keys())
    primary = pairs[0]

    # Convert primary pair first so we can safely use .index
    primary_bars = pair_bars[primary]
    if isinstance(primary_bars, pl.DataFrame):
        b_pd = primary_bars.to_pandas()
        if "timestamp_utc" in b_pd.columns:
            b_pd.set_index("timestamp_utc", inplace=True)
        pair_bars[primary] = b_pd
        primary_bars = b_pd
    idx = primary_bars.index

    # Compute log returns and ATR for each pair, aligned to primary index
    returns = {}
    atrs    = {}
    for pair, bars in pair_bars.items():
        if isinstance(bars, pl.DataFrame):
            b_pd = bars.to_pandas()
            if "timestamp_utc" in b_pd.columns:
                b_pd.set_index("timestamp_utc", inplace=True)
            bars = b_pd

        bars_aligned = bars.reindex(idx, method="ffill")
        ret = np.log(bars_aligned["close"] / bars_aligned["close"].shift(1))
        returns[pair] = ret.fillna(0.0)
        # ATR proxy: rolling true-range mean
        prev_c = bars_aligned["close"].shift(1)
        tr = pd.concat([
            bars_aligned["high"] - bars_aligned["low"],
            (bars_aligned["high"] - prev_c).abs(),
            (bars_aligned["low"] - prev_c).abs(),
        ], axis=1).max(axis=1)
        atrs[pair] = tr.rolling(atr_window, min_periods=2).mean().fillna(1e-6)

    F = pd.DataFrame(index=idx)

    # B7. Relative momentum: r_i(window) - r_j(window) for all i<j pairs
    cum_rets = {p: returns[p].rolling(momentum_window, min_periods=2).sum() for p in pairs}
    for i, pi in enumerate(pairs):
        for pj in pairs[i + 1:]:
            col = f"rel_mom_{pi}_{pj}"
            F[col] = (cum_rets[pi] - cum_rets[pj]).fillna(0.0)

    # B8. Volatility dominance: ATR_i / basket_ATR
    atr_basket = sum(atrs[p] for p in pairs) + 1e-9
    for pair in pairs:
        F[f"vol_share_{pair}"] = (atrs[pair] / atr_basket).fillna(0.0)

    # B9. Cross-pair dispersion: StdDev of short-window returns across pairs
    ret_matrix = pd.DataFrame({p: returns[p].rolling(dispersion_window, min_periods=2).sum()
                                for p in pairs})
    F["cross_dispersion"] = ret_matrix.std(axis=1).fillna(0.0)

    # B10. Time-to-barrier estimate: ATR_20 / |ΔP_5| — short = momentum, long = drift
    primary_ret5   = returns[primary].rolling(dispersion_window, min_periods=2).sum().abs() + 1e-8
    primary_atr20  = atrs[primary].rolling(20, min_periods=5).mean().fillna(1e-6)
    F["time_to_barrier_est"] = (primary_atr20 / primary_ret5).clip(0.1, 20.0).fillna(5.0)

    # B11. No-trade zone score
    # Conditions: low vol + neutral OFI-Z + choppy trend
    # Vol condition: rolling ATR below 25th percentile
    atr_pct25 = atrs[primary].rolling(200, min_periods=50).quantile(0.25)
    low_vol   = (atrs[primary] < atr_pct25).astype(float)

    # OFI-Z neutral band: approximate with cross-pair dispersion being very low
    neutral_ofi = (F["cross_dispersion"] < F["cross_dispersion"].rolling(200, min_periods=50).quantile(0.3)).astype(float)

    # Trend stability: low dispersion of returns across window -> choppy
    trend_unstable = (F["cross_dispersion"] < 1e-5).astype(float)

    F["no_trade_score"] = ((low_vol + neutral_ofi + trend_unstable) / 3.0).clip(0.0, 1.0)

    F = F.ffill().bfill().fillna(0.0)
    return pl.from_pandas(F.reset_index(drop=True))


# ════════════════════════════════════════════════════════════════════════════
# Convenience: Build all multi-pair features in one call
# ════════════════════════════════════════════════════════════════════════════

def build_multipair_features(
    pair_bars: dict[str, pl.DataFrame | pd.DataFrame],
    **kwargs
) -> pl.DataFrame:
    """
    Compute all multi-pair features (B7-B11) + VPIN + realized moments + Asia-London gap.
    """
    # B7-B11
    mp = compute_multipair_features(pair_bars, **kwargs)

    # Add VPIN for primary pair
    primary = list(pair_bars.keys())[0]
    primary_bars = pair_bars[primary]
    if isinstance(primary_bars, pl.DataFrame):
        vpin = compute_vpin(primary_bars)
        mp = pl.concat([mp, vpin.to_frame()], how="horizontal_extend")

    # Add realized moments
    if isinstance(primary_bars, pl.DataFrame):
        rm = compute_realized_moments(primary_bars["close"])
        mp = pl.concat([mp, rm], how="horizontal_extend")

    # Add Asia-London gap (needs ATR)
    atr_col = kwargs.get("atr_col", "atr_6")
    if atr_col in primary_bars.columns:
        atr = primary_bars[atr_col]
        gap = compute_asia_london_gap(primary_bars, atr)
        mp = pl.concat([mp, gap.to_frame()], how="horizontal_extend")

    return mp


if __name__ == "__main__":
    # Quick smoke test
    print("Multi-pair features module loaded")
    print("Available functions:")
    print("  - compute_vpin")
    print("  - compute_realized_moments")
    print("  - compute_asia_london_gap")
    print("  - compute_multipair_features")
    print("  - build_multipair_features")
