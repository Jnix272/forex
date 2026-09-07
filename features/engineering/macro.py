"""
features/engineering/macro.py
Macro / yield curve / carry / correlation breakdown features.
"""

import polars as pl


def carry_features(spot: str = "close", forward: str = "forward_price", window: int = 30) -> pl.Expr:
    """Spot-forward carry as log ratio of forward over spot, smoothed.
    Returns a rolling mean of the carry.
    """
    carry = (pl.col(forward) / pl.col(spot)).log()
    return carry.rolling_mean(window).alias("carry_spot_forward")


def yield_curve_slope(short_yield: str = "US2Y", long_yield: str = "US10Y") -> pl.Expr:
    """Simple slope = long-term yield minus short-term yield."""
    return (pl.col(long_yield) - pl.col(short_yield)).alias("yield_curve_slope")


def correlation_breakdown(a: str, b: str, window: int = 60) -> pl.Expr:
    """Rolling correlation delta between two series.
    Returns the absolute change of correlation over the window.
    """
    corr = pl.rolling_corr(pl.col(a), pl.col(b), window_size=window)
    delta = (corr - corr.rolling_mean(window)).abs()
    return delta.alias(f"corr_break_{a}_{b}")
