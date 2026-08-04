# config/feature_mask.py
"""
Domain-knowledge-driven feature mask.

``True``  → keep the column in the training matrix.
``False`` → drop it in ``apply_feature_mask`` (dataset_builder).
Columns absent from this dict (e.g. ``mid_close``) are kept so market
contracts for RL/backtest stay intact.

Pruned defaults disable collinear / sparse / offline-useless columns so the
mask is not a no-op placeholder.
"""

from __future__ import annotations

from typing import Any

FEATURE_MASK = {
    # Core microstructure / momentum – keep high-value ones
    "ofi": True,
    "ofi_z": True,
    "ofi_l2": True,
    "obi_proxy": True,
    "tar": True,
    "kyles_lambda": True,
    "amihud_illiq": True,
    "realized_spread": True,
    "vpin": True,

    # Volatility / ATR
    "atr_6": True,
    "atr_20": True,
    "atr_60": True,
    "atr_ratio_6_20": True,
    "atr_ratio_20_60": True,
    "vol_6": True,
    "vol_20": True,
    "vol_60": True,
    "vol_ratio_6_20": True,
    "vol_ratio_20_60": True,
    "vol_of_vol": True,

    # Bollinger Bands — bb_mid is near-collinear with price; keep width/pct
    "bb_mid": False,
    "bb_upper": True,
    "bb_lower": True,
    "bb_width": True,
    "bb_pct": True,

    # Momentum & Returns
    "rsi_14": True,
    "macd": True,
    "macd_sig": True,
    "macd_hist": True,
    "ret_5": True,
    "ret_20": True,
    "ret_60": True,

    # VWAP / Breakout & Spread
    "breakout_pressure": True,
    "liquidity_vacuum": True,
    "price_ofi_div": True,
    "spread_pips": True,
    "spread_zscore": True,
    "spread_percentile": True,
    "spread_widening_5m": True,
    "spread_widening_20m": True,
    "cost_to_atr": True,

    # Market Regime / Tradeability
    "adx_14": True,
    "chop_index": True,
    "trend_regime": True,
    "range_regime": True,
    "realized_vol_regime": True,
    "trend_quality": True,
    "volatility_regime": True,
    "hurst_exponent": True,
    "noise_to_signal_60": True,
    "trailing_volatility_60": True,

    # Higher-timeframe context
    "ret_5m": True,
    "rsi_5m": True,
    "atr_5m": True,
    "trend_slope_5m": True,
    "distance_to_vwap_5m": True,
    "volatility_regime_5m": True,
    "ret_15m": True,
    "rsi_15m": True,
    "atr_15m": True,
    "trend_slope_15m": True,
    "distance_to_vwap_15m": True,
    "volatility_regime_15m": True,
    "ret_1h": True,
    "rsi_1h": True,
    "atr_1h": True,
    "trend_slope_1h": True,
    "distance_to_vwap_1h": True,
    "volatility_regime_1h": True,

    # Temporal & Latency — offline training has no meaningful latency signal
    "time_sin": True,
    "time_cos": True,
    "day_sin": True,
    "day_cos": True,
    "london_ny": True,
    "expected_latency_ms": False,

    # Advanced groups — sparse / optional exotic cross-breaks off by default
    "carry_spot_forward": False,
    "yield_curve_slope": True,
    "corr_break_gold_DXY": False,

    # Sentiment tiers
    "sentiment_raw": True,
    "sentiment_decayed": True,

    # FinBERT projection (8 dims = SENTIMENT["finbert_proj_dim"] from settings.py)
    **{f"fb_{i}": True for i in range(8)},

    # News & Economic Calendar — pre/post are redundant with news_ok timing
    "eco_surprise": True,
    "eco_revision": True,
    "news_ok": True,
    "pre_news": False,
    "post_news": False,
    "buzz": False,
    "cat_central_bank": True,
    "cat_inflation": True,
    "cat_labor": True,
    "cat_growth": True,
    "cat_geopolitical": True,
    "cat_commentary": True,

    # News missingness / staleness tracking (drift detection)
    "sentiment_decayed_missing": True,
    "sentiment_decayed_staleness": True,
    "eco_surprise_missing": True,
    "eco_surprise_staleness": True,
    "buzz_missing": False,
    "buzz_staleness": False,

    # COT Features
    "cot_net_hf": True,
    "cot_net_comm": True,
    "cot_hf_mom_4w": True,
    "cot_extreme": True,

    # Macro Spreads
    "spread_us_de": True,
    "spread_us_jp": True,
    "spread_us_gb": True,
    "spread_us_au": True,
    "spread_us_ca": True,
    "spread_us_nz": True,
    "spread_de_gb": True,
    "spread_de_jp": True,
    "spread_us_ch": True,

    # Macro Carry Signals
    "carry_eur": True,
    "carry_jpy": True,
    "carry_gbp": True,
    "carry_aud": True,
    "carry_cad": True,
    "carry_nzd": True,
    "carry_eurgbp": True,
    "carry_eurjpy": True,
    "carry_chf": True,

    # Macro Yield Momentum & Volatility
    "yield_momentum_5d": True,
    "yield_momentum_20d": True,
    "yield_vol_20d": True,
}


def enabled_feature_names(mask: dict[str, bool] | None = None) -> list[str]:
    """Names explicitly marked True in the mask."""
    m = mask if mask is not None else FEATURE_MASK
    return [k for k, v in m.items() if v]


def disabled_feature_names(mask: dict[str, bool] | None = None) -> list[str]:
    """Names explicitly marked False (will be dropped when present)."""
    m = mask if mask is not None else FEATURE_MASK
    return [k for k, v in m.items() if not v]


def apply_feature_mask(frame: Any, mask: dict[str, bool] | None = None) -> Any:
    """
    Drop columns whose mask entry is False.

    Columns not listed in the mask are retained (price / aux columns).
    Supports pandas and Polars DataFrames.
    """
    m = mask if mask is not None else FEATURE_MASK
    if frame is None or not hasattr(frame, "columns") or not hasattr(frame, "drop"):
        return frame
    drop = [c for c in list(frame.columns) if m.get(c) is False]
    if not drop:
        return frame
    # Polars: drop(*names); pandas: drop(columns=...)
    mod = type(frame).__module__
    if mod.startswith("polars"):
        return frame.drop(drop)
    return frame.drop(columns=drop)
