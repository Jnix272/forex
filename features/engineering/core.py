"""
features/engineering/core.py
Core feature engineering: sanitize_frame, point-in-time join helpers,
and the master FeatureEngineer class.
"""

import os
import time

import numpy as np
import polars as pl

from infrastructure.logging_utils import log_feature_build

from .microstructure import (
    order_book_imbalance_proxy,
    trade_arrival_rate,
    order_flow_imbalance,
    average_true_range,
    rolling_volatility,
    bollinger_bands,
    rsi,
    macd,
    lag_returns,
    stochastic_oscillator,
    williams_r,
    cci,
    all_candlestick_patterns,
    kyles_lambda,
    amihud_illiquidity,
    realized_spread,
    vpin,
    multi_level_obi,
    order_book_imbalance_l2,
    frac_diff_proxy,
    vwap_bands,
    volume_weighted_momentum,
    liquidity_vacuum,
    vol_of_vol,
    price_ofi_divergence,
    _pip_size_for_pair,
    add_spread_cost_features,
    add_market_regime_features,
    add_higher_timeframe_context,
    add_volume_profile_features,
    add_volatility_clock_features,
    embedding_placeholders,
    vol_filter,
    circuit_breaker_features,
    position_limit_flags,
    compute_latency_feature,
    missingness_flags,
)
from .regime import (
    hmm_regime_probs,
    cpd_ret,
    regime_persistence,
    regime_gated_features,
    interaction_features,
)
from .cross_asset import CrossAssetFeatures
from .regime_gate import RegimeGateClassifier
from .sentiment import (
    sentiment_tiers,
    buzz_score,
)

# === Sanitize ==========================================================================

_SANITIZE_NO_CLIP = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid_close",
        "ask_close",
        "spread",
        "spread_pips",
        "expected_latency_ms",
        "timestamp_utc",
        "cot_net_hf",
        "cot_net_comm",
    }
)


def sanitize_frame(df: pl.DataFrame, fill_value: float = 0.0, context: str = "frame") -> pl.DataFrame:
    cols = df.select(pl.col(pl.Float32, pl.Float64)).columns
    if not cols:
        return df
    clip_cols = [c for c in cols if c not in _SANITIZE_NO_CLIP]
    noclip_cols = [c for c in cols if c in _SANITIZE_NO_CLIP]
    exprs = [
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
        .then(None)
        .otherwise(pl.col(c))
        .fill_null(fill_value)
        .clip(-1e6, 1e6)
        .alias(c)
        for c in clip_cols
    ] + [
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
        .then(None)
        .otherwise(pl.col(c))
        .fill_null(fill_value)
        .alias(c)
        for c in noclip_cols
    ]
    return df.with_columns(exprs)


# Default delay when observations only have event/publication time (not a true
# available_time). Matches eco join: avoid same-bar leakage at release.
_DEFAULT_PIT_DELAY_MINUTES = 1


def _with_available_time(
    df: pl.DataFrame | None,
    delay_minutes: int = _DEFAULT_PIT_DELAY_MINUTES,
    time_dtype: pl.DataType | None = None,
) -> pl.DataFrame | None:
    """Ensure ``available_time`` exists for point-in-time asof joins."""
    if df is None or len(df) == 0:
        return df
    dtype = time_dtype or pl.Datetime("ns", "UTC")
    if "available_time" in df.columns:
        return df.with_columns(pl.col("available_time").cast(dtype))
    if "timestamp_utc" not in df.columns:
        return df
    return df.with_columns(
        (pl.col("timestamp_utc").cast(dtype) + pl.duration(minutes=int(delay_minutes))).alias("available_time")
    )


def _join_asof_available(
    left: pl.DataFrame,
    right: pl.DataFrame | None,
    *,
    delay_minutes: int = _DEFAULT_PIT_DELAY_MINUTES,
) -> pl.DataFrame:
    """Backward asof join on when information became knowable (available_time)."""
    if right is None or len(right) == 0 or "timestamp_utc" not in left.columns:
        return left
    left_dtype = left.schema["timestamp_utc"]
    right = _with_available_time(right, delay_minutes=delay_minutes, time_dtype=left_dtype)
    if right is None or "available_time" not in right.columns:
        return left
    drop_cols = [c for c in right.columns if c in left.columns and c not in ("timestamp_utc", "available_time")]
    right_join = right.drop(drop_cols).sort("available_time")
    if "timestamp_utc" in right_join.columns:
        right_join = right_join.drop("timestamp_utc")
    right_join = right_join.with_columns(pl.col("available_time").cast(left_dtype))
    out = left.join_asof(
        right_join,
        left_on="timestamp_utc",
        right_on="available_time",
        strategy="backward",
    )
    if "available_time" in out.columns:
        out = out.drop("available_time")
    return out


# === Master Builder ====================================================================

class FeatureEngineer:
    def __init__(
        self,
        atr_window=6,
        ofi_window=20,
        tar_window=30,
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bb_window=20,
        bb_std=2.0,
        lag_windows=None,
        vol_mult=3.0,
        news_buf=2,
        decay_lam=0.1,
        fb_dim=8,
        ca_corr_window=60,
        ca_regime_window=240,
        ca_lags=(1, 5, 15),
        enable_regime_gate=True,
        enable_quality_gate=False,
        enable_no_trade_zones=False,
    ):
        if lag_windows is None:
            lag_windows = [5, 20, 60]
        self.atr_w = atr_window
        self.ofi_w = ofi_window
        self.tar_w = tar_window
        self.rsi_p = rsi_period
        self.mf = macd_fast
        self.ms = macd_slow
        self.msig = macd_signal
        self.bb_w = bb_window
        self.bb_s = bb_std
        self.lags = lag_windows
        self.vm = vol_mult
        self.nb = news_buf
        self.dl = decay_lam
        self.fb = fb_dim

        try:
            from config.settings import FEATURE_CACHE as FC
            from config.settings import FEATURE_SCALES as FS
        except ImportError:
            FS = {}
            FC = {}

        yaml_fs = {}
        yaml_fc = {}
        _cfg_path = os.environ.get("FOREX_CONFIG") or os.environ.get("FOREX_RUN_CONFIG")
        if _cfg_path:
            try:
                import yaml

                with open(_cfg_path, encoding="utf-8") as f:
                    _yaml_root = yaml.safe_load(f) or {}
                yaml_fs = _yaml_root.get("features", {}) or {}
                yaml_fc = _yaml_root.get("feature_cache", {}) or {}
            except Exception:
                yaml_fs, yaml_fc = {}, {}

        self.atr_ws = yaml_fs.get("atr_windows", FS.get("atr_windows", [self.atr_w, 20, 60]))
        self.vol_ws = yaml_fs.get("vol_windows", FS.get("vol_windows", [6, 20, 60]))
        self.ofi_ws = yaml_fs.get("ofi_windows", FS.get("ofi_windows", [5, 20, 60]))
        self.mom_ws = yaml_fs.get("momentum_windows", FS.get("momentum_windows", [5, 20, 60]))
        self.lags = yaml_fs.get("momentum_windows", FS.get("momentum_windows", lag_windows))
        self.vwap_w = int(yaml_fs.get("vwap_window", FS.get("vwap_window", 60)))
        self.chop_w = int(yaml_fs.get("chop_window", FS.get("chop_window", 14)))
        self.vol_regime_w = int(yaml_fs.get("regime_window", FS.get("regime_window", 240)))
        self.volatility_w = int(yaml_fs.get("volatility_window", FS.get("volatility_window", 120)))
        _ca_cw = int(yaml_fs.get("corr_window", FS.get("corr_window", ca_corr_window)))
        _ca_rw = int(yaml_fs.get("regime_window", FS.get("regime_window", ca_regime_window)))
        self.ca = CrossAssetFeatures(
            corr_window=_ca_cw,
            regime_window=_ca_rw,
            lags=ca_lags,
        )
        _fc = yaml_fc if yaml_fc else FC
        self.feature_cache_enabled = bool((_fc or {}).get("enabled", True))
        self.ofi_z_threshold = float((_fc or {}).get("ofi_z_threshold", 2.0))
        slow = list((_fc or {}).get("slow_cols") or [])
        self.slow_cols = ["hurst_exponent" if c == "hurst" else c for c in slow]
        self.enable_regime_gate = bool(enable_regime_gate)
        self.enable_quality_gate = bool(enable_quality_gate)
        self.enable_no_trade_zones = bool(enable_no_trade_zones)
        self.regime_gate = RegimeGateClassifier()
        try:
            from features.macro_features import MacroYieldFeatureBuilder

            self._macro_builder = MacroYieldFeatureBuilder()
        except Exception as e:
            print(f"[FeatureEngineering] WARNING: MacroYieldFeatureBuilder init failed: {e}")
            self._macro_builder = None
        self._warned_finbert_placeholder = False

    def build_chunked(
        self, bars: pl.DataFrame, chunk_size: int = 50_000, output_dir: str | None = None, **kwargs
    ) -> pl.DataFrame:
        """DS-005: Process features in overlapping chronological chunks to bound RAM."""
        n = len(bars)
        overlap = self.warmup_bars_needed
        chunks = []

        for start in range(0, n, chunk_size):
            warmup_start = max(0, start - overlap)
            end = min(start + chunk_size, n)
            chunk = bars[warmup_start:end]

            features = self.build(chunk, **kwargs)

            prefix_len = start - warmup_start
            features = features[prefix_len:]

            if output_dir:
                from pathlib import Path

                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                features.write_parquet(out_path / f"features_{start:08d}.parquet")
            else:
                chunks.append(features)

        if output_dir:
            return pl.DataFrame()
        return pl.concat(chunks, how="vertical_relaxed") if chunks else pl.DataFrame()

    def build_with_warmup(self, bars: pl.DataFrame, warmup_bars: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """DS-002: Build features with warmup context to prevent EMA look-ahead bias."""
        if warmup_bars is not None and len(warmup_bars) > 0:
            n_warmup = len(warmup_bars)
            combined = pl.concat([warmup_bars, bars], how="vertical_relaxed")
            features = self.build(combined, **kwargs)
            return features[n_warmup:]
        return self.build(bars, **kwargs)

    @property
    def warmup_bars_needed(self) -> int:
        """Minimum historical bars needed before a split to avoid EMA cold-start."""
        return 3 * self.ms  # 3 x slow MACD window

    def build(
        self,
        bars: pl.DataFrame,
        cross_asset=None,
        sentiment=None,
        eco_act=None,
        eco_fc=None,
        art_counts=None,
        finbert_embs=None,
        news_events=None,
        cot_data=None,
        pair="EURUSD",
        eco_prior=None,
        news_cats=None,
    ) -> pl.DataFrame:

        import pandas as pd

        if isinstance(bars, pd.DataFrame):
            if bars.index.name is None:
                bars.index.name = "timestamp_utc"
            bars = pl.from_pandas(bars.reset_index())

        if "timestamp" in bars.columns and "timestamp_utc" not in bars.columns:
            bars = bars.with_columns(pl.col("timestamp").alias("timestamp_utc"))
        elif "timestamp_utc" in bars.columns and "timestamp" not in bars.columns:
            bars = bars.with_columns(pl.col("timestamp_utc").alias("timestamp"))

        # Ensure timestamp is UTC and correctly cast datatypes to prevent join_asof crashes
        _ts_dtype = None
        if "timestamp_utc" in bars.columns:
            _ts_dtype = bars.schema["timestamp_utc"]
            bars = bars.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
        if sentiment is not None and "timestamp_utc" in sentiment.columns:
            sentiment = sentiment.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
        if eco_act is not None and "timestamp_utc" in eco_act.columns:
            eco_act = eco_act.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
        if eco_fc is not None and "timestamp_utc" in eco_fc.columns:
            eco_fc = eco_fc.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
        if art_counts is not None and "timestamp_utc" in art_counts.columns:
            art_counts = art_counts.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
        if cot_data is not None and "timestamp_utc" in cot_data.columns:
            cot_data = cot_data.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))

        # Add fractional differentiation before base features need it
        bars = bars.with_columns([frac_diff_proxy("close")])

        # 1. Base Features (Independent)
        F = bars.with_columns(
            [order_book_imbalance_proxy()]
            + trade_arrival_rate(self.tar_w)
            + [order_flow_imbalance(self.ofi_w)]
            + [average_true_range(w) for w in self.atr_ws]
            + [rolling_volatility(w) for w in self.vol_ws]
            + bollinger_bands(self.bb_w, self.bb_s)
            + [rsi(self.rsi_p)]
            + macd(self.mf, self.ms, self.msig)
            + lag_returns(self.lags)
            + stochastic_oscillator()
            + [williams_r()]
            + [cci()]
            + all_candlestick_patterns()
            + [kyles_lambda(20)]
            + [amihud_illiquidity(20)]
            + [realized_spread(10)]
            + [vpin(50, 50)]
            + multi_level_obi(5)
        )

        _l2_cols = ("bid_price_1", "ask_price_1", "bid_price_2", "ask_price_2", "bid_size_1", "ask_size_1")
        if all(c in F.columns for c in _l2_cols):
            F = F.with_columns([order_book_imbalance_l2()])
        elif "obi_l2" in F.columns:
            F = F.with_columns([pl.col("obi_l2").alias("ofi_l2")])
        else:
            F = F.with_columns([pl.lit(0.0).alias("ofi_l2")])

        # 2. Derived Features (Dependent on Base Features)
        tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        vol = pl.when(pl.col("volume") == 0).then(1.0).otherwise(pl.col("volume")).fill_null(1.0)
        vwap = (tp * vol).rolling_sum(self.vwap_w) / vol.rolling_sum(self.vwap_w)
        vwap_bands_expr = vwap_bands(self.vwap_w, 2.0)
        atr_col_name = f"atr_{self.atr_w}" if f"atr_{self.atr_w}" in F.columns else f"atr_{self.atr_ws[0]}"
        F = F.with_columns(
            vwap_bands_expr
            + [volume_weighted_momentum(20)]
            + [
                (
                    (pl.col("ofi") - pl.col("ofi").rolling_mean(min(120, self.ofi_w * 6)))
                    / (pl.col("ofi").rolling_std(min(120, self.ofi_w * 6)) + 1e-9)
                ).alias("ofi_z")
            ]
            + [
                ((pl.col(atr_col_name) / (pl.col("atr_20") + 1e-9)).clip(0.1, 10.0)).alias(
                    f"atr_ratio_{self.atr_w}_20"
                )
            ]
            + [((pl.col("atr_20") / (pl.col("atr_60") + 1e-9)).clip(0.1, 10.0)).alias("atr_ratio_20_60")]
            + [((pl.col("vol_6") / (pl.col("vol_20") + 1e-9)).clip(0.1, 10.0)).alias("vol_ratio_6_20")]
            + [((pl.col("vol_20") / (pl.col("vol_60") + 1e-9)).clip(0.1, 10.0)).alias("vol_ratio_20_60")]
            + [((pl.col("close") - vwap) / (pl.col("atr_20") + 1e-9)).alias("breakout_pressure")]
            + (
                [liquidity_vacuum(120)]
                if "ask_close" in F.columns and "bid_close" in F.columns
                else [pl.lit(0.0).alias("liquidity_vacuum")]
            )
        )

        # 3. Fragility
        F = F.with_columns([vol_of_vol(20, 20), price_ofi_divergence(self.ofi_w, 10)])

        pip_size = _pip_size_for_pair(pair)
        if "spread_avg" in F.columns:
            F = F.with_columns([(pl.col("spread_avg") / pip_size).alias("spread_pips")])
        elif "ask_close" in F.columns:
            F = F.with_columns([((pl.col("ask_close") - pl.col("bid_close")) / pip_size).alias("spread_pips")])
        else:
            F = F.with_columns([pl.lit(0.5).alias("spread_pips")])

        ac = f"atr_{self.atr_w}"
        F = add_spread_cost_features(F, pair=pair, atr_col=ac)
        F = add_market_regime_features(
            F,
            chop_window=self.chop_w,
            regime_window=self.vol_regime_w,
            volatility_window=self.volatility_w,
        )

        existing = set(F.columns)
        F = F.with_columns(regime_gated_features(existing))

        F = F.with_columns(
            [
                pl.when(pl.col("vol_60") <= pl.col("vol_60").rolling_quantile(0.33, window_size=60))
                .then(0)
                .when(pl.col("vol_60") <= pl.col("vol_60").rolling_quantile(0.66, window_size=60))
                .then(1)
                .otherwise(2)
                .alias("realized_vol_regime"),
                (pl.col("adx_14") * (pl.col("rsi_14") - pl.col("rsi_14").shift(5))).alias("trend_quality"),
            ]
        )

        existing = set(F.columns)
        F = F.with_columns(interaction_features(existing))

        F = add_higher_timeframe_context(F)

        F = add_volume_profile_features(F)
        F = add_volatility_clock_features(F)

        # HMM Regime Detection + CPD (Regime Detection Upgrade)
        existing = set(F.columns)
        try:
            from features.regime_detection import detect_regimes_polars

            _t0_reg = time.perf_counter()
            _reg = detect_regimes_polars(
                F,
                close_col="close",
                n_states=3,
                window=60,
                hurst_window=120,
                fractal_window=60,
                step=5,
            )
            F = F.with_columns(_reg)
            log_feature_build(
                "regime_detection",
                n_rows=F.height,
                n_cols=len(_reg.columns),
                status="success",
                t0=_t0_reg,
                note="detect_regimes_polars (HMM + Hurst + fractal)",
            )
        except Exception as _reg_exc:
            print(
                f"[FeatureEngineering] WARNING: detect_regimes_polars failed "
                f"({type(_reg_exc).__name__}: {_reg_exc}); using vol-bucket regime_class"
            )
            F = F.with_columns(hmm_regime_probs(3, 60))
            log_feature_build(
                "regime_detection",
                n_rows=F.height,
                n_cols=3,
                status="fallback_vol_bucket",
                t0=time.perf_counter(),
                exc=_reg_exc,
                note="hmmlearn/numba unavailable",
            )
        F = F.with_columns(cpd_ret("close", 60))
        F = F.with_columns(regime_persistence(20))

        # Circuit Breakers / Kill Switches
        existing = set(F.columns)
        F = F.with_columns(circuit_breaker_features())
        F = F.with_columns(position_limit_flags(atr_col=ac, pair=pair))

        # FinBERT embeddings
        if finbert_embs is not None and len(finbert_embs) > 0:
            try:
                if isinstance(finbert_embs, pl.DataFrame):
                    emb_df = finbert_embs
                else:
                    emb_df = pl.DataFrame(finbert_embs)
                if "timestamp_utc" in emb_df.columns and "timestamp_utc" in F.columns:
                    emb_df = emb_df.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
                    emb_cols = [c for c in emb_df.columns if c.startswith("embed_") or c.startswith("fb_")]
                    F = _join_asof_available(F, emb_df.select(["timestamp_utc", *emb_cols]))
                    rename_map = {}
                    for i, c in enumerate(emb_cols):
                        if c.startswith("fb_"):
                            continue
                        rename_map[c] = f"fb_{i}"
                    if rename_map:
                        F = F.rename(rename_map)
                    for i in range(self.fb):
                        col = f"fb_{i}"
                        if col in F.columns:
                            F = F.with_columns(pl.col(col).fill_null(0.0))
                        else:
                            F = F.with_columns(pl.lit(0.0).alias(col))
                else:
                    if not self._warned_finbert_placeholder:
                        print(
                            "[FeatureEngineering] WARNING: FinBERT embs missing timestamp_utc; using zero placeholders"
                        )
                        self._warned_finbert_placeholder = True
                    F = F.with_columns(embedding_placeholders(8))
            except Exception as e:
                if not self._warned_finbert_placeholder:
                    print(f"[FeatureEngineering] WARNING: FinBERT emb join failed ({e}); using zero placeholders")
                    self._warned_finbert_placeholder = True
                F = F.with_columns(embedding_placeholders(8))
        else:
            if not self._warned_finbert_placeholder:
                print("[FeatureEngineering] WARNING: no FinBERT embeddings provided; using zero placeholders")
                self._warned_finbert_placeholder = True
            F = F.with_columns(embedding_placeholders(8))

        # Cross asset
        F = self.ca.build(F, cross_asset)

        # Macro yield spreads, carry, and momentum
        if self._macro_builder is not None:
            try:
                macro_df = self._macro_builder.build(F)
                macro_cols = [c for c in macro_df.columns if c not in F.columns]
                if macro_cols:
                    F = pl.concat([F, macro_df.select(macro_cols)], how="horizontal_extend")
            except Exception as e:
                print(f"[FeatureEngineering] WARNING: Macro features build failed: {e}")

        # COT (Commitment of Traders) features
        if cot_data is not None:
            try:
                from features.cot_features import add_cot_features

                F = add_cot_features(F, cot_data, pair)
            except Exception as e:
                print(f"[FeatureEngineering] WARNING: COT features build failed: {e}")
                for _cc in ("cot_net_hf", "cot_net_comm", "cot_hf_mom_4w", "cot_extreme"):
                    if _cc not in F.columns:
                        F = F.with_columns(pl.lit(0.0).alias(_cc))
        else:
            for _cc in ("cot_net_hf", "cot_net_comm", "cot_hf_mom_4w", "cot_extreme"):
                if _cc not in F.columns:
                    F = F.with_columns(pl.lit(0.0).alias(_cc))

        # Alias columns for curriculum compatibility / deduplication
        if "gold_dxy_corr_break" in F.columns and "corr_break_gold_DXY" not in F.columns:
            F = F.with_columns(pl.col("gold_dxy_corr_break").alias("corr_break_gold_DXY"))
        if "carry_spot_forward" not in F.columns:
            F = F.with_columns(pl.lit(0.0).alias("carry_spot_forward"))
        if "us_2s10s_spread" in F.columns and "yield_curve_slope" not in F.columns:
            F = F.with_columns(pl.col("us_2s10s_spread").alias("yield_curve_slope"))
        elif "yield_curve_slope" in F.columns and "us_2s10s_spread" not in F.columns:
            F = F.with_columns(pl.col("yield_curve_slope").alias("us_2s10s_spread"))
            F = F.with_columns(pl.col("us_2s10s_spread").diff().alias("us_2s10s_spread_chg"))

        # Filters
        F = F.with_columns([vol_filter(ac, self.vm, 60)])

        # News Filter
        if news_events is not None and len(news_events) > 0:
            import pandas as pd

            buf = pd.Timedelta(minutes=self.nb)
            ts = F["timestamp_utc"].to_pandas()
            flags = pd.DataFrame({"news_ok": 1.0, "pre_news": 0.0, "post_news": 0.0}, index=ts)
            for ev in news_events:
                mask_ok = ((ts >= ev) & (ts <= ev + buf)).values
                mask_pre = ((ts >= ev - buf) & (ts < ev)).values
                mask_post = ((ts >= ev) & (ts <= ev + buf)).values
                flags.loc[mask_ok, "news_ok"] = 0.0
                flags.loc[mask_pre, "pre_news"] = 1.0
                flags.loc[mask_post, "post_news"] = 1.0
            F = F.with_columns(
                [
                    pl.Series("news_ok", flags["news_ok"].values),
                    pl.Series("pre_news", flags["pre_news"].values),
                    pl.Series("post_news", flags["post_news"].values),
                ]
            )
        else:
            F = F.with_columns(
                [pl.lit(1.0).alias("news_ok"), pl.lit(0.0).alias("pre_news"), pl.lit(0.0).alias("post_news")]
            )

        if self.enable_regime_gate:
            rbp = self.regime_gate.fit_predict(F)
            F = F.with_columns([rbp])
            F = F.with_columns(
                [
                    (0.2 + 0.8 * pl.col("regime_break_prob")).clip(0.0, 1.0).alias("gate_gold_weight"),
                    (1.0 - 0.7 * pl.col("regime_break_prob")).clip(0.0, 1.0).alias("gate_yield_weight"),
                    (0.3 + 0.7 * pl.col("regime_break_prob")).clip(0.0, 1.0).alias("gate_risk_weight"),
                ]
            )

        # Sentiment & News features
        if sentiment is not None:
            if "timestamp_utc" in sentiment.columns:
                sentiment = sentiment.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
            F = _join_asof_available(F, sentiment)
            F = sentiment_tiers(F, decay_lam=self.dl, fb_dim=self.fb)
        else:
            F = F.with_columns(
                [
                    pl.lit(0.0).alias("sentiment_raw"),
                    pl.lit(0.0).alias("sentiment_decayed"),
                    *[pl.lit(0.0).alias(f"fb_{i}") for i in range(self.fb)],
                ]
            )

        # Multi-modal sentiment fusion
        if sentiment is not None and any(c in sentiment.columns for c in ("text", "headline")):
            try:
                from features.sentiment_fusion import add_sentiment_features

                _tcol = "text" if "text" in sentiment.columns else "headline"
                _ev = sentiment.rename({_tcol: "text"}) if _tcol != "text" else sentiment
                _keep = [c for c in ("timestamp_utc", "source", "text", "sentiment") if c in _ev.columns]
                _ev_pit = _ev.select(_keep)
                if "timestamp_utc" in _ev_pit.columns:
                    _ev_pit = _ev_pit.with_columns(
                        (
                            pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC"))
                            + pl.duration(minutes=_DEFAULT_PIT_DELAY_MINUTES)
                        ).alias("timestamp_utc")
                    )
                F = add_sentiment_features(
                    F,
                    _ev_pit,
                    time_col="timestamp_utc",
                    lam=float(self.dl) if self.dl else 0.05,
                    dt_sec=3600.0,
                    n_topics=4,
                )
            except Exception as e:
                print(f"[FeatureEngineering] WARNING: multi-modal sentiment failed: {e}")

        if eco_act is not None:
            eco_act = eco_act.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
            if eco_fc is not None:
                eco_fc = eco_fc.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
                eco = eco_act.join(eco_fc, on="timestamp_utc", how="full", coalesce=True).sort("timestamp_utc")
            else:
                eco = eco_act.sort("timestamp_utc")
            if eco_prior is not None:
                eco_prior = eco_prior.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
                drop_pr = [c for c in eco_prior.columns if c in eco.columns and c != "timestamp_utc"]
                eco = eco.join(eco_prior.drop(drop_pr), on="timestamp_utc", how="full", coalesce=True).sort("timestamp_utc")
            drop_cols = [c for c in eco.columns if c in F.columns and c != "timestamp_utc"]
            left_dtype = F.schema["timestamp_utc"]
            if "available_time" not in eco.columns:
                eco = eco.with_columns(
                    (pl.col("timestamp_utc").cast(left_dtype) + pl.duration(minutes=1)).alias("available_time")
                )
            else:
                eco = eco.with_columns(pl.col("available_time").cast(left_dtype))
            eco_join = (
                eco.drop(drop_cols).with_columns(pl.col("available_time").cast(left_dtype)).sort("available_time")
            )
            join_drop = [c for c in ("timestamp_utc",) if c in eco_join.columns]
            F = F.join_asof(
                eco_join.drop(join_drop) if join_drop else eco_join,
                left_on="timestamp_utc",
                right_on="available_time",
                strategy="backward",
            )
            if "available_time" in F.columns:
                F = F.drop("available_time")
            if "actual" in F.columns and "forecast" in F.columns:
                F = F.with_columns(
                    [
                        (
                            pl.col("actual")
                            .cast(pl.String)
                            .str.replace_all(r"[^\d\.\-]", "")
                            .cast(pl.Float64, strict=False)
                            - pl.col("forecast")
                            .cast(pl.String)
                            .str.replace_all(r"[^\d\.\-]", "")
                            .cast(pl.Float64, strict=False)
                        )
                        .fill_null(0.0)
                        .alias("eco_surprise")
                    ]
                )
                if "prior" in F.columns:
                    F = F.with_columns(
                        [
                            (
                                pl.col("actual")
                                .cast(pl.String)
                                .str.replace_all(r"[^\d\.\-]", "")
                                .cast(pl.Float64, strict=False)
                                - pl.col("prior")
                                .cast(pl.String)
                                .str.replace_all(r"[^\d\.\-]", "")
                                .cast(pl.Float64, strict=False)
                            )
                            .fill_null(0.0)
                            .alias("eco_revision")
                        ]
                    )
                else:
                    F = F.with_columns([pl.lit(0.0).alias("eco_revision")])
            else:
                F = F.with_columns([pl.lit(0.0).alias("eco_surprise"), pl.lit(0.0).alias("eco_revision")])
        else:
            F = F.with_columns([pl.lit(0.0).alias("eco_surprise"), pl.lit(0.0).alias("eco_revision")])

        if art_counts is not None:
            if "timestamp_utc" in art_counts.columns:
                art_counts = art_counts.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
            F = _join_asof_available(F, art_counts)
            if "article_counts" in F.columns:
                F = F.with_columns([pl.col("article_counts").fill_null(0.0)])
            else:
                F = F.with_columns([pl.lit(0.0).alias("article_counts")])
            F = F.with_columns([buzz_score()])
        else:
            F = F.with_columns([pl.lit(0.0).alias("buzz")])

        _NEWS_CAT_COLS = [
            "cat_central_bank",
            "cat_inflation",
            "cat_labor",
            "cat_growth",
            "cat_geopolitical",
            "cat_commentary",
        ]
        if news_cats is not None and "timestamp_utc" in news_cats.columns:
            news_cats = news_cats.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
            F = _join_asof_available(F, news_cats)
            for _nc in _NEWS_CAT_COLS:
                if _nc in F.columns:
                    F = F.with_columns(pl.col(_nc).fill_null(0.0))
                else:
                    F = F.with_columns(pl.lit(0.0).alias(_nc))
        else:
            F = F.with_columns([pl.lit(0.0).alias(c) for c in _NEWS_CAT_COLS])

        fb_cols = [f"fb_{i}" for i in range(self.fb)]
        missing_fb = [c for c in fb_cols if c not in F.columns]
        if missing_fb:
            F = F.with_columns([pl.lit(0.0).alias(c) for c in missing_fb])

        # Temporal / session clock features.
        import pandas as pd

        ts_pd = F["timestamp_utc"].to_pandas()
        h = ts_pd.dt.hour
        m = ts_pd.dt.minute
        tm = h * 60 + m
        temporal = [
            pl.Series("time_sin", np.sin(2 * np.pi * tm / 1440)),
            pl.Series("time_cos", np.cos(2 * np.pi * tm / 1440)),
            pl.Series("day_sin", np.sin(2 * np.pi * ts_pd.dt.dayofweek / 5)),
            pl.Series("day_cos", np.cos(2 * np.pi * ts_pd.dt.dayofweek / 5)),
        ]
        if "london_ny" not in F.columns:
            temporal.append(pl.Series("london_ny", ((h >= 13) & (h <= 17)).astype(float)))
        F = F.with_columns(temporal)

        # Missingness
        tracked = ["sentiment_decayed", "eco_surprise", "buzz"]
        F = missingness_flags(F, tracked, 0.9)
        F = F.with_columns([pl.col(c).fill_null(strategy="forward").fill_null(0.0) for c in tracked if c in F.columns])

        # Latency
        try:
            from config.settings import EXECUTION as _EX

            _lat_base = float(_EX.get("latency_baseline_ms", 50.0))
        except Exception:
            _lat_base = 50.0

        F = F.with_columns([compute_latency_feature(F, _lat_base, ac)])

        # Final cleanup
        n0 = len(F)
        F = F.drop(
            ["actual", "forecast", "prior", "_eco_raw", "_eco_scale", "sentiment", "article_counts"], strict=False
        )
        F = F.drop_nulls(subset=[ac])
        F = F.fill_null(strategy="forward").fill_null(0.0)
        if n0 > len(F):
            print(f"[Features] Dropped {n0 - len(F):,} NaN rows -> {len(F):,}  {len(F.columns)}")

        if "timestamp_utc" in F.columns and _ts_dtype is not None:
            F = F.with_columns(pl.col("timestamp_utc").cast(_ts_dtype))

        # No-trade zones
        if self.enable_no_trade_zones:
            try:
                from features.no_trade_zones import compute_heuristic_no_trade_score

                _nt_cols = [c for c in ("atr_6", "spread_pips", "adx_14", "rsi_14") if c in F.columns]
                _nt_pd = F.select(_nt_cols).to_pandas()
                F = F.with_columns(pl.Series("no_trade_score", compute_heuristic_no_trade_score(_nt_pd)))
                print("[Features] No-trade zones enabled: added 'no_trade_score' column")
            except Exception as _e:
                print(f"[Features] WARNING: no-trade zones failed: {_e}")

        # Feature quality monitor
        if self.enable_quality_gate:
            try:
                from features.feature_quality_monitor import feature_quality_monitor, filter_features

                self.quality_report = feature_quality_monitor(F)
                _n_severe = int((self.quality_report["psi_level"] == "severe").sum())
                _n_leaky = int(self.quality_report["leak_flag"].sum())
                print(
                    f"[Features] Quality monitor: {len(self.quality_report)} features | "
                    f"{_n_severe} severe-drift | {_n_leaky} leaky"
                )
                F, _rep, _dropped = filter_features(F, drop_leaky=True, drop_severe_drift=True)
                if _dropped:
                    print(f"[Features] Quality gate dropped {len(_dropped)} features: {sorted(_dropped)[:10]}...")
            except Exception as _e:
                print(f"[Features] WARNING: feature quality monitor failed: {_e}")

        return sanitize_frame(F, fill_value=0.0, context="FeatureEngineer.build")
