"""
data/feature_definitions.py
===========================
Feature Specification Registry — Single source of truth for all features.
Each feature is defined once with its type, source, transformation, and dependencies.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class FeatureType(Enum):
    """Feature type determines storage, materialization, and serving strategy."""
    NUMERIC = "numeric"          # Scalar float/int per timestamp
    CATEGORICAL = "categorical"  # String or integer category
    EMBEDDING = "embedding"      # Vector per timestamp (e.g., FinBERT)
    TIMESTAMP = "timestamp"      # datetime index


class MaterializationStrategy(Enum):
    """How and when the feature is materialized."""
    EAGER_BATCH = "eager_batch"      # Full backfill nightly (macro, COT)
    INCREMENTAL = "incremental"      # Append-only rolling (returns, volatility)
    ON_DEMAND = "on_demand"          # Computed at inference, cached (microstructure)
    SNAPSHOT = "snapshot"            # Tagged version at promotion time


class FeatureSource(Enum):
    """Origin system for the feature."""
    PRICE = "price"              # OHLCV from tick resampling
    ORDERBOOK = "orderbook"      # L2/L3 reconstructed or real
    MACRO = "macro"              # Yields, spreads, economic calendar
    SENTIMENT = "sentiment"      # News, social, FinBERT
    COT = "cot"                  # Commitment of Traders
    EXTERNAL = "external"        # Cross-asset, alternatives
    DERIVED = "derived"          # Computed from other features


@dataclass
class FeatureSpec:
    """
    Complete specification for a single feature.
    
    The content hash (SHA256 of transformation + dependencies + params)
    enables deduplication and version detection.
    """
    name: str
    feature_type: FeatureType
    description: str
    source: FeatureSource
    transformation: str           # Human-readable: "log_ret(close, lag=1)"
    params: dict[str, Any] = field(default_factory=dict)  # e.g., {"window": 20, "lag": 1}
    dependencies: list[str] = field(default_factory=list)  # Upstream feature names
    version: int = 1
    tags: list[str] = field(default_factory=list)
    owner: str = "ml-team"
    materialization: MaterializationStrategy = MaterializationStrategy.INCREMENTAL
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat() + "Z")
    deprecated: bool = False
    # Computed fields (not in constructor)
    _hash: str = field(init=False, repr=False, default="")

    def __post_init__(self):
        """Compute content hash for versioning."""
        content = {
            "name": self.name,
            "feature_type": self.feature_type.value,
            "source": self.source.value,
            "transformation": self.transformation,
            "params": self.params,
            "dependencies": sorted(self.dependencies),
            "version": self.version,
        }
        self._hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def hash(self) -> str:
        return self._hash

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["feature_type"] = self.feature_type.value
        d["source"] = self.source.value
        d["materialization"] = self.materialization.value
        d["hash"] = self.hash
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSpec":
        data = data.copy()
        data["feature_type"] = FeatureType(data["feature_type"])
        data["source"] = FeatureSource(data["source"])
        data["materialization"] = MaterializationStrategy(data.get("materialization", "incremental"))
        spec = cls(**{k: v for k, v in data.items() if k not in ("hash", "_hash")})
        return spec


# ═══════════════════════════════════════════════════════════════════════════
# BUILT-IN FEATURE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
# This is the canonical list. Add new features here, then run materialization.

BUILTIN_FEATURES: list[FeatureSpec] = [
    # ── Price / Returns ──
    FeatureSpec(
        name="log_ret_1",
        feature_type=FeatureType.NUMERIC,
        description="Log return: log(close_t / close_{t-1})",
        source=FeatureSource.PRICE,
        transformation="log(close).diff(1)",
        params={"lag": 1},
        dependencies=["close"],
        tags=["returns", "core"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="log_ret_5",
        feature_type=FeatureType.NUMERIC,
        description="5-bar log return",
        source=FeatureSource.PRICE,
        transformation="log(close).diff(5)",
        params={"lag": 5},
        dependencies=["close"],
        tags=["returns", "momentum"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="log_ret_20",
        feature_type=FeatureType.NUMERIC,
        description="20-bar log return (daily-ish)",
        source=FeatureSource.PRICE,
        transformation="log(close).diff(20)",
        params={"lag": 20},
        dependencies=["close"],
        tags=["returns", "momentum"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="close",
        feature_type=FeatureType.NUMERIC,
        description="Close price (mid)",
        source=FeatureSource.PRICE,
        transformation="identity",
        params={},
        dependencies=[],
        tags=["price", "core"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),

    # ── Volatility ──
    FeatureSpec(
        name="atr_6",
        feature_type=FeatureType.NUMERIC,
        description="Average True Range (6-bar)",
        source=FeatureSource.PRICE,
        transformation="ATR(high, low, close, window=6)",
        params={"window": 6},
        dependencies=["high", "low", "close"],
        tags=["volatility", "core"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="atr_20",
        feature_type=FeatureType.NUMERIC,
        description="Average True Range (20-bar)",
        source=FeatureSource.PRICE,
        transformation="ATR(high, low, close, window=20)",
        params={"window": 20},
        dependencies=["high", "low", "close"],
        tags=["volatility"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="rolling_vol_20",
        feature_type=FeatureType.NUMERIC,
        description="Rolling 20-bar std of log returns",
        source=FeatureSource.PRICE,
        transformation="rolling_std(log_ret_1, window=20)",
        params={"window": 20},
        dependencies=["log_ret_1"],
        tags=["volatility"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="bollinger_upper_20",
        feature_type=FeatureType.NUMERIC,
        description="Bollinger Band upper (20, 2σ)",
        source=FeatureSource.PRICE,
        transformation="rolling_mean(close, 20) + 2 * rolling_std(close, 20)",
        params={"window": 20, "n_std": 2.0},
        dependencies=["close"],
        tags=["volatility", "mean_reversion"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="bollinger_lower_20",
        feature_type=FeatureType.NUMERIC,
        description="Bollinger Band lower (20, 2σ)",
        source=FeatureSource.PRICE,
        transformation="rolling_mean(close, 20) - 2 * rolling_std(close, 20)",
        params={"window": 20, "n_std": 2.0},
        dependencies=["close"],
        tags=["volatility", "mean_reversion"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),

    # ── Microstructure ──
    FeatureSpec(
        name="ofi_20",
        feature_type=FeatureType.NUMERIC,
        description="Order Flow Imbalance (20-bar rolling)",
        source=FeatureSource.ORDERBOOK,
        transformation="OFI(volume, close-open, window=20)",
        params={"window": 20},
        dependencies=["volume", "open", "close"],
        tags=["microstructure", "flow"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="obi_proxy",
        feature_type=FeatureType.NUMERIC,
        description="Order Book Imbalance proxy from OHLC",
        source=FeatureSource.ORDERBOOK,
        transformation="(close - low) / (high - low + eps)",
        params={},
        dependencies=["open", "high", "low", "close"],
        tags=["microstructure", "proxy"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="trade_arrival_rate_30",
        feature_type=FeatureType.NUMERIC,
        description="Normalized trade arrival rate (30-bar)",
        source=FeatureSource.ORDERBOOK,
        transformation="zscore(volume, window=30)",
        params={"window": 30},
        dependencies=["volume"],
        tags=["microstructure", "flow"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),

    # ── Session / Time ──
    FeatureSpec(
        name="session_asian",
        feature_type=FeatureType.NUMERIC,
        description="1 if Asian session (00:00-07:00 UTC)",
        source=FeatureSource.PRICE,
        transformation="hour_in_range(0, 7)",
        params={"start": 0, "end": 7},
        dependencies=["timestamp_utc"],
        tags=["session", "time"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="session_london",
        feature_type=FeatureType.NUMERIC,
        description="1 if London session (07:00-16:00 UTC)",
        source=FeatureSource.PRICE,
        transformation="hour_in_range(7, 16)",
        params={"start": 7, "end": 16},
        dependencies=["timestamp_utc"],
        tags=["session", "time"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="session_ny",
        feature_type=FeatureType.NUMERIC,
        description="1 if NY session (13:00-22:00 UTC)",
        source=FeatureSource.PRICE,
        transformation="hour_in_range(13, 22)",
        params={"start": 13, "end": 22},
        dependencies=["timestamp_utc"],
        tags=["session", "time"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="session_overlap_lon_ny",
        feature_type=FeatureType.NUMERIC,
        description="1 if London-NY overlap (13:00-16:00 UTC)",
        source=FeatureSource.PRICE,
        transformation="hour_in_range(13, 16)",
        params={"start": 13, "end": 16},
        dependencies=["timestamp_utc"],
        tags=["session", "time", "overlap"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),

    # ── Advanced Features (from advanced_features.py) ──
    FeatureSpec(
        name="hurst_120",
        feature_type=FeatureType.NUMERIC,
        description="Rolling Hurst exponent (120-bar)",
        source=FeatureSource.DERIVED,
        transformation="rolling_hurst(log_ret_1, window=120, step=20)",
        params={"window": 120, "step": 20},
        dependencies=["log_ret_1"],
        tags=["regime", "fractal"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="fractal_dim_30",
        feature_type=FeatureType.NUMERIC,
        description="Fractal dimension (30-bar)",
        source=FeatureSource.DERIVED,
        transformation="fractal_dimension(close, window=30)",
        params={"window": 30},
        dependencies=["close"],
        tags=["regime", "fractal"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="iv_proxy_20",
        feature_type=FeatureType.NUMERIC,
        description="Implied Volatility proxy (Garman-Klass, 20-bar)",
        source=FeatureSource.DERIVED,
        transformation="iv_proxy(high, low, close, open, window=20)",
        params={"window": 20},
        dependencies=["high", "low", "close", "open"],
        tags=["options_proxy", "volatility"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="skew_proxy_20",
        feature_type=FeatureType.NUMERIC,
        description="Return skewness proxy (20-bar)",
        source=FeatureSource.DERIVED,
        transformation="rolling_skew(log_ret_1, window=20)",
        params={"window": 20},
        dependencies=["log_ret_1"],
        tags=["options_proxy", "distribution"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),

    # ── Macro / Yield Spreads ──
    FeatureSpec(
        name="spread_us_de_10y",
        feature_type=FeatureType.NUMERIC,
        description="US 10Y - DE 10Y yield spread",
        source=FeatureSource.MACRO,
        transformation="us10y - de10y",
        params={},
        dependencies=["us10y", "de10y"],
        tags=["macro", "yield_spread", "fred"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),
    FeatureSpec(
        name="spread_us_jp_10y",
        feature_type=FeatureType.NUMERIC,
        description="US 10Y - JP 10Y yield spread",
        source=FeatureSource.MACRO,
        transformation="us10y - jp10y",
        params={},
        dependencies=["us10y", "jp10y"],
        tags=["macro", "yield_spread", "fred"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),
    FeatureSpec(
        name="yield_curve_slope_us",
        feature_type=FeatureType.NUMERIC,
        description="US 10Y - US 2Y slope",
        source=FeatureSource.MACRO,
        transformation="us10y - us2y",
        params={},
        dependencies=["us10y", "us2y"],
        tags=["macro", "yield_curve", "fred"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),

    # ── Sentiment ──
    FeatureSpec(
        name="sentiment_raw",
        feature_type=FeatureType.NUMERIC,
        description="Raw sentiment score (VADER/FinBERT)",
        source=FeatureSource.SENTIMENT,
        transformation="sentiment_score(headlines)",
        params={},
        dependencies=["news_headlines"],
        tags=["sentiment", "nlp"],
        materialization=MaterializationStrategy.ON_DEMAND,
    ),
    FeatureSpec(
        name="sentiment_decayed",
        feature_type=FeatureType.NUMERIC,
        description="Exponentially decayed sentiment (λ=0.1) decayed sentiment",
        source=FeatureSource.SENTIMENT,
        transformation="ewma(sentiment_raw, lambda=0.1)",
        params={"decay_lambda": 0.1},
        dependencies=["sentiment_raw"],
        tags=["sentiment", "decayed"],
        materialization=MaterializationStrategy.ON_DEMAND,
    ),

    # ── COT (Commitment of Traders) ──
    FeatureSpec(
        name="cot_net_noncom",
        feature_type=FeatureType.NUMERIC,
        description="Non-commercial net position (long - short) / (long + short)",
        source=FeatureSource.COT,
        transformation="(long_noncom - short_noncom) / (long_noncom + short_noncom + eps)",
        params={},
        dependencies=["cot_long_noncom", "cot_short_noncom"],
        tags=["cot", "positioning"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),
    FeatureSpec(
        name="cot_extreme",
        feature_type=FeatureType.NUMERIC,
        description="1 if |cot_net| > 0.7 (extreme positioning)",
        source=FeatureSource.COT,
        transformation="abs(cot_net_noncom) > 0.7",
        params={"threshold": 0.7},
        dependencies=["cot_net_noncom"],
        tags=["cot", "extreme"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),

    # ── Cross-Asset / Intermarket ──
    FeatureSpec(
        name="dxy",
        feature_type=FeatureType.NUMERIC,
        description="Dollar Index (DXY)",
        source=FeatureSource.EXTERNAL,
        transformation="dxy_close",
        params={},
        dependencies=[],
        tags=["cross_asset", "dxy"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),
    FeatureSpec(
        name="vix",
        feature_type=FeatureType.NUMERIC,
        description="VIX volatility index",
        source=FeatureSource.EXTERNAL,
        transformation="vix_close",
        params={},
        dependencies=[],
        tags=["cross_asset", "vix", "risk"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),
    FeatureSpec(
        name="wti",
        feature_type=FeatureType.NUMERIC,
        description="WTI Crude Oil price",
        source=FeatureSource.EXTERNAL,
        transformation="wti_close",
        params={},
        dependencies=[],
        tags=["cross_asset", "commodity"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),
    FeatureSpec(
        name="gold",
        feature_type=FeatureType.NUMERIC,
        description="Gold (XAUUSD) price",
        source=FeatureSource.EXTERNAL,
        transformation="gold_close",
        params={},
        dependencies=[],
        tags=["cross_asset", "commodity", "safe_haven"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),
    FeatureSpec(
        name="hlr",
        feature_type=FeatureType.NUMERIC,
        description="High-Low Ratio relative to Close",
        source=FeatureSource.PRICE,
        transformation="(high - low) / close",
        params={},
        dependencies=["close"],
        tags=["volatility", "intraday"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="spread_bps",
        feature_type=FeatureType.NUMERIC,
        description="Bid-ask spread in basis points (|close - open| / close * 10000)",
        source=FeatureSource.PRICE,
        transformation="abs(close - open) / close * 10000",
        params={},
        dependencies=["close"],
        tags=["microstructure", "cost"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="hour_sin",
        feature_type=FeatureType.NUMERIC,
        description="Sine encoding of hour of day (0-23)",
        source=FeatureSource.DERIVED,
        transformation="sin(2*pi*hour/24)",
        params={},
        dependencies=[],
        tags=["session", "cyclic"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="hour_cos",
        feature_type=FeatureType.NUMERIC,
        description="Cosine encoding of hour of day (0-23)",
        source=FeatureSource.DERIVED,
        transformation="cos(2*pi*hour/24)",
        params={},
        dependencies=[],
        tags=["session", "cyclic"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="dow_sin",
        feature_type=FeatureType.NUMERIC,
        description="Sine encoding of day of week (0=Mon .. 6=Sun)",
        source=FeatureSource.DERIVED,
        transformation="sin(2*pi*(dow-1)/7)",
        params={},
        dependencies=[],
        tags=["session", "cyclic", "weekly"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="dow_cos",
        feature_type=FeatureType.NUMERIC,
        description="Cosine encoding of day of week (0=Mon .. 6=Sun)",
        source=FeatureSource.DERIVED,
        transformation="cos(2*pi*(dow-1)/7)",
        params={},
        dependencies=[],
        tags=["session", "cyclic", "weekly"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="session_asia",
        feature_type=FeatureType.NUMERIC,
        description="Asia session indicator (23:00-06:59 UTC)",
        source=FeatureSource.DERIVED,
        transformation="hour in [23, 0, 1, 2, 3, 4, 5, 6]",
        params={"hours": [23, 0, 1, 2, 3, 4, 5, 6]},
        dependencies=[],
        tags=["session", "asia"],
        materialization=MaterializationStrategy.SNAPSHOT,
    ),
    FeatureSpec(
        name="is_monday",
        feature_type=FeatureType.NUMERIC,
        description="Monday indicator",
        source=FeatureSource.DERIVED,
        transformation="dow == 1",
        params={},
        dependencies=[],
        tags=["session", "day_of_week"],
        materialization=MaterializationStrategy.SNAPSHOT,
    ),
    FeatureSpec(
        name="is_friday",
        feature_type=FeatureType.NUMERIC,
        description="Friday indicator",
        source=FeatureSource.DERIVED,
        transformation="dow == 5",
        params={},
        dependencies=[],
        tags=["session", "day_of_week"],
        materialization=MaterializationStrategy.SNAPSHOT,
    ),
    FeatureSpec(
        name="yield_2y10y",
        feature_type=FeatureType.NUMERIC,
        description="US 2y-10y Treasury yield spread",
        source=FeatureSource.MACRO,
        transformation="us_10y_yield - us_2y_yield",
        params={},
        dependencies=[],
        tags=["macro", "yield_curve", "rates"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),
    FeatureSpec(
        name="usd_index",
        feature_type=FeatureType.NUMERIC,
        description="US Dollar Index (DXY) weighted against 6 major currencies",
        source=FeatureSource.EXTERNAL,
        transformation="dxy_close",
        params={},
        dependencies=[],
        tags=["cross_asset", "dxy", "usd"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),
    FeatureSpec(
        name="vix_close",
        feature_type=FeatureType.NUMERIC,
        description="VIX volatility index close price",
        source=FeatureSource.EXTERNAL,
        transformation="vix_close",
        params={},
        dependencies=[],
        tags=["cross_asset", "vix", "risk"],
        materialization=MaterializationStrategy.EAGER_BATCH,
    ),
    # ═════════════════════════════════════════════════════════════════════════
    # CLASSICAL INDICATORS (Improvement #2)
    # ═════════════════════════════════════════════════════════════════════════
    FeatureSpec(
        name="stoch_k",
        feature_type=FeatureType.NUMERIC,
        description="Stochastic Oscillator %K (14, 3)",
        source=FeatureSource.PRICE,
        transformation="stochastic(k=14, d=3) %K",
        params={"k_period": 14, "d_period": 3},
        dependencies=["high", "low", "close"],
        tags=["momentum", "classical"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="stoch_d",
        feature_type=FeatureType.NUMERIC,
        description="Stochastic Oscillator %D (signal line)",
        source=FeatureSource.PRICE,
        transformation="stochastic(k=14, d=3) %D",
        params={"k_period": 14, "d_period": 3},
        dependencies=["high", "low", "close"],
        tags=["momentum", "classical"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="williams_r",
        feature_type=FeatureType.NUMERIC,
        description="Williams %R (14-bar)",
        source=FeatureSource.PRICE,
        transformation="williams_r(period=14)",
        params={"period": 14},
        dependencies=["high", "low", "close"],
        tags=["momentum", "classical"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="cci",
        feature_type=FeatureType.NUMERIC,
        description="Commodity Channel Index (20-bar)",
        source=FeatureSource.PRICE,
        transformation="cci(period=20)",
        params={"period": 20},
        dependencies=["high", "low", "close"],
        tags=["momentum", "classical"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    # ═════════════════════════════════════════════════════════════════════════
    # VOLUME-WEIGHTED FEATURES (Improvement #4)
    # ═════════════════════════════════════════════════════════════════════════
    FeatureSpec(
        name="vwap",
        feature_type=FeatureType.NUMERIC,
        description="Volume-Weighted Average Price (60-bar)",
        source=FeatureSource.PRICE,
        transformation="vwap(window=60)",
        params={"window": 60},
        dependencies=["high", "low", "close", "volume"],
        tags=["volume", "vwap"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="vwap_upper",
        feature_type=FeatureType.NUMERIC,
        description="VWAP upper band (2σ)",
        source=FeatureSource.PRICE,
        transformation="vwap_upper(window=60, n_std=2.0)",
        params={"window": 60, "n_std": 2.0},
        dependencies=["high", "low", "close", "volume"],
        tags=["volume", "vwap"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="vwap_lower",
        feature_type=FeatureType.NUMERIC,
        description="VWAP lower band (2σ)",
        source=FeatureSource.PRICE,
        transformation="vwap_lower(window=60, n_std=2.0)",
        params={"window": 60, "n_std": 2.0},
        dependencies=["high", "low", "close", "volume"],
        tags=["volume", "vwap"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="vwap_zscore",
        feature_type=FeatureType.NUMERIC,
        description="VWAP z-score (price distance from VWAP in std units)",
        source=FeatureSource.PRICE,
        transformation="vwap_zscore(window=60)",
        params={"window": 60},
        dependencies=["high", "low", "close", "volume"],
        tags=["volume", "vwap"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="vwma_ret",
        feature_type=FeatureType.NUMERIC,
        description="Volume-weighted moving average of returns (20-bar)",
        source=FeatureSource.PRICE,
        transformation="volume_weighted_momentum(window=20)",
        params={"window": 20},
        dependencies=["close", "volume"],
        tags=["volume", "momentum"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    # ═════════════════════════════════════════════════════════════════════════
    # REGIME-GATED FEATURES (Improvement #3)
    # ═════════════════════════════════════════════════════════════════════════
    FeatureSpec(
        name="rsi_trend",
        feature_type=FeatureType.NUMERIC,
        description="RSI gated by trend regime (active only in trending markets)",
        source=FeatureSource.DERIVED,
        transformation="rsi_14 * trend_regime",
        params={},
        dependencies=["rsi_14", "trend_regime"],
        tags=["regime_gated", "trend"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="macd_trend",
        feature_type=FeatureType.NUMERIC,
        description="MACD gated by trend regime",
        source=FeatureSource.DERIVED,
        transformation="macd * trend_regime",
        params={},
        dependencies=["macd", "trend_regime"],
        tags=["regime_gated", "trend"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="adx_trend",
        feature_type=FeatureType.NUMERIC,
        description="ADX gated by trend regime",
        source=FeatureSource.DERIVED,
        transformation="adx_14 * trend_regime",
        params={},
        dependencies=["adx_14", "trend_regime"],
        tags=["regime_gated", "trend"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="ret5_trend",
        feature_type=FeatureType.NUMERIC,
        description="5-bar return gated by trend regime",
        source=FeatureSource.DERIVED,
        transformation="ret_5 * trend_regime",
        params={},
        dependencies=["ret_5", "trend_regime"],
        tags=["regime_gated", "trend"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="stoch_range",
        feature_type=FeatureType.NUMERIC,
        description="Stochastic %K gated by range regime",
        source=FeatureSource.DERIVED,
        transformation="stoch_k * range_regime",
        params={},
        dependencies=["stoch_k", "range_regime"],
        tags=["regime_gated", "range"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="bb_pct_range",
        feature_type=FeatureType.NUMERIC,
        description="Bollinger %B gated by range regime",
        source=FeatureSource.DERIVED,
        transformation="bb_pct * range_regime",
        params={},
        dependencies=["bb_pct", "range_regime"],
        tags=["regime_gated", "range"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="williams_range",
        feature_type=FeatureType.NUMERIC,
        description="Williams %R gated by range regime",
        source=FeatureSource.DERIVED,
        transformation="williams_r * range_regime",
        params={},
        dependencies=["williams_r", "range_regime"],
        tags=["regime_gated", "range"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="cci_range",
        feature_type=FeatureType.NUMERIC,
        description="CCI gated by range regime",
        source=FeatureSource.DERIVED,
        transformation="cci * range_regime",
        params={},
        dependencies=["cci", "range_regime"],
        tags=["regime_gated", "range"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="atr_ratio_volatile",
        feature_type=FeatureType.NUMERIC,
        description="ATR ratio gated by volatile regime",
        source=FeatureSource.DERIVED,
        transformation="atr_ratio_6_20 * volatility_regime",
        params={},
        dependencies=["atr_ratio_6_20", "volatility_regime"],
        tags=["regime_gated", "volatile"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="breakout_volatile",
        feature_type=FeatureType.NUMERIC,
        description="Breakout pressure gated by volatile regime",
        source=FeatureSource.DERIVED,
        transformation="breakout_pressure * volatility_regime",
        params={},
        dependencies=["breakout_pressure", "volatility_regime"],
        tags=["regime_gated", "volatile"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="vwap_z_volatile",
        feature_type=FeatureType.NUMERIC,
        description="VWAP z-score gated by volatile regime",
        source=FeatureSource.DERIVED,
        transformation="vwap_zscore * volatility_regime",
        params={},
        dependencies=["vwap_zscore", "volatility_regime"],
        tags=["regime_gated", "volatile"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    # ═════════════════════════════════════════════════════════════════════════
    # INTERACTION FEATURES (Improvement #5)
    # ═════════════════════════════════════════════════════════════════════════
    FeatureSpec(
        name="atr_x_ofi",
        feature_type=FeatureType.NUMERIC,
        description="ATR × OFI interaction (volatility × flow)",
        source=FeatureSource.DERIVED,
        transformation="atr_6 * ofi_z",
        params={},
        dependencies=["atr_6", "ofi_z"],
        tags=["interaction"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="atr_ratio_x_ofi",
        feature_type=FeatureType.NUMERIC,
        description="ATR ratio × OFI interaction",
        source=FeatureSource.DERIVED,
        transformation="atr_ratio_6_20 * ofi_z",
        params={},
        dependencies=["atr_ratio_6_20", "ofi_z"],
        tags=["interaction"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="rsi_x_trend",
        feature_type=FeatureType.NUMERIC,
        description="RSI × trend regime interaction",
        source=FeatureSource.DERIVED,
        transformation="rsi_14 * trend_regime",
        params={},
        dependencies=["rsi_14", "trend_regime"],
        tags=["interaction"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="macd_x_trend",
        feature_type=FeatureType.NUMERIC,
        description="MACD × trend regime interaction",
        source=FeatureSource.DERIVED,
        transformation="macd * trend_regime",
        params={},
        dependencies=["macd", "trend_regime"],
        tags=["interaction"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="stoch_x_range",
        feature_type=FeatureType.NUMERIC,
        description="Stochastic × range regime interaction",
        source=FeatureSource.DERIVED,
        transformation="stoch_k * range_regime",
        params={},
        dependencies=["stoch_k", "range_regime"],
        tags=["interaction"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="bb_x_range",
        feature_type=FeatureType.NUMERIC,
        description="Bollinger %B × range regime interaction",
        source=FeatureSource.DERIVED,
        transformation="bb_pct * range_regime",
        params={},
        dependencies=["bb_pct", "range_regime"],
        tags=["interaction"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="cost_x_volatile",
        feature_type=FeatureType.NUMERIC,
        description="Cost-to-ATR × volatile regime interaction",
        source=FeatureSource.DERIVED,
        transformation="cost_to_atr * volatility_regime",
        params={},
        dependencies=["cost_to_atr", "volatility_regime"],
        tags=["interaction"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="curve_x_risk",
        feature_type=FeatureType.NUMERIC,
        description="Yield curve slope × risk-off signal interaction",
        source=FeatureSource.DERIVED,
        transformation="yield_curve_slope * risk_off_signal",
        params={},
        dependencies=["yield_curve_slope", "risk_off_signal"],
        tags=["interaction"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="carry_x_trend",
        feature_type=FeatureType.NUMERIC,
        description="Carry × trend regime interaction",
        source=FeatureSource.DERIVED,
        transformation="carry_eur * trend_regime",
        params={},
        dependencies=["carry_eur", "trend_regime"],
        tags=["interaction"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="gold_dxy_x_risk",
        feature_type=FeatureType.NUMERIC,
        description="Gold-DXY correlation × risk-off signal interaction",
        source=FeatureSource.DERIVED,
        transformation="gold_dxy_corr * risk_off_signal",
        params={},
        dependencies=["gold_dxy_corr", "risk_off_signal"],
        tags=["interaction"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    # ═════════════════════════════════════════════════════════════════════════
    # MULTI-PAIR FEATURES (Improvement #6)
    # ═════════════════════════════════════════════════════════════════════════
    FeatureSpec(
        name="vpin",
        feature_type=FeatureType.NUMERIC,
        description="Volume-Synchronized Probability of Informed Trading",
        source=FeatureSource.ORDERBOOK,
        transformation="VPIN(bucket=50, buckets=50)",
        params={"bucket_size": 50, "n_buckets": 50},
        dependencies=["volume", "close", "open"],
        tags=["multipair", "microstructure", "toxicity"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="rolling_skew_20",
        feature_type=FeatureType.NUMERIC,
        description="Rolling skewness of returns (20-bar)",
        source=FeatureSource.DERIVED,
        transformation="rolling_skew(ret, window=20)",
        params={"window": 20},
        dependencies=["close"],
        tags=["multipair", "moments"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="rolling_kurt_20",
        feature_type=FeatureType.NUMERIC,
        description="Rolling excess kurtosis of returns (20-bar)",
        source=FeatureSource.DERIVED,
        transformation="rolling_kurt(ret, window=20)",
        params={"window": 20},
        dependencies=["close"],
        tags=["multipair", "moments"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="rvol_ratio",
        feature_type=FeatureType.NUMERIC,
        description="Up/down realized volatility ratio",
        source=FeatureSource.DERIVED,
        transformation="up_std / dn_std",
        params={"window": 20},
        dependencies=["close"],
        tags=["multipair", "volatility"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="asia_london_gap",
        feature_type=FeatureType.NUMERIC,
        description="Asia close to London open gap normalized by ATR",
        source=FeatureSource.PRICE,
        transformation="asia_london_gap(atr=atr_6)",
        params={},
        dependencies=["close", "atr_6", "timestamp_utc"],
        tags=["multipair", "session"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="rel_mom_EURUSD_USDJPY",
        feature_type=FeatureType.NUMERIC,
        description="Relative momentum EURUSD vs USDJPY (20-bar)",
        source=FeatureSource.DERIVED,
        transformation="rel_mom(EURUSD, USDJPY, window=20)",
        params={"window": 20},
        dependencies=["close_EURUSD", "close_USDJPY"],
        tags=["multipair", "relative_momentum"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="vol_share_EURUSD",
        feature_type=FeatureType.NUMERIC,
        description="EURUSD ATR share of basket ATR",
        source=FeatureSource.DERIVED,
        transformation="vol_share(EURUSD)",
        params={},
        dependencies=["atr_EURUSD", "atr_basket"],
        tags=["multipair", "volatility"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="cross_dispersion",
        feature_type=FeatureType.NUMERIC,
        description="Cross-pair return dispersion (std of 5-bar returns)",
        source=FeatureSource.DERIVED,
        transformation="cross_dispersion(window=5)",
        params={"window": 5},
        dependencies=["returns_basket"],
        tags=["multipair", "dispersion"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="time_to_barrier_est",
        feature_type=FeatureType.NUMERIC,
        description="ATR_20 / |ΔP_5| — time to barrier estimate",
        source=FeatureSource.DERIVED,
        transformation="time_to_barrier(atr_20, ret_5)",
        params={},
        dependencies=["atr_20", "ret_5"],
        tags=["multipair", "momentum"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="no_trade_score",
        feature_type=FeatureType.NUMERIC,
        description="No-trade zone score (low vol + neutral OFI + choppy)",
        source=FeatureSource.DERIVED,
        transformation="no_trade_score(low_vol, neutral_ofi, trend_unstable)",
        params={},
        dependencies=["atr", "cross_dispersion"],
        tags=["multipair", "filter"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    # ════════════════════════════════════════════════════════════════════════════
    # MICROSTRUCTURE TOXICITY (Week 2)
    # ═══════════════════════════════════════════════════════════════════════════
    FeatureSpec(
        name="kyles_lambda",
        feature_type=FeatureType.NUMERIC,
        description="Kyle\'s Lambda - price impact per unit volume (20-bar)",
        source=FeatureSource.ORDERBOOK,
        transformation="kyles_lambda(window=20)",
        params={"window": 20},
        dependencies=["close", "volume"],
        tags=["microstructure", "toxicity"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="amihud_illiq",
        feature_type=FeatureType.NUMERIC,
        description="Amihud illiquidity ratio (20-bar)",
        source=FeatureSource.ORDERBOOK,
        transformation="amihud_illiquidity(window=20)",
        params={"window": 20},
        dependencies=["close", "volume"],
        tags=["microstructure", "illiquidity"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="realized_spread",
        feature_type=FeatureType.NUMERIC,
        description="Realized spread proxy from OHLC (10-bar)",
        source=FeatureSource.ORDERBOOK,
        transformation="realized_spread(window=10)",
        params={"window": 10},
        dependencies=["high", "low", "close"],
        tags=["microstructure", "cost"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),

    # ════════════════════════════════════════════════════════════════════════════
    # REGIME QUALITY
    # ═══════════════════════════════════════════════════════════════════════════
    FeatureSpec(
        name="realized_vol_regime",
        feature_type=FeatureType.CATEGORICAL,
        description="Volatility regime: 0=low, 1=normal, 2=high (percentile-based)",
        source=FeatureSource.DERIVED,
        transformation="pd.qcut(vol_60, [0, 0.33, 0.66, 1.0], labels=[0,1,2])",
        params={"window": 60, "quantiles": [0.33, 0.66]},
        dependencies=["vol_60"],
        tags=["regime", "volatility"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
    FeatureSpec(
        name="trend_quality",
        feature_type=FeatureType.NUMERIC,
        description="ADX * sign(RSI_slope) - trend strength x direction consistency",
        source=FeatureSource.DERIVED,
        transformation="adx_14 * (rsi_14 - rsi_14.shift(5))",
        params={},
        dependencies=["adx_14", "rsi_14"],
        tags=["regime", "trend", "quality"],
        materialization=MaterializationStrategy.INCREMENTAL,
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# REGISTRY HELPERS
# ════════════════════════════════════════════════════════════════════════════

class FeatureRegistry:
    """In-memory registry with lookup helpers."""

    def __init__(self, features: list[FeatureSpec] = None):
        self._by_name: dict[str, FeatureSpec] = {}
        self._by_source: dict[FeatureSource, list[FeatureSpec]] = {}
        self._by_tag: dict[str, list[FeatureSpec]] = {}
        for f in (features or BUILTIN_FEATURES):
            self.register(f)

    def register(self, spec: FeatureSpec) -> None:
        self._by_name[spec.name] = spec
        self._by_source.setdefault(spec.source, []).append(spec)
        for tag in spec.tags:
            self._by_tag.setdefault(tag, []).append(spec)

    def get(self, name: str) -> FeatureSpec | None:
        return self._by_name.get(name)

    def get_by_source(self, source: FeatureSource) -> list[FeatureSpec]:
        return self._by_source.get(source, [])

    def get_by_tag(self, tag: str) -> list[FeatureSpec]:
        return self._by_tag.get(tag, [])

    def all(self) -> list[FeatureSpec]:
        return list(self._by_name.values())

    def resolve_dependencies(self, feature_names: list[str]) -> list[str]:
        """Return feature_names plus all transitive dependencies in topological order."""
        visited = set()
        result = []

        def visit(name: str):
            if name in visited:
                return
            spec = self._by_name.get(name)
            if not spec:
                return
            for dep in spec.dependencies:
                visit(dep)
            visited.add(name)
            result.append(name)

        for name in feature_names:
            visit(name)
        return result


# Global singleton registry
REGISTRY = FeatureRegistry(BUILTIN_FEATURES)


def get_feature_spec(name: str) -> FeatureSpec | None:
    """Quick lookup by name."""
    return REGISTRY.get(name)


def list_features(source: FeatureSource = None, tag: str = None) -> list[FeatureSpec]:
    """List features filtered by source and/or tag."""
    if source and tag:
        return [f for f in REGISTRY.get_by_source(source) if tag in f.tags]
    elif source:
        return REGISTRY.get_by_source(source)
    elif tag:
        return REGISTRY.get_by_tag(tag)
    return REGISTRY.all()


if __name__ == "__main__":
    # Demo
    print(f"Total features: {len(BUILTIN_FEATURES)}")
    print("By source:")
    for src in FeatureSource:
        feats = REGISTRY.get_by_source(src)
        if feats:
            print(f"  {src.value}: {len(feats)}")
    print(f"\nDependencies for log_ret_5: {REGISTRY.resolve_dependencies(['log_ret_5'])}")
    print(f"Dependencies for hurst_120: {REGISTRY.resolve_dependencies(['hurst_120'])}")
