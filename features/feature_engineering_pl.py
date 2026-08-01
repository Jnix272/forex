"""
features/feature_engineering.py  (v2 - Polars Edition)
All microstructure + momentum + cross-asset + sentiment features using Polars Expressions.
"""
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

# === Polars Sanitize =======================================================================================-
_SANITIZE_NO_CLIP = frozenset({
    "open", "high", "low", "close", "volume",
    "bid_close", "ask_close", "spread", "spread_pips",
    "expected_latency_ms", "timestamp_utc",
    "cot_net_hf", "cot_net_comm",
})

def sanitize_frame(df: pl.DataFrame, fill_value: float = 0.0, context: str = "frame") -> pl.DataFrame:
    cols = df.select(pl.col(pl.Float32, pl.Float64)).columns
    if not cols:
        return df
    clip_cols = [c for c in cols if c not in _SANITIZE_NO_CLIP]
    noclip_cols = [c for c in cols if c in _SANITIZE_NO_CLIP]
    exprs = [
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
          .then(None).otherwise(pl.col(c))
          .fill_null(fill_value)
          .clip(-1e6, 1e6)
          .alias(c)
        for c in clip_cols
    ] + [
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
          .then(None).otherwise(pl.col(c))
          .fill_null(fill_value)
          .alias(c)
        for c in noclip_cols
    ]
    return df.with_columns(exprs)

# === Microstructure ==========================================================================================
def order_flow_imbalance(window: int = 20) -> pl.Expr:
    d = (pl.col("close") - pl.col("open")).sign()
    bv = pl.col("volume") * d.clip(0, float("inf"))
    sv = pl.col("volume") * (-d).clip(0, float("inf"))
    vol = pl.when(pl.col("volume") == 0).then(None).otherwise(pl.col("volume"))
    return ((bv - sv) / vol).rolling_mean(window).alias("ofi")

def order_book_imbalance_proxy() -> pl.Expr:
    r = pl.col("high") - pl.col("low") + 1e-9
    return ((pl.col("close") - pl.col("low")) / r).clip(0, 1).alias("obi_proxy")

def trade_arrival_rate(window: int = 30) -> list[pl.Expr]:
    # Since we can't do conditional columns easily without knowing schema, we assume volume is present.
    # We will conditionally apply n_ticks later if it exists.
    # For now, base it on volume.
    rate = pl.col("volume").cast(pl.Float64)
    mu = rate.rolling_mean(window)
    s = rate.rolling_std(window) + 1e-9
    return [((rate - mu) / s).alias("tar")]


# ====================================================================================================================================================================================================================================
# ADVANCED MICROSTRUCTURE (Improvement: Kyle's Lambda, Multi-level OBI, VPIN)
# ====================================================================================================================================================================================================================================

def kyles_lambda(window: int = 20) -> pl.Expr:
    """Kyle's Lambda: price impact per unit *signed* volume, in basis points.

    Signed volume is approximated from bar direction (close vs open); the
    regression slope Cov(ΔP, Q_z)/Var(Q_z) with Q_z = standardized signed volume
    (Var=1) gives the scale-invariant price impact, scaled by 1e4 (bps).
    Uses signed (not total) volume so it matches the canonical Kyle's lambda.
    """
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    signed_vol = pl.col("volume").cast(pl.Float64) * (pl.col("close") - pl.col("open")).sign()
    # standardize signed volume -> mean 0, unit variance within window
    sv_z = (signed_vol - signed_vol.rolling_mean(window)) / (signed_vol.rolling_std(window_size=window) + 1e-9)
    # Cov(ret, sv_z) is the regression slope (Var(sv_z)=1 by construction)
    cov = (ret * sv_z).rolling_mean(window) - ret.rolling_mean(window) * sv_z.rolling_mean(window)
    return (cov * 1e4).alias("kyles_lambda")


def amihud_illiquidity(window: int = 20) -> pl.Expr:
    """Amihud Illiquidity Ratio: |ret| per unit *normalized* volume, in basis points.

    Volume is normalized by its rolling mean (scale-invariant across pairs/regimes)
    and the ratio is scaled by 1e4 so the feature sits in an O(0.01-1) range instead
    of the raw ~1e-7 produced by raw tick counts. Higher = more illiquid.
    """
    ret = (pl.col("close") / pl.col("close").shift(1)).log().abs()
    vol = pl.col("volume").cast(pl.Float64)
    vol_n = vol / (vol.rolling_mean(window) + 1e-9)
    return ((ret / (vol_n + 1e-9)).rolling_mean(window) * 1e4).alias("amihud_illiq")


def multi_level_obi(n_levels: int = 5, use_real_l2: bool = True) -> list[pl.Expr]:
    """Multi-level Order Book Imbalance.

    DS-009: When real L2 bid/ask size columns (bid_sz_01..N, ask_sz_01..N) are
    available from Databento MBP-10 data, uses genuine order book depth.
    Falls back to OHLC-simulated proxy when L2 data is not present.

    Args:
        n_levels: number of book levels to compute OBI for
        use_real_l2: if True, attempts to use real L2 columns first
    """
    exprs = []

    if use_real_l2:
        # Attempt to use real L2 columns from Databento
        for i in range(1, n_levels + 1):
            bid_col = f"bid_sz_{i:02d}"
            ask_col = f"ask_sz_{i:02d}"
            real_obi = (
                (pl.col(bid_col) - pl.col(ask_col)) /
                (pl.col(bid_col) + pl.col(ask_col) + 1e-9)
            ).alias(f"obi_l{i}")
            exprs.append(real_obi)

        # Weighted aggregate OBI across all levels
        bid_total = sum(pl.col(f"bid_sz_{i:02d}") for i in range(1, n_levels + 1))
        ask_total = sum(pl.col(f"ask_sz_{i:02d}") for i in range(1, n_levels + 1))
        exprs.append(
            ((bid_total - ask_total) / (bid_total + ask_total + 1e-9)).alias("obi_aggregate")
        )
        return exprs

    # Fallback: OHLC-simulated proxy (when L2 data unavailable)
    for i in range(1, n_levels + 1):
        (pl.col("high") - pl.col("low")) / n_levels
        bid_vol = pl.when(pl.col("close") > pl.col("open")).then(pl.col("volume") / n_levels).otherwise(pl.lit(0))
        ask_vol = pl.when(pl.col("close") < pl.col("open")).then(pl.col("volume") / n_levels).otherwise(pl.lit(0))
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        exprs.append(obi.alias(f"obi_l{i}"))
    return exprs


def vpin(bucket_size: int = 50, n_buckets: int = 50) -> pl.Expr:
    """Volume-Synchronized Probability of Informed Trading (VPIN).
    
    Measures order flow toxicity using volume buckets.
    Requires buy/sell volume classification (approximated from close vs open).
    """
    # Classify buy/sell volume from close vs open
    buy_vol = pl.when(pl.col("close") > pl.col("open")).then(pl.col("volume")).otherwise(0.0)
    sell_vol = pl.when(pl.col("close") < pl.col("open")).then(pl.col("volume")).otherwise(0.0)
    
    # Rolling sums over bucket_size
    buy_bucket = buy_vol.rolling_sum(window_size=bucket_size)
    sell_bucket = sell_vol.rolling_sum(window_size=bucket_size)
    total_bucket = buy_bucket + sell_bucket + 1e-9
    
    # VPIN = rolling mean of |buy - sell| / (buy + sell) over n_buckets
    vpin_val = ((buy_bucket - sell_bucket).abs() / total_bucket).rolling_mean(window_size=n_buckets)
    return vpin_val.alias("vpin")


def realized_spread(window: int = 10) -> pl.Expr:
    """Realized spread proxy via the Corwin-Schultz estimator.

    beta = sqrt( ln(H_t/L_t) * ln(H_{t-1}/L_{t-1}) )   # geometric mean of consecutive ranges
    spread = 2 * (exp(beta) - 1) / (1 + exp(beta))
    Isolates the bid-ask spread from intrabar volatility using two consecutive
    bars. For liquid FX (EURUSD) this yields ~1e-4 (≈1 pip).
    """
    hl = (pl.col("high") / pl.col("low")).log()
    beta = (hl * hl.shift(1)).sqrt()
    spread = 2 * (beta.exp() - 1) / (1 + beta.exp())
    return spread.rolling_mean(window).alias("realized_spread")


# ====================================================================================================================================================================================================================================
# REGIME DETECTION UPGRADE: HMM + CPD (Improvement: Regime Detection)
# ====================================================================================================================================================================================================================================

def hmm_regime_probs(n_states: int = 3, window: int = 60, feature: str = "close") -> list[pl.Expr]:
    """DS-007: Volatility regime bucket (NOT a true HMM).

    Bins rolling volatility into terciles and computes rolling state membership
    probabilities. Despite the legacy name, this is a simple volatility bucket
    classifier — not a Hidden Markov Model.

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
        state = pl.when(vol <= vol_q33).then(0) \
                .when(vol <= vol_q66).then(1) \
                .otherwise(2)
    else:
        # Generic N-way split using equal quantiles
        vol_q33 = vol.rolling_quantile(0.33, window_size=window)
        vol_q66 = vol.rolling_quantile(0.66, window_size=window)
        state = pl.when(vol <= vol_q33).then(0) \
                .when(vol <= vol_q66).then(1) \
                .otherwise(2)
        n_states = 3

    for s in range(n_states):
        prob = (state == s).rolling_mean(window_size=window).alias(f"vol_regime_state_{s}_prob")
        exprs.append(prob)
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


# ====================================================================================================================================================================================================================================
# CIRCUIT BREAKERS / KILL SWITCHES (Improvement: Risk Controls)
# ====================================================================================================================================================================================================================================

def circuit_breaker_features(
    max_position_pct: float = 0.05,
    max_daily_loss_pct: float = 0.03,
    max_drawdown_pct: float = 0.10,
    var_window: int = 252
) -> list[pl.Expr]:
    """Generate circuit breaker / kill switch features.
    
    Returns flags for:
    - Position size limits
    - Daily loss limits  
    - Drawdown limits
    - VaR breach
    """
    # Rolling equity curve approximation from returns
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    equity = (1 + ret).cum_prod()
    
    # Current drawdown
    peak = equity.cum_max()
    drawdown = (equity - peak) / peak
    
    # Daily P&L
    daily_ret = ret.rolling_sum(window_size=1440)  # ~1 day at 1-min bars
    
    # VaR (historical)
    var_95 = ret.rolling_quantile(0.05, window_size=var_window)
    var_99 = ret.rolling_quantile(0.01, window_size=var_window)
    
    return [
        drawdown.alias("drawdown"),
        (drawdown < -max_drawdown_pct).cast(pl.Int32).alias("drawdown_breach"),
        (daily_ret < -max_daily_loss_pct).cast(pl.Int32).alias("daily_loss_breach"),
        (ret < var_95).cast(pl.Int32).alias("var_95_breach"),
        (ret < var_99).cast(pl.Int32).alias("var_99_breach"),
    ]


def position_limit_flags(
    max_pos_pct: float = 0.05,
    atr_col: str = "atr_6",
    pair: str = "EURUSD",
) -> list[pl.Expr]:
    """Position sizing limits based on ATR and account risk.

    PIPE-002: pip_value now computed dynamically per currency pair.
    - USD-quoted pairs (EURUSD, GBPUSD): pip_value = $10/lot (0.0001 × 100,000)
    - JPY-quoted pairs (USDJPY, EURJPY): pip_value ≈ $6.5–9.3/lot (0.01 × 100,000 / rate)
      Approximated as $7.0 (conservative) since exact rate isn't available at feature time.
    """
    pair_upper = str(pair).upper()
    if "JPY" in pair_upper:
        pip_value_est = 7.0  # 100000 * 0.01 / ~143 ≈ $7.0
    elif pair_upper.endswith("USD"):
        pip_value_est = 10.0  # direct USD quote
    elif pair_upper.startswith("USD"):
        pip_value_est = 8.0  # approximate for crosses where USD is base
    else:
        pip_value_est = 9.0  # cross pairs (EURGBP, etc.) — approximate

    risk_budget = pl.lit(max_pos_pct)
    pip_value = pl.lit(pip_value_est)
    pip_mult = 10000.0 if "JPY" not in pair_upper else 100.0

    max_pos = risk_budget / (pl.col(atr_col) * pip_value * pip_mult)

    return [
        max_pos.alias("max_position_lots"),
        (pl.col(atr_col) * pip_value * pip_mult).alias("risk_per_lot"),
    ]


# ====================================================================================================================================================================================================================================
# EMBEDDING FEATURES (Improvement: FinBERT + Vector Storage)
# ====================================================================================================================================================================================================================================

def embedding_placeholders(dim: int = 8) -> list[pl.Expr]:
    """Schema-compatible placeholder columns when no embeddings are available.

    PIPE-003: These should only be used as a last resort. The preferred path is:
    1. Pre-compute FinBERT embeddings via `features/finbert_sentiment.py`
    2. Store in `data/processed/finbert_embeddings.parquet`
    3. Join via asof on timestamp_utc in FeatureEngineer.build()

    The FeatureEngineer.build() method already handles real finbert_embs when
    passed as a parameter. These placeholders only fire when no embeddings exist.
    """
    return [pl.lit(0.0).alias(f"embed_{i}") for i in range(dim)]


def compute_finbert_embeddings(
    news_df: pl.DataFrame,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 64,
    max_length: int = 128,
    output_dim: int = 32,
) -> pl.DataFrame:
    """PIPE-003: Compute real FinBERT sentiment embeddings from news headlines.

    Generates 768-dim embeddings from FinBERT, then reduces to output_dim via PCA.
    Returns DataFrame with timestamp_utc + embed_0..embed_N columns ready for asof join.

    Requires: transformers, torch, sklearn
    """
    import torch

    if news_df is None or len(news_df) == 0:
        return pl.DataFrame()

    headline_col = "headline" if "headline" in news_df.columns else "title"
    if headline_col not in news_df.columns:
        return pl.DataFrame()

    headlines = news_df[headline_col].fill_null("").to_list()
    timestamps = news_df["timestamp_utc"] if "timestamp_utc" in news_df.columns else None

    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        print(f"[FinBERT] Model load failed: {e}. Returning empty embeddings.")
        return pl.DataFrame()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt")
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    import numpy as np
    embeddings = np.vstack(all_embeddings)

    # PCA reduction to output_dim
    if embeddings.shape[1] > output_dim:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=output_dim)
        embeddings = pca.fit_transform(embeddings)

    # Build output DataFrame
    embed_cols = {f"embed_{i}": embeddings[:, i].astype(np.float32) for i in range(embeddings.shape[1])}
    result = pl.DataFrame(embed_cols)
    if timestamps is not None:
        result = result.with_columns(timestamps.alias("timestamp_utc"))

    return result


# ====================================================================================================================================================================================================================================
# FEATURE DISCOVERY / AUTO-GENERATION
# ====================================================================================================================================================================================================================================

def feature_discovery_mutual_info(
    target: str = "ret_5",
    candidates: list[str] = None,
    window: int = 20
) -> pl.Expr:
    """Mutual Information between candidates and target (approximation).
    
    Uses correlation ratio as proxy for MI.
    """
    if candidates is None:
        candidates = ["close", "volume", "rsi_14", "macd", "atr_6"]
    
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    pl.col(target)
    
    exprs = []
    for c in candidates:
        corr = pl.rolling_corr(ret, pl.col(c), window_size=20)
        mi_proxy = (corr**2).alias(f"mi_{c}_vs_{target}")
        exprs.append(mi_proxy)
    return pl.concat_list(exprs).alias("feature_mi_scores")


# ====================================================================================================================================================================================================================================
# TENSORRT / ONNX EXPORT HELPERS
# ====================================================================================================================================================================================================================================

def onnx_compatible_features() -> list[str]:
    """Return list of feature names compatible with ONNX export.

    These names MUST match the exact aliases produced by HAELTFeatureBuilder.build().
    Excludes features that require dynamic shapes or unsupported ops.
    """
    # These are the core features that export cleanly to ONNX
    return [
        "close", "open", "high", "low", "volume",
        # lag_returns() produces ret_{w} — NOT log_ret_{w}
        "ret_5", "ret_20", "ret_60",
        "atr_6", "atr_20",
        # rolling_volatility(20) produces vol_20 — NOT rolling_vol_20
        "vol_20",
        "rsi_14", "macd", "macd_sig", "macd_hist",
        "bb_pct", "bb_width",
        "stoch_k", "stoch_d", "williams_r", "cci",
        "ofi", "obi_proxy", "vpin",
        "trend_regime", "range_regime", "volatility_regime",
        # HAELTFeatureBuilder.build() produces time_sin/time_cos/day_sin/day_cos
        # NOT hour_sin/hour_cos/dow_sin/dow_cos
        "time_sin", "time_cos", "day_sin", "day_cos",
    ]


def quantize_aware_features() -> list[str]:
    """Features suitable for INT8 quantization (bounded range)."""
    return [
        "stoch_k", "stoch_d",  # 0-100
        "rsi_14",  # 0-100
        "williams_r",  # -100 to 0
        "cci",  # typically -300 to 300
        "bb_pct",  # 0-1
        # HAELTFeatureBuilder produces time_sin/time_cos/day_sin/day_cos — NOT hour_*/dow_*
        "time_sin", "time_cos",  # -1 to 1
        "day_sin", "day_cos",   # -1 to 1
    ]


# ====================================================================================================================================================================================================================================
# AUTOMATED MODEL CARDS (Improvement: Governance)
# ====================================================================================================================================================================================================================================

def model_card_template(
    model_name: str,
    version: str,
    features: list[str],
    metrics: dict,
    data_window: str,
    known_limitations: list[str] = None
) -> dict:
    """Generate standardized model card for auditability.
    
    Returns dict that can be saved as JSON alongside model.
    """
    from datetime import datetime, timezone, timedelta
    _now = datetime.now(timezone.utc)
    return {
        "model_name": model_name,
        "version": version,
        "created_at": _now.isoformat(),
        "architecture": "HAELT" if "haelt" in model_name.lower() else "UNKNOWN",
        "features": {
            "count": len(features),
            "names": features,
            "embedding_dims": sum(1 for f in features if f.startswith("embed_")),
        },
        "data": {
            "training_window": data_window,
            "pairs": ["EURUSD", "USDJPY", "GBPUSD"],  # from config
            "frequency": "1min",
        },
        "metrics": metrics,
        "known_limitations": known_limitations or [
            "Trained on 1-min data; may not generalize to other timeframes",
            "Regime-dependent performance; monitor regime features",
            "No on-chain/satellite data in current version",
        ],
        "governance": {
            "approved_for_production": False,
            "review_date": (_now + timedelta(days=91)).isoformat(),  # ~3 months ahead (safe across year-end)
        },
    }


# ====================================================================================================================================================================================================================================
# POPULATION-BASED HPO (Improvement: Better Optuna)
# ====================================================================================================================================================================================================================================

def population_based_hpo_config() -> dict:
    """Configuration for Population-Based Training (PBT) style HPO.
    
    Returns config dict for Optuna/ray tune integration.
    """
    return {
        "population_size": 10,
        "perturbation_interval": 10,  # epochs
        "hyperparameters": {
            "lr": {"distribution": "loguniform", "low": 1e-5, "high": 1e-2},
            "batch_size": {"choices": [64, 128, 256, 512]},
            "dropout": {"distribution": "uniform", "low": 0.1, "high": 0.5},
            "hidden_size": {"choices": [128, 256, 512]},
            "num_layers": {"choices": [2, 3, 4]},
            "seq_len": {"choices": [60, 80, 100, 120]},
        },
        "perturbation_factors": {
            "lr": {"distribution": "loguniform", "low": 0.8, "high": 1.25},
            "dropout": {"distribution": "uniform", "low": -0.05, "high": 0.05},
        },
        "selection": "top_20_percent",
        "resample_from": "best",
    }


# === ATR-6 + Volatility ====================================================================================
def average_true_range(window: int = 6) -> pl.Expr:
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal([
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs()
    ])
    return tr.rolling_mean(window).alias(f"atr_{window}")

def rolling_volatility(window: int = 20) -> pl.Expr:
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    return ret.rolling_std(window).alias(f"vol_{window}")

def bollinger_bands(window: int = 20, n_std: float = 2.0) -> list[pl.Expr]:
    mid = pl.col("close").rolling_mean(window)
    std = pl.col("close").rolling_std(window)
    up = mid + n_std * std
    lo = mid - n_std * std
    width = (up - lo) / (mid + 1e-9)
    pct = ((pl.col("close") - lo) / (up - lo + 1e-9)).clip(0, 1)
    return [
        mid.alias("bb_mid"),
        up.alias("bb_upper"),
        lo.alias("bb_lower"),
        width.alias("bb_width"),
        pct.alias("bb_pct")
    ]

# === Momentum ================================================================================================-
def rsi(period: int = 14) -> pl.Expr:
    d = pl.col("close").diff()
    g = d.clip(0, float("inf")).rolling_mean(period)
    l = (-d.clip(float("-inf"), 0)).rolling_mean(period) + 1e-9
    return (100 - 100 / (1 + g / l)).alias(f"rsi_{period}")

def macd(fast: int = 12, slow: int = 26, signal: int = 9) -> list[pl.Expr]:
    ef = pl.col("close").ewm_mean(span=fast, adjust=False)
    es = pl.col("close").ewm_mean(span=slow, adjust=False)
    line = ef - es
    sig = line.ewm_mean(span=signal, adjust=False)
    return [
        line.alias("macd"),
        sig.alias("macd_sig"),
        (line - sig).alias("macd_hist")
    ]

def lag_returns(windows: list[int] = [5, 20, 60]) -> list[pl.Expr]:
    lp = pl.col("close").log()
    return [(lp - lp.shift(w)).alias(f"ret_{w}") for w in windows]


# ====================================================================================================================================================================================================================================
# CLASSICAL INDICATORS (Improvement #2)
# ====================================================================================================================================================================================================================================

def stochastic_oscillator(k_period: int = 14, d_period: int = 3) -> list[pl.Expr]:
    """Stochastic Oscillator %K and %D."""
    lowest = pl.col("low").rolling_min(k_period)
    highest = pl.col("high").rolling_max(k_period)
    k = 100 * (pl.col("close") - lowest) / (highest - lowest + 1e-9)
    d = k.rolling_mean(d_period)
    return [k.alias("stoch_k"), d.alias("stoch_d")]


def williams_r(period: int = 14) -> pl.Expr:
    """Williams %R - momentum oscillator similar to Stochastic but inverted."""
    highest = pl.col("high").rolling_max(period)
    lowest = pl.col("low").rolling_min(period)
    return ((highest - pl.col("close")) / (highest - lowest + 1e-9) * -100).alias("williams_r")


def cci(period: int = 20) -> pl.Expr:
    """Commodity Channel Index - identifies cyclical turns."""
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    ma = tp.rolling_mean(period)
    md = (tp - ma).abs().rolling_mean(period) + 1e-9
    return ((tp - ma) / (0.015 * md)).alias("cci")


# ====================================================================================================================================================================================================================================
# CANDLESTICK PATTERNS (Improvement: Pattern Recognition)
# ====================================================================================================================================================================================================================================

def candle_body() -> pl.Expr:
    """Candle body size (absolute close - open)."""
    return (pl.col("close") - pl.col("open")).abs().alias("candle_body")

def candle_range() -> pl.Expr:
    """Full candle range (high - low)."""
    return (pl.col("high") - pl.col("low")).alias("candle_range")

def upper_shadow() -> pl.Expr:
    """Upper shadow length."""
    return (pl.col("high") - pl.max_horizontal("open", "close")).alias("upper_shadow")

def lower_shadow() -> pl.Expr:
    """Lower shadow length."""
    return (pl.min_horizontal("open", "close") - pl.col("low")).alias("lower_shadow")

def is_doji(threshold: float = 0.1) -> pl.Expr:
    """Doji: body <= threshold * range. Threshold is Optuna-tunable."""
    body = (pl.col("close") - pl.col("open")).abs()
    rng = pl.col("high") - pl.col("low") + 1e-9
    return (body / rng <= threshold).cast(pl.Int32).alias("is_doji")

def is_hammer(body_ratio: float = 0.3, lower_ratio: float = 0.6, upper_ratio: float = 0.1) -> pl.Expr:
    """Hammer: small body, long lower shadow, little upper shadow.

    PIPE-008: thresholds are now parameters (Optuna-tunable) instead of magic numbers.
    """
    body = (pl.col("close") - pl.col("open")).abs()
    rng = pl.col("high") - pl.col("low") + 1e-9
    lower = pl.min_horizontal("open", "close") - pl.col("low")
    upper = pl.col("high") - pl.max_horizontal("open", "close")
    cond = (body / rng <= body_ratio) & (lower / rng >= lower_ratio) & (upper / rng <= upper_ratio)
    return cond.cast(pl.Int32).alias("is_hammer")

def is_shooting_star() -> pl.Expr:
    """Shooting star: small body, long upper shadow, little lower shadow."""
    body = (pl.col("close") - pl.col("open")).abs()
    rng = pl.col("high") - pl.col("low") + 1e-9
    lower = pl.min_horizontal("open", "close") - pl.col("low")
    upper = pl.col("high") - pl.max_horizontal("open", "close")
    cond = (body / rng <= 0.3) & (upper / rng >= 0.6) & (lower / rng <= 0.1)
    return cond.cast(pl.Int32).alias("is_shooting_star")

def is_engulfing() -> pl.Expr:
    """Bullish/Bearish Engulfing: current body engulfs previous body."""
    prev_open = pl.col("open").shift(1)
    prev_close = pl.col("close").shift(1)
    (prev_close - prev_open).abs()
    (pl.col("close") - pl.col("open")).abs()
    bullish = (prev_close < prev_open) & (pl.col("close") > pl.col("open")) & \
              (pl.col("open") <= prev_close) & (pl.col("close") >= prev_open)
    bearish = (prev_close > prev_open) & (pl.col("close") < pl.col("open")) & \
              (pl.col("open") >= prev_close) & (pl.col("close") <= prev_open)
    return (bullish.cast(pl.Int32) - bearish.cast(pl.Int32)).alias("engulfing")

def is_harami() -> pl.Expr:
    """Harami: current body inside previous body."""
    prev_open = pl.col("open").shift(1)
    prev_close = pl.col("close").shift(1)
    bullish = (prev_close < prev_open) & (pl.col("close") > pl.col("open")) & \
              (pl.col("open") > prev_close) & (pl.col("close") < prev_open)
    bearish = (prev_close > prev_open) & (pl.col("close") < pl.col("open")) & \
              (pl.col("open") < prev_close) & (pl.col("close") > prev_open)
    return (bullish.cast(pl.Int32) - bearish.cast(pl.Int32)).alias("harami")

def is_morning_star() -> pl.Expr:
    """Morning Star: 3-candle bullish reversal."""
    c1_open = pl.col("open").shift(2)
    c1_close = pl.col("close").shift(2)
    c2_open = pl.col("open").shift(1)
    c2_close = pl.col("close").shift(1)
    c3_open = pl.col("open")
    c3_close = pl.col("close")
    
    c1_bearish = c1_close < c1_open
    c2_small = (c2_close - c2_open).abs() < (c1_open - c1_close).abs() * 0.3
    c2_gap_down = pl.min_horizontal(c2_open, c2_close) < c1_close
    c3_bullish = c3_close > c3_open
    c3_gap_up = pl.min_horizontal(c3_open, c3_close) > c2_close
    c3_closes_above = c3_close > (c1_open + c1_close) / 2
    
    cond = c1_bearish & c2_small & c2_gap_down & c3_bullish & c3_gap_up & c3_closes_above
    return cond.cast(pl.Int32).alias("morning_star")

def is_evening_star() -> pl.Expr:
    """Evening Star: 3-candle bearish reversal."""
    c1_open = pl.col("open").shift(2)
    c1_close = pl.col("close").shift(2)
    c2_open = pl.col("open").shift(1)
    c2_close = pl.col("close").shift(1)
    c3_open = pl.col("open")
    c3_close = pl.col("close")
    
    c1_bullish = c1_close > c1_open
    c2_small = (c2_close - c2_open).abs() < (c1_close - c1_open).abs() * 0.3
    c2_gap_up = pl.max_horizontal(c2_open, c2_close) > c1_close
    c3_bearish = c3_close < c3_open
    c3_gap_down = pl.max_horizontal(c3_open, c3_close) < c2_close
    c3_closes_below = c3_close < (c1_open + c1_close) / 2
    
    cond = c1_bullish & c2_small & c2_gap_up & c3_bearish & c3_gap_down & c3_closes_below
    return cond.cast(pl.Int32).alias("evening_star")

def is_three_white_soldiers() -> pl.Expr:
    """Three White Soldiers: 3 consecutive bullish candles with higher closes."""
    c1_bullish = pl.col("close").shift(2) > pl.col("open").shift(2)
    c2_bullish = pl.col("close").shift(1) > pl.col("open").shift(1)
    c3_bullish = pl.col("close") > pl.col("open")
    higher_closes = (pl.col("close") > pl.col("close").shift(1)) & \
                    (pl.col("close").shift(1) > pl.col("close").shift(2))
    cond = c1_bullish & c2_bullish & c3_bullish & higher_closes
    return cond.cast(pl.Int32).alias("three_white_soldiers")

def is_three_black_crows() -> pl.Expr:
    """Three Black Crows: 3 consecutive bearish candles with lower closes."""
    c1_bearish = pl.col("close").shift(2) < pl.col("open").shift(2)
    c2_bearish = pl.col("close").shift(1) < pl.col("open").shift(1)
    c3_bearish = pl.col("close") < pl.col("open")
    lower_closes = (pl.col("close") < pl.col("close").shift(1)) & \
                   (pl.col("close").shift(1) < pl.col("close").shift(2))
    cond = c1_bearish & c2_bearish & c3_bearish & lower_closes
    return cond.cast(pl.Int32).alias("three_black_crows")

def all_candlestick_patterns() -> list[pl.Expr]:
    """Return all candlestick pattern expressions."""
    return [
        candle_body(),
        candle_range(),
        upper_shadow(),
        lower_shadow(),
        is_doji(),
        is_hammer(),
        is_shooting_star(),
        is_engulfing(),
        is_harami(),
        is_morning_star(),
        is_evening_star(),
        is_three_white_soldiers(),
        is_three_black_crows(),
    ]


# ====================================================================================================================================================================================================================================
# VOLUME-WEIGHTED FEATURES (Improvement #4)
# ====================================================================================================================================================================================================================================

def vwap_bands(window: int = 60, n_std: float = 2.0) -> list[pl.Expr]:
    """VWAP with standard deviation bands."""
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    vol = pl.when(pl.col("volume") == 0).then(1.0).otherwise(pl.col("volume"))
    vwap = (tp * vol).rolling_sum(window) / vol.rolling_sum(window)
    # VWAP standard deviation (volume-weighted)
    vwap_var = ((tp - vwap)**2 * vol).rolling_sum(window) / (vol.rolling_sum(window) + 1e-9)
    vwap_std = vwap_var.sqrt()
    return [
        vwap.alias("vwap"),
        (vwap + n_std * vwap_std).alias("vwap_upper"),
        (vwap - n_std * vwap_std).alias("vwap_lower"),
        ((pl.col("close") - vwap) / (vwap_std + 1e-9)).alias("vwap_zscore"),
    ]


def volume_weighted_momentum(window: int = 20) -> pl.Expr:
    """Volume-weighted moving average of returns."""
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    vwma = (ret * pl.col("volume")).rolling_sum(window) / pl.col("volume").rolling_sum(window)
    return vwma.alias("vwma_ret")


# ====================================================================================================================================================================================================================================
# REGIME-GATED / INTERACTION FEATURES (Improvements #3, #5)
# ====================================================================================================================================================================================================================================

def regime_gated_features(existing_cols: set = None) -> list[pl.Expr]:
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


def interaction_features(existing_cols: set = None) -> list[pl.Expr]:
    """Explicit non-linear feature interactions for linear/weak non-linear models.
    
    Args:
        existing_cols: Set of column names that exist in the DataFrame.
    """
    if existing_cols is None:
        existing_cols = set()
    
    exprs = []
    
    # Volatility    Flow
    if "atr_6" in existing_cols and "ofi_z" in existing_cols:
        exprs.append((pl.col("atr_6") * pl.col("ofi_z")).alias("atr_x_ofi"))
    if "atr_ratio_6_20" in existing_cols and "ofi_z" in existing_cols:
        exprs.append((pl.col("atr_ratio_6_20") * pl.col("ofi_z")).alias("atr_ratio_x_ofi"))
    
    # Momentum    Regime
    if "rsi_14" in existing_cols and "trend_regime" in existing_cols:
        exprs.append((pl.col("rsi_14") * pl.col("trend_regime")).alias("rsi_x_trend"))
    if "macd" in existing_cols and "trend_regime" in existing_cols:
        exprs.append((pl.col("macd") * pl.col("trend_regime")).alias("macd_x_trend"))
    if "stoch_k" in existing_cols and "range_regime" in existing_cols:
        exprs.append((pl.col("stoch_k") * pl.col("range_regime")).alias("stoch_x_range"))
    
    # Spread / Cost    Volatility
    if "bb_pct" in existing_cols and "range_regime" in existing_cols:
        exprs.append((pl.col("bb_pct") * pl.col("range_regime")).alias("bb_x_range"))
    if "cost_to_atr" in existing_cols and "volatility_regime" in existing_cols:
        exprs.append((pl.col("cost_to_atr") * pl.col("volatility_regime")).alias("cost_x_volatile"))
    
    # Macro    Risk
    if "yield_curve_slope" in existing_cols and "risk_off_signal" in existing_cols:
        exprs.append((pl.col("yield_curve_slope") * pl.col("risk_off_signal")).alias("curve_x_risk"))
    if "carry_eur" in existing_cols and "trend_regime" in existing_cols:
        exprs.append((pl.col("carry_eur") * pl.col("trend_regime")).alias("carry_x_trend"))
    
    # Cross-asset    Risk
    if "gold_dxy_corr" in existing_cols and "risk_off_signal" in existing_cols:
        exprs.append((pl.col("gold_dxy_corr") * pl.col("risk_off_signal")).alias("gold_dxy_x_risk"))
    
    return exprs


# ====================================================================================================================================================================================================================================
# FEATURE QUALITY REPORT (Improvement #1)
# ====================================================================================================================================================================================================================================

def compute_quality_report(df: pl.DataFrame) -> dict:
    """Compute per-feature quality metrics for monitoring.
    
    Returns dict mapping feature_name -> quality_metrics
    """
    from datetime import datetime
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "features": {},
    }
    
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    for col in numeric_cols:
        s = df[col]
        n_null = s.null_count()
        n_total = len(s)
        vals = s.drop_nulls()
        
        if len(vals) == 0:
            report["features"][col] = {
                "null_pct": 100.0, "constant": True, "dtype": str(s.dtype),
            }
            continue
        
        # Check for constant
        is_const = vals.n_unique() == 1
        
        # Basic stats
        mean_v = float(vals.mean())
        std_v = float(vals.std())
        skew_v = float(vals.skew()) if len(vals) > 2 else 0.0
        kurt_v = float(vals.kurtosis()) if len(vals) > 3 else 0.0
        
        # Extreme values
        inf_count = int(s.is_infinite().sum())
        
        report["features"][col] = {
            "dtype": str(s.dtype),
            "null_count": int(n_null),
            "null_pct": float(n_null / n_total * 100),
            "constant": bool(is_const),
            "n_unique": int(vals.n_unique()),
            "mean": mean_v,
            "std": std_v,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "skew": skew_v,
            "kurtosis": kurt_v,
            "inf_count": inf_count,
        }
    
    return report


# === Cross-Asset =============================================================================================-
class CrossAssetFeatures:

    def __init__(self, corr_window=60, regime_window=240, lags=(1, 5, 15)):
        self.cw = corr_window
        self.rw = regime_window
        self.lags = tuple(lags)

    def build(self, bars: pl.DataFrame, data: dict = None) -> pl.DataFrame:
        len(bars)
        synthetic = self._synthetic(bars)
        
        merged = dict(synthetic)
        if data is not None:
            for k, v in data.items():
                if "timestamp_utc" in v.columns:
                    v = v.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
                merged[k] = v
        
        # Start with bars
        F = bars
        
        # Calculate forex return
        F = F.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("forex_ret")
        ])
        
        # Join assets
        for asset, s_df in merged.items():
            # Rename value column to the asset name
            if "value" in s_df.columns:
                s_df = s_df.rename({"value": asset})
            elif s_df.columns[-1] != "timestamp_utc":
                s_df = s_df.rename({s_df.columns[-1]: asset})
                
            s_df = s_df.select(["timestamp_utc", asset]).sort("timestamp_utc")
            F = F.join_asof(s_df, on="timestamp_utc", strategy="backward")
            # forward fill
            F = F.with_columns([pl.col(asset).fill_null(strategy="forward")])
            
            # Returns
            lr = (pl.col(asset).log() - pl.col(asset).shift(1).log())
            exprs = [lr.alias(f"{asset}_ret")]
            for lag in self.lags:
                exprs.append(lr.shift(lag).alias(f"{asset}_ret_l{lag}"))
                
            # rolling corr and beta
            cov = pl.rolling_cov(lr, pl.col("forex_ret"), window_size=self.cw, min_periods=max(5, self.cw//4))
            var_x = pl.col("forex_ret").rolling_var(window_size=self.cw, min_periods=max(5, self.cw//4))
            var_y = lr.rolling_var(window_size=self.cw, min_periods=max(5, self.cw//4))
            
            corr = cov / (var_x.sqrt() * var_y.sqrt() + 1e-9)
            beta = cov / (var_x + 1e-9)
            
            exprs.append(corr.alias(f"{asset}_corr"))
            exprs.append(beta.alias(f"{asset}_beta"))
            F = F.with_columns(exprs)

        cols = F.columns
        exprs = []
        if "US10Y" in cols and "DE10Y" in cols:
            exprs.extend([
                (pl.col("US10Y") - pl.col("DE10Y")).alias("yield_spread_us_de_10y"),
                (pl.col("US10Y") - pl.col("DE10Y")).diff().alias("yield_spread_us_de_10y_chg")
            ])
        if "US10Y" in cols and "US2Y" in cols:
            exprs.extend([
                (pl.col("US10Y") - pl.col("US2Y")).alias("us_2s10s_spread"),
                (pl.col("US10Y") - pl.col("US2Y")).diff().alias("us_2s10s_spread_chg")
            ])
        if "SPX" in cols and "VIX" in cols:
            spx_ret = pl.col("SPX").log().diff()
            vix_ret = pl.col("VIX").log().diff()
            exprs.append(
                (((spx_ret < 0).cast(pl.Float64) + (vix_ret > 0).cast(pl.Float64)) / 2.0).alias("risk_off_signal")
            )
        if "GOLD_ret" in cols and "DXY_ret" in cols:
            gd_cov = pl.rolling_cov(pl.col("GOLD_ret"), pl.col("DXY_ret"), window_size=self.cw)
            gd_vx = pl.col("GOLD_ret").rolling_var(window_size=self.cw)
            gd_vy = pl.col("DXY_ret").rolling_var(window_size=self.cw)
            gd_corr = gd_cov / (gd_vx.sqrt() * gd_vy.sqrt() + 1e-9)
            gd_corr_base = gd_corr.rolling_mean(self.rw)
            exprs.extend([
                gd_corr.alias("gold_dxy_corr"),
                (gd_corr - gd_corr_base).abs().alias("gold_dxy_corr_break")
            ])
        if "WTI_ret" in cols and "COPPER_ret" in cols:
            exprs.append(
                (0.6 * pl.col("COPPER_ret").shift(1) + 0.4 * pl.col("WTI_ret").shift(1)).alias("commodity_fx_lead")
            )
        
        if exprs:
            F = F.with_columns(exprs)
            
        return F.fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.0)

    def _synthetic(self, bars: pl.DataFrame) -> dict:
        # Deprecated: synthetic data generation replaced by real external assets.
        # Return an empty dict so that only provided real data is merged.
        return {}

# === Sentiment ================================================================================================-
def sentiment_decay(s_df: pl.DataFrame, lam: float = 0.1) -> pl.Series:
    if s_df is None or len(s_df) == 0:
        return np.zeros(0, dtype=float)
    import pandas as pd
    ts_ns = pd.to_datetime(s_df["timestamp_utc"].to_numpy(), utc=True).asi8.astype(np.int64)
    vals = np.nan_to_num(s_df["sentiment"].to_numpy().astype(float), nan=0.0)
    event_mask = vals != 0.0
    last_idx = np.maximum.accumulate(np.where(event_mask, np.arange(len(vals)), -1))
    dec = np.zeros(len(vals), dtype=float)
    valid = last_idx >= 0
    if valid.any():
        elapsed = (ts_ns[valid] - ts_ns[last_idx[valid]]) / 1e9
        dec[valid] = vals[last_idx[valid]] * np.exp(-float(lam) * elapsed)
    return dec

def buzz_score(window: int = 5) -> pl.Expr:
    return pl.col("article_counts").rolling_sum(window).fill_null(0.0).alias("buzz")

def proj_finbert(emb, dim=8):
    rng = np.random.default_rng(0)
    P = rng.standard_normal((768, dim)).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-9
    e = emb.reshape(1, -1) if emb.ndim == 1 else emb
    return (e @ P).squeeze()

# === Filters ===================================================================================================-
def vol_filter(atr_col: str = "atr_6", mult: float = 3.0, lb: int = 60) -> pl.Expr:
    return (pl.col(atr_col) <= mult * pl.col(atr_col).rolling_mean(lb)).cast(pl.Float64).alias("vol_ok")

# === L2 Order   Book Imbalance =================================================================================
def order_book_imbalance_l2(bid1: str = "bid_price_1", ask1: str = "ask_price_1", bid2: str = "bid_price_2", ask2: str = "ask_price_2", vol1: str = "bid_size_1", vol2: str = "ask_size_1") -> pl.Expr:
    """Compute a depth   2 order   book imbalance proxy.
    Uses volume   weighted price differences across the top two levels.
    """
    imbalance1 = (pl.col(bid1) * pl.col(vol1) - pl.col(ask1) * pl.col(vol2)) / (pl.col(vol1) + pl.col(vol2) + 1e-9)
    imbalance2 = (pl.col(bid2) * pl.col(vol1) - pl.col(ask2) * pl.col(vol2)) / (pl.col(vol1) + pl.col(vol2) + 1e-9)
    return (imbalance1 + imbalance2).rolling_mean(20).alias("ofi_l2")

# === Carry Features =============================================================================================-
def carry_features(spot: str = "close", forward: str = "forward_price", window: int = 30) -> pl.Expr:
    """Spot   forward carry as log ratio of forward over spot, smoothed.
    Returns a rolling mean of the carry.
    """
    carry = (pl.col(forward) / pl.col(spot)).log()
    return carry.rolling_mean(window).alias("carry_spot_forward")

# === Yield Curve Slope ==========================================================================================-
def yield_curve_slope(short_yield: str = "US2Y", long_yield: str = "US10Y") -> pl.Expr:
    """Simple slope = long   term yield minus short   term yield."""
    return (pl.col(long_yield) - pl.col(short_yield)).alias("yield_curve_slope")

# === Correlation Breakdown Detector =====================================================================-
def correlation_breakdown(a: str, b: str, window: int = 60) -> pl.Expr:
    """Rolling correlation delta between two series.
    Returns the absolute change of correlation over the window.
    """
    corr = pl.rolling_corr(pl.col(a), pl.col(b), window_size=window)
    delta = (corr - corr.rolling_mean(window)).abs()
    return delta.alias(f"corr_break_{a}_{b}")

# === Sentiment Tiers =============================================================================================-
def sentiment_tiers(df: pl.DataFrame, decay_lam: float = 0.1, fb_dim: int = 8) -> pl.DataFrame:
    """Add three sentiment columns:
    1. raw sentiment (if present)
    2. decayed sentiment (exponential decay)
    3. FinBERT projection to low   dim space.
    """
    # raw sentiment assumed in column "sentiment"
    if "sentiment" in df.columns:
        df = df.with_columns([pl.col("sentiment").alias("sentiment_raw")])
        df = df.with_columns(pl.Series("sentiment_decayed", sentiment_decay(df, decay_lam)))
    else:
        df = df.with_columns([pl.lit(0.0).alias("sentiment_raw"), pl.lit(0.0).alias("sentiment_decayed")])
    # FinBERT projection     placeholder zeros if embeddings not provided
    for i in range(fb_dim):
        df = df.with_columns([pl.lit(0.0).alias(f"fb_{i}")])
    return df

def news_filter_expr(news_events: list, buf_min: int = 15) -> list[pl.Expr]:
    # Polars implementation using list matching or just looping if list is small.
    # Actually, simpler to do via joining an events dataframe.
    # This will be implemented directly in the FeatureEngineer class.
    pass

# === Regime Gating ==========================================================================================-
class RegimeGateClassifier:
    def __init__(self, min_samples: int = 80, random_state: int = 42):
        self.min_samples = int(min_samples)
        self.random_state = int(random_state)
        self.model = LogisticRegression(max_iter=500, random_state=self.random_state)

    def _zscore(self, col: str, lb: int = 60) -> pl.Expr:
        mu = pl.col(col).rolling_mean(lb, min_periods=max(10, lb // 4))
        sd = pl.col(col).rolling_std(lb, min_periods=max(10, lb // 4)) + 1e-9
        return ((pl.col(col) - mu) / sd).clip(-6, 6).alias(f"{col}_z")

    def fit_predict(self, F: pl.DataFrame) -> pl.Series:
        req = ["gold_dxy_corr_break", "us_2s10s_spread_chg", "yield_spread_us_de_10y_chg", "risk_off_signal"]
        if not all(c in F.columns for c in req):
            return pl.Series("regime_break_prob", np.zeros(len(F)))

        X_df = F.select([
            self._zscore("gold_dxy_corr_break").alias("gold_break_z"),
            self._zscore("us_2s10s_spread_chg").alias("curve_chg_z"),
            self._zscore("yield_spread_us_de_10y_chg").alias("yield_chg_z"),
            self._zscore("risk_off_signal", lb=30).alias("risk_off_z")
        ])
        
        y_expr = (
            (pl.col("gold_break_z") > 1.0) & 
            ((pl.col("risk_off_z") > 0.5) | (pl.col("curve_chg_z").abs() > 1.0))
        ).cast(pl.Int32)
        
        y = X_df.select(y_expr.alias("y"))["y"].to_numpy()
        X_pd = X_df.to_pandas()
        ok = X_pd.notna().all(axis=1)
        
        if ok.sum() < self.min_samples or len(np.unique(y[ok])) < 2:
            score = (0.8 * X_pd["gold_break_z"].fillna(0) +
                     0.5 * X_pd["risk_off_z"].fillna(0) +
                     0.3 * X_pd["curve_chg_z"].abs().fillna(0))
            prob = 1.0 / (1.0 + np.exp(-score.clip(-8, 8)))
            return pl.Series("regime_break_prob", prob)

        ok_idx = np.where(ok)[0]
        fit_end = max(self.min_samples, int(len(ok_idx) * 0.70))
        fit_idx = ok_idx[:fit_end]
        
        if len(np.unique(y[fit_idx])) < 2:
            score = (0.8 * X_pd["gold_break_z"].fillna(0) +
                     0.5 * X_pd["risk_off_z"].fillna(0) +
                     0.3 * X_pd["curve_chg_z"].abs().fillna(0))
            prob = 1.0 / (1.0 + np.exp(-score.clip(-8, 8)))
            return pl.Series("regime_break_prob", prob)

        self.model.fit(X_pd.iloc[fit_idx].values, y[fit_idx])
        prob = np.zeros(len(F), dtype=np.float64)
        prob[ok] = self.model.predict_proba(X_pd.loc[ok].values)[:, 1]
        return pl.Series("regime_break_prob", prob)

# ==================================================================================================================-
# MULTI-SCALE FEATURES  (short / medium / long triples)
# ==================================================================================================================-
def multi_scale_atr(windows=(6, 20, 60)) -> list[pl.Expr]:
    exprs = []
    prev_w = None
    for w in windows:
        exprs.append(average_true_range(w))
        if prev_w is not None:
            exprs.append((pl.col(f"atr_{prev_w}") / (pl.col(f"atr_{w}") + 1e-9)).clip(0.1, 10.0).alias(f"atr_ratio_{prev_w}_{w}"))
        prev_w = w
    return exprs

def multi_scale_vol(windows=(6, 20, 60)) -> list[pl.Expr]:
    exprs = []
    prev_w = None
    for w in windows:
        exprs.append(rolling_volatility(w))
        if prev_w is not None:
            exprs.append((pl.col(f"vol_{prev_w}") / (pl.col(f"vol_{w}") + 1e-9)).clip(0.1, 10.0).alias(f"vol_ratio_{prev_w}_{w}"))
        prev_w = w
    return exprs

# ==================================================================================================================-
# B4. OFI SURPRISE       abnormal pressure vs. background
# ==================================================================================================================-
def ofi_surprise(fast=20, slow=120) -> list[pl.Expr]:
    return [
        order_flow_imbalance(fast),
        ((pl.col("ofi") - pl.col("ofi").rolling_mean(slow)) / (pl.col("ofi").rolling_std(slow) + 1e-9)).alias("ofi_z")
    ]

# ==================================================================================================================-
# B5. BREAKOUT PRESSURE       how extended price is vs. VWAP
# ==================================================================================================================-
def breakout_pressure(vwap_window=60, atr_window=20) -> list[pl.Expr]:
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    vol = pl.when(pl.col("volume") == 0).then(1.0).otherwise(pl.col("volume")).fill_null(1.0)
    vwap = (tp * vol).rolling_sum(vwap_window) / vol.rolling_sum(vwap_window)
    return [
        average_true_range(atr_window),
        ((pl.col("close") - vwap) / (pl.col(f"atr_{atr_window}") + 1e-9)).alias("breakout_pressure")
    ]

# ==================================================================================================================-
# B6. LIQUIDITY VACUUM       spread spike vs. median
# ==================================================================================================================-
def liquidity_vacuum(window=120) -> pl.Expr:
    spread = (pl.col("ask_close") - pl.col("bid_close")).abs().fill_null(0.0001)
    median_spread = spread.rolling_median(window) + 1e-9
    return (spread / median_spread).alias("liquidity_vacuum")

# ==================================================================================================================-
# FRAGILITY INDICATORS
# ==================================================================================================================-
def vol_of_vol(vol_window=20, vov_window=20) -> pl.Expr:
    return pl.col(f"vol_{vol_window}").rolling_std(vov_window).alias("vol_of_vol")

def price_ofi_divergence(ofi_window=20, price_window=10) -> pl.Expr:
    price_dir = pl.col("close").diff(price_window).sign()
    ofi_dir = pl.col("ofi").sign() # Assuming ofi is already calculated
    return (price_dir - ofi_dir).abs().rolling_mean(price_window).alias("price_ofi_div")

def _pip_size_for_pair(pair: str = "EURUSD") -> float:
    pair = str(pair or "").upper()
    return 0.01 if "JPY" in pair else 0.0001

def add_spread_cost_features(df: pl.DataFrame, pair: str = "EURUSD", atr_col: str = "atr_6") -> pl.DataFrame:
    """Add execution-cost context from trailing spread behavior."""
    if "spread_pips" not in df.columns:
        return df
    import pandas as pd

    spread = pd.Series(df["spread_pips"].to_numpy(), dtype="float64")
    mu = spread.rolling(240, min_periods=20).mean()
    sd = spread.rolling(240, min_periods=20).std().replace(0.0, np.nan)
    z = ((spread - mu) / (sd + 1e-9)).clip(-8.0, 8.0)
    percentile = 1.0 / (1.0 + np.exp(-1.702 * z.fillna(0.0)))
    widening_5 = spread / (spread.rolling(5, min_periods=2).mean() + 1e-9)
    widening_20 = spread / (spread.rolling(20, min_periods=5).mean() + 1e-9)

    if atr_col in df.columns:
        atr = pd.Series(df[atr_col].to_numpy(), dtype="float64")
        atr_pips = atr / _pip_size_for_pair(pair)
        cost_to_atr = spread / (atr_pips + 1e-9)
    else:
        cost_to_atr = pd.Series(0.0, index=spread.index)

    return df.with_columns([
        pl.Series("spread_zscore", z.fillna(0.0).to_numpy()),
        pl.Series("spread_percentile", percentile.fillna(0.5).clip(0.0, 1.0).to_numpy()),
        pl.Series("spread_widening_5m", widening_5.replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy()),
        pl.Series("spread_widening_20m", widening_20.replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy()),
        pl.Series("cost_to_atr", cost_to_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()),
    ])

def add_market_regime_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add lightweight trend/chop/volatility regime features from past OHLC."""
    import pandas as pd

    req = {"high", "low", "close"}
    if not req.issubset(set(df.columns)):
        return df

    high = pd.Series(df["high"].to_numpy(), dtype="float64")
    low = pd.Series(df["low"].to_numpy(), dtype="float64")
    close = pd.Series(df["close"].to_numpy(), dtype="float64")
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr_14 = tr.rolling(14, min_periods=5).sum()
    plus_di = 100.0 * plus_dm.rolling(14, min_periods=5).sum() / (tr_14 + 1e-9)
    minus_di = 100.0 * minus_dm.rolling(14, min_periods=5).sum() / (tr_14 + 1e-9)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(14, min_periods=5).mean()

    chop_window = 14
    tr_sum = tr.rolling(chop_window, min_periods=5).sum()
    range_hl = high.rolling(chop_window, min_periods=5).max() - low.rolling(chop_window, min_periods=5).min()
    chop = 100.0 * np.log10((tr_sum / (range_hl + 1e-9)).clip(lower=1e-9)) / np.log10(chop_window)

    atr_20 = tr.rolling(20, min_periods=5).mean()
    vol_regime = atr_20 / (atr_20.rolling(240, min_periods=30).median() + 1e-9)
    slope_20 = (close - close.shift(20)) / (atr_20 + 1e-9)
    trend_regime = ((adx > 25.0) & (slope_20.abs() > 0.5)).astype(float)
    range_regime = ((chop > 60.0) & (adx < 20.0)).astype(float)

    ret = np.log(close / close.shift(1))
    acorr = ret.rolling(120, min_periods=30).corr(ret.shift(1)).clip(-1.0, 1.0)
    hurst = (0.5 + 0.25 * acorr).clip(0.0, 1.0)
    noise_to_signal = ret.rolling(60, min_periods=10).std() / (ret.rolling(60, min_periods=10).mean().abs() + 1e-9)
    trailing_vol = ret.rolling(60, min_periods=10).std()

    return df.with_columns([
        pl.Series("adx_14", adx.fillna(0.0).to_numpy()),
        pl.Series("chop_index", chop.replace([np.inf, -np.inf], np.nan).fillna(50.0).to_numpy()),
        pl.Series("trend_regime", trend_regime.fillna(0.0).to_numpy()),
        pl.Series("range_regime", range_regime.fillna(0.0).to_numpy()),
        pl.Series("volatility_regime", vol_regime.replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy()),
        pl.Series("hurst_exponent", hurst.fillna(0.5).to_numpy()),
        pl.Series("noise_to_signal_60", noise_to_signal.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()),
        pl.Series("trailing_volatility_60", trailing_vol.fillna(0.0).to_numpy()),
    ])

def add_higher_timeframe_context(df: pl.DataFrame) -> pl.DataFrame:
    """Join completed 5m/15m/1h context back to each 1m row without lookahead."""
    import pandas as pd

    req = {"timestamp_utc", "open", "high", "low", "close", "volume"}
    if not req.issubset(set(df.columns)) or len(df) == 0:
        return df

    pdf = df.to_pandas().sort_values("timestamp_utc")
    idxed = pdf.set_index(pd.to_datetime(pdf["timestamp_utc"], utc=True)).sort_index()
    merged = pdf.copy()

    def _rsi_series(close: "pd.Series", period: int = 14) -> "pd.Series":
        d = close.diff()
        gain = d.clip(lower=0.0).rolling(period, min_periods=5).mean()
        loss = (-d.clip(upper=0.0)).rolling(period, min_periods=5).mean()
        return 100.0 - 100.0 / (1.0 + gain / (loss + 1e-9))

    for rule, suffix, median_window in (("5min", "5m", 48), ("15min", "15m", 32), ("1h", "1h", 24)):
        ohlcv = idxed.resample(rule, label="right", closed="right").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open", "high", "low", "close"])
        if ohlcv.empty:
            continue
        prev_close = ohlcv["close"].shift(1)
        tr = pd.concat([
            ohlcv["high"] - ohlcv["low"],
            (ohlcv["high"] - prev_close).abs(),
            (ohlcv["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=3).mean()
        tp = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0
        vol = ohlcv["volume"].replace(0.0, np.nan).fillna(1.0)
        vwap = (tp * vol).rolling(20, min_periods=3).sum() / (vol.rolling(20, min_periods=3).sum() + 1e-9)
        feats = pd.DataFrame(index=ohlcv.index)
        feats[f"ret_{suffix}"] = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
        feats[f"rsi_{suffix}"] = _rsi_series(ohlcv["close"])
        feats[f"atr_{suffix}"] = atr
        feats[f"trend_slope_{suffix}"] = (ohlcv["close"] - ohlcv["close"].shift(3)) / (atr + 1e-9)
        feats[f"distance_to_vwap_{suffix}"] = (ohlcv["close"] - vwap) / (atr + 1e-9)
        feats[f"volatility_regime_{suffix}"] = atr / (atr.rolling(median_window, min_periods=5).median() + 1e-9)
        feats = feats.shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feats.index.name = "timestamp_utc"
        merged = pd.merge_asof(
            merged.sort_values("timestamp_utc"),
            feats.reset_index().sort_values("timestamp_utc"),
            on="timestamp_utc",
            direction="backward",
        )

    num_cols = merged.select_dtypes(include=[np.number]).columns
    merged[num_cols] = merged[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pl.from_pandas(merged)

# ==================================================================================================================-
# MISSINGNESS FLAGS
# ==================================================================================================================-
def missingness_flags(df: pl.DataFrame, cols: list, decay: float = 0.9) -> pl.DataFrame:
    for c in cols:
        if c not in df.columns:
            df = df.with_columns([pl.lit(1.0).alias(f"{c}_missing"), pl.lit(1.0).alias(f"{c}_staleness")])
            continue
        
        is_null = df[c].is_null()
        df = df.with_columns([
            is_null.cast(pl.Float64).alias(f"{c}_missing"),
            is_null.cum_min().cast(pl.Float64).alias(f"{c}_staleness"),
        ])
    return df

# === F: Latency feature =================================================================================-
def compute_latency_feature(df: pl.DataFrame, latency_baseline_ms: float = 50.0, atr_col: str = "atr_6", atr_window_ref: int = 120) -> pl.Series:
    n = len(df)
    latency = np.full(n, latency_baseline_ms, dtype=np.float32)
    if atr_col in df.columns:
        atr_vals = df[atr_col].fill_null(strategy="forward").to_numpy()
        med_atr = df[atr_col].fill_null(strategy="forward").rolling_median(atr_window_ref, min_samples=10).fill_null(strategy="backward").to_numpy()
        atr_excess = np.maximum(0.0, atr_vals / np.maximum(med_atr, 1e-12) - 1.0)
        latency += latency_baseline_ms * 0.5 * atr_excess
    
    latency = np.clip(latency, latency_baseline_ms, latency_baseline_ms * 10.0)
    return pl.Series("expected_latency_ms", latency)

# === Master Builder ==========================================================================================
class FeatureEngineer:
    def __init__(self,atr_window=6,ofi_window=20,tar_window=30,rsi_period=14,
                 macd_fast=12,macd_slow=26,macd_signal=9,bb_window=20,bb_std=2.0,
                 lag_windows=[5,20,60],vol_mult=3.0,news_buf=2,decay_lam=0.1,fb_dim=8,
                 ca_corr_window=60,ca_regime_window=240,ca_lags=(1,5,15),
                 enable_regime_gate=True):
        self.atr_w=atr_window; self.ofi_w=ofi_window; self.tar_w=tar_window
        self.rsi_p=rsi_period; self.mf=macd_fast; self.ms=macd_slow; self.msig=macd_signal
        self.bb_w=bb_window; self.bb_s=bb_std; self.lags=lag_windows
        self.vm=vol_mult; self.nb=news_buf; self.dl=decay_lam; self.fb=fb_dim
        self.ca=CrossAssetFeatures(
            corr_window=ca_corr_window,
            regime_window=ca_regime_window,
            lags=ca_lags,
        )
        self.enable_regime_gate = bool(enable_regime_gate)
        self.regime_gate = RegimeGateClassifier()
        try:
            from features.macro_features import MacroYieldFeatureBuilder
            self._macro_builder = MacroYieldFeatureBuilder()
        except Exception as e:
            print(f"[FeatureEngineering] WARNING: MacroYieldFeatureBuilder init failed: {e}")
            self._macro_builder = None

    def build_chunked(self, bars: pl.DataFrame, chunk_size: int = 50_000,
                      output_dir: str = None, **kwargs) -> pl.DataFrame:
        """DS-005: Process features in overlapping chronological chunks to bound RAM.

        For datasets too large to fit in memory, processes chunk_size bars at a time
        with warmup overlap to maintain EMA continuity.

        Args:
            bars: full dataset
            chunk_size: bars per chunk (default 50k)
            output_dir: if set, sink chunks to parquet files and return empty frame
            **kwargs: passed to self.build()
        """
        n = len(bars)
        overlap = self.warmup_bars_needed
        chunks = []

        for start in range(0, n, chunk_size):
            warmup_start = max(0, start - overlap)
            end = min(start + chunk_size, n)
            chunk = bars[warmup_start:end]

            features = self.build(chunk, **kwargs)

            # Slice off warmup prefix
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

    def build_with_warmup(self, bars: pl.DataFrame, warmup_bars: pl.DataFrame,
                          **kwargs) -> pl.DataFrame:
        """DS-002: Build features with warmup context to prevent EMA look-ahead bias.

        Prepends warmup_bars before the target bars, builds features on the
        combined sequence, then slices off the warmup prefix. This ensures
        EMA/rolling stats are seeded from historical context, not from the
        first value of the target split.

        Args:
            bars: target split data (val/test)
            warmup_bars: historical context to prepend (last 3*slow_window bars of train)
            **kwargs: passed to self.build()
        """
        if warmup_bars is not None and len(warmup_bars) > 0:
            n_warmup = len(warmup_bars)
            combined = pl.concat([warmup_bars, bars], how="vertical_relaxed")
            features = self.build(combined, **kwargs)
            return features[n_warmup:]
        return self.build(bars, **kwargs)

    @property
    def warmup_bars_needed(self) -> int:
        """Minimum historical bars needed before a split to avoid EMA cold-start."""
        return 3 * self.ms  # 3 × slow MACD window

    def build(self,bars: pl.DataFrame,cross_asset=None,sentiment=None,eco_act=None,eco_fc=None,
              art_counts=None,finbert_embs=None,news_events=None, cot_data=None, pair="EURUSD",
              eco_prior=None, news_cats=None) -> pl.DataFrame:
        
        import pandas as pd
        if isinstance(bars, pd.DataFrame):
            if bars.index.name is None:
                bars.index.name = "timestamp_utc"
            bars = pl.from_pandas(bars.reset_index())

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
            
        # 1. Base Features (Independent)
        F = bars.with_columns(
            [order_book_imbalance_proxy()] +
            trade_arrival_rate(self.tar_w) +
            [order_flow_imbalance(self.ofi_w)] +
            [average_true_range(w) for w in [self.atr_w, 20, 60]] +
            [rolling_volatility(w) for w in [6, 20, 60]] +
            bollinger_bands(self.bb_w, self.bb_s) +
            [rsi(self.rsi_p)] +
            macd(self.mf, self.ms, self.msig) +
            lag_returns(self.lags) +
            stochastic_oscillator() +
            [williams_r()] +
            [cci()] +
            all_candlestick_patterns() +
            [kyles_lambda(20)] +
            [amihud_illiquidity(20)] +
            [realized_spread(10)] +
            [vpin(50, 50)] +
            multi_level_obi(5)
        )
        
        # 2. Derived Features (Dependent on Base Features)
        tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        vol = pl.when(pl.col("volume") == 0).then(1.0).otherwise(pl.col("volume")).fill_null(1.0)
        vwap = (tp * vol).rolling_sum(60) / vol.rolling_sum(60)
        
        # Volume-weighted features (Improvement #4)
        vwap_bands_expr = vwap_bands(60, 2.0)
        
        F = F.with_columns(
            vwap_bands_expr +
            [volume_weighted_momentum(20)] +
            # OFI Surprise
            [((pl.col("ofi") - pl.col("ofi").rolling_mean(min(120, self.ofi_w*6))) / (pl.col("ofi").rolling_std(min(120, self.ofi_w*6)) + 1e-9)).alias("ofi_z")] +
            # Multi-scale ATR
            [((pl.col(f"atr_{self.atr_w}") / (pl.col("atr_20") + 1e-9)).clip(0.1, 10.0)).alias(f"atr_ratio_{self.atr_w}_20")] +
            [((pl.col("atr_20") / (pl.col("atr_60") + 1e-9)).clip(0.1, 10.0)).alias(f"atr_ratio_20_60")] +
            # Multi-scale Volatility
            [((pl.col("vol_6") / (pl.col("vol_20") + 1e-9)).clip(0.1, 10.0)).alias("vol_ratio_6_20")] +
            [((pl.col("vol_20") / (pl.col("vol_60") + 1e-9)).clip(0.1, 10.0)).alias("vol_ratio_20_60")] +
            # Breakout Pressure
            [((pl.col("close") - vwap) / (pl.col("atr_20") + 1e-9)).alias("breakout_pressure")] +
            # Liquidity Vacuum (requires bid/ask columns)
            ([liquidity_vacuum(120)] if "ask_close" in F.columns and "bid_close" in F.columns else [pl.lit(0.0).alias("liquidity_vacuum")])
        )
        
        # 3. Fragility
        F = F.with_columns([
            vol_of_vol(20, 20),
            price_ofi_divergence(self.ofi_w, 10)
        ])
        
        pip_size = _pip_size_for_pair(pair)
        if "spread_avg" in F.columns: 
            F = F.with_columns([(pl.col("spread_avg") / pip_size).alias("spread_pips")])
        elif "ask_close" in F.columns: 
            F = F.with_columns([((pl.col("ask_close") - pl.col("bid_close")) / pip_size).alias("spread_pips")])
        else: 
            F = F.with_columns([pl.lit(0.5).alias("spread_pips")])

        ac = f"atr_{self.atr_w}"
        F = add_spread_cost_features(F, pair=pair, atr_col=ac)
        F = add_market_regime_features(F)
        
        # Regime-gated features (Improvement #3) - create regime-specific variants
        existing = set(F.columns)
        F = F.with_columns(regime_gated_features(existing))
        
        # Add regime quality features
        F = F.with_columns([
            # realized_vol_regime: 0=low, 1=normal, 2=high (percentile-based)
            pl.when(pl.col("vol_60") <= pl.col("vol_60").rolling_quantile(0.33, window_size=60))
            .then(0)
            .when(pl.col("vol_60") <= pl.col("vol_60").rolling_quantile(0.66, window_size=60))
            .then(1)
            .otherwise(2)
            .alias("realized_vol_regime"),
            # trend_quality: ADX (trend strength) x 5-bar RSI momentum (direction & pace)
            (pl.col("adx_14") * (pl.col("rsi_14") - pl.col("rsi_14").shift(5))).alias("trend_quality"),
        ])
        
        # Interaction features (Improvement #5) - explicit non-linear combinations
        existing = set(F.columns)
        F = F.with_columns(interaction_features(existing))
        
        F = add_higher_timeframe_context(F)
        
        # HMM Regime Detection + CPD (Regime Detection Upgrade)
        existing = set(F.columns)
        F = F.with_columns(hmm_regime_probs(3, 60))
        F = F.with_columns(cpd_ret("close", 60))
        F = F.with_columns(regime_persistence(20))
        
        # Circuit Breakers / Kill Switches
        existing = set(F.columns)
        F = F.with_columns(circuit_breaker_features())
        F = F.with_columns(position_limit_flags(atr_col=ac))
        
        # FinBERT embeddings: use real embeddings if provided, otherwise zero placeholders
        if finbert_embs is not None and len(finbert_embs) > 0:
            try:
                if isinstance(finbert_embs, pl.DataFrame):
                    emb_df = finbert_embs
                else:
                    emb_df = pl.DataFrame(finbert_embs)
                if "timestamp_utc" in emb_df.columns and "timestamp_utc" in F.columns:
                    emb_df = emb_df.sort("timestamp_utc")
                    emb_cols = [c for c in emb_df.columns if c.startswith("embed_") or c.startswith("fb_")]
                    F = F.join_asof(
                        emb_df.select(["timestamp_utc"] + emb_cols),
                        on="timestamp_utc",
                        strategy="backward",
                    )
                    rename_map = {c: f"embed_{i}" for i, c in enumerate(emb_cols) if not c.startswith("embed_")}
                    if rename_map:
                        F = F.rename(rename_map)
                    for i in range(8):
                        col = f"embed_{i}"
                        if col in F.columns:
                            F = F.with_columns(pl.col(col).fill_null(0.0))
                        else:
                            F = F.with_columns(pl.lit(0.0).alias(col))
                else:
                    F = F.with_columns(embedding_placeholders(8))
            except Exception:
                F = F.with_columns(embedding_placeholders(8))
        else:
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
        # yield_curve_slope and us_2s10s_spread are the same (US10Y-US2Y)     alias whichever exists
        if "us_2s10s_spread" in F.columns and "yield_curve_slope" not in F.columns:
            F = F.with_columns(pl.col("us_2s10s_spread").alias("yield_curve_slope"))
        elif "yield_curve_slope" in F.columns and "us_2s10s_spread" not in F.columns:
            F = F.with_columns(pl.col("yield_curve_slope").alias("us_2s10s_spread"))
            F = F.with_columns(pl.col("us_2s10s_spread").diff().alias("us_2s10s_spread_chg"))

        # Filters
        F = F.with_columns([vol_filter(ac, self.vm, 60)])
        
        # News Filter - Join events if any
        if news_events is not None and len(news_events) > 0:
            import pandas as pd
            buf = pd.Timedelta(minutes=self.nb)
            # Polars doesn't do non-equi joins easily without `join_where` in newer versions, 
            # so we map to numpy or pandas for speed given small event arrays
            ts = F["timestamp_utc"].to_pandas()
            flags = pd.DataFrame({"news_ok": 1.0, "pre_news": 0.0, "post_news": 0.0}, index=ts)
            for ev in news_events:
                mask_ok = ((ts >= ev-buf) & (ts <= ev+buf)).values
                mask_pre = ((ts >= ev-buf) & (ts < ev)).values
                mask_post = ((ts >= ev) & (ts <= ev+buf)).values
                flags.loc[mask_ok, "news_ok"] = 0.0
                flags.loc[mask_pre, "pre_news"] = 1.0
                flags.loc[mask_post, "post_news"] = 1.0
            F = F.with_columns([
                pl.Series("news_ok", flags["news_ok"].values),
                pl.Series("pre_news", flags["pre_news"].values),
                pl.Series("post_news", flags["post_news"].values)
            ])
        else:
            F = F.with_columns([
                pl.lit(1.0).alias("news_ok"),
                pl.lit(0.0).alias("pre_news"),
                pl.lit(0.0).alias("post_news")
            ])
            
        if self.enable_regime_gate:
            rbp = self.regime_gate.fit_predict(F)
            F = F.with_columns([rbp])
            F = F.with_columns([
                (0.2 + 0.8 * pl.col("regime_break_prob")).clip(0.0, 1.0).alias("gate_gold_weight"),
                (1.0 - 0.7 * pl.col("regime_break_prob")).clip(0.0, 1.0).alias("gate_yield_weight"),
                (0.3 + 0.7 * pl.col("regime_break_prob")).clip(0.0, 1.0).alias("gate_risk_weight")
            ])
            
        # Sentiment & News features using join_asof
        # Sentiment tiers (raw, decayed, FinBERT projection)
        if sentiment is not None:
            # timestamp already cast at top of build()
            drop_cols = [c for c in sentiment.columns if c in F.columns and c != "timestamp_utc"]
            F = F.join_asof(sentiment.sort("timestamp_utc").drop(drop_cols), on="timestamp_utc", strategy="backward")
            # Apply the tiered sentiment helper (adds raw, decay, and placeholder FinBERT cols)
            F = sentiment_tiers(F, decay_lam=self.dl, fb_dim=self.fb)
        else:
            # No sentiment data     create zeroed columns for the three tiers
            F = F.with_columns([
                pl.lit(0.0).alias("sentiment_raw"),
                pl.lit(0.0).alias("sentiment_decayed"),
                *[pl.lit(0.0).alias(f"fb_{i}") for i in range(self.fb)]
            ])
            
        if eco_act is not None:
            eco_act = eco_act.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
            if eco_fc is not None:
                eco_fc = eco_fc.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
                eco = eco_act.join(eco_fc, on="timestamp_utc", how="outer_coalesce").sort("timestamp_utc")
            else:
                eco = eco_act.sort("timestamp_utc")
            if eco_prior is not None:
                eco_prior = eco_prior.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
                drop_pr = [c for c in eco_prior.columns if c in eco.columns and c != "timestamp_utc"]
                eco = eco.join(eco_prior.drop(drop_pr), on="timestamp_utc", how="outer_coalesce").sort("timestamp_utc")
            drop_cols = [c for c in eco.columns if c in F.columns and c != "timestamp_utc"]
            F = F.join_asof(eco.drop(drop_cols), on="timestamp_utc", strategy="backward")
            if "actual" in F.columns and "forecast" in F.columns:
                F = F.with_columns([
                    (pl.col("actual").cast(pl.String).str.replace_all(r"[^\d\.\-]", "").cast(pl.Float64, strict=False) - 
                     pl.col("forecast").cast(pl.String).str.replace_all(r"[^\d\.\-]", "").cast(pl.Float64, strict=False)).fill_null(0.0).alias("eco_surprise")
                ])
                if "prior" in F.columns:
                    F = F.with_columns([
                        (pl.col("actual").cast(pl.String).str.replace_all(r"[^\d\.\-]", "").cast(pl.Float64, strict=False) -
                         pl.col("prior").cast(pl.String).str.replace_all(r"[^\d\.\-]", "").cast(pl.Float64, strict=False)).fill_null(0.0).alias("eco_revision")
                    ])
                else:
                    F = F.with_columns([pl.lit(0.0).alias("eco_revision")])
            else:
                F = F.with_columns([pl.lit(0.0).alias("eco_surprise"), pl.lit(0.0).alias("eco_revision")])
        else:
            F = F.with_columns([pl.lit(0.0).alias("eco_surprise"), pl.lit(0.0).alias("eco_revision")])
            
        if art_counts is not None:
            drop_cols = [c for c in art_counts.columns if c in F.columns and c != "timestamp_utc"]
            F = F.join_asof(art_counts.sort("timestamp_utc").drop(drop_cols), on="timestamp_utc", strategy="backward")
            F = F.with_columns([
                pl.col("article_counts").fill_null(0.0)
            ])
            F = F.with_columns([buzz_score()])
        else:
            F = F.with_columns([pl.lit(0.0).alias("buzz")])

        # News category one-hot flags (central_bank, inflation, labor, growth, geopolitical, commentary)
        _NEWS_CAT_COLS = ["cat_central_bank", "cat_inflation", "cat_labor", "cat_growth", "cat_geopolitical", "cat_commentary"]
        if news_cats is not None and "timestamp_utc" in news_cats.columns:
            news_cats = news_cats.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
            drop_cols = [c for c in news_cats.columns if c in F.columns and c != "timestamp_utc"]
            F = F.join_asof(news_cats.sort("timestamp_utc").drop(drop_cols), on="timestamp_utc", strategy="backward")
            for _nc in _NEWS_CAT_COLS:
                if _nc in F.columns:
                    F = F.with_columns(pl.col(_nc).fill_null(0.0))
                else:
                    F = F.with_columns(pl.lit(0.0).alias(_nc))
        else:
            F = F.with_columns([pl.lit(0.0).alias(c) for c in _NEWS_CAT_COLS])

        # Ensure fb_* placeholder columns exist (only if not already created by sentiment_tiers)
        fb_cols = [f"fb_{i}" for i in range(self.fb)]
        missing_fb = [c for c in fb_cols if c not in F.columns]
        if missing_fb:
            F = F.with_columns([pl.lit(0.0).alias(c) for c in missing_fb])
        
        
        # Temporal
        import pandas as pd
        ts_pd = F["timestamp_utc"].to_pandas()
        h = ts_pd.dt.hour; m = ts_pd.dt.minute; tm = h * 60 + m
        F = F.with_columns([
            pl.Series("time_sin", np.sin(2*np.pi*tm/1440)),
            pl.Series("time_cos", np.cos(2*np.pi*tm/1440)),
            pl.Series("day_sin", np.sin(2*np.pi*ts_pd.dt.dayofweek/5)),
            pl.Series("day_cos", np.cos(2*np.pi*ts_pd.dt.dayofweek/5)),
            pl.Series("london_ny", ((h >= 13) & (h <= 17)).astype(float))
        ])
        
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
            
        F = F.with_columns([
            compute_latency_feature(F, _lat_base, ac)
        ])
        
        # Final cleanup
        n0 = len(F)
        F = F.drop(["actual", "forecast", "prior", "_eco_raw", "_eco_scale", "sentiment", "article_counts"], strict=False)
        F = F.drop_nulls(subset=[ac])
        F = F.fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.0)
        if n0 > len(F):
            print(f"[Features] Dropped {n0 - len(F):,} NaN rows -> {len(F):,}  {len(F.columns)}")

        # Restore the caller's timestamp precision so feats can be joined back to bars.
        if "timestamp_utc" in F.columns and _ts_dtype is not None:
            F = F.with_columns(pl.col("timestamp_utc").cast(_ts_dtype))

        return sanitize_frame(F, fill_value=0.0, context="FeatureEngineer.build")
