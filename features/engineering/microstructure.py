"""
features/engineering/microstructure.py
Microstructure, indicators, candlestick patterns, VWAP, volume profile,
spread/cost features, market regime helpers, circuit breakers, latency,
missingness flags, higher-timeframe context, and miscellaneous utilities.
"""

import numpy as np
import polars as pl


# === Microstructure ===================================================================

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
    rate = pl.col("volume").cast(pl.Float64)
    mu = rate.rolling_mean(window)
    s = rate.rolling_std(window) + 1e-9
    return [((rate - mu) / s).alias("tar")]


def kyles_lambda(window: int = 20) -> pl.Expr:
    """Kyle's Lambda: price impact per unit *signed* volume, in basis points."""
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    signed_vol = pl.col("volume").cast(pl.Float64) * (pl.col("close") - pl.col("open")).sign()

    ret_mean = ret.rolling_mean(window)
    sv_mean = signed_vol.rolling_mean(window)
    cov = (ret * signed_vol).rolling_mean(window) - (ret_mean * sv_mean)
    sv_var = signed_vol.rolling_var(window) + 1e-9

    return ((cov / sv_var) * 1e4).alias("kyles_lambda")


def amihud_illiquidity(window: int = 20) -> pl.Expr:
    """Amihud Illiquidity Ratio: |ret| per unit *normalized* volume, in basis points."""
    ret = (pl.col("close") / pl.col("close").shift(1)).log().abs()
    vol = pl.col("volume").cast(pl.Float64)
    vol_n = vol / (vol.rolling_mean(window) + 1e-9)
    return ((ret / (vol_n + 1e-9)).rolling_mean(window) * 1e4).alias("amihud_illiq")


def multi_level_obi(n_levels: int = 5, use_real_l2: bool = False) -> list[pl.Expr]:
    """Multi-level Order Book Imbalance."""
    exprs = []

    if use_real_l2:
        for i in range(1, n_levels + 1):
            bid_col = f"bid_sz_{i:02d}"
            ask_col = f"ask_sz_{i:02d}"
            real_obi = ((pl.col(bid_col) - pl.col(ask_col)) / (pl.col(bid_col) + pl.col(ask_col) + 1e-9)).alias(
                f"obi_l{i}"
            )
            exprs.append(real_obi)

        bid_total = pl.sum_horizontal([pl.col(f"bid_sz_{i:02d}") for i in range(1, n_levels + 1)])
        ask_total = pl.sum_horizontal([pl.col(f"ask_sz_{i:02d}") for i in range(1, n_levels + 1)])
        exprs.append(((bid_total - ask_total) / (bid_total + ask_total + 1e-9)).alias("obi_aggregate"))
        return exprs

    for i in range(1, n_levels + 1):
        bid_vol = pl.when(pl.col("close") > pl.col("open")).then(pl.col("volume") / n_levels).otherwise(pl.lit(0))
        ask_vol = pl.when(pl.col("close") < pl.col("open")).then(pl.col("volume") / n_levels).otherwise(pl.lit(0))
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        exprs.append(obi.alias(f"obi_l{i}"))
    return exprs


def vpin(bucket_size: int = 50, n_buckets: int = 50) -> pl.Expr:
    """Volume-Synchronized Probability of Informed Trading (VPIN)."""
    buy_vol = pl.when(pl.col("close") > pl.col("open")).then(pl.col("volume")).otherwise(0.0)
    sell_vol = pl.when(pl.col("close") < pl.col("open")).then(pl.col("volume")).otherwise(0.0)

    buy_bucket = buy_vol.rolling_sum(window_size=bucket_size)
    sell_bucket = sell_vol.rolling_sum(window_size=bucket_size)
    total_bucket = buy_bucket + sell_bucket + 1e-9

    vpin_val = ((buy_bucket - sell_bucket).abs() / total_bucket).rolling_mean(window_size=n_buckets)
    return vpin_val.alias("vpin")


def realized_spread(window: int = 10) -> pl.Expr:
    """Realized spread proxy via the Corwin-Schultz estimator."""
    hl = (pl.col("high") / pl.col("low")).log()
    beta = (hl * hl.shift(1)).sqrt()
    spread = 2 * (beta.exp() - 1) / (1 + beta.exp())
    return spread.rolling_mean(window).alias("realized_spread")


def order_book_imbalance_l2(
    bid1: str = "bid_price_1",
    ask1: str = "ask_price_1",
    bid2: str = "bid_price_2",
    ask2: str = "ask_price_2",
    vol1: str = "bid_size_1",
    vol2: str = "ask_size_1",
) -> pl.Expr:
    """Compute a depth-2 order-book imbalance proxy."""
    imbalance1 = (pl.col(bid1) * pl.col(vol1) - pl.col(ask1) * pl.col(vol2)) / (pl.col(vol1) + pl.col(vol2) + 1e-9)
    imbalance2 = (pl.col(bid2) * pl.col(vol1) - pl.col(ask2) * pl.col(vol2)) / (pl.col(vol1) + pl.col(vol2) + 1e-9)
    return (imbalance1 + imbalance2).rolling_mean(20).alias("ofi_l2")


# === Embedding placeholders ==========================================================

def embedding_placeholders(dim: int = 8) -> list[pl.Expr]:
    """Schema-compatible FinBERT placeholders when no embeddings are available."""
    return [pl.lit(0.0).alias(f"fb_{i}") for i in range(dim)]


def compute_finbert_embeddings(
    news_df: pl.DataFrame,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 64,
    max_length: int = 128,
    output_dim: int = 32,
) -> pl.DataFrame:
    """Compute real FinBERT sentiment embeddings from news headlines."""
    import torch

    if news_df is None or len(news_df) == 0:
        return pl.DataFrame()

    headline_col = "headline" if "headline" in news_df.columns else "title"
    if headline_col not in news_df.columns:
        return pl.DataFrame()

    headlines = news_df[headline_col].fill_null("").to_list()
    timestamps = news_df["timestamp_utc"] if "timestamp_utc" in news_df.columns else None

    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        print(f"[FinBERT] Model load failed: {e}. Returning empty embeddings.")
        return pl.DataFrame()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    embeddings = np.vstack(all_embeddings)

    if embeddings.shape[1] > output_dim:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=output_dim)
        embeddings = pca.fit_transform(embeddings)

    embed_cols = {f"embed_{i}": embeddings[:, i].astype(np.float32) for i in range(embeddings.shape[1])}
    result = pl.DataFrame(embed_cols)
    if timestamps is not None:
        result = result.with_columns(timestamps.alias("timestamp_utc"))

    return result


# === ATR + Volatility ================================================================

def frac_diff_proxy(col_name: str = "close") -> pl.Expr:
    """Lightweight d=0.5 proxy (first 4 weights) for fractional differentiation."""
    return (
        pl.col(col_name)
        - 0.5 * pl.col(col_name).shift(1)
        - 0.125 * pl.col(col_name).shift(2)
        - 0.0625 * pl.col(col_name).shift(3)
    ).fill_null(pl.col(col_name)).alias(f"{col_name}_ffd")


def average_true_range(window: int = 6) -> pl.Expr:
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        [pl.col("high") - pl.col("low"), (pl.col("high") - prev_close).abs(), (pl.col("low") - prev_close).abs()]
    )
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
        pct.alias("bb_pct"),
    ]


# === Momentum ========================================================================

def _log_ret() -> pl.Expr:
    # Per-bar log return. ``close_ffd`` is a fractionally differenced price *level*
    # (~0.04 x price), not a return: summing it gave ret_w ~ w x price (|corr| > 0.995
    # with open) and pinned RSI near 100. Log returns are also pip-size free (JPY-safe).
    return pl.col("close").log().diff()


def rsi(period: int = 14) -> pl.Expr:
    d = _log_ret()
    g = d.clip(0, float("inf")).rolling_mean(period)
    l = (-d.clip(float("-inf"), 0)).rolling_mean(period) + 1e-9  # noqa: E741
    return (100 - 100 / (1 + g / l)).alias(f"rsi_{period}")


def macd(fast: int = 12, slow: int = 26, signal: int = 9) -> list[pl.Expr]:
    ef = pl.col("close").ewm_mean(span=fast, adjust=False)
    es = pl.col("close").ewm_mean(span=slow, adjust=False)
    line = (ef - es) / pl.col("close") * 1e4  # bps of price: comparable across pairs
    sig = line.ewm_mean(span=signal, adjust=False)
    return [line.alias("macd"), sig.alias("macd_sig"), (line - sig).alias("macd_hist")]


def lag_returns(windows: list[int] | None = None) -> list[pl.Expr]:
    if windows is None:
        windows = [5, 20, 60]
    return [(_log_ret().rolling_sum(w) * 1e4).alias(f"ret_{w}") for w in windows]  # bps


# === Classical Indicators ============================================================

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


# === Candlestick Patterns ============================================================

def candle_body() -> pl.Expr:
    return (pl.col("close") - pl.col("open")).abs().alias("candle_body")


def candle_range() -> pl.Expr:
    return (pl.col("high") - pl.col("low")).alias("candle_range")


def upper_shadow() -> pl.Expr:
    return (pl.col("high") - pl.max_horizontal("open", "close")).alias("upper_shadow")


def lower_shadow() -> pl.Expr:
    return (pl.min_horizontal("open", "close") - pl.col("low")).alias("lower_shadow")


def is_doji(threshold: float = 0.1) -> pl.Expr:
    body = (pl.col("close") - pl.col("open")).abs()
    rng = pl.col("high") - pl.col("low") + 1e-9
    return (body / rng <= threshold).cast(pl.Int32).alias("is_doji")


def is_hammer(body_ratio: float = 0.3, lower_ratio: float = 0.6, upper_ratio: float = 0.1) -> pl.Expr:
    body = (pl.col("close") - pl.col("open")).abs()
    rng = pl.col("high") - pl.col("low") + 1e-9
    lower = pl.min_horizontal("open", "close") - pl.col("low")
    upper = pl.col("high") - pl.max_horizontal("open", "close")
    cond = (body / rng <= body_ratio) & (lower / rng >= lower_ratio) & (upper / rng <= upper_ratio)
    return cond.cast(pl.Int32).alias("is_hammer")


def is_shooting_star() -> pl.Expr:
    body = (pl.col("close") - pl.col("open")).abs()
    rng = pl.col("high") - pl.col("low") + 1e-9
    lower = pl.min_horizontal("open", "close") - pl.col("low")
    upper = pl.col("high") - pl.max_horizontal("open", "close")
    cond = (body / rng <= 0.3) & (upper / rng >= 0.6) & (lower / rng <= 0.1)
    return cond.cast(pl.Int32).alias("is_shooting_star")


def is_engulfing() -> pl.Expr:
    prev_open = pl.col("open").shift(1)
    prev_close = pl.col("close").shift(1)
    (prev_close - prev_open).abs()
    (pl.col("close") - pl.col("open")).abs()
    bullish = (
        (prev_close < prev_open)
        & (pl.col("close") > pl.col("open"))
        & (pl.col("open") <= prev_close)
        & (pl.col("close") >= prev_open)
    )
    bearish = (
        (prev_close > prev_open)
        & (pl.col("close") < pl.col("open"))
        & (pl.col("open") >= prev_close)
        & (pl.col("close") <= prev_open)
    )
    return (bullish.cast(pl.Int32) - bearish.cast(pl.Int32)).alias("engulfing")


def is_harami() -> pl.Expr:
    prev_open = pl.col("open").shift(1)
    prev_close = pl.col("close").shift(1)
    bullish = (
        (prev_close < prev_open)
        & (pl.col("close") > pl.col("open"))
        & (pl.col("open") > prev_close)
        & (pl.col("close") < prev_open)
    )
    bearish = (
        (prev_close > prev_open)
        & (pl.col("close") < pl.col("open"))
        & (pl.col("open") < prev_close)
        & (pl.col("close") > prev_open)
    )
    return (bullish.cast(pl.Int32) - bearish.cast(pl.Int32)).alias("harami")


def is_morning_star() -> pl.Expr:
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
    c1_bullish = pl.col("close").shift(2) > pl.col("open").shift(2)
    c2_bullish = pl.col("close").shift(1) > pl.col("open").shift(1)
    c3_bullish = pl.col("close") > pl.col("open")
    higher_closes = (pl.col("close") > pl.col("close").shift(1)) & (pl.col("close").shift(1) > pl.col("close").shift(2))
    cond = c1_bullish & c2_bullish & c3_bullish & higher_closes
    return cond.cast(pl.Int32).alias("three_white_soldiers")


def is_three_black_crows() -> pl.Expr:
    c1_bearish = pl.col("close").shift(2) < pl.col("open").shift(2)
    c2_bearish = pl.col("close").shift(1) < pl.col("open").shift(1)
    c3_bearish = pl.col("close") < pl.col("open")
    lower_closes = (pl.col("close") < pl.col("close").shift(1)) & (pl.col("close").shift(1) < pl.col("close").shift(2))
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


# === Volume-weighted features =========================================================

def vwap_bands(window: int = 60, n_std: float = 2.0) -> list[pl.Expr]:
    """VWAP with standard deviation bands."""
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    vol = pl.when(pl.col("volume") == 0).then(1.0).otherwise(pl.col("volume"))
    vwap = (tp * vol).rolling_sum(window) / vol.rolling_sum(window)
    vwap_var = ((tp - vwap) ** 2 * vol).rolling_sum(window) / (vol.rolling_sum(window) + 1e-9)
    vwap_std = vwap_var.sqrt()
    return [
        vwap.alias("vwap"),
        (vwap + n_std * vwap_std).alias("vwap_upper"),
        (vwap - n_std * vwap_std).alias("vwap_lower"),
        ((pl.col("close") - vwap) / (vwap_std + 1e-9)).alias("vwap_zscore"),
    ]


def volume_weighted_momentum(window: int = 20) -> pl.Expr:
    """Volume-weighted moving average of returns."""
    ret = _log_ret() * 1e4
    vwma = (ret * pl.col("volume")).rolling_sum(window) / pl.col("volume").rolling_sum(window)
    return vwma.alias("vwma_ret")


# === Volume Profile / POC =============================================================

def add_volume_profile_features(df: pl.DataFrame, window: int = 240, n_bins: int = 10) -> pl.DataFrame:
    """Volume Profile / Point-of-Control (POC) features."""
    lo = pl.col("low").rolling_min(window)
    hi = pl.col("high").rolling_max(window)
    span = hi - lo + 1e-9
    bin_expr = ((pl.col("close") - lo) / span * n_bins).floor().clip(0, n_bins - 1).cast(pl.Int64)

    F = df.with_columns(bin_expr.alias("__vp_bin"))
    b = pl.col("__vp_bin")
    vol = pl.col("volume").cast(pl.Float64)

    rolls = [(pl.when(b == k).then(vol).otherwise(0.0).rolling_sum(window), k) for k in range(n_bins)]
    tot = vol.rolling_sum(window)

    poc_bin = pl.concat_list([r for r, _ in rolls]).list.arg_max().cast(pl.Float64)
    poc_vol = pl.max_horizontal([r for r, _ in rolls])
    vw_mean = pl.sum_horizontal([k * r for r, k in rolls]) / (tot + 1e-9)
    va_wstd = (pl.sum_horizontal([r * (k - vw_mean) ** 2 for r, k in rolls]) / (tot + 1e-9)).sqrt()
    norm = max(1, n_bins - 1)

    out = F.with_columns(
        [
            (poc_bin / norm).alias("vp_poc_pos"),
            ((b.cast(pl.Float64) - poc_bin) / n_bins).alias("vp_poc_dist"),
            (poc_vol / (tot + 1e-9)).alias("vp_poc_share"),
            (vw_mean / norm).alias("vp_vw_pos"),
            ((vw_mean / norm) - (poc_bin / norm)).alias("vp_skew"),
            ((va_wstd / n_bins) * 10.0).alias("vp_va_width"),
            pl.when((b.cast(pl.Float64) - poc_bin).abs() <= va_wstd).then(1.0).otherwise(0.0).alias("vp_in_va"),
        ]
    ).drop("__vp_bin")

    for c in ("vp_poc_pos", "vp_poc_dist", "vp_poc_share", "vp_vw_pos", "vp_skew", "vp_va_width", "vp_in_va"):
        out = out.with_columns(pl.col(c).fill_null(0.0))
    return out


# === Intraday Volatility Clock ========================================================

def add_volatility_clock_features(df: pl.DataFrame, day_period: int | None = None, k_days: int = 7) -> pl.DataFrame:
    """Intraday 'volatility clock' features.

    ``day_period`` is bars per day. It used to default to 1440 (1-minute bars):
    on 5-minute bars the 7-day lookback needed 10,080 bars, longer than any build
    window, so pace/hot were always null -> constant 0. Now inferred from the
    median bar spacing when not given.
    """
    if day_period is None:
        day_period = 1440
        if "timestamp_utc" in df.columns and df.height > 2:
            _dt = df["timestamp_utc"].diff().drop_nulls()
            if len(_dt):
                _med_s = _dt.dt.total_seconds().median()
                if _med_s and _med_s > 0:
                    day_period = max(1, int(round(86400 / float(_med_s))))
    # Lookback must fit in the frame (each build window is ~5 days + warm-up with
    # weekend gaps): with k_days=7 at 288 bars/day the rolling mean never filled
    # and pace/hot stayed constant 0.
    k_days = int(max(1, min(int(k_days), df.height // max(1, int(day_period)) - 1)))
    ret = (pl.col("close") / pl.col("close").shift(1)).log().abs()
    mins_day = pl.col("timestamp_utc").dt.hour().cast(pl.Int32) * 60 + pl.col("timestamp_utc").dt.minute().cast(
        pl.Int32
    )
    pos = mins_day / 1440.0  # fraction of the day, independent of bar size

    refs = [ret.shift(day_period * k) for k in range(1, k_days + 1)]
    ref_mean = pl.sum_horizontal(refs) / k_days
    ref_std = (pl.sum_horizontal([r**2 for r in refs]) / k_days - ref_mean**2).clip(0.0, None).sqrt()
    typical_daily = ret.rolling_mean(day_period * k_days, min_samples=day_period) * float(day_period)
    session_cum = ret.cum_sum().over(pl.col("timestamp_utc").dt.date())
    pace = session_cum / (typical_daily + 1e-9)

    out = df.with_columns(
        [
            pos.alias("vol_clock_pos"),
            (ret / (ref_mean + 1e-9)).clip(0.0, 10.0).alias("vol_clock_ratio"),
            ((ret - ref_mean) / (ref_std + 1e-9)).clip(-10.0, 10.0).alias("vol_clock_z"),
            pace.clip(0.0, 10.0).alias("vol_clock_pace"),
            (pace - pos).clip(-5.0, 10.0).alias("vol_clock_hot"),
        ]
    )
    for c in ("vol_clock_pos", "vol_clock_ratio", "vol_clock_z", "vol_clock_pace", "vol_clock_hot"):
        out = out.with_columns(pl.col(c).fill_null(0.0))
    return out


# === Multi-scale =======================================================================================

def multi_scale_atr(windows=(6, 20, 60)) -> list[pl.Expr]:
    exprs = []
    prev_w = None
    for w in windows:
        exprs.append(average_true_range(w))
        if prev_w is not None:
            exprs.append(
                (pl.col(f"atr_{prev_w}") / (pl.col(f"atr_{w}") + 1e-9)).clip(0.1, 10.0).alias(f"atr_ratio_{prev_w}_{w}")
            )
        prev_w = w
    return exprs


def multi_scale_vol(windows=(6, 20, 60)) -> list[pl.Expr]:
    exprs = []
    prev_w = None
    for w in windows:
        exprs.append(rolling_volatility(w))
        if prev_w is not None:
            exprs.append(
                (pl.col(f"vol_{prev_w}") / (pl.col(f"vol_{w}") + 1e-9)).clip(0.1, 10.0).alias(f"vol_ratio_{prev_w}_{w}")
            )
        prev_w = w
    return exprs


def ofi_surprise(fast=20, slow=120) -> list[pl.Expr]:
    return [
        order_flow_imbalance(fast),
        ((pl.col("ofi") - pl.col("ofi").rolling_mean(slow)) / (pl.col("ofi").rolling_std(slow) + 1e-9)).alias("ofi_z"),
    ]


def breakout_pressure(vwap_window=60, atr_window=20) -> list[pl.Expr]:
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    vol = pl.when(pl.col("volume") == 0).then(1.0).otherwise(pl.col("volume")).fill_null(1.0)
    vwap = (tp * vol).rolling_sum(vwap_window) / vol.rolling_sum(vwap_window)
    return [
        average_true_range(atr_window),
        ((pl.col("close") - vwap) / (pl.col(f"atr_{atr_window}") + 1e-9)).alias("breakout_pressure"),
    ]


def liquidity_vacuum(window=120) -> pl.Expr:
    spread = (pl.col("ask_close") - pl.col("bid_close")).abs().fill_null(0.0001)
    median_spread = spread.rolling_median(window) + 1e-9
    return (spread / median_spread).alias("liquidity_vacuum")


def vol_of_vol(vol_window=20, vov_window=20) -> pl.Expr:
    return pl.col(f"vol_{vol_window}").rolling_std(vov_window).alias("vol_of_vol")


def price_ofi_divergence(ofi_window=20, price_window=10) -> pl.Expr:
    price_dir = pl.col("close").diff(price_window).sign()
    ofi_dir = pl.col("ofi").sign()
    return (price_dir - ofi_dir).abs().rolling_mean(price_window).alias("price_ofi_div")


# === Filters =========================================================================

def vol_filter(atr_col: str = "atr_6", mult: float = 3.0, lb: int = 60) -> pl.Expr:
    return (pl.col(atr_col) <= mult * pl.col(atr_col).rolling_mean(lb)).cast(pl.Float64).alias("vol_ok")


# === Spread/cost features ============================================================

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
    z = pd.Series(((spread - mu) / (sd + 1e-9)).clip(-8.0, 8.0), index=spread.index, dtype="float64")
    percentile = pd.Series(1.0 / (1.0 + np.exp(-1.702 * z.fillna(0.0))), index=spread.index, dtype="float64")
    widening_5 = spread / (spread.rolling(5, min_periods=2).mean() + 1e-9)
    widening_20 = spread / (spread.rolling(20, min_periods=5).mean() + 1e-9)

    if atr_col in df.columns:
        atr = pd.Series(df[atr_col].to_numpy(), dtype="float64")
        atr_pips = atr / _pip_size_for_pair(pair)
        cost_to_atr = spread / (atr_pips + 1e-9)
    else:
        cost_to_atr = pd.Series(0.0, index=spread.index)

    return df.with_columns(
        [
            pl.Series("spread_zscore", z.fillna(0.0).to_numpy()),
            pl.Series("spread_percentile", percentile.fillna(0.5).clip(0.0, 1.0).to_numpy()),
            pl.Series("spread_widening_5m", widening_5.replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy()),
            pl.Series("spread_widening_20m", widening_20.replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy()),
            pl.Series("cost_to_atr", cost_to_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()),
        ]
    )


# === Market regime helpers ============================================================

def add_market_regime_features(
    df: pl.DataFrame,
    *,
    chop_window: int = 14,
    regime_window: int = 240,
    volatility_window: int = 120,
) -> pl.DataFrame:
    """Add lightweight trend/chop/volatility regime features from past OHLC."""
    import pandas as pd

    req = {"high", "low", "close"}
    if not req.issubset(set(df.columns)):
        return df

    high = pd.Series(df["high"].to_numpy(), dtype="float64")
    low = pd.Series(df["low"].to_numpy(), dtype="float64")
    close = pd.Series(df["close"].to_numpy(), dtype="float64")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr_14 = tr.rolling(14, min_periods=5).sum()
    plus_di = 100.0 * plus_dm.rolling(14, min_periods=5).sum() / (tr_14 + 1e-9)
    minus_di = 100.0 * minus_dm.rolling(14, min_periods=5).sum() / (tr_14 + 1e-9)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(14, min_periods=5).mean()

    chop_window = int(max(2, chop_window))
    tr_sum = tr.rolling(chop_window, min_periods=5).sum()
    range_hl = high.rolling(chop_window, min_periods=5).max() - low.rolling(chop_window, min_periods=5).min()
    chop = 100.0 * np.log10((tr_sum / (range_hl + 1e-9)).clip(lower=1e-9)) / np.log10(chop_window)

    atr_20 = tr.rolling(20, min_periods=5).mean()
    vol_regime = atr_20 / (atr_20.rolling(int(max(10, regime_window)), min_periods=30).median() + 1e-9)
    slope_20 = (close - close.shift(20)) / (atr_20 + 1e-9)
    trend_regime = ((adx > 25.0) & (slope_20.abs() > 0.5)).astype(float)
    range_regime = ((chop > 60.0) & (adx < 20.0)).astype(float)

    vw = int(max(10, volatility_window))
    ret = pd.Series(np.log(close / close.shift(1)), index=close.index, dtype="float64")
    acorr = ret.rolling(vw, min_periods=min(10, vw)).corr(ret.shift(1)).clip(-1.0, 1.0)
    hurst = (0.5 + 0.25 * acorr).clip(0.0, 1.0)
    # Bounded form of std/|mean|: the raw ratio exploded (~8k) when the mean return ~ 0.
    _n2s_mean = ret.rolling(60, min_periods=10).mean().abs()
    _n2s_std = ret.rolling(60, min_periods=10).std()
    noise_to_signal = np.log1p(_n2s_std / (_n2s_mean + _n2s_std * 1e-3 + 1e-12))
    trailing_vol = ret.rolling(60, min_periods=10).std()

    return df.with_columns(
        [
            pl.Series("adx_14", adx.fillna(0.0).to_numpy()),
            pl.Series("chop_index", chop.replace([np.inf, -np.inf], np.nan).fillna(50.0).to_numpy()),
            pl.Series("trend_regime", trend_regime.fillna(0.0).to_numpy()),
            pl.Series("range_regime", range_regime.fillna(0.0).to_numpy()),
            pl.Series("volatility_regime", vol_regime.replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy()),
            pl.Series("hurst_exponent", hurst.fillna(0.5).to_numpy()),
            pl.Series("noise_to_signal_60", noise_to_signal.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()),
            pl.Series("trailing_volatility_60", trailing_vol.fillna(0.0).to_numpy()),
        ]
    )


# === Higher Timeframe Context ==========================================================

def add_higher_timeframe_context(df: pl.DataFrame) -> pl.DataFrame:
    """Join completed 5m/15m/1h context back to each 1m row without lookahead."""
    req = {"timestamp_utc", "open", "high", "low", "close", "volume"}
    if not req.issubset(set(df.columns)) or len(df) == 0:
        return df

    out = df.sort("timestamp_utc")
    for every, suffix, median_window in (("5m", "5m", 48), ("15m", "15m", 32), ("1h", "1h", 24)):
        ohlcv = (
            out.group_by_dynamic(
                "timestamp_utc",
                every=every,
                closed="right",
                label="right",
            )
            .agg(
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            )
            .drop_nulls(subset=["open", "high", "low", "close"])
            .sort("timestamp_utc")
        )
        if ohlcv.height == 0:
            continue
        prev_close = pl.col("close").shift(1)
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        )
        atr = tr.rolling_mean(14, min_samples=3)
        tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        vol = pl.when(pl.col("volume") == 0).then(1.0).otherwise(pl.col("volume")).fill_null(1.0)
        vwap = (tp * vol).rolling_sum(20, min_samples=3) / (vol.rolling_sum(20, min_samples=3) + 1e-9)
        d = pl.col("close").diff()
        gain = d.clip(lower_bound=0.0).rolling_mean(14, min_samples=5)
        loss = (-d.clip(upper_bound=0.0)).rolling_mean(14, min_samples=5)
        rsi_htf = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-9))

        feats = ohlcv.with_columns(
            [
                (pl.col("close").log() - pl.col("close").shift(1).log()).alias(f"ret_{suffix}"),
                rsi_htf.alias(f"rsi_{suffix}"),
                atr.alias(f"atr_{suffix}"),
                ((pl.col("close") - pl.col("close").shift(3)) / (atr + 1e-9)).alias(f"trend_slope_{suffix}"),
                ((pl.col("close") - vwap) / (atr + 1e-9)).alias(f"distance_to_vwap_{suffix}"),
                (atr / (atr.rolling_median(median_window, min_samples=5) + 1e-9)).alias(f"volatility_regime_{suffix}"),
            ]
        ).select(
            [
                "timestamp_utc",
                f"ret_{suffix}",
                f"rsi_{suffix}",
                f"atr_{suffix}",
                f"trend_slope_{suffix}",
                f"distance_to_vwap_{suffix}",
                f"volatility_regime_{suffix}",
            ]
        )
        feat_cols = [c for c in feats.columns if c != "timestamp_utc"]
        feats = feats.with_columns([pl.col(c).shift(1) for c in feat_cols]).fill_null(0.0)
        out = out.join_asof(feats, on="timestamp_utc", strategy="backward")

    htf_cols = [
        c
        for c in out.columns
        if c.endswith(("_5m", "_15m", "_1h"))
        and c.startswith(("ret_", "rsi_", "atr_", "trend_slope_", "distance_to_vwap_", "volatility_regime_"))
    ]
    if htf_cols:
        out = out.with_columns(
            [
                pl.when(pl.col(c).is_infinite()).then(0.0).otherwise(pl.col(c).fill_nan(0.0).fill_null(0.0)).alias(c)
                for c in htf_cols
            ]
        )
    return out


# === Missingness flags ================================================================

def missingness_flags(df: pl.DataFrame, cols: list, decay: float = 0.9) -> pl.DataFrame:
    for c in cols:
        if c not in df.columns:
            df = df.with_columns([pl.lit(1.0).alias(f"{c}_missing"), pl.lit(1.0).alias(f"{c}_staleness")])
            continue

        # Upstream joins fill gaps with 0.0, so "missing" is null *or* exactly 0
        # (only null made it constant). Staleness = 1 - decay**(bars since the value
        # last changed): 0 right after a fresh release/headline, -> 1 as it ages.
        # (The old ``is_null.cum_min()`` was 0 after the first value, forever.)
        v = pl.col(c)
        changed = (v != v.shift(1)).fill_null(True)
        grp = changed.cast(pl.Int64).cum_sum()
        age = pl.int_range(pl.len()).over(grp)
        df = df.with_columns(
            [
                (v.is_null() | (v == 0.0)).cast(pl.Float64).alias(f"{c}_missing"),
                (1.0 - pl.lit(float(decay)).pow(age.cast(pl.Float64))).alias(f"{c}_staleness"),
            ]
        )
    return df


# === Latency feature ==================================================================

def compute_latency_feature(
    df: pl.DataFrame, latency_baseline_ms: float = 50.0, atr_col: str = "atr_6", atr_window_ref: int = 120
) -> pl.Series:
    n = len(df)
    latency = np.full(n, latency_baseline_ms, dtype=np.float32)
    if atr_col in df.columns:
        atr_vals = df[atr_col].fill_null(strategy="forward").to_numpy()
        med_atr = (
            df[atr_col]
            .fill_null(strategy="forward")
            .rolling_median(atr_window_ref, min_samples=10)
            .fill_null(0.0)
            .to_numpy()
        )
        atr_excess = np.maximum(0.0, atr_vals / np.maximum(med_atr, 1e-12) - 1.0)
        latency += latency_baseline_ms * 0.5 * atr_excess

    latency = np.clip(latency, latency_baseline_ms, latency_baseline_ms * 10.0)
    return pl.Series("expected_latency_ms", latency)


# === Circuit Breakers / Kill Switches ================================================

def circuit_breaker_features(
    max_position_pct: float = 0.05,
    max_daily_loss_pct: float = 0.03,
    max_drawdown_pct: float = 0.10,
    var_window: int = 252,
) -> list[pl.Expr]:
    """Generate circuit breaker / kill switch features."""
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    equity = (1 + ret).cum_prod()

    peak = equity.cum_max()
    drawdown = (equity - peak) / peak

    daily_ret = ret.rolling_sum(window_size=1440)

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
    """Position sizing limits based on ATR and account risk."""
    pair_upper = str(pair).upper()

    if "JPY" in pair_upper:
        pip_value = 1000.0 / pl.col("close")
    elif pair_upper.endswith("USD"):
        pip_value = pl.lit(10.0)
    elif pair_upper.startswith("USD"):
        pip_value = 10.0 / pl.col("close")
    else:
        pip_value = pl.lit(9.0)

    risk_budget = pl.lit(max_pos_pct)
    pip_mult = 10000.0 if "JPY" not in pair_upper else 100.0

    max_pos = risk_budget / (pl.col(atr_col) * pip_value * pip_mult)

    return [
        max_pos.alias("max_position_lots"),
        (pl.col(atr_col) * pip_value * pip_mult).alias("risk_per_lot"),
    ]


# === Feature Discovery / ONNX / Governance ===========================================

def feature_discovery_mutual_info(target: str = "ret_5", candidates: list[str] | None = None, window: int = 20) -> pl.Expr:
    """Mutual Information between candidates and target (approximation)."""
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


def onnx_compatible_features() -> list[str]:
    """Return list of feature names compatible with ONNX export."""
    return [
        "close", "open", "high", "low", "volume",
        "ret_5", "ret_20", "ret_60",
        "atr_6", "atr_20",
        "vol_20",
        "rsi_14",
        "macd", "macd_sig", "macd_hist",
        "bb_pct", "bb_width",
        "stoch_k", "stoch_d",
        "williams_r",
        "cci",
        "ofi",
        "obi_proxy",
        "vpin",
        "trend_regime", "range_regime", "volatility_regime",
        "time_sin", "time_cos",
        "day_sin", "day_cos",
    ]


def quantize_aware_features() -> list[str]:
    """Features suitable for INT8 quantization (bounded range)."""
    return [
        "stoch_k", "stoch_d",
        "rsi_14",
        "williams_r",
        "cci",
        "bb_pct",
        "time_sin", "time_cos",
        "day_sin", "day_cos",
    ]


def model_card_template(
    model_name: str,
    version: str,
    features: list[str],
    metrics: dict,
    data_window: str,
    known_limitations: list[str] | None = None,
) -> dict:
    """Generate standardized model card for auditability."""
    from datetime import datetime, timedelta, timezone

    UTC = timezone.utc
    _now = datetime.now(UTC)
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
            "pairs": ["EURUSD", "USDJPY", "GBPUSD"],
            "frequency": "1min",
        },
        "metrics": metrics,
        "known_limitations": known_limitations
        or [
            "Trained on 1-min data; may not generalize to other timeframes",
            "Regime-dependent performance; monitor regime features",
            "No on-chain/satellite data in current version",
        ],
        "governance": {
            "approved_for_production": False,
            "review_date": (_now + timedelta(days=91)).isoformat(),
        },
    }


def population_based_hpo_config() -> dict:
    """Configuration for Population-Based Training (PBT) style HPO."""
    return {
        "population_size": 10,
        "perturbation_interval": 10,
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
