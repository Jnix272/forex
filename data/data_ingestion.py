"""
data/data_ingestion.py
======================
High-resolution forex data ingestion and preprocessing.

Handles:
  - Loading tick / 1-min OHLCV data from CSV or Parquet
  - Fractional differentiation for stationarity
  - Bid-Ask spread validation (Golden Rule)
  - UTC timestamp normalization
  - Train/test splitting for walk-forward validation
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")


def _cfg_float(key: str, default: float) -> float:
    """Read a float from config.settings.MACRO_DATA with a safe fallback."""
    try:
        from config.settings import MACRO_DATA
        return float(MACRO_DATA.get(key, default))
    except Exception:
        return float(default)


def _cfg_str(key: str, default: str) -> str:
    """Read a string from config.settings.MACRO_DATA with a safe fallback."""
    try:
        from config.settings import MACRO_DATA
        val = MACRO_DATA.get(key, default)
        return str(val) if val is not None else default
    except Exception:
        return str(default)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR  (use when real broker data is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_tick_data(
    n_rows: int = 100_000,
    pair: str = "EURUSD",
    base_price: float = 1.0850,
    spread_pips: float = 0.5,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate realistic synthetic EUR/USD tick data for development/testing.
    Returns a Polars DataFrame.
    """
    rng = np.random.default_rng(seed)

    mu = 0.0
    sigma = 0.10
    dt = 1 / (252 * 24 * 3600)

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_rows)
    mid_prices = base_price * np.exp(np.cumsum(log_returns))

    # Fast datetime generation
    start = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
    timestamps = pd.date_range(start=start, periods=n_rows, freq="1s")

    half_spread = (spread_pips * 0.0001) / 2
    bid = mid_prices - half_spread
    ask = mid_prices + half_spread

    hour = timestamps.hour
    session_multiplier = np.where(
        ((hour >= 7) & (hour <= 10)) | ((hour >= 13) & (hour <= 16)),
        rng.uniform(2.0, 4.0, n_rows),
        rng.uniform(0.5, 1.5, n_rows),
    )
    volume = np.round(np.exp(rng.normal(3.5, 0.8, n_rows)) * session_multiplier).astype(int)

    df = pl.DataFrame({
        "timestamp_utc": timestamps,
        "bid": np.round(bid, 5),
        "ask": np.round(ask, 5),
        "mid": np.round(mid_prices, 5),
        "volume": volume,
        "spread": np.round(ask - bid, 5),
        "pair": pair,
    })
    
    # Ensure UTC timezone
    df = df.with_columns(pl.col("timestamp_utc").dt.replace_time_zone("UTC"))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_tick_data(
    filepath: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pl.DataFrame:
    """
    Load tick data from CSV or Parquet into a Polars DataFrame.
    Expects at minimum: timestamp, bid, ask columns.

    ``start``/``end`` are optional ISO UTC dates pushed into the lazy reader
    so only the requested range is materialised.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = _load_tick_data_frame(path, start=start, end=end)
    return _standardize_dataframe(df)


def _load_tick_data_frame(
    path: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pl.DataFrame:
    """
    Load tick data natively with Polars using lazy scanning so filters are
    pushed down to the file before materialisation (prevents OOM on the 16GB
    RAM development box when ingesting multi-year windows).
    """
    suffix = path.suffix.lower()

    lf = None
    if suffix == ".parquet":
        lf = pl.scan_parquet(path)
    elif suffix in (".csv", ".txt"):
        lf = pl.scan_csv(path, try_parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Push down date-range filters to the timestamp column when known.
    if start or end:
        lf = _filter_lazy_by_time(lf, start=start, end=end)

    return lf.collect()


def _filter_lazy_by_time(lf, start: Optional[str] = None, end: Optional[str] = None):
    """Apply a UTC date-range filter on a lazy frame's timestamp column."""
    ts_col = None
    for col in lf.columns:
        if col in ("timestamp_utc", "timestamp", "datetime", "ts_event", "__index_level_0__"):
            ts_col = col
            break
    if ts_col is None:
        return lf
    expr = lf
    if start:
        expr = expr.filter(pl.col(ts_col) >= pd.Timestamp(start, tz="UTC"))
    if end:
        expr = expr.filter(pl.col(ts_col) <= pd.Timestamp(end, tz="UTC"))
    return expr


def clean_bad_ticks(
    df: pl.DataFrame,
    z_thresh: float = 8.0,
    window: int = 60,
    *,
    mad_z_thresh: float = 6.0,
    spread_ratio: float = 8.0,
    spread_window: int = 120,
) -> pl.DataFrame:
    """
    Robust tick cleaning using a rolling Z-score plus a MAD-based (median
    absolute deviation) filter and spread sanity checks. Operates natively on
    a Polars DataFrame.

    The rolling mean/std Z-score path is retained for legacy parity, but the
    MAD filter is the primary detector: for heavy-tailed FX returns the mean is
    pulled by the very outliers we want to find, whereas MAD is scale-invariant
    under the median and is far more robust.

    Remediation replaces detected outlier mids with the rolling median and
    re-derives bid/ask around the corrected mid.
    """
    if "mid" not in df.columns:
        df = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    if "spread" not in df.columns:
        df = df.with_columns((pl.col("ask") - pl.col("bid")).clip(lower_bound=0.0).alias("spread"))

    # Add rolling stats
    df = df.with_columns([
        pl.col("mid").rolling_mean(window_size=window, min_samples=window).alias("rolling_mean"),
        pl.col("mid").rolling_std(window_size=window, min_samples=window).alias("rolling_std"),
        pl.col("mid").rolling_median(window_size=window).alias("rolling_median"),
    ])

    # Z-score
    df = df.with_columns(
        ((pl.col("mid") - pl.col("rolling_mean")) / (pl.col("rolling_std") + 1e-9)).abs().alias("z_score")
    )

    # MAD z-score: z = 0.6745 * (x - median) / MAD  (MAD ~ 0.6745*sigma for Gaussian)
    # MAD is computed vectorized as the rolling median of |x - rolling_median(x)|.
    df = df.with_columns(
        ((pl.col("mid") - pl.col("rolling_median")).abs()
         .rolling_median(window_size=window).alias("mad_scale"))
    )
    df = df.with_columns(
        (0.6745 * (pl.col("mid") - pl.col("rolling_median")) / (pl.col("mad_scale") + 1e-9)).abs().alias("mad_z_score")
    )

    # Spread sanity: flag ticks whose spread exceeds N x rolling median spread
    df = df.with_columns(
        pl.col("spread").rolling_median(window_size=spread_window).alias("rolling_med_spread")
    )
    df = df.with_columns(
        (pl.col("spread") > pl.col("rolling_med_spread") * spread_ratio).alias("spread_outlier")
    )

    outliers_cond = (
        ((pl.col("z_score") > z_thresh) | (pl.col("mad_z_score") > mad_z_thresh)) &
        ~pl.col("spread_outlier")  # don't replace legit wide-spread (news) ticks
    )
    outlier_count = df.filter(outliers_cond).shape[0]

    if outlier_count > 0:
        print(f"[DataIngestion] Cleaning {outlier_count} bad ticks "
              f"(z>{z_thresh} or MAD z>{mad_z_thresh})")
        # Replace outliers with median
        df = df.with_columns(
            pl.when(outliers_cond).then(pl.col("rolling_median")).otherwise(pl.col("mid")).alias("mid")
        )
        # Re-derive bid/ask
        df = df.with_columns(((pl.col("ask") - pl.col("bid")) / 2).alias("half_spread"))
        df = df.with_columns([
            pl.when(outliers_cond).then(pl.col("mid") - pl.col("half_spread")).otherwise(pl.col("bid")).alias("bid"),
            pl.when(outliers_cond).then(pl.col("mid") + pl.col("half_spread")).otherwise(pl.col("ask")).alias("ask"),
        ])

    df = df.drop([
        "rolling_mean", "rolling_std", "z_score",
        "rolling_median", "mad_scale", "mad_z_score",
        "rolling_med_spread", "spread_outlier",
    ])
    if "spread" not in df.columns:
        df = df.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))
    return df


def _standardize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Enforce UTC timezone, sort by time, validate required columns.
    """
    required = {"bid", "ask"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"DataFrame must contain columns: {required}. Got: {set(df.columns)}")

    # Ensure timestamp column and UTC timezone
    # PIPE-010: recognise pandas-exported index columns (__index_level_0__) and
    # Databento's ts_event in addition to the conventional names. Without this,
    # raw Dukascopy parquet exports raise "No timestamp column found".
    time_col = None
    for col in ["timestamp", "timestamp_utc", "datetime", "ts_event", "time", "__index_level_0__"]:
        if col in df.columns:
            time_col = col
            break
            
    if time_col:
        df = df.rename({time_col: "timestamp_utc"})
        
        # Make sure it's datetime and localized
        if df.schema["timestamp_utc"] == pl.Utf8:
            df = df.with_columns(pl.col("timestamp_utc").str.to_datetime())
            
        # Ensure timezone is UTC
        if df.schema["timestamp_utc"].time_zone is None:
            df = df.with_columns(pl.col("timestamp_utc").dt.replace_time_zone("UTC"))
        elif df.schema["timestamp_utc"].time_zone != "UTC":
            df = df.with_columns(pl.col("timestamp_utc").dt.convert_time_zone("UTC"))
    else:
        raise ValueError("No timestamp column found")

    df = df.sort("timestamp_utc")

    if "mid" not in df.columns:
        df = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))

    df = clean_bad_ticks(
        df,
        z_thresh=_cfg_float("bad_tick_z_thresh", 8.0),
        window=int(_cfg_float("bad_tick_window", 60)),
        mad_z_thresh=_cfg_float("bad_tick_mad_z_thresh", 6.0),
        spread_ratio=_cfg_float("bad_tick_spread_ratio", 8.0),
        spread_window=int(_cfg_float("bad_tick_spread_window", 120)),
    )

    if "spread" not in df.columns:
        df = df.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))

    # Golden Rule check
    bad_count = df.filter(pl.col("spread") <= 0).shape[0]
    if bad_count > 0:
        warnings.warn(
            f"⚠️  {bad_count} rows have zero or negative spread. "
            "Filtering these out — always include real bid/ask data!"
        )
        df = df.filter(pl.col("spread") > 0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLING (tick -> OHLCV bars)
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_bars(df: pl.DataFrame, freq: str = "1min") -> pl.DataFrame:
    """
    Resample tick data to OHLCV bars using Polars group_by_dynamic.
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
        
    if (len(df) == 0):
        return pl.DataFrame()

    every = _pandas_freq_to_polars(freq)

    bars = (
        df
        .sort("timestamp_utc")
        .group_by_dynamic("timestamp_utc", every=every, label="left", closed="left")
        .agg([
            pl.col("mid").first().alias("open"),
            pl.col("mid").max().alias("high"),
            pl.col("mid").min().alias("low"),
            pl.col("mid").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("spread").mean().alias("spread_avg"),
            pl.col("bid").last().alias("bid_close"),
            pl.col("ask").last().alias("ask_close"),
        ])
        .drop_nulls(subset=["open"])
        .sort("timestamp_utc")
    )

    # Shared weekend / holiday / thin-liquidity / dead-bar filters.
    bars = _apply_market_filters(bars)

    return bars


def _pandas_freq_to_polars(freq: str) -> str:
    """Translate the common pandas-style bar frequency strings."""
    normalized = freq.strip().lower()
    mapping = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    return mapping.get(normalized, normalized)


# ─────────────────────────────────────────────────────────────────────────────
# INFORMATION BARS  (López de Prado: tick / volume / dollar bars)
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_tick_bars(df: pl.DataFrame, n_ticks: int = 500) -> pl.DataFrame:
    """Build bars of exactly ``n_ticks`` ticks each (fixed-size tick bars)."""
    return _resample_information_bars(df, bar_type="tick", threshold=n_ticks)


def resample_to_volume_bars(df: pl.DataFrame, volume_target: float = 10_000.0) -> pl.DataFrame:
    """Build bars that accumulate ``volume_target`` units of volume each."""
    return _resample_information_bars(df, bar_type="volume", threshold=volume_target)


def resample_to_dollar_bars(df: pl.DataFrame, dollar_target: float = 1_000_000.0) -> pl.DataFrame:
    """Build bars that accumulate ``dollar_target`` USD of notional volume each."""
    return _resample_information_bars(df, bar_type="dollar", threshold=dollar_target)


def _resample_information_bars(
    df: pl.DataFrame,
    bar_type: str = "tick",
    threshold: float = 500,
) -> pl.DataFrame:
    """
    Shared core for information (sampled-by-event) bars.

    Instead of bucketing on wall-clock time, bars close when a threshold of
    ticks / volume / notional volume has accumulated. Event-driven sampling
    produces statistically nicer properties than time bars: the number of
    observations per unit of time adapts to market activity, which improves
    normality of returns and reduces volatility clustering artefacts.
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    if len(df) == 0:
        return pl.DataFrame()

    df = df.sort("timestamp_utc")
    if "mid" not in df.columns:
        df = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))

    # Event-size accumulator per bar.
    if bar_type == "tick":
        event = pl.lit(1.0).alias("_event")
    elif bar_type == "volume":
        event = pl.col("volume").cast(pl.Float64).fill_null(0.0).alias("_event")
    elif bar_type == "dollar":
        event = (pl.col("volume").cast(pl.Float64) * pl.col("mid")).fill_null(0.0).alias("_event")
    else:
        raise ValueError(f"Unknown bar_type: {bar_type} (tick|volume|dollar)")

    df = df.with_columns(event)
    df = df.with_columns(
        (pl.col("_event").cum_sum() / threshold).floor().cast(pl.Int64).alias("_bar_id")
    )

    # Any bar-id from the first, always-positive bucket onwards is valid; the
    # remainder after the last full bar is dropped so bars are homogeneous.
    bars = (
        df.group_by("_bar_id")
        .agg([
            pl.col("timestamp_utc").first().alias("timestamp_utc"),
            pl.col("mid").first().alias("open"),
            pl.col("mid").max().alias("high"),
            pl.col("mid").min().alias("low"),
            pl.col("mid").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("spread").mean().alias("spread_avg"),
            pl.col("bid").last().alias("bid_close"),
            pl.col("ask").last().alias("ask_close"),
        ])
        .filter(pl.col("_bar_id") > 0)
        .sort("timestamp_utc")
        .drop("_bar_id")
    )

    return _apply_market_filters(bars)


def _apply_market_filters(bars: pl.DataFrame) -> pl.DataFrame:
    """
    Apply the weekend / holiday / thin-liquidity / dead-bar filters that are
    shared between time bars and information bars.
    """
    if len(bars) == 0:
        return bars

    # 1. Strict Weekend Filter
    bars = bars.filter(
        ~((pl.col("timestamp_utc").dt.weekday() == 6) |
          ((pl.col("timestamp_utc").dt.weekday() == 5) & (pl.col("timestamp_utc").dt.hour() >= 22)) |
          ((pl.col("timestamp_utc").dt.weekday() == 7) & (pl.col("timestamp_utc").dt.hour() < 22)))
    )

    # 2. Holiday filter (exchange calendar + fixed-date fallback)
    bars = _filter_market_holidays(bars)

    # 3. Thin-liquidity day filter via volume threshold
    if "volume" in bars.columns:
        _daily_vol = bars.with_columns(
            pl.col("timestamp_utc").dt.date().alias("_date")
        ).group_by("_date").agg(pl.col("volume").sum().alias("_day_vol"))
        _median_vol = _daily_vol["_day_vol"].median()
        if _median_vol is not None and _median_vol > 0:
            _thin_days = _daily_vol.filter(
                pl.col("_day_vol") < _median_vol * 0.2
            )["_date"]
            if len(_thin_days) > 0:
                bars = bars.filter(
                    ~pl.col("timestamp_utc").dt.date().is_in(_thin_days.to_list())
                )

    # 4. Dead Bars Filter
    bars = bars.filter(
        ~((pl.col("high") == pl.col("low")) & (pl.col("volume") < 1e-6))
    )

    return bars


def _filter_market_holidays(bars: pl.DataFrame) -> pl.DataFrame:
    """
    Filter bars falling on recognised market holidays.

    Uses ``pandas_market_calendars`` FOREX calendar when available (covers
    floating holidays such as Good Friday, Thanksgiving, and JPY bank
    holidays), otherwise falls back to the fixed-date mask + a thin-liquidity
    guard handled by the caller.
    """
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("FOREX")
        start = bars["timestamp_utc"].min()
        end = bars["timestamp_utc"].max()
        if start is not None and end is not None:
            valid = cal.valid_days(
                start_date=start.date(), end_date=end.date(),
            )
            valid_tz = pl.Series("_valid_day", valid).dt.replace_time_zone("UTC")
            valid_days = valid_tz.dt.date().to_list()
            return bars.filter(pl.col("timestamp_utc").dt.date().is_in(valid_days))
    except Exception:
        pass

    # Fallback: fixed-date mask (universal + US/EUR/GBP).
    _holiday_filter = (
        ((pl.col("timestamp_utc").dt.month() == 12) & (pl.col("timestamp_utc").dt.day() == 25)) |
        ((pl.col("timestamp_utc").dt.month() == 1) & (pl.col("timestamp_utc").dt.day() == 1)) |
        ((pl.col("timestamp_utc").dt.month() == 12) & (pl.col("timestamp_utc").dt.day() == 26)) |
        ((pl.col("timestamp_utc").dt.month() == 7) & (pl.col("timestamp_utc").dt.day() == 4)) |
        ((pl.col("timestamp_utc").dt.month() == 12) & (pl.col("timestamp_utc").dt.day() == 24)) |
        ((pl.col("timestamp_utc").dt.month() == 12) & (pl.col("timestamp_utc").dt.day() == 31))
    )
    return bars.filter(~_holiday_filter)


# ─────────────────────────────────────────────────────────────────────────────
# GAP DETECTION & INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

def detect_bar_gaps(df: pl.DataFrame, freq_minutes: int = 1) -> dict:
    """
    Detect missing bars inside market-open periods.

    Returns a dict with per-gap summary: total missing rows, longest gap
    (minutes), and a sample of gap (start, end, duration) tuples.
    """
    if len(df) == 0 or "timestamp_utc" not in df.columns:
        return {"n_gaps": 0, "n_missing_rows": 0, "longest_gap_minutes": 0, "gaps": []}

    s = df.sort("timestamp_utc")["timestamp_utc"]
    diff = s.diff().dt.total_seconds().alias("_dt")
    idx = pl.Series(range(len(s)))
    d = df.select([
        idx.alias("_i"),
        s.alias("timestamp_utc"),
        diff,
    ]).filter(pl.col("_dt") > freq_minutes * 60)

    gaps = []
    for row in d.iter_rows(named=True):
        gap_min = int(row["_dt"] // 60)
        gaps.append({
            "after": row["timestamp_utc"],
            "duration_minutes": gap_min,
        })

    n_missing = sum(int(g["duration_minutes"]) - 1 for g in gaps)
    return {
        "n_gaps": len(gaps),
        "n_missing_rows": n_missing,
        "longest_gap_minutes": max((g["duration_minutes"] for g in gaps), default=0),
        "gaps": gaps[:10],
    }


def fill_gaps(df: pl.DataFrame, policy: str = "drop", freq_minutes: int = 1) -> pl.DataFrame:
    """
    Fill missing bars according to ``policy``:
      - "drop":        (default) no interpolation, return unchanged
      - "ffill":       forward-fill OHLCV/spread over gaps <= gap_max_minutes
      - "interpolate": linear interpolation of close (and spread) over gaps
    """
    if policy not in ("drop", "ffill", "interpolate"):
        raise ValueError(f"Unknown gap policy: {policy} (drop|ffill|interpolate)")
    if len(df) == 0 or policy == "drop":
        return df

    report = detect_bar_gaps(df, freq_minutes=freq_minutes)
    if report["n_gaps"] == 0:
        return df

    # Build a full regular grid and reindex.
    df = df.sort("timestamp_utc")
    s = df["timestamp_utc"]
    grid = pl.DataFrame({
        "timestamp_utc": pl.datetime_range(
            start=s.min(),
            end=s.max(),
            interval=f"{freq_minutes}m",
            time_zone="UTC",
            eager=True,
        )
    }).sort("timestamp_utc")

    # Keep grid points whose distance to the most recent real bar is within
    # gap_max_minutes. This fills small gaps but never bridges weekends/holidays
    # (a Mon 07:00 bar anchors to the Fri close, far beyond the max gap).
    max_gap = int(_cfg_float("gap_max_minutes", 5))
    anchors = df.select(s.alias("timestamp_utc"))
    grid = grid.join_asof(
        anchors,
        on="timestamp_utc",
        strategy="backward",
        coalesce=False,
    ).with_columns(
        (pl.col("timestamp_utc").cast(pl.Int64)
         - pl.col("timestamp_utc_right").cast(pl.Int64)).alias("_since_bar")
    ).filter(
        (pl.col("_since_bar").is_null())
        | (pl.col("_since_bar") <= max_gap * 60 * 1_000_000)  # µs since last bar
    ).drop(["_since_bar", "timestamp_utc_right"])

    joined = grid.join(df, on="timestamp_utc", how="left")

    if policy == "ffill":
        joined = joined.with_columns([
            pl.col("open").forward_fill(),
            pl.col("high").forward_fill(),
            pl.col("low").forward_fill(),
            pl.col("close").forward_fill(),
            pl.col("volume").fill_null(0.0),
            pl.col("spread_avg").forward_fill(),
            pl.col("bid_close").forward_fill(),
            pl.col("ask_close").forward_fill(),
        ])
    else:  # interpolate
        joined = joined.with_columns([
            pl.col("open").interpolate(),
            pl.col("high").interpolate(),
            pl.col("low").interpolate(),
            pl.col("close").interpolate(),
            pl.col("volume").fill_null(0.0),
            pl.col("spread_avg").interpolate(),
            pl.col("bid_close").interpolate(),
            pl.col("ask_close").interpolate(),
        ])

    return joined.drop_nulls(subset=["open"]).sort("timestamp_utc")


# ─────────────────────────────────────────────────────────────────────────────
# TICK SAMPLING ANALYSIS  (Lomb-Scargle)
# ─────────────────────────────────────────────────────────────────────────────

def detect_tick_sampling(df: pl.DataFrame, min_periods: int = 20) -> dict:
    """
    Estimate the dominant tick sampling frequency and how regular the arrival
    pattern is, using a Lomb-Scargle periodogram on the tick inter-arrival
    times.

    Returns:
      - ``n_ticks``, ``span_hours``
      - ``median_iat_ms``, ``std_iat_ms``      inter-arrival time stats
      - ``dominant_period_s``, ``power``       dominant period and its peak power
      - ``irregularity``                       score 0..1 (higher = more irregular)
      - ``regular``                            True when sampling is near-regular
    """
    if len(df) < min_periods or "timestamp_utc" not in df.columns:
        return {"n_ticks": len(df), "regular": None, "reason": "insufficient_ticks"}

    ts = df["timestamp_utc"].sort().dt.epoch("s").to_numpy()  # seconds since epoch
    iat = np.diff(ts)
    iat = iat[iat > 0]
    if len(iat) < min_periods:
        return {"n_ticks": len(df), "regular": None, "reason": "insufficient_iat"}

    median_iat = float(np.median(iat))
    std_iat = float(np.std(iat))

    # Regularity: coefficient of variation of inter-arrival times.
    cv = std_iat / (median_iat + 1e-9)
    irregularity = float(np.clip((cv - 0.1) / 2.0, 0.0, 1.0))

    # Lomb-Scargle on inter-arrival times (frequency axis in cycles/sec).
    dominant_period_s = None
    power = None
    try:
        from scipy.signal import lombscargle
        t = np.arange(len(iat), dtype=float)
        # Normalise iat to give a well-scaled periodogram.
        y = iat - iat.mean()
        fmax = 1.0 / (2 * max(np.median(iat), 1e-6))
        freqs = np.linspace(1e-6, fmax, 256)
        pgram = lombscargle(t, y, freqs)
        if pgram.max() > 0:
            peak_f = freqs[int(np.argmax(pgram))]
            dominant_period_s = float(1.0 / peak_f) if peak_f > 0 else None
            power = float(pgram.max())
    except Exception:
        pass

    return {
        "n_ticks": len(df),
        "span_hours": float((ts[-1] - ts[0]) / 3600.0) if len(ts) > 1 else 0.0,
        "median_iat_ms": float(median_iat * 1000.0),
        "std_iat_ms": float(std_iat * 1000.0),
        "dominant_period_s": dominant_period_s,
        "periodogram_power": power,
        "irregularity": irregularity,
        "regular": irregularity < 0.25,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FRACTIONAL DIFFERENTIATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_weights_ffd(d: float, thres: float = 1e-5) -> np.ndarray:
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])


def fracDiff_FFD(series: pl.Series, d: float = 0.4, thres: float = 1e-5) -> pl.Series:
    """
    Apply Fixed-Window Fractional Differentiation to a price series.
    Accepts and returns a Polars Series.
    """
    w = _get_weights_ffd(d, thres)
    width = len(w) - 1
    
    if width >= len(series):
        return pl.Series(series.name, [None] * len(series), dtype=pl.Float64)
        
    res_values = np.convolve(series.to_numpy(), w, mode='valid')
    # Pad with NaNs at the beginning to maintain original length
    padded = np.concatenate([np.full(width, np.nan), res_values])
    return pl.Series(series.name, padded)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class ForexDataPipeline:
    def __init__(
        self,
        bar_freq: str = "1min",
        frac_diff_order: float = 0.4,
        apply_frac_diff: bool = True,
        session_filter: bool = True,
        session_start_utc: str = "07:00",
        session_end_utc: str = "21:00",
        *,
        bar_type: str = "time",          # "time" | "tick" | "volume" | "dollar"
        info_bar_threshold: float = 500.0,
        session_mode: str = "fixed",     # "fixed" | "dst"
        add_session_label: bool = False,
        gap_policy: str = "drop",        # "drop" | "ffill" | "interpolate"
    ):
        self.bar_freq = bar_freq
        self.frac_diff_order = frac_diff_order
        self.apply_frac_diff = apply_frac_diff
        self.session_filter = session_filter
        self.bar_type = str(bar_type).lower()
        self.info_bar_threshold = float(info_bar_threshold)
        self.session_mode = str(session_mode).lower()
        self.add_session_label = bool(add_session_label)
        self.gap_policy = _cfg_str("gap_policy", gap_policy)

        h, m = map(int, session_start_utc.split(":"))
        self.session_start = h * 3600 + m * 60

        h, m = map(int, session_end_utc.split(":"))
        self.session_end = h * 3600 + m * 60

        if self.session_mode == "dst":
            # Load DST-aware session windows from the shared config.
            try:
                from config.settings import LIVE_RISK as _LR
                self._sessions = _LR["session_limits"]
            except Exception:
                self._sessions = None

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        print(f"[Pipeline] Raw tick rows: {len(df):,}")

        if self.bar_type == "time":
            bars = resample_to_bars(df, freq=self.bar_freq)
        elif self.bar_type in ("tick", "volume", "dollar"):
            thresh = self.info_bar_threshold
            _resamplers = {
                "tick": lambda d: resample_to_tick_bars(d, n_ticks=int(thresh)),
                "volume": lambda d: resample_to_volume_bars(d, volume_target=thresh),
                "dollar": lambda d: resample_to_dollar_bars(d, dollar_target=thresh),
            }
            bars = _resamplers[self.bar_type](df)
        else:
            raise ValueError(
                f"Unknown bar_type: {self.bar_type} (time|tick|volume|dollar)"
            )
        print(f"[Pipeline] Bars after resampling ({self.bar_type}): {len(bars):,}")

        # Gap detection / interpolation before session filtering.
        if self.gap_policy != "drop" and len(bars) > 0:
            report = detect_bar_gaps(bars)
            if report["n_gaps"] > 0:
                print(f"[Pipeline] Detected {report['n_gaps']} gaps "
                      f"({report['n_missing_rows']} missing rows, "
                      f"longest {report['longest_gap_minutes']} min). "
                      f"Policy: {self.gap_policy}")
            bars = fill_gaps(bars, policy=self.gap_policy)

        if self.session_filter and not (len(bars) == 0):
            if self.session_mode == "dst" and self._sessions:
                bars = self._apply_dst_sessions(bars)
                if self.add_session_label:
                    # _apply_dst_sessions already attaches a session_label.
                    pass
                else:
                    bars = bars.drop("session_label")
            else:
                # Fixed-UTC window (legacy behavior).
                bars = bars.with_columns(
                    ((pl.col("timestamp_utc").dt.hour() * 3600) +
                     (pl.col("timestamp_utc").dt.minute() * 60)).alias("_time_sec")
                )
                bars = bars.filter(
                    (pl.col("_time_sec") >= self.session_start) &
                    (pl.col("_time_sec") <= self.session_end) &
                    (pl.col("timestamp_utc").dt.weekday() <= 5)
                ).drop("_time_sec")
            print(f"[Pipeline] Bars after session filter: {len(bars):,}")

        if (len(bars) == 0):
            print("[Pipeline] Warning: No bars left after filtering. Returning empty DataFrame.")
            return bars

        if self.apply_frac_diff:
            bars = bars.with_columns(
                fracDiff_FFD(bars["close"], d=self.frac_diff_order).alias("close_ffd")
            )
            print(f"[Pipeline] Fractional diff applied (d={self.frac_diff_order})")

        bars = bars.drop_nulls()
        print(f"[Pipeline] Final bar count after cleaning: {len(bars):,}")

        return bars

    def _apply_dst_sessions(self, bars: pl.DataFrame) -> pl.DataFrame:
        """
        Label each bar with its DST-aware trading session and keep only bars
        inside a recognised session (or the configured session window).

        Session windows are defined in local time (zoneinfo) so UTC boundaries
        shift correctly across DST transitions in March / November.
        """
        sessions = self._sessions or {}
        if not sessions:
            return bars.with_columns(pl.lit("off").alias("session_label"))

        ts = bars["timestamp_utc"]
        label = pl.lit("off").cast(pl.Utf8).alias("session_label")

        # Build, for each session, a boolean mask of bars whose LOCAL time in
        # that session's timezone falls inside the session's local hours.
        local_cols = {}
        for name, cfg in sessions.items():
            if name == "off":
                continue
            tz = cfg.get("tz")
            hours = cfg.get("hours_local")
            if tz is None or hours is None:
                continue
            local = ts.dt.convert_time_zone(str(tz))
            local_cols[name] = local

        base = bars.with_columns(ts.alias("_ts"))
        col_name = "_session"
        base = base.with_columns(pl.lit("off").alias(col_name))
        for name, local in local_cols.items():
            base = base.with_columns(local.alias(f"_local_{name}"))

        # For each session, set the label where local time is within hours_local.
        for name in list(local_cols.keys()):
            start_h, start_m = sessions[name]["hours_local"][0].hour, sessions[name]["hours_local"][0].minute
            end_h, end_m = sessions[name]["hours_local"][1].hour, sessions[name]["hours_local"][1].minute
            lt = pl.col(f"_local_{name}")
            in_window = (
                ((lt.dt.hour() > start_h) | ((lt.dt.hour() == start_h) & (lt.dt.minute() >= start_m))) &
                ((lt.dt.hour() < end_h) | ((lt.dt.hour() == end_h) & (lt.dt.minute() <= end_m)))
            )
            base = base.with_columns(
                pl.when(in_window).then(pl.lit(name)).otherwise(pl.col(col_name)).alias(col_name)
            )

        drop_cols = [f"_local_{n}" for n in local_cols.keys()]
        result = base.drop(["_ts"] + drop_cols).rename({col_name: "session_label"})

        if self.add_session_label:
            return result
        return result.filter(pl.col("session_label") != "off")

    def train_test_split(
        self,
        bars: pl.DataFrame,
        train_ratio: float = 0.7,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        split_idx = int(len(bars) * train_ratio)
        return bars.slice(0, split_idx), bars.slice(split_idx)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-START HELPER
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate(
    filepath: Optional[str] = None,
    n_rows: int = 50_000,
    *,
    source: Optional[str] = None,
    pair: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pl.DataFrame:
    """
    Load real tick data or generate synthetic data.
    Now returns a Polars DataFrame.
    """
    if source is not None or (pair is not None and start is not None):
        from data.sources import ForexDataManager
        _pair = (pair or "EURUSD").upper().replace("/", "")
        _source = source or "dukascopy"
        try:
            mgr = ForexDataManager(verbose=False)
            df = mgr.load(_pair, source=_source, start=start, end=end)
            if not df.empty:
                # mgr.load returns Pandas, convert to Polars
                df_pl = pl.from_pandas(df.reset_index())
                if "timestamp" in df_pl.columns:
                    df_pl = df_pl.rename({"timestamp": "timestamp_utc"})
                return df_pl
        except Exception as e:
            print(f"[DataLoader] {_source} failed for {_pair}: {e} — falling back to synthetic data")
        
        base_prices = {
            "EURUSD": 1.0850, "GBPUSD": 1.2700, "USDJPY": 148.50,
            "AUDUSD": 0.6550, "USDCAD": 1.3600, "USDCHF": 0.8950,
            "EURGBP": 0.8540, "NZDUSD": 0.6100, "EURJPY": 161.00,
            "GBPJPY": 188.00,
        }
        base_price = base_prices.get(_pair, 1.0850)
        spread_pips = 0.02 if _pair.endswith("JPY") else 0.5
        print(f"[DataLoader] Generating synthetic ticks for {_pair}")
        return generate_synthetic_tick_data(
            n_rows=n_rows, pair=_pair,
            base_price=base_price, spread_pips=spread_pips,
        )

    if filepath and Path(filepath).exists():
        print(f"[DataLoader] Loading real data from {filepath}")
        return load_tick_data(filepath)

    print(f"[DataLoader] Generating {n_rows:,} synthetic EUR/USD ticks for development")
    return generate_synthetic_tick_data(n_rows=n_rows)


if __name__ == "__main__":
    from data.data_ingestion import (
        load_or_generate, ForexDataPipeline,
        resample_to_tick_bars, resample_to_volume_bars, resample_to_dollar_bars,
        detect_bar_gaps, fill_gaps, detect_tick_sampling, clean_bad_ticks,
    )

    ticks = load_or_generate(n_rows=10_000)
    print(f"\nSample tick data:\n{ticks.head()}")
    print(f"\nData types:\n{ticks.schema}")

    pipeline = ForexDataPipeline(bar_freq="5min")
    bars = pipeline.run(ticks)
    train, test = pipeline.train_test_split(bars)
    print(f"\nTrain: {len(train):,} bars | Test: {len(test):,} bars")

    # Information bars
    tb = resample_to_tick_bars(ticks, n_ticks=200)
    vb = resample_to_volume_bars(ticks, volume_target=500.0)
    db = resample_to_dollar_bars(ticks, dollar_target=500.0)
    print(f"Tick bars: {len(tb):,} | Volume bars: {len(vb):,} | Dollar bars: {len(db):,}")

    # Tick sampling analysis (Lomb-Scargle)
    sampling = detect_tick_sampling(ticks)
    print(f"Tick sampling: {sampling}")

    # Gap detection on a full-day bar set
    full_day = ForexDataPipeline(bar_freq="1min", session_filter=False,
                                 apply_frac_diff=False).run(ticks)
    gaps = detect_bar_gaps(full_day)
    print(f"Gaps: {gaps['n_gaps']} | missing rows: {gaps['n_missing_rows']}")

    # DST-aware session labelling
    dst_ticks = load_or_generate(n_rows=100_000)  # ~27h of 1s ticks
    dst_pipeline = ForexDataPipeline(bar_freq="5min", session_filter=True,
                                     session_mode="dst", add_session_label=True,
                                     apply_frac_diff=False)
    dst_bars = dst_pipeline.run(dst_ticks)
    if "session_label" in dst_bars.columns:
        print(f"Session labels: {dst_bars['session_label'].value_counts().to_dict()}")
    else:
        print("(session_label column not present — no recognised session windows)")

    # Real Dukascopy fixture (verifies __index_level_0__ handling)
    import glob
    real = sorted(glob.glob("data/raw/dukascopy/EURUSD/2024/01/*.parquet"))
    if real:
        df = load_tick_data(real[0])
        print(f"Loaded real Dukascopy tick file: {len(df):,} rows, cols={df.columns[:4]}")
    else:
        print("No real Dukascopy fixture found — skipping.")
