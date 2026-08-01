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

def load_tick_data(filepath: str) -> pl.DataFrame:
    """
    Load tick data from CSV or Parquet into a Polars DataFrame.
    Expects at minimum: timestamp, bid, ask columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = _load_tick_data_frame(path)
    return _standardize_dataframe(df)


def _load_tick_data_frame(path: Path) -> pl.DataFrame:
    """Load tick data natively with Polars."""
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pl.scan_parquet(path).collect()
    if suffix in (".csv", ".txt"):
        return pl.scan_csv(path, try_parse_dates=True).collect()

    raise ValueError(f"Unsupported file format: {path.suffix}")


def clean_bad_ticks(df: pl.DataFrame, z_thresh: float = 8.0, window: int = 60) -> pl.DataFrame:
    """
    Uses a rolling Z-score and median filter to remove outliers.
    Operates natively on Polars DataFrame.
    """
    if "mid" not in df.columns:
        df = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))

    # Add rolling stats
    df = df.with_columns([
        pl.col("mid").rolling_mean(window_size=window, min_samples=window).alias("rolling_mean"),
        pl.col("mid").rolling_std(window_size=window, min_samples=window).alias("rolling_std"),
    ])
    
    # Calculate Z-score
    df = df.with_columns(
        ((pl.col("mid") - pl.col("rolling_mean")) / (pl.col("rolling_std") + 1e-9)).abs().alias("z_score")
    )
    
    outliers_cond = pl.col("z_score") > z_thresh
    outlier_count = df.filter(outliers_cond).shape[0]

    if outlier_count > 0:
        print(f"[DataIngestion] Cleaning {outlier_count} bad ticks (> {z_thresh} sigma)")
        df = df.with_columns(
            pl.col("mid").rolling_median(window_size=window).alias("rolling_median")
        )
        
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
        df = df.drop(["rolling_median", "half_spread"])

    df = df.drop(["rolling_mean", "rolling_std", "z_score"])
    return df


def _standardize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Enforce UTC timezone, sort by time, validate required columns.
    """
    required = {"bid", "ask"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"DataFrame must contain columns: {required}. Got: {set(df.columns)}")

    # Ensure timestamp column and UTC timezone
    time_col = None
    for col in ["timestamp", "timestamp_utc", "datetime"]:
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

    df = clean_bad_ticks(df)

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

    # 1. Strict Weekend Filter: Drop Friday >= 22:00 UTC, Saturday, Sunday < 22:00 UTC
    # In Polars dt.weekday(), Mon=1, Fri=5, Sat=6, Sun=7
    bars = bars.filter(
        ~((pl.col("timestamp_utc").dt.weekday() == 6) |
          ((pl.col("timestamp_utc").dt.weekday() == 5) & (pl.col("timestamp_utc").dt.hour() >= 22)) |
          ((pl.col("timestamp_utc").dt.weekday() == 7) & (pl.col("timestamp_utc").dt.hour() < 22)))
    )

    # 2. Holiday Filter: DS-006 — expanded beyond just Dec 25 / Jan 1
    # Covers major forex market closures across USD, EUR, GBP, JPY calendars
    _holiday_filter = (
        # Universal
        ((pl.col("timestamp_utc").dt.month() == 12) & (pl.col("timestamp_utc").dt.day() == 25)) |  # Christmas
        ((pl.col("timestamp_utc").dt.month() == 1)  & (pl.col("timestamp_utc").dt.day() == 1))  |  # New Year
        ((pl.col("timestamp_utc").dt.month() == 12) & (pl.col("timestamp_utc").dt.day() == 26)) |  # Boxing Day (GBP)
        # US holidays (fixed dates — floating ones handled by volume filter below)
        ((pl.col("timestamp_utc").dt.month() == 7)  & (pl.col("timestamp_utc").dt.day() == 4))  |  # Independence Day
        ((pl.col("timestamp_utc").dt.month() == 12) & (pl.col("timestamp_utc").dt.day() == 24)) |  # Christmas Eve (early close)
        ((pl.col("timestamp_utc").dt.month() == 12) & (pl.col("timestamp_utc").dt.day() == 31))    # New Year's Eve (early close)
    )
    bars = bars.filter(~_holiday_filter)

    # DS-006: Detect thin-liquidity days (floating holidays like Good Friday,
    # Thanksgiving, Japanese bank holidays) via volume threshold.
    # Days with <20% of median daily volume are likely market holidays.
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
                    ~pl.col("timestamp_utc").dt.date().is_in(_thin_days)
                )

    # 3. Dead Bars Filter: Drop bars with near-zero volume and no price movement
    bars = bars.filter(
        ~((pl.col("high") == pl.col("low")) & (pl.col("volume") < 1e-6))
    )

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
    ):
        self.bar_freq = bar_freq
        self.frac_diff_order = frac_diff_order
        self.apply_frac_diff = apply_frac_diff
        self.session_filter = session_filter
        
        h, m = map(int, session_start_utc.split(":"))
        self.session_start = h * 3600 + m * 60
        
        h, m = map(int, session_end_utc.split(":"))
        self.session_end = h * 3600 + m * 60

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        print(f"[Pipeline] Raw tick rows: {len(df):,}")

        bars = resample_to_bars(df, freq=self.bar_freq)
        print(f"[Pipeline] Bars after resampling ({self.bar_freq}): {len(bars):,}")

        if self.session_filter and not (len(bars) == 0):
            # Get hour and minute in seconds to filter times
            bars = bars.with_columns(
                ((pl.col("timestamp_utc").dt.hour() * 3600) + (pl.col("timestamp_utc").dt.minute() * 60)).alias("_time_sec")
            )
            # 1 = Mon, ..., 5 = Fri, 6 = Sat, 7 = Sun in Polars (ISO 8601)
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
    ticks = load_or_generate(n_rows=10_000)
    print(f"\nSample tick data:\n{ticks.head()}")
    print(f"\nData types:\n{ticks.schema}")

    pipeline = ForexDataPipeline(bar_freq="5min")
    bars = pipeline.run(ticks)
    train, test = pipeline.train_test_split(bars)
    print(f"\nTrain: {len(train):,} bars | Test: {len(test):,} bars")
