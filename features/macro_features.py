"""
features/macro_features.py
===========================
Yield-spread, carry, and yield-curve features for all 9 forex pairs.

Interest-rate differentials are the dominant long-run driver of FX pairs:
  EUR/USD  <- US10Y - DE10Y  (US-Germany spread)
  USD/JPY  <- US10Y - JP10Y  (US-Japan spread)
  GBP/USD  <- US10Y - GB10Y
  AUD/USD  <- AU10Y - US10Y  (inverted: AUD is the high-yielder)
  USD/CAD  <- US10Y - CA10Y
  NZD/USD  <- NZ10Y - US10Y  (inverted: NZD is the high-yielder)
  EUR/GBP  <- DE10Y - GB10Y
  EUR/JPY  <- DE10Y - JP10Y
  USD/CHF  <- US10Y - CH10Y

Data sources (in priority order):
  1. FRED API (fredapi library) - free, no rate limit for daily data.
     Set FRED_API_KEY env var. Key at https://fred.stlouisfed.org/
  2. Synthetic fallback - correlated random walks calibrated to historical
     levels, used when FRED is unavailable.

Bugs fixed vs original:
  - tz_localize crash when FRED index already has a timezone.
  - Module-level warnings.filterwarnings("ignore") removed.
  - FRED retry with exponential backoff added.
  - yield_curve_slope (US 2Y-10Y) added as recession / FX driver.
  - Carry and spread features added for all 9 pairs (was only 5).
  - _align handles timezone mismatches safely.

Features produced (all daily, forward-filled to bars.index via ffill):
  spread_us_de, spread_us_jp, spread_us_gb, spread_us_au, spread_us_ca
  spread_us_nz, spread_de_gb, spread_de_jp, spread_us_ch
  carry_eur, carry_jpy, carry_gbp, carry_aud, carry_cad
  carry_nzd, carry_eurgbp, carry_eurjpy, carry_chf
  yield_momentum_5d, yield_momentum_20d      (on primary US-DE spread)
  yield_vol_20d                              (20-day rolling std, US-DE)
  yield_curve_slope                          (US 10Y - US 2Y)
"""

import os
import time

import numpy as np
import pandas as pd
import polars as pl

from infrastructure.logging_utils import log_data_load

# ── FRED series IDs ───────────────────────────────────────────────────────────
# Daily: DGS10, DGS2 (US only - FRED provides these at daily frequency)
# Monthly OECD: IRLTLT01XXM156N (all other countries - ffill to daily is fine)
FRED_SERIES = {
    "US10Y": "DGS10",  # US 10-year Treasury yield (daily)
    "US2Y": "DGS2",  # US 2-year Treasury yield (daily)
    "DE10Y": "IRLTLT01DEM156N",  # Germany 10-year Bund (monthly)
    "JP10Y": "IRLTLT01JPM156N",  # Japan 10-year JGB (monthly)
    "GB10Y": "IRLTLT01GBM156N",  # UK 10-year Gilt (monthly)
    "AU10Y": "IRLTLT01AUM156N",  # Australia 10-year (monthly)
    "CA10Y": "IRLTLT01CAM156N",  # Canada 10-year (monthly)
    "NZ10Y": "IRLTLT01NZM156N",  # New Zealand 10-year (monthly)
    "CH10Y": "IRLTLT01CHM156N",  # Switzerland 10-year (monthly)
}

# Approximate long-run mean values (used for synthetic fallback)
YIELD_DEFAULTS = {
    "US10Y": 4.50,
    "US2Y": 4.20,
    "DE10Y": 2.40,
    "JP10Y": 0.80,
    "GB10Y": 4.20,
    "AU10Y": 4.40,
    "CA10Y": 3.80,
    "NZ10Y": 4.60,
    "CH10Y": 0.90,
}

# Correlation matrix row/col order matches YIELD_DEFAULTS keys
_YIELD_NAMES = list(YIELD_DEFAULTS.keys())


# ── FRED fetch ────────────────────────────────────────────────────────────────


def _fetch_fred_series(
    series_id: str,
    start: str,
    end: str,
    fred_key: str,
    max_retries: int = 3,
) -> pd.Series:
    """
    Fetch a single FRED series. Retries with exponential backoff on transient
    errors. Handles both naive and tz-aware FRED indices safely.
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise RuntimeError("fredapi not installed - pip install fredapi")

    fred = Fred(api_key=fred_key)
    last_exc: Exception = RuntimeError("no attempts made")

    for attempt in range(max_retries):
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            if s is None or s.empty:
                return pd.Series(dtype=float)
            # Safely normalise to UTC regardless of what FRED returns
            idx = pd.DatetimeIndex(s.index)
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            s.index = idx
            return s.dropna()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # 1s, 2s, 4s

    raise last_exc


def _fetch_all_yields(
    start: pd.Timestamp,
    end: pd.Timestamp,
    fred_key: str,
) -> dict[str, pd.Series | None]:
    """Fetch all configured yield series from FRED."""
    start_s = str(start.date())
    end_s = str(end.date())
    result: dict[str, pd.Series | None] = {}

    for name, series_id in FRED_SERIES.items():
        try:
            s = _fetch_fred_series(series_id, start_s, end_s, fred_key)
            result[name] = s if not s.empty else None
            print(f"[MacroFeatures] FRED {name}: {len(s)} obs")
        except Exception as e:
            print(f"[MacroFeatures] FRED {name} failed ({e}) - using synthetic")
            result[name] = None

    return result


# ── Synthetic fallback ────────────────────────────────────────────────────────


def _synthetic_yields(
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int = 42,
) -> dict[str, pd.Series]:
    """
    Generate synthetic yield series calibrated to realistic levels and
    correlations. Used when FRED API is unavailable.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range(start.date(), end.date(), freq="B", tz="UTC")
    n = len(days)

    # 9x9 correlation matrix (order matches _YIELD_NAMES)
    corr = np.array(
        [
            # US10  US2   DE10  JP10  GB10  AU10  CA10  NZ10  CH10
            [1.00, 0.92, 0.65, 0.40, 0.80, 0.70, 0.75, 0.68, 0.55],  # US10Y
            [0.92, 1.00, 0.60, 0.35, 0.75, 0.65, 0.72, 0.63, 0.50],  # US2Y
            [0.65, 0.60, 1.00, 0.30, 0.72, 0.55, 0.60, 0.52, 0.70],  # DE10Y
            [0.40, 0.35, 0.30, 1.00, 0.35, 0.38, 0.42, 0.36, 0.28],  # JP10Y
            [0.80, 0.75, 0.72, 0.35, 1.00, 0.65, 0.68, 0.62, 0.60],  # GB10Y
            [0.70, 0.65, 0.55, 0.38, 0.65, 1.00, 0.62, 0.78, 0.48],  # AU10Y
            [0.75, 0.72, 0.60, 0.42, 0.68, 0.62, 1.00, 0.60, 0.52],  # CA10Y
            [0.68, 0.63, 0.52, 0.36, 0.62, 0.78, 0.60, 1.00, 0.45],  # NZ10Y
            [0.55, 0.50, 0.70, 0.28, 0.60, 0.48, 0.52, 0.45, 1.00],  # CH10Y
        ]
    )
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, len(_YIELD_NAMES)))
    cz = z @ L.T
    sigma = 0.03  # daily vol (~3 bps per day)
    levels = np.array([YIELD_DEFAULTS[k] for k in _YIELD_NAMES])
    paths = levels + np.cumsum(cz * sigma, axis=0)
    paths = np.clip(paths, 0.01, 20.0)

    return {name: pd.Series(paths[:, i], index=days) for i, name in enumerate(_YIELD_NAMES)}


# ── Feature builder ───────────────────────────────────────────────────────────


class MacroYieldFeatureBuilder:
    """
    Builds yield-spread, carry, and yield-curve features aligned to minute bars.

    Usage:
        builder = MacroYieldFeatureBuilder()        # no FRED key → synthetic
        df = builder.build(bars)
    """

    def __init__(
        self,
        fred_api_key: str | None = None,
        momentum_windows: tuple = (5, 20),
        vol_window: int = 20,
    ):
        self._fred_key = fred_api_key or os.getenv("FRED_API_KEY", "")
        self._mom_wins = momentum_windows
        self._vol_win = vol_window

    # ── public API ────────────────────────────────────────────────────────────

    def load_yields(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.Series]:
        """Load all yield series; fills any gaps with synthetic data."""
        _t0 = time.perf_counter()
        if self._fred_key:
            try:
                raw = _fetch_all_yields(start, end, self._fred_key)
                synth = _synthetic_yields(start, end)
                n_real = 0
                n_synth = 0
                for k in raw:
                    if raw[k] is None or raw[k].empty:  # type: ignore[union-attr]
                        raw[k] = synth.get(k)
                        n_synth += 1
                    else:
                        n_real += 1
                log_data_load(
                    "fred_yields",
                    f"FRED:{start}->{end}",
                    n_rows=n_real,
                    status="success" if n_real else "fallback_synthetic",
                    t0=_t0,
                    note=f"real={n_real}, synthetic={n_synth}",
                )
                return raw  # type: ignore[return-value]
            except Exception as e:
                log_data_load(
                    "fred_yields",
                    f"FRED:{start}->{end}",
                    n_rows=0,
                    status="fallback_synthetic",
                    t0=_t0,
                    exc=e,
                    note="batch failed",
                )

        synth = _synthetic_yields(start, end)
        log_data_load(
            "fred_yields",
            f"synthetic:{start}->{end}",
            n_rows=len(synth),
            status="fallback_synthetic",
            t0=_t0,
            note="no FRED_API_KEY",
        )
        return synth

    def build(self, bars: pl.DataFrame) -> pl.DataFrame:
        """
        Build macro yield features aligned to bars' timestamp_utc.
        Daily yields are forward-filled to bars' frequency via join_asof.
        """
        if isinstance(bars, pd.DataFrame):
            if bars.index.name is None:
                bars.index.name = "timestamp_utc"
            bars = pl.from_pandas(bars.reset_index())

        if len(bars) == 0:
            return pl.DataFrame()

        min_dt = bars["timestamp_utc"].min()
        max_dt = bars["timestamp_utc"].max()
        if min_dt is None or max_dt is None:
            return pl.DataFrame()

        # Normalise to UTC pd.Timestamp for fetching
        try:
            start = pd.Timestamp(str(min_dt)) - pd.Timedelta(days=35)
            end = pd.Timestamp(str(max_dt))
        except Exception:
            return pl.DataFrame()

        yields = self.load_yields(start, end)

        pdf = pd.DataFrame(yields)
        # Yield dates are tz-aware UTC
        pdf.index = pd.to_datetime(pdf.index, utc=True)
        pdf.index.name = "timestamp_utc"

        yield_df = pl.from_pandas(pdf.reset_index())
        yield_df = yield_df.with_columns(pl.all().exclude("timestamp_utc").forward_fill())

        # Ensure yield_df has the same tz awareness as bars
        bar_dtype = bars["timestamp_utc"].dtype
        bars_tz = getattr(bar_dtype, "time_zone", None)
        if bars_tz is None:
            yield_df = yield_df.with_columns(pl.col("timestamp_utc").dt.replace_time_zone(None))
        else:
            yield_df = yield_df.with_columns(pl.col("timestamp_utc").dt.convert_time_zone(bars_tz))
        yield_df = yield_df.with_columns(pl.col("timestamp_utc").cast(bar_dtype))

        # Join asof
        out = (
            bars.select(["timestamp_utc"])
            .sort("timestamp_utc")
            .join_asof(yield_df.sort("timestamp_utc"), on="timestamp_utc", strategy="backward")
        )

        # Forward-fill only. Backward-fill would leak future yield observations
        # into bars before the first available macro timestamp.
        out = out.with_columns(pl.all().exclude("timestamp_utc").forward_fill().fill_null(0.0))

        # ── Yield spreads (positive = USD-bullish for that pair) ──────────────
        out = out.with_columns(
            [
                (pl.col("US10Y") - pl.col("DE10Y")).alias("spread_us_de"),
                (pl.col("US10Y") - pl.col("JP10Y")).alias("spread_us_jp"),
                (pl.col("US10Y") - pl.col("GB10Y")).alias("spread_us_gb"),
                (pl.col("AU10Y") - pl.col("US10Y")).alias("spread_us_au"),
                (pl.col("US10Y") - pl.col("CA10Y")).alias("spread_us_ca"),
                (pl.col("NZ10Y") - pl.col("US10Y")).alias("spread_us_nz"),
                (pl.col("DE10Y") - pl.col("GB10Y")).alias("spread_de_gb"),
                (pl.col("DE10Y") - pl.col("JP10Y")).alias("spread_de_jp"),
                (pl.col("US10Y") - pl.col("CH10Y")).alias("spread_us_ch"),
                (pl.col("US10Y") - pl.col("US2Y")).alias("yield_curve_slope"),
            ]
        )

        # ── Carry signals: positive = long base currency of pair ──────────────
        out = out.with_columns(
            [
                (-pl.col("spread_us_de")).alias("carry_eur"),
                (pl.col("spread_us_jp")).alias("carry_jpy"),
                (-pl.col("spread_us_gb")).alias("carry_gbp"),
                (pl.col("spread_us_au")).alias("carry_aud"),
                (-pl.col("spread_us_ca")).alias("carry_cad"),
                (pl.col("spread_us_nz")).alias("carry_nzd"),
                (pl.col("spread_de_gb")).alias("carry_eurgbp"),
                (pl.col("spread_de_jp")).alias("carry_eurjpy"),
                (-pl.col("spread_us_ch")).alias("carry_chf"),
            ]
        )

        # ── Primary spread momentum + volatility (USD−EUR as primary) ─────────  # noqa: RUF003
        exprs = []
        for w in self._mom_wins:
            exprs.append((pl.col("spread_us_de") - pl.col("spread_us_de").shift(w)).alias(f"yield_momentum_{w}d"))

        exprs.append(
            pl.col("spread_us_de").rolling_std(window_size=self._vol_win).fill_null(0.0).alias("yield_vol_20d")
        )

        out = out.with_columns(exprs)

        drop_cols = ["timestamp_utc", "US10Y", "US2Y", "DE10Y", "JP10Y", "GB10Y", "AU10Y", "CA10Y", "NZ10Y", "CH10Y"]
        out = out.drop([c for c in drop_cols if c in out.columns])

        return out.fill_null(0.0)


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Macro Yield Features - smoke test")
    idx = pd.date_range("2024-01-01", periods=1440, freq="1min", tz="UTC")
    bars = pl.DataFrame({"timestamp_utc": idx, "close": np.random.randn(1440).cumsum() + 1.085})

    builder = MacroYieldFeatureBuilder()  # no FRED key → synthetic
    df = builder.build(bars)
    print(f"  Feature shape : {df.shape}")
    print(f"  Columns       : {df.columns}")
    nulls = sum(df[col].null_count() for col in df.columns)
    print(f"  NaNs          : {nulls}")
    assert nulls == 0, "NaNs found in output!"
    print(df.head(3))
    print("OK")
