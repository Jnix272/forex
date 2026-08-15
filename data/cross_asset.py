"""
data/cross_asset.py
====================
Load cross-asset market series (commodities, yields, equity proxies, risk
indicators) for intermarket feature engineering.

Provider priority per asset:
  1) Stooq public CSV  — free, no auth, covers most series back to 2000+
  2) Yahoo Finance (yfinance) — broader coverage, requires yfinance package
  3) FRED API (fredapi) — most reliable for bond yields, requires FRED_API_KEY
  4) EODHD API         — commercial, broadest coverage, requires EODHD_API_KEY

Known symbol corrections vs original:
  - DE10Y  : Yahoo "^TNX" was wrong (that is US 10Y). Using Stooq "10dey.b" only.
  - US2Y   : Yahoo "^IRX" is the 13-week T-Bill, not 2-year. Annotated as approx.
  - JP10Y, GB10Y, AU10Y, CA10Y added for macro spread features.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from infrastructure.logging_utils import log_data_load

# ── Stooq symbol candidates (tried in order until one returns data) ───────────
STOOQ_SYMBOLS: dict[str, list[str]] = {
    # Commodities
    "WTI":       ["cl.f",      "cl"],
    "GOLD":      ["xauusd",    "gc.f"],
    "COPPER":    ["hg.f",      "hg"],
    "NATGAS":    ["ng.f",      "ng"],
    "SILVER":    ["xagusd",    "si.f"],
    # FX / Dollar index
    "DXY":       ["usdidx",    "dx.f"],
    # Equities — US
    "SPX":       ["^spx",      "spx"],
    "NASDAQ100": ["^ndq",      "ndq"],
    "VIX":       ["^vix",      "vix"],
    # Equities — global (local index = local currency signal)
    "DAX":       ["^dax",      "dax"],
    "FTSE100":   ["^ftx",      "ftx"],
    "NIKKEI225": ["^nkx",      "nkx"],
    "ASX200":    ["^asx",      "asx"],
    # EM risk
    "EEM":       ["eem.us"],
    # Crypto
    "BTC":       ["btc.v"],
    # Yields — US
    "US10Y":     ["10usy.b",   "us10y"],
    "US2Y":      ["2usy.b",    "us2y"],
    # Yields — international (no correct Yahoo fallback; stooq/FRED/EODHD only)
    "DE10Y":     ["10dey.b",   "de10y"],
    "JP10Y":     ["10jpy.b",   "jp10y"],
    "GB10Y":     ["10gby.b",   "gb10y"],
    "AU10Y":     ["10auy.b",   "au10y"],
    "CA10Y":     ["10cay.b",   "ca10y"],
    "NZ10Y":     ["10nzy.b",   "nz10y"],
    "CH10Y":     ["10chy.b",   "ch10y"],
}

# ── Yahoo Finance fallback symbols ─────────────────────────────────────────────
# NOTE: Yahoo symbols for bond yields are unreliable / often wrong.
# Only use for assets where stooq fails and there is no FRED series.
YAHOO_SYMBOLS: dict[str, str] = {
    "WTI":       "CL=F",
    "GOLD":      "GC=F",
    "COPPER":    "HG=F",
    "NATGAS":    "NG=F",
    "SILVER":    "SI=F",
    "DXY":       "DX-Y.NYB",
    "SPX":       "^GSPC",
    "NASDAQ100": "^NDX",
    "VIX":       "^VIX",
    "DAX":       "^GDAXI",
    "FTSE100":   "^FTSE",
    "NIKKEI225": "^N225",
    "ASX200":    "^AXJO",
    "EEM":       "EEM",
    "BTC":       "BTC-USD",
    "US10Y":     "^TNX",           # US 10-year Treasury yield (correct)
    # US2Y  — no reliable Yahoo symbol (^IRX is 13-week T-Bill); use FRED
    # NZ10Y / CH10Y / DE10Y / JP10Y / GB10Y / AU10Y / CA10Y — no reliable Yahoo symbols; use FRED/EODHD
}

# ── FRED series IDs for bond yields (most reliable source) ────────────────────
FRED_YIELD_SYMBOLS: dict[str, str] = {
    "US10Y": "DGS10",
    "US2Y":  "DGS2",
    "DE10Y": "IRLTLT01DEM156N",   # Germany 10Y (monthly — ffill to daily)
    "JP10Y": "IRLTLT01JPM156N",   # Japan 10Y (monthly)
    "GB10Y": "IRLTLT01GBM156N",   # UK 10Y (monthly)
    "AU10Y": "IRLTLT01AUM156N",   # Australia 10Y (monthly)
    "CA10Y": "IRLTLT01CAM156N",   # Canada 10Y (monthly)
    "NZ10Y": "IRLTLT01NZM156N",   # New Zealand 10Y (monthly)
    "CH10Y": "IRLTLT01CHM156N",   # Switzerland 10Y (monthly)
}

# Assets where Yahoo fallback is known-wrong or unavailable
_NO_YAHOO_FALLBACK = {
    "DE10Y", "JP10Y", "GB10Y", "AU10Y", "CA10Y", "NZ10Y", "CH10Y",
    "US2Y",
}


# ── Stooq helpers ─────────────────────────────────────────────────────────────

def _stooq_url(symbol: str) -> str:
    return f"https://stooq.com/q/d/l/?s={symbol}&i=d"


def _read_stooq_daily(symbol: str) -> pd.Series | None:
    try:
        import io
        import urllib.request
        req = urllib.request.Request(
            _stooq_url(symbol),
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        if not raw.strip() or "<html" in raw.lower():
            return None
        df = pd.read_csv(io.StringIO(raw))
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    # Stooq returns a single row with "No data" or a date-like string when empty
    first_date = str(df.iloc[0].get("Date", "")).lower()
    if any(tok in first_date for tok in ("no_data", "no data", "brak")):
        return None
    if len(df) < 2:    # single row is likely an error response
        return None
    s   = pd.to_numeric(df["Close"], errors="coerce")
    idx = pd.to_datetime(df["Date"], utc=True, errors="coerce") + pd.Timedelta(days=1)
    out = pd.Series(s.values, index=idx, name=symbol).dropna()
    if out.empty:
        return None
    return out.sort_index()


# ── Yahoo Finance helper ───────────────────────────────────────────────────────

def _read_yahoo_daily(symbol: str, start: str, end: str) -> pd.Series | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        df = yf.download(
            symbol,
            start=str(pd.Timestamp(start).date()),
            end=str(end_ts.date()),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    col = next((c for c in ("Adj Close", "Close") if c in df.columns), None)
    if col is None:
        return None
    col_data = df[col]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
    s   = pd.to_numeric(col_data, errors="coerce")
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    out = pd.Series(s.values, index=idx, name=symbol).dropna()
    if out.empty:
        return None
    return out.sort_index()


# ── FRED helper ───────────────────────────────────────────────────────────────

def _read_fred_daily(series_id: str, start: str, end: str) -> pd.Series | None:
    """Fetch from FRED; returns None if fredapi not installed or key missing."""
    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        if s is None or s.empty:
            return None
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        else:
            s.index = s.index.tz_convert("UTC")
        return s.dropna().sort_index()
    except Exception:
        return None


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_read(cache_path: Path, start: pd.Timestamp = None, end: pd.Timestamp = None) -> pd.Series | None:
    if not cache_path.exists():
        return None
    try:
        df = pd.read_parquet(cache_path)
        idx = pd.to_datetime(df["timestamp"], utc=True)
        ser = pd.Series(df["value"].values, index=idx).dropna()
        if ser.empty:
            return None
            
        # Range check: if the cache doesn't cover the requested range, force redownload
        if start is not None and ser.index.min() > start:
            return None
        if end is not None and ser.index.max() < end:
            return None
            
        return ser.sort_index()
    except Exception:
        return None


def _cache_write(cache_path: Path, ser: pd.Series, metadata: dict = None) -> None:
    try:
        if ser.empty:
            return
            
        # 1. Deduplicate by index (keep last observation)
        ser = ser[~ser.index.duplicated(keep="last")]
        
        # 2. Sort index
        ser = ser.sort_index()
        
        # 3. Resample to Calendar Daily and Forward Fill to enforce gap policy
        ser = ser.asfreq("D").ffill()
        
        # Convert to DataFrame
        df = pd.DataFrame({
            "timestamp": ser.index,
            "value": ser.values,
        })
        
        # Save as Parquet with metadata
        table = pa.Table.from_pandas(df)
        
        # Add metadata
        if metadata is None: 
            metadata = {}
        metadata["schema_version"] = "1.0"
        metadata["processed_at"] = pd.Timestamp.utcnow().isoformat()
        
        custom_meta = {k.encode(): str(v).encode() for k, v in metadata.items()}
        existing_meta = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing_meta, **custom_meta})
        
        pq.write_table(table, cache_path)
    except Exception as e:
        print(f"Cache write failed for {cache_path}: {e}")


# ── EODHD helper ──────────────────────────────────────────────────────────────

def _read_eodhd_daily(
    asset:     str,
    start:     str,
    end:       str,
    cache_dir: Path,
) -> pd.Series | None:
    """
    Fetch a cross-asset series from EODHD as last-resort fallback.
    Only called if EODHD_API_KEY is set in the environment.
    Results are cached in cache_dir as <asset>_eodhd.csv.
    """
    try:
        from data.eodhd import EODHDLoader
    except ImportError:
        return None

    if not EODHDLoader.has_api_key():
        return None

    # Check local EODHD-specific cache first
    safe_asset = asset.replace("/", "_")
    cache_path = cache_dir / f"{safe_asset}_eodhd.parquet"
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    cached = _cache_read(cache_path, start_ts, end_ts)
    if cached is not None and not cached.empty:
        return cached

    try:
        loader  = EODHDLoader(verbose=False)
        panel   = loader.load_cross_asset(start=start, end=end, assets=[asset], use_cache=False)
        ser     = panel.get(asset)
        if ser is not None and not ser.empty:
            _cache_write(cache_path, ser, metadata={"source": "eodhd", "asset": asset})
        return ser
    except Exception:
        return None


# ── Main public function ───────────────────────────────────────────────────────

def load_cross_asset_panel(
    start:     str,
    end:       str,
    cache_dir: str,
    source:    str = "auto",
) -> dict[str, pd.Series]:
    """
    Return dict of asset -> daily price/yield series with UTC DatetimeIndex,
    clipped to [start, end].

    source : "auto" | "stooq" | "yahoo" | "fred"
      "auto" tries stooq -> yahoo -> fred in order per asset.
    """
    source = (source or "auto").strip().lower()
    if source not in {"auto", "stooq", "yahoo", "fred", "eodhd"}:
        return {}

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")

    cdir = Path(cache_dir) / "cross_asset"
    cdir.mkdir(parents=True, exist_ok=True)

    out: dict[str, pd.Series] = {}

    for asset, candidates in STOOQ_SYMBOLS.items():
        _t0_asset = time.perf_counter()
        ser: pd.Series | None = None
        used_provider = None

        provider_order = (
            ["stooq", "yahoo", "fred", "eodhd"] if source == "auto" else [source]
        )

        for provider in provider_order:
            if ser is not None and not ser.empty:
                used_provider = provider
                break

            if provider == "stooq":
                for sym in candidates:
                    safe_sym   = sym.replace("^", "idx")
                    cache_path = cdir / f"{asset}_stooq_{safe_sym}.parquet"
                    ser = _cache_read(cache_path, start_ts, end_ts)
                    if ser is None or ser.empty:
                        ser = _read_stooq_daily(sym)
                        if ser is not None and not ser.empty:
                            _cache_write(cache_path, ser, metadata={"source": "stooq", "symbol": sym, "asset": asset})
                    if ser is not None and not ser.empty:
                        log_data_load("cross_asset_provider", f"{asset}/stooq/{sym}", 
                                     n_rows=len(ser), status="success", t0=_t0_asset)
                        break

            elif provider == "yahoo":
                if asset in _NO_YAHOO_FALLBACK:
                    continue
                ysym = YAHOO_SYMBOLS.get(asset)
                if not ysym:
                    continue
                safe_sym   = ysym.replace("^", "idx").replace("=", "_")
                cache_path = cdir / f"{asset}_yahoo_{safe_sym}.parquet"
                ser = _cache_read(cache_path, start_ts, end_ts)
                if ser is None or ser.empty:
                    ser = _read_yahoo_daily(ysym, start, end)
                    if ser is not None and not ser.empty:
                        _cache_write(cache_path, ser, metadata={"source": "yahoo", "symbol": ysym, "asset": asset})
                if ser is not None and not ser.empty:
                    log_data_load("cross_asset_provider", f"{asset}/yahoo/{ysym}", 
                                 n_rows=len(ser), status="success", t0=_t0_asset)

            elif provider == "fred":
                fsym = FRED_YIELD_SYMBOLS.get(asset)
                if not fsym:
                    continue
                cache_path = cdir / f"{asset}_fred_{fsym}.parquet"
                ser = _cache_read(cache_path, start_ts, end_ts)
                if ser is None or ser.empty:
                    ser = _read_fred_daily(fsym, start, end)
                    if ser is not None and not ser.empty:
                        _cache_write(cache_path, ser, metadata={"source": "fred", "symbol": fsym, "asset": asset})
                if ser is not None and not ser.empty:
                    log_data_load("cross_asset_provider", f"{asset}/fred/{fsym}", 
                                 n_rows=len(ser), status="success", t0=_t0_asset)

            elif provider == "eodhd":
                ser = _read_eodhd_daily(asset, start, end, cdir)
                if ser is not None and not ser.empty:
                    log_data_load("cross_asset_provider", f"{asset}/eodhd", 
                                 n_rows=len(ser), status="success", t0=_t0_asset)

        if ser is None or ser.empty:
            log_data_load("cross_asset", asset, n_rows=0, status="all_providers_failed",
                         note=f"tried={provider_order}", t0=_t0_asset)
            continue

        # Shift index by 1 day to prevent look-ahead bias (daily close is known next day)
        ser.index = ser.index + pd.Timedelta(days=1)

        # Clip to requested range; fall back to full series if no overlap
        clip = ser[(ser.index >= start_ts) & (ser.index <= end_ts)]
        out[asset] = clip if not clip.empty else ser
        
        # Freshness warning
        if not out[asset].empty:
            max_dt = out[asset].index.max()
            if max_dt < end_ts - pd.Timedelta(days=14):
                import logging
                logging.warning(
                    f"Cross-asset {asset} is STALE! Max date {max_dt.date()} is >14 days "
                    f"behind requested end date {end_ts.date()}. Fills will be hallucinated."
                )

        log_data_load("cross_asset", asset, n_rows=len(out[asset]), status="success",
                      t0=_t0_asset, note=f"provider={used_provider or provider}")

    # Derived: yield curve slope (US 10Y − US 2Y) — classic recession signal
    if "US10Y" in out and "US2Y" in out:
        us10 = out["US10Y"].reindex(out["US10Y"].index.union(out["US2Y"].index)).ffill()
        us2  = out["US2Y"].reindex(us10.index).ffill()
        slope = (us10 - us2).dropna()
        if not slope.empty:
            clip  = slope[(slope.index >= start_ts) & (slope.index <= end_ts)]
            out["YIELD_CURVE_SLOPE"] = clip if not clip.empty else slope

    return out
