"""
data/eodhd.py
=============
EODHD (End-of-Day Historical Data) loader for:
  1) Forex pairs  — daily OHLCV via EODHD's /eod/ endpoint (EURUSD.FOREX etc.)
  2) Intraday     — 1m / 5m / 1h bars via /intraday/ endpoint (paid plan)
  3) Cross-asset  — equities, commodities, crypto, bond yields

Authentication
--------------
Set your API key as an environment variable:

    Windows:  set EODHD_API_KEY=your_key_here
    Linux:    export EODHD_API_KEY=your_key_here
    .env:     EODHD_API_KEY=your_key_here

Free tier gives ~20 API requests/day. Paid tiers unlock intraday + unlimited.
Sign up at: https://eodhd.com/

Supported pairs (all use the FOREX exchange in EODHD)
------------------------------------------------------
    EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, EURGBP
    EURJPY, GBPJPY, NZDUSD, USDSGD, USDHKD, USDMXN, USDZAR

Usage
-----
    from data.eodhd import EODHDLoader

    loader = EODHDLoader()               # reads EODHD_API_KEY from env
    df = loader.load("EURUSD", start="2022-01-01", end="2024-12-31")

    # Intraday (requires paid plan)
    df = loader.load_intraday("EURUSD", interval="1h", start="2024-01-01")

    # Cross-asset panel
    panel = loader.load_cross_asset(start="2022-01-01", end="2024-12-31")
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Load .env automatically (same pattern as config/settings.py)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

from data.sources import (
    _enforce_schema,
    TICK_COLUMNS,
    _DATA_DIR,
)

# ── Constants ─────────────────────────────────────────────────────────────────

EODHD_BASE_URL    = "https://eodhd.com/api"
DEFAULT_EODHD_CACHE_DIR = str(_DATA_DIR / "raw" / "eodhd")

# ── Forex pair -> EODHD ticker mapping ────────────────────────────────────────
# All forex pairs use the ".FOREX" exchange in EODHD.
EODHD_FOREX_PAIRS: Dict[str, str] = {
    "EURUSD": "EURUSD.FOREX",
    "GBPUSD": "GBPUSD.FOREX",
    "USDJPY": "USDJPY.FOREX",
    "AUDUSD": "AUDUSD.FOREX",
    "USDCAD": "USDCAD.FOREX",
    "USDCHF": "USDCHF.FOREX",
    "EURGBP": "EURGBP.FOREX",
    "EURJPY": "EURJPY.FOREX",
    "GBPJPY": "GBPJPY.FOREX",
    "NZDUSD": "NZDUSD.FOREX",
    "USDSGD": "USDSGD.FOREX",
    "USDHKD": "USDHKD.FOREX",
    "USDMXN": "USDMXN.FOREX",
    "USDZAR": "USDZAR.FOREX",
}

# Typical half-spread per pair (price units) for bid/ask synthesis.
_TYPICAL_HALF_SPREAD: Dict[str, float] = {
    "EURUSD": 0.00005,
    "GBPUSD": 0.00008,
    "USDJPY": 0.005,
    "AUDUSD": 0.00008,
    "USDCAD": 0.00010,
    "USDCHF": 0.00010,
    "NZDUSD": 0.00012,
    "EURGBP": 0.00008,
    "EURJPY": 0.008,
    "GBPJPY": 0.012,
    "USDSGD": 0.00015,
    "USDHKD": 0.0003,
    "USDMXN": 0.02,
    "USDZAR": 0.03,
}
_DEFAULT_HALF_SPREAD = 0.00010


def _pair_half_spread(pair: str) -> float:
    return _TYPICAL_HALF_SPREAD.get(pair.upper(), _DEFAULT_HALF_SPREAD)


# ── Cross-asset EODHD symbols ──────────────────────────────────────────────────
# Maps the project's cross-asset asset name -> EODHD ticker (code.EXCHANGE).
# Exchanges used:
#   .INDX  — indices (S&P500, VIX, DXY, etc.)
#   .COMM  — commodities (oil, copper)
#   .FOREX — FX-quoted spot prices (gold, silver)
#   .CC    — crypto
#   .GBOND — government bond yields
EODHD_CROSS_ASSET: Dict[str, str] = {
    # Commodities
    "WTI":        "OILCRUDEWTI.COMM",
    "GOLD":       "XAUUSD.FOREX",
    "COPPER":     "HG.COMM",
    "NATGAS":     "NATURALGAS.COMM",
    "SILVER":     "XAGUSD.FOREX",
    # FX / Dollar index
    "DXY":        "DXY.INDX",
    # Equities — global
    "SPX":        "GSPC.INDX",
    "NASDAQ100":  "NDX.INDX",
    "VIX":        "VIX.INDX",
    "DAX":        "GDAXI.INDX",        # Germany  — EUR signal
    "FTSE100":    "FTSE.INDX",         # UK       — GBP signal
    "NIKKEI225":  "N225.INDX",         # Japan    — JPY risk-on/off
    "ASX200":     "AXJO.INDX",         # Australia — AUD signal
    # Broad EM risk
    "EEM":        "EEM.US",            # MSCI EM ETF — risk appetite
    # Crypto
    "BTC":        "BTC-USD.CC",
    # US yields
    "US10Y":      "US10Y.GBOND",
    "US2Y":       "US2Y.GBOND",
    # International yields
    "DE10Y":      "DE10Y.GBOND",
    "JP10Y":      "JP10Y.GBOND",
    "GB10Y":      "GB10Y.GBOND",
    "AU10Y":      "AU10Y.GBOND",
    "CA10Y":      "CA10Y.GBOND",
    "NZ10Y":      "NZ10Y.GBOND",       # New Zealand — NZD carry
    "CH10Y":      "CH10Y.GBOND",       # Switzerland — CHF safe-haven
}


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _eodhd_get(
    endpoint:  str,
    params:    Dict[str, str],
    api_key:   str,
    timeout:   int = 30,
    max_retry: int = 3,
) -> Optional[list]:
    """
    Make a GET request to EODHD API, returning parsed JSON or None on failure.
    Retries with exponential back-off on rate-limit (429) responses.
    """
    import json
    import urllib.request
    import urllib.parse

    params["api_token"] = api_key
    params["fmt"]       = "json"
    url = f"{EODHD_BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"

    for attempt in range(max_retry):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (forex-scaling-model)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except Exception as exc:
            code = getattr(getattr(exc, "code", None), "real", None) or getattr(exc, "code", 0)
            if code == 429 and attempt < max_retry - 1:
                time.sleep(2 ** attempt)
                continue
            if attempt == max_retry - 1:
                return None
    return None


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_eod_response(data: list, pair: str) -> pd.DataFrame:
    """
    Convert EODHD /eod/ JSON list to a bid/ask DataFrame in tick schema.

    EODHD daily bars have: date, open, high, low, close, adjusted_close, volume
    We synthesise bid/ask from close +/- typical half-spread.
    """
    if not data:
        return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

    records = []
    half    = _pair_half_spread(pair)
    for row in data:
        try:
            ts    = pd.Timestamp(row["date"], tz="UTC")
            close = float(row.get("adjusted_close") or row.get("close") or 0)
            if close <= 0:
                continue
            records.append({
                "timestamp": ts,
                "bid":       close - half,
                "ask":       close + half,
                "volume":    float(row.get("volume") or 1.0),
            })
        except (KeyError, ValueError, TypeError):
            continue

    if not records:
        return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    return _enforce_schema(df, pair, "eodhd")


def _parse_intraday_response(data: list, pair: str) -> pd.DataFrame:
    """
    Convert EODHD /intraday/ JSON list to tick schema.

    Intraday bars have: timestamp (unix), gmtoffset, datetime, open, high, low, close, volume
    """
    if not data:
        return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

    records = []
    half    = _pair_half_spread(pair)
    for row in data:
        try:
            # 'timestamp' is unix epoch seconds
            ts    = pd.Timestamp(int(row["timestamp"]), unit="s", tz="UTC")
            close = float(row.get("close") or 0)
            if close <= 0:
                continue
            records.append({
                "timestamp": ts,
                "bid":       close - half,
                "ask":       close + half,
                "volume":    float(row.get("volume") or 1.0),
            })
        except (KeyError, ValueError, TypeError):
            continue

    if not records:
        return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    return _enforce_schema(df, pair, "eodhd")


def _parse_cross_asset_response(data: list, asset: str) -> Optional[pd.Series]:
    """
    Convert EODHD /eod/ JSON list to a daily price Series (UTC DatetimeIndex).
    """
    if not data:
        return None

    dates  = []
    values = []
    for row in data:
        try:
            ts  = pd.Timestamp(row["date"], tz="UTC")
            val = float(row.get("adjusted_close") or row.get("close") or 0)
            if val > 0:
                dates.append(ts)
                values.append(val)
        except (KeyError, ValueError, TypeError):
            continue

    if not dates:
        return None

    ser = pd.Series(values, index=pd.DatetimeIndex(dates), name=asset).sort_index()
    return ser if not ser.empty else None


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path_for_pair(cache_dir: str, pair: str, suffix: str = "daily") -> Path:
    return Path(cache_dir) / pair / f"{pair}_{suffix}.parquet"


def _cache_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _cache_write_parquet(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except Exception:
        pass


def _cache_path_cross(cache_dir: str, asset: str) -> Path:
    return Path(cache_dir) / "cross_asset" / f"{asset}_eodhd.csv"


def _cache_read_cross(path: Path) -> Optional[pd.Series]:
    if not path.exists():
        return None
    try:
        df  = pd.read_csv(path)
        idx = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        val = pd.to_numeric(df["value"], errors="coerce")
        ser = pd.Series(val.values, index=idx).dropna()
        return ser if not ser.empty else None
    except Exception:
        return None


def _cache_write_cross(path: Path, ser: pd.Series) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "timestamp": ser.index.astype(str),
            "value":     ser.values,
        }).to_csv(path, index=False)
    except Exception:
        pass


# ── Main loader class ─────────────────────────────────────────────────────────

class EODHDLoader:
    """
    Load forex OHLCV and cross-asset data from EODHD API.

    Parameters
    ----------
    api_key : str, optional
        EODHD API token. Falls back to ``EODHD_API_KEY`` environment variable.
    cache_dir : str
        Root directory for caching responses as Parquet / CSV.
    verbose : bool
        Print progress messages.

    Raises
    ------
    ValueError
        If no API key is found (neither argument nor env var).
    """

    def __init__(
        self,
        api_key:   str = "",
        cache_dir: str = DEFAULT_EODHD_CACHE_DIR,
        verbose:   bool = True,
    ):
        self.api_key   = api_key or os.getenv("EODHD_API_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.verbose   = verbose
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _require_key(self) -> str:
        if not self.api_key:
            raise ValueError(
                "EODHD API key not set. "
                "Pass api_key= or set EODHD_API_KEY environment variable. "
                "Get a free key at: https://eodhd.com/"
            )
        return self.api_key

    # ── Forex daily ──────────────────────────────────────────────────────────

    def load(
        self,
        pair:  str,
        start: Optional[str] = None,
        end:   Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load daily OHLCV bars for a forex pair.

        Returns a DataFrame in the project tick schema (bid/ask/mid/spread/
        volume/pair/source) with UTC DatetimeIndex.

        Parameters
        ----------
        pair      : e.g. "EURUSD" or "EUR/USD"
        start     : ISO date "YYYY-MM-DD" (default: 2000-01-01)
        end       : ISO date "YYYY-MM-DD" (default: today)
        use_cache : Return cached Parquet if available and date range is covered.
        """
        pair   = pair.upper().replace("/", "")
        ticker = EODHD_FOREX_PAIRS.get(pair)
        if not ticker:
            if self.verbose:
                print(f"[EODHD] No ticker mapping for {pair}")
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

        key    = self._require_key()
        start  = start or "2000-01-01"
        end    = end   or str(pd.Timestamp.now(tz="UTC").date())

        cpath = _cache_path_for_pair(str(self.cache_dir), pair)
        if use_cache:
            cached = _cache_read_parquet(cpath)
            if cached is not None:
                cached_start = str(cached.index.min().date())
                cached_end   = str(cached.index.max().date())
                if cached_start <= start and cached_end >= end:
                    df = cached
                    df = df[(df.index >= pd.Timestamp(start, tz="UTC")) &
                            (df.index <= pd.Timestamp(end,   tz="UTC"))]
                    if self.verbose:
                        print(f"[EODHD] {pair}: {len(df):,} bars (cache)")
                    return df

        # Fetch from API
        code, exchange = ticker.split(".", 1)
        data = _eodhd_get(
            f"eod/{ticker}",
            {"from": start, "to": end},
            api_key=key,
        )
        if not data:
            if self.verbose:
                print(f"[EODHD] {pair}: no data returned from API")
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

        df = _parse_eod_response(data, pair)
        if df.empty:
            if self.verbose:
                print(f"[EODHD] {pair}: empty response")
            return df

        _cache_write_parquet(cpath, df)
        if self.verbose:
            print(f"[EODHD] {pair}: {len(df):,} bars ({df.index[0].date()} -> {df.index[-1].date()})")

        return df[
            (df.index >= pd.Timestamp(start, tz="UTC")) &
            (df.index <= pd.Timestamp(end,   tz="UTC"))
        ]

    def load_multiple(
        self,
        pairs: List[str],
        start: Optional[str] = None,
        end:   Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple pairs. Returns dict of pair -> DataFrame."""
        results = {}
        for pair in pairs:
            results[pair.upper().replace("/", "")] = self.load(
                pair, start=start, end=end, use_cache=use_cache
            )
        return results

    # ── Forex intraday (paid plan) ────────────────────────────────────────────

    def load_intraday(
        self,
        pair:      str,
        interval:  str = "1h",
        start:     Optional[str] = None,
        end:       Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load intraday bars for a forex pair (requires EODHD paid plan).

        Parameters
        ----------
        pair     : e.g. "EURUSD"
        interval : "1m", "5m", or "1h"
        start    : ISO date "YYYY-MM-DD"
        end      : ISO date "YYYY-MM-DD"
        """
        pair   = pair.upper().replace("/", "")
        ticker = EODHD_FOREX_PAIRS.get(pair)
        if not ticker:
            if self.verbose:
                print(f"[EODHD] No ticker mapping for {pair}")
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

        key   = self._require_key()
        start = start or str((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)).date())
        end   = end   or str(pd.Timestamp.now(tz="UTC").date())

        cpath = _cache_path_for_pair(str(self.cache_dir), pair, suffix=f"intraday_{interval}")
        if use_cache:
            cached = _cache_read_parquet(cpath)
            if cached is not None and not cached.empty:
                if self.verbose:
                    print(f"[EODHD] {pair} {interval}: {len(cached):,} bars (cache)")
                df = cached
                return df[
                    (df.index >= pd.Timestamp(start, tz="UTC")) &
                    (df.index <= pd.Timestamp(end,   tz="UTC"))
                ]

        params = {
            "interval": interval,
            "from":     str(int(pd.Timestamp(start, tz="UTC").timestamp())),
            "to":       str(int(pd.Timestamp(end,   tz="UTC").timestamp())),
        }
        data = _eodhd_get(f"intraday/{ticker}", params, api_key=key)
        if not data:
            if self.verbose:
                print(f"[EODHD] {pair} {interval}: no intraday data (check plan tier)")
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

        df = _parse_intraday_response(data, pair)
        if not df.empty:
            _cache_write_parquet(cpath, df)
            if self.verbose:
                print(
                    f"[EODHD] {pair} {interval}: {len(df):,} bars "
                    f"({df.index[0].date()} -> {df.index[-1].date()})"
                )

        return df[
            (df.index >= pd.Timestamp(start, tz="UTC")) &
            (df.index <= pd.Timestamp(end,   tz="UTC"))
        ]

    # ── Cross-asset ───────────────────────────────────────────────────────────

    def load_cross_asset(
        self,
        start:     Optional[str] = None,
        end:       Optional[str] = None,
        assets:    Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.Series]:
        """
        Load the full cross-asset panel from EODHD.

        Parameters
        ----------
        start    : ISO date "YYYY-MM-DD"
        end      : ISO date "YYYY-MM-DD"
        assets   : List of asset names to load (default: all in EODHD_CROSS_ASSET)
        use_cache: Use cached CSV if available.

        Returns
        -------
        dict of asset_name -> pd.Series with UTC DatetimeIndex
        """
        key   = self._require_key()
        start = start or "2000-01-01"
        end   = end   or str(pd.Timestamp.now(tz="UTC").date())

        target_assets = assets or list(EODHD_CROSS_ASSET.keys())
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts   = pd.Timestamp(end,   tz="UTC")

        out: Dict[str, pd.Series] = {}

        for asset in target_assets:
            ticker = EODHD_CROSS_ASSET.get(asset)
            if not ticker:
                continue

            cpath = _cache_path_cross(str(self.cache_dir), asset)

            if use_cache:
                ser = _cache_read_cross(cpath)
                if ser is not None and not ser.empty:
                    clip = ser[(ser.index >= start_ts) & (ser.index <= end_ts)]
                    if not clip.empty:
                        out[asset] = clip
                        if self.verbose:
                            print(f"[EODHD] {asset:<18} {len(clip):>6,} pts (cache)")
                        continue

            # Fetch from API
            data = _eodhd_get(
                f"eod/{ticker}",
                {"from": start, "to": end},
                api_key=key,
            )
            if not data:
                if self.verbose:
                    print(f"[EODHD] {asset:<18} -- (no data)")
                continue

            ser = _parse_cross_asset_response(data, asset)
            if ser is None or ser.empty:
                if self.verbose:
                    print(f"[EODHD] {asset:<18} -- (empty)")
                continue

            _cache_write_cross(cpath, ser)
            clip = ser[(ser.index >= start_ts) & (ser.index <= end_ts)]
            out[asset] = clip if not clip.empty else ser
            if self.verbose:
                d0 = str(out[asset].index.min().date())
                d1 = str(out[asset].index.max().date())
                print(f"[EODHD] {asset:<18} {len(out[asset]):>6,} pts ({d0} -> {d1})")

        # Derived: yield curve slope
        if "US10Y" in out and "US2Y" in out:
            us10  = out["US10Y"].reindex(
                out["US10Y"].index.union(out["US2Y"].index)
            ).ffill()
            us2   = out["US2Y"].reindex(us10.index).ffill()
            slope = (us10 - us2).dropna()
            if not slope.empty:
                clip = slope[(slope.index >= start_ts) & (slope.index <= end_ts)]
                out["YIELD_CURVE_SLOPE"] = clip if not clip.empty else slope

        return out

    # ── Convenience ───────────────────────────────────────────────────────────

    def check_api_key(self) -> bool:
        """
        Validate the API key with a minimal request.
        Returns True if the key works, False otherwise.
        """
        if not self.api_key:
            return False
        data = _eodhd_get("eod/EURUSD.FOREX", {"from": "2024-01-02", "to": "2024-01-03"}, self.api_key)
        return bool(data)

    @staticmethod
    def has_api_key() -> bool:
        """Return True if EODHD_API_KEY env var is set and non-empty."""
        return bool(os.getenv("EODHD_API_KEY", "").strip())
