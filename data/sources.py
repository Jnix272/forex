"""
data/sources.py
================
Production data connectors for all three specified sources:

  1. DukascopyLoader   — Free tick data (EUR/USD, GBP/USD, 2003–present)
  2. TickDataSuite     — Paid Ducascopy-based data with spread reconstruction
  3. LMAXLoader        — Institutional LMAX Exchange Level-2 tick feed

All loaders produce the same output schema:
  DataFrame with UTC DatetimeIndex and columns:
    bid, ask, mid, spread, volume, pair, source

This unified schema means any loader can drop straight into the
feature engineering pipeline with zero downstream changes.

Dukascopy format notes
-----------------------
Raw files are LZMA-compressed binaries at:
  https://datafeed.dukascopy.com/datafeed/{PAIR}/{YEAR}/{MONTH:02d}/{DAY:02d}/{HOUR:02d}h_ticks.bi5

Each .bi5 file covers exactly one hour of ticks.
Binary format per tick (20 bytes):
  milliseconds_offset : uint32 (ms since start of hour)
  ask_price_scaled    : uint32 (price × 100000 for JPY pairs, × 10000 others)
  bid_price_scaled    : uint32
  ask_volume          : float32
  bid_volume          : float32

LMAX format notes
------------------
LMAX provides FIX 4.4 and REST API access. Free historical data is
available at: https://www.lmax.com/exchange/market-data
Institutional feed requires LMAX brokerage account.
The REST endpoint returns per-bar OHLCV + spread data which we
reconstruct to tick resolution using the Ask/Bid bar prices.
"""

import os
import sys
import asyncio
import aiohttp
import time
import lzma
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional, List, Dict, Tuple
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
try:
    import duckdb
    _DUCKDB = True
except ImportError:
    duckdb = None
    _DUCKDB = False

try:
    import wandb
except ImportError:
    wandb = None


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SCHEMA ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

TICK_COLUMNS = ["bid", "ask", "mid", "spread", "volume", "pair", "source"]
PIP_SIZES    = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "USDCAD": 0.0001, "USDCHF": 0.0001, "NZDUSD": 0.0001, "EURGBP": 0.0001,
    "USDJPY": 0.01,   "EURJPY": 0.01,   "GBPJPY": 0.01,
}


def _enforce_schema(df: pd.DataFrame, pair: str, source: str) -> pd.DataFrame:
    """
    Ensure output always has the unified tick schema.
    Adds mid and spread if missing, tags pair and source.
    Enforces UTC index named 'timestamp'.
    """
    if df.empty:
        return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

    # Keep a cheap owning frame to avoid chained-assignment/view warnings
    # without forcing a deep copy of the underlying arrays.
    df = df.copy(deep=False)

    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    if "mid" not in df.columns:
        df["mid"] = ((df["bid"] + df["ask"]) / 2.0).astype(np.float32, copy=False)
    if "spread" not in df.columns:
        df["spread"] = (df["ask"] - df["bid"]).astype(np.float32, copy=False)
    if "volume" not in df.columns:
        df["volume"] = np.ones(len(df), dtype=np.float32)

    for col in ("bid", "ask", "mid", "spread", "volume"):
        if col in df.columns and df[col].dtype != np.float32:
            df[col] = df[col].astype(np.float32, copy=False)

    df["pair"]   = pd.Categorical.from_codes(np.zeros(len(df), dtype=np.int8), categories=[pair])
    df["source"] = pd.Categorical.from_codes(np.zeros(len(df), dtype=np.int8), categories=[source])

    # Golden Rule: drop rows where bid >= ask (data corruption)
    bad = df["bid"].to_numpy(copy=False) >= df["ask"].to_numpy(copy=False)
    if bad.any():
        print(f"  [Schema] Dropped {bad.sum()} rows with bid >= ask")
        df = df.loc[~bad]

    dupes = int(df.index.duplicated(keep="first").sum())
    if dupes:
        print(f"  [Schema] Dropped {dupes} duplicate timestamps")
        df = df.loc[~df.index.duplicated(keep="first")]

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    if list(df.columns) != TICK_COLUMNS:
        df = df.loc[:, TICK_COLUMNS]
    # Return without sorting to reduce memory usage
    return df[TICK_COLUMNS]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DUKASCOPY LOADER
# ─────────────────────────────────────────────────────────────────────────────

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{pair}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

# Dukascopy uses pair codes like "EURUSD" but the URL uses uppercase pair without slash
DUKA_PAIR_MAP = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD", "EURGBP": "EURGBP", "EURJPY": "EURJPY",
    "GBPJPY": "GBPJPY",
}

# Local cache roots (next to this package: data/raw/…)
_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_DUKASCOPY_CACHE_DIR = str(_DATA_DIR / "raw" / "dukascopy")
DEFAULT_DUKASCOPY_COMPACT_DIR = str(_DATA_DIR / "compact" / "dukascopy")
DEFAULT_TDS_DATA_DIR = str(_DATA_DIR / "raw" / "tickdatasuite")
DEFAULT_LMAX_DATA_DIR = str(_DATA_DIR / "raw" / "lmax")

# Point value for price descaling (non-JPY default: ÷100000, JPY pairs: ÷1000)
DUKA_POINT = {
    "USDJPY": 1000, "EURJPY": 1000, "GBPJPY": 1000,
}


def _parse_bi5_hour(raw_bytes: bytes, dt_hour: datetime, pair: str) -> pd.DataFrame:
    """
    Parse one Dukascopy .bi5 hour file into a DataFrame.
    Binary layout: 20 bytes per tick.
      [0:4]  ms offset from start of hour (uint32 big-endian)
      [4:8]  ask price scaled (uint32 big-endian)
      [8:12] bid price scaled (uint32 big-endian)
      [12:16] ask volume (float32 big-endian)
      [16:20] bid volume (float32 big-endian)
    """
    if not raw_bytes:
        return pd.DataFrame()

    point = DUKA_POINT.get(pair, 100000)
    n     = len(raw_bytes) // 20
    if n == 0:
        return pd.DataFrame()

    ticks = np.frombuffer(raw_bytes, dtype=">u4,>u4,>u4,>f4,>f4")[:n]
    ms_offsets  = ticks["f0"].astype(np.int64)
    ask_scaled  = ticks["f1"].astype(np.float32, copy=False)
    bid_scaled  = ticks["f2"].astype(np.float32, copy=False)
    ask_vol     = ticks["f3"].astype(np.float32, copy=False)
    bid_vol     = ticks["f4"].astype(np.float32, copy=False)

    ask = ask_scaled / point
    bid = bid_scaled / point
    vol = ((ask_vol + bid_vol) / 2.0).round(2).astype(np.float32, copy=False)

    epoch_ms = int(dt_hour.timestamp() * 1000) + ms_offsets
    idx = pd.to_datetime(epoch_ms, unit="ms", utc=True)

    return pd.DataFrame({
        "bid":    bid,
        "ask":    ask,
        "volume": vol,
    }, index=idx)


def _run_dukascopy_async(coro):
    """
    Run an async coroutine for Dukascopy I/O.

    On Windows, the default Proactor event loop + aiohttp can log noisy
    ConnectionResetError tracebacks during connection teardown. Selector loop
    avoids most of that.
    """
    if sys.platform == "win32":
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass
    return asyncio.run(coro)


class DukascopyLoader:
    """
    Async Dukascopy tick loader with on-disk Parquet cache per hour-file.

    Defaults favour **reliability over speed** (low concurrency, one pair at a
    time for ``load_multiple``). Increase ``concurrency`` and
    ``max_parallel_pairs`` if the feed tolerates it.

    Usage
    -----
        loader = DukascopyLoader()
        df = loader.load("EURUSD", start="2024-01-01", end="2024-01-31")
    """

    def __init__(
        self,
        cache_dir:    str  = DEFAULT_DUKASCOPY_CACHE_DIR,
        request_delay: float = 0.05,
        max_retries:   int  = 8,
        verbose:       bool = True,
        concurrency:   int  = 12,
        max_parallel_pairs: int = 1,
    ):
        self.cache_dir    = Path(cache_dir)
        self.delay        = request_delay
        self.max_retries  = max_retries
        self.verbose      = verbose
        self.concurrency  = max(1, concurrency)
        self.max_parallel_pairs = max(1, max_parallel_pairs)
        self._print_lock  = Lock()
        self._parse_failures = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_tasks_dt(
        self,
        start_dt: datetime,
        end_dt: datetime,
        hours: Optional[List[int]],
    ) -> List[datetime]:
        """Build the weekday/hour download schedule for a request."""
        hour_list = list(range(24)) if hours is None else list(hours)
        tasks_dt: List[datetime] = []
        current = start_dt
        while current <= end_dt:
            if current.weekday() < 5:
                for hour in hour_list:
                    tasks_dt.append(current.replace(hour=hour))
            current += timedelta(days=1)
        return tasks_dt

    def _hour_coverage_report(self, df: pd.DataFrame, requested_hours: List[datetime]) -> dict:
        """Measure duplicate timestamps and requested-hour coverage."""
        if df.empty:
            idx = pd.DatetimeIndex([], tz="UTC")
        else:
            idx = pd.to_datetime(df.index, utc=True)

        present_hours = set(idx.floor("h").unique().to_pydatetime().tolist())
        missing_hours = [dt for dt in requested_hours if dt not in present_hours]
        return {
            "duplicate_timestamps": int(idx.duplicated(keep="first").sum()),
            "missing_hours": missing_hours,
            "missing_hours_count": len(missing_hours),
            "requested_hours_count": len(requested_hours),
            "present_hours_count": len(present_hours),
        }

    def _remove_cached_hours(self, pair: str, datetimes: List[datetime]) -> None:
        """Delete cached parquet files for specific hours before a retry."""
        for dt in datetimes:
            try:
                self._cache_path(pair, dt).unlink(missing_ok=True)
            except Exception:
                pass

    def _validate_cached_hour(self, df: pd.DataFrame, dt: datetime) -> bool:
        """
        Return True when an hour frame is safe to reuse from cache.

        We keep this intentionally conservative: reject empty frames, NaNs in
        core price columns, duplicate timestamps, or rows that spill outside the
        requested UTC hour. Invalid cached files are deleted and re-fetched.
        """
        if df is None or df.empty:
            return False

        try:
            idx = pd.to_datetime(df.index, utc=True)
        except Exception:
            return False

        hour_end = dt + timedelta(hours=1)
        if ((idx < dt) | (idx >= hour_end)).any():
            return False
        if idx.duplicated(keep="first").any():
            return False

        required_cols = {"bid", "ask"}
        if not required_cols.issubset(df.columns):
            return False
        if df.loc[:, ["bid", "ask"]].isna().any().any():
            return False
        if (df["ask"] <= df["bid"]).any():
            return False

        return True

    def _load_and_validate(
        self,
        pair: str,
        datetimes: List[datetime],
        *,
        auto_redownload: bool = True,
        max_redownload_passes: int = 2,
    ) -> pd.DataFrame:
        """Load requested hours and retry any hours still missing afterwards."""
        result = _run_dukascopy_async(self._load_all_async(pair, datetimes))
        report = self._hour_coverage_report(result, datetimes)

        passes = 0
        while auto_redownload and report["missing_hours"] and passes < max_redownload_passes:
            passes += 1
            missing_hours = report["missing_hours"]
            if self.verbose:
                print(
                    f"  [Dukascopy] {pair} missing {len(missing_hours)} hour(s); "
                    f"redownload pass {passes}/{max_redownload_passes}"
                )
            self._remove_cached_hours(pair, missing_hours)
            retry_df = _run_dukascopy_async(self._load_all_async(pair, missing_hours))
            if not retry_df.empty:
                result = _enforce_schema(pd.concat([result, retry_df], copy=False), pair, "dukascopy")
            report = self._hour_coverage_report(result, datetimes)

        if self.verbose:
            print(
                f"  [Dukascopy] {pair} coverage: "
                f"{report['present_hours_count']}/{report['requested_hours_count']} hours"
            )
            if report["missing_hours"]:
                print(f"  [Dukascopy] {pair} still missing {report['missing_hours_count']} hour(s)")

        return result

    def _cache_path(self, pair: str, dt: datetime) -> Path:
        return (self.cache_dir / pair /
                f"{dt.year}" / f"{dt.month:02d}" /
                f"{dt.day:02d}_{dt.hour:02d}.parquet")

    async def _fetch_hour_async(
        self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
        pair: str, dt: datetime,
    ) -> Optional[bytes]:
        """Download one hour of .bi5 data. Releases the semaphore between retries."""
        url = DUKASCOPY_URL.format(
            pair  = DUKA_PAIR_MAP.get(pair, pair),
            year  = dt.year,
            month = dt.month - 1,   # Dukascopy months are 0-indexed (Jan=00, Dec=11)
            day   = dt.day,
            hour  = dt.hour,
        )
        req_timeout = aiohttp.ClientTimeout(total=60, sock_connect=20, sock_read=45)

        for attempt in range(self.max_retries):
            async with semaphore:
                if self.delay > 0:
                    await asyncio.sleep(self.delay)
                try:
                    async with session.get(url, timeout=req_timeout) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            # Truncated TCP bodies are common on flaky links; LZMA then fails
                            # with "end-of-stream marker" — reject and retry instead of parsing.
                            cl = resp.headers.get("Content-Length")
                            if asyncio.iscoroutine(cl):
                                cl = await cl
                            if cl is not None:
                                try:
                                    if int(cl) != len(data):
                                        data = b""
                                except ValueError:
                                    pass
                            if not data:
                                pass  # fall through to backoff + retry
                            else:
                                if self.delay > 0:
                                    await asyncio.sleep(self.delay)
                                return data
                        elif resp.status == 404:
                            return None
                        elif resp.status == 429:
                            pass  # release semaphore, then backoff below
                        else:
                            pass
                except (asyncio.TimeoutError, aiohttp.ClientError, OSError):
                    pass
            # Backoff happens OUTSIDE the semaphore so slots stay free
            backoff = min(20, 1.5 ** attempt) if attempt > 0 else 0.75
            await asyncio.sleep(backoff)
        return None

    async def _load_hour_async(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        pair: str,
        dt: datetime,
        executor: ThreadPoolExecutor,
    ) -> pd.DataFrame:
        """Load one hour — from disk cache or network fetch."""
        cache_file = self._cache_path(pair, dt)

        if cache_file.exists():
            try:
                loop = asyncio.get_running_loop()
                cached_df = await loop.run_in_executor(executor, pd.read_parquet, cache_file)
                if self._validate_cached_hour(cached_df, dt):
                    return cached_df
            except Exception:
                pass

            # Corrupted or incomplete parquet — delete and refetch so future
            # runs don't keep hitting the same bad file.
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass

        loop = asyncio.get_running_loop()
        hour_lbl = f"{dt.year}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}h"

        # Extra refetch passes: truncated HTTP bodies yield LZMAError even when status=200.
        for body_try in range(3):
            try:
                raw_lzma = await asyncio.wait_for(
                    self._fetch_hour_async(session, semaphore, pair, dt),
                    timeout=180,
                )
            except asyncio.TimeoutError:
                return pd.DataFrame()

            if raw_lzma is None:
                return pd.DataFrame()

            def _process(payload: bytes) -> pd.DataFrame:
                try:
                    raw_bytes = lzma.decompress(payload)
                    if len(raw_bytes) % 20 != 0:
                        raise ValueError(
                            f"truncated Dukascopy payload: {len(raw_bytes)} bytes is not divisible by 20"
                        )
                    df = _parse_bi5_hour(raw_bytes, dt, pair)
                    if not self._validate_cached_hour(df, dt):
                        raise ValueError("parsed hour failed integrity validation")
                except (lzma.LZMAError, ValueError):
                    raise
                except Exception as e:
                    self._parse_failures += 1
                    if self.verbose and self._parse_failures <= 5:
                        print(f"  [Dukascopy] parse error {pair} {hour_lbl}: {e!r}")
                    return pd.DataFrame()
                if not df.empty:
                    try:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        df.to_parquet(cache_file)
                    except Exception:
                        pass
                return df

            try:
                return await loop.run_in_executor(
                    executor, _process, raw_lzma,
                )
            except (lzma.LZMAError, ValueError) as e:
                self._parse_failures += 1
                if self.verbose and self._parse_failures <= 8:
                    print(
                        f"  [Dukascopy] invalid .bi5 {pair} {hour_lbl} "
                        f"(retry {body_try + 1}/3): {e!r}"
                    )
                await asyncio.sleep(0.75 * (body_try + 1))

        return pd.DataFrame()

    def load(
        self,
        pair:  str,
        start: str,
        end:   str,
        hours: Optional[List[int]] = None,
        auto_redownload: bool = True,
        max_redownload_passes: int = 2,
    ) -> pd.DataFrame:
        """
        Synchronous wrapper for modern async loading.
        Maintains compatibility with existing training scripts.
        """
        pair     = pair.upper().replace("/", "")
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        tasks_dt = self._build_tasks_dt(start_dt, end_dt, hours)

        if not tasks_dt:
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

        self._parse_failures = 0
        if self.verbose:
            print(f"[Dukascopy] Async Loading {pair} | {start} -> {end} | "
                  f"{len(tasks_dt)} hours | Concurrency: {self.concurrency}")

        return self._load_and_validate(
            pair,
            tasks_dt,
            auto_redownload=auto_redownload,
            max_redownload_passes=max_redownload_passes,
        )

    async def _load_all_async(
        self,
        pair:      str,
        datetimes: List[datetime],
        session:   Optional[aiohttp.ClientSession] = None,
        semaphore: Optional[asyncio.Semaphore]     = None,
        executor:  Optional[ThreadPoolExecutor]    = None,
    ) -> pd.DataFrame:
        """
        Core async orchestrator. Accepts shared session/semaphore/executor so
        load_multiple can reuse one connection pool across all pairs.
        When called standalone (load()), creates its own resources.
        """
        own_session  = session  is None
        own_executor = executor is None

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.concurrency)
        if own_executor:
            executor = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4))

        async def _run(sess: aiohttp.ClientSession) -> pd.DataFrame:
            total     = len(datetimes)
            completed = 0
            results: List[Optional[pd.DataFrame]] = [None] * total
            log_step  = max(total // 40, 25)
            loop      = asyncio.get_running_loop()
            t0        = loop.time()

            async def _wrapped(idx: int, dt: datetime):
                return idx, await self._load_hour_async(sess, semaphore, pair, dt, executor)

            tasks = [asyncio.create_task(_wrapped(i, dt)) for i, dt in enumerate(datetimes)]

            for coro in asyncio.as_completed(tasks):
                idx, df = await coro
                results[idx] = df
                completed += 1
                if self.verbose and (completed % log_step == 0 or completed == total):
                    elapsed = loop.time() - t0
                    rate    = completed / max(elapsed, 0.1)
                    eta     = (total - completed) / max(rate, 0.01)
                    pct_done = completed * 100 / total
                    print(f"  [{pair}] {int(pct_done):3d}% | {completed}/{total}"
                          f" | {rate:.0f} h/s | ETA {eta:.0f}s")
                    if wandb and wandb.run:
                        wandb.log({
                            f"dukascopy/{pair}/progress_pct": pct_done,
                            f"dukascopy/{pair}/hours_fetched": completed,
                            f"dukascopy/{pair}/rate_hps": rate,
                            f"dukascopy/{pair}/eta_s": eta,
                        })

            non_empty = [df for df in results if df is not None and not df.empty]
            if not non_empty:
                return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))
            combined = pd.concat(non_empty, copy=False)
            return _enforce_schema(combined, pair, "dukascopy")

        if own_session:
            conn_limit = self.concurrency + 5
            connector  = aiohttp.TCPConnector(
                limit=conn_limit,
                limit_per_host=conn_limit,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                keepalive_timeout=30,
            )
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=30),
                connector=connector,
                headers={"User-Agent": "ForexScaler/2.0 (research)"},
            ) as sess:
                result = await _run(sess)
        else:
            result = await _run(session)

        if own_executor:
            executor.shutdown(wait=True)
        return result

    def load_multiple(
        self,
        pairs: List[str],
        start: str,
        end:   str,
        hours: Optional[List[int]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple pairs (see ``max_parallel_pairs`` — default is one at a time)."""
        self._parse_failures = 0
        return _run_dukascopy_async(self._load_multiple_async(pairs, start, end, hours))

    async def _load_multiple_async(
        self, pairs: List[str], start: str, end: str, hours: Optional[List[int]]
    ) -> Dict[str, pd.DataFrame]:
        hours    = list(range(24)) if hours is None else hours
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        n_pairs  = max(len(pairs), 1)
        pair_parallel = min(n_pairs, self.max_parallel_pairs)

        # Pool size: only ``pair_parallel`` pairs may hit the host at once.
        per_pair = self.concurrency + 5
        host_cap = per_pair * pair_parallel
        connector  = aiohttp.TCPConnector(
            limit=host_cap,
            limit_per_host=host_cap,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=30,
        )
        executor   = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4))
        semaphores = {p.upper().replace("/", ""): asyncio.Semaphore(self.concurrency)
                      for p in pairs}

        async def _one_pair(sess: aiohttp.ClientSession, pair: str) -> tuple:
            pair = pair.upper().replace("/", "")
            tasks_dt = self._build_tasks_dt(start_dt, end_dt, hours)
            if not tasks_dt:
                return pair, pd.DataFrame(columns=pd.Index(TICK_COLUMNS))
            return pair, await self._load_all_async(
                pair, tasks_dt,
                session=sess, semaphore=semaphores[pair], executor=executor,
            )

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=30),
            connector=connector,
            headers={"User-Agent": "ForexScaler/2.0 (research)"},
        ) as sess:
            pair_results: List[tuple] = []
            n = len(pairs)
            batch = max(1, self.max_parallel_pairs)
            for i in range(0, n, batch):
                chunk = pairs[i : i + batch]
                part = await asyncio.gather(*[_one_pair(sess, p) for p in chunk])
                pair_results.extend(part)

        executor.shutdown(wait=True)
        return {p: df for p, df in pair_results}

    def load_eurusd_gbpusd(
        self,
        start: str,
        end:   str,
        session_only: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Convenience: load EUR/USD and GBP/USD (the two most liquid pairs).
        session_only=True loads only London session + NY open (07–17 UTC),
        reducing file count by ~54% while keeping all tradeable hours.
        """
        hours = list(range(7, 18)) if session_only else None  # London session + NY open
        return self.load_multiple(["EURUSD", "GBPUSD"], start, end, hours)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TICK DATA SUITE LOADER
# ─────────────────────────────────────────────────────────────────────────────

class TickDataSuiteLoader:
    """
    Loader for data processed by Tick Data Suite (TDS).

    Tick Data Suite is a MetaTrader plugin that:
      - Downloads Dukascopy raw tick data
      - Reconstructs real bid/ask spreads (Dukascopy only provides one-sided
        spread in its free feed — TDS uses a proprietary spread model)
      - Exports to CSV, FXT, or HSTv401 formats for MT4/MT5 backtesting

    TDS export formats supported here:
      - Standard CSV  : timestamp, open, high, low, close, tickvol, vol, spread
      - Tick CSV      : date, time, bid, ask, volume
      - Parquet       : pre-converted for faster loading

    How to export from TDS:
      1. Open TDS -> Data -> Export
      2. Choose "Tick data" format
      3. Set date range and pair
      4. Export as CSV with semicolon delimiter
      5. Point data_dir to the export folder

    TDS CSV tick format (semicolon-delimited):
      20240101 00:00:00.123;1.10500;1.10502;0.75
      (datetime;bid;ask;volume)
    """

    # TDS uses semicolons and no header by default
    TDS_DTYPES = {"bid": np.float64, "ask": np.float64, "volume": np.float64}

    def __init__(
        self,
        data_dir: str = DEFAULT_TDS_DATA_DIR,
        verbose:  bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.verbose  = verbose

    def _detect_format(self, filepath: Path) -> str:
        """Detect whether file is TDS tick CSV, TDS bar CSV, or Parquet."""
        if filepath.suffix == ".parquet":
            return "parquet"
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
        # Tick CSV: date and time in separate columns or combined
        if ";" in first:
            parts = first.split(";")
            if len(parts) >= 4:
                return "tick_csv_semicolon"
        if "," in first:
            parts = first.split(",")
            if len(parts) >= 4:
                return "tick_csv_comma"
        return "bar_csv"

    def _load_tick_csv(self, filepath: Path, delimiter: str = ";") -> pd.DataFrame:
        """
        Parse TDS tick CSV export.
        Handles both:
          (a) "20240101 00:00:00.123;bid;ask;vol"
          (b) "2024.01.01,00:00:00,bid,ask,vol"  (MT4 style)
        """
        # Try to detect column count first
        with open(filepath, "r") as f:
            sample = f.readline().strip()
        parts = sample.split(delimiter)

        if len(parts) == 4:
            # Combined datetime
            names = ["datetime", "bid", "ask", "volume"]
            df = pd.read_csv(
                str(filepath),
                sep=delimiter,
                header=None,
                names=names,
                dtype={"bid": float, "ask": float, "volume": float},
                iterator=False,
            )
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.dropna(subset=("datetime",)).set_index("datetime")

        elif len(parts) == 5:
            # Separate date and time columns
            names = ["date", "time", "bid", "ask", "volume"]
            df = pd.read_csv(
                str(filepath),
                sep=delimiter,
                header=None,
                names=names,
                dtype={"bid": float, "ask": float, "volume": float},
                iterator=False,
            )
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                utc=True, errors="coerce"
            )
            df = df.dropna(subset=("datetime",)).set_index("datetime")

        else:
            raise ValueError(f"Unexpected TDS CSV column count: {len(parts)} in {filepath}")

        return df[["bid", "ask", "volume"]]

    def load_file(self, filepath: str, pair: str) -> pd.DataFrame:
        """Load a single TDS export file."""
        fp  = Path(filepath)
        fmt = self._detect_format(fp)

        if fmt == "parquet":
            df = pd.read_parquet(fp)
        elif fmt == "tick_csv_semicolon":
            df = self._load_tick_csv(fp, delimiter=";")
        elif fmt == "tick_csv_comma":
            df = self._load_tick_csv(fp, delimiter=",")
        else:
            raise ValueError(f"Unsupported TDS format: {fmt}  ({filepath})")

        df = _enforce_schema(df, pair.upper().replace("/",""), "tickdatasuite")
        if self.verbose:
            pip = PIP_SIZES.get(pair.upper().replace("/", ""), 0.0001)
            print(f"[TDS] Loaded {fp.name} | {len(df):,} ticks | "
                  f"Spread: {df['spread'].mean()/pip:.2f} pips avg")
        return df

    def load_directory(
        self,
        pair:  str,
        start: Optional[str] = None,
        end:   Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load all TDS files for a pair from data_dir.
        Expects files named: EURUSD_2024*.csv  or  EURUSD/*.parquet

        The directory layout TDS exports to by default:
          /data_dir/EURUSD/2024_01.csv
          /data_dir/EURUSD/2024_02.csv
          ...
        """
        pair_clean = pair.upper().replace("/", "")
        search_dirs = [
            self.data_dir / pair_clean,
            self.data_dir,
        ]
        files = []
        for d in search_dirs:
            if d.exists():
                files += list(d.glob(f"*{pair_clean}*.csv"))
                files += list(d.glob(f"*{pair_clean}*.parquet"))
                files += list(d.glob("*.csv"))
                files += list(d.glob("*.parquet"))

        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(
                f"No TDS files found for {pair} in {self.data_dir}\n"
                f"Export tick data from Tick Data Suite and place in:\n"
                f"  {self.data_dir / pair_clean}/"
            )

        frames = []
        for f in files:
            try:
                frames.append(self.load_file(str(f), pair))
            except Exception as e:
                print(f"  [TDS] Skipped {f.name}: {e}")

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # Date filter
        if start: combined = combined[combined.index >= pd.Timestamp(start, tz="UTC")]
        if end:   combined = combined[combined.index <= pd.Timestamp(end,   tz="UTC")]

        if self.verbose:
            print(f"[TDS] {pair_clean}: {len(combined):,} ticks | "
                  f"{combined.index[0]} -> {combined.index[-1]}")
        return combined

    def convert_to_parquet(self, pair: str):
        """
        Convert all CSV files for a pair to Parquet for faster future loads.
        Run once after exporting from TDS.
        """
        pair_clean = pair.upper().replace("/", "")
        csv_files  = list((self.data_dir / pair_clean).glob("*.csv"))
        for f in csv_files:
            out = f.with_suffix(".parquet")
            if out.exists():
                continue
            try:
                df = self.load_file(str(f), pair)
                df.to_parquet(out)
                print(f"  Converted {f.name} -> {out.name}")
            except Exception as e:
                print(f"  Skip {f.name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LMAX EXCHANGE LOADER
# ─────────────────────────────────────────────────────────────────────────────

LMAX_REST_BASE = "https://trade.lmax.com"
LMAX_HIST_BASE = "https://www.lmax.com/exchange/market-data"

# LMAX instrument IDs for major FX pairs
LMAX_INSTRUMENTS = {
    "EURUSD": "4001",   "GBPUSD": "4002",   "USDJPY": "4003",
    "AUDUSD": "4004",   "USDCAD": "4005",   "EURGBP": "4006",
    "USDCHF": "4007",   "NZDUSD": "4009",   "EURJPY": "4012",
}


class LMAXLoader:
    """
    Loader for LMAX Exchange tick / order book data.

    LMAX is an institutional FX ECN with:
      - No last-look execution (true ECN)
      - Tight raw spreads (0.1–0.3 pips EUR/USD typical)
      - Level-2 order book depth (10 price levels)
      - FIX 4.4 and REST API access

    Two modes
    ----------
    1. Historical CSV (free, no account needed):
       Download from https://www.lmax.com/exchange/market-data
       1-minute OHLCV + spread per pair, 2010–present.

    2. Live REST API (requires LMAX brokerage account):
       Real-time L1 bid/ask + L2 order book depth.
       Set LMAX_USERNAME and LMAX_PASSWORD environment variables.

    Historical data format (LMAX CSV):
      DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose,Volume

    Live API returns JSON order book snapshots at ~100ms resolution.
    """

    def __init__(
        self,
        data_dir:    str  = DEFAULT_LMAX_DATA_DIR,
        username:    Optional[str] = None,
        password:    Optional[str] = None,
        verbose:     bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.username = username or os.getenv("LMAX_USERNAME")
        self.password = password or os.getenv("LMAX_PASSWORD")
        self.verbose  = verbose
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._session_token: Optional[str] = None

    # ── Historical (free) ─────────────────────────────────────────────────────

    def load_historical_csv(
        self,
        pair:  str,
        filepath: Optional[str] = None,
        start: Optional[str] = None,
        end:   Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load LMAX historical 1-minute OHLCV CSV data.

        Download from: https://www.lmax.com/exchange/market-data
        Select pair -> Download CSV -> place in data_dir/LMAX/{pair}/

        CSV format expected:
          DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose,Volume
        """
        pair_clean = pair.upper().replace("/", "")

        if filepath:
            files = [Path(filepath)]
        else:
            search_dir = self.data_dir / pair_clean
            files = sorted(search_dir.glob("*.csv")) + sorted(search_dir.glob("*.parquet"))

        if not files:
            raise FileNotFoundError(
                f"No LMAX data files found for {pair} in {self.data_dir}/{pair_clean}/\n"
                f"Download from: https://www.lmax.com/exchange/market-data\n"
                f"Place CSVs in: {self.data_dir}/{pair_clean}/"
            )

        frames = []
        for f in files:
            if f.suffix == ".parquet":
                df = pd.read_parquet(f)
            else:
                df = self._parse_lmax_csv(f)
            frames.append(df)

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        if start: combined = combined[combined.index >= pd.Timestamp(start, tz="UTC")]
        if end:   combined = combined[combined.index <= pd.Timestamp(end,   tz="UTC")]

        combined = _enforce_schema(combined, pair_clean, "lmax_historical")
        if self.verbose:
            pip = PIP_SIZES.get(pair_clean, 0.0001)
            print(f"[LMAX] Historical {pair_clean}: {len(combined):,} bars | "
                  f"Spread: {combined['spread'].mean()/pip:.2f} pips avg | "
                  f"{combined.index[0]} -> {combined.index[-1]}")
        return combined

    def _parse_lmax_csv(self, filepath: Path) -> pd.DataFrame:
        """Parse LMAX historical CSV into bid/ask OHLCV bars."""
        # Try with and without header
        try:
            df = pd.read_csv(
                str(filepath),
                parse_dates=True,
                index_col=0,
                iterator=False,
            )
        except Exception:
            df = pd.read_csv(
                str(filepath),
                header=None,
                parse_dates=True,
                index_col=0,
                names=["dt", "bo", "bh", "bl", "bc", "ao", "ah", "al", "ac", "vol"],
                iterator=False,
            )

        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.dropna()

        # Normalize column names (case-insensitive)
        df.columns = df.columns.str.lower().str.strip()
        col_map = {}
        for c in df.columns:
            if "bidclose"  in c or c == "bc":  col_map[c] = "bid"
            elif "askclose" in c or c == "ac": col_map[c] = "ask"
            elif "volume"   in c or c == "vol":col_map[c] = "volume"
        df = df.rename(columns=col_map)

        # Fall back to synthesizing bid/ask from BidHigh/Low if BidClose missing
        if "bid" not in df.columns:
            bh_col = next((c for c in df.columns if "bidhi" in c or c=="bh"), None)
            bl_col = next((c for c in df.columns if "bidlo" in c or c=="bl"), None)
            if bh_col and bl_col:
                df["bid"] = (df[bh_col] + df[bl_col]) / 2.0
        if "ask" not in df.columns:
            ah_col = next((c for c in df.columns if "askhi" in c or c=="ah"), None)
            al_col = next((c for c in df.columns if "asklo" in c or c=="al"), None)
            if ah_col and al_col:
                df["ask"] = (df[ah_col] + df[al_col]) / 2.0

        return df[["bid", "ask"] + (["volume"] if "volume" in df.columns else [])]

    # ── Live REST API (requires LMAX account) ─────────────────────────────────

    def login(self) -> bool:
        """Authenticate with LMAX REST API. Returns True on success."""
        if not self.username or not self.password:
            print("[LMAX] No credentials. Set LMAX_USERNAME and LMAX_PASSWORD env vars.")
            return False
        try:
            import json
            payload = json.dumps({"username": self.username,
                                   "password": self.password}).encode()
            req = Request(
                f"{LMAX_REST_BASE}/public/security/authenticate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                self._session_token = data.get("sessionId") or data.get("token")
                if self._session_token:
                    print(f"[LMAX] Authenticated | session: {self._session_token[:8]}...")
                    return True
        except Exception as e:
            print(f"[LMAX] Login failed: {e}")
        return False

    def fetch_orderbook(self, pair: str) -> Optional[dict]:
        """
        Fetch live L2 order book snapshot for a pair.
        Requires active session (call login() first).

        Returns dict with keys: bid_levels, ask_levels, timestamp
        Each level: {"price": float, "quantity": float}
        """
        if not self._session_token:
            if not self.login():
                return None
        try:
            import json
            instr_id = LMAX_INSTRUMENTS.get(pair.upper().replace("/",""), "4001")
            req = Request(
                f"{LMAX_REST_BASE}/public/orderbook/{instr_id}/data",
                headers={
                    "Session-Id": self._session_token,
                    "Content-Type": "application/json",
                },
            )
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return {
                    "timestamp":  pd.Timestamp.now(tz="UTC"),
                    "pair":       pair,
                    "bid_levels": data.get("bids", [])[:10],
                    "ask_levels": data.get("asks", [])[:10],
                    "best_bid":   float(data["bids"][0]["price"]) if data.get("bids") else None,
                    "best_ask":   float(data["asks"][0]["price"]) if data.get("asks") else None,
                }
        except Exception as e:
            print(f"[LMAX] Order book error: {e}")
            return None

    def stream_ticks(
        self,
        pair:      str,
        n_ticks:   int = 1000,
        callback = None,
    ) -> pd.DataFrame:
        """
        Poll the LMAX REST API for live tick data.
        In production, replace with FIX 4.4 streaming (lower latency).

        callback: optional function(tick_dict) called on each new tick.
        """
        if not self._session_token:
            if not self.login():
                return pd.DataFrame()

        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "stream_ticks() uses blocking I/O and must not be called from "
                "an async context. Use an executor or a separate thread."
            )
        except RuntimeError as e:
            if "stream_ticks" in str(e):
                raise

        ticks = []
        print(f"[LMAX] Streaming {n_ticks} ticks for {pair}...")
        while len(ticks) < n_ticks:
            ob = self.fetch_orderbook(pair)
            if ob and ob["best_bid"] and ob["best_ask"]:
                tick = {
                    "bid":    ob["best_bid"],
                    "ask":    ob["best_ask"],
                    "volume": 1,
                }
                ticks.append((ob["timestamp"], tick))
                if callback: callback(tick)
            time.sleep(0.1)   # 10 Hz polling — replace with FIX 4.4 for production

        idx = pd.DatetimeIndex([t[0] for t in ticks], tz="UTC")
        df  = pd.DataFrame([t[1] for t in ticks], index=idx)
        return _enforce_schema(df, pair.upper().replace("/",""), "lmax_live")

    def convert_to_parquet(self, pair: str):
        """Convert all LMAX CSVs to Parquet for fast future loads."""
        pair_clean = pair.upper().replace("/", "")
        for f in (self.data_dir / pair_clean).glob("*.csv"):
            out = f.with_suffix(".parquet")
            if out.exists():
                continue
            df = self._parse_lmax_csv(f)
            df.to_parquet(out)
            print(f"  {f.name} -> {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED DATA MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ForexDataManager:
    """
    Single interface across all three data sources.

    Priority order (best quality -> lowest latency):
      LMAX live    -> best spread accuracy, institutional quality
      TDS          -> reconstructed spreads, best for backtesting
      Dukascopy    -> free, vast history (2003+), good for research

    Usage
    -----
        mgr = ForexDataManager()

        # Research / backtest — free Dukascopy data
        df = mgr.load("EURUSD", start="2023-01-01", end="2023-12-31",
                       source="dukascopy")

        # Backtest with real spreads — TDS export
        df = mgr.load("EURUSD", start="2023-01-01", end="2023-12-31",
                       source="tds")

        # Live trading — LMAX ECN feed
        df = mgr.load("EURUSD", source="lmax_live", n_ticks=10000)

        # Auto-select best available source
        df = mgr.load("EURUSD", start="2023-01-01", end="2023-12-31")
    """

    SOURCE_PRIORITY = ["lmax_live", "tds", "databento", "dukascopy", "eodhd", "myfxbook"]

    def __init__(
        self,
        dukascopy_dir:  str = DEFAULT_DUKASCOPY_CACHE_DIR,
        dukascopy_compact_dir: str = DEFAULT_DUKASCOPY_COMPACT_DIR,
        tds_dir:        str = DEFAULT_TDS_DATA_DIR,
        lmax_dir:       str = DEFAULT_LMAX_DATA_DIR,
        myfxbook_dir:   str = "",          # auto-resolved from data/myfxbook.py default
        eodhd_api_key:  str = "",          # falls back to EODHD_API_KEY env var
        eodhd_cache_dir: str = "",         # auto-resolved from data/eodhd.py default
        lmax_username:  Optional[str] = None,
        lmax_password:  Optional[str] = None,
        verbose:        bool = True,
    ):
        from data.myfxbook import MyfxbookLoader, DEFAULT_MYFXBOOK_DATA_DIR
        from data.eodhd import EODHDLoader, DEFAULT_EODHD_CACHE_DIR
        from data.databento_loader import DatabentoLoader
        self.duka     = DukascopyLoader(dukascopy_dir, verbose=verbose)
        self.tds      = TickDataSuiteLoader(tds_dir, verbose=verbose)
        self.lmax     = LMAXLoader(lmax_dir, lmax_username, lmax_password, verbose=verbose)
        self.databento = DatabentoLoader(verbose=verbose)
        self.myfxbook = MyfxbookLoader(myfxbook_dir or DEFAULT_MYFXBOOK_DATA_DIR, verbose=verbose)
        self.eodhd    = EODHDLoader(
            api_key   = eodhd_api_key,
            cache_dir = eodhd_cache_dir or DEFAULT_EODHD_CACHE_DIR,
            verbose   = verbose,
        )
        self.duka_compact_dir = Path(dukascopy_compact_dir)
        self.duka_compact_dir.mkdir(parents=True, exist_ok=True)
        self.verbose  = verbose

    def load(
        self,
        pair:     str,
        source:   str  = "auto",
        start:    Optional[str] = None,
        end:      Optional[str] = None,
        n_ticks:  int  = 10_000,       # for live sources
        session_only: bool = True,     # London session + NY open 07-17 UTC (Dukascopy)
    ) -> pd.DataFrame:

        pair = pair.upper().replace("/", "")

        if source == "dukascopy":
            if not start or not end:
                raise ValueError("Dukascopy requires start and end dates")
            
            try:
                df = self.query_dukascopy_duckdb(pair, start, end)
                if not df.empty:
                    if session_only:
                        df = df[(df['timestamp_utc'].dt.hour >= 7) & (df['timestamp_utc'].dt.hour < 18)]
                    return df
            except Exception as e:
                print(f"[DuckDB Error] {e}")
                
            hours = list(range(7, 18)) if session_only else None
            return self.duka.load(pair, start, end, hours)

        elif source == "tds":
            return self.tds.load_directory(pair, start, end)

        elif source == "databento":
            return self.databento.load(pair, start, end)

        elif source == "hybrid":
            # Hybrid: stitch best available data per date range.
            # Priority per segment: databento mbp-10 > databento trades > dukascopy
            # This covers the case where mbp-10 only exists for part of the range.
            from pathlib import Path as _Path

            db_dir = _Path(self.databento.data_dir)
            mbp10_files  = sorted(db_dir.glob(f"{pair}_mbp-10_*.parquet"))
            trades_files = sorted(db_dir.glob(f"{pair}_trades_*.parquet"))

            frames = []

            # 1. Load databento mbp-10 where available (highest quality)
            if mbp10_files:
                df_mbp = self.databento.load(pair, start, end)
                if not (len(df_mbp) == 0) if hasattr(df_mbp, 'is_empty') else not df_mbp.empty:
                    import pandas as _pd
                    if hasattr(df_mbp, 'to_pandas'):
                        df_mbp = df_mbp.to_pandas()
                    df_mbp = _enforce_schema(df_mbp.set_index("timestamp_utc") if "timestamp_utc" in df_mbp.columns else df_mbp, pair, "databento")
                    frames.append(("mbp10", df_mbp))

            # Determine gap: dates NOT covered by mbp-10
            if frames:
                mbp10_end = frames[0][1].index.max()
            else:
                mbp10_end = None

            # 2. Fill gap with databento trades
            if trades_files and start and end:
                import pandas as _pd
                gap_start = (mbp10_end + _pd.Timedelta("1ms")).strftime("%Y-%m-%d") if mbp10_end else start
                if gap_start < end:
                    # Temporarily swap the glob pattern to load trades files
                    from data.databento_loader import DatabentoLoader as _DBL
                    _DBL(data_dir=str(db_dir), verbose=self.verbose)
                    # Patch to load trades schema files
                    trades_pattern = f"{pair}_trades_*.parquet"
                    import polars as _pl
                    try:
                        lazy = _pl.scan_parquet(str(db_dir / trades_pattern))
                        schema = lazy.collect_schema()
                        ts_col = "ts_event" if "ts_event" in schema.names() else "ts_recv"
                        import datetime as _dt
                        _s = _dt.datetime.strptime(gap_start, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
                        _e = _dt.datetime.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=_dt.timezone.utc)
                        lazy = lazy.filter((_pl.col(ts_col) >= _s) & (_pl.col(ts_col) <= _e))
                        first_px = lazy.select(_pl.col("bid_px_00").first()).collect()["bid_px_00"][0]
                        scale = 1e9 if first_px and first_px > 10000 else 1.0
                        df_tr = lazy.select([
                            _pl.col(ts_col).alias("timestamp_utc"),
                            (_pl.col("bid_px_00") / scale).cast(_pl.Float32).alias("bid"),
                            (_pl.col("ask_px_00") / scale).cast(_pl.Float32).alias("ask"),
                            ((_pl.col("bid_sz_00") + _pl.col("ask_sz_00")) / 2.0).cast(_pl.Float32).alias("volume"),
                        ]).collect().to_pandas().set_index("timestamp_utc")
                        df_tr = _enforce_schema(df_tr, pair, "databento_trades")
                        if not df_tr.empty:
                            frames.append(("trades", df_tr))
                    except Exception as _e2:
                        if self.verbose:
                            print(f"[Hybrid] trades load failed: {_e2}")

            # 3. Fill any remaining gap with Dukascopy
            covered_end = max(f.index.max() for _, f in frames) if frames else None
            import pandas as _pd2
            duka_start = (covered_end + _pd2.Timedelta("1ms")).strftime("%Y-%m-%d") if covered_end else start
            if duka_start and end and duka_start < end:
                hours = list(range(7, 18)) if session_only else None
                try:
                    df_duka = self.duka.load(pair, duka_start, end, hours)
                    if not df_duka.empty:
                        frames.append(("dukascopy", df_duka))
                except Exception as _e3:
                    if self.verbose:
                        print(f"[Hybrid] Dukascopy fill failed: {_e3}")

            if not frames:
                return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

            combined = pd.concat([f for _, f in frames], copy=False).sort_index()
            combined = combined[~combined.index.duplicated(keep="first")]
            if self.verbose:
                sources_used = [s for s, _ in frames]
                print(f"[Hybrid] {pair}: {len(combined):,} ticks | sources={sources_used} | {combined.index.min()} -> {combined.index.max()}")
            return combined

        elif source == "lmax_historical":
            return self.lmax.load_historical_csv(pair, start=start, end=end)

        elif source == "lmax_live":
            return self.lmax.stream_ticks(pair, n_ticks=n_ticks)

        elif source == "myfxbook":
            return self.myfxbook.load(pair, start, end)

        elif source == "eodhd":
            return self.eodhd.load(pair, start=start, end=end)

        elif source == "auto":
            # Try sources in priority order
            errors = []
            for src in self.SOURCE_PRIORITY:
                try:
                    return self.load(pair, src, start, end, n_ticks, session_only)
                except (FileNotFoundError, ValueError) as e:
                    errors.append(f"  {src}: {e}")
            # All failed -> synthetic fallback
            print(f"[DataManager] All sources failed for {pair}. Using synthetic data.")
            if self.verbose:
                for e in errors: print(e)
            from data.data_ingestion import generate_synthetic_tick_data
            return generate_synthetic_tick_data(n_rows=n_ticks)

        else:
            raise ValueError(f"Unknown source '{source}'. "
                             f"Options: dukascopy, tds, lmax_historical, lmax_live, myfxbook, eodhd, auto")

    def load_all_pairs(
        self,
        source: str = "dukascopy",
        start:  Optional[str] = None,
        end:    Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load all 9 configured pairs. Uses load_multiple for Dukascopy (shared session)."""
        if source == "dukascopy":
            if not start or not end:
                raise ValueError("Dukascopy requires start and end dates")
            return self.duka.load_multiple(list(DUKA_PAIR_MAP.keys()), start, end)
        return {p: self.load(p, source, start, end) for p in DUKA_PAIR_MAP}

    def quality_report(self, df: pd.DataFrame, pair: str) -> dict:
        """Compute data quality metrics for any loaded tick DataFrame."""
        if df.empty:
            return {"error": "empty DataFrame"}

        gaps = df.index.to_series().diff().dt.total_seconds().dropna()
        n_spread_anomalies = (df["spread"] <= 0).sum()
        n_bid_ask_inversion = (df["bid"] >= df["ask"]).sum()
        n_duplicate_timestamps = int(pd.Index(df.index).duplicated(keep="first").sum())

        report = {
            "pair":             pair,
            "source":           df["source"].iloc[0] if "source" in df.columns else "unknown",
            "n_ticks":          len(df),
            "date_range":       f"{df.index[0]} -> {df.index[-1]}",
            "avg_spread_pips":  round(df["spread"].mean() / PIP_SIZES.get(pair, 0.0001), 3),
            "min_spread_pips":  round(df["spread"].min()  / PIP_SIZES.get(pair, 0.0001), 3),
            "max_spread_pips":  round(df["spread"].max()  / PIP_SIZES.get(pair, 0.0001), 3),
            "avg_gap_seconds":  round(gaps.mean(), 3),
            "max_gap_seconds":  round(gaps.max(), 1),
            "n_gaps_over_1min": int((gaps > 60).sum()),
            "n_spread_anomalies": int(n_spread_anomalies),
            "n_bid_ask_inversions": int(n_bid_ask_inversion),
            "n_duplicate_timestamps": n_duplicate_timestamps,
            "quality_score":    round(
                100 * (1 - (n_spread_anomalies + n_bid_ask_inversion + n_duplicate_timestamps) / max(len(df), 1)), 2
            ),
        }
        return report

    def _normalize_pair(self, pair: str) -> str:
        return pair.upper().replace("/", "")

    def _normalize_granularity(self, granularity: str) -> str:
        granularity = granularity.lower()
        if granularity not in {"daily", "monthly"}:
            raise ValueError("granularity must be 'daily' or 'monthly'")
        return granularity

    def _parse_date_bounds(
        self,
        start: Optional[str],
        end: Optional[str],
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        start_ts = pd.Timestamp(start, tz="UTC") if start else None
        end_ts = pd.Timestamp(end, tz="UTC") if end else None
        return start_ts, end_ts

    def _iter_dukascopy_hour_files(
        self,
        pair: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Tuple[datetime, Path]]:
        pair = self._normalize_pair(pair)
        pair_root = Path(self.duka.cache_dir) / pair
        if not pair_root.exists():
            return []

        start_ts, end_ts = self._parse_date_bounds(start, end)
        files: List[Tuple[datetime, Path]] = []
        for fp in sorted(pair_root.rglob("*.parquet")):
            try:
                year = int(fp.parent.parent.name)
                month = int(fp.parent.name)
                day_str, hour_str = fp.stem.split("_", maxsplit=1)
                dt = datetime(year, month, int(day_str), int(hour_str), tzinfo=timezone.utc)
            except (ValueError, IndexError):
                continue

            dt_ts = pd.Timestamp(dt)
            if start_ts is not None and dt_ts < start_ts:
                continue
            if end_ts is not None and dt_ts > end_ts + pd.Timedelta(days=1) - pd.Timedelta(hours=1):
                continue
            files.append((dt, fp))
        return files

    def _partition_key_for_hour(self, dt: datetime, granularity: str) -> pd.Timestamp:
        granularity = self._normalize_granularity(granularity)
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if granularity == "daily":
            return ts.floor("D")
        return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def _compact_partition_path(self, pair: str, partition_ts: pd.Timestamp, granularity: str) -> Path:
        pair = self._normalize_pair(pair)
        granularity = self._normalize_granularity(granularity)
        base = (
            self.duka_compact_dir
            / f"granularity={granularity}"
            / f"pair={pair}"
            / f"year={partition_ts.year:04d}"
            / f"month={partition_ts.month:02d}"
        )
        if granularity == "daily":
            base = base / f"day={partition_ts.day:02d}"
        return base / "ticks.parquet"

    def compact_dukascopy_cache(
        self,
        pairs: List[str],
        *,
        granularity: str = "daily",
        start: Optional[str] = None,
        end: Optional[str] = None,
        overwrite: bool = True,
    ) -> Dict[str, dict]:
        """
        Compact hourly Dukascopy cache files into daily or monthly parquet partitions.

        The raw hourly cache remains the recovery layer; compacted partitions are the
        faster analysis/query layer.
        """
        granularity = self._normalize_granularity(granularity)
        summary: Dict[str, dict] = {}

        for raw_pair in pairs:
            pair = self._normalize_pair(raw_pair)
            hour_files = self._iter_dukascopy_hour_files(pair, start=start, end=end)
            grouped: Dict[pd.Timestamp, List[Path]] = {}
            for dt, fp in hour_files:
                grouped.setdefault(self._partition_key_for_hour(dt, granularity), []).append(fp)

            pair_summary = {
                "granularity": granularity,
                "hour_files": len(hour_files),
                "partitions_written": 0,
                "ticks_written": 0,
                "outputs": [],
            }

            for partition_ts, paths in sorted(grouped.items()):
                out_path = self._compact_partition_path(pair, partition_ts, granularity)
                if out_path.exists() and not overwrite:
                    pair_summary["outputs"].append(str(out_path))
                    continue

                frames = []
                for path in paths:
                    try:
                        frames.append(pd.read_parquet(path))
                    except Exception:
                        continue
                if not frames:
                    continue

                combined = _enforce_schema(pd.concat(frames, copy=False), pair, "dukascopy")
                if combined.empty:
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                combined.to_parquet(out_path)
                pair_summary["partitions_written"] += 1
                pair_summary["ticks_written"] += len(combined)
                pair_summary["outputs"].append(str(out_path))

            summary[pair] = pair_summary
            if self.verbose:
                print(
                    f"[DataManager] Compacted {pair}: "
                    f"{pair_summary['partitions_written']} {granularity} partition(s)"
                )

        return summary

    def _duckdb_partition_glob(
        self,
        granularity: str,
        pair: Optional[str] = None,
    ) -> str:
        granularity = self._normalize_granularity(granularity)
        root = self.duka_compact_dir / f"granularity={granularity}"
        if pair:
            root = root / f"pair={self._normalize_pair(pair)}"
        return str(root / "**" / "*.parquet").replace("\\", "/")

    def build_dukascopy_duckdb(
        self,
        *,
        db_path: Optional[str] = None,
        granularity: str = "daily",
        pair: Optional[str] = None,
        view_name: str = "dukascopy_ticks",
        as_table: bool = False,
    ) -> str:
        """
        Create or refresh a DuckDB view (or table) over the compacted Dukascopy partitions.
        """
        if not _DUCKDB:
            raise ImportError("duckdb is not installed. Add it to the environment to use DuckDB queries.")

        granularity = self._normalize_granularity(granularity)
        glob_path = self._duckdb_partition_glob(granularity, pair=pair)
        db_target = db_path or str(self.duka_compact_dir / "dukascopy_ticks.duckdb")
        escaped_glob_path = glob_path.replace("'", "''")

        target_type = "TABLE" if as_table else "VIEW"

        with duckdb.connect(db_target) as conn:
            if as_table:
                conn.execute(f"DROP TABLE IF EXISTS {view_name}")
            else:
                conn.execute(f"DROP VIEW IF EXISTS {view_name}")
            conn.execute(
                f"""
                CREATE OR REPLACE {target_type} {view_name} AS
                SELECT
                    CAST(timestamp AS TIMESTAMPTZ) AS timestamp,
                    bid, ask, mid, spread, volume,
                    CAST(pair AS VARCHAR) AS pair,
                    CAST(source AS VARCHAR) AS source
                FROM read_parquet('{escaped_glob_path}', hive_partitioning = true)
                """
            )
        return db_target

    def query_dukascopy_duckdb(
        self,
        pair: str,
        start: str,
        end: str,
        *,
        db_path: Optional[str] = None,
        granularity: str = "daily",
        view_name: str = "dukascopy_ticks",
    ) -> pd.DataFrame:
        """
        Directly read the parquet files without DuckDB overhead.
        """
        pair = self._normalize_pair(pair)
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        
        granularity = self._normalize_granularity(granularity)
        pair_dir = self.duka_compact_dir / f"granularity={granularity}" / f"pair={pair}"
        
        dfs = []
        for dt in pd.date_range(start_ts, end_ts, freq="D", tz="UTC"):
            path = pair_dir / f"year={dt.year}" / f"month={dt.month:02d}" / f"day={dt.day:02d}" / "ticks.parquet"
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    dfs.append(df)
                except Exception as e:
                    if self.verbose: print(f"Error reading {path}: {e}")
        
        if not dfs:
            return pd.DataFrame()
            
        df = pd.concat(dfs, ignore_index=False)
        # The parquet files have 'timestamp' as the index. pd.concat preserves it.
        if df.index.name in ['timestamp', 'timestamp_utc']:
            df = df.reset_index()
            if 'timestamp' in df.columns:
                df.rename(columns={'timestamp': 'timestamp_utc'}, inplace=True)
        elif 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'timestamp_utc'}, inplace=True)
        
        df = df[(df['timestamp_utc'] >= start_ts) & (df['timestamp_utc'] < end_ts)]
        
        if 'pair' not in df.columns: df['pair'] = pair
        if 'source' not in df.columns: df['source'] = "dukascopy"
        
        return df.sort_values('timestamp_utc').reset_index(drop=True)

    def query_dukascopy_compacted(
        self,
        pair: str,
        start: str,
        end: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Backward-compatible alias for querying compacted Dukascopy parquet."""
        return self.query_dukascopy_duckdb(pair, start, end, **kwargs)

    def download_dukascopy_year_by_year(
        self,
        pairs: List[str],
        start_year: int,
        end_year: int,
        session_only: bool = True,
        max_redownload_passes: int = 2,
        fail_on_missing: bool = True,
    ) -> Dict[str, Dict[int, dict]]:
        """
        Download Dukascopy data pair-by-pair and year-by-year.

        Each pair/year slice is checked before moving on. By default the method
        stops immediately if any year still has missing requested hours after the
        automatic redownload passes.
        """
        if end_year < start_year:
            raise ValueError("end_year must be >= start_year")

        hours = list(range(7, 18)) if session_only else None
        results: Dict[str, Dict[int, dict]] = {}
        now = datetime.now(timezone.utc)

        for raw_pair in pairs:
            pair = raw_pair.upper().replace("/", "")
            results[pair] = {}
            if self.verbose:
                print(f"[DataManager] Starting pair {pair}")

            for year in range(start_year, end_year + 1):
                year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
                year_end = min(datetime(year, 12, 31, tzinfo=timezone.utc), now)
                if year_start > year_end:
                    continue

                start = year_start.strftime("%Y-%m-%d")
                end = year_end.strftime("%Y-%m-%d")
                if self.verbose:
                    print(f"[DataManager] Downloading {pair} for {year}")

                df = self.duka.load(
                    pair,
                    start,
                    end,
                    hours=hours,
                    auto_redownload=True,
                    max_redownload_passes=max_redownload_passes,
                )
                requested = self.duka._build_tasks_dt(year_start, year_end, hours)
                coverage = self.duka._hour_coverage_report(df, requested)
                quality = self.quality_report(df, pair) if not df.empty else {"error": "empty DataFrame"}
                results[pair][year] = {
                    "start": start,
                    "end": end,
                    "ticks": len(df),
                    "coverage": coverage,
                    "quality": quality,
                }

                if fail_on_missing and coverage["missing_hours"]:
                    raise RuntimeError(
                        f"{pair} {year} still missing {coverage['missing_hours_count']} hour(s) after retries"
                    )

        return results


# ─────────────────────────────────────────────────────────────────────────────
# QUICK DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Data Sources — connection test")
    print("=" * 55)

    mgr = ForexDataManager(verbose=True)

    # 1. Dukascopy — try loading 3 days of EURUSD
    print("\n[1] Dukascopy free tick data")
    try:
        df_duka = mgr.load(
            "EURUSD", source="dukascopy",
            start="2024-01-02", end="2024-01-04",
            session_only=True,
        )
        print(f"    Ticks: {len(df_duka):,}")
        print(f"    Sample:\n{df_duka.head(3)}")
        rpt = mgr.quality_report(df_duka, "EURUSD")
        print(f"    Quality: {rpt['quality_score']}% | "
              f"Avg spread: {rpt['avg_spread_pips']} pips")
    except Exception as e:
        print(f"    Dukascopy error: {e}")
        print("    (Expected if no internet — offline / air-gapped)")

    # 2. TDS — check if export files exist
    print("\n[2] Tick Data Suite")
    tds_dir = Path(DEFAULT_TDS_DATA_DIR) / "EURUSD"
    if tds_dir.exists() and list(tds_dir.glob("*.csv")):
        try:
            df_tds = mgr.load("EURUSD", source="tds",
                               start="2024-01-01", end="2024-01-31")
            print(f"    Ticks: {len(df_tds):,}")
        except Exception as e:
            print(f"    TDS load error: {e}")
    else:
        print(f"    No TDS files found in {tds_dir}/")
        print("    Export tick data from Tick Data Suite -> place CSVs there")

    # 3. LMAX — historical CSV check
    print("\n[3] LMAX Exchange")
    lmax_dir = Path(DEFAULT_LMAX_DATA_DIR) / "EURUSD"
    if lmax_dir.exists() and list(lmax_dir.glob("*.csv")):
        try:
            df_lmax = mgr.load("EURUSD", source="lmax_historical",
                                start="2024-01-01", end="2024-01-31")
            print(f"    Bars: {len(df_lmax):,}")
        except Exception as e:
            print(f"    LMAX load error: {e}")
    else:
        print(f"    No LMAX files found in {lmax_dir}/")
        print("    Download from: https://www.lmax.com/exchange/market-data")
        print("    For live feed: set LMAX_USERNAME + LMAX_PASSWORD env vars")

    print("\n[Schema] All sources produce the same columns:")
    print(f"    {TICK_COLUMNS}")

