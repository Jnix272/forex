"""
data/myfxbook.py
================
Loader for Myfxbook historical daily OHLCV data.

Myfxbook (myfxbook.com -> Tools -> Historical Data) exports a CSV with a
title row, a header row, then daily bars newest-first:

    EURGBP Historical Data
    Date,Open,High,Low,Close,Change(Pips),Change(%)
    04/07/2026 00:00,0.8721,0.8725,0.87181,0.87239,2.9,0.03,
    04/06/2026 00:00,0.87278,...

Place exported CSVs here:
    data/raw/myfxbook/<PAIR>/<any_name>.csv

The loader:
  - Skips the title row and parses the header automatically
  - Parses MM/DD/YYYY HH:MM dates to UTC DatetimeIndex
  - Synthesizes bid/ask from OHLCV mid using typical pair spread
  - Produces the project-standard tick schema so MyfxbookLoader output
    can be used anywhere DukascopyLoader output is used

Usage
-----
    from data.myfxbook import MyfxbookLoader

    loader = MyfxbookLoader()
    df = loader.load("EURGBP")
    print(df.head())

    # Load a specific file directly
    df = loader.load_file("path/to/EURGBP_historical_data.csv", "EURGBP")
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from data.sources import (
    _DATA_DIR,
    TICK_COLUMNS,
    _enforce_schema,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MYFXBOOK_DATA_DIR = str(_DATA_DIR / "raw" / "myfxbook")

# Typical half-spread per pair (in price units) used to synthesise bid/ask
# when only OHLCV mid prices are available.
# Values represent ~1–2 pip typical spread for each pair.  # noqa: RUF003
_TYPICAL_HALF_SPREAD: dict[str, float] = {
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
}
_DEFAULT_HALF_SPREAD = 0.00010


def _pair_half_spread(pair: str) -> float:
    return _TYPICAL_HALF_SPREAD.get(pair.upper(), _DEFAULT_HALF_SPREAD)


# ── Parsing ───────────────────────────────────────────────────────────────────


def _parse_myfxbook_csv(filepath: str | Path, pair: str) -> pd.DataFrame:
    """
    Parse one Myfxbook CSV export.

    Returns a DataFrame with columns [open, high, low, close,
    change_pips, change_pct] and a UTC DatetimeIndex.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Myfxbook CSV not found: {path}")

    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    if len(lines) < 3:
        return pd.DataFrame()

    # Row 0 is the title e.g. "EURGBP Historical Data"
    # Row 1 is the header "Date,Open,High,Low,Close,Change(Pips),Change(%)"
    # Find the header row (first row that starts with "Date")
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("date"):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    from io import StringIO

    body = "".join(lines[header_idx:])
    # Strip trailing commas on each data line (Myfxbook quirk)
    body = re.sub(r",\s*\n", "\n", body)

    try:
        df = pd.read_csv(StringIO(body))
    except Exception:
        return pd.DataFrame()

    # Normalise column names
    df.columns = [
        c.strip().lower().replace("(", "_").replace(")", "").replace("%", "pct").replace("/", "_") for c in df.columns
    ]
    # Expected: date, open, high, low, close, change_pips, change_pct

    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()

    # Parse date (MM/DD/YYYY HH:MM -> UTC)
    try:
        df["timestamp"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M", utc=True, errors="coerce")
    except Exception:
        df["timestamp"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["change_pips"] = pd.to_numeric(df.get("change_pips", np.nan), errors="coerce")
    df["change_pct"] = pd.to_numeric(df.get("change_pct", np.nan), errors="coerce")

    return df.dropna(subset=["close"])


# ── Loader class ──────────────────────────────────────────────────────────────


class MyfxbookLoader:
    """
    Load Myfxbook daily OHLCV bars and convert to the project-standard
    tick schema (bid/ask synthesised from mid + typical half-spread).

    Directory layout
    ----------------
        data/raw/myfxbook/
            EURGBP/
                EURGBP_historical_data.csv
                EURGBP_2024.csv
            EURUSD/
                EURUSD_historical_data.csv

    Parameters
    ----------
    data_dir : str
        Root directory containing per-pair sub-folders.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_MYFXBOOK_DATA_DIR,
        verbose: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────────

    def load_file(self, filepath: str, pair: str) -> pd.DataFrame:
        """
        Parse a single Myfxbook CSV and return a tick-schema DataFrame.

        bid/ask are synthesised: bid = close - half_spread,
                                 ask = close + half_spread.
        The daily bar is represented as a single "tick" at midnight UTC.
        """
        pair = pair.upper().replace("/", "")
        raw = _parse_myfxbook_csv(filepath, pair)
        if raw.empty:
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))
        return self._to_tick_schema(raw, pair)

    def load(
        self,
        pair: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Load and concatenate all CSVs found in data_dir/<pair>/.

        Parameters
        ----------
        pair  : e.g. "EURGBP" or "EUR/GBP"
        start : optional ISO date filter "YYYY-MM-DD"
        end   : optional ISO date filter "YYYY-MM-DD"
        """
        pair = pair.upper().replace("/", "")
        pair_dir = self.data_dir / pair

        if not pair_dir.exists():
            if self.verbose:
                print(f"[Myfxbook] No data directory for {pair}: {pair_dir}")
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

        csv_files = sorted(pair_dir.glob("*.csv"))
        if not csv_files:
            if self.verbose:
                print(f"[Myfxbook] No CSV files found in {pair_dir}")
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

        frames = []
        for f in csv_files:
            try:
                raw = _parse_myfxbook_csv(f, pair)
                if not raw.empty:
                    frames.append(raw)
                    if self.verbose:
                        print(f"[Myfxbook] {f.name}: {len(raw):,} bars")
            except Exception as e:
                if self.verbose:
                    print(f"[Myfxbook] Skip {f.name}: {e}")

        if not frames:
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

        combined = pd.concat(frames).sort_index()[~pd.concat(frames).sort_index().index.duplicated(keep="last")]

        # Date filter
        if start:
            combined = combined[combined.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            combined = combined[combined.index <= pd.Timestamp(end, tz="UTC")]

        result = self._to_tick_schema(combined, pair)

        if self.verbose:
            print(
                f"[Myfxbook] {pair}: {len(result):,} bars loaded | "
                f"{result.index[0].date()} -> {result.index[-1].date()}"
            )
        return result

    def load_ohlcv(
        self,
        pair: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Return raw daily OHLCV DataFrame (no bid/ask synthesis).

        Columns: open, high, low, close, change_pips, change_pct
        Index:   UTC DatetimeIndex (daily)
        """
        pair = pair.upper().replace("/", "")
        pair_dir = self.data_dir / pair

        if not pair_dir.exists():
            return pd.DataFrame()

        frames = []
        for f in sorted(pair_dir.glob("*.csv")):
            try:
                raw = _parse_myfxbook_csv(f, pair)
                if not raw.empty:
                    frames.append(raw)
            except Exception:
                pass

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

        if start:
            combined = combined[combined.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            combined = combined[combined.index <= pd.Timestamp(end, tz="UTC")]

        return combined

    def ingest_file(self, filepath: str, pair: str) -> Path:
        """
        Copy a downloaded Myfxbook CSV into the correct data_dir/<pair>/ folder.

        Returns the destination path.
        """
        pair = pair.upper().replace("/", "")
        dest_dir = self.data_dir / pair
        dest_dir.mkdir(parents=True, exist_ok=True)

        src = Path(filepath)
        dest = dest_dir / src.name
        import shutil

        shutil.copy2(src, dest)
        if self.verbose:
            print(f"[Myfxbook] Ingested {src.name} -> {dest}")
        return dest

    # ── internal ──────────────────────────────────────────────────────────────

    def _to_tick_schema(self, raw: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Convert parsed OHLCV rows to the project tick schema."""
        half = _pair_half_spread(pair)

        mid = raw["close"].values.astype(np.float64)
        bid = mid - half
        ask = mid + half
        vol = np.ones(len(raw), dtype=np.float64)  # daily volume not in export

        result = pd.DataFrame(
            {"bid": bid, "ask": ask, "volume": vol},
            index=raw.index,
        )
        return _enforce_schema(result, pair, "myfxbook")
