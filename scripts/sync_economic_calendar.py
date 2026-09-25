"""
scripts/sync_economic_calendar.py
==================================
Synchronizes current economic calendar events from the Fair Economy Media CDN
into data/raw/eco_calendar/events.csv.

Bypasses Cloudflare bot challenges by consuming the official weekly JSON feed:
  https://nfs.faireconomy.media/ff_calendar_thisweek.json

Features:
  - Local caching in data/raw/eco_calendar/ff_calendar_thisweek.json (TTL: 4 hours).
  - Handles HTTP 429 with Retry-After backoff.
  - Normalizes ISO timestamps with timezone offsets to standard UTC ISO (Z).
  - Appends and de-duplicates events into data/raw/eco_calendar/events.csv.
  - Verifies DuckDB slicing for EconomicCalendarGuard.

Usage:
  python scripts/sync_economic_calendar.py
  python scripts/sync_economic_calendar.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sync_economic_calendar")

EVENTS_CSV_PATH = Path("data/raw/eco_calendar/events.csv")
CACHE_JSON_PATH = Path("data/raw/eco_calendar/ff_calendar_thisweek.json")
CDN_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

CSV_FIELDS = [
    "timestamp_utc",
    "event_type",
    "currency",
    "impact",
    "headline",
    "actual",
    "forecast",
    "prior",
    "source",
    "url",
]
DEDUP_KEY = ("timestamp_utc", "currency", "headline")

_NUM_RE = re.compile(r"^[<>~]?\s*(-?\d[\d,]*\.?\d*)\s*([kmbt%])?\s*$", re.IGNORECASE)
_MULTIPLIER = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def clean_numeric(raw: str) -> str:
    s = str(raw or "").strip()
    if not s or s in {"-", "--"}:
        return ""
    m = _NUM_RE.match(s)
    if not m:
        return s
    number = float(m.group(1).replace(",", ""))
    suffix = (m.group(2) or "").lower()
    if suffix in _MULTIPLIER:
        number *= _MULTIPLIER[suffix]
    if number == int(number):
        return str(int(number))
    return repr(number)


def fetch_calendar_data(force: bool = False, cache_ttl_sec: float = 14400.0) -> list[dict]:
    """Fetch calendar data from local cache or CDN."""
    now = time.time()
    if not force and CACHE_JSON_PATH.exists():
        age = now - CACHE_JSON_PATH.stat().st_mtime
        if age < cache_ttl_sec:
            logger.info(f"Loading cached calendar data from {CACHE_JSON_PATH} (age: {age/60:.1f} min)")
            try:
                return json.loads(CACHE_JSON_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to read cache file: {e}; re-fetching from CDN...")

    logger.info(f"Fetching weekly calendar from CDN: {CDN_URL}")
    req = urllib.request.Request(
        CDN_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        },
    )

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                data = json.loads(body)
                CACHE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
                CACHE_JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
                logger.info(f"Successfully fetched {len(data)} events from CDN and saved cache.")
                return data
        except urllib.error.HTTPError as he:
            if he.code == 429 and attempt < max_attempts:
                retry_after_hdr = he.headers.get("Retry-After")
                retry_after = float(retry_after_hdr) if retry_after_hdr else 30.0
                logger.warning(f"HTTP 429 Rate limited. Waiting {retry_after:.0f}s before retry {attempt+1}/{max_attempts}...")
                time.sleep(retry_after + 1.0)
            else:
                logger.error(f"HTTP error fetching CDN calendar: {he.code} {he.reason}")
                break
        except Exception as exc:
            logger.error(f"Network error fetching CDN calendar: {exc}")
            if attempt < max_attempts:
                time.sleep(5.0)
            else:
                break

    # If fetch failed but stale cache exists, fall back to it
    if CACHE_JSON_PATH.exists():
        logger.warning("Falling back to stale local cache file.")
        try:
            return json.loads(CACHE_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return []


def parse_and_sync(events_data: list[dict], out_path: Path = EVENTS_CSV_PATH) -> int:
    if not events_data:
        logger.warning("No events to sync.")
        return 0

    # 1. Read existing events to prevent duplicates
    seen_keys = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                seen_keys.add(tuple(row.get(k, "") for k in DEDUP_KEY))

    new_rows = []
    for item in events_data:
        raw_date = item.get("date", "")
        if not raw_date:
            continue
        try:
            dt = datetime.fromisoformat(raw_date).astimezone(UTC)
            ts_utc = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            continue

        headline = str(item.get("title", "")).strip()
        currency = str(item.get("country", "")).strip().upper()
        impact = str(item.get("impact", "")).strip().lower()

        key = (ts_utc, currency, headline)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        row = {
            "timestamp_utc": ts_utc,
            "event_type": "calendar",
            "currency": currency,
            "impact": impact,
            "headline": headline,
            "actual": clean_numeric(item.get("actual", "")),
            "forecast": clean_numeric(item.get("forecast", "")),
            "prior": clean_numeric(item.get("previous", "")),
            "source": "faireconomy",
            "url": f"https://www.forexfactory.com/calendar?week={dt.strftime('%b').lower()}{dt.day}.{dt.year}",
        }
        new_rows.append(row)

    if not new_rows:
        logger.info("Calendar is already up to date. No new rows appended.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists() and out_path.stat().st_size > 0
    with out_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    logger.info(f"Successfully appended {len(new_rows)} new economic events to {out_path}")
    return len(new_rows)


def verify_duckdb_slice() -> None:
    """Verify DuckDB slicing against events.csv for current time window."""
    try:
        import duckdb
        import pandas as pd
        now = pd.Timestamp.utcnow()
        start_ts = (now - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_ts = (now + pd.Timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        con = duckdb.connect()
        csv_file = str(EVENTS_CSV_PATH).replace("\\", "/")
        query = f"""
            SELECT count(*) as cnt, count(CASE WHEN impact = 'high' THEN 1 END) as high_cnt
            FROM read_csv_auto('{csv_file}')
            WHERE timestamp_utc >= '{start_ts}' AND timestamp_utc <= '{end_ts}'
        """
        res = con.execute(query).fetchone()
        con.close()
        total_events, high_events = res[0], res[1]
        logger.info(f"[DuckDB Check] Active events found in window [{start_ts} to {end_ts}]: total={total_events}, high_impact={high_events}")
    except Exception as exc:
        logger.error(f"[DuckDB Check] Verification error: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Sync live economic calendar from CDN.")
    parser.add_argument("--force", action="store_true", help="Force re-fetch from CDN ignoring cache")
    args = parser.parse_args()

    data = fetch_calendar_data(force=args.force)
    appended = parse_and_sync(data)
    verify_duckdb_slice()
    return 0


if __name__ == "__main__":
    sys.exit(main())
