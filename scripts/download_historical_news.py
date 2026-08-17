"""
Download forex-focused historical news/calendar data for offline training.

Outputs match data.historical_news.load_historical_news_bundle:

  data/raw/news/historical_news_combined.parquet
  data/raw/eco_calendar/events.csv

Sources:
  - GDELT DOC 2.0 article list API (free, no key)
  - EODHD news API (requires EODHD_API_KEY)
  - EODHD economic events API (requires EODHD_API_KEY, plan-dependent)

Key improvements over prior version:
  - Parallel pair fetching via --workers N (ThreadPoolExecutor)
  - Incremental CSV append with threading.Lock (no full-rewrite per checkpoint)
  - Auto-split cap-hit windows (--gdelt-split-on-cap): when GDELT returns
    exactly maxrecords, re-query AM+PM sub-windows to recover missed articles
  - --retry-failures: replay all entries from the failures CSV
  - tqdm progress bar with graceful plain-print fallback
  - --dry-run: preview chunks without making any HTTP calls
  - Per-chunk flush by default (--checkpoint-every 1)
  - Thread-safe writes via threading.Lock
"""

from __future__ import annotations

import argparse
import csv
import email.utils
import json
import os
import random
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import requests

_csv_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_limit)
        break
    except OverflowError:
        _csv_limit //= 10
from requests.adapters import HTTPAdapter

_SESSION = requests.Session()
_SESSION.mount("https://", HTTPAdapter(pool_connections=16, pool_maxsize=32))

try:
    import wandb
except ImportError:
    wandb = None


class _RateLimiter:
    """
    Thread-safe minimum-interval throttle shared across worker threads.

    Ensures consecutive calls to wait() are spaced at least `min_interval`
    seconds apart, regardless of how many threads are calling it. This is the
    only reliable way to stay under GDELT's burst limit when fetching multiple
    pairs in parallel (--workers > 1), because each pair otherwise fires
    requests back-to-back with no spacing.
    """

    def __init__(self, min_interval: float = 0.0) -> None:
        self._min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def set_interval(self, min_interval: float) -> None:
        with self._lock:
            self._min_interval = max(0.0, float(min_interval))

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + self._min_interval
                    return
                sleep_for = self._next_allowed - now
            time.sleep(sleep_for)


# Global throttle applied before every GDELT request (configured in main()).
_GDELT_LIMITER = _RateLimiter(0.0)


# ── Optional tqdm ----------------------------------------------------------------─────────────
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, *, desc: str = "", total: int | None = None, leave: bool = True):
        return _tqdm(iterable, desc=desc, total=total, leave=leave, unit="chunk", dynamic_ncols=True)

    HAS_TQDM = True
except ImportError:  # pragma: no cover
    HAS_TQDM = False

    class _FallbackBar:  # minimal shim
        def __init__(self, iterable, *, desc="", total=None, leave=True):
            self._it = iter(iterable)
            self._desc = desc
            self._n = 0
            self._total = total
            if desc:
                print(f"[progress] {desc} (no tqdm installed)", flush=True)

        def __iter__(self):
            return self

        def __next__(self):
            val = next(self._it)
            self._n += 1
            pct = f"{100 * self._n / self._total:.0f}%" if self._total else str(self._n)
            print(f"\r[progress] {self._desc} {pct}", end="", flush=True)
            return val

        def close(self):
            print(flush=True)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def _progress(iterable, *, desc: str = "", total: int | None = None, leave: bool = True):
        return _FallbackBar(iterable, desc=desc, total=total, leave=leave)


# ── Constants ----------------------------------------------------------------─────────────────
SCRIPT_VERSION = "2026-05-21-parallel-incremental"

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
EODHD_NEWS_URL = "https://eodhd.com/api/news"
EODHD_ECONOMIC_URLS = (
    "https://eodhd.com/api/economic-events",
    "https://eodhd.com/api/calendar/economic",
)

OFFICIAL_FEEDS: dict[str, dict[str, str]] = {
    "FED": {
        "currency": "USD",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
    },
    "ECB": {
        "currency": "EUR",
        "url": "https://www.ecb.europa.eu/rss/press.html",
    },
    "BOE": {
        "currency": "GBP",
        "url": "https://www.bankofengland.co.uk/boeapps/rss/news.xml",
    },
    "BOJ": {
        "currency": "JPY",
        "url": "https://www.boj.or.jp/en/rss/whatsnew.xml",
    },
    "RBA": {
        "currency": "AUD",
        "url": "https://www.rba.gov.au/rss/rss-cb-media-releases.xml",
    },
    "BOC": {
        "currency": "CAD",
        "url": "https://www.bankofcanada.ca/feed/",
    },
}

DEFAULT_NEWS_OUT = Path("data/raw/news/historical_news_combined.parquet")
DEFAULT_CAL_OUT = Path("data/raw/eco_calendar/events.csv")

CSV_FIELDS = [
    "timestamp_utc",
    "event_type",
    "currency",
    "impact",
    "headline",
    "actual",
    "forecast",
    "source",
    "url",
    # event_category is APPENDED last for backward compatibility: older news
    # CSVs written before this column existed still load cleanly because every
    # reader/writer normalizes rows via `.get(field, "")` (see _read_existing,
    # _IncrementalCSVWriter.append/flush_dedup, and write_rows), so a missing
    # column resolves to "" rather than raising.
    "event_category",
]
# Calendar rows carry an extra optional `prior` (previous reading) used to
# normalize economic surprises. Kept separate from the shared news CSV_FIELDS so
# the (large) news file schema is untouched; events.csv is rewritten in full by
# write_rows, so adding the column here is safe/backward-compatible.
CAL_FIELDS = [*CSV_FIELDS, "prior"]
FAILURE_FIELDS = ["source", "pair", "start_utc", "end_utc", "reason"]

# Map EODHD country codes (and a few aliases) to ISO currency codes. The loader's
# relevance filter matches ISO currencies (USD/EUR/...), so calendar rows that
# only expose a country must be translated here.
_COUNTRY_TO_CCY = {
    "US": "USD",
    "USA": "USD",
    "EU": "EUR",
    "EZ": "EUR",
    "EA": "EUR",
    "EMU": "EUR",
    "DE": "EUR",
    "FR": "EUR",
    "IT": "EUR",
    "ES": "EUR",
    "NL": "EUR",
    "GB": "GBP",
    "UK": "GBP",
    "JP": "JPY",
    "AU": "AUD",
    "NZ": "NZD",
    "CA": "CAD",
    "CH": "CHF",
}


def _calendar_currency(event: dict, title: str) -> str:
    """Resolve an ISO currency for a calendar event from currency/country fields."""
    for token in (
        str(event.get("currency", "")).strip().upper(),
        str(event.get("country", "")).strip().upper(),
    ):
        if not token:
            continue
        if token in _COUNTRY_TO_CCY:
            return _COUNTRY_TO_CCY[token]
        if len(token) == 3 and token.isalpha():
            return token
    return _event_currency_from_text(title, "GLOBAL")


PAIR_KEYWORDS: dict[str, list[str]] = {
    "EUR": ['"European Central Bank"', "ECB", "euro", "Eurozone"],
    "USD": ['"Federal Reserve"', "Fed", "FOMC", "CPI", "NFP", '"nonfarm payrolls"', "inflation"],
    "GBP": ['"Bank of England"', "BOE", "sterling", "pound", "UK CPI"],
    "JPY": ['"Bank of Japan"', "BOJ", "yen", "Japan CPI"],
    "AUD": ['"Reserve Bank of Australia"', "RBA", "Aussie"],
    "NZD": ['"Reserve Bank of New Zealand"', "RBNZ", "kiwi"],
    "CAD": ['"Bank of Canada"', "BOC", "loonie", "Canada CPI"],
    "CHF": ['"Swiss National Bank"', "SNB", "franc"],
}


# ── Date helpers ----------------------------------------------------------------──────────────


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


def _daterange(start: datetime, end: datetime, step_days: int) -> Iterable[tuple[datetime, datetime]]:
    cur = start
    step = timedelta(days=max(1, int(step_days)))
    while cur <= end:
        chunk_end = min(
            cur + step - timedelta(seconds=1),
            end + timedelta(days=1) - timedelta(seconds=1),
        )
        yield cur, chunk_end
        cur = chunk_end + timedelta(seconds=1)


def _chunk_dates(start: datetime, end: datetime) -> set[str]:
    dates: set[str] = set()
    cur = start.date()
    last = end.date()
    while cur <= last:
        dates.add(cur.isoformat())
        cur += timedelta(days=1)
    return dates


# ── HTTP helpers ------------------------------------------------------------------------------


def _parse_json_response(raw: str, url: str) -> object:
    text = raw.lstrip("\ufeff").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as strict_exc:
        decoder = json.JSONDecoder(strict=False)
        try:
            payload, idx = decoder.raw_decode(text)
        except json.JSONDecodeError:
            snippet = text.replace("\n", " ")[:240].encode("ascii", "replace").decode("ascii")
            raise ValueError(f"Non-JSON response from {url}: {snippet}") from strict_exc
        trailing = text[idx:].strip()
        if trailing:
            snippet = trailing.replace("\n", " ")[:120].encode("ascii", "replace").decode("ascii")
            raise ValueError(f"JSON response from {url} has trailing data: {snippet}") from strict_exc
        return payload


def _http_json(url: str, params: dict, timeout: int = 45) -> object:
    headers = {"User-Agent": "forex-scaling-model/2.0", "Accept-Encoding": "gzip, deflate"}
    response = _SESSION.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    raw = response.text
    return _parse_json_response(raw, url)


def _retry_sleep_seconds(exc: Exception, sleep_s: float, attempt: int) -> float:
    base = max(float(sleep_s), 1.0)
    retry_after = None
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None and exc.response.status_code == 429:
        try:
            retry_after = float(exc.response.headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            retry_after = None
        if retry_after is None or retry_after <= 0:
            # GDELT often omits Retry-After on 429s. Short exponential retries
            # can keep the client in a penalty box, so cool down decisively.
            return min(300.0, 60.0 * (2 ** max(0, attempt - 1))) + random.uniform(1.0, 5.0)
    if retry_after is not None and retry_after > 0:
        return retry_after + random.uniform(0.25, 1.0)
    return min(120.0, base * (2 ** max(0, attempt - 1))) + random.uniform(0.25, 1.0)


# ── Pair / text helpers ----------------------------------------------------------------───────


def _pair_currencies(pair: str) -> list[str]:
    clean = str(pair).replace("/", "").replace("_", "").upper()
    if len(clean) >= 6:
        return [clean[:3], clean[3:6]]
    return [clean]


def _gdelt_datetime(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _event_currency_from_text(text: str, pair: str) -> str:
    upper = str(text).upper()
    hits = [ccy for ccy in _pair_currencies(pair) if ccy and ccy in upper]
    if hits:
        return ",".join(sorted(set(hits)))
    for ccy, kws in PAIR_KEYWORDS.items():
        if any(str(k).strip('"').upper() in upper for k in kws):
            return ccy
    return "GLOBAL"


# ── Event categorization ──────────────────────────────────────────────────────
# Ordered category precedence (first match wins):
#   central_bank > inflation > labor > growth > geopolitical > commentary
# Matching is word-boundary aware (regex \b) to reduce false positives such as
# "war" matching "warning"/"forward", or "ism" matching "prism".
EVENT_CATEGORIES = [
    "central_bank",
    "inflation",
    "labor",
    "growth",
    "geopolitical",
    "commentary",  # catch-all default
]

# Keyword groups for the five non-default categories, in precedence order.
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "central_bank",
        (
            "fomc",
            "ecb",
            "boe",
            "boj",
            "rba",
            "rbnz",
            "boc",
            "snb",
            "fed",
            "federal reserve",
            "central bank",
            "rate decision",
            "interest rate",
            "interest rates",
            "monetary policy",
            "rate hike",
            "rate cut",
            "rate hikes",
            "rate cuts",
            "raises rates",
            "cuts rates",
            "hikes rates",
            "rate rises",
        ),
    ),
    (
        "inflation",
        (
            "cpi",
            "ppi",
            "pce",
            "inflation",
            "core prices",
            "deflation",
            "consumer prices",
            "producer prices",
        ),
    ),
    (
        "labor",
        (
            "nfp",
            "nonfarm payrolls",
            "non-farm payrolls",
            "payrolls",
            "jobs report",
            "unemployment",
            "jobless claims",
            "employment change",
            "labor market",
            "labour market",
        ),
    ),
    (
        "growth",
        (
            "gdp",
            "pmi",
            "ism",
            "industrial production",
            "retail sales",
            "manufacturing",
        ),
    ),
    (
        "geopolitical",
        (
            "war",
            "sanctions",
            "conflict",
            "election",
            "crisis",
            "tariff",
            "tariffs",
            "geopolitical",
            "opec",
            "attack",
        ),
    ),
]

# Precompile one word-boundary-anchored alternation per category.
_CATEGORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        name,
        re.compile(
            r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b",
            re.IGNORECASE,
        ),
    )
    for name, kws in _CATEGORY_KEYWORDS
]

# Category -> news impact mapping. central_bank/inflation/labor/geopolitical are
# market-moving ("high"); growth is "medium"; commentary falls back to the
# caller-supplied default (kept "medium" for backward compatibility).
_CATEGORY_IMPACT = {
    "central_bank": "high",
    "inflation": "high",
    "labor": "high",
    "growth": "medium",
    "geopolitical": "high",
    "commentary": "medium",
}


def _event_category_from_text(text: str) -> str:
    """Classify a headline/event into one of EVENT_CATEGORIES.

    Uses ordered precedence (first match wins) with word-boundary-aware keyword
    matching. Falls back to "commentary" when nothing matches.
    """
    s = str(text or "")
    for name, pattern in _CATEGORY_PATTERNS:
        if pattern.search(s):
            return name
    return "commentary"


def _impact_from_category(category: str, default: str = "medium") -> str:
    """Derive an impact level ("high"/"medium") from an event category."""
    return _CATEGORY_IMPACT.get(str(category), default)


def _impact_from_text(text: str, default: str = "medium") -> str:
    """Derive impact directly from a headline by delegating to the category fn.

    Kept for backward compatibility with existing call sites. For "commentary"
    (no category keyword found) it returns the caller-supplied default so prior
    behavior (medium for routine headlines, high for market-movers) is preserved.
    """
    category = _event_category_from_text(text)
    if category == "commentary":
        return default
    return _impact_from_category(category, default)


def _gdelt_query(pair: str) -> str:
    """Backward-compatible broad query string for one FX pair."""
    return " OR ".join(q for _, q in _gdelt_queries(pair))


def _gdelt_term(term: str) -> str:
    """Return a GDELT-safe search term, quoting phrases when needed."""
    raw = str(term or "").strip()
    if not raw:
        return ""
    if raw.startswith('"') and raw.endswith('"'):
        return raw
    if any(ch.isspace() for ch in raw):
        return f'"{raw}"'
    return raw


def _gdelt_queries(pair: str) -> list[tuple[str, str]]:
    """
    Build smaller GDELT queries for one pair.

    A single broad OR query across both currencies can time out on GDELT's DOC
    API, especially over dense macro/news days. Querying one currency side at a
    time keeps requests smaller while CSV dedupe removes duplicate headlines.
    """
    currencies = _pair_currencies(pair)
    clean_pair = str(pair).replace("/", "").replace("_", "").upper()
    slash_pair = f"{clean_pair[:3]}/{clean_pair[3:6]}" if len(clean_pair) >= 6 else clean_pair
    pair_terms = [clean_pair, slash_pair]

    queries: list[tuple[str, str]] = []
    for ccy in currencies:
        terms = [*pair_terms, *PAIR_KEYWORDS.get(ccy, [ccy])]
        cleaned = [_gdelt_term(t) for t in dict.fromkeys(terms)]
        cleaned = [t for t in cleaned if t]
        if cleaned:
            queries.append((ccy, "(" + " OR ".join(cleaned) + ")"))

    if not queries:
        fallback = [_gdelt_term(t) for t in pair_terms if _gdelt_term(t)]
        queries.append((clean_pair or "PAIR", "(" + " OR ".join(fallback) + ")"))
    return queries


def _dedupe_news_rows(rows: list[dict]) -> list[dict]:
    """Deduplicate rows gathered from split GDELT queries before appending."""
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for row in rows:
        key = (
            str(row.get("timestamp_utc", "")),
            str(row.get("headline", "")),
            str(row.get("url", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


# ── Timestamp coercion ----------------------------------------------------------------────────


def _coerce_timestamp(value: str) -> datetime | None:
    raw = str(value).strip()
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return ts.astimezone(UTC) if ts.tzinfo else ts.replace(tzinfo=UTC)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[: len(fmt)], fmt).replace(tzinfo=UTC)
        except Exception:
            continue
    return None


def _coerce_feed_timestamp(value: str) -> datetime | None:
    """Parse RSS/Atom timestamps, including RFC-822 pubDate strings."""
    ts = _coerce_timestamp(value)
    if ts is not None:
        return ts
    try:
        parsed = email.utils.parsedate_to_datetime(str(value).strip())
        if parsed is None:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except Exception:
        return None


def _xml_text(node: ET.Element, names: tuple[str, ...]) -> str:
    """Return first non-empty child text, namespace agnostic."""
    for child in list(node):
        local = child.tag.rsplit("}", 1)[-1].lower()
        if local in names:
            text = "".join(child.itertext()).strip()
            if text:
                return text
    return ""


def _xml_link(node: ET.Element) -> str:
    """Return RSS/Atom link text/href, namespace agnostic."""
    for child in list(node):
        local = child.tag.rsplit("}", 1)[-1].lower()
        if local != "link":
            continue
        href = str(child.attrib.get("href", "")).strip()
        if href:
            return href
        text = "".join(child.itertext()).strip()
        if text:
            return text
    return ""


def _parse_official_feed_items(raw_xml: str, feed_name: str, currency: str, url: str) -> list[dict]:
    """Parse RSS or Atom XML into normalized historical-news rows."""
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as exc:
        raise ValueError(f"invalid XML from {feed_name}: {exc}") from exc

    nodes = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1].lower() in {"item", "entry"}]
    rows: list[dict] = []
    for node in nodes:
        title = _xml_text(node, ("title",))
        date_raw = _xml_text(node, ("pubdate", "published", "updated", "date"))
        ts = _coerce_feed_timestamp(date_raw)
        if not title or ts is None:
            continue
        category = _event_category_from_text(title)
        rows.append(
            {
                "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
                "event_type": "official_feed",
                "currency": currency,
                "impact": _impact_from_category(category, "high"),
                "headline": title,
                "actual": "",
                "forecast": "",
                "source": f"official_{feed_name.lower()}",
                "url": _xml_link(node) or url,
                "event_category": category,
            }
        )
    return rows


def fetch_official_feeds(
    start: datetime,
    end: datetime,
    *,
    pairs: list[str],
    feeds: dict[str, dict[str, str]] | None = None,
    timeout: int = 30,
    dry_run: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Fetch free official central-bank RSS/Atom feeds.

    These feeds mostly expose recent official headlines, not full multi-year
    archives. They still add high-quality policy events to the same CSV schema
    used by GDELT and ForexFactory.
    """
    wanted = set()
    for pair in pairs:
        wanted.update(_pair_currencies(pair))

    rows: list[dict] = []
    failures: list[dict] = []
    feed_defs = feeds or OFFICIAL_FEEDS
    for name, cfg in feed_defs.items():
        currency = str(cfg.get("currency", "")).upper()
        url = str(cfg.get("url", "")).strip()
        if not url or (wanted and currency not in wanted):
            continue
        if dry_run:
            print(f"  [DRY-RUN] official feed {name} {currency}: {url}", flush=True)
            continue
        try:
            payload = _SESSION.get(
                url,
                headers={"User-Agent": "forex-scaling-model/2.0"},
                timeout=timeout,
            )
            payload.raise_for_status()
            feed_rows = _parse_official_feed_items(payload.text, name, currency, url)
            feed_rows = [row for row in feed_rows if start <= (_coerce_timestamp(row["timestamp_utc"]) or start) <= end]
            rows.extend(feed_rows)
            print(f"  [Official] {name:<4} {currency} | {len(feed_rows)} rows", flush=True)
        except Exception as exc:
            print(f"  [Official] WARN {name} failed: {exc}", flush=True)
            failures.append(
                {
                    "source": f"official_{name.lower()}",
                    "pair": ",".join(pairs),
                    "start_utc": start.isoformat().replace("+00:00", "Z"),
                    "end_utc": end.isoformat().replace("+00:00", "Z"),
                    "reason": str(exc),
                }
            )
    return _dedupe_news_rows(rows), failures


# ── Thread-safe incremental CSV writer ────────────────────────────────────────


class _IncrementalCSVWriter:
    """
    Appends new rows to a CSV file atomically (thread-safe).
    Deduplication runs only once at the final flush_dedup() call, keeping the
    hot path (per-chunk append) cheap -- no full-file rewrite each time.

    The file existence check is done on every append (not cached) so the
    writer correctly re-creates the header if the output file is deleted
    between checkpoints.
    """

    def __init__(self, path: Path, fields: list[str]) -> None:
        self.path = path
        self.fields = fields
        self._lock = threading.Lock()
        # DS-008: Use a set of hashes instead of full tuples to bound memory.
        # Each entry is a 16-byte hash instead of a multi-string tuple.
        self._seen_hashes: set = set()
        if self.path.exists() and self.path.stat().st_size > 0:
            self._load_existing_hashes()

    def _row_hash(self, row_tuple: tuple) -> bytes:
        """Compact hash of a row tuple for memory-efficient deduplication."""
        import hashlib

        return hashlib.md5("|".join(str(v) for v in row_tuple).encode()).digest()

    def _load_existing_hashes(self) -> None:
        """Load hashes of existing rows in batches to avoid OOM."""
        import csv as _csv

        try:
            with self.path.open("r", encoding="utf-8", newline="") as fh:
                reader = _csv.DictReader(fh)
                batch_size = 100_000
                batch = []
                for row in reader:
                    key = tuple(row.get(f, "") for f in self.fields)
                    batch.append(self._row_hash(key))
                    if len(batch) >= batch_size:
                        self._seen_hashes.update(batch)
                        batch.clear()
                if batch:
                    self._seen_hashes.update(batch)
        except Exception:
            self._seen_hashes.clear()

    def _ensure_header(self) -> None:
        """Create the file with a header row if it doesn't exist yet."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.fields).writeheader()
        except FileExistsError:
            pass

    def append(self, rows: list[dict]) -> None:
        """Thread-safe incremental append of new rows to the CSV."""
        if not rows:
            return
        normalized = [{f: r.get(f, "") for f in self.fields} for r in rows]

        pre_filtered = []
        for r in normalized:
            key = tuple(r[f] for f in self.fields)
            h = self._row_hash(key)
            if h not in self._seen_hashes:
                pre_filtered.append((h, r))

        if not pre_filtered:
            return

        final_rows = []
        with self._lock:
            for h, r in pre_filtered:
                if h not in self._seen_hashes:
                    self._seen_hashes.add(h)
                    final_rows.append(r)

            if not final_rows:
                return

            if not self.path.exists():
                self._ensure_header()
            with self.path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.fields)
                writer.writerows(final_rows)

    def flush_dedup(self) -> int:
        """
        Final pass: read the whole file, deduplicate, sort, rewrite cleanly.
        Called once at the very end -- not on every checkpoint.
        """
        with self._lock:
            existing = _read_existing(self.path)
        seen: set = set()
        deduped: list[dict] = []
        for row in existing:
            normalized = {f: row.get(f, "") for f in self.fields}
            key = (
                normalized["timestamp_utc"],
                normalized["event_type"],
                normalized["currency"],
                normalized["headline"],
                normalized["source"],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        deduped.sort(key=lambda r: (r["timestamp_utc"], r["event_type"], r["currency"], r["headline"]))
        with self._lock, self.path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(deduped)
        return len(deduped)


# ── Per-(pair, date) progress manifest for accurate resume ────────────────────


class _ProgressManifest:
    """
    Thread-safe record of which (pair, date) GDELT chunks have been fetched AND
    persisted to the news CSV.

    Why this exists:
        The news CSV stores `currency` (parsed from headline text), not the pair
        that fetched each row, so it is impossible to tell from the CSV alone
        which pairs already covered a given date. The previous resume logic
        skipped a date for *all* pairs as soon as *any* pair wrote a row for it,
        silently dropping coverage for the other pairs. This manifest keys on
        (pair, date) so resume is exact.

    A (pair, date) entry is only written *after* the corresponding rows have been
    flushed to disk, so a crash never leaves a date marked done with its rows
    lost (it would simply be re-fetched and de-duplicated on the next run).
    """

    FIELDS = ["pair", "date"]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._done: set[tuple[str, str]] = set()
        if path.exists():
            try:
                with path.open("r", encoding="utf-8", newline="") as fh:
                    for row in csv.DictReader(fh):
                        p = str(row.get("pair", "")).strip()
                        d = str(row.get("date", "")).strip()
                        if p and d:
                            self._done.add((p, d))
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._done)

    def is_done(self, pair: str, dates: set[str]) -> bool:
        if not dates:
            return False
        with self._lock:
            return all((pair, d) in self._done for d in dates)

    def mark(self, pair: str, dates: set[str]) -> None:
        if not dates:
            return
        with self._lock:
            new_rows = []
            for d in sorted(dates):
                key = (pair, d)
                if key not in self._done:
                    self._done.add(key)
                    new_rows.append({"pair": pair, "date": d})
            if not new_rows:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not (self.path.exists() and self.path.stat().st_size > 0)
            with self.path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerows(new_rows)


# ── Legacy write helpers (for calendar / failures, which are less frequent) ──


def _read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_existing_days(path: Path, *, source: str) -> set[tuple[str, str]]:
    days: set[tuple[str, str]] = set()
    for row in _read_existing(path):
        if str(row.get("source", "")).strip() != source:
            continue
        pair = str(row.get("currency", "")).strip()
        ts = _coerce_timestamp(str(row.get("timestamp_utc", "")))
        if ts is not None:
            days.add((pair, ts.date().isoformat()))
    return days


def write_rows(path: Path, rows: list[dict], *, append: bool, fields: list[str] | None = None) -> int:
    fields = fields or CSV_FIELDS
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = (_read_existing(path) if append else []) + rows
    seen: set = set()
    deduped: list[dict] = []
    for row in all_rows:
        normalized = {field: row.get(field, "") for field in fields}
        key = (
            normalized["timestamp_utc"],
            normalized["event_type"],
            normalized["currency"],
            normalized["headline"],
            normalized["source"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    deduped.sort(key=lambda r: (r["timestamp_utc"], r["event_type"], r["currency"], r["headline"]))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(deduped)
    return len(deduped)


def append_failures(path: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(path)
    seen = {(r.get("source", ""), r.get("pair", ""), r.get("start_utc", ""), r.get("end_utc", "")) for r in existing}
    new_rows: list[dict] = []
    for row in rows:
        normalized = {field: row.get(field, "") for field in FAILURE_FIELDS}
        key = (
            normalized["source"],
            normalized["pair"],
            normalized["start_utc"],
            normalized["end_utc"],
        )
        if key in seen:
            continue
        seen.add(key)
        new_rows.append(normalized)
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (r.get("source", ""), r.get("pair", ""), r.get("start_utc", "")))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return len(new_rows)


def resolve_failures(path: Path, resolved_keys: set[tuple]) -> None:
    """Remove resolved failure entries from the failures CSV."""
    if not path.exists() or not resolved_keys:
        return
    existing = _read_existing(path)
    remaining = [
        r
        for r in existing
        if (r.get("source", ""), r.get("pair", ""), r.get("start_utc", ""), r.get("end_utc", "")) not in resolved_keys
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        writer.writerows(remaining)


# ── GDELT fetcher ----------------------------------------------------------------─────────────


def _gdelt_fetch_one_chunk(
    pair: str,
    chunk_start: datetime,
    chunk_end: datetime,
    *,
    query: str,
    query_label: str = "",
    max_records: int,
    sleep_s: float,
    retries: int,
    split_on_cap: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Fetch a single GDELT chunk.  If the result hits the maxrecords cap AND
    split_on_cap is True, automatically retry with AM / PM sub-windows.
    Returns (rows, failures).
    """

    def _query_window(ws: datetime, we: datetime) -> tuple[list | None, str]:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "startdatetime": _gdelt_datetime(ws),
            "enddatetime": _gdelt_datetime(we),
            "maxrecords": int(max_records),
            "sort": "HybridRel",
        }
        last_error = ""
        for attempt in range(1, max(1, int(retries)) + 1):
            try:
                _GDELT_LIMITER.wait()
                payload = _http_json(GDELT_DOC_URL, params)
                return payload, ""
            except Exception as exc:
                last_error = str(exc)
                secs = _retry_sleep_seconds(exc, sleep_s, attempt)
                print(
                    f"  [GDELT] WARN {pair} {ws.date()} attempt {attempt}/{retries}: {exc}; retry in {secs:.1f}s",
                    flush=True,
                )
                time.sleep(secs)
        return None, last_error

    def _articles_from_payload(payload) -> list:
        return payload.get("articles", []) if isinstance(payload, dict) else []

    def _parse_articles(articles: list) -> list[dict]:
        rows: list[dict] = []
        for art in articles:
            title = str(art.get("title", "")).strip()
            seen = str(art.get("seendate", "")).strip()
            if not title or not seen:
                continue
            try:
                ts = datetime.strptime(seen.replace("Z", ""), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
            except Exception:
                continue
            category = _event_category_from_text(title)
            rows.append(
                {
                    "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
                    "event_type": "headline",
                    "currency": _event_currency_from_text(title, pair),
                    "impact": _impact_from_category(category, "medium"),
                    "headline": title,
                    "actual": "",
                    "forecast": "",
                    "source": "gdelt",
                    "url": str(art.get("url", "")),
                    "event_category": category,
                }
            )
        return rows

    def _fetch_window_recursive(ws: datetime, we: datetime, depth: int = 0) -> tuple[list[dict], list[dict]]:
        payload, err = _query_window(ws, we)
        if payload is None:
            failure = {
                "source": "gdelt",
                "pair": pair,
                "start_utc": ws.isoformat().replace("+00:00", "Z"),
                "end_utc": we.isoformat().replace("+00:00", "Z"),
                "reason": f"{query_label}: {err}" if query_label else err,
            }
            return [], [failure]

        articles = _articles_from_payload(payload)
        cap_hit = len(articles) >= int(max_records)

        if depth == 0:
            label = f"{pair}{('/' + query_label) if query_label else ''} {ws.date()}->{we.date()}"
        else:
            label = f"sub {ws.strftime('%Y-%m-%d %H:%M')}->{we.strftime('%H:%M')}"

        if cap_hit and split_on_cap:
            duration = (we - ws).total_seconds()
            if duration <= 900:  # 15 minutes minimum window
                indent = "  " * (depth + 1)
                print(
                    f"{indent}[GDELT] CAP {label} | {len(articles)} articles (window <= 15m, stopping recursion)",
                    flush=True,
                )
                return _parse_articles(articles), []

            mid = ws + (we - ws) / 2
            left_end = mid - timedelta(seconds=1)
            right_start = mid

            indent = "  " * (depth + 1)
            print(
                f"{indent}[GDELT] CAP {label} ({len(articles)} articles) -- recursively splitting",
                flush=True,
            )

            time.sleep(max(0.0, float(sleep_s)))
            left_rows, left_failures = _fetch_window_recursive(ws, left_end, depth + 1)

            time.sleep(max(0.0, float(sleep_s)))
            right_rows, right_failures = _fetch_window_recursive(right_start, we, depth + 1)

            return left_rows + right_rows, left_failures + right_failures

        indent = "  " * (depth + 1) if depth > 0 else "  "
        note = " (cap hit - use --gdelt-split-on-cap)" if cap_hit and not split_on_cap else ""
        print(f"{indent}[GDELT] {label} | {len(articles)} articles{note}", flush=True)
        return _parse_articles(articles), []

    return _fetch_window_recursive(chunk_start, chunk_end, 0)


def fetch_gdelt_pair(
    pair: str,
    start: datetime,
    end: datetime,
    *,
    step_days: int,
    max_records: int,
    sleep_s: float,
    retries: int,
    split_on_cap: bool,
    manifest: _ProgressManifest | None,
    resume: bool,
    checkpoint_every: int,
    writer: _IncrementalCSVWriter | None,
    dry_run: bool = False,
) -> tuple[list[dict], list[dict], set[str]]:
    """
    Fetch all GDELT chunks for one pair.

    Returns (leftover_rows, failures, pending_dates) where pending_dates are
    dates whose rows are in `leftover_rows` and must be marked in the manifest
    by the caller *after* it flushes those rows.
    """
    rows: list[dict] = []
    failures: list[dict] = []
    pending_dates: set[str] = set()
    query_specs = _gdelt_queries(pair)

    chunks = list(_daterange(start, end, step_days))
    total_chunks = len(chunks)
    skip_count = 0
    fetch_count = 0
    consecutive_failures = 0

    def _process_chunk(chunk_start, chunk_end, i):
        if resume and manifest is not None:
            chunk_days = _chunk_dates(chunk_start, chunk_end)
            if manifest.is_done(pair, chunk_days):
                if (i + 1) % 100 == 0:
                    print(
                        f"  [GDELT Skip] {pair}: skipped {i + 1}/{total_chunks} chunks ({(i + 1) / total_chunks * 100:.1f}%)",
                        flush=True,
                    )
                return "skip", None, None

        if dry_run:
            print(
                f"  [DRY-RUN] GDELT {pair} {chunk_start.date()}->{chunk_end.date()} ({i + 1}/{total_chunks} - {(i + 1) / total_chunks * 100:.1f}%)",
                flush=True,
            )
            return "dry", None, None

        c_rows: list[dict] = []
        c_fails: list[dict] = []
        for query_label, query in query_specs:
            q_rows, q_fails = _gdelt_fetch_one_chunk(
                pair,
                chunk_start,
                chunk_end,
                query=query,
                query_label=query_label,
                max_records=max_records,
                sleep_s=sleep_s,
                retries=retries,
                split_on_cap=split_on_cap,
            )
            c_rows.extend(q_rows)
            c_fails.extend(q_fails)
        return "ok", _dedupe_news_rows(c_rows), c_fails

    # Use 1 worker per pair for chunk concurrency to prevent 429 Too Many Requests
    with ThreadPoolExecutor(max_workers=1) as pool:
        future_to_chunk = {pool.submit(_process_chunk, cs, ce, i): (i, cs, ce) for i, (cs, ce) in enumerate(chunks)}

        for future in as_completed(future_to_chunk):
            _i, cs, ce = future_to_chunk[future]
            try:
                status, chunk_rows, chunk_failures = future.result()
                if status == "skip":
                    skip_count += 1
                    continue
                if status == "dry":
                    fetch_count += 1
                    continue

                rows.extend(chunk_rows)
                failures.extend(chunk_failures)
                fetch_count += 1

                # A chunk counts as "done" (safe to skip on resume) only when it
                # was queried successfully. Failed chunks are left unmarked so a
                # future run re-fetches them.
                if not chunk_failures:
                    pending_dates |= _chunk_dates(cs, ce)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 50:
                        raise RuntimeError(
                            f"Aborting GDELT scraper: 50 consecutive chunks failed for {pair}. Check rate limits or start dates."
                        )

                if fetch_count % 5 == 0 or fetch_count == total_chunks:
                    progress_pct = fetch_count / total_chunks * 100
                    print(
                        f"  [GDELT Progress] {pair}: {fetch_count}/{total_chunks} chunks ({progress_pct:.1f}%)",
                        flush=True,
                    )
                    if wandb and wandb.run:
                        wandb.log(
                            {
                                f"gdelt/{pair}/progress_pct": progress_pct,
                                f"gdelt/{pair}/chunks_fetched": fetch_count,
                                f"gdelt/{pair}/consecutive_failures": consecutive_failures,
                                f"gdelt/{pair}/in_flight_articles": len(rows),
                            }
                        )

                if writer and checkpoint_every > 0 and fetch_count % checkpoint_every == 0:
                    if rows:
                        writer.append(rows)
                        print(f"  [GDELT] checkpoint: flushed {len(rows)} rows for {pair}", flush=True)
                        rows = []
                    # Rows are now on disk -> safe to record progress.
                    if manifest is not None and pending_dates:
                        manifest.mark(pair, pending_dates)
                        pending_dates = set()

            except Exception as exc:
                print(f"  [GDELT] Error processing chunk {cs.date()}->{ce.date()} for {pair}: {exc}", flush=True)

    if skip_count:
        print(f"[GDELT] {pair}: skipped {skip_count} already-covered chunks", flush=True)

    # caller must writer.append(rows) then manifest.mark(pair, pending_dates)
    return rows, failures, pending_dates


# ── EODHD news fetcher ----------------------------------------------------------------────────


def fetch_eodhd_news(
    pair: str,
    start: datetime,
    end: datetime,
    *,
    api_key: str,
    limit: int,
    sleep_s: float,
    dry_run: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    symbol = f"{pair.replace('/', '').replace('_', '').upper()}.FOREX"
    while True:
        if dry_run:
            print(f"  [DRY-RUN] EODHD news {pair} offset={offset}", flush=True)
            break
        params = {
            "api_token": api_key,
            "fmt": "json",
            "s": symbol,
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "limit": int(limit),
            "offset": offset,
        }
        try:
            payload = _http_json(EODHD_NEWS_URL, params)
        except Exception as exc:
            print(f"  [EODHD news] WARN {pair} offset={offset}: {exc}", flush=True)
            break
        articles = (
            payload if isinstance(payload, list) else payload.get("data", []) if isinstance(payload, dict) else []
        )
        print(f"  [EODHD news] {pair} offset={offset} | {len(articles)} articles", flush=True)
        if not articles:
            break
        for art in articles:
            title = str(art.get("title", art.get("headline", ""))).strip()
            date_raw = str(art.get("date", art.get("published_at", art.get("datetime", "")))).strip()
            if not title or not date_raw:
                continue
            ts = _coerce_timestamp(date_raw)
            if ts is None:
                continue
            category = _event_category_from_text(title)
            rows.append(
                {
                    "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
                    "event_type": "headline",
                    "currency": _event_currency_from_text(title, pair),
                    "impact": _impact_from_category(category, "medium"),
                    "headline": title,
                    "actual": "",
                    "forecast": "",
                    "source": "eodhd_news",
                    "url": str(art.get("link", art.get("url", ""))),
                    "event_category": category,
                }
            )
        if len(articles) < int(limit):
            break
        offset += int(limit)
        time.sleep(max(0.0, float(sleep_s)))
    return rows


# ── EODHD calendar fetcher ----------------------------------------------------------------───


def _calendar_year_chunks(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    """Yield (chunk_start, chunk_end) bounded to calendar years.

    A single multi-year from/to request is truncated by the API (~1000 rows), so
    we split the span year-by-year and paginate within each chunk.
    """
    cur = start
    while cur <= end:
        year_end = cur.replace(month=12, day=31, hour=23, minute=59, second=59)
        chunk_end = min(year_end, end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(seconds=1)


def _parse_calendar_events(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data", payload.get("events", []))
        return data if isinstance(data, list) else []
    return []


def _fetch_calendar_chunk(
    url: str,
    chunk_start: datetime,
    chunk_end: datetime,
    *,
    api_key: str,
    sleep_s: float,
    retries: int,
    limit: int,
) -> list[dict]:
    """Fetch one URL/date-chunk with pagination + retry/backoff. Returns events."""
    events: list[dict] = []
    offset = 0
    while True:
        params = {
            "api_token": api_key,
            "fmt": "json",
            "from": chunk_start.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "limit": int(limit),
            "offset": int(offset),
        }
        payload = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                payload = _http_json(url, params)
                break
            except Exception as exc:
                if attempt >= retries:
                    print(
                        f"  [EODHD calendar] WARN {url} [{chunk_start:%Y-%m-%d}..{chunk_end:%Y-%m-%d}] off={offset}: {exc}",
                        flush=True,
                    )
                    payload = None
                    break
                secs = _retry_sleep_seconds(exc, sleep_s, attempt)
                time.sleep(secs)
        page = _parse_calendar_events(payload)
        if not page:
            break
        events.extend(page)
        time.sleep(max(0.0, float(sleep_s)))
        if len(page) < int(limit):
            break
        offset += int(limit)
    return events


def fetch_eodhd_calendar(
    start: datetime,
    end: datetime,
    *,
    api_key: str,
    sleep_s: float,
    dry_run: bool = False,
    retries: int = 4,
    limit: int = 1000,
) -> list[dict]:
    if dry_run:
        print(f"  [DRY-RUN] EODHD calendar {EODHD_ECONOMIC_URLS}", flush=True)
        return []

    rows: list[dict] = []
    for chunk_start, chunk_end in _calendar_year_chunks(start, end):
        # Try each endpoint; only accept the first that yields events for this
        # chunk. An HTTP 200 with an empty list must fall through to the next URL
        # rather than ending the search prematurely.
        events: list[dict] = []
        for url in EODHD_ECONOMIC_URLS:
            events = _fetch_calendar_chunk(
                url,
                chunk_start,
                chunk_end,
                api_key=api_key,
                sleep_s=sleep_s,
                retries=retries,
                limit=limit,
            )
            if events:
                break
        print(f"  [EODHD calendar] {chunk_start:%Y-%m-%d}..{chunk_end:%Y-%m-%d} | {len(events)} events", flush=True)
        for event in events:
            title = str(event.get("event", event.get("title", event.get("name", "")))).strip()
            date_raw = str(event.get("date", event.get("datetime", event.get("timestamp", "")))).strip()
            ts = _coerce_timestamp(date_raw)
            if ts is None or not title:
                continue
            category = _event_category_from_text(title)
            impact = str(event.get("impact", event.get("importance", ""))).strip().lower() or _impact_from_category(
                category, "medium"
            )
            currency = _calendar_currency(event, title)
            rows.append(
                {
                    "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
                    "event_type": "calendar",
                    "currency": currency,
                    "impact": impact,
                    "headline": title,
                    "actual": event.get("actual", ""),
                    "forecast": event.get("forecast", event.get("estimate", "")),
                    "prior": event.get("previous", event.get("prior", "")),
                    "source": "eodhd_calendar",
                    "url": "",
                    "event_category": category,
                }
            )
    return rows


# ── Retry-failures loader ----------------------------------------------------------------────


def load_failures_to_retry(path: Path) -> list[dict]:
    """Read the failures CSV and return all entries as retry specs."""
    rows = _read_existing(path)
    print(f"[Retry] Found {len(rows)} failed chunk(s) in {path}", flush=True)
    return rows


# ── Argument parsing ----------------------------------------------------------------─────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download GDELT/EODHD historical forex news.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Date range
    p.add_argument("--start", required=False, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=False, help="End date YYYY-MM-DD (default: today)")
    # Pairs & source
    p.add_argument(
        "--pairs",
        nargs="+",
        default=["EURUSD"],
        help="Forex pairs, e.g. EURUSD GBPUSD USDJPY",
    )
    p.add_argument(
        "--source",
        choices=["gdelt", "eodhd", "official", "both", "free"],
        default="gdelt",
        help="News source(s): free=GDELT + official central-bank feeds; both=GDELT + EODHD",
    )
    # Outputs
    p.add_argument("--news-out", default=str(DEFAULT_NEWS_OUT))
    p.add_argument("--calendar-out", default=str(DEFAULT_CAL_OUT))
    p.add_argument(
        "--failures-out",
        default="data/raw/news/historical_news_failures.csv",
    )
    p.add_argument(
        "--gdelt-progress-out",
        default="data/raw/news/gdelt_progress.csv",
        help="Per-(pair, date) progress manifest used for accurate --resume",
    )
    # Modes
    p.add_argument(
        "--append",
        action="store_true",
        help="Merge/dedupe with existing output files",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip GDELT dates that already have GDELT rows in --news-out",
    )
    p.add_argument(
        "--retry-failures",
        action="store_true",
        help="Re-attempt all entries recorded in --failures-out",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chunks that would be fetched without making any HTTP calls",
    )
    # GDELT tuning
    p.add_argument("--gdelt-step-days", type=int, default=1, help="GDELT query window size")
    p.add_argument(
        "--gdelt-max-records",
        type=int,
        default=250,
        help="GDELT max records per request",
    )
    p.add_argument("--gdelt-retries", type=int, default=5, help="Retries per GDELT chunk")
    p.add_argument(
        "--gdelt-min-interval",
        type=float,
        default=10.0,
        help="Minimum seconds between GDELT requests, enforced globally across "
        "all workers (prevents 429 Too Many Requests). Raise if you still "
        "see 429s; lower to go faster at higher 429 risk.",
    )
    p.add_argument(
        "--gdelt-split-on-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-split dense windows that hit the maxrecords cap into AM/PM halves",
    )
    # Checkpoint
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Flush rows to CSV every N successful chunks (1 = every chunk)",
    )
    # EODHD tuning
    p.add_argument("--eodhd-limit", type=int, default=100, help="EODHD news page size")
    p.add_argument("--eodhd-api-key", default="", help="Overrides EODHD_API_KEY env var")
    p.add_argument(
        "--include-eodhd-calendar",
        action="store_true",
        help="Also fetch EODHD economic calendar events",
    )
    p.add_argument(
        "--include-official-feeds",
        action="store_true",
        help="Also fetch free official central-bank RSS/Atom feeds into historical_news_combined.parquet",
    )
    # Parallelism
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel pair-fetching threads (1 = sequential)",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds between API calls",
    )
    # Post-download sentiment pre-warming (Feature 2)
    p.add_argument(
        "--score-sentiment",
        action="store_true",
        help="After the final dedup/flush, run sentiment scoring over the "
        "downloaded headlines to pre-warm the cache "
        "(data/embeddings/sentiment_cache.pkl). Skipped during --dry-run.",
    )
    p.add_argument(
        "--sentiment-workers",
        type=int,
        default=4,
        help="Parallel workers for post-download sentiment scoring (only affects the Ollama backend).",
    )
    p.add_argument(
        "--sentiment-backend",
        default="",
        help="Force a sentiment backend (ollama/finbert/vader). Empty = auto-detect (Ollama -> FinBERT -> VADER).",
    )
    return p.parse_args()


# ── Post-download sentiment pre-warming ───────────────────────────────────────


def run_post_download_sentiment(
    news_path: Path,
    *,
    workers: int = 4,
    backend: str = "",
    batch_size: int = 256,
) -> int:
    """Pre-warm the sentiment cache for headlines in `news_path`.

    Loads the unique headlines from the news CSV and runs them through
    features.finbert_sentiment.SentimentPipeline.score_headlines_batch in
    batches. The pipeline caches every result keyed by MD5(headline) to
    data/embeddings/sentiment_cache.pkl, so this is fully resumable/idempotent:
    a re-run skips already-cached headlines and only scores new ones. The giant
    news CSV is NOT modified - only the pkl cache is warmed.

    Returns the number of unique headlines submitted for scoring.
    """
    if not news_path.exists():
        print(f"[Sentiment] No news file at {news_path}; nothing to score.", flush=True)
        return 0

    # Collect unique, non-empty headlines (dedup keeps the work minimal).
    seen: set[str] = set()
    headlines: list[str] = []
    with news_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            h = str(row.get("headline", "")).strip()
            if h and h not in seen:
                seen.add(h)
                headlines.append(h)

    total = len(headlines)
    if total == 0:
        print("[Sentiment] No headlines found to score.", flush=True)
        return 0

    # Make the project root importable so `features.finbert_sentiment` resolves
    # regardless of the current working directory.
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from features.finbert_sentiment import SentimentPipeline
    except Exception as exc:  # pragma: no cover - import-time/env failure
        print(f"[Sentiment] Could not import SentimentPipeline: {exc}", flush=True)
        return 0

    pipe = SentimentPipeline(
        prefer_backend=(backend.strip() or None),
        use_cache=True,
        max_workers=max(1, int(workers)),
    )
    print(
        f"[Sentiment] Pre-warming cache for {total:,} unique headline(s) "
        f"using backend='{pipe.active_backend()}' workers={workers} "
        f"(resumable; cache hits are skipped).",
        flush=True,
    )

    step = max(1, int(batch_size))
    done = 0
    for i in range(0, total, step):
        batch = headlines[i : i + step]
        pipe.score_headlines_batch(batch)
        done += len(batch)
        if done % (step * 4) == 0 or done == total:
            print(
                f"  [Sentiment] {done:,}/{total:,} ({done / total * 100:.1f}%) headlines processed",
                flush=True,
            )

    # Force a final flush so the warmed cache survives the process exit.
    pipe.flush_cache()
    print(f"[Sentiment] Done - cache warmed for {total:,} unique headline(s).", flush=True)
    return total


# ── Main -------------------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    _GDELT_LIMITER.set_interval(args.gdelt_min_interval)

    if wandb and os.getenv("WANDB_RUN_GROUP"):
        try:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "forex-scaling-model"),
                group=os.environ["WANDB_RUN_GROUP"],
                name="news_scraper",
                job_type="data-download-child",
            )
            print(f"[W&B] News scraper attached to group {os.environ['WANDB_RUN_GROUP']}", flush=True)
        except Exception as e:
            print(f"[W&B] Failed to attach to run group: {e}", flush=True)

    print(
        f"[Downloader] version={SCRIPT_VERSION} source={args.source} "
        f"pairs={args.pairs} workers={args.workers} "
        f"gdelt_min_interval={args.gdelt_min_interval}s "
        f"dry_run={args.dry_run} split_on_cap={args.gdelt_split_on_cap}",
        flush=True,
    )

    # ── Resolve date range ----------------------------------------------------------------────
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    if args.retry_failures:
        # --start/--end not required when retrying
        start_dt = end_dt = None
    else:
        if not args.start:
            raise SystemExit("--start is required (or use --retry-failures)")
        start_dt = _parse_date(args.start)
        end_dt = _parse_date(args.end or today)
        if end_dt < start_dt:
            raise SystemExit("--end must be >= --start")
    requested_start_dt = start_dt
    requested_end_dt = end_dt

    sources = (
        {"gdelt"}
        if args.source == "gdelt"
        else {"eodhd"}
        if args.source == "eodhd"
        else {"official"}
        if args.source == "official"
        else {"gdelt", "official"}
        if args.source == "free"
        else {"gdelt", "eodhd"}
    )
    if args.include_official_feeds:
        sources.add("official")

    gdelt_start_dt = start_dt
    if "gdelt" in sources and start_dt is not None:
        gdelt_min_date = datetime(2017, 2, 15, tzinfo=UTC)
        if end_dt is not None and end_dt < gdelt_min_date:
            print(
                f"[GDELT] No DOC 2.0 coverage before {gdelt_min_date.date()}; "
                f"skipping GDELT for requested range {start_dt.date()} -> {end_dt.date()}. "
                "Training will use calendar/official rows when available and neutral headline features otherwise.",
                flush=True,
            )
            sources.discard("gdelt")
            gdelt_start_dt = None
        elif start_dt < gdelt_min_date:
            print(
                f"[GDELT] Partial coverage: requested start {start_dt.date()} is before "
                f"GDELT DOC 2.0 start {gdelt_min_date.date()}; fetching GDELT from "
                f"{gdelt_min_date.date()} onward and leaving earlier headline sentiment neutral.",
                flush=True,
            )
            gdelt_start_dt = gdelt_min_date

    news_out = Path(args.news_out)
    failures_path = Path(args.failures_out)

    # ── Shared incremental writer ──────────────────────────────────────────────
    writer = _IncrementalCSVWriter(news_out, CSV_FIELDS)

    # Pre-seed with existing rows if appending
    if (args.append or args.resume) and news_out.exists():
        print(
            "[News] Existing file detected - incremental append mode",
            flush=True,
        )
        writer._ensure_header()  # don't overwrite

    # ── Retry-failures mode ----------------------------------------------------------------───
    if args.retry_failures:
        if not failures_path.exists():
            print(f"[Retry] No failures file at {failures_path}", flush=True)
            return 0

        failure_entries = load_failures_to_retry(failures_path)
        if not failure_entries:
            print("[Retry] Failures file is empty.", flush=True)
            return 0

        resolved_keys: set[tuple] = set()
        still_failing: list[dict] = []

        for entry in failure_entries:
            src = entry.get("source", "gdelt")
            pair = entry.get("pair", "EURUSD")
            s = _coerce_timestamp(entry.get("start_utc", ""))
            e = _coerce_timestamp(entry.get("end_utc", ""))
            if s is None or e is None:
                continue

            print(f"[Retry] {src} {pair} {s.date()}->{e.date()}", flush=True)

            if src == "gdelt":
                chunk_rows: list[dict] = []
                chunk_failures: list[dict] = []
                for query_label, query in _gdelt_queries(pair):
                    q_rows, q_failures = _gdelt_fetch_one_chunk(
                        pair,
                        s,
                        e,
                        query=query,
                        query_label=query_label,
                        max_records=args.gdelt_max_records,
                        sleep_s=args.sleep,
                        retries=args.gdelt_retries,
                        split_on_cap=args.gdelt_split_on_cap,
                    )
                    chunk_rows.extend(q_rows)
                    chunk_failures.extend(q_failures)
                chunk_rows = _dedupe_news_rows(chunk_rows)
                if chunk_rows:
                    writer.append(chunk_rows)
                if not chunk_failures:
                    resolved_keys.add((src, pair, entry.get("start_utc", ""), entry.get("end_utc", "")))
                still_failing.extend(chunk_failures)
            time.sleep(max(0.0, float(args.sleep)))

        # Remove resolved entries from failures file
        if resolved_keys:
            resolve_failures(failures_path, resolved_keys)
            print(
                f"[Retry] Resolved {len(resolved_keys)} chunk(s); {len(still_failing)} still failing",
                flush=True,
            )

        final_count = writer.flush_dedup()
        print(f"[News] Final deduplicated rows: {final_count:,} -> {news_out}", flush=True)
        return 0

    # ── Normal download mode ----------------------------------------------------------------──
    cal_rows: list[dict] = []
    all_failures: list[dict] = []
    pair_stats: dict[str, int] = {}  # pair -> row count

    manifest: _ProgressManifest | None = None
    if "gdelt" in sources:
        manifest = _ProgressManifest(Path(args.gdelt_progress_out))
        if args.resume and len(manifest):
            print(
                f"[GDELT] Resume: {len(manifest):,} (pair, date) chunk(s) already "
                f"recorded in {args.gdelt_progress_out}",
                flush=True,
            )
        elif args.resume:
            print(
                f"[GDELT] Resume requested but progress manifest "
                f"{args.gdelt_progress_out} is empty/missing - all chunks will be "
                "fetched (existing rows are still de-duplicated on write).",
                flush=True,
            )

    def _fetch_pair_gdelt(pair: str) -> tuple[str, list[dict], list[dict]]:
        """Worker function for one pair - runs in thread pool."""
        print(f"[GDELT] >> starting {pair}", flush=True)
        rows, failures, pending_dates = fetch_gdelt_pair(
            pair,
            gdelt_start_dt,
            end_dt,
            step_days=args.gdelt_step_days,
            max_records=args.gdelt_max_records,
            sleep_s=args.sleep,
            retries=args.gdelt_retries,
            split_on_cap=args.gdelt_split_on_cap,
            manifest=manifest,
            resume=args.resume,
            checkpoint_every=max(0, int(args.checkpoint_every)),
            writer=writer,
            dry_run=args.dry_run,
        )
        # Flush any remaining rows from this pair, then record their progress.
        if rows and not args.dry_run:
            writer.append(rows)
            print(f"[GDELT] {pair}: final flush of {len(rows)} rows", flush=True)
        if manifest is not None and pending_dates and not args.dry_run:
            manifest.mark(pair, pending_dates)
        return pair, rows, failures

    if "gdelt" in sources:
        workers = max(1, int(args.workers))
        pairs = args.pairs

        if args.dry_run:
            print(
                f"[DRY-RUN] Would fetch GDELT for {len(pairs)} pair(s) with {workers} worker(s)",
                flush=True,
            )

        if workers == 1:
            for pair in pairs:
                _, pair_rows, pair_failures = _fetch_pair_gdelt(pair)
                pair_stats[pair] = len(pair_rows)
                all_failures.extend(pair_failures)
        else:
            print(
                f"[GDELT] Parallel mode: {workers} workers for {len(pairs)} pairs",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_fetch_pair_gdelt, pair): pair for pair in pairs}
                for future in as_completed(futures):
                    pair = futures[future]
                    try:
                        _, pair_rows, pair_failures = future.result()
                        pair_stats[pair] = len(pair_rows)
                        all_failures.extend(pair_failures)
                        print(f"[GDELT] OK {pair} done", flush=True)
                    except Exception as exc:
                        print(f"[GDELT] ERR {pair} raised: {exc}", flush=True)
                        raise SystemExit(f"[GDELT] Fatal Error: {exc}")

    if "eodhd" in sources:
        api_key = args.eodhd_api_key or os.getenv("EODHD_API_KEY", "")
        if not api_key:
            print("[EODHD] WARN: EODHD_API_KEY not set; skipping EODHD downloads", flush=True)
        else:
            eodhd_rows: list[dict] = []
            for pair in args.pairs:
                pair_news = fetch_eodhd_news(
                    pair,
                    start_dt,
                    end_dt,
                    api_key=api_key,
                    limit=args.eodhd_limit,
                    sleep_s=args.sleep,
                    dry_run=args.dry_run,
                )
                eodhd_rows.extend(pair_news)
                pair_stats[pair] = pair_stats.get(pair, 0) + len(pair_news)

            if eodhd_rows and not args.dry_run:
                writer.append(eodhd_rows)

            if args.include_eodhd_calendar:
                cal_rows = fetch_eodhd_calendar(
                    start_dt,
                    end_dt,
                    api_key=api_key,
                    sleep_s=args.sleep,
                    dry_run=args.dry_run,
                )

    if "official" in sources:
        official_rows, official_failures = fetch_official_feeds(
            requested_start_dt,
            requested_end_dt,
            pairs=args.pairs,
            dry_run=args.dry_run,
        )
        if official_rows and not args.dry_run:
            writer.append(official_rows)
            print(f"[Official] appended {len(official_rows):,} official central-bank row(s)", flush=True)
        if official_rows:
            pair_stats["OFFICIAL"] = pair_stats.get("OFFICIAL", 0) + len(official_rows)
        all_failures.extend(official_failures)

    # ── Final dedup + sort pass ────────────────────────────────────────────────
    final_count = 0
    if not args.dry_run:
        final_count = writer.flush_dedup()
        print(f"\n[News] Final deduplicated rows: {final_count:,} -> {news_out}", flush=True)
    else:
        print("\n[DRY-RUN] No files written.", flush=True)

    # ── Optional post-download sentiment pre-warming ───────────────────────────
    # Runs only after the final flush and never during --dry-run, so the cache
    # is warmed against the freshly de-duplicated headline set.
    if args.score_sentiment and not args.dry_run:
        run_post_download_sentiment(
            news_out,
            workers=args.sentiment_workers,
            backend=args.sentiment_backend,
        )
    elif args.score_sentiment and args.dry_run:
        print("[Sentiment] --dry-run set; skipping sentiment scoring.", flush=True)

    # ── Calendar ----------------------------------------------------------------──────────────
    if cal_rows and not args.dry_run:
        cal_count = write_rows(Path(args.calendar_out), cal_rows, append=args.append, fields=CAL_FIELDS)
        print(f"[Calendar] Wrote {cal_count:,} rows -> {args.calendar_out}", flush=True)

    # ── Failures ----------------------------------------------------------------──────────────
    if all_failures and not args.dry_run:
        new_fail = append_failures(failures_path, all_failures)
        print(
            f"[Failures] Recorded {new_fail:,} new failed chunk(s) -> {failures_path}",
            flush=True,
        )
        print(
            "           Re-run with --retry-failures to attempt these again.",
            flush=True,
        )

    # --- Per-pair summary -------------------------------------------──────────────────────────
    if pair_stats:
        print("\n--- Per-pair summary -------------------------------------------", flush=True)
        for pair, count in sorted(pair_stats.items()):
            print(f"  {pair:<12} {count:>6} rows (in-flight; see total above)", flush=True)
        print("----------------------------------------------------------------", flush=True)

    gdelt_failures = [f for f in all_failures if str(f.get("source", "")) == "gdelt"]
    if "gdelt" in sources and not args.dry_run and final_count == 0 and gdelt_failures:
        print(
            "[GDELT] ERROR: no news rows were saved and failed chunks were recorded. "
            "Check data/raw/news/historical_news_failures.csv, then retry with "
            "--retry-failures or a higher --gdelt-min-interval.",
            flush=True,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
