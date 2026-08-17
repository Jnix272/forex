"""
Scrape the ForexFactory economic calendar into data/raw/eco_calendar/events.csv.

This is the project's REAL economic-calendar source. The output is consumed by
``data.historical_news.load_historical_news_bundle`` (the ``calendar_file`` path)
which feeds the eco_act / eco_fc / eco_surprise features in training.

Hardening (mirrors the news downloader, scripts/download_historical_news.py):
  - Output schema exactly matches the loader, including an extra ``prior``
    column (ForexFactory "Previous" reading) used for eco_surprise scaling.
  - DST-aware timezone handling: ForexFactory renders the logged-out calendar in
    US Eastern wall-clock time. Times are localized in that zone (handles
    EST/EDT automatically) and converted to UTC. Configurable via --tz.
  - All-day / Tentative / multi-day ("Day 1") rows have no clock time; they are
    placed at end-of-day local time (no look-ahead) instead of a wrong midnight.
  - Impact mapped from the row's color class (red=high, orange=medium,
    yellow=low, grey=low/holiday).
  - Currency read directly from the row's currency cell (never guessed).
  - Actual/Forecast/Previous numeric strings ("236K", "4.2%", "1.5M", "1,024")
    are normalized to plain numbers so the loader's numeric coercion keeps them.
  - HTTP fetches wrapped in exponential-backoff retries (429/5xx/network).
  - Polite, jittered rate limiting between weekly pages + realistic User-Agent.
  - --resume skips weeks already present in events.csv; new rows are appended and
    de-duplicated on a stable key (timestamp_utc + currency + headline) rather
    than overwriting the whole file.
  - Robust per-row parsing: a single bad row/week never aborts the run; failed
    weeks are recorded to a failures CSV for re-run.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import random
import re
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

SCRIPT_VERSION = "2026-05-29-forexfactory-hardened"

# ── Output schema -------------------------------------------------------------
# Must match data.historical_news.load_historical_news_bundle. ``prior`` is the
# ForexFactory "Previous" reading; the loader reads it when present (and ignores
# unknown columns), so appending it is backward-compatible. The feature side
# (features/feature_engineering.py) prefers |prior| as the eco_surprise scale.
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

# Stable dedup / resume key: a later revision of Actual/Forecast must NOT create
# a duplicate row, so the key intentionally excludes those mutable fields.
DEDUP_KEY = ("timestamp_utc", "currency", "headline")

DEFAULT_OUT = Path("data/raw/eco_calendar/events.csv")
DEFAULT_FAILURES_OUT = Path("data/raw/eco_calendar/events_failures.csv")
FAILURE_FIELDS = ["week", "url", "reason"]

# ForexFactory's logged-out calendar renders event times in US Eastern wall
# clock (DST-aware). Localize naive parsed times in this zone, then -> UTC.
DEFAULT_FF_TZ = "America/New_York"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}

_SESSION = requests.Session()
_SESSION.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8))


# ── CSV helpers ---------------------------------------------------------------


def _read_existing(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class _CalendarCSVWriter:
    """Append-and-dedup writer for events.csv.

    Pre-loads existing rows so that (a) duplicates across runs are suppressed on
    the stable DEDUP_KEY and (b) already-scraped weeks can be skipped on
    --resume. Each ``append`` call writes only genuinely new rows; the file is
    never fully rewritten, so an interrupted run keeps prior progress.
    """

    def __init__(self, path: Path, fields: list[str]) -> None:
        self.path = path
        self.fields = fields
        self._seen: set[tuple] = set()
        self._weeks_present: set[str] = set()
        for row in _read_existing(path):
            self._seen.add(tuple(row.get(k, "") for k in DEDUP_KEY))
            wk = _week_from_url(row.get("url", ""))
            if wk:
                self._weeks_present.add(wk)

    def has_week(self, week_str: str) -> bool:
        return week_str in self._weeks_present

    def _ensure_header(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.fields).writeheader()
        except FileExistsError:
            pass

    def append(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        new_rows: list[dict] = []
        for r in rows:
            normalized = {f: r.get(f, "") for f in self.fields}
            key = tuple(normalized.get(k, "") for k in DEDUP_KEY)
            if key in self._seen:
                continue
            self._seen.add(key)
            new_rows.append(normalized)
            wk = _week_from_url(normalized.get("url", ""))
            if wk:
                self._weeks_present.add(wk)
        if not new_rows:
            return 0
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._ensure_header()
        with self.path.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.fields).writerows(new_rows)
        return len(new_rows)


def _append_failures(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(path)
    seen = {(r.get("week", ""), r.get("url", "")) for r in existing}
    fresh = [r for r in rows if (r.get("week", ""), r.get("url", "")) not in seen]
    if not fresh:
        return
    write_header = not (path.exists() and path.stat().st_size > 0)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FAILURE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in FAILURE_FIELDS} for r in fresh)


# ── Week-string helpers -------------------------------------------------------


def _week_url(week_str: str) -> str:
    return f"https://www.forexfactory.com/calendar?week={week_str}"


def _week_from_url(url: str) -> str:
    m = re.search(r"[?&]week=([a-zA-Z]{3}\d{1,2}\.\d{4})", str(url or ""))
    return m.group(1).lower() if m else ""


def _get_sunday_of_week(dt: datetime) -> datetime:
    """ForexFactory weeks start on Sunday. Returns the Sunday on/before ``dt``."""
    days_since_sunday = (dt.weekday() + 1) % 7  # weekday(): Mon=0..Sun=6
    return dt - timedelta(days=days_since_sunday)


def _parse_ff_week(week_str: str) -> datetime:
    month = _MONTHS[week_str[:3].lower()]
    day, year = week_str[3:].split(".")
    return datetime(int(year), month, int(day))


def _generate_ff_weeks(start_dt: datetime, end_dt: datetime) -> list[str]:
    """Generate week strings like 'jan1.2018' for each Sunday in [start, end]."""
    weeks: set[str] = set()
    cur = _get_sunday_of_week(start_dt)
    end = _get_sunday_of_week(end_dt)
    while cur <= end:
        weeks.add(f"{cur.strftime('%b').lower()}{cur.day}.{cur.year}")
        cur += timedelta(days=7)
    return sorted(weeks, key=_parse_ff_week)


# ── Timezone-aware timestamp parsing ------------------------------------------

_ZONE_CACHE: dict[str, ZoneInfo] = {}


def _get_zone(tz_name: str) -> ZoneInfo:
    if tz_name not in _ZONE_CACHE:
        _ZONE_CACHE[tz_name] = ZoneInfo(tz_name)
    return _ZONE_CACHE[tz_name]


def _is_specific_time(time_str: str) -> bool:
    t = (time_str or "").strip().lower()
    if not t:
        return False
    if "all day" in t or "tentative" in t or t.startswith("day "):
        return False
    return ("am" in t or "pm" in t) and ":" in t


def _parse_ff_time(date_str: str, time_str: str, week_year: int, week_str: str, tz_name: str) -> datetime | None:
    """Convert a ForexFactory date/time row into a UTC datetime.

    ``date_str`` looks like "Mon Jan 1"; ``time_str`` like "8:30am",
    "All Day", "Tentative" or "Day 1". Naive local times are interpreted in
    ``tz_name`` (DST-aware) and converted to UTC. Rows without a specific time
    are placed at 23:59 local (conservative, no look-ahead) on the correct date.
    Returns None when the date cannot be parsed.
    """
    if not date_str:
        return None
    parts = date_str.split()
    if len(parts) < 3:
        return None
    row_month_key = parts[1][:3].lower()
    if row_month_key not in _MONTHS:
        return None
    try:
        day = int(parts[2])
    except ValueError:
        return None

    # Resolve the calendar year across Dec/Jan week boundaries.
    week_month = week_str[:3].lower()
    year = week_year
    if week_month == "dec" and row_month_key == "jan":
        year += 1
    elif week_month == "jan" and row_month_key == "dec":
        year -= 1

    month = _MONTHS[row_month_key]
    if _is_specific_time(time_str):
        try:
            tm = datetime.strptime(time_str.strip().lower().replace(" ", ""), "%I:%M%p")
        except ValueError:
            return None
        try:
            naive = datetime(year, month, day, tm.hour, tm.minute)
        except ValueError:
            return None
    else:
        try:
            naive = datetime(year, month, day, 23, 59)
        except ValueError:
            return None

    try:
        local = naive.replace(tzinfo=_get_zone(tz_name))
    except ZoneInfoNotFoundError:
        local = naive.replace(tzinfo=UTC)
    return local.astimezone(UTC)


# ── Value cleaning ------------------------------------------------------------

_NUM_RE = re.compile(r"^[<>~]?\s*(-?\d[\d,]*\.?\d*)\s*([kmbt%])?\s*$", re.IGNORECASE)
_MULTIPLIER = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def _clean_numeric(raw: str) -> str:
    """Normalize a ForexFactory figure to a plain number string.

    "236K" -> "236000", "4.2%" -> "4.2", "1.5M" -> "1500000", "1,024" -> "1024".
    Unparseable values are returned stripped (the loader coerces them to NaN).
    Empty/placeholder values return "".
    """
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


# ── HTTP fetch with retries ---------------------------------------------------


def _retry_sleep_seconds(exc: Exception, sleep_s: float, attempt: int) -> float:
    base = max(float(sleep_s), 1.0)
    retry_after = None
    resp = getattr(exc, "response", None)
    if isinstance(exc, requests.exceptions.HTTPError) and resp is not None and resp.status_code == 429:
        try:
            retry_after = float(resp.headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            retry_after = None
    if retry_after is not None and retry_after > 0:
        return retry_after + random.uniform(0.25, 1.0)
    return min(120.0, base * (2 ** max(0, attempt - 1))) + random.uniform(0.25, 1.0)


def _fetch_week_html(url: str, *, sleep_s: float, retries: int, timeout: int = 20) -> str:
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
    }
    last_error = ""
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            resp = _SESSION.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_error = str(exc)
            if attempt >= attempts:
                break
            secs = _retry_sleep_seconds(exc, sleep_s, attempt)
            print(
                f"    [FF] WARN attempt {attempt}/{attempts}: {exc}; retry in {secs:.1f}s",
                flush=True,
            )
            time.sleep(secs)
    raise RuntimeError(f"failed to fetch after {attempts} attempt(s): {last_error}")


# ── Parsing -------------------------------------------------------------------


def _impact_from_cell(impact_td) -> str:
    """Map ForexFactory impact color to high/medium/low."""
    if impact_td is None:
        return "low"
    icon = impact_td.find("span")
    cls = " ".join(icon.get("class", [])) if icon is not None else ""
    title = (icon.get("title", "") if icon is not None else "") or impact_td.get_text(" ")
    blob = f"{cls} {title}".lower()
    if "red" in blob or "high" in blob:
        return "high"
    if "ora" in blob or "orange" in blob or "medium" in blob:
        return "medium"
    # yellow=low, grey/gray=holiday/non-economic -> treated as low impact.
    return "low"


def _cell_text(tr, class_name: str) -> str:
    td = tr.find("td", class_=class_name)
    if td is None:
        return ""
    # Use a space separator + whitespace collapse: FF nests the date as
    # <span class="date">Sun <span>Dec 31</span></span>, which would otherwise
    # concatenate to "SunDec 31" and break date parsing.
    return " ".join(td.get_text(" ", strip=True).split())


def parse_week_html(html: str, week_str: str, *, tz_name: str) -> list[dict]:
    """Parse a ForexFactory weekly calendar page into event rows."""
    url = _week_url(week_str)
    year = int(week_str.split(".")[1])
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="calendar__table")
    if table is None:
        return []

    rows: list[dict] = []
    current_date = ""
    current_time = ""

    for tr in table.find_all("tr", class_="calendar__row"):
        try:
            date_txt = _cell_text(tr, "calendar__date")
            if date_txt:
                current_date = " ".join(date_txt.split())

            time_txt = _cell_text(tr, "calendar__time")
            if time_txt:
                current_time = time_txt

            currency = _cell_text(tr, "calendar__currency")
            if not currency:
                # Spacer / non-event row.
                continue

            impact = _impact_from_cell(tr.find("td", class_="calendar__impact"))
            headline = _cell_text(tr, "calendar__event")

            ts = _parse_ff_time(current_date, current_time, year, week_str, tz_name)
            rows.append(
                {
                    "timestamp_utc": ts.isoformat().replace("+00:00", "Z") if ts else "",
                    "event_type": "calendar",
                    "currency": currency,
                    "impact": impact,
                    "headline": headline,
                    "actual": _clean_numeric(_cell_text(tr, "calendar__actual")),
                    "forecast": _clean_numeric(_cell_text(tr, "calendar__forecast")),
                    "prior": _clean_numeric(_cell_text(tr, "calendar__previous")),
                    "source": "forexfactory",
                    "url": url,
                }
            )
        except Exception as exc:
            print(f"    [FF] WARN skipping malformed row in {week_str}: {exc}", flush=True)
            continue

    return rows


def fetch_forexfactory_week(week_str: str, *, tz_name: str, sleep_s: float, retries: int) -> list[dict]:
    html = _fetch_week_html(_week_url(week_str), sleep_s=sleep_s, retries=retries)
    return parse_week_html(html, week_str, tz_name=tz_name)


# ── CLI -----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape the ForexFactory economic calendar into events.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output events CSV path")
    p.add_argument(
        "--failures-out",
        default=str(DEFAULT_FAILURES_OUT),
        help="CSV recording weeks that failed to scrape (for re-run)",
    )
    p.add_argument(
        "--tz",
        default=DEFAULT_FF_TZ,
        help="IANA timezone ForexFactory renders times in (logged-out default is US Eastern)",
    )
    p.add_argument("--sleep", type=float, default=1.5, help="Base seconds between weekly requests")
    p.add_argument("--retries", type=int, default=5, help="HTTP retries per week (429/5xx/network)")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip weeks already present in the output CSV",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List the weeks that would be scraped without making any HTTP calls",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise SystemExit("--end must be >= --start")

    # Validate the timezone up front so failures are obvious, not silent.
    try:
        _get_zone(args.tz)
    except ZoneInfoNotFoundError as exc:
        raise SystemExit(
            f"Timezone {args.tz!r} not found ({exc}). Install the 'tzdata' package "
            "or pass a valid --tz (e.g. America/New_York)."
        )

    weeks = _generate_ff_weeks(start_dt, end_dt)
    out_path = Path(args.out)
    failures_path = Path(args.failures_out)
    writer = _CalendarCSVWriter(out_path, CSV_FIELDS)

    print(
        f"[ForexFactory] version={SCRIPT_VERSION} weeks={len(weeks)} tz={args.tz} "
        f"resume={args.resume} dry_run={args.dry_run} out={out_path}",
        flush=True,
    )

    if args.dry_run:
        skipped = sum(1 for w in weeks if args.resume and writer.has_week(w))
        for w in weeks:
            tag = "SKIP (present)" if args.resume and writer.has_week(w) else "fetch"
            print(f"  [DRY-RUN] {w:<12} {tag}", flush=True)
        print(f"[DRY-RUN] would fetch {len(weeks) - skipped} week(s), skip {skipped}.", flush=True)
        return 0

    total_new = 0
    failures: list[dict] = []
    for i, week_str in enumerate(weeks, 1):
        if args.resume and writer.has_week(week_str):
            print(f"  [{i}/{len(weeks)}] {week_str} -> skip (already present)", flush=True)
            continue
        try:
            rows = fetch_forexfactory_week(week_str, tz_name=args.tz, sleep_s=args.sleep, retries=args.retries)
            added = writer.append(rows)
            total_new += added
            print(f"  [{i}/{len(weeks)}] {week_str} -> {len(rows)} events (+{added} new)", flush=True)
        except Exception as exc:
            print(f"  [{i}/{len(weeks)}] {week_str} -> ERROR: {exc}", flush=True)
            failures.append({"week": week_str, "url": _week_url(week_str), "reason": str(exc)})

        # Polite, jittered delay between page requests.
        time.sleep(max(0.0, float(args.sleep)) + random.uniform(0.0, 0.5))

    if failures:
        _append_failures(failures_path, failures)
        print(
            f"[ForexFactory] {len(failures)} week(s) failed -> {failures_path} "
            "(re-run with --resume to retry only missing weeks)",
            flush=True,
        )

    print(f"[ForexFactory] Done. {total_new} new row(s) -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
