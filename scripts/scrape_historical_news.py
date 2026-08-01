"""
Stage resumable historical-news scraping for model training.

This is the friendly entrypoint for long-running news collection. It delegates
the actual source adapters to scripts/download_historical_news.py so the output
continues to match data.historical_news.load_historical_news_bundle:

  data/raw/news/historical_news_combined.parquet
  data/raw/eco_calendar/events.csv

Default source is "free", which uses GDELT DOC 2.0 plus official central-bank
feeds. EODHD can be used when EODHD_API_KEY is available.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


DEFAULT_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
]

GDELT_DOC_START = date(2017, 2, 15)
DEFAULT_NEWS_OUT = "data/raw/news/historical_news_combined.parquet"
DEFAULT_CALENDAR_OUT = "data/raw/eco_calendar/events.csv"
DEFAULT_FAILURES_OUT = "data/raw/news/historical_news_failures.csv"
DEFAULT_PROGRESS_OUT = "data/raw/news/gdelt_progress.csv"


@dataclass(frozen=True)
class Window:
    start: date
    end: date


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_windows(start: date, end: date, days: int) -> list[Window]:
    if end < start:
        raise ValueError("--end must be >= --start")
    step = timedelta(days=max(1, int(days)))
    windows: list[Window] = []
    cur = start
    while cur <= end:
        stop = min(end, cur + step - timedelta(days=1))
        windows.append(Window(cur, stop))
        cur = stop + timedelta(days=1)
    return windows


def build_download_command(args: argparse.Namespace, window: Window) -> list[str]:
    script = Path(__file__).with_name("download_historical_news.py")
    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--start",
        window.start.isoformat(),
        "--end",
        window.end.isoformat(),
        "--pairs",
        *args.pairs,
        "--source",
        args.source,
        "--news-out",
        args.news_out,
        "--calendar-out",
        args.calendar_out,
        "--failures-out",
        args.failures_out,
        "--gdelt-progress-out",
        args.gdelt_progress_out,
        "--workers",
        str(args.workers),
        "--gdelt-step-days",
        str(args.gdelt_step_days),
        "--gdelt-min-interval",
        str(args.gdelt_min_interval),
        "--gdelt-max-records",
        str(args.gdelt_max_records),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--sleep",
        str(args.sleep),
        "--append",
        "--resume",
    ]
    if args.no_split_on_cap:
        cmd.append("--no-gdelt-split-on-cap")
    if args.include_eodhd_calendar:
        cmd.append("--include-eodhd-calendar")
    if args.eodhd_api_key:
        cmd.extend(["--eodhd-api-key", args.eodhd_api_key])
    if args.score_sentiment:
        cmd.extend(
            [
                "--score-sentiment",
                "--sentiment-workers",
                str(args.sentiment_workers),
            ]
        )
        if args.sentiment_backend:
            cmd.extend(["--sentiment-backend", args.sentiment_backend])
    return cmd


def shell_join(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def parse_args() -> argparse.Namespace:
    today = datetime.now(tz=timezone.utc).date().isoformat()
    p = argparse.ArgumentParser(
        description="Scrape historical forex news in resumable training windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", default=GDELT_DOC_START.isoformat())
    p.add_argument("--end", default=today)
    p.add_argument("--window-days", type=int, default=365)
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    p.add_argument(
        "--source",
        choices=["gdelt", "eodhd", "official", "both", "free"],
        default="free",
        help="free = GDELT + official central-bank feeds; both = GDELT + EODHD",
    )
    p.add_argument("--news-out", default=DEFAULT_NEWS_OUT)
    p.add_argument("--calendar-out", default=DEFAULT_CALENDAR_OUT)
    p.add_argument("--failures-out", default=DEFAULT_FAILURES_OUT)
    p.add_argument("--gdelt-progress-out", default=DEFAULT_PROGRESS_OUT)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--gdelt-step-days", type=int, default=1)
    p.add_argument("--gdelt-min-interval", type=float, default=10.0)
    p.add_argument("--gdelt-max-records", type=int, default=250)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--no-split-on-cap", action="store_true")
    p.add_argument("--include-eodhd-calendar", action="store_true")
    p.add_argument("--eodhd-api-key", default="")
    p.add_argument("--score-sentiment", action="store_true")
    p.add_argument("--sentiment-workers", type=int, default=4)
    p.add_argument("--sentiment-backend", default="")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print staged commands without downloading or writing files.",
    )
    p.add_argument(
        "--stop-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop at the first failed window.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start = parse_date(args.start)
    end = parse_date(args.end)
    if start < GDELT_DOC_START and args.source in {"gdelt", "free", "both"}:
        print(
            f"[scraper] GDELT DOC 2.0 starts on {GDELT_DOC_START}; "
            f"clamping start from {start} to {GDELT_DOC_START}.",
            flush=True,
        )
        start = GDELT_DOC_START

    windows = iter_windows(start, end, args.window_days)
    print(
        f"[scraper] {len(windows)} window(s), source={args.source}, "
        f"pairs={','.join(args.pairs)}, out={args.news_out}",
        flush=True,
    )

    failures = 0
    for idx, window in enumerate(windows, start=1):
        cmd = build_download_command(args, window)
        print(
            f"\n[scraper] window {idx}/{len(windows)}: "
            f"{window.start} -> {window.end}",
            flush=True,
        )
        if args.dry_run:
            print(shell_join(cmd), flush=True)
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures += 1
            print(
                f"[scraper] window failed with exit code {result.returncode}: "
                f"{window.start} -> {window.end}",
                flush=True,
            )
            if args.stop_on_failure:
                return result.returncode

    if failures:
        print(f"[scraper] finished with {failures} failed window(s).", flush=True)
        return 1
    print("[scraper] finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
