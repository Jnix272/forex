from __future__ import annotations

import argparse
from datetime import date

from scripts.scrape_historical_news import (
    Window,
    build_download_command,
    iter_windows,
)


def test_iter_windows_splits_inclusive_ranges():
    windows = iter_windows(date(2024, 1, 1), date(2024, 1, 5), days=2)

    assert windows == [
        Window(date(2024, 1, 1), date(2024, 1, 2)),
        Window(date(2024, 1, 3), date(2024, 1, 4)),
        Window(date(2024, 1, 5), date(2024, 1, 5)),
    ]


def test_build_download_command_is_resumable_and_schema_compatible():
    args = argparse.Namespace(
        pairs=["EURUSD", "GBPUSD"],
        source="free",
        news_out="data/raw/news/historical_news_combined.parquet",
        calendar_out="data/raw/eco_calendar/events.csv",
        failures_out="data/raw/news/historical_news_failures.csv",
        gdelt_progress_out="data/raw/news/gdelt_progress.csv",
        workers=2,
        gdelt_step_days=7,
        gdelt_min_interval=2.0,
        gdelt_max_records=250,
        checkpoint_every=1,
        sleep=0.5,
        no_split_on_cap=False,
        include_eodhd_calendar=False,
        eodhd_api_key="",
        score_sentiment=False,
        sentiment_workers=4,
        sentiment_backend="",
    )

    cmd = build_download_command(args, Window(date(2024, 1, 1), date(2024, 12, 31)))

    assert "download_historical_news.py" in cmd[2]
    assert "--append" in cmd
    assert "--resume" in cmd
    assert cmd[cmd.index("--source") + 1] == "free"
    assert cmd[cmd.index("--news-out") + 1] == "data/raw/news/historical_news_combined.parquet"
    assert cmd[cmd.index("--pairs") + 1: cmd.index("--source")] == ["EURUSD", "GBPUSD"]
