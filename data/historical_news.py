"""
Historical news and economic-calendar alignment helpers.

This module keeps offline training honest: historical bars only receive news
that was known at or before the bar timestamp. The returned bundle maps cleanly
onto FeatureEngineer.build(...) using Polars DataFrames.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import polars as pl

# Economic-calendar values arrive as unit-suffixed strings (e.g. ForexFactory:
# "148K", "0.3%", "-5.9B"). Plain numeric casts coerce these to null.
_ECO_SUFFIX_SCALE = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_ECO_NUM_RE = re.compile(r"^([+-]?\d*\.?\d+)\s*([KMBT]?)$", re.IGNORECASE)


def _parse_eco_value(value) -> float:
    s = str(value).strip().replace(",", "").replace("%", "")
    if s == "" or s.lower() in {"nan", "none", "n/a"}:
        return np.nan
    m = _ECO_NUM_RE.match(s)
    if not m:
        return np.nan
    num = float(m.group(1))
    suffix = m.group(2).upper()
    return num * _ECO_SUFFIX_SCALE.get(suffix, 1.0)


def _eco_numeric_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all(",", "")
        .str.replace("%", "")
        .map_elements(_parse_eco_value, return_dtype=pl.Float64)
        .alias(col)
    )


_DEFAULT_NEWS_FILE = Path(__file__).resolve().parent / "raw" / "news" / "historical_news_combined.parquet"
_DEFAULT_CAL_FILE = Path(__file__).resolve().parent / "raw" / "eco_calendar" / "events.csv"

_GLOBAL_CURRENCIES = {"", "ALL", "GLOBAL", "G10", "WORLD", "MARKET", "MARKETS"}

NEWS_CATEGORIES = [
    "central_bank",
    "inflation",
    "labor",
    "growth",
    "geopolitical",
    "commentary",
]


@dataclass
class HistoricalNewsBundle:
    """Polars-native news bundle for FeatureEngineer.build(...)."""

    sentiment: Optional[pl.DataFrame] = None
    news_events: Optional[list] = None
    article_counts: Optional[pl.DataFrame] = None
    eco_actual: Optional[pl.DataFrame] = None
    eco_forecast: Optional[pl.DataFrame] = None
    eco_prior: Optional[pl.DataFrame] = None
    finbert_embeddings: Optional[np.ndarray] = None
    category_flags: Optional[pl.DataFrame] = None
    news_events_df: Optional[pl.DataFrame] = None


def empty_news_bundle() -> HistoricalNewsBundle:
    return HistoricalNewsBundle()


def _build_category_flags(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    if (len(df) == 0) or "timestamp_utc" not in df.columns:
        return None
    if "event_category" in df.columns:
        cat = (
            pl.col("event_category")
            .fill_null("")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.to_lowercase()
        )
    else:
        cat = pl.lit("commentary")
    cat = pl.when(cat.is_in(NEWS_CATEGORIES)).then(cat).otherwise(pl.lit("commentary"))
    onehot_exprs = [(cat == name).cast(pl.Float32).alias(f"cat_{name}") for name in NEWS_CATEGORIES]
    flags = (
        df.select(["timestamp_utc", *onehot_exprs])
        .group_by("timestamp_utc")
        .max()
        .sort("timestamp_utc")
    )
    return flags


def _pair_currencies(pair: str) -> set[str]:
    p = str(pair or "").replace("/", "").replace("_", "").upper()
    if len(p) >= 6:
        return {p[:3], p[3:6], "GLOBAL", "ALL"}
    return {p, "GLOBAL", "ALL"} if p else {"GLOBAL", "ALL"}


@lru_cache(maxsize=4)
def _read_table(path: Path, start: str = None, end: str = None) -> pl.DataFrame:
    """PIPE-004: Uses lazy scanning for Parquet files to avoid OOM on large datasets."""
    if not path.exists():
        return pl.DataFrame()
    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return pl.DataFrame(rows) if rows else pl.DataFrame()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("events", payload.get("headlines", payload.get("rows", [])))
        return pl.DataFrame(payload) if payload else pl.DataFrame()
    if path.suffix.lower() == ".parquet":
        # PIPE-004: lazy scan with predicate pushdown instead of eager read
        lf = pl.scan_parquet(path)
        if start and "timestamp_utc" in lf.collect_schema().names():
            lf = lf.filter(pl.col("timestamp_utc") >= start)
        if end and "timestamp_utc" in lf.collect_schema().names():
            lf = lf.filter(pl.col("timestamp_utc") <= end)
        return lf.collect()
    return pl.read_csv(path, try_parse_dates=True, infer_schema_length=10_000)


def _normalise_columns(df: pl.DataFrame) -> pl.DataFrame:
    if (len(df) == 0):
        return df
    aliases = {
        "time": "timestamp_utc",
        "timestamp": "timestamp_utc",
        "datetime": "timestamp_utc",
        "date": "timestamp_utc",
        "title": "headline",
        "name": "headline",
        "event": "headline",
        "importance": "impact",
        "ccy": "currency",
        "curr": "currency",
    }
    rename = {k: v for k, v in aliases.items() if k in df.columns}
    out = df.rename(rename) if rename else df
    out = out.rename({c: str(c).strip().lower() for c in out.columns})
    return out


def _filter_relevant(df: pl.DataFrame, start, end, pair: str) -> pl.DataFrame:
    if (len(df) == 0) or "timestamp_utc" not in df.columns:
        return pl.DataFrame()
    out = df.filter(
        (pl.col("timestamp_utc") >= start) & (pl.col("timestamp_utc") <= end)
    )
    if (len(out) == 0):
        return out
    if "currency" in out.columns:
        keep = _pair_currencies(pair)

        def _cur_ok(raw: str) -> bool:
            parts = {x.strip() for x in str(raw).replace("/", ",").split(",")}
            return bool(parts & keep) or str(raw).strip() in _GLOBAL_CURRENCIES

        mask = out["currency"].fill_null("").cast(pl.Utf8).map_elements(_cur_ok, return_dtype=pl.Boolean)
        out = out.filter(mask)
    return out.sort("timestamp_utc")


import duckdb
def _load_events(news_file: Optional[str], calendar_file: Optional[str], start_ts=None, end_ts=None) -> pl.DataFrame:
    paths = []
    if news_file:
        paths.append(Path(news_file))
    elif os.getenv("HISTORICAL_NEWS_FILE"):
        paths.append(Path(os.getenv("HISTORICAL_NEWS_FILE", "")))
    else:
        paths.append(_DEFAULT_NEWS_FILE)

    augmented_news = Path(__file__).resolve().parent / "raw" / "news" / "historical_news_augmented.csv"
    if augmented_news.exists() and _DEFAULT_NEWS_FILE in paths:
        paths.append(augmented_news)

    if calendar_file:
        paths.append(Path(calendar_file))
    elif os.getenv("ECONOMIC_CALENDAR_FILE"):
        paths.append(Path(os.getenv("ECONOMIC_CALENDAR_FILE", "")))
    else:
        paths.append(_DEFAULT_CAL_FILE)
        
    frames = []
    for p in paths:
        if not p or not p.exists():
            continue
        # If it's a massive CSV file, use DuckDB to slice it!
        if p.suffix.lower() == ".csv":
            try:
                con = duckdb.connect()
                # If we have start and end, slice it directly from disk!
                if start_ts and end_ts:
                    s_str = start_ts.strftime('%Y-%m-%d %H:%M:%S')
                    e_str = end_ts.strftime('%Y-%m-%d %H:%M:%S')
                    query = f"SELECT * FROM read_csv_auto('{str(p)}', ignore_errors=true) WHERE TRY_CAST(timestamp_utc AS TIMESTAMP) >= '{s_str}' AND TRY_CAST(timestamp_utc AS TIMESTAMP) <= '{e_str}'"
                else:
                    query = f"SELECT * FROM read_csv_auto('{str(p)}', ignore_errors=true)"
                df_slice = con.execute(query).pl()
                con.close()
                if not (len(df_slice) == 0):
                    frames.append(_normalise_columns(df_slice))
            except Exception:
                # Fallback to standard read if duckdb fails
                s_str_pass = start_ts.strftime('%Y-%m-%d %H:%M:%S') if start_ts else None
                e_str_pass = end_ts.strftime('%Y-%m-%d %H:%M:%S') if end_ts else None
                frames.append(_normalise_columns(_read_table(p, start=s_str_pass, end=e_str_pass)))
        else:
            s_str_pass = start_ts.strftime('%Y-%m-%d %H:%M:%S') if start_ts else None
            e_str_pass = end_ts.strftime('%Y-%m-%d %H:%M:%S') if end_ts else None
            frames.append(_normalise_columns(_read_table(p, start=s_str_pass, end=e_str_pass)))
            
    frames = [f for f in frames if not (len(f) == 0)]
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames, how="diagonal_relaxed")
    if "timestamp_utc" in df.columns:
        df = df.with_columns(
            pl.col("timestamp_utc").cast(pl.Utf8, strict=False)
            .str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
        ).drop_nulls("timestamp_utc")
    return df


def _eco_pair_frames(eco: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    agg_exprs = [
        pl.col("actual").mean().alias("actual"),
        pl.col("forecast").mean().alias("forecast"),
    ]
    if "prior" in eco.columns:
        agg_exprs.append(pl.col("prior").mean().alias("prior"))
    agg = eco.group_by("timestamp_utc").agg(agg_exprs).sort("timestamp_utc")
    actual = agg.select("timestamp_utc", pl.col("actual").cast(pl.Float32))
    forecast = agg.select("timestamp_utc", pl.col("forecast").cast(pl.Float32))
    if "prior" in agg.columns:
        prior = agg.select("timestamp_utc", pl.col("prior").cast(pl.Float32))
    else:
        prior = pl.DataFrame(schema={"timestamp_utc": pl.Datetime("us", "UTC"), "prior": pl.Float32})
    return actual, forecast, prior


def load_historical_news_bundle(
    start,
    end,
    pair: str,
    *,
    mode: str = "calendar",
    news_file: Optional[str] = None,
    calendar_file: Optional[str] = None,
) -> HistoricalNewsBundle:
    """Return historical news inputs for FeatureEngineer.build(...)."""
    mode = str(mode or "calendar").lower()
    if mode == "off":
        return empty_news_bundle()

    import pandas as pd

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    raw = _load_events(news_file, calendar_file, start_ts=start_ts - pd.Timedelta(days=2), end_ts=end_ts)
    df = _filter_relevant(raw, start_ts - pd.Timedelta(days=2), end_ts, pair)
    if (len(df) == 0):
        return empty_news_bundle()

    category_flags = _build_category_flags(df)

    news_events = []
    if "timestamp_utc" in df.columns:
        if "impact" in df.columns:
            imp = pl.col("impact").fill_null("").cast(pl.Utf8).str.to_lowercase()
            high = (
                imp.is_in(["high", "red", "3", "major"])
                | imp.str.contains("high|major|rate|fomc|cpi|nfp")
            )
            news_events = df.filter(high)["timestamp_utc"].to_list()
        else:
            news_events = df["timestamp_utc"].to_list()

    eco_actual = eco_forecast = eco_prior = None
    if "actual" in df.columns and "forecast" in df.columns:
        eco = df.drop_nulls(["actual", "forecast"]).with_columns([
            _eco_numeric_expr("actual"),
            _eco_numeric_expr("forecast"),
        ])
        if "prior" in eco.columns:
            eco = eco.with_columns(_eco_numeric_expr("prior"))
        eco = eco.drop_nulls(["actual", "forecast"])
        if not (len(eco) == 0):
            eco_actual, eco_forecast, eco_prior = _eco_pair_frames(eco)

    news_events_df = df if mode == "full" else None

    if mode != "full":
        return HistoricalNewsBundle(
            news_events=news_events,
            eco_actual=eco_actual,
            eco_forecast=eco_forecast,
            eco_prior=eco_prior,
            category_flags=category_flags,
            news_events_df=news_events_df,
        )

    # DS-004: removed bag-of-words _sentiment_score fallback which inverts
    # signals for financial contexts (e.g. "weak dollar" = bullish EURUSD).
    # When no pre-computed sentiment is available, use 0.0 (neutral) instead.
    target_col = None
    if "sentiment" in df.columns:
        target_col = "sentiment"
    elif "sentiment_score" in df.columns:
        target_col = "sentiment_score"

    if target_col:
        sent_col = pl.col(target_col).cast(pl.Float64, strict=False).fill_null(0.0)
    else:
        sent_col = pl.lit(0.0)

    sentiment = (
        df.select("timestamp_utc", sent_col.alias("sentiment"))
        .group_by("timestamp_utc")
        .agg(pl.col("sentiment").mean().cast(pl.Float32))
        .sort("timestamp_utc")
    )
    article_counts = (
        df.select("timestamp_utc")
        .group_by("timestamp_utc")
        .len()
        .rename({"len": "article_counts"})
        .with_columns(pl.col("article_counts").cast(pl.Float32))
        .sort("timestamp_utc")
    )

    return HistoricalNewsBundle(
        sentiment=sentiment,
        news_events=news_events,
        article_counts=article_counts,
        eco_actual=eco_actual,
        eco_forecast=eco_forecast,
        eco_prior=eco_prior,
        category_flags=category_flags,
        news_events_df=news_events_df,
    )


def collect_headlines_for_range(
    start: str,
    end: str,
    pairs: Iterable[str] = ("EURUSD", "GBPUSD", "USDJPY"),
    *,
    news_file: Optional[str] = None,
    calendar_file: Optional[str] = None,
) -> list[str]:
    """Return unique headline strings across the full date range.

    Used for upfront FinBERT cache-warming so the per-window loop never
    waits on model inference.
    """
    import pandas as pd

    start_ts = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=2)
    end_ts = pd.Timestamp(end, tz="UTC")

    raw = _load_events(news_file, calendar_file, start_ts=start_ts, end_ts=end_ts)
    if len(raw) == 0 or "headline" not in raw.columns:
        return []

    pair_currencies: set[str] = set()
    for p in pairs:
        pair_currencies |= _pair_currencies(p)

    if "currency" in raw.columns:
        keep = pair_currencies

        def _cur_ok(val: str) -> bool:
            parts = {x.strip() for x in str(val).replace("/", ",").split(",")}
            return bool(parts & keep) or str(val).strip() in _GLOBAL_CURRENCIES

        mask = raw["currency"].fill_null("").cast(pl.Utf8).map_elements(
            _cur_ok, return_dtype=pl.Boolean,
        )
        raw = raw.filter(mask)

    headlines = (
        raw.select(pl.col("headline").fill_null("").cast(pl.Utf8))
        .unique()
        .to_series()
        .to_list()
    )
    return [h for h in headlines if h.strip()]
