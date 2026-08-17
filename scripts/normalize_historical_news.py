"""Normalize historical_news_combined.parquet for the training news loader.

BigQuery/GDELT exports often mark rows by pair (EURUSD, USDJPY, GBPUSD), while
data.historical_news filters by ISO currencies. This script maps pair labels to
currency lists, fills blank macro rows, adds the project schema columns, and
deduplicates rows before training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

PAIR_TO_CURRENCIES = {
    "EURUSD": "EUR,USD",
    "USDJPY": "USD,JPY",
    "GBPUSD": "GBP,USD",
}


def _headline() -> pl.Expr:
    return pl.col("headline").fill_null("").cast(pl.Utf8).str.to_lowercase()


def _pair_currency_expr() -> pl.Expr:
    cur = pl.col("currency").fill_null("").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    h = _headline()
    return (
        pl.when(cur == "EURUSD")
        .then(pl.lit("EUR,USD"))
        .when(cur == "USDJPY")
        .then(pl.lit("USD,JPY"))
        .when(cur == "GBPUSD")
        .then(pl.lit("GBP,USD"))
        .when(cur.is_in(["EUR", "USD", "JPY", "GBP", "GLOBAL", "ALL"]))
        .then(cur)
        .when(h.str.contains("european central bank|\\becb\\b|\\beuro\\b|eur/usd|eurusd"))
        .then(pl.lit("EUR,USD"))
        .when(h.str.contains("bank of japan|\\bboj\\b|\\byen\\b|usd/jpy|usdjpy"))
        .then(pl.lit("USD,JPY"))
        .when(h.str.contains("bank of england|\\bboe\\b|sterling|\\bpound\\b|gbp/usd|gbpusd"))
        .then(pl.lit("GBP,USD"))
        .when(h.str.contains("federal reserve|\\bfomc\\b|nonfarm payrolls|\\bnfp\\b|\\bcpi\\b|inflation"))
        .then(pl.lit("USD"))
        .otherwise(pl.lit("GLOBAL"))
        .alias("currency")
    )


def _impact_expr() -> pl.Expr:
    h = _headline()
    return (
        pl.when(h.str.contains("\\bfomc\\b|nonfarm payrolls|\\bnfp\\b|\\bcpi\\b|rate decision|central bank"))
        .then(pl.lit("High"))
        .otherwise(pl.lit("Medium"))
        .alias("impact")
    )


def _category_expr() -> pl.Expr:
    h = _headline()
    return (
        pl.when(h.str.contains("central bank|federal reserve|\\bfomc\\b|\\becb\\b|\\bboj\\b|\\bboe\\b|rate decision"))
        .then(pl.lit("central_bank"))
        .when(h.str.contains("inflation|\\bcpi\\b|consumer price|ppi"))
        .then(pl.lit("inflation"))
        .when(h.str.contains("nonfarm payrolls|\\bnfp\\b|jobs|employment|unemployment|wages"))
        .then(pl.lit("labor"))
        .when(h.str.contains("gdp|growth|recession|pmi|retail sales"))
        .then(pl.lit("growth"))
        .when(h.str.contains("war|conflict|sanction|geopolitical|election"))
        .then(pl.lit("geopolitical"))
        .otherwise(pl.lit("commentary"))
        .alias("event_category")
    )


def normalize_news(input_path: Path, output_path: Path) -> None:
    if input_path.suffix == ".parquet":
        df = pl.read_parquet(input_path)
    else:
        df = pl.read_csv(input_path, infer_schema_length=10_000)
    required = ["timestamp_utc", "event_type", "currency", "headline", "url", "source"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {missing}")

    out = (
        df.with_columns(
            [
                pl.col("timestamp_utc").cast(pl.Utf8).str.strip_chars(),
                pl.col("event_type").fill_null("headline").cast(pl.Utf8),
                pl.col("headline").fill_null("").cast(pl.Utf8),
                pl.col("url").fill_null("").cast(pl.Utf8),
                pl.col("source").fill_null("gdelt_bq").cast(pl.Utf8),
                _pair_currency_expr(),
                _impact_expr(),
                _category_expr(),
                pl.lit("").alias("actual"),
                pl.lit("").alias("forecast"),
            ]
        )
        .select(
            [
                "timestamp_utc",
                "event_type",
                "currency",
                "impact",
                "headline",
                "actual",
                "forecast",
                "source",
                "url",
                "event_category",
            ]
        )
        .unique(subset=["timestamp_utc", "currency", "headline", "url"], keep="first")
        .sort("timestamp_utc")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        out.write_parquet(output_path)
    else:
        out.write_csv(output_path)
    print(f"normalized rows={out.height:,} -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize data/raw/news/historical_news_combined.parquet")
    parser.add_argument("--input", default="data/raw/news/historical_news_combined.parquet")
    parser.add_argument("--output", default="data/raw/news/historical_news.normalized.csv")
    parser.add_argument(
        "--replace", action="store_true", help="Replace input with output and keep a .raw_bq_backup.csv copy"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    normalize_news(input_path, output_path)

    if args.replace:
        backup = input_path.with_suffix(".raw_bq_backup.csv")
        if not backup.exists():
            input_path.replace(backup)
        output_path.replace(input_path)
        print(f"backup={backup}")
        print(f"active={input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
