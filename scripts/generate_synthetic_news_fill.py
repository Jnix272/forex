"""
Generate reviewable synthetic historical-news rows for missing pair coverage.

Default target pairs are the active training set:
  EURUSD, GBPJPY, USDJPY

The script does not modify the combined Parquet by default. It writes a CSV that
can be inspected, then included in a later merge if desired.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_NEWS_FILE = "data/raw/news/historical_news_combined.parquet"
DEFAULT_OUT = "data/raw/news/historical_news_synthetic_fill.csv"
DEFAULT_PAIRS = ["EURUSD", "GBPJPY", "USDJPY"]

# From the verified GBPJPY coverage gap in historical_news_combined.parquet.
DEFAULT_MISSING_MONTHS = {
    "GBPJPY": [
        "2010-06",
        "2010-08",
        "2010-09",
        "2010-10",
        "2010-11",
        "2010-12",
        "2011-01",
        "2011-02",
        "2011-04",
        "2011-06",
        "2011-08",
        "2011-11",
        "2011-12",
        "2012-01",
        "2012-02",
        "2012-07",
        "2012-08",
        "2012-10",
        "2012-11",
        "2014-03",
    ],
}

EVENTS = [
    ("central_bank", "high", "{ccy} central bank guidance keeps {pair} traders cautious"),
    ("inflation", "high", "{ccy} inflation expectations shift ahead of key data"),
    ("labor", "medium", "{ccy} labor market indicators draw attention from currency desks"),
    ("growth", "medium", "{ccy} growth outlook update influences {pair} positioning"),
    ("geopolitical", "medium", "Risk sentiment drives renewed focus on {pair}"),
    ("commentary", "low", "Analysts note range-bound trading conditions for {pair}"),
    ("central_bank", "medium", "{ccy} rate outlook remains in focus for FX markets"),
    ("growth", "low", "{pair} liquidity steady as macro calendar stays light"),
]

FIELDS = [
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
    "sentiment_score",
]


def pair_currencies(pair: str) -> list[str]:
    clean = pair.upper().replace("/", "").replace("_", "")
    if len(clean) < 6:
        return [clean]
    return [clean[:3], clean[3:6]]


def parse_month(month: str) -> tuple[int, int]:
    try:
        year_s, month_s = month.split("-", 1)
        year = int(year_s)
        month_i = int(month_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid month {month!r}; expected YYYY-MM") from exc
    if month_i < 1 or month_i > 12:
        raise argparse.ArgumentTypeError(f"Invalid month {month!r}; month must be 01..12")
    return year, month_i


def month_days(month: str) -> list[date]:
    year, month_i = parse_month(month)
    last_day = calendar.monthrange(year, month_i)[1]
    days = [date(year, month_i, day) for day in range(1, last_day + 1)]
    business = [day for day in days if day.weekday() < 5]
    return business or days


def stable_pick(items: list, key: str):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return items[int.from_bytes(digest[:4], "big") % len(items)]


def event_timestamp(day: date, pair: str, currency: str, idx: int) -> str:
    hour = stable_pick([7, 8, 9, 10, 13, 14, 15, 16], f"{day}-{pair}-{currency}-{idx}")
    minute = stable_pick([0, 15, 30, 45], f"{pair}-{currency}-{day}-{idx}-minute")
    dt = datetime.combine(day, time(hour, minute), tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def headline_filter_sql(currency: str, pair: str) -> str:
    keywords = {
        "EUR": ["eur", "euro", "ecb", "eurozone", "european central bank"],
        "USD": ["usd", "dollar", "fed", "federal reserve", "fomc", "treasury"],
        "GBP": ["gbp/usd", "gbpjpy", "gbp jpy", "pound sterling", "sterling", "boe", "bank of england", "uk inflation", "uk rates", "britain economy", "british economy"],
        "JPY": ["jpy", "yen", "boj", "bank of japan", "japan", "tokyo"],
    }
    terms = keywords.get(currency, [currency.lower(), pair.lower()])
    clauses = [f"lower(headline) LIKE '%{term.replace(chr(39), chr(39) + chr(39))}%'" for term in terms]
    include = "(" + " OR ".join(clauses) + ")"
    noise_terms = [
        "shares",
        "stock",
        "dividend",
        "donates",
        "protein",
        "compounders",
        "compound",
        "yielding",
        "market by top countries",
        "buy kensington",
    ]
    exclusions = " AND ".join(
        f"lower(headline) NOT LIKE '%{term.replace(chr(39), chr(39) + chr(39))}%'"
        for term in noise_terms
    )
    return f"({include} AND {exclusions})"


def template_rows_for_month(pair: str, month: str, events_per_currency: int) -> Iterable[dict[str, str]]:
    days = month_days(month)
    for currency in pair_currencies(pair):
        for idx in range(events_per_currency):
            day = days[(idx * 3 + len(currency)) % len(days)]
            category, impact, template = stable_pick(EVENTS, f"{pair}-{currency}-{month}-{idx}")
            headline = template.format(pair=pair, ccy=currency)
            yield {
                "timestamp_utc": event_timestamp(day, pair, currency, idx),
                "event_type": "headline",
                "currency": currency,
                "impact": impact,
                "headline": headline,
                "actual": "",
                "forecast": "",
                "source": "synthetic_gap_fill",
                "url": "",
                "event_category": category,
                "sentiment_score": "0.0",
            }


def rows_from_real_news(
    con,
    rel: str,
    pair: str,
    month: str,
    events_per_currency: int,
) -> list[dict[str, str]]:
    days = month_days(month)
    rows: list[dict[str, str]] = []

    for currency in pair_currencies(pair):
        relevance = headline_filter_sql(currency, pair)
        samples = con.execute(
            f"""
            SELECT
                timestamp_utc,
                coalesce(event_type, 'headline') AS event_type,
                upper(coalesce(currency, '')) AS currency,
                coalesce(impact, 'medium') AS impact,
                headline,
                coalesce(actual, '') AS actual,
                coalesce(forecast, '') AS forecast,
                coalesce(source, 'historical_news') AS source,
                coalesce(url, '') AS url,
                coalesce(event_category, 'commentary') AS event_category,
                '0.0' AS sentiment_score
            FROM {rel}
            WHERE headline IS NOT NULL
              AND length(trim(headline)) > 12
              AND upper(coalesce(currency, '')) = '{currency}'
              AND {relevance}
              AND strftime(try_cast(timestamp_utc AS TIMESTAMP), '%Y-%m') <> '{month}'
            ORDER BY hash('{pair}-{month}-{currency}' || coalesce(headline, '') || coalesce(url, ''))
            LIMIT {int(events_per_currency)}
            """
        ).fetchall()

        if len(samples) < events_per_currency:
            samples = con.execute(
                f"""
                SELECT
                    timestamp_utc,
                    coalesce(event_type, 'headline') AS event_type,
                    upper(coalesce(currency, '')) AS currency,
                    coalesce(impact, 'medium') AS impact,
                    headline,
                    coalesce(actual, '') AS actual,
                    coalesce(forecast, '') AS forecast,
                    coalesce(source, 'historical_news') AS source,
                    coalesce(url, '') AS url,
                    coalesce(event_category, 'commentary') AS event_category,
                    '0.0' AS sentiment_score
                FROM {rel}
                WHERE headline IS NOT NULL
                  AND length(trim(headline)) > 12
                  AND upper(coalesce(currency, '')) IN ({', '.join("'" + c + "'" for c in pair_currencies(pair))})
                  AND ({' OR '.join(headline_filter_sql(c, pair) for c in pair_currencies(pair))})
                  AND strftime(try_cast(timestamp_utc AS TIMESTAMP), '%Y-%m') <> '{month}'
                ORDER BY hash('{pair}-{month}-{currency}-fallback' || coalesce(headline, '') || coalesce(url, ''))
                LIMIT {int(events_per_currency)}
                """
            ).fetchall()

        for idx, sample in enumerate(samples[:events_per_currency]):
            day = days[(idx * 3 + len(currency)) % len(days)]
            row = dict(zip(FIELDS, sample))
            original_source = str(row.get("source") or "historical_news")
            row["timestamp_utc"] = event_timestamp(day, pair, currency, idx)
            row["currency"] = currency
            row["source"] = f"synthetic_from_real:{original_source}"
            rows.append({field: "" if row.get(field) is None else str(row.get(field)) for field in FIELDS})

    return rows


def detect_missing_months(news_file: str, pairs: list[str]) -> dict[str, list[str]]:
    try:
        import duckdb
    except Exception as exc:
        raise SystemExit(
            "DuckDB is required for --detect-missing. Use --month or install DuckDB."
        ) from exc

    path = Path(news_file)
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"News file is missing or empty: {path}")

    con = duckdb.connect(database=":memory:")
    rel = f"read_parquet('{str(path).replace(chr(39), chr(39) + chr(39))}')"
    result: dict[str, list[str]] = {}

    for pair in pairs:
        currencies = pair_currencies(pair) + ["GLOBAL", "ALL"]
        sql_currencies = ", ".join("'" + c + "'" for c in currencies)
        months = [
            row[0]
            for row in con.execute(
                f"""
                SELECT DISTINCT strftime(try_cast(timestamp_utc AS TIMESTAMP), '%Y-%m') AS ym
                FROM {rel}
                WHERE timestamp_utc IS NOT NULL
                  AND try_cast(timestamp_utc AS TIMESTAMP) IS NOT NULL
                  AND upper(coalesce(currency, '')) IN ({sql_currencies})
                ORDER BY ym
                """
            ).fetchall()
            if row[0]
        ]
        if not months:
            result[pair] = []
            continue

        present = set(months)
        start_y, start_m = parse_month(months[0])
        end_y, end_m = parse_month(months[-1])
        missing: list[str] = []
        y, m = start_y, start_m
        while (y, m) <= (end_y, end_m):
            ym = f"{y:04d}-{m:02d}"
            if ym not in present:
                missing.append(ym)
            m += 1
            if m == 13:
                y += 1
                m = 1
        result[pair] = missing

    con.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic historical news rows for missing pair-month coverage."
    )
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--news-file", default=DEFAULT_NEWS_FILE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--month",
        action="append",
        default=[],
        help="Month to generate, YYYY-MM. Can be repeated. Applies to all pairs.",
    )
    parser.add_argument(
        "--detect-missing",
        action="store_true",
        help="Detect missing months from --news-file instead of using built-in GBPJPY gaps.",
    )
    parser.add_argument("--events-per-currency", type=int, default=8)
    parser.add_argument(
        "--method",
        choices=["real-derived", "template"],
        default="real-derived",
        help="real-derived samples existing historical news; template uses deterministic generic rows.",
    )
    parser.add_argument("--append", action="store_true", help="Append to --out instead of overwriting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = [p.upper().replace("/", "") for p in args.pairs]

    if args.detect_missing:
        months_by_pair = detect_missing_months(args.news_file, pairs)
    elif args.month:
        months = sorted(set(args.month))
        for month in months:
            parse_month(month)
        months_by_pair = {pair: months for pair in pairs}
    else:
        months_by_pair = {pair: DEFAULT_MISSING_MONTHS.get(pair, []) for pair in pairs}

    rows: list[dict[str, str]] = []
    if args.method == "real-derived":
        try:
            import duckdb
        except Exception as exc:
            raise SystemExit("DuckDB is required for --method real-derived") from exc
        news_path = Path(args.news_file)
        if not news_path.exists() or news_path.stat().st_size == 0:
            raise SystemExit(f"News file is missing or empty: {news_path}")
        con = duckdb.connect(database=":memory:")
        rel = f"read_parquet('{str(news_path).replace(chr(39), chr(39) + chr(39))}')"
        for pair in pairs:
            for month in months_by_pair.get(pair, []):
                rows.extend(rows_from_real_news(con, rel, pair, month, args.events_per_currency))
        con.close()
    else:
        for pair in pairs:
            for month in months_by_pair.get(pair, []):
                rows.extend(template_rows_for_month(pair, month, args.events_per_currency))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and out_path.exists() else "w"
    write_header = mode == "w"

    with out_path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} synthetic rows to {out_path}")
    for pair in pairs:
        months = months_by_pair.get(pair, [])
        print(f"  {pair}: {len(months)} months -> {', '.join(months) if months else 'none'}")


if __name__ == "__main__":
    main()
