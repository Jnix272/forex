import argparse
import os
import shutil
import time

import duckdb


DEFAULT_INPUTS = [
    "data/raw/news/Hugging_Face.csv",
    "data/raw/news/historical_news_2021_2025.csv",
    "data/raw/news/historical_news_fnspid_full.csv",
]
DEFAULT_OUTPUT = "data/raw/news/historical_news_combined.parquet"
DEFAULT_PAIRS = ["EURUSD", "USDJPY", "GBPUSD"]
FX_MACRO_TERMS = [
    "forex", "foreign exchange", "fx",
    "federal reserve", "fed", "fomc", "ecb", "bank of england", "boe",
    "bank of japan", "boj", "central bank", "monetary policy",
    "interest rate", "rate hike", "rate cut",
    "treasury yield", "treasury yields", "bond yield", "bond yields",
    "government bond", "gilts", "bund",
    "inflation", "cpi", "ppi", "gdp", "recession",
    "unemployment", "payroll", "payrolls", "nfp", "jobs report", "wage growth",
    "retail sales", "consumer confidence", "durable goods",
    "industrial production", "manufacturing pmi", "services pmi", "pmi", "ism",
    "trade balance", "current account", "budget deficit",
    "quantitative easing", "qe", "stimulus", "tariff", "trade war",
    "brexit", "eurozone", "euro zone",
]
CURRENCY_TERMS = [
    "currency", "currencies", "dollar", "greenback", "euro", "yen",
    "sterling", "british pound", "pound sterling", "usd", "eur", "jpy", "gbp",
]
CURRENCY_CONTEXT_TERMS = [
    "forex", "foreign exchange", "fx", "market", "markets", "trader", "traders",
    "trade", "trades", "trading", "against", "versus", "vs", "pair", "pairs",
    "rise", "rises", "rising", "rose", "gain", "gains", "gained", "higher",
    "fall", "falls", "falling", "fell", "drop", "drops", "dropped", "lower",
    "slip", "slips", "slipped", "weaken", "weakens", "weakened", "weak",
    "strengthen", "strengthens", "strengthened", "strong", "mixed", "subdued",
    "weighed", "rally", "rebound", "volatility", "safe haven", "carry trade",
    "libor", "interbank",
]
GENERAL_NEWS_EXCLUDE_TERMS = [
    "sport", "football", "soccer", "tennis", "cricket", "olympic",
    "movie", "film", "music", "celebrity", "fashion", "restaurant",
    "recipe", "weather", "earthquake", "hurricane", "fire", "crime",
    "murder", "shooting", "killed", "dead", "wedding", "divorce",
    "rebel", "rebels", "violence", "war", "iraq", "kenya",
    "net asset value", "portfolio update", "transaction in own shares",
    "name change", "etf", "stock", "stocks", "share", "shares",
    "trillion-dollar", "valuation", "contract", "acquires", "awarded",
    "distribution", "announces", "plc", "inc", "corp",
    "etfs", "software", "integration", "ipo", "hydrogen",
]


def _pair_currencies(pairs):
    currencies = {"GLOBAL", "ALL", "G10", "WORLD", "MARKET", "MARKETS", ""}
    for pair in pairs:
        p = str(pair or "").upper().replace("/", "").replace("_", "")
        if len(p) >= 6:
            currencies.add(p[:3])
            currencies.add(p[3:6])
    return sorted(currencies)


def _sql_list(values):
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def _regex_from_terms(terms):
    escaped_terms = []
    for term in terms:
        escaped = str(term).lower().replace("\\", "\\\\")
        for char in ".^$*+?{}[]|()":
            escaped = escaped.replace(char, "\\" + char)
        escaped_terms.append(r"\b" + escaped.replace(" ", r"\s+") + r"\b")
    return "(" + "|".join(escaped_terms) + ")"


def _news_relation(path):
    escaped = str(path).replace("'", "''")
    if str(path).lower().endswith(".parquet"):
        return f"read_parquet('{escaped}')"
    return f"""read_csv('{escaped}',
                    ignore_errors=true,
                    header=true,
                    null_padding=true,
                    auto_detect=false,
                    parallel=false,
                    columns={{'timestamp_utc':'VARCHAR', 'event_type':'VARCHAR', 'currency':'VARCHAR', 'impact':'VARCHAR', 'headline':'VARCHAR', 'actual':'VARCHAR', 'forecast':'VARCHAR', 'source':'VARCHAR', 'url':'VARCHAR', 'sentiment_score':'VARCHAR'}}
                )"""


def merge_massive_datasets(
    files=None,
    output_parquet=DEFAULT_OUTPUT,
    pairs=None,
    start_year=2008,
    end_year=2026,
    strict_fx_filter=True,
    keep_temp=False,
):
    print("Starting ultra-fast massive dataset merge via DuckDB (Partition -> Dedup -> Merge)...")
    start_time = time.time()
    files = files or DEFAULT_INPUTS
    pairs = pairs or DEFAULT_PAIRS
    currencies = _pair_currencies(pairs)

    existing_files = [f for f in files if os.path.exists(f)]
    master_file = "data/raw/news/historical_news_master.csv"
    if files == DEFAULT_INPUTS and os.path.exists(master_file):
        existing_files = [master_file]
    if not existing_files:
        print("No news CSV files found to merge.")
        return

    print(f"Found {len(existing_files)} massive news files to merge.")
    print(f"Keeping currencies for pairs {', '.join(pairs)}: {', '.join(c for c in currencies if c)}")
    if strict_fx_filter:
        print("Applying strict FX/macro relevance filter to headlines.")

    partition_dir = "data/raw/news/partitions"
    dedup_dir = "data/raw/news/deduped"
    tmp_output = output_parquet + ".tmp"

    for temp_dir in (partition_dir, dedup_dir):
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    if os.path.exists(tmp_output):
        os.remove(tmp_output)

    con = duckdb.connect(database=':memory:')
    con.execute("SET preserve_insertion_order=false;")
    con.execute("PRAGMA memory_limit='4GB';")

    # -------------------------------------------------------------------------
    # STEP 1: Filter, deduplicate, and write the final Parquet in one pass.
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Filtering and deduplicating in one pass...")
    queries = []
    relevance_filter = ""
    if strict_fx_filter:
        positive_re = _regex_from_terms(FX_MACRO_TERMS).replace("'", "''")
        currency_re = _regex_from_terms(CURRENCY_TERMS).replace("'", "''")
        currency_context_re = _regex_from_terms(CURRENCY_CONTEXT_TERMS).replace("'", "''")
        negative_re = _regex_from_terms(GENERAL_NEWS_EXCLUDE_TERMS).replace("'", "''")
        # GDELT bulk exports often have blank headlines but valid URL + tone.
        # Keep those rows; apply FX/macro headline filter only when a headline exists.
        relevance_filter = f"""
                  AND (
                        lower(coalesce(source, '')) LIKE '%gdelt%'
                        OR (
                            length(trim(coalesce(headline, ''))) > 0
                            AND (
                                regexp_matches(
                                    lower(coalesce(headline, '') || ' ' || coalesce(event_type, '') || ' ' || coalesce(source, '')),
                                    '{positive_re}'
                                )
                                OR (
                                    regexp_matches(lower(coalesce(headline, '')), '{currency_re}')
                                    AND regexp_matches(lower(coalesce(headline, '')), '{currency_context_re}')
                                )
                            )
                            AND NOT regexp_matches(lower(coalesce(headline, '')), '{negative_re}')
                        )
                  )"""
    for f in existing_files:
        q = f"""SELECT
                    timestamp_utc, event_type, currency, impact,
                    CASE
                        WHEN length(trim(coalesce(headline, ''))) > 0 THEN headline
                        ELSE regexp_replace(coalesce(url, ''), 'https?://[^/]+/', '')
                    END AS headline,
                    actual, forecast, source, url, sentiment_score,
                    TRY_CAST(SUBSTRING(timestamp_utc, 1, 4) AS INTEGER) as part_year,
                    -- Dedupe key: use URL when headline is blank (GDELT bulk)
                    CASE
                        WHEN length(trim(coalesce(headline, ''))) > 0 THEN headline
                        ELSE coalesce(url, '')
                    END AS dedupe_key
                FROM {_news_relation(f)}
                WHERE timestamp_utc IS NOT NULL
                  AND (
                        length(trim(coalesce(headline, ''))) > 0
                        OR (
                            length(trim(coalesce(url, ''))) > 0
                            AND lower(coalesce(source, '')) LIKE '%gdelt%'
                        )
                  )
                  AND upper(coalesce(currency, '')) IN ({_sql_list(currencies)})
                  {relevance_filter}"""
        queries.append(q)

    union_query = " UNION ALL ".join(queries)
    con.execute(f"""
        COPY (
            SELECT
                timestamp_utc,
                ANY_VALUE(event_type) as event_type,
                ANY_VALUE(currency) as currency,
                ANY_VALUE(impact) as impact,
                ANY_VALUE(headline) as headline,
                ANY_VALUE(actual) as actual,
                ANY_VALUE(forecast) as forecast,
                ANY_VALUE(source) as source,
                ANY_VALUE(url) as url,
                ANY_VALUE(sentiment_score) as sentiment_score
            FROM ({union_query})
            WHERE part_year >= {int(start_year)} AND part_year <= {int(end_year)}
            GROUP BY timestamp_utc, dedupe_key
            ORDER BY timestamp_utc
        ) TO '{tmp_output}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """)
    os.replace(tmp_output, output_parquet)

    # Cleanup temp dirs (optional, but good for space)
    if keep_temp:
        print("Keeping temporary partitions.")
    else:
        print("Cleaning up temporary partitions...")
        try:
            for temp_dir in (partition_dir, dedup_dir):
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        except Exception as e:
            print("Cleanup failed:", e)

    elapsed = time.time() - start_time
    file_size_gb = os.path.getsize(output_parquet) / (1024**3)
    print(f"\n[SUCCESS] Pipeline complete in {elapsed:.2f} seconds!")
    print(f"[SUCCESS] Generated {output_parquet} ({file_size_gb:.2f} GB)")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge large news CSV/Parquet files into a pair-scoped Parquet dataset.")
    parser.add_argument("--input", nargs="+", default=DEFAULT_INPUTS, help="News CSV or Parquet files to merge")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output Parquet path")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="Pairs to keep")
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--no-strict-fx-filter",
        action="store_true",
        help="Disable the default FX/macro headline relevance filter.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Do not delete partition/dedup temp dirs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_massive_datasets(
        files=args.input,
        output_parquet=args.output,
        pairs=[p.upper().replace("/", "") for p in args.pairs],
        start_year=args.start_year,
        end_year=args.end_year,
        strict_fx_filter=not args.no_strict_fx_filter,
        keep_temp=args.keep_temp,
    )
