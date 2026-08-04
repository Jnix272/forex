import argparse
import io
import random
import time
import zipfile
from datetime import datetime
from datetime import time as dtime
from pathlib import Path

import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover - fallback for very old runtimes
    ZoneInfo = None

# Map CFTC contract names to our FX pairs.
# NOTE: CFTC futures are quoted as FOREIGN-CCY vs USD. For pairs where USD is the
# BASE (USDJPY, USDCAD) the net-positioning sign is inverted relative to our pair
# convention and is corrected after netting (see USD-base handling below).
CONTRACT_MAP = {
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "EURUSD",
    "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE": "GBPUSD",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "USDJPY",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "AUDUSD",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "USDCAD",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "USDCHF",
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE": "NZDUSD",
}

CFTC_TZ = "America/New_York"
# TFF reports measure positioning as of Tuesday but are released the following
# Friday at ~15:30 ET. We stamp rows with the release datetime (UTC) to avoid
# look-ahead bias when joining onto intraday bars.
COT_RELEASE_TIME_ET = dtime(15, 30)

OUT_FILE = Path("data/raw/cot/cot_financials_cleaned.parquet")


def _retry_sleep_seconds(exc: Exception, sleep_s: float, attempt: int) -> float:
    """Exponential backoff with jitter; honours HTTP 429 Retry-After when present."""
    base = max(float(sleep_s), 1.0)
    retry_after = None
    if (
        isinstance(exc, requests.exceptions.HTTPError)
        and getattr(exc, "response", None) is not None
        and exc.response.status_code == 429
    ):
        try:
            retry_after = float(exc.response.headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            retry_after = None
    if retry_after is not None and retry_after > 0:
        return retry_after + random.uniform(0.25, 1.0)
    return min(120.0, base * (2 ** max(0, attempt - 1))) + random.uniform(0.25, 1.0)


def _download_with_retries(url: str, *, retries: int = 4, sleep_s: float = 1.0, timeout: int = 30) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as exc:  # network / HTTP errors
            last_exc = exc
            if attempt >= retries:
                break
            secs = _retry_sleep_seconds(exc, sleep_s, attempt)
            print(f"  -> attempt {attempt}/{retries} failed: {exc}; retrying in {secs:.1f}s")
            time.sleep(secs)
    raise last_exc if last_exc else RuntimeError(f"Failed to download {url}")


def _release_timestamp_utc(report_dates: pd.Series) -> pd.Series:
    """Map each CFTC report (Tuesday) to its public release datetime in UTC.

    Release = the Friday of the same week at 15:30 America/New_York. Computed via
    weekday offset (Friday=4) so it is robust to occasional non-Tuesday dates.
    """
    rd = pd.to_datetime(report_dates)
    # Days from the report weekday to Friday within the same week.
    offsets = rd.dt.weekday.apply(lambda wd: (4 - int(wd)))
    release_dates = rd.dt.normalize() + pd.to_timedelta(offsets, unit="D")
    release_dates = release_dates + pd.Timedelta(hours=COT_RELEASE_TIME_ET.hour,
                                                  minutes=COT_RELEASE_TIME_ET.minute)
    if ZoneInfo is not None:
        local = release_dates.dt.tz_localize(CFTC_TZ, ambiguous="NaT", nonexistent="shift_forward")
        return local.dt.tz_convert("UTC")
    # Fallback: treat ET as a fixed -05:00 offset (approximate, no DST).
    return (release_dates + pd.Timedelta(hours=5)).dt.tz_localize("UTC")


def _existing_years(parquet_path: Path) -> set[int]:
    if not parquet_path.exists():
        return set()
    try:
        existing = pd.read_parquet(parquet_path)
    except Exception as exc:
        print(f"  -> Could not read existing parquet ({exc}); treating as empty")
        return set()
    if "report_year" in existing.columns:
        return set(int(y) for y in existing["report_year"].dropna().unique())
    if "timestamp_utc" in existing.columns:
        # Legacy files lacking report_year: approximate from the release year.
        return set(int(y) for y in pd.to_datetime(existing["timestamp_utc"]).dt.year.dropna().unique())
    return set()


def download_cot_data(start_year=2008, end_year=None, *, retries=4, sleep_s=1.0):
    if end_year is None:
        end_year = datetime.now().year

    out_dir = OUT_FILE.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    have_years = _existing_years(OUT_FILE)
    all_data = []

    # Financial Futures (TFF) report
    for year in range(start_year, end_year + 1):
        if year in have_years:
            print(f"Skipping {year} (already present in {OUT_FILE.name})")
            continue
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
        print(f"Downloading {url} ...")
        try:
            content = _download_with_retries(url, retries=retries, sleep_s=sleep_s)
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                txt_filename = z.namelist()[0]
                with z.open(txt_filename) as f:
                    df = pd.read_csv(f, low_memory=False)
                    df["Year"] = year
                    all_data.append(df)
                    print(f"  -> Successfully parsed {len(df)} rows for {year}")
        except Exception as e:
            print(f"  -> Failed to download/parse {year}: {e}")
        # Be polite to the CFTC host between yearly requests.
        time.sleep(max(0.0, float(sleep_s)))

    if not all_data:
        print("No new COT years downloaded; nothing to do.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # Clean up column names (strip whitespace)
    full_df.columns = full_df.columns.str.strip()

    # Clean up column values (older files have trailing whitespace)
    full_df["Market_and_Exchange_Names"] = full_df["Market_and_Exchange_Names"].str.strip()

    # Filter to only the Forex contracts we care about
    full_df = full_df[full_df["Market_and_Exchange_Names"].isin(CONTRACT_MAP.keys())]

    if len(full_df) == 0:
        print("No FX contracts found in downloaded data.")
        return

    # Map to our standard pair names
    full_df["pair"] = full_df["Market_and_Exchange_Names"].map(CONTRACT_MAP)

    # Convert report dates to the public RELEASE datetime in UTC (anti look-ahead).
    # Older TFF files (≤2012) use "Report_Date_as_MM_DD_YYYY"; newer use "Report_Date_as_YYYY-MM-DD".
    date_col = None
    for candidate in ("Report_Date_as_YYYY-MM-DD", "Report_Date_as_MM_DD_YYYY"):
        if candidate in full_df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise KeyError("Cannot find a report-date column in the downloaded data")
    full_df = full_df.dropna(subset=[date_col]).copy()
    full_df["_report_date"] = pd.to_datetime(full_df[date_col])
    full_df["report_year"] = full_df["_report_date"].dt.year
    full_df["timestamp_utc"] = _release_timestamp_utc(full_df["_report_date"])
    full_df.drop(columns=["_report_date"], inplace=True)

    cols_to_keep = [
        "timestamp_utc",
        "report_year",
        "pair",
        "Lev_Money_Positions_Long_All",
        "Lev_Money_Positions_Short_All",
        "Dealer_Positions_Long_All",
        "Dealer_Positions_Short_All",
        "Open_Interest_All",
    ]

    # Ensure columns exist, fill missing with 0
    for c in cols_to_keep:
        if c not in full_df.columns:
            full_df[c] = 0

    clean_df = full_df[cols_to_keep].copy()

    # Calculate net positions (Leveraged Funds = Hedge Funds, Dealer = Commercial)
    clean_df["net_hedge_fund"] = clean_df["Lev_Money_Positions_Long_All"] - clean_df["Lev_Money_Positions_Short_All"]
    clean_df["net_commercial"] = clean_df["Dealer_Positions_Long_All"] - clean_df["Dealer_Positions_Short_All"]

    # Sign correction: CFTC contracts are FOREIGN-CCY vs USD. When USD is the BASE
    # of our mapped pair (e.g. USDJPY, USDCAD), a long foreign-ccy future is short
    # the pair, so the net sign must be inverted. Pairs where USD is the quote
    # (EURUSD/GBPUSD/AUDUSD) are already aligned.
    usd_base = clean_df["pair"].str.startswith("USD")
    clean_df.loc[usd_base, ["net_hedge_fund", "net_commercial"]] *= -1

    # Merge with any existing data (incremental append) and dedup.
    if OUT_FILE.exists():
        try:
            prev = pd.read_parquet(OUT_FILE)
            clean_df = pd.concat([prev, clean_df], ignore_index=True, sort=False)
        except Exception as exc:
            print(f"  -> Could not merge existing parquet ({exc}); writing fresh")

    clean_df = clean_df.drop_duplicates(subset=["pair", "timestamp_utc"], keep="last")
    clean_df.sort_values(["pair", "timestamp_utc"], inplace=True)

    clean_df.to_parquet(OUT_FILE, index=False)
    print(f"Saved {len(clean_df)} COT records to {OUT_FILE}")


def parse_args():
    p = argparse.ArgumentParser(description="Download & clean CFTC TFF (financial futures) COT data.")
    p.add_argument("--start-year", type=int, default=2008)
    p.add_argument("--end-year", type=int, default=None, help="Defaults to the current year.")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--sleep", type=float, default=1.0, help="Base seconds to sleep/backoff between requests.")
    p.add_argument("--dry-run", action="store_true", help="Parse args and print the plan without downloading.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    end_year = args.end_year if args.end_year is not None else datetime.now().year
    if args.dry_run:
        have = _existing_years(OUT_FILE)
        todo = [y for y in range(args.start_year, end_year + 1) if y not in have]
        print(f"[DRY-RUN] Would download years: {todo or '(none; all present)'}")
        print(f"[DRY-RUN] Output: {OUT_FILE}")
    else:
        download_cot_data(
            start_year=args.start_year,
            end_year=end_year,
            retries=args.retries,
            sleep_s=args.sleep,
        )
