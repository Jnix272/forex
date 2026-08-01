import polars as pl
import argparse
import os
import subprocess
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

def audit_and_repair(start_date: str, end_date: str, csv_path: str):
    print(f"Auditing {csv_path} from {start_date} to {end_date}...")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist. All months are missing.")
    else:
        # Load dataset and extract unique Year-Month combinations
        print("Scanning dataset for existing months...")
        df = pl.scan_csv(csv_path, ignore_errors=True)
        # Parse timestamp to extract Year and Month
        df_months = df.select([
            pl.col("timestamp_utc").str.slice(0, 7).alias("year_month")
        ]).filter(pl.col("year_month").is_not_null()).unique().collect()
        
        existing_months = set(df_months["year_month"].to_list())
        print(f"Found {len(existing_months)} unique months in dataset.")

    # Generate expected months
    start_dt = datetime.strptime(start_date[:7], "%Y-%m")
    end_dt = datetime.strptime(end_date[:7], "%Y-%m")
    
    expected_months = []
    curr_dt = start_dt
    while curr_dt <= end_dt:
        expected_months.append(curr_dt.strftime("%Y-%m"))
        curr_dt += relativedelta(months=1)
        
    missing_months = [m for m in expected_months if not os.path.exists(csv_path) or m not in existing_months]
    
    if not missing_months:
        print("Audit Complete: No missing months found!")
        return

    print(f"Audit Complete: Found {len(missing_months)} missing months.")
    print(f"Missing: {missing_months[:5]} ...")
    
    # Group missing months into continuous blocks
    missing_months.sort()
    
    # Let's categorize the missing months based on our available sources
    # pre-2017 -> Kaggle (download_2008_news.py)
    # post-2017 -> GDELT (download_historical_news.py)
    
    pre_2017 = [m for m in missing_months if m < "2017-02"]
    post_2017 = [m for m in missing_months if m >= "2017-02"]
    
    if pre_2017:
        print(f"Repairing pre-2017 gaps ({len(pre_2017)} months) using Kaggle 2008 dataset...")
        # Since the Kaggle dataset is bulk, we just run the download script once.
        # It covers 2008 to 2016.
        # Check if the download script exists
        if os.path.exists("scripts/download_2008_news.py"):
            import sys
            subprocess.run([sys.executable, "scripts/download_2008_news.py"], check=True)
            # Merge logic is handled in the download_2008_news.py script itself or we do it here.
        else:
            print("Warning: scripts/download_2008_news.py not found.")
            
    if post_2017:
        print(f"Repairing post-2017 gaps ({len(post_2017)} months) using GDELT API...")
        # We find the min and max of post-2017 to run download_historical_news.py
        start_gdelt = post_2017[0] + "-01"
        # For end_gdelt, we take the last month and set it to the end of that month
        # Simplification: just use the first of the next month
        end_gdelt_month = datetime.strptime(post_2017[-1], "%Y-%m") + relativedelta(months=1)
        end_gdelt = end_gdelt_month.strftime("%Y-%m-01")
        
        print(f"Running GDELT download from {start_gdelt} to {end_gdelt}...")
        import sys
        subprocess.run([
            sys.executable, "scripts/download_historical_news.py",
            "--start", start_gdelt,
            "--end", end_gdelt,
            "--workers", "4"
        ], check=True)
        
    print("Repair process finished. Re-run audit to verify.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2008-01-01", help="Start date to audit (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.now(timezone.utc).strftime("%Y-%m-%d"), help="End date to audit")
    parser.add_argument("--csv", type=str, default="data/raw/news/historical_news_combined.parquet", help="Path to historical news CSV")
    args = parser.parse_args()
    
    audit_and_repair(args.start, args.end, args.csv)
