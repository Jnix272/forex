import io
import zipfile
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# GDELT 2.0 Event CSV has 61 columns. We care about:
# 1: SQLDATE (YYYYMMDD) or 2: MonthYear
# 7: Actor1CountryCode
# 17: Actor2CountryCode
# 34: AvgTone
# 60: SOURCEURL
# However, sometimes URLs are 60, sometimes 57 depending on parsing. 
# GDELT 2.0 format has 61 columns exactly.
# Let's map column indices.
COL_SQLDATE = 1
COL_ACTOR1_COUNTRY = 7
COL_ACTOR2_COUNTRY = 17
COL_AVG_TONE = 34
COL_SOURCEURL = 60

# We map GDELT Country Codes to our currencies.
# This is a rough mapping. GDELT uses 3-character FIPS 10-4 country codes for some, or 3-char ISO?
# Actually GDELT uses 3-letter codes like 'USA', 'GBR', 'EUR' (EU), 'JPN', 'CAN', 'AUS'
COUNTRY_TO_CURRENCY = {
    'USA': 'USD',
    'GBR': 'GBP',
    'EUR': 'EUR',
    'JPN': 'JPY',
    'CAN': 'CAD',
    'AUS': 'AUD',
}

def generate_urls(start_date: datetime, end_date: datetime):
    """Generate GDELT 2.0 Event CSV zip URLs for every 15 minutes."""
    urls = []
    current = start_date
    while current <= end_date:
        # format: YYYYMMDDHHMM00
        ds_str = current.strftime("%Y%m%d%H%M00")
        url = f"http://data.gdeltproject.org/gdeltv2/{ds_str}.export.CSV.zip"
        urls.append((current, url))
        current += timedelta(minutes=15)
    return urls

def download_and_parse_gdelt(url_info):
    dt, url = url_info
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return dt, None
        
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            # There should be exactly one .csv file inside
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                # Read without headers
                df = pd.read_csv(f, sep='\t', header=None, dtype=str, on_bad_lines='skip')
                
                # Extract relevant columns
                if len(df.columns) < 61:
                    return dt, None
                
                df_subset = df[[COL_SQLDATE, COL_ACTOR1_COUNTRY, COL_ACTOR2_COUNTRY, COL_AVG_TONE, COL_SOURCEURL]].copy()
                df_subset.columns = ['sqldate', 'actor1', 'actor2', 'avgtone', 'url']
                
                # Filter rows where either actor is in our target list
                targets = list(COUNTRY_TO_CURRENCY.keys())
                mask = df_subset['actor1'].isin(targets) | df_subset['actor2'].isin(targets)
                df_filtered = df_subset[mask].copy()
                
                if df_filtered.empty:
                    return dt, pd.DataFrame()
                
                # Map to currency (just pick the first matching one)
                def get_currency(r):
                    if r['actor1'] in targets:
                        return COUNTRY_TO_CURRENCY[r['actor1']]
                    return COUNTRY_TO_CURRENCY[r['actor2']]
                
                df_filtered['currency'] = df_filtered.apply(get_currency, axis=1)
                
                # Convert avgtone to float
                df_filtered['sentiment_score'] = pd.to_numeric(df_filtered['avgtone'], errors='coerce')
                df_filtered = df_filtered.dropna(subset=['sentiment_score'])
                
                # Reset index so assignment to df_final aligns properly
                df_filtered = df_filtered.reset_index(drop=True)
                
                # Format final dataframe
                # Note: sqldate is YYYYMMDD. For precise timestamp we can use the 15-minute file datetime.
                df_final = pd.DataFrame()
                df_final['timestamp_utc'] = [dt.strftime('%Y-%m-%dT%H:%M:%SZ')] * len(df_filtered)
                df_final['event_type'] = 'headline'
                df_final['currency'] = df_filtered['currency']
                df_final['impact'] = 'medium'
                df_final['headline'] = ''
                df_final['actual'] = ''
                df_final['forecast'] = ''
                df_final['source'] = 'gdelt2_bulk'
                df_final['url'] = df_filtered['url']
                df_final['sentiment_score'] = df_filtered['sentiment_score']
                
                return dt, df_final
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return dt, None

def main():
    parser = argparse.ArgumentParser(description="GDELT 2.0 15-minute Bulk Downloader")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel downloads")
    parser.add_argument("--out", type=str, default="data/raw/news/historical_news_gdelt2.csv")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    # End of day
    end_dt = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1, minutes=-15)

    urls = generate_urls(start_dt, end_dt)
    print(f"Generated {len(urls)} URLs to download (15-min intervals).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    columns = ["timestamp_utc", "event_type", "currency", "impact", "headline", 
               "actual", "forecast", "source", "url", "sentiment_score"]
               
    existing_dts = set()
    if out_path.exists():
        import polars as pl
        print("Scanning existing CSV to resume...")
        try:
            df_exist = pl.scan_csv(str(out_path), ignore_errors=True).select("timestamp_utc").unique().collect()
            existing_dts = set(df_exist["timestamp_utc"].to_list())
            print(f"Found {len(existing_dts)} unique timestamps already processed.")
            
            # Filter URLs
            filtered_urls = []
            for dt, url in urls:
                if dt.strftime('%Y-%m-%dT%H:%M:%SZ') not in existing_dts:
                    filtered_urls.append((dt, url))
            print(f"Skipping {len(urls) - len(filtered_urls)} URLs. {len(filtered_urls)} remaining to download.")
            urls = filtered_urls
        except Exception as e:
            print(f"Could not read existing CSV for resume: {e}")
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(",".join(columns) + "\n")

    success_count = 0
    fail_count = 0
    total_rows = 0

    chunk_size = 5000
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_and_parse_gdelt, u): u for u in chunk}
            
            for future in as_completed(futures):
                dt, df = future.result()
                if df is None:
                    fail_count += 1
                else:
                    success_count += 1
                    if not df.empty:
                        # Append to CSV
                        df[columns].to_csv(out_path, mode='a', header=False, index=False)
                        total_rows += len(df)
                
                # Print progress every 100 files
                completed = success_count + fail_count
                if completed % 100 == 0:
                    print(f"Progress: {completed}/{len(urls)} files processed. Rows extracted: {total_rows}")
        
        # Force garbage collection between chunks to clear ThreadPool memory
        import gc
        gc.collect()

    print(f"\nFinished! Processed {success_count} files successfully, {fail_count} failed.")
    print(f"Saved {total_rows} rows to {out_path}")
    print("Note: The 'headline' column is intentionally left blank. You can merge this file into historical_news_combined.parquet.")

if __name__ == "__main__":
    main()
