import polars as pl
import os
import time

def main():
    start_time = time.time()
    print("Starting massive Polars out-of-core merge...")
    
    # Define file paths
    file_hf = "data/raw/news/Hugging_Face.csv"
    file_gdelt = "data/raw/news/historical_news_2021_2025.csv"
    file_fnspid = "data/raw/news/historical_news_fnspid_full.csv"
    output_file = "data/raw/news/historical_news_master.parquet"
    
    schema_override = {
        "timestamp_utc": pl.Utf8,
        "event_type": pl.Utf8,
        "currency": pl.Utf8,
        "impact": pl.Utf8,
        "headline": pl.Utf8,
        "actual": pl.Utf8,
        "forecast": pl.Utf8,
        "source": pl.Utf8,
        "url": pl.Utf8,
        "sentiment_score": pl.Utf8
    }
    
    lazy_frames = []
    
    if os.path.exists(file_hf):
        print(f"Found {file_hf}")
        lf1 = pl.scan_csv(file_hf, schema_overrides=schema_override, ignore_errors=True)
        lazy_frames.append(lf1)
        
    if os.path.exists(file_gdelt):
        print(f"Found {file_gdelt}")
        lf2 = pl.scan_csv(file_gdelt, schema_overrides=schema_override, ignore_errors=True)
        lazy_frames.append(lf2)
        
    if os.path.exists(file_fnspid):
        print(f"Found {file_fnspid}")
        lf3 = pl.scan_csv(file_fnspid, schema_overrides=schema_override, ignore_errors=True)
        lazy_frames.append(lf3)
        
    if not lazy_frames:
        print("No datasets found to merge!")
        return
        
    print("Building execution graph...")
    # Concatenate all lazy frames with diagonal strategy to handle missing columns
    master_lf = pl.concat(lazy_frames, how="diagonal")
    
    # String comparison is incredibly fast and memory efficient for ISO dates.
    # Drop completely invalid rows first, then filter for >= 2003
    master_lf = master_lf.filter(
        pl.col("timestamp_utc").is_not_null()
    ).filter(
        pl.col("timestamp_utc") >= "2003-01-01"
    )
    
    # Sort chronologically
    master_lf = master_lf.sort("timestamp_utc")
    
    # PIPE-010: Sink to compressed Parquet instead of flat CSV (10-50× faster reads)
    # Add year column for optional partitioning
    master_lf = master_lf.with_columns(
        pl.col("timestamp_utc").str.slice(0, 4).alias("year")
    )
    print(f"Executing graph and streaming to {output_file}...")
    master_lf.sink_parquet(output_file, compression="zstd")
    
    elapsed = time.time() - start_time
    print(f"Merge completed successfully in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    main()
