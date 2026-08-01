import duckdb
import time

def main():
    start_time = time.time()
    print("Starting bulletproof DuckDB out-of-core merge...")
    
    # Connect to a file-backed DuckDB database to prevent RAM exhaustion
    # Using a file database ensures DuckDB spills to disk automatically
    con = duckdb.connect("data/raw/news/merge_db.duckdb")
    
    # Configure DuckDB to use max 12GB of RAM (adjust if needed)
    con.execute("PRAGMA memory_limit='12GB'")
    import os
    os.makedirs('data/raw/news/tmp', exist_ok=True)
    # Configure temp directory for out-of-core spilling
    con.execute("PRAGMA temp_directory='data/raw/news/tmp'")
    
    output_file = "data/raw/news/historical_news_master.csv"
    
    query = f"""
    COPY (
        SELECT timestamp_utc, event_type, currency, impact, headline, actual, forecast, source, url, sentiment_score
        FROM (
            SELECT timestamp_utc, event_type, currency, impact, headline, actual, forecast, source, url, sentiment_score
            FROM read_csv_auto('data/raw/news/historical_news_fnspid_full.csv', ignore_errors=true)
            WHERE timestamp_utc >= '2003-01-01'
            
            UNION ALL
            
            SELECT timestamp_utc, event_type, currency, impact, headline, actual, forecast, source, url, sentiment_score
            FROM read_csv_auto('data/raw/news/historical_news_2021_2025.csv', ignore_errors=true)
            WHERE timestamp_utc >= '2003-01-01'
            
            UNION ALL
            
            SELECT timestamp_utc, event_type, currency, impact, headline, actual, forecast, source, url, null as sentiment_score
            FROM read_csv_auto('data/raw/news/Hugging_Face.csv', ignore_errors=true)
            WHERE timestamp_utc >= '2003-01-01'
        )
        ORDER BY timestamp_utc ASC
    ) TO '{output_file}' (HEADER, DELIMITER ',');
    """
    
    print("Executing massive join and sort (this may take a while)...")
    try:
        con.execute(query)
        elapsed = time.time() - start_time
        print(f"Merge completed successfully in {elapsed:.2f} seconds!")
    except Exception as e:
        print(f"Error during DuckDB merge: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    main()
