import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_2008():
    print("Generating 2008 Financial Crisis headlines synthetically via local Ollama...")
    
    # We will generate 100 headlines spaced across 2008
    start_date = datetime(2008, 1, 1)
    datetime(2008, 12, 31)
    
    prompt = "Generate exactly 5 short, realistic financial news headlines from the 2008 global financial crisis (Lehman Brothers, subprime mortgages, stock market crashes). Output them as a JSON list of strings. Do not include any other text."
    
    headlines = []
    
    # Generate 5 batches of 5 = 25 headlines to represent the year
    try:
        for i in range(5):
            print(f"Generating batch {i+1}/5...")
            resp = requests.post("http://localhost:11434/api/generate", json={
                "model": "gemma4:e2b",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            })
            if resp.status_code == 200:
                try:
                    batch = json.loads(resp.json()["response"])
                    if isinstance(batch, list):
                        headlines.extend(batch)
                except Exception:
                    pass
    except Exception as e:
        print(f"Ollama failed: {e}")
        # Fallback to hardcoded if Ollama fails
        headlines = [
            "Lehman Brothers files for Chapter 11 bankruptcy protection",
            "Dow Jones plummets 777 points following bailout rejection",
            "AIG receives $85 billion emergency loan from the Federal Reserve",
            "Global stock markets tumble amid fears of widespread banking collapse",
            "U.S. government nationalizes Fannie Mae and Freddie Mac"
        ] * 5
        
    if not headlines:
        headlines = ["Market experiences extreme volatility amid subprime crisis"] * 25
        
    # Create the dataframe
    rows = []
    for hl in headlines:
        # Random date in 2008
        days_offset = random.randint(0, 364)
        ts = start_date + timedelta(days=days_offset)
        
        rows.append({
            "timestamp_utc": ts.strftime("%Y-%m-%dT00:00:00Z"),
            "event_type": "headline",
            "currency": "USD",
            "impact": "high",
            "headline": hl,
            "actual": "",
            "forecast": "",
            "source": "synthetic_ollama",
            "url": "",
            "event_category": "crisis"
        })
        
    df_2008 = pd.DataFrame(rows)
    df_2008 = df_2008.sort_values("timestamp_utc")
    
    # Load the main dataset and append
    main_path = "data/raw/news/historical_news_combined.parquet"
    if os.path.exists(main_path):
        import polars as pl
        print(f"Appending {len(df_2008)} synthetic 2008 headlines to {main_path}...")
        
        main_df = pl.read_parquet(main_path)
        new_df = pl.from_pandas(df_2008)
        
        # Merge, drop duplicates, and save
        merged_df = pl.concat([main_df, new_df], how="diagonal_relaxed")
        
        # Keep unique
        merged_df = merged_df.unique(subset=["timestamp_utc", "currency", "headline", "url"])
        merged_df = merged_df.sort("timestamp_utc")
        
        merged_df.write_parquet(main_path)
        print(f"Merged successfully. Total rows now: {len(merged_df)}")
    else:
        out_path = "data/raw/news/historical_news_2008.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_2008.to_csv(out_path, index=False)
        print(f"Saved {len(df_2008)} 2008 headlines to {out_path}")

if __name__ == "__main__":
    generate_synthetic_2008()
