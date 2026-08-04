import os

import pandas as pd
from datasets import load_dataset


def main():
    print("Loading HuggingFace dataset ashraq/financial-news...")
    # Load dataset
    ds = load_dataset('ashraq/financial-news', split='train')

    print(f"Loaded {len(ds)} articles. Converting to pandas...")
    df = ds.to_pandas()

    print("Formatting columns...")
    # Convert 'date' to 'timestamp_utc'
    # Format is '2020-06-01 00:00:00'
    # We will safely convert errors to NaT, then drop them
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df = df.dropna(subset=['date_dt'])
    df['timestamp_utc'] = df['date_dt'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    df['event_type'] = 'headline'

    # Simple heuristic: if headline contains 'EUR', 'GBP', 'JPY' etc we can tag it,
    # but mostly it's US stock news so we default to USD/GLOBAL
    df['currency'] = 'USD'
    df.loc[df['headline'].str.contains(' ECB |Euro|EUR', case=False, na=False), 'currency'] = 'EUR'
    df.loc[df['headline'].str.contains(' BOE |Bank of England|GBP|Pound', case=False, na=False), 'currency'] = 'GBP'
    df.loc[df['headline'].str.contains(' BOJ |Bank of Japan|JPY|Yen', case=False, na=False), 'currency'] = 'JPY'

    df['impact'] = 'medium'
    df['actual'] = ''
    df['forecast'] = ''
    df['source'] = 'hf_ashraq'

    # Rename if necessary (headline and url are already named correctly)

    # Select final columns
    columns = [
        "timestamp_utc",
        "event_type",
        "currency",
        "impact",
        "headline",
        "actual",
        "forecast",
        "source",
        "url",
    ]

    df_final = df[columns]

    print("Deduplicating...")
    df_final = df_final.drop_duplicates(subset=['timestamp_utc', 'headline']).sort_values('timestamp_utc')

    out_dir = "data/raw/news"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/Hugging_Face.csv"

    print(f"Saving {len(df_final)} rows to {out_path}...")
    df_final.to_csv(out_path, index=False)
    print("Done! Completely replaced slow news scraper with HF dataset.")

if __name__ == '__main__':
    main()
