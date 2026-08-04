import os

import pandas as pd
from datasets import load_dataset


def main():
    print("Loading FNSPID dataset from Hugging Face...")
    # Stream the dataset to avoid downloading the entire 15M rows at once into RAM
    dataset = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)

    print("Extracting ALL data (1999-2023)...")
    output_file = "data/raw/news/historical_news_fnspid_full.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write header if new file
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("timestamp_utc,event_type,currency,impact,headline,actual,forecast,source,url,sentiment_score\n")

    records = []
    chunk_size = 100000
    total_count = 0
    skip_count = 0

    if os.path.exists(output_file):
        print("Counting existing rows to resume...")
        with open(output_file, encoding='utf-8') as f:
            skip_count = sum(1 for _ in f) - 1 # minus header
        print(f"Skipping first {skip_count} rows...")
        total_count = skip_count

    try:
        for i, row in enumerate(dataset):
            if i < skip_count:
                continue

            records.append(row)
            total_count += 1

            # Write to disk in chunks to save RAM
            if len(records) >= chunk_size:
                df = pd.DataFrame(records)

                df_out = pd.DataFrame()
                df_out['timestamp_utc'] = df.get('Date', '')
                df_out['event_type'] = 'headline'
                df_out['currency'] = 'USD'
                df_out['impact'] = 'medium'

                # Replace newlines/commas in headlines to prevent CSV breakage
                df_out['headline'] = df['Article_title'].astype(str).str.replace('\n', ' ').str.replace('\r', '').str.replace(',', '')

                df_out['actual'] = ''
                df_out['forecast'] = ''
                df_out['source'] = 'fnspid_full'
                df_out['url'] = df.get('Url', '')
                df_out['sentiment_score'] = ''

                df_out.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
                print(f"Written chunk... Total rows saved: {total_count}")
                records = [] # clear RAM

    except Exception as e:
        print(f"Streaming finished or interrupted: {e}")

    # Write remaining records
    if len(records) > 0:
        df = pd.DataFrame(records)
        df_out = pd.DataFrame()
        df_out['timestamp_utc'] = df.get('Date', '')
        df_out['event_type'] = 'headline'
        df_out['currency'] = 'USD'
        df_out['impact'] = 'medium'
        df_out['headline'] = df['Article_title'].astype(str).str.replace('\n', ' ').str.replace('\r', '').str.replace(',', '')
        df_out['actual'] = ''
        df_out['forecast'] = ''
        df_out['source'] = 'fnspid_full'
        df_out['url'] = df.get('Url', '')
        df_out['sentiment_score'] = ''
        df_out.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
        print(f"Written final chunk... Total rows saved: {total_count}")

    print(f"\nSuccessfully finished! Saved {total_count} rows to {output_file}")

if __name__ == "__main__":
    main()
