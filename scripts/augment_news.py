import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd
import requests

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e2b"
PROMPT_TEMPLATE = """Rewrite the following financial news headline in 3 different ways.
Return ONLY a JSON list of 3 strings. Do not include any other text.
Keep the factual meaning and sentiment identical.

Headline: "{headline}"
"""

def generate_variations(row: dict) -> list[dict]:
    """Ask Ollama to generate 3 variations of the headline."""
    headline = row["headline"]
    prompt = PROMPT_TEMPLATE.format(headline=headline)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 200
        },
        "format": "json"
    }

    variations = []
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            response_text = data.get("response", "").strip()

            # Parse the output as JSON
            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, list):
                    lines = parsed
                elif isinstance(parsed, dict):
                    lines = next((v for v in parsed.values() if isinstance(v, list)), [])
                else:
                    lines = []
            except json.JSONDecodeError:
                lines = []

            # We only keep up to 3 variations
            for line in lines[:3]:
                if line.lower() != headline.lower() and len(line) > 5:
                    new_row = row.copy()
                    new_row["headline"] = line
                    variations.append(new_row)
    except Exception as e:
        print(f"Error generating for '{headline}': {e}")

    return variations


def augment_dataset(df: pd.DataFrame, max_concurrent: int = 10) -> pd.DataFrame:
    rows = df.to_dict(orient="records")
    augmented_rows = []

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(generate_variations, row): row for row in rows}
        for i, f in enumerate(as_completed(futures)):
            result = f.result()
            augmented_rows.extend(result)
            if i % 10 == 0:
                print(f"Processed {i}/{len(rows)} headlines...", flush=True)

    return pd.DataFrame(augmented_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2020-02-01", help="Start date")
    parser.add_argument("--end", type=str, default="2020-05-01", help="End date")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process for testing")
    parser.add_argument("--concurrency", type=int, default=10, help="Ollama API concurrency")
    parser.add_argument("--out", type=str, default="data/raw/news/historical_news_augmented.csv")
    args = parser.parse_args()

    news_file = Path("data/raw/news/historical_news_combined.parquet")
    if not news_file.exists():
        print(f"Error: {news_file} not found.")
        return

    print(f"Loading headlines from {args.start} to {args.end}...", flush=True)
    con = duckdb.connect()

    query = f"""
        SELECT * FROM read_parquet('{news_file!s}')
        WHERE timestamp_utc >= '{args.start}' AND timestamp_utc <= '{args.end}'
        AND headline IS NOT NULL
    """
    if args.limit:
        query += f" LIMIT {args.limit}"

    df = con.execute(query).df()
    con.close()

    print(f"Loaded {len(df):,} headlines. Starting augmentation with {MODEL_NAME}...", flush=True)

    augmented_df = augment_dataset(df, max_concurrent=args.concurrency)

    if len(augmented_df) == 0:
        print("No variations generated. Is Ollama running?")
        return

    print(f"Generated {len(augmented_df):,} synthetic variations.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        augmented_df.to_csv(out_path, mode='a', header=False, index=False)
        print(f"Appended to {out_path}")
    else:
        augmented_df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

    # Print a few examples
    print("\nSample variations:")
    for _, row in augmented_df.head(5).iterrows():
        print(f" - {row['headline']}")

if __name__ == "__main__":
    main()
