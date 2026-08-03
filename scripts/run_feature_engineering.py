import os
import sys
import yaml
import polars as pl
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from features.feature_engineering_pl import FeatureEngineer

def load_config(config_path="config/run_normal.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_pipeline():
    config = load_config()
    pairs = config["data"]["pairs"]
    
    macro_path = config["news"].get("economic_calendar_file")
    news_path = config["news"].get("historical_news_file")
    
    print("Loading external data...")
    # Load COT (we just pass the path to FeatureEngineer or load it)
    cot_path = "data/raw/cot/cot_financials_cleaned.parquet"
    if os.path.exists(cot_path):
        cot_df = pl.read_parquet(cot_path)
    else:
        cot_df = None
        
    if macro_path and os.path.exists(macro_path):
        macro_df = pl.read_csv(macro_path, ignore_errors=True, try_parse_dates=True)
        if "timestamp_utc" in macro_df.columns:
            try:
                # If it's loaded as String due to complex format, force conversion first
                if macro_df.schema["timestamp_utc"] == pl.String:
                    macro_df = macro_df.with_columns(pl.col("timestamp_utc").str.to_datetime(time_unit="ns", time_zone="UTC"))
                else:
                    macro_df = macro_df.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
            except Exception as e:
                print("Failed to convert macro timestamp:", e)
    else:
        macro_df = None
        
    if news_path and os.path.exists(news_path):
        news_path_obj = Path(news_path)
        if news_path_obj.suffix.lower() == ".parquet":
            news_df = pl.read_parquet(news_path_obj)
        else:
            news_df = pl.read_csv(news_path_obj, ignore_errors=True, try_parse_dates=True)
        if "timestamp_utc" in news_df.columns:
            try:
                if news_df.schema["timestamp_utc"] == pl.String:
                    news_df = news_df.with_columns(pl.col("timestamp_utc").str.to_datetime(time_unit="ns", time_zone="UTC"))
                else:
                    news_df = news_df.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
            except Exception as e:
                print("Failed to convert news timestamp:", e)
    else:
        news_df = None

    print("Loading OANDA data...")
    oanda_sidecar_dir = Path("data/oanda_sentiment")
    oanda_files = list(oanda_sidecar_dir.glob("*.parquet")) if oanda_sidecar_dir.exists() else []
    oanda_path = "data/raw/oanda_sentiment.csv"
    if oanda_files:
        oanda_master_df = pl.concat([pl.read_parquet(path) for path in oanda_files])
    elif os.path.exists(oanda_path):
        oanda_master_df = pl.read_csv(oanda_path, ignore_errors=True)
    else:
        oanda_master_df = None

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        print(f"\n=========================================")
        print(f"Processing {pair}...")
        print(f"=========================================")
        
        # Load raw tick data using glob
        glob_path = f"data/raw/dukascopy/{pair}/*/*/*.parquet"
        print(f"Scanning tick data from {glob_path}...")
        
        try:
            # We use scan_parquet to lazily load all tick data across the years
            ticks = pl.scan_parquet(glob_path)
            
            # Fix schema from raw Dukascopy Pandas Parquets
            ticks = ticks.rename({
                "__index_level_0__": "timestamp_utc",
                "bid": "bid_price",
                "ask": "ask_price"
            })
            
            # Resample ticks into 1-minute OHLCV bars...
            print("Resampling ticks into 1-minute OHLCV bars...")
            
            # Ensure timestamp is datetime and sorted
            # If the index was an integer, we might need to cast. Let's assume it's already datetime or parseable.
            ticks = ticks.sort("timestamp_utc")

            # Resample
            # Dukascopy schema is usually: timestamp_utc, ask_price, ask_volume, bid_price, bid_volume
            bars = ticks.group_by_dynamic("timestamp_utc", every="1m").agg([
                pl.col("bid_price").first().alias("open"),
                pl.col("bid_price").max().alias("high"),
                pl.col("bid_price").min().alias("low"),
                pl.col("bid_price").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("ask_price").last().alias("ask_close"),
                pl.col("bid_price").last().alias("bid_close"),
            ]).collect(engine="streaming")
            
            # Fill forward any missing values if liquidity was 0 for a minute
            bars = bars.fill_null(strategy="forward")
            
            print(f"Generated {len(bars):,} 1-minute bars for {pair}.")
            
            # Map datasets to match FeatureEngineer.build() expectations
            
            # Cross asset dict
            cross_asset_dict = {}
            if cot_df is not None:
                cross_asset_dict["cot_net_hf"] = cot_df
                
            # For macro/eco calendar, FeatureEngineer expects eco_act and eco_fc, but since macro_df has both actual and forecast, we just pass it to both to let it merge.
            eco_act = macro_df
            eco_fc = macro_df
            
            # For news, FeatureEngineer expects `sentiment` DataFrame with 'sentiment' column
            sentiment_df = None
            if news_df is not None:
                if "sentiment_string" in news_df.columns:
                    # Simple mapping for string to float
                    sentiment_df = news_df.with_columns(
                        pl.when(pl.col("sentiment_string") == "positive").then(1.0)
                        .when(pl.col("sentiment_string") == "negative").then(-1.0)
                        .otherwise(0.0).alias("sentiment")
                    )
                else:
                    sentiment_df = news_df

            # Filter OANDA data
            oanda_df = None
            if oanda_master_df is not None:
                # Convert standard pair (e.g., EURUSD) to OANDA format (EUR_USD)
                oanda_pair = f"{pair[:3]}_{pair[3:]}"
                if "instrument" in oanda_master_df.columns:
                    oanda_df = oanda_master_df.filter(pl.col("instrument") == oanda_pair)

            # Run Feature Engineering
            print("Running Feature Engineering Pipeline...")
            engineer = FeatureEngineer()
            
            processed = engineer.build_chunked(
                bars=bars,
                cross_asset=cross_asset_dict,
                sentiment=sentiment_df,
                eco_act=eco_act,
                eco_fc=eco_fc,
                oanda_data=oanda_df,
                chunk_size=50_000
            )
            
            # Save the result
            out_file = out_dir / f"{pair}_features.parquet"
            processed.write_parquet(out_file)
            print(f"Successfully saved feature dataset to {out_file}")
            
        except Exception as e:
            print(f"Error processing {pair}: {e}")

if __name__ == "__main__":
    run_pipeline()
