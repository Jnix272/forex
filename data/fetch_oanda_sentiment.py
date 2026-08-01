import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OANDA_API_TOKEN = os.getenv("OANDA_API_TOKEN", "").strip()
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "").strip()
OANDA_BASE_URL = "https://api-fxpractice.oanda.com/v3"

SIDECAR_DIR = Path("data/oanda_sentiment")
SIDECAR_DIR.mkdir(parents=True, exist_ok=True)

def _get_headers():
    return {
        "Authorization": f"Bearer {OANDA_API_TOKEN}",
        "Accept-Datetime-Format": "RFC3339"
    }

def fetch_snapshot(endpoint_type, instrument="EUR_USD", mock=False, allow_mock=True):
    """
    Fetches the raw snapshot for orderBook or positionBook.
    Returns (raw_json, is_mock).
    """
    if mock or not OANDA_API_TOKEN:
        if not allow_mock:
            raise RuntimeError(f"OANDA {endpoint_type} unavailable: missing token and mock fallback disabled")
        if endpoint_type == "orderBook":
            return _generate_mock_order_book(instrument), True
        else:
            return _generate_mock_position_book(instrument), True

    url = f"{OANDA_BASE_URL}/instruments/{instrument}/{endpoint_type}"
    resp = requests.get(url, headers=_get_headers())
    if resp.status_code == 200:
        return resp.json().get(endpoint_type, {}), False
    else:
        print(f"[OANDA API] Failed {endpoint_type} for {instrument}: {resp.status_code}")
        if not allow_mock:
            raise RuntimeError(f"OANDA {endpoint_type} request failed with status {resp.status_code}")
        if endpoint_type == "orderBook":
            return _generate_mock_order_book(instrument), True
        else:
            return _generate_mock_position_book(instrument), True

def _generate_mock_order_book(instrument):
    price = 1.0850
    bids = [{"price": str(price - (i*0.0010)), "bucketWidth": "0.0010", "longCountPercent": str(np.random.uniform(0.1, 2.0)), "shortCountPercent": str(np.random.uniform(0.1, 2.0))} for i in range(1, 20)]
    asks = [{"price": str(price + (i*0.0010)), "bucketWidth": "0.0010", "longCountPercent": str(np.random.uniform(0.1, 2.0)), "shortCountPercent": str(np.random.uniform(0.1, 2.0))} for i in range(1, 20)]
    bids[5]["shortCountPercent"] = "15.0"
    asks[8]["longCountPercent"] = "12.0"
    return {"instrument": instrument, "time": datetime.now(timezone.utc).isoformat() + "Z", "price": str(price), "bucketWidth": "0.0010", "bids": bids, "asks": asks}

def _generate_mock_position_book(instrument):
    price = 1.0850
    return {
        "instrument": instrument, 
        "time": datetime.now(timezone.utc).isoformat() + "Z", 
        "price": str(price), 
        "bucketWidth": "0.0010",
        "bids": [{"price": str(price - (i*0.0010)), "longCountPercent": str(np.random.uniform(0.1, 2.0)), "shortCountPercent": str(np.random.uniform(0.1, 2.0))} for i in range(1, 20)],
        "asks": [{"price": str(price + (i*0.0010)), "longCountPercent": str(np.random.uniform(0.1, 2.0)), "shortCountPercent": str(np.random.uniform(0.1, 2.0))} for i in range(1, 20)]
    }

def engineer_features(ob, pb, instrument, is_mock):
    """
    Engineers current timestep features without history (z-scores happen later in the pipeline).
    """
    current_price = float(ob.get("price", 0.0))
    if current_price == 0.0:
        return None

    long_pct_total = sum(float(b.get("longCountPercent", 0.0)) for b in pb.get("bids", []) + pb.get("asks", []))
    short_pct_total = sum(float(b.get("shortCountPercent", 0.0)) for b in pb.get("bids", []) + pb.get("asks", []))
    total_pos = long_pct_total + short_pct_total
    retail_long_ratio = (long_pct_total / total_pos) if total_pos > 0 else 0.5

    max_sell_stop_pct = 0.0
    sell_stop_price = current_price
    for bucket in ob.get("bids", []):
        pct = float(bucket.get("shortCountPercent", 0.0))
        if pct > max_sell_stop_pct:
            max_sell_stop_pct = pct
            sell_stop_price = float(bucket.get("price", current_price))
            
    max_buy_stop_pct = 0.0
    buy_stop_price = current_price
    for bucket in ob.get("asks", []):
        pct = float(bucket.get("longCountPercent", 0.0))
        if pct > max_buy_stop_pct:
            max_buy_stop_pct = pct
            buy_stop_price = float(bucket.get("price", current_price))

    stop_loss_cluster_dist_long = (current_price - sell_stop_price) * 10000.0
    stop_loss_cluster_dist_short = (buy_stop_price - current_price) * 10000.0

    nearest_side = 1.0 if stop_loss_cluster_dist_long < stop_loss_cluster_dist_short else -1.0
    nearest_dist = min(stop_loss_cluster_dist_long, stop_loss_cluster_dist_short)

    pending_buy_limits = sum(float(b.get("longCountPercent", 0.0)) for b in ob.get("bids", []))
    pending_sell_limits = sum(float(b.get("shortCountPercent", 0.0)) for b in ob.get("asks", []))
    total_limits = pending_buy_limits + pending_sell_limits
    order_imbalance = (pending_buy_limits / total_limits) if total_limits > 0 else 0.5

    return {
        "timestamp": ob.get("time", datetime.now(timezone.utc).isoformat()),
        "instrument": instrument,
        "is_mock": is_mock,
        "source_status": "mock" if is_mock else "ok",
        "retail_long_ratio": retail_long_ratio,
        "stop_loss_cluster_dist_long": stop_loss_cluster_dist_long,
        "stop_loss_cluster_dist_short": stop_loss_cluster_dist_short,
        "nearest_stop_cluster_side": nearest_side,
        "nearest_stop_cluster_distance_pips": nearest_dist,
        "order_imbalance": order_imbalance,
        # Save raw snapshots as JSON strings for later inspection if needed
        "raw_order_book": json.dumps(ob),
        "raw_position_book": json.dumps(pb)
    }

def run_collector(instrument="EUR_USD", mock=False, allow_mock_write=False):
    """
    Fetches data, engineers base features, and appends to the daily parquet sidecar.
    """
    if "_" not in instrument and len(instrument) == 6:
        instrument = f"{instrument[:3]}_{instrument[3:]}"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Collecting {instrument} sidecar snapshot...")
    allow_mock = bool(mock or allow_mock_write)
    ob_raw, ob_mock = fetch_snapshot("orderBook", instrument, mock=mock, allow_mock=allow_mock)
    pb_raw, pb_mock = fetch_snapshot("positionBook", instrument, mock=mock, allow_mock=allow_mock)
    
    is_mock = ob_mock or pb_mock
    if is_mock and not allow_mock_write:
        raise RuntimeError(
            "OANDA collector received mock data. Refusing to write sidecar unless "
            "allow_mock_write=True is explicitly set."
        )
    features = engineer_features(ob_raw, pb_raw, instrument, is_mock)
    
    if not features:
        return None

    df = pd.DataFrame([features])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    date_str = df["timestamp"].iloc[0].strftime("%Y-%m-%d")
    parquet_path = SIDECAR_DIR / f"{instrument}_{date_str}.parquet"

    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=["timestamp"], keep="last")
        combined_df.to_parquet(parquet_path, index=False)
    else:
        df.to_parquet(parquet_path, index=False)
        
    print(f"   -> Saved snapshot to {parquet_path} (is_mock={is_mock})")
    return features

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect OANDA order/position book sidecar snapshots.")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--mock", action="store_true", help="Use generated mock data.")
    parser.add_argument(
        "--allow-mock-write",
        action="store_true",
        help="Allow mock snapshots to be written. Never use for production training data.",
    )
    args = parser.parse_args()
    run_collector(args.instrument, mock=args.mock, allow_mock_write=args.allow_mock_write)
