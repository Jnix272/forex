import json
from pathlib import Path

import pandas as pd

MIN_PASS_SNAPSHOTS = 60
MAX_PASS_CADENCE_MINUTES = 10.0
MAX_WARN_CADENCE_MINUTES = 60.0

def generate_report(sidecar_dir="data/oanda_sentiment", output_json="data/oanda_sentiment_report.json"):
    dir_path = Path(sidecar_dir)
    if not dir_path.exists():
        print(f"Directory {sidecar_dir} does not exist.")
        return

    # Group parquets by instrument
    files = list(dir_path.glob("*.parquet"))
    pairs = {}
    for f in files:
        # e.g., EUR_USD_2026-06-19.parquet
        name_parts = f.stem.split("_")
        if len(name_parts) >= 3:
            pair = f"{name_parts[0]}_{name_parts[1]}"
            if pair not in pairs:
                pairs[pair] = []
            pairs[pair].append(f)

    report = {
        "summary": {
            "total_pairs_covered": len(pairs),
            "generated_at": pd.Timestamp.utcnow().isoformat()
        },
        "pairs": {}
    }

    for pair, file_list in pairs.items():
        dfs = [pd.read_parquet(f) for f in file_list]
        df = pd.concat(dfs).sort_values("timestamp")

        # Calculate stats
        total_rows = len(df)
        min_time = df["timestamp"].min()
        max_time = df["timestamp"].max()

        # Freshness / Cadence
        time_diffs = df["timestamp"].diff().dt.total_seconds() / 60.0
        avg_cadence_mins = time_diffs.dropna().mean()
        avg_cadence_value = None if pd.isna(avg_cadence_mins) else round(float(avg_cadence_mins), 2)

        # Missing minutes (assuming 1-min expected cadence)
        expected_minutes = ((max_time - min_time).total_seconds() / 60.0) + 1 if pd.notnull(max_time) else 0
        missing_minutes = max(0, expected_minutes - total_rows)

        # Stale snapshots (no change in data)
        # We can approximate staleness by looking for duplicate feature vectors
        features = ["retail_long_ratio", "order_imbalance"]
        stale_count = df.duplicated(subset=features).sum()

        # Mock data tracking
        mock_count = df["is_mock"].sum() if "is_mock" in df.columns else 0

        # Null rates
        null_rates = (df.isnull().sum() / total_rows).to_dict() if total_rows > 0 else {}

        # Status calculation
        status = "PASS"
        reasons = []
        if mock_count > 0:
            status = "FAIL (Contains Mock Data)"
            reasons.append("contains_mock_data")
        elif total_rows < 2:
            status = "FAIL (Insufficient Snapshots)"
            reasons.append("insufficient_snapshots")
        elif total_rows < MIN_PASS_SNAPSHOTS:
            status = "WARN (Limited Coverage)"
            reasons.append("limited_coverage")
        elif null_rates.get("retail_long_ratio", 0) > 0.05:
            status = "WARN (High Null Rate)"
            reasons.append("high_null_rate")
        elif avg_cadence_value is not None and avg_cadence_value > MAX_WARN_CADENCE_MINUTES:
            status = "FAIL (Low Cadence)"
            reasons.append("low_cadence")
        elif avg_cadence_value is not None and avg_cadence_value > MAX_PASS_CADENCE_MINUTES:
            status = "WARN (Low Cadence)"
            reasons.append("low_cadence")

        report["pairs"][pair] = {
            "status": status,
            "reasons": reasons,
            "timestamp_range": [min_time.isoformat(), max_time.isoformat()] if pd.notnull(min_time) else [],
            "total_snapshots": int(total_rows),
            "avg_update_cadence_minutes": avg_cadence_value,
            "missing_minutes": int(missing_minutes),
            "stale_snapshot_count": int(stale_count),
            "mock_snapshot_count": int(mock_count),
            "null_rates": {k: float(v) for k, v in null_rates.items()}
        }

    with open(output_json, "w") as f:
        json.dump(report, f, indent=4, allow_nan=False)

    print(f"Generated report at {output_json}")
    for p, stats in report["pairs"].items():
        print(f"  [{p}] Status: {stats['status']} | Snapshots: {stats['total_snapshots']} | Mock: {stats['mock_snapshot_count']}")

if __name__ == "__main__":
    generate_report()
