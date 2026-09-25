"""
Audit script for live engine on 2026-09-24.
Queries the local telemetry server on port 8002 and inspects system logs and metrics.
"""
import urllib.request
import urllib.parse
import json
import sys
from datetime import datetime, timezone

def query_duckdb(sql: str) -> dict:
    url = f"http://127.0.0.1:8002/query?sql={urllib.parse.quote(sql)}"
    req = urllib.request.Request(url, headers={"User-Agent": "LiveAudit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"status": "error", "error": str(e)}

def fetch_summary() -> dict:
    url = "http://127.0.0.1:8002/summary"
    req = urllib.request.Request(url, headers={"User-Agent": "LiveAudit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"status": "error", "error": str(e)}

def run_audit():
    print("=== LIVE ENGINE TELEMETRY AUDIT (2026-09-24) ===")
    
    # 0. Summary
    summ = fetch_summary()
    print("Summary:", json.dumps({k: v for k, v in summ.items() if not k.startswith("latest_")}, indent=2))
    
    # 1. Tick Ingestion per pair
    print("\n--- 1. TICK INGESTION (live_ticks) ---")
    res = query_duckdb("SELECT pair, count(*) as n_ticks, min(timestamp) as min_ts, max(timestamp) as max_ts, avg(spread_pips) as avg_spread, max(spread_pips) as max_spread, min(spread_pips) as min_spread FROM live_ticks WHERE strftime(timestamp, '%Y-%m-%d') = '2026-09-24' GROUP BY pair ORDER BY pair")
    print("Ticks per pair:", res)

    # Check null ticks
    res = query_duckdb("SELECT count(*) as null_ticks FROM live_ticks WHERE bid IS NULL OR ask IS NULL OR mid IS NULL")
    print("Null ticks:", res)
    
    # 2. Bar Ingestion per pair
    print("\n--- 2. BAR COMPLETION & CONTINUITY (live_bars) ---")
    res = query_duckdb("SELECT pair, count(*) as n_bars, min(bar_idx) as min_idx, max(bar_idx) as max_idx, min(timestamp) as min_ts, max(timestamp) as max_ts FROM live_bars WHERE strftime(timestamp, '%Y-%m-%d') = '2026-09-24' GROUP BY pair ORDER BY pair")
    print("Bars per pair:", res)
    
    # Check null prices in bars
    res = query_duckdb("SELECT count(*) as null_price_bars FROM live_bars WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL")
    print("Null price bars:", res)
    
    # 3. Model execution & Latency
    print("\n--- 3. MODEL INFERENCE & LATENCY ---")
    res = query_duckdb("SELECT pair, count(*) as bars, avg(latency_ms) as avg_lat, min(latency_ms) as min_lat, max(latency_ms) as max_lat, quantile_cont(latency_ms, 0.5) as p50, quantile_cont(latency_ms, 0.95) as p95, quantile_cont(latency_ms, 0.99) as p99 FROM live_bars WHERE strftime(timestamp, '%Y-%m-%d') = '2026-09-24' GROUP BY pair ORDER BY pair")
    print("Latency stats:", res)
    
    res = query_duckdb("SELECT pair, action, count(*) as count FROM live_bars WHERE strftime(timestamp, '%Y-%m-%d') = '2026-09-24' GROUP BY pair, action ORDER BY pair, action")
    print("Action distribution:", res)
    
    res = query_duckdb("SELECT DISTINCT model FROM live_bars WHERE strftime(timestamp, '%Y-%m-%d') = '2026-09-24'")
    print("Distinct models:", res)
    
    res = query_duckdb("SELECT model_weights, count(*) as count FROM live_bars WHERE strftime(timestamp, '%Y-%m-%d') = '2026-09-24' GROUP BY model_weights ORDER BY count DESC LIMIT 10")
    print("Model weights samples:", res)

    # Check for NaN / null in actions or latency
    res = query_duckdb("SELECT count(*) FROM live_bars WHERE action IS NULL OR latency_ms IS NULL OR isnan(latency_ms) OR isnan(close)")
    print("NaN or NULL bars:", res)
    
    # 4. Trades & Guard Blocks
    print("\n--- 4. TRADES & GUARD EVENTS (live_trades) ---")
    res = query_duckdb("SELECT event, reason, count(*) as count FROM live_trades WHERE strftime(timestamp, '%Y-%m-%d') = '2026-09-24' GROUP BY event, reason ORDER BY count DESC")
    print("Events and reasons:", res)
    
    res = query_duckdb("SELECT timestamp, pair, event, action, lots, price, order_id, reason, details FROM live_trades WHERE strftime(timestamp, '%Y-%m-%d') = '2026-09-24' ORDER BY timestamp ASC")
    print(f"Total trade events today: {len(res.get('rows', []))}")
    for row in res.get('rows', []):
        print("  TRADE ROW:", row)
        
    # Check FIFO violations
    res = query_duckdb("SELECT count(*) as fifo_violations FROM live_trades WHERE lower(reason) LIKE '%fifo%' OR lower(details) LIKE '%fifo%'")
    print("FIFO violation count:", res)

if __name__ == "__main__":
    run_audit()
