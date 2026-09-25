import concurrent.futures
import json
import time
import urllib.parse
import urllib.request

QUERIES = {
    "ticks": "SELECT pair, count(*) as n_ticks, min(timestamp) as min_ts, max(timestamp) as max_ts, avg(spread_pips) as avg_spread, max(spread_pips) as max_spread, min(spread_pips) as min_spread, count(CASE WHEN bid IS NULL OR ask IS NULL OR mid IS NULL THEN 1 END) as null_ticks FROM live_ticks GROUP BY pair",
    "bars_summary": "SELECT pair, count(*) as n_bars, min(bar_idx) as min_idx, max(bar_idx) as max_idx, min(timestamp) as min_ts, max(timestamp) as max_ts, avg(latency_ms) as avg_lat, min(latency_ms) as min_lat, max(latency_ms) as max_lat, count(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL THEN 1 END) as null_bars, count(CASE WHEN action IS NULL THEN 1 END) as null_actions FROM live_bars GROUP BY pair",
    "action_distribution": "SELECT pair, action, model, count(*) as cnt, avg(latency_ms) as avg_lat FROM live_bars GROUP BY pair, action, model ORDER BY pair, action",
    "latency_quantiles": "SELECT pair, quantile_cont(latency_ms, 0.50) as p50, quantile_cont(latency_ms, 0.90) as p90, quantile_cont(latency_ms, 0.95) as p95, quantile_cont(latency_ms, 0.99) as p99 FROM live_bars GROUP BY pair",
    "trades_summary": "SELECT pair, event, reason, count(*) as count, min(timestamp) as first_ts, max(timestamp) as last_ts FROM live_trades GROUP BY pair, event, reason ORDER BY first_ts",
    "all_trades": "SELECT timestamp, pair, event, action, lots, price, order_id, reason, details FROM live_trades ORDER BY timestamp ASC",
    "all_bars": "SELECT timestamp, pair, bar_idx, latency_ms, close, action, model, model_weights FROM live_bars ORDER BY pair, timestamp ASC",
    "weights_distinct": "SELECT pair, model_weights, count(*) as cnt FROM live_bars GROUP BY pair, model_weights"
}

def run_single_query(name, sql):
    url = f"http://127.0.0.1:8002/query?sql={urllib.parse.quote(sql)}"
    req = urllib.request.Request(url, headers={"User-Agent": "AuditScript/1.0"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return name, True, data, time.time() - t0
    except Exception as exc:
        return name, False, str(exc), time.time() - t0

def main():
    print(f"Starting parallel execution of {len(QUERIES)} audit queries...")
    t_start = time.time()
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(QUERIES)) as executor:
        futures = [executor.submit(run_single_query, name, sql) for name, sql in QUERIES.items()]
        for future in concurrent.futures.as_completed(futures):
            name, ok, data, elapsed = future.result()
            print(f"[{elapsed:.2f}s] Query '{name}' finished (ok={ok})")
            results[name] = data if ok else {"error": data}
            
    print(f"All queries finished in {time.time()-t_start:.2f}s")
    
    with open("scripts/audit_results_today.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved results to scripts/audit_results_today.json")

if __name__ == "__main__":
    main()
