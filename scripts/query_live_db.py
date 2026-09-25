"""
scripts/query_live_db.py
========================
CLI tool to inspect and query live trading telemetry from DuckDB.
Seamlessly queries the running live daemon via its local telemetry API (port 8002),
or queries the local data/store/live_trading.duckdb file directly if the daemon is stopped.

Usage:
  python scripts/query_live_db.py
  python scripts/query_live_db.py --sql "SELECT pair, count(*) FROM live_ticks GROUP BY pair"
  python scripts/query_live_db.py --ticks 20
"""

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
import duckdb


def fetch_from_http(url: str, timeout: float = 1.5) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ForexLiveDBQuery/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read().decode("utf-8"))
        except Exception:
            return {"status": "error", "error": f"HTTP {exc.code}: {exc.reason}"}
    except Exception:
        return None


def print_summary(data: dict[str, Any], source: str) -> None:
    print(f"\n{'=' * 65}")
    print(f" LIVE DUCKDB TELEMETRY SUMMARY ({source})")
    print(f"{'=' * 65}")
    print(f" Total Live Ticks Captured : {data.get('ticks_count', 0):,}")
    print(f" Total Live Bars Completed : {data.get('bars_count', 0):,}")
    print(f" Total Trade / Guard Events: {data.get('trades_count', 0):,}")
    print(f"{'=' * 65}\n")

    # Latest Ticks
    latest_ticks = data.get("latest_ticks") or []
    if latest_ticks:
        print(f"--- LATEST TICKS ({len(latest_ticks)}) ---")
        for t in latest_ticks:
            ts = str(t[0])[:19]
            print(f"  {ts} | {t[1]:<6} | Bid: {t[2]:.5f} | Ask: {t[3]:.5f} | Mid: {t[4]:.5f} | Spread: {t[5]:.1f} pips")
        print()

    # Latest Bars
    latest_bars = data.get("latest_bars") or []
    if latest_bars:
        print(f"--- LATEST COMPLETED BARS ({len(latest_bars)}) ---")
        for b in latest_bars:
            ts = str(b[0])[:19]
            act_str = {0: "BUY", 1: "HOLD", 2: "SELL", 3: "CLOSE"}.get(b[4], str(b[4]))
            weights_str = f" | Weights: {b[8]}" if len(b) > 8 and b[8] else ""
            print(f"  {ts} | {b[1]:<6} | Bar #{b[2]} | Close: {b[3]:.5f} | Action: {act_str:<4} | Model: {b[5]} | Lat: {b[7]:.1f}ms{weights_str}")
        print()

    # Latest Trades
    latest_trades = data.get("latest_trades") or []
    if latest_trades:
        print(f"--- LATEST TRADES & GUARD EVENTS ({len(latest_trades)}) ---")
        for tr in latest_trades:
            ts = str(tr[0])[:19]
            print(f"  {ts} | {tr[1]:<6} | Event: {tr[2]} | Action: {tr[3]} | Lots: {tr[4]} | Price: {tr[5]} | Reason: {tr[6]}")
        print()


def query_file_direct(db_path: Path, sql: str | None = None) -> None:
    if not db_path.exists():
        print(f"[LiveDB] Database {db_path} does not exist.")
        return

    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except Exception as exc:
        print(f"[LiveDB] Could not open {db_path} directly: {exc}")
        print("Hint: If the live trading engine is running, use HTTP query on port 8002.")
        return

    try:
        if sql:
            cur = conn.cursor()
            res = cur.execute(sql)
            cols = [desc[0] for desc in res.description] if res.description else []
            rows = res.fetchall()
            print(f"\n--- SQL Query: {sql} ---")
            print(" | ".join(cols))
            print("-" * 50)
            for r in rows[:50]:
                print(" | ".join(str(v) for v in r))
            print(f"Total rows: {len(rows)}\n")
        else:
            tick_count = conn.execute("SELECT count(*) FROM live_ticks").fetchone()[0]
            bar_count = conn.execute("SELECT count(*) FROM live_bars").fetchone()[0]
            trade_count = conn.execute("SELECT count(*) FROM live_trades").fetchone()[0]
            latest_ticks = conn.execute(
                "SELECT timestamp, pair, bid, ask, mid, spread_pips FROM live_ticks ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()
            latest_bars = conn.execute(
                "SELECT timestamp, pair, bar_idx, close, action, model, sentiment, latency_ms FROM live_bars ORDER BY timestamp DESC LIMIT 5"
            ).fetchall()
            latest_trades = conn.execute(
                "SELECT timestamp, pair, event, action, lots, price, reason FROM live_trades ORDER BY timestamp DESC LIMIT 5"
            ).fetchall()
            summary = {
                "ticks_count": tick_count,
                "bars_count": bar_count,
                "trades_count": trade_count,
                "latest_ticks": latest_ticks,
                "latest_bars": latest_bars,
                "latest_trades": latest_trades,
            }
            print_summary(summary, f"Direct DuckDB File: {db_path}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Query Live DuckDB Telemetry")
    parser.add_argument("--db-path", default="data/store/live_trading.duckdb", help="DuckDB file path")
    parser.add_argument("--port", type=int, default=8002, help="HTTP telemetry port (default: 8002)")
    parser.add_argument("--sql", default=None, help="Execute custom SELECT SQL query")
    parser.add_argument("--file-only", action="store_true", default=False, help="Force direct file read (bypass HTTP)")
    args = parser.parse_args()

    http_base = f"http://127.0.0.1:{args.port}"
    db_file = Path(args.db_path)

    if not args.file_only:
        # Try live HTTP API first
        if args.sql:
            encoded_sql = urllib.parse.quote(args.sql)
            res = fetch_from_http(f"{http_base}/query?sql={encoded_sql}")
            if res:
                if res.get("status") == "ok":
                    cols = res.get("columns", [])
                    rows = res.get("rows", [])
                    print(f"\n--- SQL Query (Live Engine): {args.sql} ---")
                    print(" | ".join(cols))
                    print("-" * 50)
                    for r in rows[:50]:
                        print(" | ".join(str(v) for v in r))
                    print(f"Total rows: {len(rows)}\n")
                    return
                elif res.get("status") == "error":
                    print(f"\n[LiveDB Error] Query rejected by live engine: {res.get('error')}\n")
                    return
        else:
            summary = fetch_from_http(f"{http_base}/summary")
            if summary:
                if summary.get("status") == "ok":
                    print_summary(summary, f"Live Engine at {http_base}")
                    return
                elif summary.get("status") == "error":
                    print(f"\n[LiveDB Error] {summary.get('error')}\n")
                    return

    # Fallback to direct file reading
    query_file_direct(db_file, sql=args.sql)


if __name__ == "__main__":
    main()
