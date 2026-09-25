"""
Detailed analysis script for today (2026-09-24) using scripts/audit_results_today.json
"""
import json
from datetime import datetime, timezone
import dateutil.parser

with open('scripts/audit_results_today.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

# 1. Analyze Bars Today (2026-09-24)
bars_cols = d['all_bars']['columns']
bars_rows = d['all_bars']['rows']

# Columns: ['timestamp', 'pair', 'bar_idx', 'latency_ms', 'close', 'action', 'model', 'model_weights']
col_map = {c: i for i, c in enumerate(bars_cols)}

bars_today = []
bars_yesterday = []

for r in bars_rows:
    ts_str = str(r[col_map['timestamp']])
    # Check if ts is on 2026-09-24
    if '2026-09-24' in ts_str:
        bars_today.append(r)
    else:
        bars_yesterday.append(r)

print(f"Total bars in DB: {len(bars_rows)}")
print(f"Bars yesterday (2026-09-23): {len(bars_yesterday)}")
print(f"Bars today (2026-09-24): {len(bars_today)}")

# Group bars today by pair
bars_by_pair = {}
for r in bars_today:
    p = r[col_map['pair']]
    bars_by_pair.setdefault(p, []).append(r)

print("\n--- BARS TODAY PER PAIR ---")
for p, rows in sorted(bars_by_pair.items()):
    ts_list = [dateutil.parser.parse(str(r[col_map['timestamp']])) for r in rows]
    lat_list = [float(r[col_map['latency_ms']]) for r in rows]
    idx_list = [int(r[col_map['bar_idx']]) for r in rows]
    actions = [int(r[col_map['action']]) for r in rows]
    action_counts = {0: actions.count(0), 1: actions.count(1), 2: actions.count(2)}
    
    # Check bar_idx continuity
    idx_gaps = []
    for i in range(1, len(idx_list)):
        # Check if idx increases by 1 or if engine was restarted
        diff = idx_list[i] - idx_list[i-1]
        if diff != 1 and not (idx_list[i] == 0): # 0 is engine restart
            idx_gaps.append((idx_list[i-1], idx_list[i], str(ts_list[i])))
            
    # Check time gaps between consecutive bars (> 310 seconds for 5m bars)
    time_gaps = []
    for i in range(1, len(ts_list)):
        dt = (ts_list[i] - ts_list[i-1]).total_seconds()
        if dt > 360: # more than 6 minutes
            time_gaps.append((dt, str(ts_list[i-1]), str(ts_list[i])))
            
    # Check delays (>10s after 5m boundary)
    # A 5m bar at MM:SS is on schedule if seconds past MM:00 are <= 15s (polling interval ~500ms + bar close)
    delayed_bars = []
    for r, ts in zip(rows, ts_list):
        sec_in_bar = (ts.minute % 5) * 60 + ts.second
        lat = float(r[col_map['latency_ms']])
        if sec_in_bar > 20: # more than 20s past 5m boundary
            delayed_bars.append((sec_in_bar, str(ts), lat))

    print(f"Pair: {p}")
    print(f"  Bars count: {len(rows)}")
    print(f"  Min ts: {ts_list[0]} | Max ts: {ts_list[-1]}")
    print(f"  Actions: {action_counts}")
    print(f"  Latency: mean={sum(lat_list)/len(lat_list):.1f}ms, min={min(lat_list):.1f}ms, max={max(lat_list):.1f}ms, median={sorted(lat_list)[len(lat_list)//2]:.1f}ms")
    print(f"  Index gaps: {len(idx_gaps)} {idx_gaps[:3] if idx_gaps else ''}")
    print(f"  Time gaps (>6m): {len(time_gaps)} {time_gaps if time_gaps else ''}")
    print(f"  Delayed bars (>20s into 5m window): {len(delayed_bars)} {delayed_bars[:3] if delayed_bars else ''}")

# 2. Analyze Trades Today (2026-09-24)
trades_cols = d['all_trades']['columns']
trades_rows = d['all_trades']['rows']
t_col_map = {c: i for i, c in enumerate(trades_cols)}

trades_today = [r for r in trades_rows if '2026-09-24' in str(r[t_col_map['timestamp']])]
trades_yesterday = [r for r in trades_rows if '2026-09-23' in str(r[t_col_map['timestamp']])]

print(f"\n--- TRADES & GUARDS ---")
print(f"Total trades/guard events in DB: {len(trades_rows)}")
print(f"Yesterday events: {len(trades_yesterday)}")
print(f"Today events: {len(trades_today)}")

events_today = {}
for r in trades_today:
    ev = r[t_col_map['event']]
    reason = r[t_col_map['reason']]
    pair = r[t_col_map['pair']]
    key = (ev, reason)
    events_today.setdefault(key, []).append(r)

for (ev, reason), rows in sorted(events_today.items()):
    ts_first = rows[0][t_col_map['timestamp']]
    ts_last = rows[-1][t_col_map['timestamp']]
    pairs = [r[t_col_map['pair']] for r in rows]
    pair_counts = {p: pairs.count(p) for p in set(pairs)}
    print(f"Event: {ev} | Reason: {reason} | Count: {len(rows)} | Pairs: {pair_counts}")
    print(f"  First: {ts_first} | Last: {ts_last}")

# 3. Model Weights Inspection
print("\n--- MODEL WEIGHTS TODAY ---")
weights_seen = {}
for r in bars_today:
    mw = r[col_map['model_weights']]
    p = r[col_map['pair']]
    weights_seen.setdefault(p, set()).add(mw)
for p, w_set in sorted(weights_seen.items()):
    print(f"Pair {p} distinct weights: {w_set}")
