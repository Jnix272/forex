import urllib.request
import urllib.parse
import json

sql = "SELECT pair, count(*) as count, min(timestamp) as min_ts, max(timestamp) as max_ts, avg(spread_pips) as avg_spread, max(spread_pips) as max_spread FROM live_ticks WHERE strftime(timestamp, '%Y-%m-%d') = '2026-09-24' GROUP BY pair ORDER BY pair"
url = f"http://127.0.0.1:8002/query?sql={urllib.parse.quote(sql)}"
with urllib.request.urlopen(url, timeout=30) as r:
    data = json.loads(r.read())
print(json.dumps(data, indent=2))
