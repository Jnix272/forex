import glob
import os
import datetime

live_logs = glob.glob("logs/live/*")
print(f"Total files in logs/live/: {len(live_logs)}")
today_files = []
for f in live_logs:
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    if mtime.date() == datetime.date(2026, 9, 24):
        today_files.append((mtime, f, os.path.getsize(f)))

today_files.sort(key=lambda x: x[0], reverse=True)
print(f"Files modified today ({len(today_files)}):")
for mtime, f, sz in today_files[:20]:
    print(f"  {mtime.strftime('%Y-%m-%d %H:%M:%S')} | {sz:>10} bytes | {f}")
