import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add current directory to path so we can import from data.cross_asset
sys.path.append(str(Path(".").resolve()))

from data.cross_asset import load_cross_asset_panel

cache_dir = Path("data/raw")
cache_dir.mkdir(parents=True, exist_ok=True)

# Fetch from 2008 to today
start = "2008-01-01"
end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

print(f"Downloading cross asset data from {start} to {end}...")
data = load_cross_asset_panel(start, end, str(cache_dir), source="auto")
print(f"Downloaded {len(data)} assets.")
