from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def _open_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="r+")
    except TypeError:
        return zarr.open(str(path), mode="r+")


def _sanitize_array(arr, *, chunk_size: int) -> int:
    changed = 0
    n = int(arr.shape[0])
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        vals = np.asarray(arr[start:stop], dtype=np.float32)
        bad = ~np.isfinite(vals)
        bad_count = int(bad.sum())
        if bad_count:
            vals[bad] = 0.0
            arr[start:stop] = vals
            changed += bad_count
    return changed


def sanitize_cache(root: Path, *, chunk_size: int) -> int:
    total_changed = 0

    for zarr_path in sorted(root.glob("*.zarr")):
        group = _open_group(zarr_path)
        if "y" not in group:
            continue
        changed = _sanitize_array(group["y"], chunk_size=chunk_size)
        print(f"{zarr_path}: sanitized {changed:,} label value(s)")
        total_changed += changed

    for npy_path in sorted(root.glob("*_y.npy")):
        arr = np.load(npy_path, mmap_mode="r+")
        changed = _sanitize_array(arr, chunk_size=chunk_size)
        print(f"{npy_path}: sanitized {changed:,} label value(s)")
        total_changed += changed

    return total_changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace NaN/Inf cached labels with 0.0.")
    parser.add_argument("--data-cache", default="data/processed", help="Directory containing .zarr or *_y.npy caches")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    root = Path(args.data_cache).expanduser()
    if not root.exists():
        raise SystemExit(f"data cache not found: {root}")

    changed = sanitize_cache(root, chunk_size=max(1, args.chunk_size))
    print(f"Done. Total sanitized label value(s): {changed:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
