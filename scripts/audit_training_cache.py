from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None


ZARR_KEYS = ("X", "y", "y_cls", "pq", "diff", "close", "atr", "spread")
NPY_SUFFIXES = {
    "X": "_X.npy",
    "y": "_y.npy",
    "y_cls": "_y_cls.npy",
    "pq": "_pq.npy",
    "diff": "_diff.npy",
    "close": "_close.npy",
    "atr": "_atr.npy",
    "spread": "_spread.npy",
}


def _open_zarr_group(path: Path):
    if zarr is None:
        raise RuntimeError("zarr is not installed")
    try:
        return zarr.open_group(str(path), mode="r")
    except TypeError:
        return zarr.open(str(path), mode="r")


def _iter_cache_paths(path: Path) -> Iterable[Path]:
    if path.suffix == ".zarr" or path.name.endswith("_X.npy"):
        yield path
        return
    if path.exists() and path.is_dir():
        yield from sorted(path.glob("*.zarr"))
        yield from sorted(path.glob("*_X.npy"))


def _base_from_x_npy(path: Path) -> Path:
    name = path.name
    if not name.endswith("_X.npy"):
        return path
    return path.with_name(name[:-6])


def _scan_array(arr, *, chunk_size: int) -> dict[str, object]:
    shape = tuple(int(v) for v in arr.shape)
    dtype = str(getattr(arr, "dtype", "unknown"))
    total = 0
    nan = 0
    posinf = 0
    neginf = 0
    amin = float("inf")
    amax = float("-inf")

    n = int(arr.shape[0]) if shape else 1
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        vals = np.asarray(arr[start:stop] if shape else arr, dtype=np.float64)
        total += int(vals.size)
        nan += int(np.isnan(vals).sum())
        posinf += int(np.isposinf(vals).sum())
        neginf += int(np.isneginf(vals).sum())
        finite = vals[np.isfinite(vals)]
        if finite.size:
            amin = min(amin, float(finite.min()))
            amax = max(amax, float(finite.max()))

    return {
        "shape": shape,
        "dtype": dtype,
        "values": total,
        "nan": nan,
        "posinf": posinf,
        "neginf": neginf,
        "min": None if amin == float("inf") else amin,
        "max": None if amax == float("-inf") else amax,
    }


def _audit_zarr(path: Path, *, chunk_size: int) -> dict[str, dict[str, object]]:
    group = _open_zarr_group(path)
    return {key: _scan_array(group[key], chunk_size=chunk_size) for key in ZARR_KEYS if key in group}


def _audit_npy(path: Path, *, chunk_size: int) -> dict[str, dict[str, object]]:
    base = _base_from_x_npy(path)
    report = {}
    for key, suffix in NPY_SUFFIXES.items():
        fp = Path(str(base) + suffix)
        if fp.exists():
            report[key] = _scan_array(np.load(fp, mmap_mode="r"), chunk_size=chunk_size)
    return report


def audit_cache(path: Path, *, chunk_size: int) -> dict[str, dict[str, object]]:
    if path.suffix == ".zarr":
        return _audit_zarr(path, chunk_size=chunk_size)
    return _audit_npy(path, chunk_size=chunk_size)


def _print_report(path: Path, report: dict[str, dict[str, object]]) -> int:
    print(f"\nCache: {path}")
    if not report:
        print("  no auditable arrays found")
        return 1

    bad_total = 0
    for key, stats in report.items():
        bad = int(stats["nan"]) + int(stats["posinf"]) + int(stats["neginf"])
        bad_total += bad
        print(
            f"  {key:<7} shape={stats['shape']} dtype={stats['dtype']} "
            f"nan={stats['nan']:,} +inf={stats['posinf']:,} -inf={stats['neginf']:,} "
            f"min={stats['min']} max={stats['max']}"
        )
    print(f"  nonfinite_total={bad_total:,}")

    # Schema sidecar vs X feature width (TPA-K03 / curriculum freeze contract).
    schema_code = 0
    try:
        import json

        base = path if path.suffix == ".zarr" else _base_from_x_npy(path)
        schema_path = Path(str(base) + "_feature_schema.json")
        if schema_path.exists() and "X" in report:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            n_schema = len(schema) if isinstance(schema, list) else 0
            x_shape = report["X"]["shape"]
            n_feat = int(x_shape[-1]) if isinstance(x_shape, tuple) and x_shape else 0
            if n_schema and n_feat and n_schema != n_feat:
                print(f"  SCHEMA_MISMATCH schema_cols={n_schema} X_features={n_feat} ({schema_path.name})")
                schema_code = 2
            elif n_schema:
                print(f"  schema_ok cols={n_schema} ({schema_path.name})")
        elif "X" in report:
            print("  schema_missing (no *_feature_schema.json)")
            schema_code = 1
    except Exception as exc:
        print(f"  schema_check_failed: {exc}")
        schema_code = 1

    return max(0 if bad_total == 0 else 2, schema_code)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only finite-value audit for training caches.")
    parser.add_argument(
        "path", nargs="?", default="data/processed", help="Cache path, *_X.npy path, or cache directory"
    )
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    root = Path(args.path).expanduser()
    if not root.exists():
        raise SystemExit(f"cache path not found: {root}")

    paths = list(_iter_cache_paths(root))
    if not paths:
        print(f"No training caches found under {root}")
        return 1

    exit_code = 0
    for path in paths:
        code = _print_report(path, audit_cache(path, chunk_size=max(1, args.chunk_size)))
        exit_code = max(exit_code, code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
