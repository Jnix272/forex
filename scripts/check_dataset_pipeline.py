#!/usr/bin/env python3
"""Fast raw-data and dataset-cache pipeline audit.

This is intentionally metadata-first. It checks the local artifacts needed by
the training pipeline without crawling every tick file or loading full datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
]


@dataclass
class Finding:
    section: str
    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / 1_000_000, 3)
    except OSError:
        return 0.0


def _add(findings: list[Finding], section: str, status: str, message: str, **detail: Any) -> None:
    findings.append(Finding(section=section, status=status, message=message, detail=detail))


def _direct_dirs(path: Path) -> list[Path]:
    try:
        return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)
    except OSError:
        return []


def _year_dirs(path: Path, prefix: str = "") -> list[int]:
    years: list[int] = []
    for child in _direct_dirs(path):
        name = child.name
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]
        if name.isdigit():
            years.append(int(name))
    return sorted(set(years))


def _first_file(root: Path, name: str | None = None, suffix: str | None = None, max_dirs: int = 5000) -> Path | None:
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        seen += 1
        if seen > max_dirs:
            return None
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            if name is not None and filename != name:
                continue
            if suffix is not None and not filename.endswith(suffix):
                continue
            return Path(dirpath) / filename
    return None


def _parquet_meta(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        return {
            "path": _rel(path),
            "rows": int(pf.metadata.num_rows),
            "row_groups": int(pf.metadata.num_row_groups),
            "columns": list(pf.schema_arrow.names),
            "size_mb": _size_mb(path),
        }
    except Exception as exc:
        return {"path": _rel(path), "error": str(exc), "size_mb": _size_mb(path)}


def check_source_dirs(findings: list[Finding]) -> None:
    raw = ROOT / "data" / "raw"
    required = ["dukascopy", "news", "cot", "eco_calendar"]
    optional = ["eodhd", "lmax"]
    if not raw.exists():
        _add(findings, "sources", "FAIL", "data/raw is missing", path=_rel(raw))
        return
    for name in required + optional:
        path = raw / name
        if not path.exists():
            _add(findings, "sources", "WARN" if name in optional else "FAIL", f"source directory missing: {name}", path=_rel(path))
            continue
        populated = any(path.iterdir())
        _add(
            findings,
            "sources",
            "OK" if populated else "WARN",
            f"source directory {'populated' if populated else 'empty'}: {name}",
            path=_rel(path),
        )


def check_raw_dukascopy(findings: list[Finding], pairs: list[str], expected_start: int, expected_end: int) -> None:
    raw_root = ROOT / "data" / "raw" / "dukascopy"
    if not raw_root.exists():
        _add(findings, "raw_dukascopy", "FAIL", "raw Dukascopy directory missing", path=_rel(raw_root))
        return
    expected_years = set(range(expected_start, expected_end + 1))
    for pair in pairs:
        pair_dir = raw_root / pair
        if not pair_dir.exists():
            _add(findings, "raw_dukascopy", "FAIL", f"{pair}: raw directory missing", pair=pair, path=_rel(pair_dir))
            continue
        years = _year_dirs(pair_dir)
        missing = sorted(expected_years - set(years))
        status = "OK" if not missing else ("WARN" if years else "FAIL")
        _add(
            findings,
            "raw_dukascopy",
            status,
            f"{pair}: raw year coverage {years[0] if years else 'N/A'}-{years[-1] if years else 'N/A'} ({len(years)} years)",
            pair=pair,
            years=years,
            missing_years=missing[:20],
            missing_year_count=len(missing),
            path=_rel(pair_dir),
        )


def check_compact_dukascopy(
    findings: list[Finding],
    pairs: list[str],
    expected_start: int,
    expected_end: int,
    sample_parquet: bool,
) -> None:
    compact = ROOT / "data" / "compact" / "dukascopy" / "granularity=daily"
    if not compact.exists():
        _add(findings, "compact_dukascopy", "FAIL", "compact Dukascopy hive directory missing", path=_rel(compact))
        return
    expected_years = set(range(expected_start, expected_end + 1))
    expected_month_count = len(expected_years) * 12
    for pair in pairs:
        pair_dir = compact / f"pair={pair}"
        if not pair_dir.exists():
            _add(findings, "compact_dukascopy", "FAIL", f"{pair}: compact partition missing", pair=pair, path=_rel(pair_dir))
            continue
        years = _year_dirs(pair_dir, prefix="year=")
        months = 0
        for ydir in _direct_dirs(pair_dir):
            months += len([p for p in _direct_dirs(ydir) if p.name.startswith("month=")])
        missing = sorted(expected_years - set(years))
        month_gap = max(0, expected_month_count - months)
        status = "OK" if not missing and month_gap == 0 else ("WARN" if years else "FAIL")
        detail: dict[str, Any] = {
            "pair": pair,
            "years": years,
            "months": months,
            "expected_months": expected_month_count,
            "missing_month_count": month_gap,
            "missing_years": missing[:20],
            "missing_year_count": len(missing),
            "path": _rel(pair_dir),
        }
        if sample_parquet:
            sample = _first_file(pair_dir, name="ticks.parquet", max_dirs=2500)
            detail["sample_parquet"] = _parquet_meta(sample) if sample else None
            if not sample:
                status = "WARN" if status == "OK" else status
        _add(
            findings,
            "compact_dukascopy",
            status,
            (
                f"{pair}: compact coverage {years[0] if years else 'N/A'}-{years[-1] if years else 'N/A'} "
                f"({len(years)} years, {months}/{expected_month_count} month dirs)"
            ),
            **detail,
        )


def check_duckdb(findings: list[Finding], deep: bool) -> None:
    db = ROOT / "data" / "store" / "forex_ticks.duckdb"
    if not db.exists():
        _add(findings, "duckdb", "FAIL", "consolidated DuckDB store missing", path=_rel(db))
        return
    try:
        import duckdb

        con = duckdb.connect(str(db), read_only=True)
        tables = [r[0] for r in con.execute("show tables").fetchall()]
        if "ticks" not in tables:
            _add(findings, "duckdb", "FAIL", "DuckDB missing ticks table", tables=tables, path=_rel(db), size_mb=_size_mb(db))
            return
        schema = con.execute("describe ticks").fetchall()
        cols = [r[0] for r in schema]
        detail: dict[str, Any] = {"path": _rel(db), "size_mb": _size_mb(db), "columns": cols}
        try:
            detail["rows"] = int(con.execute("select count(*) from ticks").fetchone()[0])
        except Exception as exc:
            detail["row_count_error"] = str(exc)
        if deep:
            try:
                rows = con.execute(
                    "select pair, count(*) as n, min(timestamp) as start_ts, max(timestamp) as end_ts "
                    "from ticks group by pair order by pair"
                ).fetchall()
                detail["pairs"] = [
                    {"pair": pair, "rows": int(n), "start": str(start), "end": str(end)}
                    for pair, n, start, end in rows
                ]
            except Exception as exc:
                detail["pair_summary_error"] = str(exc)
        required = {"timestamp", "bid", "ask", "mid", "spread", "volume", "pair", "source"}
        missing = sorted(required - set(cols))
        _add(
            findings,
            "duckdb",
            "OK" if not missing else "FAIL",
            f"DuckDB ticks table readable ({detail.get('rows', 'unknown')} rows)",
            missing_columns=missing,
            **detail,
        )
    except Exception as exc:
        _add(findings, "duckdb", "FAIL", f"DuckDB unreadable: {exc}", path=_rel(db), size_mb=_size_mb(db))


def check_news(findings: list[Finding]) -> None:
    news_dir = ROOT / "data" / "raw" / "news"
    if not news_dir.exists():
        _add(findings, "news", "FAIL", "news directory missing", path=_rel(news_dir))
        return
    files = sorted([p for p in news_dir.iterdir() if p.is_file()], key=lambda p: p.name)
    csv_files = [p for p in files if p.suffix.lower() == ".csv"]
    parquet_files = [p for p in files if p.suffix.lower() in {".parquet", ".parq"}]
    _add(
        findings,
        "news",
        "OK" if files else "WARN",
        f"news files present: {len(files)} files",
        csv_files=[{"path": _rel(p), "size_mb": _size_mb(p)} for p in csv_files],
        parquet_files=[{"path": _rel(p), "size_mb": _size_mb(p)} for p in parquet_files],
    )
    combined = news_dir / "historical_news_combined.parquet"
    if combined.exists():
        meta = _parquet_meta(combined)
        status = "OK" if "error" not in meta and meta.get("rows", 0) > 0 else "FAIL"
        _add(findings, "news", status, "combined news parquet metadata", **meta)
    else:
        _add(findings, "news", "WARN", "combined news parquet missing", path=_rel(combined))
    if len(csv_files) > 1 and parquet_files:
        _add(
            findings,
            "news",
            "WARN",
            "multiple large CSV news files coexist with parquet; check for duplicate storage",
            csv_total_mb=round(sum(p.stat().st_size for p in csv_files) / 1_000_000, 3),
            parquet_total_mb=round(sum(p.stat().st_size for p in parquet_files) / 1_000_000, 3),
        )


def check_csv_head(findings: list[Finding], section: str, path: Path, required_any: list[str]) -> None:
    if not path.exists():
        _add(findings, section, "WARN", f"{_rel(path)} missing", path=_rel(path))
        return
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        missing_all = not any(col in header for col in required_any)
        _add(
            findings,
            section,
            "WARN" if missing_all else "OK",
            f"{_rel(path)} readable header",
            path=_rel(path),
            size_mb=_size_mb(path),
            columns=header[:50],
        )
    except Exception as exc:
        _add(findings, section, "FAIL", f"{_rel(path)} unreadable: {exc}", path=_rel(path), size_mb=_size_mb(path))


def check_auxiliary_sources(findings: list[Finding]) -> None:
    cot = ROOT / "data" / "raw" / "cot" / "cot_financials_cleaned.parquet"
    if cot.exists():
        meta = _parquet_meta(cot)
        _add(findings, "cot", "OK" if "error" not in meta and meta.get("rows", 0) > 0 else "WARN", "COT parquet metadata", **meta)
    else:
        _add(findings, "cot", "WARN", "COT parquet missing", path=_rel(cot))
    check_csv_head(findings, "eco_calendar", ROOT / "data" / "raw" / "eco_calendar" / "events.csv", ["timestamp_utc", "date", "event"])
    cross = ROOT / "data" / "processed" / "cross_asset"
    if cross.exists():
        files = [p for p in cross.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".parquet"}]
        _add(
            findings,
            "cross_asset",
            "OK" if files else "WARN",
            f"cross-asset processed files present: {len(files)}",
            path=_rel(cross),
            sample=[{"path": _rel(p), "size_mb": _size_mb(p)} for p in sorted(files)[:20]],
        )
    else:
        _add(findings, "cross_asset", "WARN", "processed cross-asset directory missing", path=_rel(cross))


def _zarray_meta(path: Path) -> dict[str, Any]:
    meta_path = path / ".zarray"
    if not meta_path.exists():
        return {"exists": False, "path": _rel(meta_path)}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "exists": True,
            "path": _rel(meta_path),
            "shape": data.get("shape"),
            "chunks": data.get("chunks"),
            "dtype": data.get("dtype"),
            "compressor": data.get("compressor", {}).get("id") if isinstance(data.get("compressor"), dict) else data.get("compressor"),
        }
    except Exception as exc:
        return {"exists": False, "path": _rel(meta_path), "error": str(exc)}


def check_zarr_caches(findings: list[Finding]) -> None:
    processed = ROOT / "data" / "processed"
    caches = sorted(processed.glob("*.zarr"))
    if not caches:
        _add(findings, "processed_zarr", "WARN", "no processed zarr datasets found", path=_rel(processed))
        return
    required = ["X", "y"]
    sidecars = ["y_cls", "pq", "diff", "close", "atr", "spread"]
    for cache in caches:
        arrays: dict[str, Any] = {}
        for name in required + sidecars:
            arrays[name] = _zarray_meta(cache / name)
        problems: list[str] = []
        x_shape = arrays["X"].get("shape")
        y_shape = arrays["y"].get("shape")
        if not arrays["X"].get("exists"):
            problems.append("missing X")
        if not arrays["y"].get("exists"):
            problems.append("missing y")
        if x_shape and y_shape and x_shape[0] != y_shape[0]:
            problems.append(f"X rows {x_shape[0]} != y rows {y_shape[0]}")
        if x_shape:
            for name in sidecars:
                shape = arrays[name].get("shape")
                if shape and shape[0] != x_shape[0]:
                    problems.append(f"{name} rows {shape[0]} != X rows {x_shape[0]}")
            if len(x_shape) != 3:
                problems.append(f"X shape is not 3D: {x_shape}")
            elif x_shape[0] < 10_000:
                problems.append(f"low row count for training cache: {x_shape[0]}")
        if arrays["pq"].get("exists") and arrays["pq"].get("dtype") not in {"<f4", "|f4", "float32"}:
            problems.append(f"pq dtype suspicious: {arrays['pq'].get('dtype')}")
        if arrays["diff"].get("exists") and arrays["diff"].get("dtype") not in {"|u1", "uint8"}:
            problems.append(f"diff dtype suspicious: {arrays['diff'].get('dtype')}")
        manifest_candidates = [
            cache.with_name(cache.name + "_manifest.json"),
            processed / "dataset_manifest.json",
        ]
        manifest = next((p for p in manifest_candidates if p.exists()), None)
        schema = cache.with_name(cache.name + "_feature_schema.json")
        schema_audit = cache.with_name(cache.name + "_feature_schema_audit.json")
        if x_shape and len(x_shape) == 3 and schema.exists():
            try:
                schema_len = len(json.loads(schema.read_text(encoding="utf-8")))
                if schema_len != int(x_shape[2]):
                    problems.append(f"feature schema length {schema_len} != X feature dim {x_shape[2]}")
            except Exception as exc:
                problems.append(f"feature schema unreadable: {exc}")
        elif x_shape and len(x_shape) == 3 and not schema.exists() and not schema_audit.exists():
            problems.append("feature schema sidecar missing")
        status = "OK" if not problems else ("WARN" if x_shape else "FAIL")
        _add(
            findings,
            "processed_zarr",
            status,
            f"{cache.name}: zarr metadata {'ok' if not problems else 'has issues'}",
            path=_rel(cache),
            arrays=arrays,
            manifest=_rel(manifest) if manifest else None,
            feature_schema=_rel(schema) if schema.exists() else None,
            feature_schema_audit=_rel(schema_audit) if schema_audit.exists() else None,
            problems=problems,
        )


def check_pipeline_entrypoints(findings: list[Finding]) -> None:
    scripts = [
        ROOT / "scripts" / "run_pipeline.py",
        ROOT / "scripts" / "download_all.py",
        ROOT / "scripts" / "migrate_to_duckdb.py",
        ROOT / "scripts" / "validate_data_quality.py",
        ROOT / "scripts" / "train.py",
    ]
    for script in scripts:
        _add(
            findings,
            "pipeline_entrypoints",
            "OK" if script.exists() else "FAIL",
            f"{_rel(script)} {'exists' if script.exists() else 'missing'}",
            path=_rel(script),
        )
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "all", "--dry-run"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        _add(
            findings,
            "pipeline_entrypoints",
            "OK" if proc.returncode == 0 else "FAIL",
            "run_pipeline.py all --dry-run",
            returncode=proc.returncode,
            stdout=proc.stdout[-2000:],
            stderr=proc.stderr[-2000:],
        )
    except Exception as exc:
        _add(findings, "pipeline_entrypoints", "FAIL", f"pipeline dry-run failed to execute: {exc}")


def summarize(findings: list[Finding]) -> dict[str, Any]:
    counts = {"OK": 0, "WARN": 0, "FAIL": 0}
    for finding in findings:
        counts[finding.status] = counts.get(finding.status, 0) + 1
    return {
        "status": "FAIL" if counts.get("FAIL", 0) else ("WARN" if counts.get("WARN", 0) else "OK"),
        "counts": counts,
    }


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("=" * 78)
    print("DATASET / PIPELINE CHECK")
    print("=" * 78)
    print(f"Overall: {summary['status']} | OK={summary['counts'].get('OK', 0)} WARN={summary['counts'].get('WARN', 0)} FAIL={summary['counts'].get('FAIL', 0)}")
    print(f"Report:  {report['output_path']}")
    print("-" * 78)
    for status in ("FAIL", "WARN", "OK"):
        rows = [f for f in report["findings"] if f["status"] == status]
        if not rows:
            continue
        print(f"\n{status} ({len(rows)})")
        for row in rows:
            print(f"  [{row['section']}] {row['message']}")
            problems = row.get("detail", {}).get("problems")
            if problems:
                for problem in problems[:8]:
                    print(f"    - {problem}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast raw-data and dataset-cache pipeline audit")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--expected-start-year", type=int, default=2008)
    parser.add_argument("--expected-end-year", type=int, default=2025)
    parser.add_argument("--sample-parquet", action="store_true", help="Read one compact parquet footer per pair")
    parser.add_argument("--deep-duckdb", action="store_true", help="Run grouped pair/date summaries over the DuckDB ticks table")
    parser.add_argument("--output", default="logs/dataset_pipeline_check.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    started = time.time()
    findings: list[Finding] = []
    pairs = [p.strip().upper() for p in args.pairs if p.strip()]

    check_source_dirs(findings)
    check_raw_dukascopy(findings, pairs, args.expected_start_year, args.expected_end_year)
    check_compact_dukascopy(findings, pairs, args.expected_start_year, args.expected_end_year, args.sample_parquet)
    check_duckdb(findings, deep=args.deep_duckdb)
    check_news(findings)
    check_auxiliary_sources(findings)
    check_zarr_caches(findings)
    check_pipeline_entrypoints(findings)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "pairs": pairs,
        "summary": summarize(findings),
        "output_path": _rel(output),
        "findings": [asdict(f) for f in findings],
    }
    output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    if not args.quiet:
        print_report(report)
    return 1 if report["summary"]["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
