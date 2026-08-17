"""
scripts/run_pipeline.py
=======================
Top-level pipeline orchestrator: download, migrate, features, train, backtest.

Usage (PowerShell):
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_pipeline.py download --start 2017-02-18
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_pipeline.py migrate
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_pipeline.py data --start 2017-02-18
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_pipeline.py train --quick
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_pipeline.py all --start 2017-02-18

`download` auto-runs the DuckDB migration as its final step. `train` and `backtest`
auto-run the migration first when the consolidated
data/store/forex_ticks.duckdb is missing or stale (disable with --no-auto-migrate).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _python_exe() -> str:
    from scripts._python_env import python_exe as _resolve

    return _resolve()


def _run_script(script: str, passthrough: list[str]) -> int:
    cmd = [_python_exe(), str(_ROOT / "scripts" / script), *passthrough]
    print(f"\n[pipeline] $ {' '.join(cmd)}\n", flush=True)
    return int(subprocess.run(cmd, cwd=str(_ROOT)).returncode)


def _latest_compact_mtime(compact: Path) -> float:
    """Newest mtime of any ticks.parquet under the compacted Dukascopy store."""
    latest = 0.0
    for root, _, files in os.walk(compact):
        for name in files:
            if name != "ticks.parquet":
                continue
            p = Path(root) / name
            try:
                latest = max(latest, p.stat().st_mtime)
            except OSError:
                continue
    return latest


def _duckdb_is_stale() -> bool:
    """True when the consolidated DuckDB is missing, older than compact data, or missing pairs."""
    compact = _ROOT / "data" / "compact" / "dukascopy" / "granularity=daily"
    if not compact.is_dir():
        return False
    db = _ROOT / "data" / "store" / "forex_ticks.duckdb"
    if not db.is_file():
        return True

    try:
        import duckdb

        with duckdb.connect(str(db), read_only=True) as c:
            db_pairs = {r[0] for r in c.execute("SELECT DISTINCT pair FROM ticks").fetchall()}
    except Exception:
        return True

    compact_pairs = {
        p.name.replace("pair=", "") for p in compact.iterdir() if p.is_dir() and p.name.startswith("pair=")
    }
    if not compact_pairs.issubset(db_pairs):
        return True

    return _latest_compact_mtime(compact) > db.stat().st_mtime + 60


def _maybe_auto_migrate() -> bool:
    """Run the DuckDB migration before data consumers if the database is stale."""
    if not _duckdb_is_stale():
        return True
    print("[pipeline] Consolidated DuckDB is stale/missing - running migration", flush=True)
    rc = _run_script("migrate_to_duckdb.py", [])
    if rc != 0:
        print(f"[pipeline] Auto-migration failed with code {rc}", flush=True)
        return False
    return True


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            description="Run download, features, train, backtest, or the full pipeline.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "command",
            choices=("download", "migrate", "validate", "data", "train", "backtest", "all"),
            help="download | migrate | validate | data | train | backtest | all",
        )
        parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Flags forwarded to the respective script")
        parser.print_help()
        return 0

    command = argv[0]
    passthrough = argv[1:]

    # WIRE-007: Full pipeline now covers backtest stages
    if command == "download":
        return _run_script("download_all.py", passthrough)
    if command == "migrate":
        return _run_script("migrate_to_duckdb.py", passthrough)
    if command == "validate":
        rc = _run_script("validate_data_quality.py", passthrough)
        if rc != 0:
            return rc
        return _run_script("../training/data_coverage.py", passthrough)
    if command in ("train", "backtest"):
        passthrough = [a for a in passthrough if a != "--no-auto-migrate"]
        auto_migrate = "--no-auto-migrate" not in argv
        if auto_migrate and not _maybe_auto_migrate():
            return 1
        return _run_script("backtest_model.py" if command == "backtest" else "train.py", passthrough)
    if command == "data":
        download_args = []
        train_args = []
        for arg in passthrough:
            if arg in ("--skip-news", "--skip-prices", "--skip-cot", "--skip-eco", "--dry-run"):
                download_args.append(arg)
            else:
                download_args.append(arg)
                train_args.append(arg)

        if "--dry-run" in passthrough:
            print("[pipeline] --dry-run detected. Aborting before full data processing.", flush=True)
            return 0

        for script in (
            "download_all.py",
            "migrate_to_duckdb.py",
            "validate_data_quality.py",
            "../training/data_coverage.py",
        ):
            if script == "download_all.py":
                script_passthrough = download_args
            elif script == "migrate_to_duckdb.py":
                script_passthrough = ["--dry-run"] if "--dry-run" in download_args else []
            elif script in ("validate_data_quality.py", "../training/data_coverage.py"):
                script_passthrough = []
            else:
                script_passthrough = train_args

            rc = _run_script(script, script_passthrough)
            if rc != 0:
                print(f"[pipeline] Stage {script} failed with code {rc}", flush=True)
                return rc
        return 0
    if command == "all":
        # Separate download-specific arguments from general/train arguments
        download_args = []
        train_args = []
        for arg in passthrough:
            if arg in ("--skip-news", "--skip-prices", "--skip-cot", "--skip-eco", "--dry-run"):
                download_args.append(arg)
            else:
                download_args.append(arg)
                train_args.append(arg)

        if "--dry-run" in passthrough:
            print("[pipeline] --dry-run detected. Aborting before execution of train/backtest.", flush=True)
            return 0

        for script in (
            "download_all.py",
            "migrate_to_duckdb.py",
            "validate_data_quality.py",
            "../training/data_coverage.py",
            "train.py",
            "backtest_model.py",
        ):
            script_path = _ROOT / "scripts" / script
            if not script_path.exists():
                print(f"[pipeline] ERROR: Required script {script} not found! Aborting.", flush=True)
                sys.exit(1)

            if script == "download_all.py":
                script_passthrough = download_args
            elif script == "migrate_to_duckdb.py":
                # migration auto-discovers all compacted pairs and is incremental; no passthrough
                script_passthrough = ["--dry-run"] if "--dry-run" in download_args else []
            elif script in ("validate_data_quality.py", "../training/data_coverage.py"):
                script_passthrough = []
            else:
                script_passthrough = train_args
            rc = _run_script(script, script_passthrough)
            if rc != 0:
                print(f"[pipeline] Stage {script} failed with code {rc}", flush=True)
                return rc
        return 0

    print(
        f"[pipeline] Unknown command: {command!r}. Use: download | migrate | validate | data | train | backtest | all",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
