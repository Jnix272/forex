"""
scripts/run_pipeline.py
=======================
Top-level pipeline: download data, train, or both.

Usage (PowerShell):
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_pipeline.py download --start 2017-02-18
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_pipeline.py train --quick
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_pipeline.py all --start 2017-02-18
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            description="Run download, features, train, backtest, or the full pipeline.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "command", choices=("download", "features", "train", "backtest", "all"),
            help="download | features | train | backtest | all",
        )
        parser.add_argument("passthrough", nargs=argparse.REMAINDER,
                            help="Flags forwarded to the respective script")
        parser.print_help()
        return 0

    command = argv[0]
    passthrough = argv[1:]

    # WIRE-007: Full pipeline now covers features, backtest stages
    if command == "download":
        return _run_script("download_all.py", passthrough)
    if command == "features":
        return _run_script("build_features.py", passthrough)
    if command == "train":
        return _run_script("train.py", passthrough)
    if command == "backtest":
        return _run_script("run_backtest.py", passthrough)
    if command == "all":
        for script in ("download_all.py", "build_features.py", "train.py", "run_backtest.py"):
            script_path = _ROOT / "scripts" / script
            if not script_path.exists():
                print(f"[pipeline] Skipping {script} (not found)", flush=True)
                continue
            rc = _run_script(script, passthrough)
            if rc != 0:
                print(f"[pipeline] Stage {script} failed with code {rc}", flush=True)
                return rc
        return 0

    print(f"[pipeline] Unknown command: {command!r}. Use: download | features | train | backtest | all", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
