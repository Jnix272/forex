"""
End-to-end full training test using real data.
This test runs the main model training pipeline (train_gpu.py) over real downloaded data
and logs the entire output to a file.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Paths
_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _ROOT / "logs" / "tests"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "e2e_real_data_test.log"


def _check_real_data() -> bool:
    """Check if real Dukascopy data is available to run full tests against."""
    raw_root = _ROOT / "data" / "raw" / "dukascopy"
    if not raw_root.exists():
        return False
    return any(pair_dir.is_dir() and any(pair_dir.rglob("*.parquet")) for pair_dir in raw_root.iterdir())


def _discover_trading_window(min_files: int = 3) -> tuple[str, str] | None:
    """Pick a contiguous weekday window covered by local Dukascopy parquet files."""
    raw_root = _ROOT / "data" / "raw" / "dukascopy"
    dates: set[datetime] = set()

    # Layout: data/raw/dukascopy/EURUSD/YYYY/MM/DD_HH.parquet
    for pair_dir in raw_root.iterdir():
        if not pair_dir.is_dir():
            continue
        for year_dir in sorted(pair_dir.iterdir(), reverse=True):
            if not (year_dir.is_dir() and year_dir.name.isdigit() and len(year_dir.name) == 4):
                continue
            for month_dir in sorted(year_dir.iterdir(), reverse=True):
                if not (month_dir.is_dir() and month_dir.name.isdigit() and len(month_dir.name) == 2):
                    continue
                for parquet in month_dir.glob("*.parquet"):
                    stem = parquet.stem  # e.g. 02_07 or 02
                    day_token = stem.split("_", 1)[0]
                    if not (day_token.isdigit() and len(day_token) == 2):
                        continue
                    try:
                        dates.add(datetime(int(year_dir.name), int(month_dir.name), int(day_token)))
                    except ValueError:
                        continue
                if len(dates) >= 40:
                    break
            if len(dates) >= 40:
                break
        if len(dates) >= 40:
            break

    if len(dates) < min_files:
        return None

    ordered = sorted(dates)
    # Prefer Mon-Fri spans of at least 3 calendar days with data.
    for start in ordered:
        if start.weekday() >= 5:
            continue
        end = start + timedelta(days=4)
        covered = [d for d in ordered if start <= d <= end and d.weekday() < 5]
        if len(covered) >= min(3, min_files):
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    start, end = ordered[0], ordered[min(len(ordered) - 1, 4)]
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


@pytest.fixture(scope="module")
def setup_logging():
    # Close any existing handlers on this logger before unlinking the log file
    existing_logger = logging.getLogger("e2e_real_data")
    for h in existing_logger.handlers[:]:
        h.close()
        existing_logger.removeHandler(h)

    # Setup a fresh log file for this run (truncate if locked by another process)
    if _LOG_FILE.exists():
        try:
            _LOG_FILE.unlink()
        except PermissionError:
            _LOG_FILE.write_text("", encoding="utf-8")

    logger = logging.getLogger("e2e_real_data")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)

    yield logger

    logger.info("E2E Test Session Completed.")


@pytest.mark.slow
@pytest.mark.skipif(
    not _check_real_data(), reason="No real Dukascopy data downloaded. Run scripts/download_data.py first."
)
def test_full_pipeline_with_real_data(setup_logging, request):
    """
    Run train_gpu.py via subprocess to ensure the actual training CLI
    works end-to-end on real data.
    """
    logger = setup_logging
    logger.info("Starting End-to-End full model training test on real data...")

    window = _discover_trading_window()
    if window is None:
        pytest.skip("Dukascopy files exist but no usable date window could be discovered")
    data_start, data_end = window
    logger.info(f"Using discovered data window {data_start} -> {data_end}")

    python_exe = sys.executable
    try:
        from scripts._python_env import python_exe as _resolve

        python_exe = _resolve()
    except Exception:
        pass

    cmd = [
        python_exe,
        "training/train_gpu.py",
        "--data-source",
        "dukascopy",
        "--model",
        "haelt",
        "--epochs",
        "2",
        "--batch-size",
        "128",
        "--no-wandb",
        "--force-rebuild",
        "--quick-mode",
        "--data-start",
        data_start,
        "--data-end",
        data_end,
    ]

    if request.config.getoption("--quick-mode", default=False):
        pass

    logger.info(f"Executing training pipeline: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["DISCORD_WEBHOOK_URL"] = ""

    process = subprocess.Popen(
        cmd,
        cwd=str(_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )

    output_lines = []
    if process.stdout:
        for line in process.stdout:
            cleaned_line = line.strip()
            output_lines.append(cleaned_line)
            logger.info(f"[train_gpu] {cleaned_line}")
            try:
                print(f"[train_gpu] {cleaned_line}")
            except UnicodeEncodeError:
                print(f"[train_gpu] {cleaned_line.encode('ascii', errors='replace').decode('ascii')}")

    exit_code = process.wait()
    logger.info(f"Training pipeline finished with exit_code: {exit_code}")
    output_str = "\n".join(output_lines)
    output_l = output_str.lower()

    if exit_code != 0 and any(
        marker in output_l for marker in ("no bars", "empty", "0 rows", "insufficient data", "no data")
    ):
        pytest.skip(
            f"Discovered window {data_start}->{data_end} had insufficient Dukascopy bars "
            f"(exit={exit_code}). See {_LOG_FILE}."
        )

    assert exit_code == 0, f"train_gpu.py failed with exit code {exit_code}. See logs at {_LOG_FILE} for details."
    assert "training complete" in output_l or "evaluation" in output_l or "model=" in output_l, (
        f"Training finished but success markers missing. See {_LOG_FILE}."
    )

    logger.info("End-to-end real data test passed successfully!")
