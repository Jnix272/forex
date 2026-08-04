"""
scripts/download_all.py
=======================
One-command download for all offline training data.

Runs in order:
  1. Dukascopy tick + cross-asset panel  (scripts/download_data.py)
  2. GDELT historical news               (scripts/download_historical_news.py)
  3. CFTC COT positioning                (scripts/download_cot.py)
  4. ForexFactory economic calendar      (scripts/scrape_forexfactory.py)

Usage (PowerShell):
  .\\.venv-gpu\\Scripts\\python.exe scripts\\download_all.py --start 2008-01-01 --end 2025-12-31
  .\\.venv-gpu\\Scripts\\python.exe scripts\\download_all.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - keeps --help usable in tiny envs
    yaml = None

try:
    import wandb
    WANDB = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    WANDB = False

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env (WANDB_API_KEY, EODHD keys, etc.) — same pattern as config/settings.py
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass

DEFAULT_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "EURGBP", "NZDUSD", "EURJPY", "GBPJPY",
]

# ── Data directory paths for metrics ─────────────────────────────────────────
_DATA_PATHS = {
    "prices":   _ROOT / "data" / "raw" / "dukascopy",
    "news":     _ROOT / "data" / "raw" / "news",
    "cot":      _ROOT / "data" / "raw" / "cot",
    "calendar": _ROOT / "data" / "raw" / "eco_calendar",
}


def _dir_stats(path: Path) -> dict[str, int]:
    """Count files and total bytes under a directory (non-recursive is too slow; use recursive)."""
    n_files = 0
    total_bytes = 0
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                n_files += 1
                try:
                    total_bytes += f.stat().st_size
                except OSError:
                    pass
    return {"files": n_files, "bytes": total_bytes}


def _python_exe() -> str:
    from scripts._python_env import python_exe as _resolve
    py = _resolve()
    try:
        subprocess.run(
            [py, "--version"],
            cwd=str(_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=True,
        )
        return py
    except Exception:
        print(f"[download_all] WARN: {py} did not start; using {sys.executable}", flush=True)
        return sys.executable


def _load_yaml_defaults() -> dict[str, Any]:
    """Load downloader defaults from config/run.yaml, with safe long-history fallbacks."""
    defaults: dict[str, Any] = {
        "pairs": list(DEFAULT_PAIRS),
        "start": "2008-01-01",
        "end": datetime.now(UTC).strftime("%Y-%m-%d"),
        "full_day": False,
        "yearly": True,
        "keep_going": True,
        "verify": True,
        "verify_fix": True,
        "check_missing_months": True,
        "check_missing_scope": "both",
        "verify_min_ticks": 0,
    }
    if yaml is None:
        return defaults

    for rel in ("config/run.yaml", "config/run_ubuntu.yaml"):
        path = _ROOT / rel
        if not path.is_file():
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            data = cfg.get("data") or {}
            dl = cfg.get("download") or {}

            if dl.get("pairs"):
                defaults["pairs"] = [str(p).upper().replace("/", "") for p in dl["pairs"]]
            elif data.get("pairs") and len(data["pairs"]) >= len(DEFAULT_PAIRS):
                defaults["pairs"] = [str(p).upper().replace("/", "") for p in data["pairs"]]
            if data.get("start"):
                defaults["start"] = str(data["start"])
            if data.get("end"):
                defaults["end"] = str(data["end"])

            defaults["full_day"] = bool(data.get("full_day_data", defaults["full_day"]))
            for key in (
                "yearly",
                "keep_going",
                "verify",
                "verify_fix",
                "check_missing_months",
                "check_missing_scope",
                "verify_min_ticks",
            ):
                if key in dl:
                    defaults[key] = dl[key]
            break
        except Exception:
            pass
    return defaults


def _valid_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc
    return value


def _header(step: int, total: int, name: str) -> None:
    bar = "=" * 66
    print(f"\n{bar}\n  Step {step}/{total}: {name}\n{bar}", flush=True)


def _run_step(cmd: list[str], *, dry_run: bool, step_label: str) -> int:
    print(f"  $ {' '.join(cmd)}", flush=True)
    if dry_run:
        print(f"  [DRY-RUN] skipped {step_label}", flush=True)
        return 0
    result = subprocess.run(cmd, cwd=str(_ROOT))
    return int(result.returncode)


# ── W&B helpers ──────────────────────────────────────────────────────────────

def _wandb_init(args: argparse.Namespace) -> Any:
    """Initialize a W&B run for the download pipeline. Returns the run or None."""
    if not WANDB:
        print("[W&B] Skipped: wandb not installed", flush=True)
        return None
    if args.no_wandb:
        print("[W&B] Disabled (--no-wandb)", flush=True)
        return None
    if not os.getenv("WANDB_API_KEY"):
        print("[W&B] Skipped: WANDB_API_KEY not set", flush=True)
        return None

    run = wandb.init(
        project=args.wandb_project,
        name=f"download_{args.start}_to_{args.end}",
        job_type="data-download",
        config={
            "pairs": args.pairs,
            "start": args.start,
            "end": args.end,
            "yearly": args.yearly,
            "full_day": args.full_day,
            "verify": args.verify,
            "verify_fix": args.verify_fix,
            "gdelt_workers": args.gdelt_workers,
            "gdelt_min_interval": args.gdelt_min_interval,
            "gdelt_step_days": args.gdelt_step_days,
        },
        tags=["download", "data-pipeline"],
    )
    if run is not None:
        os.environ["WANDB_RUN_GROUP"] = str(run.name)
        os.environ["WANDB_PROJECT"] = args.wandb_project
    print(f"[W&B] Run started: {run.url}", flush=True)
    return run


def _wandb_log_step(
    run: Any,
    step_name: str,
    step_idx: int,
    duration_s: float,
    return_code: int,
    data_key: str | None = None,
) -> None:
    """Log metrics for a completed download step."""
    if run is None:
        return

    metrics: dict[str, Any] = {
        f"steps/{step_name}/duration_s": round(duration_s, 1),
        f"steps/{step_name}/duration_min": round(duration_s / 60, 2),
        f"steps/{step_name}/return_code": return_code,
        f"steps/{step_name}/success": int(return_code == 0),
        "step_index": step_idx,
    }

    # Count files and bytes for the relevant data directory
    if data_key and data_key in _DATA_PATHS:
        stats = _dir_stats(_DATA_PATHS[data_key])
        metrics[f"data/{data_key}/files"] = stats["files"]
        metrics[f"data/{data_key}/size_mb"] = round(stats["bytes"] / (1024 * 1024), 1)
        metrics[f"data/{data_key}/size_gb"] = round(stats["bytes"] / (1024**3), 3)

    run.log(metrics)


def _wandb_log_summary(run: Any, total_duration_s: float, failed: list[str]) -> None:
    """Log final summary metrics."""
    if run is None:
        return

    # Aggregate all data directories
    total_files = 0
    total_bytes = 0
    for key, path in _DATA_PATHS.items():
        stats = _dir_stats(path)
        total_files += stats["files"]
        total_bytes += stats["bytes"]
        run.summary[f"final/{key}_files"] = stats["files"]
        run.summary[f"final/{key}_gb"] = round(stats["bytes"] / (1024**3), 3)

    run.summary["final/total_files"] = total_files
    run.summary["final/total_gb"] = round(total_bytes / (1024**3), 2)
    run.summary["final/total_duration_min"] = round(total_duration_s / 60, 1)
    run.summary["final/total_duration_hr"] = round(total_duration_s / 3600, 2)
    run.summary["final/failed_steps"] = len(failed)
    run.summary["final/failed_names"] = ", ".join(failed) if failed else "none"
    run.summary["final/success"] = len(failed) == 0


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    defaults = _load_yaml_defaults()
    p = argparse.ArgumentParser(
        description="Download all forex ML data in one command: prices -> GDELT -> COT -> ForexFactory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", type=_valid_date, default=defaults["start"], help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=_valid_date, default=defaults["end"], help="End date YYYY-MM-DD")
    p.add_argument("--pairs", nargs="+", default=defaults["pairs"], metavar="PAIR",
                   help="FX pairs for price and GDELT news downloads")

    p.add_argument("--skip-prices", action="store_true", help="Skip Dukascopy tick + cross-asset")
    p.add_argument("--skip-news", action="store_true", help="Skip free historical news download")
    p.add_argument("--skip-cot", action="store_true", help="Skip CFTC COT download")
    p.add_argument("--skip-eco", action="store_true", help="Skip ForexFactory calendar scrape")
    p.add_argument("--continue-on-error", action="store_true", help="Run remaining steps even if one step fails")
    p.add_argument("--dry-run", action="store_true", help="Print planned commands without launching child scripts")

    p.add_argument("--yearly", action=argparse.BooleanOptionalAction, default=bool(defaults["yearly"]),
                   help="Download prices pair-by-pair/year-by-year with repair gates")
    p.add_argument("--full-day", action=argparse.BooleanOptionalAction, default=bool(defaults["full_day"]),
                   help="Download all 24 hours/day instead of session hours")
    p.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=bool(defaults["keep_going"]),
                   help="Continue later price years/pairs after a failed year")
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=bool(defaults["verify"]),
                   help="Run price data verification after download")
    p.add_argument("--verify-fix", action=argparse.BooleanOptionalAction, default=bool(defaults["verify_fix"]),
                   help="Auto-repair duplicate/gap issues during price verification")
    p.add_argument("--check-missing-months", action=argparse.BooleanOptionalAction,
                   default=bool(defaults["check_missing_months"]),
                   help="Check and redownload suspicious missing price months")
    p.add_argument("--check-missing-scope", choices=["months", "years", "pairs", "both"],
                   default=str(defaults["check_missing_scope"]), help="Missing-data summary scope")
    p.add_argument("--verify-min-ticks", type=int, default=int(defaults["verify_min_ticks"]),
                   help="Flag price files with fewer than N ticks as suspicious")

    p.add_argument("--score-sentiment", action="store_true",
                   help="After news download, pre-warm sentiment cache")
    p.add_argument("--gdelt-min-interval", type=float, default=5.0,
                   help="Minimum seconds between GDELT API requests")
    p.add_argument("--gdelt-step-days", type=int, default=1, help="GDELT query window size")
    p.add_argument("--gdelt-max-records", type=int, default=250, help="GDELT max records per request")
    p.add_argument("--gdelt-retries", type=int, default=5, help="Retries per GDELT chunk")
    p.add_argument("--gdelt-workers", type=int, default=2, help="Parallel GDELT pair workers")
    p.add_argument("--retry-news-failures", action="store_true", help="Retry rows in the news failure manifest")
    p.add_argument("--official-feeds", action=argparse.BooleanOptionalAction, default=True,
                   help="Include free official central-bank RSS/Atom feeds in historical_news_combined.parquet")

    # W&B options
    p.add_argument("--wandb-project", type=str, default="forex-scaling-model",
                   help="Weights & Biases project name")
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if datetime.strptime(args.start, "%Y-%m-%d") > datetime.strptime(args.end, "%Y-%m-%d"):
        print(f"[download_all] ERROR: --start {args.start} is after --end {args.end}", flush=True)
        return 2

    py = _python_exe()
    pairs = [p.upper().replace("/", "") for p in args.pairs]
    start_year = int(args.start[:4])

    # Map step labels to their data directory key for W&B metrics
    step_data_keys: dict[str, str] = {}
    steps: list[tuple[str, list[str]]] = []

    if not args.skip_prices:
        price_cmd = [
            py, str(_ROOT / "scripts" / "download_data.py"),
            "--pairs", *pairs,
            "--start", args.start,
            "--end", args.end,
            "--no-eodhd",
            "--no-eodhd-cross-asset",
        ]
        if args.full_day:
            price_cmd.append("--full-day")
        if args.yearly:
            price_cmd.append("--yearly")
        if args.keep_going:
            price_cmd.append("--keep-going")
        if args.verify:
            price_cmd.append("--verify")
        if args.verify_fix:
            price_cmd.append("--verify-fix")
        if args.check_missing_months:
            price_cmd.extend(["--check-missing-months", "--check-missing-scope", args.check_missing_scope])
        if args.verify_min_ticks > 0:
            price_cmd.extend(["--verify-min-ticks", str(args.verify_min_ticks)])
        label = "Price data (Dukascopy + cross-asset)"
        steps.append((label, price_cmd))
        step_data_keys[label] = "prices"

    if not args.skip_news:
        news_cmd = [
            py, str(_ROOT / "scripts" / "download_historical_news.py"),
            "--source", "gdelt",
            "--pairs", *pairs,
            "--start", args.start,
            "--end", args.end,
            "--resume",
            "--append",
            "--gdelt-min-interval", str(args.gdelt_min_interval),
            "--gdelt-step-days", str(args.gdelt_step_days),
            "--gdelt-max-records", str(args.gdelt_max_records),
            "--gdelt-retries", str(args.gdelt_retries),
            "--workers", str(args.gdelt_workers),
        ]
        if args.official_feeds:
            news_cmd.append("--include-official-feeds")
        if args.score_sentiment:
            news_cmd.append("--score-sentiment")
        if args.retry_news_failures:
            news_cmd.append("--retry-failures")
        if args.dry_run:
            news_cmd.append("--dry-run")
        label = "Historical news"
        steps.append((label, news_cmd))
        step_data_keys[label] = "news"

    if not args.skip_cot:
        cot_cmd = [py, str(_ROOT / "scripts" / "download_cot.py"), "--start-year", str(start_year)]
        if args.dry_run:
            cot_cmd.append("--dry-run")
        label = "CFTC COT"
        steps.append((label, cot_cmd))
        step_data_keys[label] = "cot"

    if not args.skip_eco:
        eco_cmd = [
            py, str(_ROOT / "scripts" / "scrape_forexfactory.py"),
            "--start", args.start,
            "--end", args.end,
            "--resume",
        ]
        if args.dry_run:
            eco_cmd.append("--dry-run")
        label = "ForexFactory economic calendar"
        steps.append((label, eco_cmd))
        step_data_keys[label] = "calendar"

    if not steps:
        print("[download_all] Nothing to do - all steps skipped.", flush=True)
        return 0

    # ── Initialize W&B ───────────────────────────────────────────────────────
    wb_run = _wandb_init(args)

    print("\n" + "=" * 66)
    print("  Forex Scaling Model - Unified Data Download")
    print("=" * 66)
    print(f"  pairs : {', '.join(pairs)}")
    print(f"  range : {args.start} -> {args.end}")
    print(f"  steps : {len(steps)}")
    print(f"  price : yearly={args.yearly} verify={args.verify} fix={args.verify_fix} missing_months={args.check_missing_months}")
    print(f"  news  : gdelt_workers={args.gdelt_workers} min_interval={args.gdelt_min_interval}s step_days={args.gdelt_step_days} official_feeds={args.official_feeds}")
    print(f"  wandb : {'ON -> ' + wb_run.url if wb_run else 'OFF'}")
    if args.dry_run:
        print("  mode  : DRY-RUN")
    print(flush=True)

    pipeline_t0 = time.monotonic()
    failed: list[str] = []
    for i, (label, cmd) in enumerate(steps, 1):
        _header(i, len(steps), label)
        step_t0 = time.monotonic()
        rc = _run_step(cmd, dry_run=args.dry_run, step_label=label)
        step_duration = time.monotonic() - step_t0

        # Log step to W&B
        _wandb_log_step(
            wb_run,
            step_name=label.lower().replace(" ", "_").replace("(", "").replace(")", ""),
            step_idx=i,
            duration_s=step_duration,
            return_code=rc,
            data_key=step_data_keys.get(label),
        )

        if rc != 0:
            failed.append(label)
            print(f"\n[download_all] FAILED: {label} (exit {rc})", flush=True)
            if not args.continue_on_error:
                break

    total_duration = time.monotonic() - pipeline_t0

    # ── Log final summary to W&B ─────────────────────────────────────────────
    _wandb_log_summary(wb_run, total_duration, failed)
    if wb_run:
        wb_run.finish()
        print(f"[W&B] Run finished: {wb_run.url}", flush=True)

    print("\n" + "=" * 66, flush=True)
    if failed:
        print(f"  Download finished with errors: {', '.join(failed)}", flush=True)
        print("=" * 66 + "\n", flush=True)
        return 1

    print("  All download steps completed successfully.", flush=True)
    print("=" * 66 + "\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

