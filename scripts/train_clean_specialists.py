"""
scripts/train_clean_specialists.py
===================================
Automated zero-contention pipeline to retrain and promote Single-Pair Specialists
with the verified Sharpe optimization suite:
  1. Waits for the active USDJPY training run (PID 13596 / task-9690) to finish cleanly.
  2. Sequences clean walk-forward GPU training for:
       - EURUSD (config/run_eurusd.yaml)
       - USDCAD (config/run_usdcad.yaml)
       - USDJPY (config/run_usdjpy.yaml)
  3. Features active in each run:
       - Stationary feature set (raw prices open, high, low, close, bid_close, ask_close dropped)
       - Dynamic Reliability Guard (min_reliable_val_trades: 30) preserving peak validation Sharpes
       - Quantile trade selection (FOREX_TRADE_FRACTION=0.05)
       - Anti-forgetting continual learning (SI + Dynamic SI, EWC, SWA)
       - Self-Paced Learning (mode: self_paced) with Online Miner feedback
       - Promotion gate with quantile holdout backtesting
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPECIALIST_RUNS = [
    ("EURUSD", ROOT / "config" / "run_eurusd.yaml"),
    ("USDCAD", ROOT / "config" / "run_usdcad.yaml"),
    ("USDJPY", ROOT / "config" / "run_usdjpy.yaml"),
]


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok and exit_code.value == STILL_ACTIVE)
    except Exception:
        return False


def wait_for_active_run(wait_pid: int, poll_sec: int = 30) -> None:
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistRetrain] "
        f"Waiting for active GPU training run (PID={wait_pid}) to complete...",
        flush=True,
    )
    while _pid_alive(wait_pid):
        time.sleep(poll_sec)
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistRetrain] "
        f"Active GPU run (PID={wait_pid}) has completed! Starting Clean Specialist Suite.",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Wait for active run and train clean Single-Pair Specialists.")
    ap.add_argument("--wait-pid", type=int, default=13596, help="PID of active training process to wait for")
    ap.add_argument("--poll-sec", type=int, default=30, help="Polling interval in seconds")
    ap.add_argument("--skip-wait", action="store_true", help="Start immediately without waiting")
    ap.add_argument("--pairs", type=str, default="EURUSD,USDCAD,USDJPY", help="Comma-separated list of pairs to train")
    ap.add_argument(
        "--model",
        type=str,
        default="haelt",
        choices=["haelt", "mamba", "tft", "patchtst", "timesnet", "timemixer", "itransformer", "gnn"],
        help="Architecture to train for all single-pair specialists",
    )
    ap.add_argument("--epochs", type=int, default=None, help="Optional epoch count override")
    args = ap.parse_args()

    if not args.skip_wait and args.wait_pid > 0:
        wait_for_active_run(args.wait_pid, poll_sec=args.poll_sec)

    py_exe = str(ROOT / ".venv311" / "Scripts" / "python.exe")
    if not Path(py_exe).exists():
        py_exe = sys.executable

    env = os.environ.copy()
    env["FOREX_PURE_BCE"] = "1"
    env["FOREX_TRADE_FRACTION"] = "0.05"
    env["PYTHONPATH"] = str(ROOT)

    selected_pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    runs = [(p, cfg) for p, cfg in SPECIALIST_RUNS if p in selected_pairs]

    print("\n" + "=" * 76, flush=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistRetrain] Beginning clean retrain for: {selected_pairs} (model={args.model})", flush=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistRetrain] Settings: trade_fraction=0.05, min_reliable_val_trades=30, pure_bce=1", flush=True)
    print("=" * 76, flush=True)

    for idx, (pair, cfg_path) in enumerate(runs, start=1):
        print("\n" + "=" * 76, flush=True)
        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [Run {idx}/{len(runs)}] "
            f"Launching Dedicated Single-Pair Specialist Retrain for {pair} ({cfg_path.name}) with {args.model.upper()}...",
            flush=True,
        )
        print("=" * 76, flush=True)

        cmd = [
            py_exe,
            "-u",
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(cfg_path),
            "--full",
            "--model",
            args.model,
        ]
        ckpt_dir = f"checkpoints/{args.model.lower()}_{pair.lower()}_v1"
        extra_args = ["--", "--checkpoint-dir", ckpt_dir]
        if args.epochs is not None and args.epochs > 0:
            extra_args.extend(["--epochs", str(args.epochs)])
        cmd.extend(extra_args)
        res = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if res.returncode != 0:
            print(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistRetrain] "
                f"WARNING: Specialist training for {pair} exited with code {res.returncode}",
                flush=True,
            )
        else:
            print(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistRetrain] "
                f"SUCCESS: Specialist training for {pair} completed cleanly!",
                flush=True,
            )

    print("\n" + "=" * 76, flush=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistRetrain] All requested specialist models completed!", flush=True)
    print("=" * 76, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
