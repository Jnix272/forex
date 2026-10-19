"""
scripts/run_specialist_pipeline.py
==================================
Automated handoff launcher for the Dedicated Single-Pair Specialist Architecture:
  1. Monitors the active Gen-2 3-pair HAELT training run (PID / checkpoints/haelt_relabelled_v1)
     until Fold 5 + CV refit + Promotion Gate complete cleanly (without interrupting it).
  2. Slices the 3-pair Zarr dataset into 3 isolated single-pair Zarr datasets
     (EURUSD, USDCAD, USDJPY; 191,271 x 120 x 146 each) in a single sequential pass
     via scripts/slice_single_pair_caches.py.
  3. Trains each Single-Pair Specialist HAELT model sequentially with:
       - batch_size: 300
       - FOREX_PURE_BCE=1, FOREX_TRADE_FRACTION=0.08
       - Full stationary Advanced Features (all 11 curriculum groups)
       - SI + Dynamic SI, EWC, SWA, Self-Paced Learning (SPL), Online Hard-Example Miner
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

GEN2_SUMMARY = ROOT / "checkpoints" / "haelt_relabelled_v1" / "haelt" / "train_summary.json"
GEN2_GATE = ROOT / "checkpoints" / "haelt_relabelled_v1" / "promotion_gate.json"

SPECIALIST_CONFIGS = [
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


def wait_for_gen2(wait_pid: int, poll_sec: int = 30) -> None:
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistLauncher] "
        f"Waiting for Gen-2 3-pair HAELT training (PID={wait_pid}) to complete...",
        flush=True,
    )
    while _pid_alive(wait_pid):
        time.sleep(poll_sec)
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistLauncher] "
        f"Gen-2 HAELT process (PID={wait_pid}) has finished! Proceeding to Single-Pair Specialist pipeline.",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Wait for Gen-2 HAELT and run Single-Pair Specialist pipeline.")
    ap.add_argument("--wait-pid", type=int, default=3664, help="PID of active Gen-2 train_gpu.py process to wait for")
    ap.add_argument("--poll-sec", type=int, default=30, help="Polling interval in seconds while waiting")
    ap.add_argument("--skip-wait", action="store_true", help="Start immediately without waiting for PID")
    args = ap.parse_args()

    if not args.skip_wait and args.wait_pid > 0:
        wait_for_gen2(args.wait_pid, poll_sec=args.poll_sec)

    py_exe = str(ROOT / ".venv311" / "Scripts" / "python.exe")
    if not Path(py_exe).exists():
        py_exe = sys.executable

    env = os.environ.copy()
    env["FOREX_PURE_BCE"] = "1"
    env["FOREX_TRADE_FRACTION"] = "0.05"
    env["PYTHONPATH"] = str(ROOT)

    # Step 1: Slice 3-pair Zarr into 3 single-pair Zarr stores
    print("\n" + "=" * 72, flush=True)
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [Step 1/4] "
        "Slicing 3-pair Zarr cache into EURUSD, USDCAD, USDJPY single-pair Zarr caches...",
        flush=True,
    )
    print("=" * 72, flush=True)
    slice_cmd = [py_exe, "-u", str(ROOT / "scripts" / "slice_single_pair_caches.py")]
    res_slice = subprocess.run(slice_cmd, cwd=str(ROOT), env=env)
    if res_slice.returncode != 0:
        print(f"[SpecialistLauncher] ERROR: Zarr slicing failed with exit code {res_slice.returncode}", flush=True)
        return int(res_slice.returncode)

    # Step 2..4: Train each Single-Pair Specialist sequentially
    for idx, (pair, cfg_path) in enumerate(SPECIALIST_CONFIGS, start=2):
        print("\n" + "=" * 72, flush=True)
        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [Step {idx}/4] "
            f"Launching Dedicated Single-Pair Specialist Training for {pair} ({cfg_path.name}, batch_size=300)...",
            flush=True,
        )
        print("=" * 72, flush=True)
        train_cmd = [
            py_exe,
            "-u",
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(cfg_path),
            "--full",
        ]
        res = subprocess.run(train_cmd, cwd=str(ROOT), env=env)
        if res.returncode != 0:
            print(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistLauncher] "
                f"WARNING: Specialist training for {pair} exited with code {res.returncode}",
                flush=True,
            )
        else:
            print(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistLauncher] "
                f"Completed Specialist training for {pair} successfully!",
                flush=True,
            )

    print(
        f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [SpecialistLauncher] "
        "All 3 Dedicated Single-Pair Specialist models (EURUSD, USDCAD, USDJPY) finished!",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
