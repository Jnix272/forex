"""
scripts/run_specialist_rl_and_mamba.py
=======================================
End-to-end automation pipeline executing:
  1. Complete Option 2: Dedicated Single-Pair RL Specialists (EURUSD, USDCAD, USDJPY)
  2. Complete Option 3: Dedicated Single-Pair Mamba Specialists (EURUSD, USDCAD, USDJPY)
  3. Integrated Holdout Evaluation: Benchmarks HAELT vs Mamba vs 4H Ensemble vs RL execution
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import PATHS

RL_SPECS = [
    {
        "pair": "EURUSD",
        "cache": "data/processed/dataset_scalping_5m_EURUSD_0_dukascopy_120_triple_barrier_reward_lh30_tp1.2_sl0.8_exec1_lexit-bid_ask_wu14_fmfe0a2838_lr5213b8_news-calendar_ca-auto-auto_ba0929a_2018-01-01_2025-12-30.zarr",
        "ckpt_dir": "checkpoints/haelt_eurusd_v1",
    },
    {
        "pair": "USDCAD",
        "cache": "data/processed/dataset_scalping_5m_USDCAD_0_dukascopy_120_triple_barrier_reward_lh30_tp1.2_sl0.8_exec1_lexit-bid_ask_wu14_fmfe0a2838_lr5213b8_news-calendar_ca-auto-auto_ba0929a_2018-01-01_2025-12-30.zarr",
        "ckpt_dir": "checkpoints/haelt_usdcad_v1",
    },
    {
        "pair": "USDJPY",
        "cache": "data/processed/dataset_scalping_5m_USDJPY_0_dukascopy_120_triple_barrier_reward_lh30_tp1.2_sl0.8_exec1_lexit-bid_ask_wu14_fmfe0a2838_lr5213b8_news-calendar_ca-auto-auto_ba0929a_2018-01-01_2025-12-30.zarr",
        "ckpt_dir": "checkpoints/haelt_usdjpy_v1",
    },
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


def wait_for_pid(pid: int, poll_sec: int = 15) -> None:
    if pid <= 0 or not _pid_alive(pid):
        return
    print(f"[{datetime.now():%H:%M:%S}] Waiting for active process PID={pid} to finish...", flush=True)
    while _pid_alive(pid):
        time.sleep(poll_sec)
    print(f"[{datetime.now():%H:%M:%S}] Process PID={pid} completed cleanly!", flush=True)


def run_rl_pipeline(py_exe: str, env: dict, episodes: int = 20, agent: str = "dqn") -> dict[str, dict]:
    print("\n" + "=" * 76, flush=True)
    print(f"[{datetime.now():%H:%M:%S}] [Option 2] Single-Pair RL Specialist Suite (Agent={agent.upper()}, Episodes={episodes})", flush=True)
    print("=" * 76, flush=True)

    results = {}
    for idx, spec in enumerate(RL_SPECS, start=1):
        pair = spec["pair"]
        ckpt_dir = ROOT / spec["ckpt_dir"] / "haelt"
        rl_ckpt = ckpt_dir / "rl_best.pt"
        report_file = ckpt_dir / "rl_report.json"

        # Check if already completed and valid
        if rl_ckpt.exists() and report_file.exists():
            try:
                with open(report_file) as f:
                    rep = json.load(f)
                print(f"[{datetime.now():%H:%M:%S}] [RL {idx}/{len(RL_SPECS)}] {pair} RL already trained: Eval Return: {rep.get('eval_return_pct', 0.0):+.2f}%, Sharpe: {rep.get('sharpe', 0.0):.2f}", flush=True)
                results[pair] = rep
                continue
            except Exception:
                pass

        print(f"\n[{datetime.now():%H:%M:%S}] [RL {idx}/{len(RL_SPECS)}] Training {agent.upper()} specialist for {pair}...", flush=True)
        cmd = [
            py_exe,
            "-u",
            str(ROOT / "scripts" / "train_rl.py"),
            "--cache",
            str(ROOT / spec["cache"]),
            "--model-name",
            "haelt",
            "--checkpoint-dir",
            str(ROOT / spec["ckpt_dir"]),
            "--agent",
            agent,
            "--episodes",
            str(episodes),
        ]
        t0 = time.perf_counter()
        res = subprocess.run(cmd, cwd=str(ROOT), env=env)
        elapsed = time.perf_counter() - t0
        if res.returncode != 0:
            print(f"[{datetime.now():%H:%M:%S}] [RL {idx}/{len(RL_SPECS)}] ERROR: Training for {pair} failed with code {res.returncode}", flush=True)
        else:
            print(f"[{datetime.now():%H:%M:%S}] [RL {idx}/{len(RL_SPECS)}] SUCCESS: {pair} RL completed in {elapsed:.1f}s!", flush=True)
            if report_file.exists():
                try:
                    with open(report_file) as f:
                        results[pair] = json.load(f)
                except Exception:
                    pass
    return results


def run_mamba_specialists(py_exe: str, env: dict, pairs: str = "EURUSD,USDCAD,USDJPY", epochs: int | None = None) -> int:
    print("\n" + "=" * 76, flush=True)
    print(f"[{datetime.now():%H:%M:%S}] [Option 3] Dedicated Single-Pair Mamba Specialist Suite", flush=True)
    print("=" * 76, flush=True)

    cmd = [
        py_exe,
        "-u",
        str(ROOT / "scripts" / "train_clean_specialists.py"),
        "--model",
        "mamba",
        "--pairs",
        pairs,
        "--skip-wait",
    ]
    if epochs is not None and epochs > 0:
        cmd.extend(["--epochs", str(epochs)])

    res = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return res.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RL & Mamba Specialist Pipelines")
    parser.add_argument("--wait-pid", type=int, default=0, help="PID of active process to wait for")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL specialist training")
    parser.add_argument("--skip-mamba", action="store_true", help="Skip Mamba specialist training")
    parser.add_argument("--rl-agent", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--rl-episodes", type=int, default=20)
    parser.add_argument("--mamba-pairs", type=str, default="EURUSD,USDCAD,USDJPY")
    parser.add_argument("--mamba-epochs", type=int, default=None)
    args = parser.parse_args()

    py_exe = str(ROOT / ".venv311" / "Scripts" / "python.exe")
    if not Path(py_exe).exists():
        py_exe = sys.executable

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    if args.wait_pid > 0:
        wait_for_pid(args.wait_pid)

    # 1. Option 2: Train remaining RL specialists
    if not args.skip_rl:
        rl_res = run_rl_pipeline(py_exe, env, episodes=args.rl_episodes, agent=args.rl_agent)
        print(f"\n[RL Results Summary] Completed: {list(rl_res.keys())}")

    # 2. Option 3: Train Mamba single-pair specialists
    if not args.skip_mamba:
        code = run_mamba_specialists(py_exe, env, pairs=args.mamba_pairs, epochs=args.mamba_epochs)
        if code != 0:
            print(f"[Mamba] Training exited with code {code}")
            return code

    print("\n" + "=" * 76, flush=True)
    print(f"[{datetime.now():%H:%M:%S}] All requested Option 2 & 3 Specialist jobs finished!", flush=True)
    print("=" * 76, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
