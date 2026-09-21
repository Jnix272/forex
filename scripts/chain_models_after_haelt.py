"""
scripts/chain_models_after_haelt.py
====================================
Watches for the completion of the flagship HAELT training process (PID 23604).
Once HAELT completes its final walk-forward fold (Fold 6) and writes its final
artifacts (train_summary.json / promotion_gate.json):
1. Terminates the legacy process (preventing it from running the old tft,transformer order).
2. Sequentially executes:
   - 1st: GNN (Cross-Asset Structure, ClusterTSCL pretrain, seq_len=80)
   - 2nd: Mamba (Long-Sequence Efficiency, ForecastPretext pretrain, seq_len=60)
   - 3rd: TFT (Temporal Fusion Transformer, MaskedRecon pretrain, seq_len=60)
3. Directs all outputs to D:\\forex-main\\train_out.log for continuous monitoring.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT = Path("d:/forex-main")
VENV_PYTHON = ROOT / ".venv311" / "Scripts" / "python.exe"
LOG_FILE = ROOT / "train_out.log"
QUEUE_LOG_FILE = ROOT / "auto_queue.log"
RAELT_DIR = ROOT / "checkpoints" / "forex_4pair_2015_2025_haelt" / "haelt"
SUMMARY_JSON = RAELT_DIR / "train_summary.json"
PROMOTION_JSON = RAELT_DIR / "promotion_gate.json"
QUEUE_STATUS_FILE = ROOT / "checkpoints" / "model_queue_status.json"

TARGET_PID = 23604

QUEUE = [
    {
        "model": "gnn",
        "description": "GNN (Cross-Asset Structure)",
        "cmd": [
            str(VENV_PYTHON),
            "-u",
            "-m",
            "training.train_gpu",
            "--config",
            "config/run.yaml",
            "--model",
            "gnn",
            "--pretrain",
            "--pretrain-method",
            "cluster",
            "--pretrain-epochs",
            "10",
            "--seq-len",
            "120",
            "--checkpoint-dir",
            "checkpoints/forex_4pair_2015_2025_gnn",
            "--resume",
        ],
    },
    {
        "model": "mamba",
        "description": "Mamba (Long Sequence Efficiency)",
        "cmd": [
            str(VENV_PYTHON),
            "-u",
            "-m",
            "training.train_gpu",
            "--config",
            "config/run.yaml",
            "--model",
            "mamba",
            "--pretrain",
            "--pretrain-method",
            "forecast",
            "--pretrain-epochs",
            "14",
            "--seq-len",
            "120",
            "--checkpoint-dir",
            "checkpoints/forex_4pair_2015_2025_mamba",
            "--resume",
        ],
    },
    {
        "model": "tft",
        "description": "TFT (Temporal Fusion Transformer)",
        "cmd": [
            str(VENV_PYTHON),
            "-u",
            "-m",
            "training.train_gpu",
            "--config",
            "config/run.yaml",
            "--model",
            "tft",
            "--pretrain",
            "--pretrain-method",
            "masked",
            "--pretrain-ablation",
            "false",
            "--seq-len",
            "120",
            "--lr",
            "0.001",
            "--checkpoint-dir",
            "checkpoints/forex_4pair_2015_2025_tft",
            "--resume",
        ],
    },
]


def log(msg: str) -> None:
    now_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{now_str}] [AutoQueue] {msg}"
    print(line, flush=True)
    try:
        with open(QUEUE_LOG_FILE, "a", encoding="utf-8") as fp:
            fp.write(f"{line}\n")
    except Exception:
        pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as fp:
            fp.write(f"\n{line}\n")
    except Exception:
        pass


def update_status(status_dict: dict) -> None:
    QUEUE_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_STATUS_FILE, "w", encoding="utf-8") as fp:
        json.dump(status_dict, fp, indent=2)


def is_haelt_complete() -> bool:
    """Check if HAELT has fully finished Fold 6 and written post-training artifacts."""
    # Check 1: train_summary.json exists with 7 folds or completed_at
    if SUMMARY_JSON.exists():
        try:
            with open(SUMMARY_JSON, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("n_folds", 0) >= 6 and data.get("model_name") == "haelt":
                return True
            if data.get("completed_at") and data.get("model_name") == "haelt":
                return True
        except Exception:
            pass

    # Check 2: promotion_gate.json exists for haelt
    if PROMOTION_JSON.exists():
        return True

    # Check 3: haelt_fold6_calibrated.pt or haelt_fold6_swa.pt exists
    if (RAELT_DIR / "haelt_fold6_calibrated.pt").exists() or (RAELT_DIR / "haelt_fold6_swa.pt").exists():
        return True

    # Check 4: Check if PID 23604 has terminated
    if not psutil.pid_exists(TARGET_PID):
        fold6_last = RAELT_DIR / "haelt_fold6_last.pt"
        if fold6_last.exists():
            return True

    return False


def terminate_legacy_process() -> None:
    """Gracefully stop legacy training process to prevent running old tft/transformer order."""
    log(f"Stopping legacy training process (PID {TARGET_PID})...")
    try:
        proc = psutil.Process(TARGET_PID)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except psutil.TimeoutExpired:
            proc.kill()
        log("Legacy training process terminated successfully.")
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        log(f"Legacy process PID {TARGET_PID} already terminated: {e}")


def run_queue() -> None:
    log("=" * 60)
    log("Auto-training daemon started. Monitoring HAELT PID 23604 for completion...")
    log(f"Configured queue: {' -> '.join(item['description'] for item in QUEUE)}")
    log("=" * 60)

    status = {
        "status": "waiting_for_haelt",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "haelt_pid": TARGET_PID,
        "completed_models": [],
        "current_model": None,
    }
    update_status(status)

    while True:
        if is_haelt_complete():
            log("DETECTED: HAELT training has completely finished!")
            break
        time.sleep(30)

    # HAELT has finished! Terminate old process if still alive
    if psutil.pid_exists(TARGET_PID):
        terminate_legacy_process()

    # Small delay for GPU memory cleanup
    time.sleep(10)

    # Execute queue
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)

    for idx, item in enumerate(QUEUE):
        model_name = item["model"]
        desc = item["description"]
        cmd = item["cmd"]

        log("=" * 60)
        log(f"STARTING QUEUE ITEM {idx + 1}/{len(QUEUE)}: {desc}")
        log(f"Command: {' '.join(cmd)}")
        log("=" * 60)

        status["status"] = f"training_{model_name}"
        status["current_model"] = model_name
        update_status(status)

        try:
            out_fp = open(LOG_FILE, "a", encoding="utf-8")
        except PermissionError:
            out_fp = open(QUEUE_LOG_FILE, "a", encoding="utf-8")
        with out_fp:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=out_fp,
                stderr=subprocess.STDOUT,
            )
            status["current_pid"] = proc.pid
            update_status(status)
            rc = proc.wait()

        if rc == 0:
            log(f"SUCCESS: {desc} completed with exit code 0!")
            status["completed_models"].append(model_name)
        else:
            log(f"WARNING: {desc} exited with code {rc}!")

        # Cooldown between models
        time.sleep(15)

    log("=" * 60)
    log("ALL QUEUED MODELS (GNN -> Mamba -> TFT) HAVE COMPLETED!")
    log("=" * 60)
    status["status"] = "all_models_completed"
    status["current_model"] = None
    update_status(status)


if __name__ == "__main__":
    run_queue()
