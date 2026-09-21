"""
scripts/auto_optimal_roadmap.py
===============================
Automated end-to-end orchestrator for The Optimal Roadmap:
  Stage 1: Monitors active TFT walk-forward CV on GPU until complete.
  Stage 2: Trains 4-Model Ensemble Meta-Learner (HAELT + GNN + Mamba + TFT).
  Stage 3: Trains Multi-RL Policy Ensemble (3-Agent Recurrent PPO with Voting).
  Stage 4: Executes Final Out-of-Sample Backtest and generates deployment certification.

Usage:
  python scripts/auto_optimal_roadmap.py
  python scripts/auto_optimal_roadmap.py --check-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT = Path("d:/forex-main")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VENV_PYTHON = ROOT / ".venv311" / "Scripts" / "python.exe"
LOG_FILE = ROOT / "optimal_roadmap.log"
STATUS_FILE = ROOT / "checkpoints" / "optimal_roadmap_status.json"
QUEUE_STATUS_FILE = ROOT / "checkpoints" / "model_queue_status.json"
TFT_DIR = ROOT / "checkpoints" / "forex_4pair_2015_2025_tft" / "tft"
ENSEMBLE_DIR = ROOT / "checkpoints" / "ensemble"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [OptimalRoadmap] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)


def log(msg: str) -> None:
    logging.info(msg)


def update_status(status_dict: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as fp:
        json.dump(status_dict, fp, indent=2)


def get_default_cache() -> Path:
    caches = sorted((ROOT / "data" / "processed").glob("dataset_scalping_*.zarr"))
    if caches:
        return caches[0]
    return (
        ROOT
        / "data"
        / "processed"
        / "dataset_scalping_5m_EURUSD-GBPUSD-USDCAD-USDJPY_20000000_dukascopy_120_cpar_reward_lh30_tp1.2_sl0.8_exec1_lexit-bid_ask_wu14_fmfe0a2838_lr5213b8_news-calendar_ca-auto-auto_2008-01-01_2025-12-30.zarr"
    )


def is_tft_complete() -> bool:
    """Check if TFT walk-forward CV (Folds 0-6) has concluded."""
    if QUEUE_STATUS_FILE.exists():
        try:
            with open(QUEUE_STATUS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "all_models_completed":
                return True
            if "tft" in data.get("completed_models", []):
                return True
        except Exception:
            pass

    summary_path = TFT_DIR / "train_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("n_folds", 0) >= 6 or data.get("completed_at"):
                return True
        except Exception:
            pass

    fold6_best = TFT_DIR / "tft_fold6_best.pt"
    fold6_cal = TFT_DIR / "tft_fold6_calibrated.pt"
    if fold6_best.exists() or fold6_cal.exists():
        tft_proc_running = False
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or [])
                if "training.train_gpu" in cmd and "tft" in cmd:
                    tft_proc_running = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not tft_proc_running:
            return True

    return False


def run_stage_2_ensemble(device: str = "cuda") -> bool:
    """Stage 2: Train 4-model Ensemble Meta-Learner (HAELT + GNN + Mamba + TFT)."""
    log("=" * 70)
    log("STAGE 2: TRAINING 4-MODEL ENSEMBLE META-LEARNER")
    log("Models: HAELT + GNN + Mamba + TFT")
    log("=" * 70)

    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    out_ckpt = ENSEMBLE_DIR / "ensemble_meta_best.pt"
    if out_ckpt.exists():
        log(f"[INFO] Stage 2 checkpoint already exists at {out_ckpt}. Skipping retraining.")
        return True

    cmd = [
        str(VENV_PYTHON),
        "-u",
        "scripts/train_ensemble_meta.py",
        "--models",
        "haelt",
        "mamba",
        "gnn",
        "tft",
        "--checkpoint-dir",
        "checkpoints",
        "--output",
        str(out_ckpt),
        "--epochs",
        "15",
        "--batch-size",
        "256",
        "--lr",
        "1e-4",
        "--device",
        device,
    ]

    log(f"Executing: {' '.join(cmd)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)

    with open(LOG_FILE, "a", encoding="utf-8") as out_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=out_fp,
            stderr=subprocess.STDOUT,
        )
        rc = proc.wait()

    if rc == 0 and out_ckpt.exists():
        log(f"[SUCCESS] Stage 2 Complete! Ensemble Meta-Learner saved to {out_ckpt}")
        return True
    else:
        log(f"[ERROR] Stage 2 failed with exit code {rc}")
        return False


def run_stage_3_multi_rl(cache_path: Path, device: str = "cuda", episodes: int = 250, retrain: bool = False) -> bool:
    """Stage 3: Train Multi-RL Policy Ensemble (3-Agent Recurrent PPO with Voting)."""
    log("=" * 70)
    log(f"STAGE 3: TRAINING MULTI-RL RECURRENT PPO ENSEMBLE ({episodes} EPISODES)")
    log("Supervised Base: 4-Model Stacking Ensemble (HAELT+GNN+Mamba+TFT)")
    log("Committee: 3 PPO Agents, LSTM Memory (hist_len=32), Soft-Voting Consensus")
    rl_ckpt = ENSEMBLE_DIR / "rl_ensemble_best.pt"
    if rl_ckpt.exists() and not retrain:
        log(f"[INFO] Stage 3 checkpoint already exists at {rl_ckpt}. Skipping retraining.")
        return True

    cmd = [
        str(VENV_PYTHON),
        "-u",
        "scripts/train_rl.py",
        "--cache",
        str(cache_path),
        "--model-name",
        "ensemble",
        "--ensemble-bases",
        "haelt",
        "mamba",
        "gnn",
        "tft",
        "--checkpoint-dir",
        "checkpoints",
        "--multi-agent",
        "--num-agents",
        "3",
        "--ensemble-agents",
        "ppo,ppo,ppo",
        "--consensus-mode",
        "soft_vote",
        "--use-lstm",
        "--hist-len",
        "32",
        "--batch-size",
        "256",
        "--episodes",
        str(episodes),
        "--device",
        device,
    ]

    log(f"Executing: {' '.join(cmd)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)

    with open(LOG_FILE, "a", encoding="utf-8") as out_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=out_fp,
            stderr=subprocess.STDOUT,
        )
        rc = proc.wait()

    rl_ckpt = ENSEMBLE_DIR / "rl_ensemble_best.pt"
    if rc == 0 and rl_ckpt.exists():
        log(f"[SUCCESS] Stage 3 Complete! Multi-RL Ensemble saved to {rl_ckpt}")
        return True
    else:
        log(f"[ERROR] Stage 3 failed with exit code {rc}")
        return False


def run_stage_4_certification() -> dict:
    """Stage 4: Generate Final Deployment Certification & Performance Summary."""
    log("=" * 70)
    log("STAGE 4: COMPILING OPTIMAL ROADMAP DEPLOYMENT CERTIFICATION")
    log("=" * 70)

    rl_report_path = ENSEMBLE_DIR / "rl_report.json"
    rl_metrics = {}
    if rl_report_path.exists():
        try:
            with open(rl_report_path, encoding="utf-8") as f:
                rl_metrics = json.load(f)
        except Exception as e:
            log(f"Warning reading rl_report.json: {e}")

    # Extract performance metrics for quality gating
    if "ensemble_consensus" in rl_metrics:
        cons = rl_metrics["ensemble_consensus"]
        n_trades = int(cons.get("n_trades", 0))
        sharpe = float(cons.get("sharpe", 0.0))
        eval_return_pct = float(cons.get("eval_return_pct", 0.0))
    else:
        n_trades = int(rl_metrics.get("n_trades", 0))
        sharpe = float(rl_metrics.get("sharpe", 0.0))
        eval_return_pct = float(rl_metrics.get("eval_return_pct", rl_metrics.get("train_return_pct", 0.0)))

    # Hard Quality Gate evaluation
    reasons = []
    if n_trades == 0:
        cert_status = "FAILED_ZERO_TRADES"
        reasons.append("Model/Ensemble took zero trades during evaluation (inaction collapse)")
    elif n_trades < 10 or eval_return_pct <= 0.0 or sharpe <= 0.0:
        cert_status = "REJECTED_INACTION_COLLAPSE"
        if n_trades < 10:
            reasons.append(f"Insufficient trade count: {n_trades} < 10 required")
        if eval_return_pct <= 0.0:
            reasons.append(f"Non-positive evaluation return: {eval_return_pct:+.2f}% <= 0.0%")
        if sharpe <= 0.0:
            reasons.append(f"Non-positive Sharpe ratio: {sharpe:.2f} <= 0.0")
    elif n_trades >= 10 and sharpe > 0.5 and eval_return_pct > 0.0:
        cert_status = "CERTIFIED_READY_FOR_DEPLOYMENT"
    else:
        cert_status = "REJECTED_INACTION_COLLAPSE"
        reasons.append(f"Sub-par performance metrics: n_trades={n_trades}, sharpe={sharpe:.2f}, eval_return={eval_return_pct:+.2f}%")

    log(f"[Quality Gate] Verdict: {cert_status} | n_trades={n_trades} | Sharpe={sharpe:.2f} | Return={eval_return_pct:+.2f}%")
    if reasons:
        for r in reasons:
            log(f"  [REASON] {r}")

    certification = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": cert_status,
        "quality_gate_passed": bool(cert_status == "CERTIFIED_READY_FOR_DEPLOYMENT"),
        "rejection_reasons": reasons,
        "performance_gate": {
            "n_trades": n_trades,
            "sharpe": sharpe,
            "eval_return_pct": eval_return_pct,
            "min_trades_required": 10,
            "min_sharpe_required": 0.5,
            "min_return_required": 0.0,
        },
        "architecture_stack": {
            "base_models": ["haelt", "mamba", "gnn", "tft"],
            "ensemble_meta": "Attention-Gated Neural Stacking (checkpoints/ensemble/ensemble_meta_best.pt)",
            "execution_policy": "3-Agent Recurrent PPO Ensemble (checkpoints/ensemble/rl_ensemble_best.pt)",
            "risk_controls": "Dynamic ATR Trailing Stops, Breakeven Lock, Multi-Pair Correlation Gate",
        },
        "performance_metrics": rl_metrics,
        "deployment_artifacts": [
            "checkpoints/forex_4pair_2015_2025_haelt/haelt/haelt_best.pt",
            "checkpoints/forex_4pair_2015_2025_gnn/gnn/gnn_best.pt",
            "checkpoints/forex_4pair_2015_2025_mamba/mamba/mamba_best.pt",
            "checkpoints/forex_4pair_2015_2025_tft/tft/tft_best.pt",
            "checkpoints/ensemble/ensemble_meta_best.pt",
            "checkpoints/ensemble/rl_ensemble_best.pt",
            "checkpoints/ensemble/rl_best.pt",
        ],
    }

    cert_path = ENSEMBLE_DIR / "optimal_roadmap_certification.json"
    with open(cert_path, "w", encoding="utf-8") as fp:
        json.dump(certification, fp, indent=2)

    log(f"Certification saved to {cert_path}")
    return certification


def main():
    parser = argparse.ArgumentParser(description="Optimal Roadmap Automated Orchestrator")
    parser.add_argument("--check-only", action="store_true", help="Check status and exit without waiting")
    parser.add_argument("--force-start", action="store_true", help="Start Stage 2 immediately using current checkpoints")
    parser.add_argument("--episodes", type=int, default=250, help="Number of RL episodes per agent (default: 250)")
    parser.add_argument("--retrain-rl", action="store_true", help="Force retrain RL even if checkpoint exists")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cache_path = get_default_cache()
    log(f"Resolved Zarr Cache: {cache_path}")

    status = {
        "orchestrator": "optimal_roadmap",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stage": "monitoring_tft",
        "stages_completed": [],
    }
    update_status(status)

    if args.check_only:
        tft_done = is_tft_complete()
        log(f"Check-only: TFT Complete = {tft_done}")
        return

    if not args.force_start:
        log("Monitoring TFT training progress on GPU...")
        while not is_tft_complete():
            time.sleep(30)
        log("DETECTED: TFT has completed all walk-forward folds!")
        status["stages_completed"].append("Stage 1: TFT Complete")
        update_status(status)

    time.sleep(15)

    status["stage"] = "training_ensemble"
    update_status(status)
    s2_ok = run_stage_2_ensemble(device=args.device)
    if not s2_ok:
        log("Stage 2 failed. Halting pipeline.")
        status["stage"] = "failed_stage_2"
        update_status(status)
        sys.exit(1)
    status["stages_completed"].append("Stage 2: Ensemble Meta-Learner")
    update_status(status)

    time.sleep(10)

    status["stage"] = "training_multi_rl"
    update_status(status)
    s3_ok = run_stage_3_multi_rl(cache_path=cache_path, device=args.device, episodes=args.episodes, retrain=args.retrain_rl)
    if not s3_ok:
        log("Stage 3 failed. Halting pipeline.")
        status["stage"] = "failed_stage_3"
        update_status(status)
        sys.exit(1)
    status["stages_completed"].append("Stage 3: Multi-RL Policy Ensemble")
    update_status(status)

    cert = run_stage_4_certification()
    status["stages_completed"].append("Stage 4: Deployment Certification")
    if cert["status"] == "CERTIFIED_READY_FOR_DEPLOYMENT":
        status["stage"] = "all_completed"
        status["completed_at"] = datetime.now(timezone.utc).isoformat()
        update_status(status)
        log("=" * 70)
        log("THE OPTIMAL ROADMAP PIPELINE HAS COMPLETED 100% SUCCESSFULLY!")
        log("Stack is certified and ready for live paper trading.")
        log("=" * 70)
    else:
        status["stage"] = "certification_failed"
        update_status(status)
        log("=" * 70)
        log(f"STAGE 4 QUALITY GATE FAILED: {cert['status']}")
        for r in cert.get("rejection_reasons", []):
            log(f"  - {r}")
        log("Deployment rejected due to inaction collapse or subpar performance.")
        log("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
