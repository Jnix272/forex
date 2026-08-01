"""
scripts/train.py
================
Simple entry point for GPU training — wraps training/train_gpu.py.

Usage (PowerShell):
  .\\.venv-gpu\\Scripts\\python.exe scripts\\train.py
  .\\.venv-gpu\\Scripts\\python.exe scripts\\train.py --quick
  .\\.venv-gpu\\Scripts\\python.exe scripts\\train.py --full --pretrain --resume
  .\\.venv-gpu\\Scripts\\python.exe scripts\\train.py --rebuild-cache
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_CONFIG = _ROOT / "config" / "run.yaml"
NEWS_CSV = _ROOT / "data" / "raw" / "news" / "historical_news_combined.parquet"
ECO_CSV = _ROOT / "data" / "raw" / "eco_calendar" / "events.csv"
COT_PARQUET = _ROOT / "data" / "raw" / "cot" / "cot_financials_cleaned.parquet"
PRICE_ROOT = _ROOT / "data" / "raw" / "dukascopy"


def _python_exe() -> str:
    from scripts._python_env import python_exe as _resolve
    return _resolve()


def _load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _configured_pairs(cfg: dict) -> list[str]:
    data = cfg.get("data") or {}
    pairs = data.get("pairs") or [data.get("pair", "EURUSD")]
    return [str(p).upper().replace("/", "") for p in pairs]


def _pair_has_price_data(pair: str) -> bool:
    pair_dir = PRICE_ROOT / pair
    if not pair_dir.is_dir():
        return False
    return any(pair_dir.rglob("*.parquet")) or any(pair_dir.rglob("*.bi5"))


def check_data_paths(cfg: dict) -> list[str]:
    """Return fatal error messages (empty list = OK to train)."""
    errors: list[str] = []
    pairs = _configured_pairs(cfg)
    missing_pairs = [p for p in pairs if not _pair_has_price_data(p)]
    if missing_pairs:
        errors.append(
            "Price data missing for: "
            + ", ".join(missing_pairs)
            + f"\n  Expected under: {PRICE_ROOT / '<PAIR>'}/"
            + f"\n  Run: {_python_exe()} scripts/download_all.py"
        )
    return errors


def warn_optional_data(cfg: dict) -> list[str]:
    """Return non-fatal warnings about auxiliary datasets."""
    warnings: list[str] = []
    news_mode = str((cfg.get("news") or {}).get("historical_mode", "off")).lower()

    if not NEWS_CSV.is_file():
        warnings.append(
            f"News CSV not found: {NEWS_CSV}\n"
            "  GDELT headlines will be neutral placeholders unless you run download_all."
        )
    if not ECO_CSV.is_file():
        warnings.append(
            f"Economic calendar not found: {ECO_CSV}\n"
            "  Eco features (eco_act/eco_fc/eco_surprise) will be empty."
        )
    if not COT_PARQUET.is_file():
        warnings.append(
            f"COT parquet not found: {COT_PARQUET}\n"
            "  COT positioning features will be unavailable."
        )
    if news_mode == "full" and (not NEWS_CSV.is_file() or not ECO_CSV.is_file()):
        warnings.append(
            f"config news.historical_mode is '{news_mode}' but news/calendar files are missing."
        )
    return warnings


def print_run_summary(cfg: dict, args: argparse.Namespace, train_cmd: list[str]) -> None:
    data = cfg.get("data") or {}
    training = cfg.get("training") or {}
    model = cfg.get("model") or {}
    strategy = cfg.get("strategy") or {}

    mode = "quick smoke" if args.quick else ("full" if args.full else "supervised (config defaults)")
    extras = []
    if args.pretrain:
        extras.append("pretrain")
    if args.rl:
        extras.append("RL")
    if args.resume:
        extras.append("resume")
    if getattr(args, "warm_start", False):
        extras.append("warm-start")
    if getattr(args, "rebuild_cache", False):
        extras.append("rebuild-cache")
    if getattr(args, "fair_sweep", False):
        extras.append("fair-sweep")
    if args.model_profile is True:
        extras.append("model-profile")
    elif args.model_profile is False:
        extras.append("no-model-profile")

    print("\n" + "=" * 66)
    print("  Forex Scaling Model — Training")
    print("=" * 66)
    print(f"  config : {args.config}")
    print(f"  mode   : {mode}" + (f" + {', '.join(extras)}" if extras else ""))
    print(f"  model  : {model.get('name', 'haelt')}")
    print(f"  strat  : {strategy.get('mode', 'scalping')} ({strategy.get('bar_freq', '1min')})")
    print(f"  pairs  : {', '.join(_configured_pairs(cfg))}")
    print(f"  range  : {data.get('start', '?')} -> {data.get('end', '?')}")
    print(f"  epochs : {training.get('epochs', '?')}")
    print(f"  source : {data.get('source', 'dukascopy')}")
    print(f"\n  command: {' '.join(train_cmd)}")
    print("=" * 66 + "\n", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the forex scaling model (wrapper around training/train_gpu.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="YAML run config passed to train_gpu.py")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="Smoke run: <=8 epochs, <=2 folds, no ensemble/RL")
    mode.add_argument("--full", action="store_true",
                      help="Use config as-is (default behaviour)")
    p.add_argument("--pretrain", action="store_true", help="Enable contrastive pre-training")
    p.add_argument("--ablate-pretrain", action="store_true", help="Run ablation test on pretraining")
    p.add_argument("--rl", action="store_true", help="Enable RL training after supervised")
    p.add_argument("--resume", action="store_true",
                   help="Resume interrupted run from *_last.pt (optimizer + epoch state)")
    p.add_argument("--warm-start", action="store_true",
                   help="Load production/best weights and continue training (fine-tune)")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force dataset cache rebuild (passes --force-rebuild to train_gpu.py)")
    p.add_argument("--skip-data-check", action="store_true",
                   help="Skip pre-flight data path checks")
    p.add_argument("--fair-sweep", action="store_true",
                   help="Fair architecture bake-off: pass --no-model-profile to train_gpu")
    p.add_argument("--model-profile", dest="model_profile", action="store_const", const=True,
                   default=None, help="Force per-architecture profiles on (train_gpu default: on)")
    p.add_argument("--no-model-profile", dest="model_profile", action="store_const", const=False,
                   help="Disable per-architecture profiles (shared run.yaml hyperparams)")
    p.add_argument("--teacher-model", type=str, default=None,
                   help="Name of teacher model to distill from (e.g., haelt, ensemble)")
    p.add_argument("--distill-weight", type=float, default=0.5,
                   help="Weight of distillation loss relative to supervised loss")
    p.add_argument("--distill-temperature", type=float, default=2.0,
                   help="Temperature for distillation (if applicable)")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra flags forwarded to train_gpu.py (prefix with --)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.config)

    if not args.skip_data_check:
        fatal = check_data_paths(cfg)
        if fatal:
            print("\n[train] Cannot start — required data is missing:\n", flush=True)
            for msg in fatal:
                print(f"  • {msg}\n", flush=True)
            return 1
        for msg in warn_optional_data(cfg):
            print(f"[train] WARNING: {msg}\n", flush=True)

    py = _python_exe()
    train_cmd = [py, str(_ROOT / "training" / "train_gpu.py"), "--config", str(args.config)]
    if args.teacher_model:
        train_cmd.extend(["--teacher-model", args.teacher_model])
    if args.distill_weight != 0.5:
        train_cmd.extend(["--distill-weight", str(args.distill_weight)])
    if args.distill_temperature != 2.0:
        train_cmd.extend(["--distill-temperature", str(args.distill_temperature)])
        
    if args.quick:
        train_cmd.append("--quick-mode")
    if args.pretrain:
        train_cmd.append("--pretrain")
    if args.ablate_pretrain:
        train_cmd.append("--ablate-pretrain")
    if args.rl:
        train_cmd.append("--rl-train")
    if args.resume:
        train_cmd.append("--resume")
    if args.warm_start:
        train_cmd.append("--finetune-warm-start")
    if args.rebuild_cache:
        train_cmd.append("--force-rebuild")
    if args.fair_sweep:
        train_cmd.append("--fair-sweep")
    if args.model_profile is True:
        train_cmd.append("--model-profile")
    elif args.model_profile is False:
        train_cmd.append("--no-model-profile")
    if args.extra:
        train_cmd.extend(args.extra)

    print_run_summary(cfg, args, train_cmd)

    if args.ablate_pretrain:
        import json

        print("\n" + "=" * 66)
        print("  [Ablation] Pass 1/2: No Pretrain")
        print("=" * 66 + "\n")
        cmd_no_pt = [x for x in train_cmd if x != "--pretrain" and x != "--ablate-pretrain"]
        res_no_pt = subprocess.run(cmd_no_pt, cwd=str(_ROOT))
        if res_no_pt.returncode != 0:
            return int(res_no_pt.returncode)
            
        # Extract metrics
        ckpt_dir = _ROOT / "checkpoints"  # Fallback
        if "training" in cfg and "checkpoint_dir" in cfg["training"]:
            ckpt_dir = _ROOT / cfg["training"]["checkpoint_dir"]
            
        model_name = cfg.get("model", {}).get("name", "haelt")
        summary_path = ckpt_dir / model_name / "train_summary.json"
        
        summary_no_pt = {}
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary_no_pt = json.load(f)
                
        print("\n" + "=" * 66)
        print("  [Ablation] Pass 2/2: With Pretrain")
        print("=" * 66 + "\n")
        cmd_pt = [x for x in train_cmd if x != "--ablate-pretrain"]
        if "--pretrain" not in cmd_pt:
            cmd_pt.append("--pretrain")
            
        res_pt = subprocess.run(cmd_pt, cwd=str(_ROOT))
        if res_pt.returncode != 0:
            return int(res_pt.returncode)
            
        summary_pt = {}
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary_pt = json.load(f)
                
        val_sharpe_diff = (summary_pt.get("best_val_sharpe") or 0) - (summary_no_pt.get("best_val_sharpe") or 0)
        val_loss_diff = (summary_pt.get("best_val_loss") or 0) - (summary_no_pt.get("best_val_loss") or 0)
        gen_gap_diff = (summary_pt.get("gen_gap_final") or 0) - (summary_no_pt.get("gen_gap_final") or 0)
        if val_sharpe_diff > 0 and val_loss_diff <= 0 and gen_gap_diff <= 0.02:
            verdict = "pretrain_helped"
        elif val_sharpe_diff < 0 or (val_loss_diff > 0 and val_sharpe_diff <= 0):
            verdict = "pretrain_hurt"
        else:
            verdict = "mixed"

        ablation_results = {
            "model_name": model_name,
            "comparison": {
                "verdict": verdict,
                "deltas_pretrain_minus_baseline": {
                    "best_val_sharpe": val_sharpe_diff,
                    "best_val_loss": val_loss_diff,
                    "final_train_val_gap": gen_gap_diff,
                },
            },
            "no_pretrain": summary_no_pt,
            "with_pretrain": summary_pt,
            "improvement": {
                "val_sharpe_diff": val_sharpe_diff,
                "val_loss_diff": val_loss_diff,
                "gen_gap_diff": gen_gap_diff,
            }
        }
        
        ablation_file = ckpt_dir / model_name / "pretrain_ablation.json"
        with open(ablation_file, "w") as f:
            json.dump(ablation_results, f, indent=2)
            
        print(f"\n[Ablation] Complete. Saved results to {ablation_file}")
        return 0
    else:
        result = subprocess.run(train_cmd, cwd=str(_ROOT))
        return int(result.returncode)

if __name__ == "__main__":
    raise SystemExit(main())
