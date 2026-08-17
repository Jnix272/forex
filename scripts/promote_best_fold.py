"""
scripts/promote_best_fold.py
===========================
Standalone utility to scan cross-validation folds and promote the best one
to the primary checkpoint filename, applying a stability penalty for overfitting.

Usage:
    python scripts/promote_best_fold.py --model haelt --dir checkpoints/haelt_run
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional


def promote_best_fold(model_name: str, checkpoint_dir: str, metric: str = "sharpe"):
    ckpt_dir = Path(checkpoint_dir)
    use_sharpe = metric == "sharpe"
    best_fold: int | None = None
    best_adjusted_score: float | None = None
    folds_found = []

    print(f"[Promote] Scanning {ckpt_dir} for {model_name} folds with stability penalty...")

    # 1. Scan for fold config files: <model_name>_fold<N>_config.json
    for cfg_path in ckpt_dir.glob(f"{model_name}_fold*_config.json"):
        try:
            # Extract fold index from filename
            fold_str = cfg_path.name.replace(f"{model_name}_fold", "").replace("_config.json", "")
            fi = int(fold_str)

            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)

            if use_sharpe:
                raw_score = cfg.get("best_val_sharpe_proxy")
            else:
                raw = cfg.get("best_val_loss")
                raw_score = -raw if raw is not None else None  # Negate so higher=better

            train_val_gap = cfg.get("train_val_loss_gap", 0.0)

            if raw_score is not None:
                # Apply stability penalty (overfitting penalty)
                gap_penalty = (train_val_gap * 0.1) if train_val_gap is not None and train_val_gap > 0 else 0.0
                adjusted_score = raw_score - gap_penalty

                folds_found.append(
                    {
                        "fold": fi,
                        "raw_score": raw_score,
                        "train_val_loss_gap": train_val_gap,
                        "gap_penalty": gap_penalty,
                        "adjusted_score": adjusted_score,
                    }
                )

                if best_adjusted_score is None or adjusted_score > best_adjusted_score:
                    best_adjusted_score = adjusted_score
                    best_fold = fi
        except Exception as e:
            print(f"  [Warn] Failed to parse {cfg_path.name}: {e}")

    if best_fold is None:
        print(f"[Error] No valid folds found for {model_name} in {checkpoint_dir}")
        return

    # 2. Perform promotion
    src = ckpt_dir / f"{model_name}_fold{best_fold}_best.pt"
    dst = ckpt_dir / f"{model_name}_best.pt"

    if not src.exists():
        print(f"[Error] Best fold checkpoint not found: {src}")
        return

    shutil.copy2(src, dst)

    metric_label = "sharpe" if use_sharpe else "val_loss"
    winning_fold_data = next(f for f in folds_found if f["fold"] == best_fold)

    print(f"{'=' * 60}")
    print(f"  SUCCESS: Fold {best_fold} promoted to selected_stable_fold!")
    print(f"  Metric: {metric_label}")
    print(f"  Raw Score:      {winning_fold_data['raw_score']:.4f}")
    print(
        f"  Gap Penalty:   -{winning_fold_data['gap_penalty']:.4f} (train/val gap: {winning_fold_data['train_val_loss_gap']:.4f})"
    )
    print(f"  Adjusted Score: {winning_fold_data['adjusted_score']:.4f}")
    print(f"  Source: {src.name}")
    print(f"  Target: {dst.name}")
    print(f"{'=' * 60}")

    # 3. Write summary
    summary = {
        "model": model_name,
        "selected_stable_fold": best_fold,
        "metric": metric_label,
        "winning_adjusted_score": best_adjusted_score,
        "folds": folds_found,
    }
    summary_path = ckpt_dir / f"{model_name}_fold_selection.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Promote most stable CV fold to primary checkpoint.")
    p.add_argument("--model", type=str, required=True, help="Model name (e.g. haelt)")
    p.add_argument("--dir", type=str, required=True, help="Directory containing checkpoints")
    p.add_argument(
        "--metric",
        type=str,
        default="sharpe",
        choices=["sharpe", "loss"],
        help="Metric to use for promotion (default: sharpe)",
    )

    args = p.parse_args()
    promote_best_fold(args.model, args.dir, args.metric)


if __name__ == "__main__":
    main()
