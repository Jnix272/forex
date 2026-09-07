"""Checkpoint / walk-forward CV resume-status helpers."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from training.core import _log_warn


def _model_completion_status(model_name: str, checkpoint_dir: str | Path) -> tuple[bool, str]:
    """Return whether an all-model member appears fully trained.

    Crash checkpoints are deliberately ignored: they prove the model started,
    not that it produced a clean resume/best artifact.
    """
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    best_paths = [
        ckpt_dir / f"{model}_best.pt",
        ckpt_dir / model / f"{model}_best.pt",
    ]
    has_best = any(p.exists() for p in best_paths)
    has_manifest = (ckpt_dir / "manifest.json").exists()
    has_train_summary = (ckpt_dir / "train_summary.json").exists()
    has_fold_selection = (ckpt_dir / "fold_selection.json").exists()
    has_deployment = (ckpt_dir / "deployment.json").exists()
    crash_files = sorted(ckpt_dir.glob(f"*{model}*_crash.pt"))

    if has_deployment and (has_best or has_manifest):
        return True, "deployment.json + completed checkpoint metadata"
    if has_fold_selection and has_best:
        return True, "fold_selection.json + best checkpoint"
    if has_manifest and has_best:
        return True, "manifest.json + best checkpoint"
    if has_train_summary and has_best and not has_fold_selection:
        return True, "train_summary.json + best checkpoint"
    if crash_files and not has_best:
        return False, f"crash checkpoint only ({crash_files[-1].name})"
    if has_best:
        return False, "best checkpoint exists but completion metadata is missing"
    return False, "no completed artifacts"


def _baseline_ablation_completion_status(model_name: str, checkpoint_dir: str | Path, args) -> tuple[bool, str]:
    """Return whether the no-pretrain baseline proof for a model is already complete.

    Baseline ablation artifacts live under <checkpoint_dir>/baseline and do not go
    through the full model promotion/deployment path, so the generic completion
    helper is too weak here. For walk-forward runs we require every expected fold
    best checkpoint before skipping baseline on resume.
    """
    model = str(model_name).lower().strip()
    baseline_dir = Path(checkpoint_dir) / "baseline"
    if not baseline_dir.exists():
        return False, "baseline artifact directory missing"

    walk_forward = bool(getattr(args, "walk_forward_cv", False))
    if walk_forward:
        n_folds = max(1, int(getattr(args, "walk_forward_folds", 1)))
        missing = []
        for fi in range(n_folds):
            fold_best = baseline_dir / f"baseline_{model}_fold{fi}_best.pt"
            if not fold_best.exists():
                missing.append(fold_best.name)
        if not missing:
            return True, f"all {n_folds} baseline fold checkpoints present"
        return False, f"missing baseline fold checkpoints: {', '.join(missing[:3])}" + (
            " ..." if len(missing) > 3 else ""
        )

    single_best = baseline_dir / f"baseline_{model}_best.pt"
    if single_best.exists():
        return True, "single-split baseline checkpoint present"
    return False, f"missing {single_best.name}"


def _supervised_resume_status(model_name: str, checkpoint_dir: str | Path, args) -> tuple[bool, str]:
    """Return whether supervised training already started for this model."""
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    walk_forward = bool(getattr(args, "walk_forward_cv", False))

    if walk_forward:
        n_folds = max(1, int(getattr(args, "walk_forward_folds", 1)))
        last_paths = [ckpt_dir / f"{model}_fold{fi}_last.pt" for fi in range(n_folds)]
        best_paths = [ckpt_dir / f"{model}_fold{fi}_best.pt" for fi in range(n_folds)]
        existing_last = [p for p in last_paths if p.exists()]
        if existing_last:
            latest = max(existing_last, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
            return True, f"supervised resume checkpoint present ({latest.name})"
        existing_best = [p for p in best_paths if p.exists()]
        if existing_best:
            latest = max(existing_best, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
            return True, f"supervised fold checkpoint present ({latest.name})"
        return False, "no supervised fold checkpoints found"

    last_path = ckpt_dir / f"{model}_last.pt"
    best_path = ckpt_dir / f"{model}_best.pt"
    if last_path.exists():
        return True, f"supervised resume checkpoint present ({last_path.name})"
    if best_path.exists():
        return True, f"supervised best checkpoint present ({best_path.name})"
    return False, "no supervised checkpoints found"


def _latest_resumable_fold(model_name: str, checkpoint_dir: str | Path, n_folds: int) -> int | None:
    """Return the latest fold index with a resumable supervised checkpoint."""
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    candidates: list[tuple[float, int]] = []
    for fi in range(max(1, int(n_folds))):
        for path in (
            ckpt_dir / f"{model}_fold{fi}_last.pt",
            ckpt_dir / f"{model}_fold{fi}_best.pt",
        ):
            if path.exists():
                try:
                    candidates.append((path.stat().st_mtime, fi))
                except Exception:
                    candidates.append((0.0, fi))
                break
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[-1][1])


def _load_cv_fold_entry(
    model_name: str,
    checkpoint_dir: str | Path,
    fold_idx: int,
) -> dict | None:
    """Rebuild one walk-forward fold summary from its checkpoint artifacts."""
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    fold_suffix = f"_fold{int(fold_idx)}"
    last_path = ckpt_dir / f"{model}{fold_suffix}_last.pt"
    best_path = ckpt_dir / f"{model}{fold_suffix}_best.pt"
    ckpt_path = last_path if last_path.exists() else (best_path if best_path.exists() else None)
    if ckpt_path is None:
        return None
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    history = ck.get("history")
    if not isinstance(history, dict) or not history:
        return None
    best_metric = None
    if history.get("val_sharpe"):
        best_metric = float(max(history["val_sharpe"]))
    elif history.get("val_loss"):
        best_metric = float(min(history["val_loss"]))
    elif ck.get("best_sharpe") is not None:
        best_metric = float(ck["best_sharpe"])
    elif ck.get("best_val_loss") is not None:
        best_metric = float(ck["best_val_loss"])
    return {"fold": int(fold_idx), "best_metric": best_metric, "history": history}


def _load_walk_forward_resume_history(
    model_name: str,
    checkpoint_dir: str | Path,
    log_dir: Path,
    run_name_slug: str,
    model_slug: str,
    start_fold: int,
) -> list[dict]:
    """Load completed fold metrics for folds [0, start_fold) when resuming walk-forward CV."""
    if start_fold <= 0:
        return []
    entries: list[dict] = []
    cv_path = Path(log_dir) / f"{run_name_slug}_{model_slug}_cv.json"
    if cv_path.exists():
        try:
            data = json.loads(cv_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries = [e for e in data if isinstance(e, dict) and int(e.get("fold", -1)) < int(start_fold)]
        except Exception as exc:
            _log_warn(f"[WalkForward] Could not read prior cv.json ({exc}); rebuilding from checkpoints.")
    loaded_folds = {int(e.get("fold", -1)) for e in entries}
    for fi in range(int(start_fold)):
        if fi in loaded_folds:
            continue
        entry = _load_cv_fold_entry(model_name, checkpoint_dir, fi)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda e: int(e.get("fold", 0)))
    return entries
