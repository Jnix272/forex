"""
training/train_min.py

Minimal training entrypoint:
- Load cached dataset (or build if missing)
- Train a single model (no pretrain / no ensemble / no RL / no walk-forward)
- Save best checkpoint

This intentionally reuses the existing dataset + model code from `training/train_gpu.py`,
but implements a much smaller training loop.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required. Install torch in your venv.") from e


# Reuse the proven plumbing (cache load/build, datasets, model factory, loss helpers).
from training.train_gpu import (  # noqa: E402
    build_dataset_chunked,
    build_model,
    build_criterion,
    train_epoch,
    validate_epoch,
    ZarrStreamDataset,
    parse_args,
)


@dataclass
class _EarlyStop:
    best_metric: float
    best_epoch: int
    bad_epochs: int = 0


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_config(dev: torch.device, use_amp: bool) -> Tuple[bool, torch.dtype, torch.amp.GradScaler]:
    if not use_amp or dev.type != "cuda":
        return False, torch.float32, torch.amp.GradScaler(enabled=False)

    major, _minor = torch.cuda.get_device_capability(dev)
    amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    return True, amp_dtype, scaler


def _make_loaders(args, cache_path: str, n_samples: int) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    split = int(n_samples * (1.0 - float(args.val_split)))
    train_idx = np.arange(0, split, dtype=np.int64)
    val_idx = np.arange(split, n_samples, dtype=np.int64)

    train_ds = ZarrStreamDataset(cache_path, train_idx, shuffle_chunks=True)
    val_ds = ZarrStreamDataset(cache_path, np.sort(val_idx), shuffle_chunks=False)

    nw = 0 if os.name == "nt" else int(getattr(args, "num_workers", 0) or 0)
    pf = int(getattr(args, "prefetch_factor", 2) or 2) if nw > 0 else None
    pin_mem = os.name != "nt"

    # Keep it simple/robust on Windows: no persistent workers by default.
    train_dl = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=False,  # iterable dataset handles internal shuffles
        num_workers=nw,
        pin_memory=pin_mem,
        persistent_workers=False,
        prefetch_factor=pf,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(args.batch_size) * 2,
        shuffle=False,
        num_workers=0,  # deterministic and avoids extra mmap handles
        pin_memory=pin_mem,
    )
    return train_dl, val_dl, train_idx, val_idx


def _save_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer, epoch: int, metrics: dict, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "args": vars(args),
    }
    torch.save(obj, str(path))


def _model_checkpoint_dir(base_dir: str, model_name: str) -> Path:
    base = Path(str(base_dir or "checkpoints")).expanduser()
    return base if base.name.lower() == model_name.lower() else base / model_name


def main() -> int:
    args = parse_args()

    # Force "only necessary things"
    args.all_models = False
    args.walk_forward_cv = False
    args.pretrain = False
    args.train_ensemble = False
    args.rl_train = False

    dev = _device()

    cache_path, n_samples, n_features, _scaler = build_dataset_chunked(args)
    train_dl, val_dl, train_idx, _val_idx = _make_loaders(args, cache_path, n_samples)

    model_name = str(getattr(args, "model", "tft")).lower()
    classification = str(getattr(args, "loss", "cross_entropy")).lower() == "cross_entropy"

    model = build_model(model_name, n_features, args).to(dev)

    use_amp, amp_dtype, scaler = _amp_config(dev, bool(getattr(args, "amp", True)))

    crit = build_criterion(
        args,
        dev,
        cache_path=cache_path if classification else None,
        train_idx=train_idx if classification else None,
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(args, "lr", 1e-4)),
        weight_decay=float(getattr(args, "weight_decay", 1e-4)),
    )

    ckpt_dir = _model_checkpoint_dir(getattr(args, "checkpoint_dir", "checkpoints"), model_name)
    ckpt_path = ckpt_dir / f"{model_name}_min_best.pt"

    stop_metric = str(getattr(args, "early_stop_metric", "sharpe")).lower()
    higher_is_better = stop_metric == "sharpe"
    es = _EarlyStop(best_metric=(-1e9 if higher_is_better else 1e9), best_epoch=-1)

    print(f"[MinTrain] device={dev} | amp={use_amp} dtype={amp_dtype} | workers={getattr(args,'num_workers',0)}")
    print(f"[MinTrain] cache={cache_path} | samples={n_samples:,} | features={n_features} | seq_len={args.seq_len}")
    print(f"[MinTrain] model={model_name} | loss={args.loss} | epochs={args.epochs} | batch={args.batch_size}")

    t0 = time.time()
    for ep in range(int(args.epochs)):
        tr_loss = train_epoch(
            model=model,
            loader=train_dl,
            opt=opt,
            crit=crit,
            scaler_amp=scaler,
            device=dev,
            use_amp=use_amp,
            classification=classification,
            grad_clip=float(getattr(args, "grad_clip", 1.0)),
            pbar=None,
            amp_dtype=amp_dtype,
            thermal_limit=int(getattr(args, "thermal_limit", 83)),
            feature_mask=None,
        )

        va_loss, dir_acc, sharpe = validate_epoch(
            model=model,
            loader=val_dl,
            crit=crit,
            device=dev,
            classification=classification,
            pbar=None,
            amp=use_amp,
            amp_dtype=amp_dtype,
        )

        metric = sharpe if higher_is_better else va_loss
        improved = (metric > es.best_metric) if higher_is_better else (metric < es.best_metric)
        if improved:
            es.best_metric = float(metric)
            es.best_epoch = int(ep)
            es.bad_epochs = 0
            _save_ckpt(
                ckpt_path,
                model=model,
                opt=opt,
                epoch=ep,
                metrics={
                    "train_loss": float(tr_loss),
                    "val_loss": float(va_loss),
                    "dir_acc": float(dir_acc),
                    "sharpe": float(sharpe),
                },
                args=args,
            )
            tag = "BEST"
        else:
            es.bad_epochs += 1
            tag = f"pat={es.bad_epochs}/{int(args.patience)}"

        elapsed = time.time() - t0
        print(
            f"[MinTrain] ep {ep+1:03d}/{int(args.epochs)} | "
            f"tr={tr_loss:.4f} va={va_loss:.4f} acc={dir_acc:.3f} sharpe={sharpe:.3f} | "
            f"{tag} | {elapsed/60:.1f}m"
        )

        if es.bad_epochs >= int(args.patience):
            print(f"[MinTrain] early stop at ep={ep+1} (best ep={es.best_epoch+1})")
            break

    print(f"[MinTrain] done. best_ckpt={ckpt_path}")
    return 0


if __name__ == "__main__":
    # Avoid torch.compile/inductor surprises on Windows; keep eager.
    os.environ.setdefault("TORCH_LOGS", "")
    raise SystemExit(main())

