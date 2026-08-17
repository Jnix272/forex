"""
PyTorch Lightning training infrastructure for Forex models.

Wraps existing model architectures and data loading for Lightning training,
with PLCurriculumCallback integration for curriculum learning.

Usage:
    python training/train_gpu.py --training-framework lightning --model haelt --epochs 40
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    pl = None
    Trainer = None

from models.architectures import build_model as _build_arch_model
from training.gpu_datasets import ZarrStreamDataset, wrap_loader_prefetch
from training.model_factory import _core_model
from training.model_factory import build_model as _build_training_model


class ForexLightningModule(pl.LightningModule):
    """Lightning module wrapping Forex model architectures."""

    def __init__(
        self,
        model_name: str,
        n_features: int,
        args: Any,
        device: torch.device,
        cache_path: str = "",
        train_idx: np.ndarray | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["args", "device"])
        self.model_name = model_name
        self.n_features = n_features
        self.args = args
        self.cache_path = cache_path
        self.train_idx = train_idx

        # Build model
        try:
            self.model = _build_training_model(model_name, n_features, args)
        except Exception:
            self.model = _build_arch_model(model_name, input_size=n_features, seq_len=args.seq_len)

        # Build criterion using the supervised loop's build_criterion
        classification = args.loss in ("cross_entropy", "multi_task", "asymmetric_directional")
        multitask = bool(getattr(args, "multitask", False))
        self.classification = classification
        self.multitask = multitask

        from training.supervised_loop import build_criterion

        self.criterion = build_criterion(
            args,
            device,
            cache_path=cache_path if (classification or multitask) else None,
            train_idx=train_idx if (classification or multitask) else None,
        )

        # Metrics tracking
        self._val_losses = []
        self._val_sharpes = []
        self._dir_accs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with loss computation."""
        # Unpack batch (ZarrStreamDataset returns tuples)
        if isinstance(batch, (list, tuple)):
            xb = batch[0]
            yb = batch[1] if len(batch) > 1 else None
            y_cls = batch[2] if len(batch) > 2 else None
        else:
            xb = batch
            yb = None

        # Forward pass
        outputs = self.model(xb)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        # Compute loss
        if self.multitask and isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
            # MultiTask output: (direction_logits, return_hat, confidence)
            loss = self.criterion(outputs, yb, y_cls)
        elif self.classification and y_cls is not None:
            loss = self.criterion(outputs, y_cls.long().clamp(0, 2))
        elif yb is not None:
            loss = self.criterion(outputs, yb.float())
        else:
            loss = self.criterion(outputs, yb)

        # Log
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=xb.shape[0])
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Validation step with metrics."""
        if isinstance(batch, (list, tuple)):
            xb = batch[0]
            yb = batch[1] if len(batch) > 1 else None
            y_cls = batch[2] if len(batch) > 2 else None
        else:
            xb = batch
            yb = None

        outputs = self.model(xb)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        # Loss
        if self.multitask and isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
            val_loss = self.criterion(outputs, yb, y_cls)
        elif self.classification and y_cls is not None:
            val_loss = self.criterion(outputs, y_cls.long().clamp(0, 2))
        elif yb is not None:
            val_loss = self.criterion(outputs, yb.float())
        else:
            val_loss = self.criterion(outputs, yb)

        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, batch_size=xb.shape[0])

        # Direction accuracy
        if self.classification and y_cls is not None:
            preds = outputs.argmax(dim=-1) if outputs.dim() > 1 else (outputs > 0).long()
            acc = (preds == y_cls.long().clamp(0, 2)).float().mean()
            self.log("val_dir_acc", acc, prog_bar=True, on_epoch=True, batch_size=xb.shape[0])

        # Sharpe proxy (simplified)
        if yb is not None and outputs.dim() == 1:
            returns = (outputs * yb.float()).detach()
            if returns.std() > 1e-8:
                sharpe = returns.mean() / (returns.std() + 1e-8)
                self.log("val_sharpe", sharpe, prog_bar=True, on_epoch=True, batch_size=xb.shape[0])

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        from training.gpu_device import build_adamw

        opt = build_adamw(
            self.model.parameters(),
            lr=float(self.args.lr),
            weight_decay=float(getattr(self.args, "weight_decay", 1e-4)),
        )

        # LR scheduler
        _sched_kind = str(getattr(self.args, "lr_schedule", "warmup_cosine")).lower()
        _total_steps = self.trainer.estimated_stepping_batches

        if _sched_kind == "warmup_cosine":
            _warmup_pct = float(getattr(self.args, "lr_warmup_pct", 0.1))
            _min_ratio = float(getattr(self.args, "lr_min_ratio", 0.05))
            _warmup_steps = max(1, int(_total_steps * _warmup_pct))
            _decay_steps = max(1, int(_total_steps) - _warmup_steps)

            def _warmup_cosine(step):
                import math

                if step < _warmup_steps:
                    return (step + 1) / max(1, _warmup_steps)
                progress = min(1.0, (step - _warmup_steps) / _decay_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return _min_ratio + (1.0 - _min_ratio) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_warmup_cosine)
        else:
            max_lr = float(getattr(self.args, "onecycle_max_lr_mult", 10.0)) * float(self.args.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=max_lr,
                total_steps=_total_steps,
                pct_start=float(getattr(self.args, "onecycle_pct_start", 0.1)),
            )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_validation_epoch_end(self):
        """Track best metrics."""
        metrics = self.trainer.callback_metrics
        val_loss = metrics.get("val_loss", float("inf"))
        val_sharpe = metrics.get("val_sharpe", 0.0)
        self._val_losses.append(float(val_loss))
        self._val_sharpes.append(float(val_sharpe))

    def get_best_metrics(self) -> dict:
        """Return best metrics after training."""
        if not self._val_sharpes:
            return {"best_val_loss": float("inf"), "best_sharpe": 0.0}
        return {
            "best_val_loss": min(self._val_losses),
            "best_sharpe": max(self._val_sharpes),
            "final_val_loss": self._val_losses[-1],
            "final_sharpe": self._val_sharpes[-1],
            "history": {
                "train_loss": [],
                "val_loss": self._val_losses,
                "val_sharpe": self._val_sharpes,
                "dir_acc": self._dir_accs,
            },
        }


class ForexDataModule(pl.LightningDataModule):
    """DataModule wrapping ZarrStreamDataset for Lightning."""

    def __init__(
        self,
        cache_path: str,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        args: Any,
        multitask: bool = False,
    ):
        super().__init__()
        self.cache_path = cache_path
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.args = args
        self.multitask = multitask
        self.train_dataset = None
        self.val_dataset = None
        self._shuffle_buffer = int(getattr(args, "shuffle_buffer_size", 16384) or 16384)

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_dataset = ZarrStreamDataset(
                self.cache_path,
                self.train_idx,
                shuffle_chunks=True,
                shuffle_buffer_size=self._shuffle_buffer,
                multitask_targets=self.multitask,
                return_indices=True,
            )
            self.val_dataset = ZarrStreamDataset(
                self.cache_path,
                np.sort(self.val_idx),
                shuffle_chunks=False,
                multitask_targets=self.multitask,
                return_indices=False,
            )

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        nw = int(getattr(self.args, "num_workers", 4) or 4)
        pf = int(getattr(self.args, "prefetch_factor", 4) or 4)
        pin = bool(getattr(self.args, "pin_memory", True))
        dl = DataLoader(
            self.train_dataset,
            batch_size=int(self.args.batch_size),
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=nw > 0,
            prefetch_factor=pf if nw > 0 else None,
        )
        return wrap_loader_prefetch(dl, self.args)

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        val_nw = max(0, int(getattr(self.args, "val_num_workers", 2) or 2))
        pin = bool(getattr(self.args, "pin_memory", True))
        dl = DataLoader(
            self.val_dataset,
            batch_size=int(self.args.batch_size),
            shuffle=False,
            num_workers=val_nw,
            pin_memory=pin,
            persistent_workers=val_nw > 0,
        )
        return wrap_loader_prefetch(dl, self.args)


def run_lightning_training(
    model_name: str,
    cache_path: str,
    n_samples: int,
    n_features: int,
    args: Any,
    device: torch.device,
    n_gpus: int = 1,
    run: Any = None,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    fold_id: int | None = None,
    amp_dtype: torch.dtype = torch.float32,
) -> tuple[dict, dict]:
    """
    Main entry point for Lightning-based training.

    Returns:
        (history_dict, best_metrics_dict)
    """
    if not LIGHTNING_AVAILABLE:
        raise ImportError("PyTorch Lightning not installed. pip install pytorch-lightning")

    fold_suffix = f"_fold{fold_id}" if fold_id is not None else ""
    multitask = bool(getattr(args, "multitask", False))

    # Create DataModule
    datamodule = ForexDataModule(
        cache_path=cache_path,
        train_idx=train_idx,
        val_idx=val_idx,
        args=args,
        multitask=multitask,
    )

    # Create Lightning Module
    lightning_module = ForexLightningModule(
        model_name=model_name,
        n_features=n_features,
        args=args,
        device=device,
        cache_path=cache_path,
        train_idx=train_idx,
    )

    # Build callbacks
    callbacks = []

    # 1. Curriculum callback (if enabled)
    if getattr(args, "curriculum_manager", False):
        from training.curriculum_callbacks import PLCurriculumCallback

        _diff_arr = None
        try:
            from training.direction_control import _load_diff_array

            _diff = _load_diff_array(cache_path, n_samples)
            if _diff is not None and len(_diff) == n_samples and train_idx is not None:
                _diff_arr = np.asarray(_diff[train_idx], dtype=float)
        except Exception:
            pass

        pl_callback = PLCurriculumCallback(
            difficulty_scores=_diff_arr,
            pace_function=str(getattr(args, "curriculum_manager_mode", "linear")),
            total_epochs=int(args.epochs),
            min_fraction=0.1,
            max_fraction=1.0,
            use_loss_weighting=bool(getattr(args, "use_loss_weighting", False)),
            verbose=True,
        )
        callbacks.append(pl_callback)
        print(f"[Lightning] PLCurriculumCallback enabled (mode={getattr(args, 'curriculum_manager_mode', 'linear')})")

    # 2. Model checkpoint (match existing naming convention)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{model_name}{fold_suffix}_best",
        monitor="val_sharpe" if args.early_stop_metric == "sharpe" else "val_loss",
        mode="max" if args.early_stop_metric == "sharpe" else "min",
        save_top_k=1,
        save_last=False,
    )
    callbacks.append(ckpt_callback)

    # 3. Early stopping
    early_stop = EarlyStopping(
        monitor="val_sharpe" if args.early_stop_metric == "sharpe" else "val_loss",
        mode="max" if args.early_stop_metric == "sharpe" else "min",
        patience=int(getattr(args, "patience", 6)),
        min_delta=float(getattr(args, "early_stop_min_delta", 0.0)),
    )
    callbacks.append(early_stop)

    # 4. SWA (if enabled)
    if getattr(args, "swa_enabled", False):
        from pytorch_lightning.callbacks import StochasticWeightAveraging

        swa_callback = StochasticWeightAveraging(
            swa_lrs=float(getattr(args, "swa_lr", 1e-5)),
            swa_epoch_start=max(1, int(args.epochs * float(getattr(args, "swa_start_frac", 0.75)))),
        )
        callbacks.append(swa_callback)
        print(f"[Lightning] SWA enabled (start at {int(args.epochs * 0.75)} epochs)")

    # Create Trainer
    # Lightning 1.x uses "16"/"32"; 2.x renamed to "16-mixed"/"32-true".
    if tuple(int(p) for p in pl.__version__.split(".")[:2]) >= (2, 0):
        precision = "16-mixed" if args.amp and device.type == "cuda" else "32-true"
    else:
        precision = "16" if args.amp and device.type == "cuda" else "32"
    if Trainer is None:
        raise ImportError("pytorch_lightning is required but not installed.")
    trainer = Trainer(
        max_epochs=int(args.epochs),
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=min(n_gpus, 1) if device.type == "cuda" else 1,
        precision=precision,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=True,
        gradient_clip_val=float(getattr(args, "grad_clip", 1.0)),
        enable_checkpointing=True,
        accumulate_grad_batches=int(getattr(args, "grad_accum_steps", 1) or 1),
        check_val_every_n_epoch=1,
        inference_mode=False,
    )

    # Train
    print(f"\n[Lightning] Training {model_name.upper()} for {args.epochs} epochs with Lightning")
    t0 = time.time()
    trainer.fit(lightning_module, datamodule=datamodule)
    elapsed = time.time() - t0

    # Get best metrics
    metrics = lightning_module.get_best_metrics()
    metrics["elapsed_s"] = elapsed

    # Save best model (matching existing checkpoint format)
    best_path = ckpt_dir / f"{model_name}{fold_suffix}_best.pt"
    if not best_path.exists():
        # Save if Lightning didn't save via callback
        core = _core_model(lightning_module.model)
        torch.save(core.state_dict(), str(best_path))

    # Save config sidecar
    cfg_path = ckpt_dir / f"{model_name}{fold_suffix}_config.json"
    with open(cfg_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "n_features": n_features,
                "seq_len": args.seq_len,
                "d_model": getattr(args, "d_model", 256),
                "nhead": getattr(args, "nhead", 8),
                "hidden_size": getattr(args, "hidden_size", 256),
                "num_layers": getattr(args, "num_layers", 3),
                "dropout": getattr(args, "dropout", 0.1),
                "best_val_loss": metrics.get("best_val_loss", float("inf")),
                "best_val_sharpe_proxy": metrics.get("best_sharpe", 0.0),
                "epoch": trainer.current_epoch,
                "n_samples": n_samples,
                "loss": args.loss,
                "fold_id": fold_id,
                "framework": "lightning",
                "timestamp": datetime.now(UTC).isoformat(),
            },
            f,
            indent=2,
        )

    print(f"[Lightning] Training complete in {elapsed:.1f}s")
    print(f"[Lightning] Best Sharpe: {metrics.get('best_sharpe', 0):.4f}")
    print(f"[Lightning] Best Val Loss: {metrics.get('best_val_loss', float('inf')):.4f}")
    print(f"[Lightning] Checkpoint: {best_path}")

    history = metrics.get("history", {})
    history.setdefault("train_loss", [])
    history.setdefault("val_loss", metrics.get("final_val_loss", []))
    history.setdefault("val_sharpe", metrics.get("final_sharpe", []))
    history.setdefault("dir_acc", [])
    history.setdefault("lr", [])

    return history, metrics


def is_lightning_available() -> bool:
    """Check if Lightning is available."""
    return LIGHTNING_AVAILABLE
