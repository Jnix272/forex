"""
training/scale_model.py
=======================
Knowledge Distillation script for scaling models.
Allows training a Student model (either larger or smaller) to mimic a pre-trained Teacher model.
By using the Teacher's outputs as "soft labels", the Student learns the market structure
much faster than learning from raw labels alone.

Usage:
  python training/scale_model.py --teacher-model mamba --teacher-ckpt checkpoints/mamba_best.pt \\
                                 --student-model haelt --epochs 50
"""

import argparse
import os
import sys
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_gpu import (
    build_dataset_chunked,
    ZarrStreamDataset,
    walk_forward_splits,
    setup_device,
    _class_weights_tensor,
    _log_nan,
    _ThreadPrefetchLoader,
    _recover_nonfinite_training_state,
    _gradients_are_finite,
    _apply_yaml_config,
    _match_target_shape,
    labels_to_class_index,
    build_model,
    run_preflight_sanity_checks,
)

from config.settings import TRAINING, PATHS

def parse_args():
    p = argparse.ArgumentParser(description="Knowledge Distillation (Scale Model)")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config to use as defaults, e.g. config/run_fast.yaml")
    
    # Distillation specific
    p.add_argument("--teacher-model", type=str, required=True, help="Architecture of the teacher model")
    p.add_argument("--teacher-ckpt", type=str, required=True, help="Path to teacher weights (.pt)")
    p.add_argument("--student-model", type=str, required=True, help="Architecture of the new student model")
    
    p.add_argument("--alpha", type=float, default=0.5, 
                   help="Weight of the distillation loss vs task loss (default 0.5)")
    p.add_argument("--temperature", type=float, default=2.0, 
                   help="Temperature for softening teacher logits in CrossEntropy distillation")
                   
    # Standard training args (matching train_gpu.py)
    p.add_argument("--n-ticks", type=int, default=20_000_000)
    p.add_argument("--chunk-size", type=int, default=500_000)
    p.add_argument("--real-data-window-days", type=int, default=0)
    p.add_argument("--data-source", type=str, default="synthetic")
    p.add_argument("--data-start", type=str, default="2020-01-01")
    p.add_argument("--data-end", type=str, default="2023-12-31")
    p.add_argument("--pair", type=str, default="EURUSD")
    p.add_argument("--pairs", type=str, default=None)
    p.add_argument("--pair-embed-dim", type=int, default=0)
    p.add_argument("--corr-window", type=int, default=20)
    p.add_argument("--corr-window-long", type=int, default=60)
    p.add_argument("--momentum-window", type=int, default=20)
    p.add_argument("--pair-align", type=str, default="inner", choices=["inner", "outer"])
    p.add_argument("--full-day-data", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-split", type=float, default=None)
    p.add_argument("--loss", type=str, default="huber")
    p.add_argument("--label-method", type=str, default="rl_reward")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--student-hidden-size", type=int, default=None)
    p.add_argument("--student-num-layers", type=int, default=None)
    p.add_argument("--student-d-model", type=int, default=None)
    p.add_argument("--student-nhead", type=int, default=None)
    p.add_argument("--student-dropout", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--multitask", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--quick-mode", action="store_true")
    p.add_argument("--checkpoint-dir", type=str, default=PATHS["checkpoints"])
    p.add_argument("--data-cache", type=str, default=PATHS["data_processed"])
    p.add_argument("--cross-asset-mode", type=str, default="auto")
    p.add_argument("--historical-news-mode", type=str, default="calendar", choices=["off", "calendar", "full"])
    p.add_argument("--historical-news-file", type=str, default=None)
    p.add_argument("--economic-calendar-file", type=str, default=None)
    p.add_argument("--integrity-gate", action="store_true", default=True)
    p.add_argument("--auto-rebuild-on-mismatch", action="store_true")
    p.add_argument("--no-amp", action="store_true", help="Force FP32 and disable AMP (overrides --amp and --dtype)")

    pre, _ = p.parse_known_args()
    if pre.config:
        _apply_yaml_config(p, pre.config)

    args = p.parse_args()
    
    # Force FP32 if --no-amp is set
    if args.no_amp:
        args.amp = False
        args.dtype = "fp32"
        print("[Distill] --no-amp flag detected: forcing FP32 and disabling AMP for stability.")

    if args.val_split is None:
        args.val_split = float(TRAINING["val_split"])
    if args.quick_mode:
        args.epochs = min(args.epochs, 5)
        args.patience = 2
    return args


def _student_args(args):
    out = copy.copy(args)
    out.model = args.student_model
    for src, dst in (
        ("student_hidden_size", "hidden_size"),
        ("student_num_layers", "num_layers"),
        ("student_d_model", "d_model"),
        ("student_nhead", "nhead"),
        ("student_dropout", "dropout"),
    ):
        val = getattr(args, src, None)
        if val is not None:
            setattr(out, dst, val)
    return out


def _checkpoint_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _model_checkpoint_dir(base_dir: str, model_name: str) -> Path:
    base = Path(str(base_dir or PATHS["checkpoints"])).expanduser()
    return base if base.name.lower() == model_name.lower() else base / model_name

def _is_multitask_checkpoint(state_dict):
    return any("dir_head" in k or "ret_head" in k for k in state_dict.keys())

def _ensemble_meta_path(ckpt_path: Path) -> Path | None:
    candidates = [
        ckpt_path.with_suffix(ckpt_path.suffix + ".json"),
        ckpt_path.with_name(f"{ckpt_path.name}.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_ensemble_teacher(args, n_features: int, seq_len: int, device: torch.device):
    """Load EnsembleMetaLearner plus its base checkpoints for distillation."""
    try:
        from models.ensemble import EnsembleMetaLearner
        from scripts.train_ensemble_meta import load_base_model, resolve_checkpoint
    except Exception as exc:
        raise RuntimeError(f"Could not import ensemble loader helpers: {exc}") from exc

    ckpt_path = Path(args.teacher_ckpt)
    meta_path = _ensemble_meta_path(ckpt_path)
    meta_payload = {}
    if meta_path is not None:
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta_payload = {}
    meta = meta_payload.get("meta", {}) if isinstance(meta_payload, dict) else {}
    base_names = list(meta.get("base_names") or ["mamba", "haelt", "tft"])
    ckpt_dir = Path(args.checkpoint_dir)

    bases = []
    loaded_names = []
    for name in base_names:
        base_ckpt = resolve_checkpoint(str(name), ckpt_dir)
        if base_ckpt is None:
            raise FileNotFoundError(f"Missing ensemble base checkpoint for '{name}' under {ckpt_dir}")
        base_model, _cfg = load_base_model(str(name), base_ckpt, n_features, seq_len, device)
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
        bases.append(base_model)
        loaded_names.append(str(name))

    teacher = EnsembleMetaLearner(bases, context_dim=32, hidden=64, base_names=loaded_names).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    teacher.load_state_dict(_checkpoint_state_dict(ckpt), strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"[Distill] Loaded ensemble teacher from {ckpt_path} with bases={loaded_names}")
    return teacher


def _teacher_output(raw, *, teacher_is_ensemble: bool):
    if teacher_is_ensemble and isinstance(raw, (tuple, list)):
        return raw[0]
    return raw

def distillation_loss_fn(student_out, teacher_out, target, task_loss_fn, args, teacher_is_mt=None):
    """
    Computes a combined loss = (1 - alpha) * TaskLoss + alpha * DistillationLoss
    Handles both single-scalar outputs and MultiTaskHead tuples.
    """
    alpha = args.alpha
    temp = args.temperature

    if args.multitask and (teacher_is_mt is not False):
        # student_out is (direction_logits, return_hat, confidence)
        s_dir, s_ret, s_conf = student_out
        
        if isinstance(teacher_out, tuple):
            t_dir, t_ret, t_conf = teacher_out
        else:
            # Fallback: teacher is single-head, student is multi-head
            # Map teacher output to the most appropriate student head
            t_dir = teacher_out if teacher_out.ndim > 1 else None
            t_ret = teacher_out if teacher_out.ndim <= 1 else None
            t_conf = None
        
        # 1. Task loss (from true labels)
        task_loss = task_loss_fn(
            s_dir, s_ret, s_conf, 
            labels_to_class_index(target), target
        )
        
        # 2. Distillation loss (match the teacher)
        # Direction: KL Divergence with temperature scaling
        s_dir_log_prob = F.log_softmax(s_dir / temp, dim=-1)
        t_dir_prob = F.softmax(t_dir / temp, dim=-1)
        distill_dir = F.kl_div(s_dir_log_prob, t_dir_prob, reduction="batchmean") * (temp ** 2)
        
        # Return/Confidence: Mean Squared Error matching
        distill_ret = F.mse_loss(s_ret, t_ret)
        distill_conf = F.mse_loss(s_conf, t_conf)
        
        distill_loss = distill_dir + distill_ret + distill_conf
        
    else:
        if student_out.ndim >= 2 and student_out.shape[-1] > 1:
            y_cls = labels_to_class_index(target)
            task_loss = task_loss_fn(student_out, y_cls)
            s_log_prob = F.log_softmax(student_out / temp, dim=-1)
            t_prob = F.softmax(teacher_out / temp, dim=-1)
            distill_loss = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temp ** 2)
        else:
            target = _match_target_shape(student_out, target)
            teacher_out = _match_target_shape(student_out, teacher_out)
            task_loss = task_loss_fn(student_out, target)
            distill_loss = F.mse_loss(student_out, teacher_out)

    return (1.0 - alpha) * task_loss + alpha * distill_loss

def run_distillation():
    args = parse_args()
    dev, _, amp_dtype = setup_device(dtype_override=getattr(args, "dtype", "auto"))
    use_amp = bool(args.amp and dev.type == "cuda" and amp_dtype != torch.float32)
    
    # 1. Build Data
    cache_path, n_samples, n_features, scaler = build_dataset_chunked(args)
    if n_samples == 0:
        print("[Distill] No data. Exiting.")
        return
        
    from training.train_gpu import _embargo_bars, _purge_bars, _validation_method
    splits = walk_forward_splits(n_samples, 1, _embargo_bars(args), _purge_bars(args), _validation_method(args))
    train_idx, val_idx = splits[-1]
    
    ds_train = ZarrStreamDataset(cache_path, train_idx, shuffle_chunks=True)
    train_nw_safe = 0 if os.name == "nt" else args.num_workers
    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, num_workers=train_nw_safe,
        prefetch_factor=args.prefetch_factor if train_nw_safe > 0 else None,
        pin_memory=(dev.type == "cuda" and os.name != "nt"), drop_last=True
    )
    
    # Stability: On Windows, validation loader worker processes crash after long runs.
    # num_workers=0 runs loading in-process — safe and fast enough for validation.
    val_nw_safe = 0 if os.name == "nt" else args.num_workers
    
    ds_val = ZarrStreamDataset(cache_path, val_idx, shuffle_chunks=False)
    loader_val = DataLoader(
        ds_val, batch_size=args.batch_size, num_workers=val_nw_safe,
        prefetch_factor=args.prefetch_factor if val_nw_safe > 0 else None,
        pin_memory=False, # Stability: don't pin for val on Windows
        drop_last=False
    )
    
    if train_nw_safe == 0:
        loader_train = _ThreadPrefetchLoader(loader_train, prefetch=2)
        loader_val = _ThreadPrefetchLoader(loader_val, prefetch=2)
        
    teacher_is_ensemble = str(args.teacher_model).lower() == "ensemble"
    if teacher_is_ensemble:
        teacher = _load_ensemble_teacher(args, n_features, args.seq_len, dev)
        teacher_is_mt = False
    else:
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
        state_dict = _checkpoint_state_dict(ckpt)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Bug fix: Detect if teacher is multitask independently of student settings
        teacher_is_mt = _is_multitask_checkpoint(state_dict)
        t_args = copy.copy(args)
        t_args.multitask = teacher_is_mt
        
        teacher = build_model(args.teacher_model, n_features, t_args)
        teacher.load_state_dict(state_dict, strict=False)
        
        teacher.to(dev)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False  # Freeze teacher
        
    # 3. Initialize Student Model
    s_args = _student_args(args)
    student = build_model(args.student_model, n_features, s_args)
    student.to(dev)

    # -- Preflight Checks ------------------------------------------------------
    run_preflight_sanity_checks(student, dev, loader_train, args)
    
    # 4. Optimizer and Loss
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    grad_scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    
    if args.multitask:
        from models.architectures import MultiTaskLoss
        class_w = _class_weights_tensor(cache_path, train_idx, dev)
        task_loss_fn = MultiTaskLoss(class_weights=class_w)
    else:
        if args.loss == "cross_entropy":
            class_w = _class_weights_tensor(cache_path, train_idx, dev)
            task_loss_fn = nn.CrossEntropyLoss(weight=class_w)
        elif args.loss == "huber":
            from models.architectures import HuberLoss
            task_loss_fn = HuberLoss(delta=1.0)
        else:
            task_loss_fn = nn.MSELoss()

    # 5. Training Loop
    print("\n[Distill] Starting Knowledge Distillation...")
    best_loss = float("inf")
    patience_ctr = 0
    
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        batches = 0
        
        for batch_idx, (Xb, yb) in enumerate(loader_train):
            Xb, yb = Xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                with autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                    t_out = _teacher_output(teacher(Xb), teacher_is_ensemble=teacher_is_ensemble)
                    
            # Train student
            with autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                s_out = student(Xb)
                loss = distillation_loss_fn(s_out, t_out, yb, task_loss_fn, args, teacher_is_mt)
                loss = loss / max(1, int(args.grad_accum_steps))

            if not torch.isfinite(loss):
                _recover_nonfinite_training_state(student, opt)
                _log_nan(batch_idx, epoch, 1)
                continue
                
            do_step = (
                ((batch_idx + 1) % max(1, int(args.grad_accum_steps)) == 0)
                or (batch_idx + 1 == len(loader_train))
            )
            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                if do_step:
                    grad_scaler.unscale_(opt)
                    if _gradients_are_finite(student):
                        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                        grad_scaler.step(opt)
                        grad_scaler.update()
                        opt.zero_grad(set_to_none=True)
                    else:
                        _recover_nonfinite_training_state(student, opt)
                        grad_scaler.update()
                        _log_nan(batch_idx, epoch, 1)
            else:
                loss.backward()
                if do_step:
                    if _gradients_are_finite(student):
                        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    else:
                        _recover_nonfinite_training_state(student, opt)
                        _log_nan(batch_idx, epoch, 1)
                    
            total_loss += loss.item() * max(1, int(args.grad_accum_steps))
            batches += 1
            
        epoch_loss = total_loss / max(1, batches)
        
        # Validation Loop
        student.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X_val, y_val in loader_val:
                X_val, y_val = X_val.to(dev, non_blocking=True), y_val.to(dev, non_blocking=True)
                
                with autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                    t_out_val = teacher(X_val)
                    t_out_val = _teacher_output(t_out_val, teacher_is_ensemble=teacher_is_ensemble)
                    s_out_val = student(X_val)
                    v_loss = distillation_loss_fn(s_out_val, t_out_val, y_val, task_loss_fn, args)
                    
                val_loss += v_loss.item()
                val_batches += 1
                
        epoch_val_loss = val_loss / max(1, val_batches)
        print(f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.5f} | Val Loss: {epoch_val_loss:.5f}")
        
        # Early Stopping logic based on Val Loss
        if epoch_val_loss < best_loss - 1e-4:
            best_loss = epoch_val_loss
            patience_ctr = 0
            # Save student
            out_path = _model_checkpoint_dir(args.checkpoint_dir, args.student_model) / f"{args.student_model}_student_best.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": best_loss
            }, out_path)
            print(f"   -> Saved new best student checkpoint to {out_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[Distill] Early stopping at epoch {epoch} (Val Loss: {epoch_val_loss:.5f})")
                break

if __name__ == "__main__":
    run_distillation()
