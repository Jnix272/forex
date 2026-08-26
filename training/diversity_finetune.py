"""Joint diversity fine-tuning across trained models.

Extracted verbatim from ``training.supervised_loop`` (refactor R5);
re-exported there for import-path stability."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.architectures import MODEL_ROLES, DiversityLoss
from training.cache_integrity import _on_disk_sequence_count
from training.core import _crop_to_seq_len
from training.direction_control import _direction_class_index
from training.gpu_cli import _model_build_args
from training.gpu_datasets import ZarrStreamDataset
from training.gpu_device import build_adamw
from training.gpu_losses import _match_target_shape
from training.loop_batches import _sanitize_batch_tensors, _unpack_batch
from training.loop_optim import _centralize_gradients
from training.model_factory import _strict_load_report, build_model
from training.post_train import _safe_save

def run_diversity_finetune(
    checkpoint_dir: str,
    model_names: list,
    cache_path: str,
    n_features: int,
    args,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-5,
    div_weight: float = 0.10,
    same_role_mult: float = 2.0,
    batch_size: int = 512,
    max_batches: int = 200,
) -> None:
    """
    C: Joint diversity fine-tuning across all trained models.

    Loads each model's *_best.pt checkpoint, then for each batch:
      loss_i = task_loss_i  +  DiversityLoss(all model outputs)
    All models are updated simultaneously so the diversity gradient flows
    into each model while maintaining their individual task performance.

    Args:
        checkpoint_dir : Directory containing *_best.pt files.
        model_names    : List of model name strings to include.
        cache_path     : Zarr cache path (same as used during training).
        n_features     : Feature dimension.
        args           : Training args namespace (loss, seq_len, etc.).
        device         : torch.device.
        epochs         : Fine-tuning epochs (3 is usually sufficient).
        lr             : Learning rate (much lower than training -- fine-tuning).
        div_weight     : DiversityLoss weight multiplier.
        same_role_mult : Extra multiplier for same-role pairs.
        batch_size     : Batch size for joint forward pass.
        max_batches    : Max batches per epoch (limits GPU time).
    """
    ckpt_dir = Path(checkpoint_dir)
    classification = getattr(args, "loss", "cross_entropy") in ("cross_entropy", "multi_task", "asymmetric_directional")

    # -- Load checkpoints ------------------------------------------------------
    loaded_models = {}
    loaded_roles = []
    loaded_names = []
    loaded_seq_lens = []
    for name in model_names:
        # Support both per-model subfolder layout (<base>/<model>/<model>_best.pt)
        # and the legacy flat layout (<base>/<model>_best.pt).
        ckpt_path = ckpt_dir / name / f"{name}_best.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / f"{name}_best.pt"
        if not ckpt_path.exists():
            print(f"  [DivFT] Skipping {name} -- checkpoint not found at {ckpt_path}")
            continue
        try:
            model_args = _model_build_args(args, name)
            m = build_model(name, n_features, model_args).to(device)  # type: ignore
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

            try:
                m.load_state_dict(state, strict=True)
            except Exception:
                # If strict loading failed, load into backbone with an asserting
                # report so a near-empty load fails loudly (A-H2).
                core = m.backbone if hasattr(m, "backbone") else m
                _strict_load_report(core, state, f"DivFT:{name}", min_frac_loaded=0.6)

            m.train()
            loaded_models[name] = m
            loaded_roles.append(MODEL_ROLES.get(name, "unknown"))
            loaded_names.append(name)
            loaded_seq_lens.append(int(getattr(model_args, "seq_len", getattr(args, "seq_len", 0)) or 0))
        except Exception as e:
            print(f"  [DivFT] Could not load {name}: {e}")

    if len(loaded_models) < 2:
        print("  [DivFT] Need >=2 models for diversity fine-tuning -- skipping.")
        return

    print(f"\n[DivFT] Diversity fine-tuning: {loaded_names}")
    print(f"        Roles: {loaded_roles}")
    print(f"        epochs={epochs}  lr={lr}  div_weight={div_weight}  same_role_mult={same_role_mult}")

    # -- Shared optimizer across all models -----------------------------------
    all_params = []
    for m in loaded_models.values():
        all_params += list(m.parameters())
    opt = build_adamw(all_params, lr=lr, weight_decay=1e-4)

    # -- Criterion (same as supervised training) -------------------------------
    if classification:
        crit = torch.nn.CrossEntropyLoss()
    else:
        crit = torch.nn.HuberLoss(delta=1.0)

    # -- Diversity loss -----------------------------------------------------
    div_loss_fn = DiversityLoss(
        weight=div_weight,
        same_role_mult=same_role_mult,
        roles=loaded_roles,
    ).to(device)  # type: ignore

    # -- Data loader (val split -- fine-tune on held-out data only) ------------
    n_samples = int(_on_disk_sequence_count(cache_path) or 0)
    if n_samples <= 0:
        print("  [DivFT] Could not determine dataset size -- skipping.")
        return
    val_start = int(n_samples * 0.80)
    val_idx = np.arange(val_start, n_samples)
    ds = ZarrStreamDataset(cache_path, val_idx, shuffle_chunks=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    model_list = list(loaded_models.values())

    for ep in range(epochs):
        ep_task_loss = 0.0
        ep_div_loss = 0.0
        n_batches = 0

        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            xb, yb, y_cls_b, y_conf_b, _ = _unpack_batch(batch, device)
            xb, yb, y_cls_b, y_conf_b, keep = _sanitize_batch_tensors(xb, yb, y_cls_b, y_conf_b)
            if keep is not None and not bool(keep.all()):
                if not bool(keep.any()):
                    continue
                xb, yb = xb[keep], yb[keep]
                if y_cls_b is not None:
                    y_cls_b = y_cls_b[keep]
                if y_conf_b is not None:
                    y_conf_b = y_conf_b[keep]
            y_cls_idx = _direction_class_index(yb, y_cls_b, classification=classification)

            opt.zero_grad(set_to_none=True)

            # Forward all models on the same batch
            outputs = []
            task_loss = torch.tensor(0.0, device=device)
            for m, m_seq_len in zip(model_list, loaded_seq_lens, strict=False):
                xb_m = _crop_to_seq_len(xb, m_seq_len)
                out = m(xb_m)
                outputs.append(out)
                # Per-model task loss
                if isinstance(out, tuple):
                    logits, _ret_hat, _conf = out
                    task_loss = task_loss + crit(logits, y_cls_idx)
                elif classification:
                    task_loss = task_loss + crit(out, y_cls_idx)
                else:
                    task_loss = task_loss + crit(out, _match_target_shape(out, yb))
            task_loss = task_loss / len(model_list)

            # Scalar outputs for diversity loss -- extract scalar per model per sample
            scalar_outs = []
            for out in outputs:
                if isinstance(out, tuple):
                    # regression head for diversity comparison
                    scalar_outs.append(out[1].reshape(-1))  # return_hat
                elif out.ndim == 2 and out.shape[-1] > 1:
                    # classification logits -> use argmax-weighted scalar
                    scalar_outs.append(out.softmax(-1)[:, -1] - out.softmax(-1)[:, 0])
                else:
                    scalar_outs.append(out.reshape(-1))

            diversity = div_loss_fn(scalar_outs)
            loss = task_loss + diversity
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            _centralize_gradients(all_params)
            opt.step()

            ep_task_loss += task_loss.item()
            ep_div_loss += diversity.item()
            n_batches += 1

        avg_t = ep_task_loss / max(n_batches, 1)
        avg_d = ep_div_loss / max(n_batches, 1)
        print(f"  [DivFT] Epoch {ep + 1}/{epochs} | task={avg_t:.5f}  diversity={avg_d:.5f}")

    # -- Save updated checkpoints ---------------------------------------------
    for name, m in loaded_models.items():
        ckpt_path = ckpt_dir / name / f"{name}_best.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / f"{name}_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt["model_state_dict"] = m.cpu().state_dict()
        ckpt["diversity_finetuned"] = True
        _safe_save(ckpt, ckpt_path)
        print(f"  [DivFT] Saved updated checkpoint: {ckpt_path.name}")


