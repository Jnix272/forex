"""Optimizer-step machinery for the supervised loop.

Extracted verbatim from ``training.supervised_loop`` (refactor R5);
re-exported there for import-path stability.
"""
from __future__ import annotations

import torch.nn as nn

from training.core import _log_nan
from training.direction_control import (
    _gradients_are_finite,
    _recover_nonfinite_training_state,
)

def _centralize_gradients(params) -> None:
    """Apply Gradient Centralization (Yong et al.) to weight grads in-place.

    Subtracts the mean over non-batch dims for every ``dim > 1`` gradient so
    weight updates stay mean-zero. Accepts an ``nn.Module`` or a parameter
    iterable so AMP / non-AMP / diversity-finetune share one hot path.
    """
    iterable = params.parameters() if isinstance(params, nn.Module) else params
    for p in iterable:
        g = p.grad
        if g is None or g.dim() <= 1:
            continue
        g.sub_(g.mean(dim=tuple(range(1, g.dim())), keepdim=True))


def _maybe_warn_grad_norm(model, batch_idx: int) -> None:
    if batch_idx % 100 != 0:
        return
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    total_norm = total_norm**0.5
    if total_norm > 50.0:
        print(f"[Stability] WARNING: High grad norm ({total_norm:.2f}) at batch {batch_idx}")


def _optimizer_step(
    model,
    opt,
    scaler_amp,
    use_fp16_scaler,
    do_step,
    grad_clip,
    scheduler,
    batch_idx,
    *,
    nan_skips: int,
    epoch: int,
    pbar,
    si_module=None,
) -> tuple[bool, int]:
    """Backward is assumed done. Returns (stepped_ok, updated_nan_skips)."""
    if not do_step:
        return True, nan_skips
    if use_fp16_scaler:
        scaler_amp.unscale_(opt)
    if not _gradients_are_finite(model):
        nan_skips += 1
        _log_nan(batch_idx, epoch, nan_skips)
        _recover_nonfinite_training_state(model, opt)
        if use_fp16_scaler:
            scaler_amp.update()
        if nan_skips <= 3 or nan_skips % 10 == 0:
            print(f"[Train] NaN/Inf gradients at batch {batch_idx} (skip {nan_skips})")
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss="NaN-grad-skip")
        return False, nan_skips
    # Snapshot params + RAW gradients BEFORE any in-place modification of p.grad
    # (clipping, centralization).  Must come first so the SI path integral
    # accumulates the true backward() gradients, not the clipped/centred ones.
    if si_module is not None:
        si_module.pre_step()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    _maybe_warn_grad_norm(model, batch_idx)
    # Shared GC for both AMP (after unscale) and non-AMP paths.
    _centralize_gradients(model)
    if use_fp16_scaler:
        scaler_amp.step(opt)
        scaler_amp.update()
    else:
        opt.step()
    if si_module is not None:
        si_module.post_step()
    opt.zero_grad(set_to_none=True)
    if scheduler is not None:
        scheduler.step()
    return True, nan_skips


