"""Batch preparation / sanitization for the supervised loop.

Extracted verbatim from ``training.supervised_loop`` (refactor R5);
re-exported there for import-path stability."""
from __future__ import annotations

import torch

from training.core import _log_warn

def _unpack_batch(batch, device):
    """Return (xb, yb, y_cls, y_conf, sample_idx) from 2- to 5-tuple batches.

    Always returns a 5-element tuple.  ``sample_idx`` is ``None`` when the
    loader does not carry index information.
    """
    n = len(batch) if isinstance(batch, (tuple, list)) else 1
    if n >= 5:
        xb, yb, y_cls, y_conf, sample_idx = batch[0], batch[1], batch[2], batch[3], batch[4]
        return (
            xb.to(device, non_blocking=True),
            yb.to(device, non_blocking=True),
            y_cls.to(device, non_blocking=True),
            y_conf.to(device, non_blocking=True),
            sample_idx.to(device, non_blocking=True),
        )
    if n == 4:
        xb, yb, y_cls, y_conf = batch[0], batch[1], batch[2], batch[3]
        return (
            xb.to(device, non_blocking=True),
            yb.to(device, non_blocking=True),
            y_cls.to(device, non_blocking=True),
            y_conf.to(device, non_blocking=True),
            None,
        )
    if n == 3:
        xb, yb, sample_idx = batch[0], batch[1], batch[2]
        return (
            xb.to(device, non_blocking=True),
            yb.to(device, non_blocking=True),
            None,
            None,
            sample_idx.to(device, non_blocking=True),
        )
    xb, yb = batch[0], batch[1]
    return (xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), None, None, None)


_SANITIZE_STATS: dict[str, int] = {
    "feature_nonfinite": 0,
    "batches_with_feature_clamp": 0,
    "target_rows_dropped": 0,
    "batches_with_target_drops": 0,
}


def _sanitize_batch_tensors(
    xb: torch.Tensor,
    yb: torch.Tensor,
    y_cls: torch.Tensor | None,
    y_conf: torch.Tensor | None,
    *,
    skip_bad_targets: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Sanitize features; surface non-finite targets instead of silently zeroing them.

    Returns ``(xb, yb, y_cls, y_conf, keep_mask)``. ``keep_mask`` is a bool
    vector over the batch dim. Features may be clamped (counted + WARN);
    targets are **never** replaced with 0 - bad rows are flagged for the
    caller to drop (or raise when ``skip_bad_targets=False``).
    """
    global _SANITIZE_STATS
    xb_f = xb.float()
    feat_bad = ~torch.isfinite(xb_f)
    n_feat_bad = int(feat_bad.sum().item())
    if n_feat_bad:
        _SANITIZE_STATS["feature_nonfinite"] += n_feat_bad
        _SANITIZE_STATS["batches_with_feature_clamp"] += 1
        n_clamp_batches = _SANITIZE_STATS["batches_with_feature_clamp"]
        if n_clamp_batches <= 3 or n_clamp_batches % 50 == 0:
            msg = (
                f"[Sanitize] WARN: clamped {n_feat_bad} non-finite feature value(s) "
                f"(batches={n_clamp_batches}, total_vals={_SANITIZE_STATS['feature_nonfinite']})"
            )
            print(msg)
            try:
                _log_warn(msg)
            except Exception:
                pass
    xb = torch.nan_to_num(xb_f, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

    # Targets: detect non-finite rows - never nan_to_num → 0.
    yb_f = yb.float()
    bsz = int(yb_f.shape[0])
    keep = torch.isfinite(yb_f).reshape(bsz, -1).all(dim=-1)

    if y_cls is not None:
        y_cls_f = y_cls.float()
        keep = keep & torch.isfinite(y_cls_f).reshape(bsz, -1).all(dim=-1)
        y_cls = y_cls_f.clamp(-1.0, 1.0)
    if y_conf is not None:
        y_conf_f = y_conf.float()
        keep = keep & torch.isfinite(y_conf_f).reshape(bsz, -1).all(dim=-1)
        y_conf = y_conf_f.clamp(0.0, 1.0)

    if not bool(keep.all()):
        n_bad = int((~keep).sum().item())
        _SANITIZE_STATS["target_rows_dropped"] += n_bad
        _SANITIZE_STATS["batches_with_target_drops"] += 1
        if not skip_bad_targets:
            raise ValueError(
                f"[Sanitize] {n_bad}/{bsz} non-finite target rows (fail-closed; "
                f"not zeroed - fix data or set skip_bad_targets=True to drop)"
            )
        n_drop_batches = _SANITIZE_STATS["batches_with_target_drops"]
        if n_drop_batches <= 3 or n_drop_batches % 50 == 0:
            msg = (
                f"[Sanitize] WARN: dropping {n_bad}/{bsz} rows with non-finite targets "
                f"(not zeroed; batches={n_drop_batches}, "
                f"total_rows={_SANITIZE_STATS['target_rows_dropped']})"
            )
            print(msg)
            try:
                _log_warn(msg)
            except Exception:
                pass
    return xb, yb_f, y_cls, y_conf, keep


def sanitize_stats() -> dict[str, int]:
    """Return a copy of running sanitize counters (features clamped / targets dropped)."""
    return dict(_SANITIZE_STATS)


def reset_sanitize_stats() -> None:
    for key in _SANITIZE_STATS:
        _SANITIZE_STATS[key] = 0


def _prepare_train_batch(
    batch,
    device,
    *,
    seq_len: int | None,
    feature_mask: torch.Tensor | None,
    adversarial_gen,
    adversarial_feature_names: list[str] | None,
    model=None,
    crit=None,
    classification: bool = False,
    multitask: bool = False,
):
    """Unpack, sanitize, mask, and optionally adversarially perturb one batch.

    Returns ``(xb, yb, y_cls, y_conf, sample_idx)`` or ``None`` when every
    row has non-finite targets and the batch must be skipped.
    """
    xb, yb, y_cls_b, y_conf_b, batch_idx_t = _unpack_batch(batch, device)
    if seq_len is not None and xb.shape[1] > seq_len:
        xb = xb[:, -seq_len:, :]
    xb, yb, y_cls_b, y_conf_b, keep = _sanitize_batch_tensors(
        xb,
        yb,
        y_cls_b,
        y_conf_b,
        skip_bad_targets=True,
    )
    if keep is not None and not bool(keep.all()):
        if not bool(keep.any()):
            return None
        xb = xb[keep]
        yb = yb[keep]
        if y_cls_b is not None:
            y_cls_b = y_cls_b[keep]
        if y_conf_b is not None:
            y_conf_b = y_conf_b[keep]
        if batch_idx_t is not None:
            batch_idx_t = batch_idx_t[keep]

    if feature_mask is not None:
        xb = xb * feature_mask

    if adversarial_gen is not None:
        try:
            if classification or multitask:
                y_adv = y_cls_b if y_cls_b is not None else yb
                # Class targets are cached as {-1,0,1} floats; CE needs long
                # {0,1,2}. Match MultiTaskLoss.forward / lightning_trainer by
                # clamping to [0,2] so the -1 (Sell) rows don't hit the CUDA
                # nll kernel's `t >= 0 && t < n_classes` device-side assert.
                if y_adv.dtype in (torch.float16, torch.float32, torch.float64):
                    y_adv = y_adv.long().clamp(0, 2)
            else:
                y_adv = yb
            proxy_crit = (
                (lambda o, y: torch.nn.functional.cross_entropy(o, y))
                if multitask
                else crit
            )
            res = adversarial_gen(model, xb, y_adv, proxy_crit)
            xb = res[0] if isinstance(res, (tuple, list)) else res
        except TypeError:
            with torch.no_grad():
                xb = adversarial_gen(xb, adversarial_feature_names)

    return xb, yb, y_cls_b, y_conf_b, batch_idx_t
