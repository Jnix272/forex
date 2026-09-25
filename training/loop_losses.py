"""Loss assembly for the supervised loop.
# AUDIT: Verified Huber loss usage with delta parameter, bet_size weighting applied, and auxiliary loss scaling for regression.
"""

# Extracted verbatim from ``training.supervised_loop`` (refactor R5);
# re-exported there for import-path stability.
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from config.settings import TRAINING
from models.architectures import (
    AsymmetricDirectionalLoss,
    HuberLoss,
    MultiPairMultiTaskLoss,
    MultiTaskLoss,
    OverconfidencePenalty,
)
from training.direction_control import (
    _class_prior_tensor,
    _class_weights_tensor,
    _direction_class_index,
)
from training.ewc import apply_ewc_loss
from training.gpu_losses import (
    DirectionalHuberLoss,
    SharpeProxyLoss,
    _match_target_shape,
)
from training.synaptic_intelligence import apply_si_loss

_OVERCONF_PENALTY: OverconfidencePenalty | None = None  # D: set in supervised_train

def _apply_bet_size(base: torch.Tensor, bet_size: torch.Tensor | None) -> torch.Tensor:
    if bet_size is None or base.numel() <= 1:
        return base
    w = bet_size.clamp_min(0.0)
    if w.numel() != base.numel():
        w = w.view(w.shape[0], *[1] * (base.dim() - 1))
    else:
        w = w.reshape_as(base)
    return base * w

def _compute_loss(
    model_out: torch.Tensor | tuple,
    crit: nn.Module | Callable,
    yb: torch.Tensor,
    classification: bool,
    y_cls: torch.Tensor | None = None,
    y_conf: torch.Tensor | None = None,
    multitask: bool = False,
    direction_only: bool = False,
    bet_size: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Unified loss for single-head and MultiTaskWrapper outputs.

    ``direction_only=True`` (direction warmup / probe) always uses class-index
    CE against the direction head - never MultiTaskLoss's full (dir+ret+conf)
    signature, even when ``multitask=True``.
    """
    if isinstance(model_out, tuple):
        logits = model_out[0]
        ret_hat = model_out[1]
        conf = model_out[2] if len(model_out) > 2 else None

        y_cls_idx = _direction_class_index(
            yb,
            y_cls,
            classification=classification or direction_only,
        )
        if direction_only:
            # Warmup/probe: Huber on direction logits only.
            # MultiTaskLoss.forward expects (logits, ret, conf, y_cls, y_cont, ...);
            # calling it with 2 args would TypeError or mis-bind.
            if isinstance(crit, (MultiTaskLoss, MultiPairMultiTaskLoss)):
                # Keep batch and pair dimensions intact for proper bet_size broadcast
                y_cont = _match_target_shape(logits, yb)
                l_hub = cast(Any, crit).hub(logits, y_cont)
                return _apply_bet_size(l_hub, bet_size).mean()
            try:
                if isinstance(crit, nn.CrossEntropyLoss):
                    if logits.ndim == 3 and y_cls_idx.ndim == 2:
                        base = cast(Any, crit)(logits.reshape(-1, logits.shape[-1]), y_cls_idx.reshape(-1).long())
                    elif logits.ndim == 2 and y_cls_idx.ndim == 2:
                        base = cast(Any, crit)(logits, y_cls_idx[:, -1].long())
                    else:
                        print(f"DEBUG_CE: type(crit)={type(crit)} MultiTaskLoss={MultiTaskLoss} isinstance={isinstance(crit, MultiTaskLoss)} logits {logits.shape}", flush=True)
                        base = cast(Any, crit)(logits, y_cls_idx.long())
                else:
                    # Pure regression direction probe - target is continuous yb
                    y_cont = _match_target_shape(logits, yb)
                    base = cast(Any, crit)(logits, y_cont)
            except TypeError:
                # Fallback: treat as multitask-shaped criterion
                y_cont = _match_target_shape(ret_hat, yb)
                return cast(Any, crit)(logits, ret_hat, conf, y_cls_idx, y_cont, None, bet_size=bet_size)
            return _apply_bet_size(base, bet_size).mean() if base.numel() > 1 else base

        if isinstance(crit, MultiPairMultiTaskLoss):
            y_cont = _match_target_shape(ret_hat, yb)
            q_low = model_out[3] if len(model_out) > 3 else None
            q_high = model_out[4] if len(model_out) > 4 else None
            return crit(
                logits=logits,
                ret_hat=ret_hat,
                conf=conf,
                y_cls=y_cls_idx,
                y_cont=y_cont,
                y_conf=y_conf,
                bet_size=bet_size,
                q_low=q_low,
                q_high=q_high,
            )

        if multitask or isinstance(crit, MultiTaskLoss):
            y_cont = _match_target_shape(ret_hat, yb)
            # Detect multi-pair target or prediction shape (B, P)
            if (logits.ndim == 2 and logits.shape[1] > 1) or (isinstance(y_cont, torch.Tensor) and y_cont.ndim == 2 and y_cont.shape[1] > 1):
                P = logits.shape[1] if (logits.ndim == 2 and logits.shape[1] > 1) else y_cont.shape[1]
                q_low = model_out[3] if len(model_out) > 3 else None
                q_high = model_out[4] if len(model_out) > 4 else None
                pair_losses = []
                for p in range(P):
                    l_p = logits[:, p] if logits.ndim == 2 else logits
                    r_p = ret_hat[:, p] if ret_hat.ndim == 2 else ret_hat
                    c_p = conf[:, p] if conf is not None and conf.ndim == 2 else conf
                    yc_idx_p = y_cls_idx[:, p] if y_cls_idx.ndim == 2 else y_cls_idx
                    y_cont_p = y_cont[:, p] if y_cont.ndim == 2 else y_cont
                    y_conf_p = y_conf[:, p] if y_conf is not None and y_conf.ndim == 2 else y_conf
                    ql_p = q_low[:, p] if q_low is not None and q_low.ndim == 2 else q_low
                    qh_p = q_high[:, p] if q_high is not None and q_high.ndim == 2 else q_high
                    pair_losses.append(
                        cast(Any, crit)(
                            l_p, r_p, c_p, yc_idx_p, y_cont_p, y_conf_p, None, None, None, None, bet_size, ql_p, qh_p
                        )
                    )
                return torch.stack(pair_losses).mean()

            # Quantile heads (optional 4th/5th outputs)
            q_low = model_out[3] if len(model_out) > 3 else None
            q_high = model_out[4] if len(model_out) > 4 else None
            # Handle aux heads shifting quantile positions when return_aux=True
            # In that case model_out = (logits, ret, conf, q_low, q_high, recon, vol)
            # Detect recon as 3D/2D tensor with feature dim >1
            if isinstance(crit, MultiTaskLoss) and getattr(crit, "w_quantile", 0) > 0:
                return cast(Any, crit)(
                    logits, ret_hat, conf, y_cls_idx, y_cont, y_conf, None, None, None, None, bet_size, q_low, q_high
                )
            return cast(Any, crit)(
                logits, ret_hat, conf, y_cls_idx, y_cont, y_conf, None, None, None, None, bet_size
            )

        # Tuple output but non-multitask criterion (rare) - direction CE
        if classification:
            if isinstance(crit, nn.CrossEntropyLoss):
                if logits.ndim == 3 and y_cls_idx.ndim == 2:
                    base = crit(logits.reshape(-1, logits.shape[-1]), y_cls_idx.reshape(-1).long())
                elif logits.ndim == 2 and y_cls_idx.ndim == 2:
                    base = crit(logits, y_cls_idx[:, -1].long())
                else:
                    print(f"DEBUG_CE2: logits {logits.shape} y_cls_idx {y_cls_idx.shape}", flush=True)
                    base = crit(logits, y_cls_idx.long())
            else:
                if logits.ndim == 2 and y_cls_idx.ndim == 2:
                    base = crit(logits, y_cls_idx[:, -1].long())
                else:
                    base = crit(logits, y_cls_idx)
        else:
            y_cont = _match_target_shape(ret_hat, yb)
            base = crit(ret_hat, y_cont)
            
        return _apply_bet_size(base, bet_size).mean() if base.numel() > 1 else base

    if classification:
        base = crit(model_out, _direction_class_index(yb, y_cls))
    else:
        yb = _match_target_shape(model_out, yb)
        try:
            base = crit(model_out, yb, weight=y_conf)
        except TypeError:
            base = crit(model_out, yb)

    base = _apply_bet_size(base, bet_size)
    if base.numel() > 1:
        base = base.mean()

    if _OVERCONF_PENALTY is not None:
        return base + _OVERCONF_PENALTY(model_out, yb)
    return base


def _apply_online_miner(online_miner, pred, yb, y_cls_b, batch_idx_t, classification, multitask):
    if online_miner is None or batch_idx_t is None:
        return
    try:
        with torch.no_grad():
            pred_flat = pred[0] if isinstance(pred, tuple) else pred
            if classification or multitask:
                y_cls_idx = _direction_class_index(yb, y_cls_b, classification=True)
                per_sample = (pred_flat.argmax(-1) != y_cls_idx).float()
            else:
                per_sample = torch.abs(pred_flat.ravel() - yb.ravel())
            online_miner.update_batch(
                batch_idx_t.detach().cpu().numpy(),
                per_sample.detach().cpu().numpy(),
            )
    except Exception as exc:
        print(f"[Train] online_miner.update_batch failed: {exc}")


def _apply_kd_loss(pred, teacher_model, xb, loss, distill_weight: float):
    if teacher_model is None:
        return loss
    with torch.no_grad():
        t_pred = teacher_model(xb)
    if isinstance(pred, tuple):
        p_out = pred[0]
        t_out = t_pred[0] if isinstance(t_pred, tuple) else t_pred
    else:
        p_out = pred
        t_out = t_pred if not isinstance(t_pred, tuple) else t_pred[0]
    kd_loss = torch.nn.functional.mse_loss(p_out, t_out)
    return (1.0 - distill_weight) * loss + distill_weight * kd_loss


class _CurriculumProviderConfig:
    """Minimal config facade for providers without a dataclass ``config``.

    The consumer inspects ``provider.config.difficulty`` for the difficulty
    level count; the CustomCurriculumAdapter exposes ``mode``/``kwargs``
    instead, so this shim reports no difficulty sub-config (falls back to
    a default level count).
    """

    def __init__(self, provider):
        self._provider = provider

    @property
    def difficulty(self):
        return None


class _CurriculumProvider:
    """Unified surface over CurriculumManager / CustomCurriculumAdapter.

    Both providers expose update()/get_sample_weights()/get_inclusion_mask(),
    but with different signatures. This shim lets the training loop call either
    through one interface (used for the P3-2 factory wiring).
    """

    def __init__(self, provider):
        self._provider = provider

    @property
    def config(self):
        cfg = getattr(self._provider, "config", None)
        if cfg is not None and hasattr(cfg, "difficulty"):
            return cfg
        return _CurriculumProviderConfig(self._provider)

    def get_inclusion_mask(self, epoch=None):
        return self._provider.get_inclusion_mask()

    def get_sample_weights(self):
        return self._provider.get_sample_weights()

    def set_losses(self, losses) -> None:
        """Feed per-sample epoch losses back into the underlying provider."""
        fn = getattr(self._provider, "set_losses", None)
        if fn is not None:
            fn(losses)

    def update(self, epoch, **kwargs):
        try:
            return self._provider.update(epoch, **kwargs)
        except TypeError:
            allowed = {}
            if "losses" in kwargs:
                allowed["losses"] = kwargs["losses"]
            return self._provider.update(epoch, **allowed)


def _apply_curriculum_weights(loss, pred, yb, crit, classification, batch_idx_t, sample_weight_lookup):
    """Fold per-sample curriculum weights into the batch loss as a weighted mean.

    Only supported for single-output regression criteria that accept a
    ``weight=`` kwarg (HuberLoss, AsymmetricDirectionalLoss, SharpeProxyLoss).
    CE / multitask / tuple-output paths return the plain loss unchanged.
    """
    if classification or isinstance(pred, tuple):
        return loss
    try:
        idx_np = batch_idx_t.detach().cpu().numpy()
    except (AttributeError, RuntimeError):
        return loss
    try:
        sw = torch.as_tensor(
            np.asarray(sample_weight_lookup)[idx_np],
            dtype=torch.float32,
            device=batch_idx_t.device,
        )
    except (IndexError, ValueError, TypeError):
        return loss
    if sw.numel() == 0:
        return loss
    try:
        weighted = crit(pred, yb, weight=sw)
    except TypeError:
        return loss
    if not torch.isfinite(weighted):
        return loss
    return weighted


def _build_train_loss(
    model,
    xb,
    yb,
    y_cls_b,
    y_conf_b,
    bet_size_b,
    crit,
    classification,
    multitask,
    direction_only,
    teacher_model,
    distill_weight,
    ewc_module,
    ewc_lambda,
    si_module,
    si_lambda,
    loader,
    batch_idx_t,
    online_miner,
    accum_steps,
    sample_weight_lookup=None,
):
    """Shared forward + loss assembly used by both AMP and non-AMP paths.

    ``si_lambda`` is the effective Î» for this epoch, computed by the caller
    from the FeatureStabilityMonitor's regime-drift estimate
    (``epoch_si_lambda``): ``Î» = base_Î» / (1 + max_shiftÂ²)`` so the SI penalty
    relaxes under severe distribution shift and re-locks when the regime
    stabilizes.
    """
    pred = model(xb)
    loss = _compute_loss(
        pred,
        crit,
        yb,
        classification,
        y_cls=y_cls_b,
        y_conf=y_conf_b,
        multitask=multitask,
        direction_only=direction_only,
        bet_size=bet_size_b,
    )
    if sample_weight_lookup is not None and batch_idx_t is not None:
        loss = _apply_curriculum_weights(
            loss,
            pred,
            yb,
            crit,
            classification,
            batch_idx_t,
            sample_weight_lookup,
        )
    _apply_online_miner(online_miner, pred, yb, y_cls_b, batch_idx_t, classification, multitask)
    if hasattr(loader, "update_priorities") and batch_idx_t is not None:
        try:
            loader.update_priorities(batch_idx_t, loss.detach())
        except Exception as exc:
            print(f"[Train] update_priorities failed: {exc}")
    loss = _apply_kd_loss(pred, teacher_model, xb, loss, distill_weight)
    if ewc_module is not None:
        # Compute penalty in fp32 to prevent overflow when AMP runs fp16 forward.
        # ewc_lambda * (param - saved)**2 weighted by Fisher can exceed fp16 range
        # (default lambda=1000); cast back to the main loss dtype after addition.
        _loss_dtype = loss.dtype
        loss = apply_ewc_loss(loss.float(), ewc_module, ewc_lambda).to(_loss_dtype)
    if si_module is not None:
        _loss_dtype = loss.dtype
        loss = apply_si_loss(loss.float(), si_module, si_lambda).to(_loss_dtype)
    return pred, loss / accum_steps


def build_criterion(
    args,
    device: torch.device,
    cache_path: str | None = None,
    train_idx: np.ndarray | None = None,
):
    """
    Huber / asymmetric / directional_huber / sharpe_huber regression,
    weighted CE on {-1,0,+1}, or MultiTaskLoss.
    MultiTaskLoss is selected when --multitask is passed and combines:
      w_dir*CE(direction) + w_ret*Huber(return_hat) + w_conf*BCE(confidence)
    """
    multitask = getattr(args, "multitask", False)
    d = float(TRAINING.get("huber_delta", 1.0))

    per_pair_heads = getattr(args, "per_pair_heads", False)
    if per_pair_heads:
        return MultiPairMultiTaskLoss(
            w_dir=1.0,
            w_ret=float(getattr(args, "mt_w_ret", 0.5)),
            w_conf=float(getattr(args, "mt_w_conf", 0.3)),
            huber_delta=d,
            w_quantile=float(getattr(args, "mt_w_quantile", getattr(args, "w_quantile", 0.2))),
            quantiles=(0.05, 0.95),
        ).to(device)

    if multitask:
        cw = None
        cp = None
        if cache_path is not None and train_idx is not None:
            cw = _class_weights_tensor(
                cache_path,
                train_idx,
                device,
                use_direction_sidecar=True,
            )
            cp = _class_prior_tensor(
                cache_path,
                train_idx,
                device,
                use_direction_sidecar=True,
            )
        loss_str = getattr(args, "loss", "huber").lower()
        w_sharpe = float(getattr(args, "sharpe_weight", 0.0)) if loss_str == "sharpe_huber" else 0.0
        from training.train_gpu import _sharpe_ann_factor
        sharpe_ann = _sharpe_ann_factor(args) if w_sharpe > 0 else 1.0
        return MultiTaskLoss(  # type: ignore
            class_weights=cw,
            w_dir=1.0,
            w_ret=float(getattr(args, "mt_w_ret", 0.5)),
            w_conf=float(getattr(args, "mt_w_conf", 0.3)),
            huber_delta=d,
            class_balance_weight=float(getattr(args, "mt_class_balance_weight", 0.0)),
            entropy_weight=float(getattr(args, "mt_entropy_weight", 0.0)),
            direction_weight_floor=float(getattr(args, "mt_direction_weight_floor", 0.0)),
            focal_gamma=float(getattr(args, "mt_focal_gamma", 0.0)),
            class_prior=cp,
            w_sharpe=w_sharpe,
            sharpe_ann=sharpe_ann,
            label_smoothing=float(getattr(args, "label_smoothing", TRAINING.get("label_smoothing", 0.05))),
            w_quantile=float(getattr(args, "mt_w_quantile", getattr(args, "w_quantile", 0.2))),
            quantiles=(0.05, 0.95),
        ).to(device)  # type: ignore

    if args.loss == "cross_entropy":
        print("[Warning] cross_entropy is deprecated for primary objective. Defaulting to HuberLoss.")
        return HuberLoss(delta=d, reduction="none").to(device)  # type: ignore
        
    if args.loss == "asymmetric":
        sw = float(TRAINING.get("asymmetric_sign_weight", 2.0))
        return AsymmetricDirectionalLoss(delta=d, sign_weight=sw, reduction="none").to(device)  # type: ignore
    if args.loss == "directional_huber":
        return DirectionalHuberLoss(
            delta=d,
            direction_weight=float(getattr(args, "direction_weight", 0.5)),
            reduction="none",
        ).to(device)  # type: ignore
    if args.loss == "sharpe_huber":
        from training.train_gpu import _sharpe_ann_factor
        ann = _sharpe_ann_factor(args)
        return SharpeProxyLoss(delta=d, sharpe_weight=float(getattr(args, "sharpe_weight", 0.2)), ann=ann, reduction="none").to(device)  # type: ignore
    return HuberLoss(delta=d, reduction="none").to(device)  # type: ignore


# Public / backward-compatible alias for batch loss calculation
compute_batch_loss = _compute_loss











