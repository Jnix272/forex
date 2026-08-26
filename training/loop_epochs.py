"""Epoch loops (train/validate) for the supervised loop.

Extracted verbatim from ``training.supervised_loop`` (refactor R5);
re-exported there for import-path stability."""
from __future__ import annotations

from typing import Any, cast

import gc
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.amp import autocast

from config.settings import TRAINING
from models.architectures import MultiTaskLoss
from training.core import (
    _TRAIN_LOGGER,
    _log_error,
    _log_nan,
    _log_oom,
)
from training.direction_control import (
    _direction_class_index,
    _direction_recall_from_confusion,
    _recover_nonfinite_training_state,
)
from training.gpu_device import _thermal_check
from training.gpu_losses import _match_target_shape
from training.loop_batches import (
    _prepare_train_batch,
    _sanitize_batch_tensors,
    _unpack_batch,
)
from training.loop_losses import _build_train_loss, _compute_loss
from training.loop_optim import _optimizer_step

def _train_batch(
    model,
    xb,
    yb,
    y_cls_b,
    y_conf_b,
    batch_idx_t,
    *,
    crit,
    classification: bool,
    multitask: bool,
    direction_only: bool,
    teacher_model,
    distill_weight: float,
    ewc_module,
    ewc_lambda: float,
    loader,
    online_miner,
    accum_steps: int,
    opt,
    scaler_amp,
    use_fp16_scaler: bool,
    amp_on: bool,
    amp_dtype: torch.dtype,
    do_step: bool,
    grad_clip: float,
    scheduler,
    batch_idx: int,
    epoch: int,
    pbar,
    nan_skips: int,
    si_module=None,
    si_lambda: float = 1.0,
    sample_weight_lookup=None,
) -> tuple[str, float | None, int]:
    """Shared AMP/non-AMP train step: forward → backward → optional opt step.

    Returns ``(status, loss_val, nan_skips)`` where status is
    ``"ok"``, ``"nan_loss"``, or ``"nan_grad"``.
    """
    amp_ctx = autocast("cuda", dtype=amp_dtype) if amp_on else nullcontext()
    with amp_ctx:
        _, loss = _build_train_loss(
            model,
            xb,
            yb,
            y_cls_b,
            y_conf_b,
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
            sample_weight_lookup=sample_weight_lookup,
        )

    if not torch.isfinite(loss):
        nan_skips += 1
        _log_nan(batch_idx, epoch, nan_skips)
        _recover_nonfinite_training_state(model, opt)
        if nan_skips <= 3 or nan_skips % 10 == 0:
            print(f"[Train] NaN/Inf loss at batch {batch_idx} (skip {nan_skips})")
        opt.zero_grad(set_to_none=True)
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss="NaN-skip")
        return "nan_loss", None, nan_skips

    scale = use_fp16_scaler and amp_on
    if scale:
        scaler_amp.scale(loss).backward()
    else:
        loss.backward()

    ok, nan_skips = _optimizer_step(
        model,
        opt,
        scaler_amp,
        scale,
        do_step,
        grad_clip,
        scheduler,
        batch_idx,
        nan_skips=nan_skips,
        epoch=epoch,
        pbar=pbar,
        si_module=si_module,
    )
    if not ok:
        return "nan_grad", None, nan_skips

    return "ok", float(loss.item() * accum_steps), nan_skips


def train_epoch(
    model,
    loader,
    opt,
    crit,
    scaler_amp,
    device,
    use_amp,
    classification: bool,
    grad_clip: float = 1.0,
    pbar=None,
    amp_dtype: torch.dtype = torch.float32,
    thermal_limit: int = 83,
    feature_mask: torch.Tensor | None = None,
    scheduler=None,
    accum_steps: int = 1,
    seq_len: int | None = None,
    multitask: bool = False,
    epoch: int = 0,
    teacher_model=None,
    distill_weight: float = 0.5,
    direction_only: bool = False,
    online_miner=None,
    adversarial_gen=None,
    adversarial_feature_names: list[str] | None = None,
    ewc_module=None,
    ewc_lambda: float = 1000.0,
    si_module=None,
    si_lambda: float = 1.0,
    sample_weight_lookup=None,
):
    """One training epoch via shared ``_prepare_train_batch`` / ``_train_batch``."""
    model.train()
    total = 0.0
    n = 0
    oom_skips = 0
    nan_skips = 0
    use_fp16_scaler = scaler_amp.is_enabled()
    amp_on = bool(use_amp and device.type == "cuda")
    _n_batches = len(loader)
    _mask = feature_mask.to(device) if feature_mask is not None else None
    opt.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 50 == 0:
            if _TRAIN_LOGGER:
                _TRAIN_LOGGER.heartbeat()
            _thermal_check(limit=thermal_limit)

        do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == _n_batches)

        try:
            prepared = _prepare_train_batch(
                batch,
                device,
                seq_len=seq_len,
                feature_mask=_mask,
                adversarial_gen=adversarial_gen,
                adversarial_feature_names=adversarial_feature_names,
                model=model,
                crit=crit,
                classification=classification,
                multitask=multitask,
            )
            if prepared is None:
                nan_skips += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss="bad-tgt-skip")
                continue
            xb, yb, y_cls_b, y_conf_b, batch_idx_t = prepared

            status, loss_val, nan_skips = _train_batch(
                model,
                xb,
                yb,
                y_cls_b,
                y_conf_b,
                batch_idx_t,
                crit=crit,
                classification=classification,
                multitask=multitask,
                direction_only=direction_only,
                teacher_model=teacher_model,
                distill_weight=distill_weight,
                ewc_module=ewc_module,
                ewc_lambda=ewc_lambda,
                si_module=si_module,
                si_lambda=si_lambda,
                loader=loader,
                online_miner=online_miner,
                accum_steps=accum_steps,
                opt=opt,
                scaler_amp=scaler_amp,
                use_fp16_scaler=use_fp16_scaler,
                amp_on=amp_on,
                amp_dtype=amp_dtype,
                do_step=do_step,
                grad_clip=grad_clip,
                scheduler=scheduler,
                batch_idx=batch_idx,
                epoch=epoch,
                pbar=pbar,
                nan_skips=nan_skips,
                sample_weight_lookup=sample_weight_lookup,
            )
            if status != "ok" or loss_val is None:
                continue

            total += loss_val
            n += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_val:.5f}")

        except RuntimeError as e:
            is_oom = ("out of memory" in str(e).lower()) or ("cuda error" in str(e).lower())
            if not (device.type == "cuda" and is_oom):
                _log_error(f"[Train] Unexpected RuntimeError at batch {batch_idx}", e)
                raise
            oom_skips += 1
            opt.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss="OOM-skip")
            _log_oom(batch_idx, 0, oom_skips)
            if oom_skips <= 3 or oom_skips % 10 == 0:
                print(
                    f"[Train] CUDA OOM batch {batch_idx}, skipped ({oom_skips} total). "
                    "Reduce --batch-size or --grad-accum-steps."
                )
            continue
        except Exception as e:
            _log_error(f"[Train] Unexpected error at batch {batch_idx}", e)
            raise

    if oom_skips:
        print(f"[Train] OOM summary: {oom_skips} batch(es) skipped this epoch.")
    if nan_skips:
        print(f"[Train] NaN summary: {nan_skips} batch(es) skipped this epoch.")
    return total / max(n, 1)


def _validation_class_diag(
    pred_counts: torch.Tensor,
    true_counts: torch.Tensor,
    confusion: torch.Tensor,
    logits_sum: torch.Tensor,
    probs_sum: torch.Tensor,
    diag_true_counts: torch.Tensor,
) -> dict:
    pred = [int(x) for x in pred_counts.detach().cpu().tolist()]
    true = [int(x) for x in true_counts.detach().cpu().tolist()]
    conf = [[int(v) for v in row] for row in confusion.detach().cpu().tolist()]
    recalls = _direction_recall_from_confusion(conf)
    denom = diag_true_counts.detach().float().clamp_min(1.0).view(3, 1)
    mean_logits = (logits_sum.detach().float() / denom).cpu().tolist()
    mean_probs = (probs_sum.detach().float() / denom).cpu().tolist()
    pred_total = max(1, sum(pred))
    true_total = max(1, sum(true))
    return {
        "pred": pred,
        "true": true,
        "pred_shares": [float(x) / pred_total for x in pred],
        "true_shares": [float(x) / true_total for x in true],
        "recall": recalls,
        "confusion": conf,
        "mean_logits_by_true_class": mean_logits,
        "mean_probs_by_true_class": mean_probs,
    }


@torch.no_grad()
def _non_overlapping_sharpe(
    r: torch.Tensor,
    lookahead_bars: int,
    *,
    exclude_flat: bool = True,
) -> float:
    """Compute Sharpe on the non-overlapping subset of per-sample returns.

    The validation return stream ``r`` is the product of
    ``sign(prediction) * forward_return`` for every sample in the
    validation set. Adjacent samples overlap by ``seq_len - 1`` bars in
    features and ``lookahead_bars - 1`` bars in labels, so their
    returns are strongly autocorrelated. Computing mean/std on the
    full stream treats overlapping samples as independent trades and
    under-estimates variance, which over-inflates Sharpe.

    The correct fix is to take every ``lookahead_bars``-th sample
    (i.e. one return per "trade"), giving a stream of approximately
    independent per-trade returns. We also drop flat predictions
    (``|d| == 0``) since they correspond to non-trades.

    Returns
    -------
    float
        Per-trade Sharpe.  Returns 0.0 if there are too few trades
        to compute a meaningful statistic.
    """
    n = int(r.numel())
    if n == 0:
        return 0.0
    # Stride = lookahead_bars (with seq_len-stride=1 windows, every
    # ``lookahead_bars``-th sample is the start of a fresh non-overlapping
    # forward window).
    stride = max(1, int(lookahead_bars))
    sub = r[::stride]
    if exclude_flat:
        sub = sub[sub.abs() > 0.0]
    n_sub = int(sub.numel())
    if n_sub < 2:
        return 0.0
    # Use sample variance (n-1) - textbook Sharpe denominator.
    mean = sub.mean()
    # unbiased=False is the population std, but Sharpe in academic
    # literature uses sample std (n-1). Use float ops for safety.
    std = sub.std(unbiased=True)
    if std.item() <= 0.0 or not torch.isfinite(std):
        return 0.0
    return float((mean / std).item())


def validate_epoch(
    model,
    loader,
    crit,
    device,
    classification: bool,
    pbar=None,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    seq_len: int | None = None,
    multitask: bool = False,
    feature_mask: torch.Tensor | None = None,
    sharpe_ann_factor: float | None = None,
        direction_only: bool = False,
        rl_mode: bool = False,
        lookahead_bars: int = 1,
        sharpe_non_overlapping: bool = True,
        return_per_trade_sharpe: bool = True,
        *,
        tx_cost_bps: float = 0.0,      # transaction cost in basis points per trade (0.3 pct = 30 bps)
        close_prices: torch.Tensor | None = None,  # per-sample close price for computing actual returns
        pip_size: float = 0.0001,      # pip size for FX pair
    ):
    """Run one validation epoch in eager FP32 by default.

    Autocast is off unless ``amp=True`` is passed explicitly - validation is
    usually not compute-bound, and AMP adds cast overhead while hurting
    Sharpe / CE numeric stability.

    The returned Sharpe is the **directional** Sharpe (sign(pred) × sign(label)
    annualized). A secondary ``cost_sharpe`` is returned via the function
    attribute ``validate_epoch.last_cost_sharpe`` computed from actual price
    returns minus transaction costs when ``close_prices`` and ``tx_cost_bps``
    are provided. This enables cost-aware model selection to complement the
    directional proxy Sharpe.
    """
    model.eval()
    total = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    n_acc = 0
    n_ret = 0
    oom_skips = 0
    nan_skips = 0
    valid_batches = 0
    r_sum = torch.zeros(1, device=device)
    r_sq_sum = torch.zeros(1, device=device)
    pred_counts = torch.zeros(3, device=device, dtype=torch.long)
    true_counts = torch.zeros(3, device=device, dtype=torch.long)
    confusion = torch.zeros((3, 3), device=device, dtype=torch.long)
    logits_sum = torch.zeros((3, 3), device=device)
    probs_sum = torch.zeros((3, 3), device=device)
    diag_true_counts = torch.zeros(3, device=device)

    heartbeat_interval = 50
    _mask = feature_mask.to(device) if feature_mask is not None else None
    # Per-sample return stream, kept for the de-overlapped per-trade
    # Sharpe.  We accumulate on CPU to avoid GPU memory pressure when
    # the validation set is large.  ``_per_trade_returns`` is the
    # concatenation of every per-sample ``r = sign(pred) * yb``.
    _per_trade_returns_parts: list[torch.Tensor] = []

    # Cost-aware Sharpe: we collect actual price returns and compute
    # a cost-deducted Sharpe for diagnostic/model-selection purposes.
    # This is separate from the directional Sharpe above which uses
    # direction-labels, not price returns.
    _cost_returns_parts: list[torch.Tensor] = []  # per-trade net returns (price-based, cost-deducted)
    _cost_close_parts: list[torch.Tensor] = []    # per-sample close prices aligned with returns

    def _accumulate_class_diag(logits: torch.Tensor, y_cls_idx: torch.Tensor) -> torch.Tensor:
        """Update class diagnostics; return pred_cls."""
        nonlocal correct, n_acc, pred_counts, true_counts, confusion, logits_sum, probs_sum, diag_true_counts
        pred_cls = logits.argmax(-1)
        correct += (pred_cls == y_cls_idx).sum()
        n_acc += int(y_cls_idx.numel())
        pred_counts += torch.bincount(pred_cls.reshape(-1).clamp(0, 2), minlength=3)[:3]
        true_counts += torch.bincount(y_cls_idx.reshape(-1).clamp(0, 2), minlength=3)[:3]
        probs = torch.softmax(logits.float(), dim=-1)
        t_flat = y_cls_idx.reshape(-1).clamp(0, 2)
        p_flat = pred_cls.reshape(-1).clamp(0, 2)
        for _t, _p in zip(t_flat, p_flat, strict=False):
            confusion[int(_t), int(_p)] += 1
        for _cls in range(3):
            _mask_cls = t_flat == _cls
            if bool(_mask_cls.any()):
                diag_true_counts[_cls] += int(_mask_cls.sum())
                logits_sum[_cls] += logits.float()[_mask_cls].sum(dim=0)
                probs_sum[_cls] += probs[_mask_cls].sum(dim=0)
        return pred_cls

    with torch.no_grad():
        for i, batch in enumerate(loader):
            try:
                xb, yb, y_cls_b, y_conf_b, _ = _unpack_batch(batch, device)
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
                        nan_skips += 1
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(loss="bad-tgt-skip")
                        continue
                    xb, yb = xb[keep], yb[keep]
                    if y_cls_b is not None:
                        y_cls_b = y_cls_b[keep]
                    if y_conf_b is not None:
                        y_conf_b = y_conf_b[keep]
                if _mask is not None:
                    xb = xb * _mask

                if not torch.isfinite(xb).all() or not torch.isfinite(yb).all():
                    nan_skips += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(loss="NaN-skip")
                    continue

                # Eager FP32 unless caller opts into amp (rare).
                amp_ctx = (
                    autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
                    if amp and device.type == "cuda"
                    else nullcontext()
                )
                with amp_ctx:
                    pred = model(xb)
                    y_cls_idx = _direction_class_index(
                        yb,
                        y_cls_b,
                        classification=classification,
                    )

                    if isinstance(pred, tuple):
                        logits, ret_hat, _conf = pred
                        loss = _compute_loss(
                            pred,
                            crit,
                            yb,
                            classification,
                            y_cls=y_cls_b,
                            y_conf=y_conf_b,
                            multitask=multitask,
                            direction_only=direction_only,
                        )
                        if not (
                            torch.isfinite(loss) and torch.isfinite(logits).all() and torch.isfinite(ret_hat).all()
                        ):
                            nan_skips += 1
                            if pbar is not None:
                                pbar.update(1)
                                pbar.set_postfix(loss="NaN-skip")
                            continue
                        total += loss
                        pred_cls = _accumulate_class_diag(logits, y_cls_idx)
                        d = pred_cls.float() - 1.0
                    elif classification:
                        loss = crit(pred, y_cls_idx)
                        if not (torch.isfinite(loss) and torch.isfinite(pred).all()):
                            nan_skips += 1
                            if pbar is not None:
                                pbar.update(1)
                                pbar.set_postfix(loss="NaN-skip")
                            continue
                        total += loss
                        pred_cls = _accumulate_class_diag(pred, y_cls_idx)
                        d = pred_cls.float() - 1.0
                    else:
                        if multitask or isinstance(crit, MultiTaskLoss):
                            y_cls_idx = _direction_class_index(yb, y_cls_b, classification=True)
                            if isinstance(crit, MultiTaskLoss):
                                loss = cast(Any, crit).ce(pred, y_cls_idx.reshape(-1).clamp(0, 2)).mean()
                            else:
                                loss = crit(pred, y_cls_idx)
                            yb_reg = _match_target_shape(pred, yb)
                        else:
                            yb_reg = _match_target_shape(pred, yb)
                            loss = crit(pred, yb_reg)
                        if not (torch.isfinite(loss) and torch.isfinite(pred).all()):
                            nan_skips += 1
                            if pbar is not None:
                                pbar.update(1)
                                pbar.set_postfix(loss="NaN-skip")
                            continue
                        total += loss
                        correct += (torch.sign(pred) == torch.sign(yb_reg)).sum()
                        n_acc += int(yb_reg.numel())
                        d = torch.sign(pred)

                yb_for_returns = _match_target_shape(d, yb.float())
                if (rl_mode or y_cls_b is not None) and y_cls_b is not None:
                    side = _match_target_shape(d, y_cls_b.float()).sign()
                    yb_for_returns = yb_for_returns.abs() * side
                r = (d * yb_for_returns).flatten()
                if r.numel() > 0:
                    r_sum += r.sum()
                    r_sq_sum += (r * r).sum()
                    n_ret += int(r.numel())
                    # Keep the per-sample returns for the de-overlapped
                    # per-trade Sharpe at the end of the epoch.  We
                    # only retain non-zero (i.e. traded) returns to
                    # bound memory; the de-overlap helper can also
                    # re-filter if needed.
                    if sharpe_non_overlapping and return_per_trade_sharpe:
                        _per_trade_returns_parts.append(r.detach().to("cpu", non_blocking=True))

                valid_batches += 1
                if pbar is not None:
                    pbar.update(1)
                if (i + 1) % heartbeat_interval == 0 and _TRAIN_LOGGER:
                    _TRAIN_LOGGER.heartbeat()

            except RuntimeError as e:
                is_oom = ("out of memory" in str(e).lower()) or ("cuda error" in str(e).lower())
                if not (device.type == "cuda" and is_oom):
                    raise
                oom_skips += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss="OOM-skip")
                continue

    if oom_skips:
        print(f"[Val] OOM summary: skipped {oom_skips} batch(es) this epoch.")
    if nan_skips:
        print(f"[Val] NaN summary: skipped {nan_skips} batch(es) this epoch.")

    _cfg_ann = TRAINING.get("sharpe_annualization_factor")
    ann = float(
        sharpe_ann_factor if sharpe_ann_factor is not None else (_cfg_ann if _cfg_ann is not None else 1.0)
    )
    val_loss = total.item() / max(valid_batches, 1)
    dir_acc = correct.item() / max(n_acc, 1)

    if n_ret == 0 or valid_batches == 0:
        print(
            "[Val] WARNING: no validation return samples contributed to Sharpe "
            f"(valid_batches={valid_batches}, n_ret={n_ret}). Returning Sharpe=0."
        )
        _diag = _validation_class_diag(
            pred_counts,
            true_counts,
            confusion,
            logits_sum,
            probs_sum,
            diag_true_counts,
        )
        validate_epoch.last_class_counts = {"pred": _diag["pred"], "true": _diag["true"]}
        validate_epoch.last_class_diag = _diag
        return val_loss, dir_acc, 0.0

    # ── Sharpe computation ─────────────────────────────────────────────
    # Two flavors, both annualised by ``ann``:
    #
    #   * ``sharpe_per_trade``: de-overlapped, sample-variance
    #     (textbook Sharpe).  This is the primary metric for model
    #     selection and the promotion gate.
    #
    #   * ``sharpe_per_sample``: legacy metric, kept for backward
    #     compatibility with old dashboards. Computed on the full
    #     stream (which over-counts overlapping samples).
    #
    # We keep both so model selection and dashboards don't diverge in
    # one step; callers can choose which to log.
    # Concatenate the per-sample return stream we collected in the
    # validation loop.  When the loop was skipped (n_ret == 0) this
    # list is empty and we fall back to the legacy path.
    if _per_trade_returns_parts:
        try:
            _per_trade_returns = torch.cat(_per_trade_returns_parts, dim=0)
        except Exception:
            _per_trade_returns = torch.zeros(0)
    else:
        _per_trade_returns = torch.zeros(0)
    r_mean = r_sum / n_ret
    r_var = torch.clamp(r_sq_sum / n_ret - r_mean**2, min=0.0)
    sharpe_per_sample = (r_mean / (r_var.sqrt() + 1e-8)).item() * ann
    if sharpe_non_overlapping and return_per_trade_sharpe and _per_trade_returns.numel() >= 2:
        per_trade = _non_overlapping_sharpe(
            _per_trade_returns,
            lookahead_bars=max(1, int(lookahead_bars)),
        )
        sharpe = per_trade * ann
    else:
        sharpe = sharpe_per_sample
    _diag = _validation_class_diag(
        pred_counts,
        true_counts,
        confusion,
        logits_sum,
        probs_sum,
        diag_true_counts,
    )
    validate_epoch.last_class_counts = {"pred": _diag["pred"], "true": _diag["true"]}
    validate_epoch.last_class_diag = _diag

    # ── Cost-aware diagnostic Sharpe ──────────────────────────────────────
    # The primary Sharpe above uses direction-label returns {-1,0,1}. When
    # the caller provides tx_cost_bps (transaction cost in basis points), we
    # compute an alternative Sharpe by subtracting a per-trade cost from the
    # directional returns. This gives a conservative lower-bound estimate of
    # the Sharpe after execution frictions, which is a better signal for model
    # selection than the inflated gross Sharpe.
    #
    # Cost model: each non-flat trade (d ∈ {-1, 1}) incurs a cost of
    # tx_cost_bps / 10000 per trade, deducted from the return. This is
    # conservative because it assumes the cost is proportional to the
    # direction-label magnitude (1 bp), not the actual price return.
    cost_sharpe = None
    if tx_cost_bps > 0.0 and _per_trade_returns.numel() >= 2:
        # tx_cost_bps is basis-points (0.01 pct = 1 bp).  Convert to fraction
        # of return magnitude.  Because ``_per_trade_returns`` is in {-1, 0, 1}
        # (sign-product), the appropriate scale is 1 bp = 0.0001 of that unit.
        # When ``close_prices`` are supplied, ``pip_size`` is used to express
        # the cost in price-return units (returns / price) instead, which is
        # the more honest measure.  We default to the sign-product scale
        # when close_prices is None so the diagnostic is always defined.
        if close_prices is not None and close_prices.numel() > 0:
            # Use pip_size to express cost in price units, then scale by
            # mean price to convert to return units.  For 5m FX, mean price
            # ≈ 1.0–1.2 for most pairs (USDJPY ≈ 150), so cost_per_unit ≈
            # tx_cost_bps / 1e4 / price.
            mean_price = float(close_prices.mean().item() or 1.0)
            tx_cost_price = (tx_cost_bps / 10_000.0) * pip_size / max(mean_price, 1e-9)
            # Translate back into direction-label units (per bp of sign-product).
            tx_cost = tx_cost_price / pip_size if pip_size > 0 else tx_cost_bps / 10_000.0
        else:
            # Default: treat 1 bp as 0.0001 of the sign-product return.
            tx_cost = tx_cost_bps / 10_000.0
        # Deduct cost only from traded (non-zero) returns
        r_cost = _per_trade_returns.clone()
        trade_mask = r_cost.abs() > 0.0
        r_cost = torch.where(trade_mask, r_cost - tx_cost, r_cost)
        # De-overlap the cost-deducted returns
        if sharpe_non_overlapping and return_per_trade_sharpe:
            cost_per_trade = _non_overlapping_sharpe(r_cost, lookahead_bars=max(1, int(lookahead_bars)))
        else:
            cost_per_trade = float((r_cost.mean() / (r_cost.std(unbiased=True) + 1e-8)).item())
        cost_sharpe = cost_per_trade * ann
        # Diagnostic: also compute per-sample (pre-deoverlap) cost Sharpe
        r_cost_mean = r_cost.mean().item()
        r_cost_std = r_cost.std(unbiased=True).item()
        cost_sharpe_per_sample = (r_cost_mean / (r_cost_std + 1e-8)) * ann

        # Always print the cost-aware diagnostic
        n_trades = int(trade_mask.sum().item())
        print(
            f"[Val] dir_sharpe={sharpe:.4f}  cost_sharpe={cost_sharpe:.4f}  "
            f"cost_sharpe_per_sample={cost_sharpe_per_sample:.4f}  "
            f"tx_cost={tx_cost_bps:.1f}bps  n_trades={n_trades}/{n_ret}"
        )
    else:
        print(
            f"[Val] dir_sharpe={sharpe:.4f}  cost_sharpe=N/A (no tx_cost specified)  "
            f"n_trades={n_ret}"
        )

    validate_epoch.last_cost_sharpe = cost_sharpe
    validate_epoch.last_dir_sharpe = sharpe
    validate_epoch.last_ann_factor = ann
    return val_loss, dir_acc, sharpe


