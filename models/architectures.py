"""
models/architectures.py
========================
All ten model architectures specified:
  1. TFT           - Temporal Fusion Transformer
  2. iTransformer  - Variate-dimension attention
  3. HAELTHybrid   - LSTM + Transformer parallel
  4. MambaScalper  - State Space Model (low latency)
  5. GNNCrossAsset - Graph Neural Network for cross-asset correlations
  6. EXPERTEncoder - Exchange-Rate Transformer (conv FFN, no positional enc)
  7. PatchTST      - Patch Time Series Transformer (channel-independent)
  8. GLM           - Generalized Linear baseline
  9. TimesNet      - Temporal 2D-Variation multiscale (FFT + Inception)
  10. TimeMixer    - Decomposable multiscale mixing (trend/seasonal)

Shared interface: forward(x) -> (batch,) scalars if num_classes==1, else (batch, num_classes) logits.
"""

import argparse
import inspect
import warnings
from types import SimpleNamespace
from typing import Any, cast

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint  # noqa: F401  (attribute access via torch.utils.checkpoint)

    TORCH = True
except ImportError:
    TORCH = False
    warnings.warn("PyTorch not installed. pip install torch", stacklevel=2)


def build_model(name: str, input_size: int, seq_len: Any | None = 60, **kwargs) -> Any:
    """Runtime-safe model factory that works in both torch and stub environments."""
    if not TORCH:
        print(f"[Model] Stub for {name} (torch not installed)")
        return None

    base_name = name.lower()
    if base_name.startswith("baseline_"):
        base_name = base_name.replace("baseline_", "", 1)
        
    if base_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")

    cls = MODEL_REGISTRY[base_name]
    if isinstance(seq_len, (argparse.Namespace, SimpleNamespace)):
        for k, v in vars(seq_len).items():
            if k not in kwargs:
                kwargs[k] = v
        seq_len = getattr(seq_len, "seq_len", 60)

    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if "input_size" in params and "input_size" not in kwargs:
        kwargs["input_size"] = input_size
    if "seq_len" in params and "seq_len" not in kwargs:
        kwargs["seq_len"] = seq_len if isinstance(seq_len, int) else 60

    _ALIASES = {
        "hidden": "hidden_size",
        "hidden_channels": "hidden",
        "heads": "nhead",
        "lstm_hidden": "hidden_size",
        "n_layers": "num_layers",
        "lstm_layers": "num_layers",
        "dim_ff": "dim_feedforward",
    }
    for canon, alt in _ALIASES.items():
        if canon in params and canon not in kwargs and alt in kwargs:
            kwargs[canon] = kwargs[alt]
    for canon, alt in _ALIASES.items():
        if alt in params and alt not in kwargs and canon in kwargs:
            kwargs[alt] = kwargs[canon]

    if name.lower() == "haelt":
        if "d_model" in kwargs:
            kwargs["d_model"] = kwargs["d_model"] // 2
        if "nhead" in kwargs:
            kwargs["nhead"] = max(2, kwargs["nhead"] // 2)
        if "lstm_hidden" in kwargs and "hidden_size" in kwargs:
            kwargs["lstm_hidden"] = kwargs["hidden_size"] // 2

    # Default to a single continuous scalar output (Linear head) for regression.
    if "num_classes" in params and "num_classes" not in kwargs:
        kwargs["num_classes"] = 1

    valid_kwargs = {k: v for k, v in kwargs.items() if k in params}
    try:
        model = cls(**valid_kwargs)
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            print(f"[Model] WARN: {name} init failed with filtered kwargs, retrying minimal...")
            model = cls(input_size=input_size)
        else:
            raise e

    # -- Multi-pair embedding wrapper ------------------------------------------
    # Restored from the original train_gpu.build_model: when training on
    # multiple pairs with pair_embed_dim > 0, wrap the backbone in
    # MultiPairWrapper. The backbone must be built with the combined input
    # width (pairs_flat + cross-interaction features), so recompute it here
    # and rebuild when the caller passed the flat width.
    n_pairs = int(kwargs.get("_n_pairs", 1) or 1)
    embed_dim = int(kwargs.get("pair_embed_dim", 0) or 0)
    f_per_pair = int(kwargs.get("_f_per_pair", 0) or 0)
    use_pair_emb = n_pairs > 1 and embed_dim > 0
    if use_pair_emb:
        _fpp = f_per_pair if f_per_pair > 0 else max(1, int(input_size) // n_pairs)
        _n_cross = n_pairs * (n_pairs - 1) // 2
        _n_interaction = 3 * _n_cross + n_pairs + 2
        backbone_input = n_pairs * (_fpp + embed_dim) + _n_interaction
        _orig_input = int(valid_kwargs.get("input_size", input_size))
        if _orig_input != backbone_input:
            rebuild_kwargs = dict(valid_kwargs)
            rebuild_kwargs["input_size"] = backbone_input
            model = cls(**rebuild_kwargs)
        m = MultiPairWrapper(
            model,
            n_pairs=n_pairs,
            f_per_pair=_fpp,
            embed_dim=embed_dim,
            corr_window=int(kwargs.get("corr_window", 20) or 20),
            corr_window_long=int(kwargs.get("corr_window_long", 60) or 60),
            momentum_window=int(kwargs.get("momentum_window", 20) or 20),
        )
        print(
            f"[Model] {name.upper()} | MultiPair wrapper "
            f"({n_pairs}P x {_fpp}F + {embed_dim}E | "
            f"corr={kwargs.get('corr_window', 20)}/{kwargs.get('corr_window_long', 60)}bar "
            f"mom={kwargs.get('momentum_window', 20)}bar) applied."
        )
        model = m

    try:
        n_params = sum(p.numel() for p in model.parameters())
    except ValueError:
        n_params = 0
    print(f"[Model] {name.upper()} | {n_params:,} parameters | applied_params={list(valid_kwargs.keys())}")

    per_pair_heads = bool(
        kwargs.get("per_pair_heads", False)
        or getattr(kwargs.get("args", None), "per_pair_heads", False)
        or kwargs.get("n_pair_heads", None) is not None
        or getattr(kwargs.get("args", None), "n_pair_heads", None) is not None
    )
    if kwargs.get("multitask", False) or getattr(kwargs.get("args", None), "multitask", False) or per_pair_heads:
        head_in = getattr(model, "d_model", getattr(model, "hidden_size", getattr(model, "embed_dim", 128)))
        if name.lower() == "haelt":
            head_in = kwargs.get("d_model", 128) * 2  # haelt uses cat(lstm, transformer)
        
        if TORCH:
            if per_pair_heads:
                n_pair_heads = kwargs.get("n_pair_heads", None) or getattr(kwargs.get("args", None), "n_pair_heads", None)
                pairs_arg = kwargs.get("pairs", None) or getattr(kwargs.get("args", None), "pairs", None)
                if pairs_arg is None:
                    pairs_arg = n_pair_heads if n_pair_heads is not None else n_pairs
                model = MultiPairMultiTaskWrapper(
                    model,
                    head_in=head_in,
                    pairs=pairs_arg,
                    hidden=64,
                    dropout=kwargs.get("dropout", 0.1),
                    proj_threshold=1024,
                    proj_to=256,
                    force_project=True,
                    quantile_enabled=kwargs.get("quantile_enabled", True),
                )
                print(f"[Model] {name.upper()} | MultiPairMultiTask wrapper (pairs={pairs_arg}, head_in={head_in}) applied.")
            else:
                model = MultiTaskWrapper(
                    model,
                    head_in=head_in,
                    hidden=64,
                    dropout=kwargs.get("dropout", 0.1),
                    proj_threshold=1024,
                    proj_to=256,
                    force_project=True,
                )
                print(f"[Model] {name.upper()} | MultiTask wrapper (head_in={head_in}) applied.")

    return model


if TORCH:
    # â”€â”€ Shared building blocks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _kaiming_init_module(mod: "nn.Module") -> None:
        """Xavier/Kaiming-stable init; moderate head gain keeps outputs ~O(1)."""
        for name, m in mod.named_modules():
            if isinstance(m, nn.Linear):
                # Final prediction heads: gain=0.1 (was 0.01 - too vanishing for deep nets).
                is_head = name == "head" or name.startswith("head.") or name.endswith(".head") or ".head." in name
                gain = 0.1 if is_head else 1.0
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                for name_param, p in m.named_parameters():
                    if "weight" in name_param:
                        nn.init.xavier_uniform_(p)
                    elif "bias" in name_param:
                        nn.init.zeros_(p)
                        if isinstance(m, nn.LSTM) and ("bias_hh" in name_param or "bias_ih" in name_param):
                            n = p.size(0)
                            # PyTorch gate ordering: input, forget, cell, output
                            start, end = n // 4, n // 2
                            p.data[start:end].fill_(1.0)

    def _maybe_checkpoint(fn, *args, enabled: bool = True):
        """Gradient checkpointing when training; no-op at eval / when disabled."""
        if enabled and torch.is_grad_enabled() and any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    class _FlashMHA(nn.Module):
        """
        Multi-head self-attention that calls F.scaled_dot_product_attention directly.

        Unlike nn.MultiheadAttention, this bypasses PyTorch's internal dispatch
        guards and unconditionally routes to the Flash Attention / mem-efficient
        SDPA kernel when tensors are in FP16 or BF16 (autocast context).

        Requirements for Flash Attention kernel (PyTorch >= 2.0, CUDA):
          â€¢ Tensors must be FP16 or BF16 - satisfied by AMP autocast.
          â€¢ No custom attn_mask - self-attention only, no causal mask needed.
          â€¢ Head dim should be 16, 32, 64, or 128 for best performance.
        """

        def __init__(self, d_model: int, heads: int, dropout: float = 0.0):
            super().__init__()
            assert d_model % heads == 0, f"d_model ({d_model}) must be divisible by heads ({heads})"
            self.heads = heads
            self.head_dim = d_model // heads
            self.scale = self.head_dim**-0.5
            self.dropout = dropout
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, d_model)
            B, T, _ = x.shape
            qkv = self.qkv(x)  # (B, T, 3*d)
            q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, d)
            # Reshape to (B, heads, T, head_dim) for SDPA
            q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
            # F.scaled_dot_product_attention -> Flash Attention when FP16/BF16
            attn_drop = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
            # Merge heads: (B, heads, T, head_dim) -> (B, T, d_model)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            return self.out_proj(out)

    class HuberLoss(nn.Module):
        def __init__(self, delta=1.0, sign_weight=0.0, reduction="mean"):
            super().__init__()
            self.delta = delta
            self.sign_weight = sign_weight
            self.reduction = reduction

        def forward(self, p, t, weight=None):
            e = p - t
            a = e.abs()
            base = torch.where(a <= self.delta, 0.5 * e**2, self.delta * (a - 0.5 * self.delta))
            dir_penalty = (torch.sign(p) != torch.sign(t)).float() * self.sign_weight * a
            loss = base + dir_penalty
            if weight is not None:
                loss = loss * weight
            if self.reduction == "none":
                return loss
            return loss.mean()

    class AsymmetricDirectionalLoss(nn.Module):
        """
        Huber on residuals plus extra penalty when prediction and target disagree in
        sign (direction), as discussed for asymmetric economic risk in directional
        forecasting. Targets are typically {-1,0,+1} bar labels.
        """

        def __init__(self, delta=1.0, sign_weight=2.0, reduction="mean"):
            super().__init__()
            self.delta = delta
            self.sign_weight = sign_weight
            self.reduction = reduction

        def forward(self, pred, target, weight=None):
            e = pred - target
            a = e.abs()
            huber = torch.where(
                a <= self.delta,
                0.5 * e**2,
                self.delta * (a - 0.5 * self.delta),
            )
            tnz = target.abs() > 0.05
            wrong = tnz & (torch.sign(pred) != torch.sign(target))
            extra = wrong.float() * a * target.abs().clamp(min=0.1)
            loss = huber + self.sign_weight * extra
            if weight is not None:
                loss = loss * weight
            if self.reduction == "none":
                return loss
            return loss.mean()

    # â”€â”€ Multi-task head, loss, and backbone wrapper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    class MultiTaskHead(nn.Module):
        """
        Three-output prediction head for multi-task supervision:
          direction  - scalar proxy for direction (Huber)
          return_hat - scalar magnitude regression (Huber)
          confidence - predicted |return|, clipped to [0,1] (BCE)

        Training on all three signals simultaneously prevents the backbone from
        learning 'correct direction / wrong magnitude' solutions and gives a
        natural confidence signal for downstream position sizing.
        """

        def __init__(
            self,
            in_features: int,
            hidden: int = 64,
            dropout: float = 0.1,
            return_aux: bool = False,
            recon_out_features: int | None = None,
            quantile_enabled: bool = False,
            quantiles: tuple[float, float] = (0.05, 0.95),
        ):
            super().__init__()
            self.return_aux = bool(return_aux)
            self.quantile_enabled = bool(quantile_enabled)
            self.quantiles = tuple(quantiles)
            h2 = max(hidden // 2, 16)
            self.direction = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self.return_hat = nn.Sequential(
                nn.Linear(in_features, h2),
                nn.GELU(),
                nn.Linear(h2, 1),
            )
            self.confidence = nn.Sequential(
                nn.Linear(in_features, h2),
                nn.GELU(),
                nn.Linear(h2, 1),
                # Sigmoid removed - BCEWithLogitsLoss in MultiTaskLoss fuses it
                # safely under AMP. Do NOT add Sigmoid back here.
            )
            if self.quantile_enabled:
                # VaR/CVaR quantile heads for risk-aware sizing
                self.quantile_low = nn.Sequential(
                    nn.Linear(in_features, h2),
                    nn.GELU(),
                    nn.Linear(h2, 1),
                )
                self.quantile_high = nn.Sequential(
                    nn.Linear(in_features, h2),
                    nn.GELU(),
                    nn.Linear(h2, 1),
                )
            if self.return_aux:
                recon_dim = int(recon_out_features) if recon_out_features is not None else in_features
                self.recon = nn.Sequential(
                    nn.Linear(in_features, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, recon_dim),
                )
                self.vol_hat = nn.Sequential(
                    nn.Linear(in_features, h2),
                    nn.GELU(),
                    nn.Linear(h2, 1),
                )

        def forward(self, h: torch.Tensor):
            """h: (B, in_features) - backbone hidden state BEFORE any prediction head."""
            ret = self.return_hat(h)  # (B, 1)
            conf = self.confidence(h)  # (B, 1)
            dir_pred = self.direction(h)  # (B, 1)
            outs = (
                dir_pred.reshape(-1),  # (B,)
                ret.reshape(-1),  # (B,)
                conf.reshape(-1),  # (B,)
            )
            if self.quantile_enabled:
                q_low = self.quantile_low(h).reshape(-1)  # (B,) 5th percentile VaR
                q_high = self.quantile_high(h).reshape(-1)  # (B,) 95th percentile
                outs = (*outs, q_low, q_high)
            if self.return_aux:
                return (*outs, self.recon(h), self.vol_hat(h).reshape(-1))
            return outs

    class MultiTaskLoss(nn.Module):
        """
        Multi-task objective (audit 2026-09-25 S1-S6):

          direction : BCE(logit, y_cls == BUY) on BUY/SELL rows only (HOLD rows are
                      masked out), with per-class weights, label smoothing and an
                      optional focal factor. The CPAR y_cls labels are cost-aware,
                      so the head learns "which side clears costs", not the sign of
                      the mid move.
          trade     : BCE(conf, y_cls != HOLD) - the confidence head predicts
                      whether a trade clears costs at all; decide() gates on it.
          return    : Huber(return_hat, clip(y_cont, +/-target_clip)) - tails of
                      the ATR-normalised reward no longer dominate.
          balance   : class_balance_weight * (mean P(buy) - 0.5)^2 - discourages
                      collapsing to one side.
          sharpe    : optional -w_sharpe * batch Sharpe of tanh(logit) * y_cont.
          quantiles : pinball loss on q_low / q_high (clipped target).

        Every term is a per-sample vector combined with ``bet_size`` (per-sample
        weights: period balance, curriculum) as a weighted mean. Unknown keyword
        arguments raise, so config options can no longer be silently dropped.
        """

        _ACCEPTED = {"class_weights", "focal_gamma", "label_smoothing", "class_balance_weight",
                     "w_sharpe", "sharpe_ann", "target_clip", "recon_w", "vol_w"}

        def __init__(
            self,
            w_dir: float = 1.0,
            w_ret: float = 0.5,
            w_conf: float = 0.3,
            huber_delta: float = 1.0,
            recon_w: float = 0.1,
            vol_w: float = 0.05,
            w_quantile: float = 0.2,
            quantiles: tuple[float, float] = (0.05, 0.95),
            class_weights: "torch.Tensor | None" = None,
            focal_gamma: float = 0.0,
            label_smoothing: float = 0.0,
            class_balance_weight: float = 0.0,
            w_sharpe: float = 0.0,
            sharpe_ann: float = 1.0,
            target_clip: float = 5.0,
            **kwargs,
        ):
            super().__init__()
            if kwargs:
                raise TypeError(f"MultiTaskLoss: unsupported options {sorted(kwargs)} (they would be ignored)")
            self.w_dir = w_dir
            self.w_ret = w_ret
            self.w_conf = w_conf
            self.w_quantile = float(w_quantile)
            self.quantiles = tuple(quantiles)
            self.hub = nn.HuberLoss(delta=huber_delta, reduction="none")
            self.bce = nn.BCEWithLogitsLoss(reduction="none")
            self.recon_w = recon_w
            self.vol_w = vol_w
            self.focal_gamma = float(focal_gamma or 0.0)
            self.label_smoothing = float(label_smoothing or 0.0)
            self.class_balance_weight = float(class_balance_weight or 0.0)
            self.w_sharpe = float(w_sharpe or 0.0)
            self.sharpe_ann = float(sharpe_ann or 1.0)
            self.target_clip = float(target_clip) if target_clip else 0.0
            cw = torch.ones(3) if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32).reshape(-1)[:3]
            self.register_buffer("class_weights", cw.float())

        @staticmethod
        def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
            """Pinball loss for quantile q in (0,1)."""
            diff = target - pred
            return torch.where(diff >= 0, q * diff, (q - 1) * diff)

        @staticmethod
        def _wmean(v: torch.Tensor, w: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            if mask is not None:
                w = w * mask.float()
            den = w.sum()
            return (v * w).sum() / den if float(den) > 0 else v.sum() * 0.0

        def forward(
            self,
            logits: torch.Tensor,
            ret_hat: torch.Tensor,
            conf: torch.Tensor,
            y_cls: torch.Tensor,
            y_cont: torch.Tensor,
            y_conf: torch.Tensor | None = None,
            recon_hat: torch.Tensor | None = None,
            recon_tgt: torch.Tensor | None = None,
            vol_hat: torch.Tensor | None = None,
            vol_tgt: torch.Tensor | None = None,
            bet_size: torch.Tensor | None = None,
            q_low: torch.Tensor | None = None,
            q_high: torch.Tensor | None = None,
        ) -> torch.Tensor:
            logit = logits.reshape_as(y_cont).float()
            ret = ret_hat.reshape_as(y_cont).float()
            cf = conf.reshape_as(y_cont).float()
            # y_cls arrives as a class index {0: SELL, 1: HOLD, 2: BUY}; without it
            # fall back to the sign of the reward.
            if y_cls is None:
                cls = (torch.sign(y_cont) + 1).long().clamp(0, 2)
            else:
                cls = y_cls.reshape_as(y_cont).long().clamp(0, 2)
            w = torch.ones_like(y_cont, dtype=torch.float32)
            if bet_size is not None:
                w = w * bet_size.reshape(-1, *([1] * (y_cont.dim() - 1))).expand_as(y_cont).float()

            # Direction: BUY vs SELL on tradable rows only.
            trade = cls != 1
            tgt = (cls == 2).float()
            if self.label_smoothing > 0:
                tgt = tgt * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            l_dir = self.bce(logit, tgt)
            if self.focal_gamma > 0:
                p = torch.sigmoid(logit)
                p_t = torch.where(cls == 2, p, 1 - p)
                l_dir = l_dir * (1 - p_t).clamp(min=0).pow(self.focal_gamma)
            w_cls = self.class_weights.to(logit.device)[cls]
            loss = self.w_dir * self._wmean(l_dir, w * w_cls, trade)

            # Trade (abstain) head: does this row clear costs at all?
            tgt_trade = trade.float() if y_conf is None else y_conf.reshape_as(y_cont).float()
            loss = loss + self.w_conf * self._wmean(self.bce(cf, tgt_trade), w)

            # Return head on a winsorised target.
            y_reg = y_cont.float()
            if self.target_clip > 0:
                y_reg = y_reg.clamp(-self.target_clip, self.target_clip)
            loss = loss + self.w_ret * self._wmean(self.hub(ret, y_reg), w)

            if self.class_balance_weight > 0:
                loss = loss + self.class_balance_weight * (torch.sigmoid(logit).mean() - 0.5).pow(2)
            if self.w_sharpe > 0 and y_cont.numel() > 2:
                pnl = torch.tanh(logit) * y_reg
                sr = pnl.mean() / (pnl.std() + 1e-6) * (self.sharpe_ann ** 0.5)
                loss = loss - self.w_sharpe * sr
            if recon_hat is not None and recon_tgt is not None and self.recon_w > 0:
                loss = loss + self.recon_w * self.hub(recon_hat, recon_tgt).mean()
            if vol_hat is not None and vol_tgt is not None and self.vol_w > 0:
                loss = loss + self.vol_w * self.hub(vol_hat, vol_tgt).mean()
            if self.w_quantile > 0:
                if q_low is not None:
                    l_q = self._pinball_loss(q_low.reshape_as(y_reg), y_reg, self.quantiles[0])
                    loss = loss + self.w_quantile * self._wmean(l_q, w)
                if q_high is not None:
                    l_q = self._pinball_loss(q_high.reshape_as(y_reg), y_reg, self.quantiles[1])
                    loss = loss + self.w_quantile * self._wmean(l_q, w)
            return loss

    class MultiTaskWrapper(nn.Module):
        """
        Wraps any of the 6 backbone architectures, replacing its .head with
        nn.Identity() to expose the pre-head hidden state, then routing that
        state through a MultiTaskHead.

        After wrapping, forward() returns (direction_logits, return_hat, confidence)
        instead of the backbone's scalar/logit prediction.

        When the backbone's pre-head dimension exceeds proj_threshold (e.g.
        iTransformer whose head_in = d_model x n_features â‰ˆ 18k), an extra
        Linear+GELU projection to proj_to=256 is inserted automatically.

        Usage:
            base  = HAELTHybrid(input_size=73, num_classes=1)
            model = MultiTaskWrapper(base, head_in=256)
            logits, ret, conf = model(x)   # x: (B, T, F)
        """

        def __init__(
            self,
            backbone: "nn.Module",
            head_in: int,
            hidden: int = 64,
            dropout: float = 0.1,
            proj_threshold: int = 1024,
            proj_to: int = 256,
            force_project: bool = False,
            return_aux: bool = False,
            recon_out_features: int | None = None,
        ):
            super().__init__()
            self.backbone = backbone

            if force_project:
                # MultiPairWrapper and iTransformer can change the flattened
                # pre-head width after build-time estimates. LazyLinear binds
                # to the actual tensor width on the first forward pass.
                self.proj = nn.Sequential(nn.LazyLinear(proj_to), nn.GELU())
                actual_in = proj_to
            elif head_in > proj_threshold:
                self.proj = nn.Sequential(nn.Linear(head_in, proj_to), nn.GELU())
                actual_in = proj_to
            else:
                self.proj = nn.Identity()
                actual_in = head_in

            self.mt_head = MultiTaskHead(
                actual_in,
                hidden,
                dropout,
                return_aux=return_aux,
                recon_out_features=recon_out_features,
            )
            # Walk through wrapper chain to find the actual model with a head.
            # Pylance cannot safely infer the dynamic "backbone" chain on the base
            # nn.Module type, so narrow via Any at the runtime boundary.
            target: Any = backbone
            while hasattr(target, "backbone"):
                target = cast(Any, target).backbone
            cast(Any, target).head = nn.Identity()

        def initialize_parameters(self, dummy_input: "torch.Tensor | None" = None) -> None:
            """Materialize LazyLinear parameters eagerly for PyTorch DDP and ONNX export."""
            if dummy_input is None:
                seq = int(getattr(self.backbone, "seq_len", 120) or 120)
                feat = int(getattr(self.backbone, "input_size", 584) or 584)
                p = next(self.parameters(), None)
                dev = p.device if p is not None else torch.device("cpu")
                dummy_input = torch.zeros(2, seq, feat, device=dev)
            with torch.no_grad():
                _ = self.forward(dummy_input)

        def forward(self, x: torch.Tensor, *args, **kwargs):
            h = self.backbone(x, *args, **kwargs)  # (B, head_in) - features from backbone
            h = self.proj(h)
            return self.mt_head(h)

    class MultiPairMultiTaskHead(nn.Module):
        """
        Independent multi-task prediction heads for each currency pair.

        Contains a dedicated MultiTaskHead for each pair in ``pairs``. When
        evaluated on a shared backbone representation h: (B, in_features), each
        pair head independently computes:
          - direction logits
          - return_hat forecast
          - confidence logit
          - (optional) VaR/CVaR risk quantiles: q_low (5%), q_high (95%)

        Forward returns:
          (logits, ret_hat, conf) where each tensor is shaped (B, n_pairs).
          If quantile_enabled:
          (logits, ret_hat, conf, q_low, q_high) where each tensor is shaped (B, n_pairs).
          If return_aux:
          appends (recon, vol_hat) across pairs.
        """

        def __init__(
            self,
            in_features: int,
            pairs: list[str] | int = 4,
            hidden: int = 64,
            dropout: float = 0.1,
            quantile_enabled: bool = True,
            return_aux: bool = False,
            recon_out_features: int | None = None,
            quantiles: tuple[float, float] = (0.05, 0.95),
            **kwargs,
        ):
            super().__init__()
            self.in_features = int(in_features)
            if isinstance(pairs, (list, tuple)):
                self.pair_names = [str(p) for p in pairs]
            else:
                self.pair_names = [f"pair_{i}" for i in range(int(pairs))]
            self.n_pairs = len(self.pair_names)
            self.quantile_enabled = bool(quantile_enabled)
            self.return_aux = bool(return_aux)
            self.quantiles = tuple(quantiles)

            self.heads = nn.ModuleDict({
                p.replace("/", "_"): MultiTaskHead(
                    in_features=in_features,
                    hidden=hidden,
                    dropout=dropout,
                    return_aux=return_aux,
                    recon_out_features=recon_out_features,
                    quantile_enabled=quantile_enabled,
                    quantiles=quantiles,
                )
                for p in self.pair_names
            })

        def __len__(self) -> int:
            return self.n_pairs

        def __getitem__(self, idx: int | str) -> MultiTaskHead:
            if isinstance(idx, int):
                return list(self.heads.values())[idx]
            return self.heads[str(idx).replace("/", "_")]

        def forward(self, h: torch.Tensor):
            """h: (B, in_features) - shared backbone representation."""
            head_outs = [head(h) for head in self.heads.values()]

            logits = torch.stack([out[0] for out in head_outs], dim=1)
            ret_hat = torch.stack([out[1] for out in head_outs], dim=1)
            conf = torch.stack([out[2] for out in head_outs], dim=1)
            outs = (logits, ret_hat, conf)

            idx = 3
            if self.quantile_enabled:
                q_low = torch.stack([out[idx] for out in head_outs], dim=1)
                q_high = torch.stack([out[idx + 1] for out in head_outs], dim=1)
                outs = (*outs, q_low, q_high)
                idx += 2

            if self.return_aux:
                recon = torch.stack([out[idx] for out in head_outs], dim=1)
                vol_hat = torch.stack([out[idx + 1] for out in head_outs], dim=1)
                outs = (*outs, recon, vol_hat)

            return outs

    class MultiPairMultiTaskLoss(nn.Module):
        """
        Computes multi-task loss across all P pairs:
          L = (1 / P) * sum_{p=0}^{P-1} MultiTaskLoss(logits[:, p], ret_hat[:, p], conf[:, p], y_cls[:, p], y_cont[:, p])

        Supports optional per-pair weighting.
        """

        def __init__(
            self,
            w_dir: float = 1.0,
            w_ret: float = 0.5,
            w_conf: float = 0.3,
            huber_delta: float = 1.0,
            recon_w: float = 0.1,
            vol_w: float = 0.05,
            w_quantile: float = 0.2,
            quantiles: tuple[float, float] = (0.05, 0.95),
            pair_weights: list[float] | torch.Tensor | None = None,
            **kwargs,
        ):
            super().__init__()
            self.single_loss = MultiTaskLoss(
                w_dir=w_dir,
                w_ret=w_ret,
                w_conf=w_conf,
                huber_delta=huber_delta,
                recon_w=recon_w,
                vol_w=vol_w,
                w_quantile=w_quantile,
                quantiles=quantiles,
                **kwargs,
            )
            self.w_dir = self.single_loss.w_dir
            self.w_ret = self.single_loss.w_ret
            self.w_conf = self.single_loss.w_conf
            self.w_quantile = self.single_loss.w_quantile
            self.quantiles = self.single_loss.quantiles
            self.hub = self.single_loss.hub
            self.bce = self.single_loss.bce

            if pair_weights is not None:
                pw = torch.as_tensor(pair_weights, dtype=torch.float32)
                pw_sum = pw.sum()
                if pw_sum > 1e-8:
                    self.register_buffer("pair_weights", pw / pw_sum)
                else:
                    self.register_buffer("pair_weights", torch.full_like(pw, 1.0 / max(1, len(pw))))
            else:
                self.pair_weights = None

        def forward(
            self,
            logits: torch.Tensor,
            ret_hat: torch.Tensor,
            conf: torch.Tensor,
            y_cls: torch.Tensor,
            y_cont: torch.Tensor,
            y_conf: torch.Tensor | None = None,
            recon_hat: torch.Tensor | None = None,
            recon_tgt: torch.Tensor | None = None,
            vol_hat: torch.Tensor | None = None,
            vol_tgt: torch.Tensor | None = None,
            bet_size: torch.Tensor | None = None,
            q_low: torch.Tensor | None = None,
            q_high: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            if ret_hat.ndim == 1:
                ret_hat = ret_hat.unsqueeze(1)
            if conf.ndim == 1:
                conf = conf.unsqueeze(1)
            if y_cls.ndim == 1:
                y_cls = y_cls.unsqueeze(1)
            if y_cont.ndim == 1:
                y_cont = y_cont.unsqueeze(1)
            if y_conf is not None and y_conf.ndim == 1:
                y_conf = y_conf.unsqueeze(1)
            if q_low is not None and q_low.ndim == 1:
                q_low = q_low.unsqueeze(1)
            if q_high is not None and q_high.ndim == 1:
                q_high = q_high.unsqueeze(1)

            P = logits.shape[1]
            if y_cls.shape[1] == 1 and P > 1:
                y_cls = y_cls.expand(-1, P)
            if y_cont.shape[1] == 1 and P > 1:
                y_cont = y_cont.expand(-1, P)
            if y_conf is not None and y_conf.shape[1] == 1 and P > 1:
                y_conf = y_conf.expand(-1, P)

            losses = []
            for p in range(P):
                loss_p = self.single_loss(
                    logits=logits[:, p],
                    ret_hat=ret_hat[:, p],
                    conf=conf[:, p],
                    y_cls=y_cls[:, p],
                    y_cont=y_cont[:, p],
                    y_conf=y_conf[:, p] if y_conf is not None else None,
                    recon_hat=recon_hat[:, p] if recon_hat is not None and recon_hat.ndim >= 2 else None,
                    recon_tgt=recon_tgt[:, p] if recon_tgt is not None and recon_tgt.ndim >= 2 else None,
                    vol_hat=vol_hat[:, p] if vol_hat is not None and vol_hat.ndim >= 2 else None,
                    vol_tgt=vol_tgt[:, p] if vol_tgt is not None and vol_tgt.ndim >= 2 else None,
                    bet_size=bet_size,
                    q_low=q_low[:, p] if q_low is not None else None,
                    q_high=q_high[:, p] if q_high is not None else None,
                )
                losses.append(loss_p)

            loss_stack = torch.stack(losses)
            if self.pair_weights is not None:
                pw = self.pair_weights.to(loss_stack.device)
                if pw.numel() != P:
                    pw = torch.ones(P, device=loss_stack.device) / float(P)
                return (loss_stack * pw).sum()
            return loss_stack.mean()

    class MultiPairMultiTaskWrapper(nn.Module):
        """
        Wraps any backbone architecture (HAELT, MAMBA, GNN, TFT, etc.), replacing
        its .head with nn.Identity() to expose the pre-head hidden state, then routing
        that state through a MultiPairMultiTaskHead.

        After wrapping, forward(x) returns per-pair multi-task predictions:
          (direction_logits, return_hat, confidence[, q_low, q_high])
        where each tensor is shaped (B, n_pairs).
        """

        def __init__(
            self,
            backbone: "nn.Module",
            head_in: int,
            pairs: list[str] | int = 4,
            hidden: int = 64,
            dropout: float = 0.1,
            proj_threshold: int = 1024,
            proj_to: int = 256,
            force_project: bool = False,
            return_aux: bool = False,
            recon_out_features: int | None = None,
            quantile_enabled: bool = True,
            quantiles: tuple[float, float] = (0.05, 0.95),
            **kwargs,
        ):
            super().__init__()
            self.backbone = backbone
            self.pairs = pairs

            if force_project:
                # Dynamic pre-head width adaptation: LazyLinear binds to actual tensor width
                # on first forward pass, accommodating dynamic penultimate features.
                self.proj = nn.Sequential(nn.LazyLinear(proj_to), nn.GELU())
                actual_in = proj_to
            elif head_in > proj_threshold:
                self.proj = nn.Sequential(nn.Linear(head_in, proj_to), nn.GELU())
                actual_in = proj_to
            else:
                self.proj = nn.Identity()
                actual_in = head_in

            self.mt_head = MultiPairMultiTaskHead(
                actual_in,
                pairs=pairs,
                hidden=hidden,
                dropout=dropout,
                return_aux=return_aux,
                recon_out_features=recon_out_features,
                quantile_enabled=quantile_enabled,
                quantiles=quantiles,
            )

            target: Any = backbone
            while hasattr(target, "backbone"):
                target = cast(Any, target).backbone
            if hasattr(target, "head"):
                cast(Any, target).head = nn.Identity()

        def initialize_parameters(self, dummy_input: "torch.Tensor | None" = None) -> None:
            """Materialize LazyLinear parameters eagerly for PyTorch DDP and ONNX export."""
            if dummy_input is None:
                seq = int(getattr(self.backbone, "seq_len", 120) or 120)
                feat = int(getattr(self.backbone, "input_size", 584) or 584)
                p = next(self.parameters(), None)
                dev = p.device if p is not None else torch.device("cpu")
                dummy_input = torch.zeros(2, seq, feat, device=dev)
            with torch.no_grad():
                _ = self.forward(dummy_input)

        def forward(self, x: torch.Tensor, *args, **kwargs):
            h = self.backbone(x, *args, **kwargs)
            h = self.proj(h)
            return self.mt_head(h)

    class MultiPairWrapper(nn.Module):
        """
        Structured multi-pair wrapper with cross-pair interaction.

        Improvements over naive concatenation:
          A. Cross-pair interaction channels:
             - Rolling Pearson correlation at two timescales (short + long window)
               for all i<j pairs - explicit co-movement signal the model need not infer.
             - Windowed relative momentum r_i(W) âˆ’ r_j(W) over momentum_window bars.
             - Volatility share per pair: ATR_i / basket_ATR.
             - Cross-pair return dispersion (std across pairs at each bar).
          B. Regime-conditioned pair weighting - a small attention network weights
             pairs based on the last-bar regime context (vol, OFI-Z, session).
          C. Per-pair LayerNorm - normalises each pair's features independently
             before concatenation, preventing high-vol pairs from dominating.
          D. Alignment confidence gate - tracks how many pairs are present (non-zero)
             and feeds this as an extra scalar feature; missing data = regime signal.

        Input shape:  (B, T, n_pairs * f_per_pair)  - pairs concatenated on feat axis.
        Output: backbone prediction with enriched cross-pair features.

        New feature count fed to backbone:
            n_pairs * (f_per_pair + embed_dim)      [pair features + learned embedding]
          + n_pairs * (n_pairs - 1) // 2            [pairwise windowed relative momentum]
          + n_pairs * (n_pairs - 1) // 2            [pairwise short rolling correlation]
          + n_pairs * (n_pairs - 1) // 2            [pairwise long rolling correlation]
          + n_pairs                                  [vol share per pair]
          + 1                                        [cross-pair dispersion]
          + 1                                        [alignment confidence]
        """  # noqa: RUF002

        def __init__(
            self,
            backbone: "nn.Module",
            n_pairs: int,
            f_per_pair: int,
            embed_dim: int = 16,
            regime_ctx_dim: int = 8,  # regime context features from last bar
            corr_window: int = 20,  # short rolling correlation window (bars)
            corr_window_long: int = 60,  # long rolling correlation window (bars)
            momentum_window: int = 20,  # windowed relative momentum lookback
            return_idx: int = 0,  # per-pair return proxy column (fallback 0)
            atr_idx: int = 1,  # per-pair ATR proxy column (fallback 1)
            corr_dropout_p: float = 0.3,
        ):
            super().__init__()
            self.backbone = backbone
            self.n_pairs = n_pairs
            self.f_per_pair = f_per_pair
            self.embed_dim = embed_dim
            self.corr_window = corr_window
            self.corr_window_long = corr_window_long
            self.momentum_window = momentum_window
            self.return_idx = int(return_idx)
            self.atr_idx = int(atr_idx)
            self.corr_dropout_p = corr_dropout_p

            # C. Per-pair independent normalisation
            self.pair_norms = nn.ModuleList([nn.LayerNorm(f_per_pair) for _ in range(n_pairs)])

            # A. Learned pair embedding (static identity)
            self.pair_embeds = nn.Embedding(n_pairs, embed_dim)

            # B. Regime-conditioned pair attention weights
            # Input: last-bar concatenated pair features + interaction features -> softmax
            n_cross = n_pairs * (n_pairs - 1) // 2
            # RelMom + ShortCorr + LongCorr + VolShare + Dispersion + Confidence
            n_interaction = 3 * n_cross + n_pairs + 1 + 1
            self.regime_attn = nn.Sequential(
                nn.Linear(f_per_pair + n_interaction, regime_ctx_dim),
                nn.GELU(),
                nn.Linear(regime_ctx_dim, n_pairs),
            )  # outputs (B, n_pairs) unnormalised weights -> softmax

        @property
        def head(self) -> "nn.Module":
            return self.backbone.head

        def __setattr__(self, name: str, value: "Any") -> None:
            if name == "head":
                self.backbone.head = value
            else:
                super().__setattr__(name, value)

        @staticmethod
        def _rolling_corr(x: "torch.Tensor", y: "torch.Tensor", window: int) -> "torch.Tensor":
            """
            Pearson correlation of x and y computed over a rolling window.

            x, y : (B, T) - one time series each
            Returns (B, T) - clipped to [-1, 1]; first (window-1) bars are 0.

            Implemented as differentiable 1-D convolutions so it runs on-GPU
            and participates in autograd for regime_attn gradients.
            """
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y.float(), nan=0.0, posinf=0.0, neginf=0.0)
            _B, _T = x.shape
            kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / window
            pad = window - 1

            def _rmean(t: "torch.Tensor") -> "torch.Tensor":
                return F.conv1d(
                    F.pad(t.unsqueeze(1), (pad, 0)),
                    kernel,
                ).squeeze(1)  # (B, T)

            mx = _rmean(x)
            my = _rmean(y)
            mxy = _rmean(x * y)
            mx2 = _rmean(x * x)
            my2 = _rmean(y * y)

            cov = mxy - mx * my
            var_x = (mx2 - mx * mx).clamp(min=1e-8)
            var_y = (my2 - my * my).clamp(min=1e-8)
            corr = cov / (var_x.sqrt() * var_y.sqrt())
            return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

        def _cross_pair_features(self, xp: "torch.Tensor") -> "torch.Tensor":
            """
            Compute cross-pair interaction features from (B, T, P, F).

            Returns (B, T, n_interaction) containing:
              - RelMom_{i,j}     : windowed r_i(W) âˆ’ r_j(W) for all i<j pairs
              - ShortCorr_{i,j}  : rolling Pearson corr (corr_window) for all i<j
              - LongCorr_{i,j}   : rolling Pearson corr (corr_window_long) for all i<j
              - VolShare_i       : ATR_i / sum(ATR_j) for each pair
              - Dispersion_t     : StdDev of cross-pair returns at each bar
              - AlignConf_t      : fraction of pairs with non-zero features
            """  # noqa: RUF002
            xp = torch.nan_to_num(xp.float(), nan=0.0, posinf=0.0, neginf=0.0)
            B, T, P, Fp = xp.shape  # Fp = features per pair (avoid shadowing F=functional)

            ri, ai = self.return_idx, self.atr_idx
            ri = min(max(ri, 0), Fp - 1)
            ai = min(max(ai, 0), Fp - 1)
            close_feat = xp[..., ri]  # (B, T, P) return proxy
            atr_feat = xp[..., ai].abs() + 1e-8

            # D. Alignment confidence
            has_data = (xp.abs().sum(-1) > 1e-6).float()  # (B, T, P)
            conf = has_data.mean(-1, keepdim=True)  # (B, T, 1)

            # A. Windowed relative momentum - rolling mean return difference
            W = self.momentum_window
            kernel_m = torch.ones(1, 1, W, device=xp.device, dtype=xp.dtype) / W
            pad_m = W - 1
            # rolling mean return per pair: (B, T, P)
            cf_t = close_feat.permute(0, 2, 1)  # (B, P, T)
            cf_padded = F.pad(cf_t.reshape(B * P, 1, T), (pad_m, 0))
            mom_all = F.conv1d(cf_padded, kernel_m).reshape(B, P, T).permute(0, 2, 1)  # (B,T,P)

            rel_mom_parts = []
            for i in range(P):
                for j in range(i + 1, P):
                    rel_mom_parts.append((mom_all[..., i] - mom_all[..., j]).unsqueeze(-1))
            rel_mom = (
                torch.cat(rel_mom_parts, dim=-1) if rel_mom_parts else torch.zeros(B, T, 0, device=xp.device)
            )  # (B, T, n_cross)

            # A. Rolling Pearson correlation - short and long windows
            short_corr_parts = []
            long_corr_parts = []
            for i in range(P):
                for j in range(i + 1, P):
                    xi = close_feat[..., i]  # (B, T)
                    xj = close_feat[..., j]
                    sc = self._rolling_corr(xi, xj, self.corr_window).unsqueeze(-1)
                    lc = self._rolling_corr(xi, xj, self.corr_window_long).unsqueeze(-1)

                    if self.training and self.corr_dropout_p > 0.0:
                        xi_c = xi - xi.mean(dim=-1, keepdim=True)
                        xj_c = xj - xj.mean(dim=-1, keepdim=True)
                        var_i = (xi_c ** 2).sum(dim=-1).clamp(min=1e-8)
                        var_j = (xj_c ** 2).sum(dim=-1).clamp(min=1e-8)
                        pair_corr = (xi_c * xj_c).sum(dim=-1) / torch.sqrt(var_i * var_j)

                        drop = (pair_corr.abs() > 0.90) & (torch.rand_like(pair_corr) < self.corr_dropout_p)
                        drop = drop.view(-1, 1, 1).expand_as(sc)

                        sc = sc.masked_fill(drop, 0.0)
                        lc = lc.masked_fill(drop, 0.0)

                    short_corr_parts.append(sc)
                    long_corr_parts.append(lc)
            if short_corr_parts:
                short_corr = torch.cat(short_corr_parts, dim=-1)  # (B, T, n_cross)
                long_corr = torch.cat(long_corr_parts, dim=-1)
            else:
                short_corr = torch.zeros(B, T, 0, device=xp.device)
                long_corr = torch.zeros(B, T, 0, device=xp.device)

            # A. Volatility share per pair: ATR_i / basket_ATR
            atr_sum = atr_feat.sum(-1, keepdim=True) + 1e-8
            vol_share = atr_feat / atr_sum  # (B, T, P)

            # A. Cross-pair return dispersion
            dispersion = (close_feat.var(dim=-1, keepdim=True, unbiased=False) + 1e-8).sqrt()  # (B, T, 1)

            cross = torch.cat(
                [rel_mom, short_corr, long_corr, vol_share, dispersion, conf], dim=-1
            )  # (B, T, 3*n_cross + P + 2)
            return torch.nan_to_num(cross, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)

        def forward(self, x: "torch.Tensor", *args, **kwargs) -> "torch.Tensor":
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)
            B, T, _ = x.shape

            # Split into (B, T, P, F) and apply per-pair LayerNorm
            xp = x.reshape(B, T, self.n_pairs, self.f_per_pair)
            normed = torch.stack(
                [self.pair_norms[i](xp[..., i, :]) for i in range(self.n_pairs)],
                dim=2,
            )  # (B, T, P, F)

            # Compute cross-pair interaction features
            cross = self._cross_pair_features(normed)  # (B, T, n_interaction)

            # B. Regime-conditioned pair weights from last bar
            last_bar_feat = normed[:, -1, 0, :]  # (B, F) - primary pair, last bar
            last_cross = cross[:, -1, :]  # (B, n_interaction)
            ctx = torch.nan_to_num(
                torch.cat([last_bar_feat, last_cross], dim=-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )  # (B, F + n_inter)
            attn_logits = torch.nan_to_num(self.regime_attn(ctx), nan=0.0, posinf=0.0, neginf=0.0)
            pair_weights = torch.softmax(attn_logits.float(), dim=-1).to(normed.dtype)  # (B, P)
            pair_weights = torch.nan_to_num(pair_weights, nan=1.0 / self.n_pairs, posinf=0.0, neginf=0.0)

            # Apply pair weights: scale each pair's features by learned attention
            pw = pair_weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, P, 1)
            normed = normed * pw  # (B, T, P, F)

            # Append static pair embeddings
            ids = torch.arange(self.n_pairs, device=x.device)
            emb = self.pair_embeds(ids).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
            normed = torch.cat([normed, emb], dim=-1)  # (B, T, P, F+E)

            # Flatten pairs + append cross-pair interaction
            flat = normed.reshape(B, T, self.n_pairs * (self.f_per_pair + self.embed_dim))
            full = torch.nan_to_num(
                torch.cat([flat, cross], dim=-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(-10.0, 10.0)  # (B, T, pairs_flat + n_inter)

            return self.backbone(full, *args, **kwargs)

    # ── 1. Temporal Fusion Transformer (simplified) ──────────────────────────────────────────

    class VariableSelectionNetwork(nn.Module):
        """Learns which features matter at each timestep using Gated Feature Selection."""

        def __init__(self, input_size, hidden, dropout=0.1):
            super().__init__()
            self.norm = nn.LayerNorm(input_size)
            self.grn = nn.Sequential(
                nn.Linear(input_size, hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, input_size),
            )
            self.gate = nn.Sigmoid()

        def forward(self, x):
            x_norm = self.norm(x)
            # Sigmoid gating scaled by 2.0 so initial expected weight is 2.0 * 0.5 = 1.0,
            # avoiding the 1/F (1/584) attenuation caused by Softmax over hundreds of features.
            weights = 2.0 * self.gate(self.grn(x_norm))
            return x * weights, weights

    class TFTScalper(nn.Module):
        """
        Temporal Fusion Transformer for multi-horizon forex forecasting.
        Uses Variable Selection Networks to identify which features matter,
        LSTM for local sequential patterns, and Self-Attention for long-range.
        Pre-norm residual blocks (TM-012) for training stability.

        A4 fix (2026-08-07): added a learnable positional embedding applied
        after the LSTM. The LSTM is sequence-aware but the self-attention on
        top is permutation-equivariant; without positions, attention cannot
        distinguish timesteps. The original TFT paper uses relative position
        encodings - we use a learnable absolute positional embedding, which
        is the simplest equivalent that restores temporal ordering awareness
        for the attention layer.
        """

        def __init__(
            self,
            input_size=64,
            hidden=128,
            heads=4,
            lstm_layers=2,
            dropout=0.1,
            num_classes=1,
            use_gradient_checkpointing: bool = True,
            max_seq_len: int = 240,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.vsn = VariableSelectionNetwork(input_size, hidden, dropout)
            self.lstm = nn.LSTM(input_size, hidden, lstm_layers, batch_first=True, dropout=dropout)
            # A4: positional embedding for the post-LSTM self-attention block.
            # max_seq_len=240 covers typical daily/hourly seq lengths; the
            # forward gracefully slices or cycles if T differs.
            self.pos_emb = nn.Embedding(max_seq_len, hidden)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
            self.max_seq_len = int(max_seq_len)
            self.attn = _FlashMHA(hidden, heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(hidden)
            self.ffn = nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden)
            )
            self.norm2 = nn.LayerNorm(hidden)
            self.norm_out = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, num_classes)
            _kaiming_init_module(self)

        def _attn_block(self, h):
            return h + self.attn(self.norm1(h))

        def _ffn_block(self, h):
            return h + self.ffn(self.norm2(h))

        def _add_pos(self, lstm_out):
            """Add positional embedding to (B, T, hidden) tensor.
            Handles T == max_seq_len (exact), T < max_seq_len (forward slice),
            and T > max_seq_len (cyclic reuse - fallback for longer inputs).
            """
            T = lstm_out.size(1)
            if self.max_seq_len >= T:
                pos = self.pos_emb.weight[:T]  # (T, hidden)
            else:
                idx = torch.arange(T, device=lstm_out.device) % self.max_seq_len
                pos = self.pos_emb.weight[idx]
            return lstm_out + pos.unsqueeze(0)

        def forward(self, x):
            # x: (B, T, F) - pre-norm: x + f(norm(x))
            x_sel, _ = self.vsn(x)
            lstm_out, _ = self.lstm(x_sel)
            # A4: inject positional embedding before permutation-equivariant attention
            lstm_out = self._add_pos(lstm_out)
            h = _maybe_checkpoint(
                self._attn_block,
                lstm_out,
                enabled=self.use_gradient_checkpointing,
            )
            h = _maybe_checkpoint(
                self._ffn_block,
                h,
                enabled=self.use_gradient_checkpointing,
            )
            out = self.head(self.norm_out(h[:, -1, :]))
            if isinstance(self.head, nn.Identity):
                return out
            return out.squeeze(-1) if self.num_classes == 1 else out

    # â”€â”€ 2. iTransformer (variate-dimension attention) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    class iTransformerScalper(nn.Module):
        """
        iTransformer: applies attention across the feature (variate) dimension.
        Treats EUR/USD price and US 10Y yield as different 'tokens',
        learning their interactions as a differentiable map.
        Outperforms standard time-dimension Transformers on multivariate series.
        """

        def __init__(
            self,
            input_size=64,
            seq_len=60,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_ff=256,
            dropout=0.1,
            num_classes=1,
            use_gradient_checkpointing: bool = True,
            dim_feedforward: int | None = None,
            use_pos_encoding: bool = False,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.seq_len = seq_len
            self.d_model = d_model
            self.hidden_size = d_model
            self.input_size = input_size
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.use_pos_encoding = bool(use_pos_encoding)

            ff_dim = dim_feedforward if dim_feedforward is not None else dim_ff
            # Project each variate's time-series into d_model token
            self.variate_proj = nn.Linear(seq_len, d_model)
            if self.use_pos_encoding:
                self.pos_emb = nn.Embedding(max(input_size, 1024), d_model)
                nn.init.normal_(self.pos_emb.weight, std=0.02)
            else:
                self.pos_emb = None
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)
            self.norm_out = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model * input_size, num_classes)
            _kaiming_init_module(self)

        def forward(self, x, mask=None, src_key_padding_mask=None):
            # x: (B, T, F)  ->  treat F as sequence, T as embedding
            B, T, _n_feat = x.shape
            tokens = x.permute(0, 2, 1)  # (B, F, T)
            # Curriculum may slice T below build-time seq_len; resample to match variate_proj.
            if self.seq_len != T:
                tokens = F.interpolate(
                    tokens,
                    size=self.seq_len,
                    mode="linear",
                    align_corners=False,
                )
            tokens = self.variate_proj(tokens)  # (B, F, d_model)
            if self.pos_emb is not None:
                F_count = tokens.size(1)
                pos = self.pos_emb(torch.arange(F_count, device=tokens.device) % self.pos_emb.num_embeddings)
                tokens = tokens + pos.unsqueeze(0)
            def _enc_step(tok):
                return self.encoder(tok, mask=mask, src_key_padding_mask=src_key_padding_mask)
            out = _maybe_checkpoint(
                _enc_step,
                tokens,
                enabled=self.use_gradient_checkpointing,
            )
            out = self.norm_out(out)
            # When head is Identity (MultiTaskWrapper), mean-pool variates to
            # (B, d_model) instead of materializing (B, F*d_model).
            if isinstance(self.head, nn.Identity):
                return out.mean(dim=1)  # normalize, matching all other archs
            out = out.reshape(B, -1)  # (B, F*d_model)
            o = self.head(out)
            return o.squeeze(-1) if self.num_classes == 1 else o

    # â”€â”€ 3. HAELT Hybrid (LSTM + Transformer in parallel) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    class HAELTHybrid(nn.Module):
        """
        Hybrid Attentive Ensemble Learning Transformer.
        LSTM branch captures local microstructure; Transformer captures
        long-range cross-asset correlations. Both run in parallel and are
        fused with a learned attention gate.

        A4 fix (2026-08-07): the Transformer branch is permutation-equivariant
        over time - without positional information it cannot distinguish
        bar 0 from bar 59, undermining the "long-range cross-asset correlations"
        docstring claim. We add a learnable ``nn.Embedding(seq_len, d_model)``
        positional embedding injected after the input projection.
        """

        def __init__(
            self,
            input_size=64,
            seq_len=60,
            lstm_hidden=64,
            d_model=64,
            nhead=4,
            n_layers=2,
            dropout=0.1,
            num_classes=1,
            use_gradient_checkpointing: bool = True,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.seq_len = int(seq_len)
            self.lstm = nn.LSTM(input_size, lstm_hidden, 2, batch_first=True, dropout=dropout)
            self.proj = nn.Linear(input_size, d_model)
            # A4: learnable positional embedding for the Transformer branch.
            # The LSTM is sequence-aware and does not need positions; the
            # attention is permutation-equivariant and DOES.
            self.pos_emb = nn.Embedding(self.seq_len, d_model)
            # Init positional embedding with small values so the model starts
            # near identity (no position signal) and learns to use it.
            nn.init.normal_(self.pos_emb.weight, std=0.02)
            enc = nn.TransformerEncoderLayer(
                d_model, nhead, d_model * 4, dropout=dropout, batch_first=True, norm_first=True
            )
            self.trf = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
            self.attn_pool_lstm = nn.Linear(lstm_hidden, 1)
            self.attn_pool_trf = nn.Linear(d_model, 1)
            fused = lstm_hidden + d_model
            self.gate = nn.Sequential(nn.Linear(fused, fused), nn.Sigmoid())
            self.norm_out = nn.LayerNorm(fused)
            self.head = nn.Sequential(nn.Linear(fused, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, num_classes))
            self._init_weights()

        def _init_weights(self):
            _kaiming_init_module(self)

        def forward(self, x):
            # PIPE-006: soft clipping via tanh scaling instead of hard clamp at Â±10
            # Preserves signal magnitude for high-impact news events while bounding values
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.where(x.abs() > 10.0, 10.0 * torch.tanh(x / 10.0), x)
            # LSTM branch (sequence-aware - no position needed)
            lout, _ = self.lstm(x)
            attn_w_l = torch.softmax(self.attn_pool_lstm(lout), dim=1)
            lf = (lout * attn_w_l).sum(dim=1)
            # Transformer branch with A4 positional encoding
            h = self.proj(x)
            # Inject positional embedding: shape (T, d_model) broadcast over (B, T, d_model)
            T = h.size(1)
            if self.seq_len == T:
                pos = self.pos_emb.weight  # (seq_len, d_model)
                h = h + pos.unsqueeze(0)
            elif self.seq_len >= T:
                # Forward slicing if input is shorter than the training seq_len
                pos = self.pos_emb.weight[:T]
                h = h + pos.unsqueeze(0)
            else:
                # Longer input than training seq_len - pad by reusing positions cyclically.
                # This is a fallback; the canonical use-case has T == seq_len at training time.
                idx = torch.arange(T, device=h.device) % self.seq_len
                pos = self.pos_emb.weight[idx]  # (T, d_model)
                h = h + pos.unsqueeze(0)
            tout = _maybe_checkpoint(
                self.trf,
                h,
                enabled=self.use_gradient_checkpointing,
            )
            attn_w_t = torch.softmax(self.attn_pool_trf(tout), dim=1)
            tf = (tout * attn_w_t).sum(dim=1)
            c = torch.cat([lf, tf], dim=-1)
            c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            g = self.gate(c)
            o = self.head(self.norm_out(c + g * c))  # residual through gate (TM-016)
            if isinstance(self.head, nn.Identity):
                return o
            return o.squeeze(-1) if self.num_classes == 1 else o

    # â”€â”€ 4. Mamba State Space Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    class MambaBlock(nn.Module):
        """
        Causal conv + SiLU + softplus-dt gated residual (Mamba-*inspired*).

        This is **not** a selective SSM (no discretized A/B/C state, no scan).
        ``dt_proj`` only scales the conv features. Prefer this for speed; do not
        expect strict load of older checkpoints that had ``A_log``/``d_state``.
        Architecture tag: ``mamba_gated_v2`` (see ``MambaScalper.arch_tag``).
        """

        def __init__(self, d_model=128, d_conv=4, expand=2, dropout=0.1):
            super().__init__()
            d_inner = d_model * expand
            self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
            # A8/A9 fix (2026-08-07): use asymmetric LEFT-ONLY padding so the
            # 1D conv is genuinely causal. The previous symmetric
            # `padding=d_conv-1` + `[:, :, :T]` truncation leaked `d_conv-1`
            # future bars into every output position (clearest in MambaScalper).
            # Note: Conv1d only accepts symmetric padding; we manually pad in forward()
            self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=0, groups=d_inner, bias=True)
            self.conv1d_pad = d_conv - 1
            self.act = nn.SiLU()
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
            self.norm = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)
            self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        def forward(self, x):
            # x: (B, T, d_model) - pre-norm residual (TM-012)
            _B, _T, _D = x.shape
            res = x
            x_n = self.norm(x)
            xz = self.in_proj(x_n)  # (B, T, d_inner*2)
            x2, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)
            # 1D conv along time (causal - asymmetric LEFT pad, no future leakage)
            # A8/A9 fix: manually pad left with zeros, then conv with padding=0
            # This avoids the 2-tuple padding issue in Conv1d
            x2_perm = x2.permute(0, 2, 1).contiguous()  # (B, d_inner, T)
            x2_pad = F.pad(x2_perm, (self.conv1d_pad, 0))  # (B, d_inner, T + pad)
            x2c = self.conv1d(x2_pad)  # (B, d_inner, T)
            x2c = x2c.permute(0, 2, 1).contiguous()
            x2c = self.act(x2c)
            # Softplus-dt feature gate (not an SSM recurrence)
            y = x2c * F.softplus(self.dt_proj(x2c))
            y = y * torch.sigmoid(z)  # gating
            out = self.out_proj(y)
            return res + self.drop(out)

    class MambaScalper(nn.Module):
        """
        Stack of MambaBlocks for low-latency HFT inference.
        Handles long sequences with O(L) cost vs transformer O(LÂ²).
        Not a true Mamba SSM - see MambaBlock docstring.
        """

        arch_tag = "mamba_gated_v2"

        def __init__(
            self,
            input_size=64,
            d_model=128,
            d_conv=4,
            expand=2,
            num_layers=4,
            dropout=0.1,
            num_classes=1,
            use_gradient_checkpointing: bool = True,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.embed = nn.Linear(input_size, d_model)
            self.layers = nn.ModuleList([MambaBlock(d_model, d_conv, expand, dropout) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)
            _kaiming_init_module(self)

        def forward(self, x):
            h = self.embed(x)
            for layer in self.layers:
                h = _maybe_checkpoint(layer, h, enabled=self.use_gradient_checkpointing)
            o = self.head(self.norm(h[:, -1, :]))
            if isinstance(self.head, nn.Identity):
                return o
            return o.squeeze(-1) if self.num_classes == 1 else o

    # â”€â”€ 5. GNN Cross-Asset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    class GNNCrossAsset(nn.Module):
        """
        Graph Neural Network for cross-asset modelling.

        Treats features (per node) as graph nodes. Message passing uses a
        learned dense adjacency (sigmoid(adj_logits)), not a fixed correlation
        graph.

        This is a GAT-style implementation without torch_geometric.
        """

        def __init__(self, node_features=32, hidden=64, num_layers=3, heads=4, n_nodes=6, dropout=0.1, num_classes=1):
            super().__init__()
            self.n_nodes = n_nodes
            self.num_classes = num_classes
            self.node_embed = nn.Linear(node_features, hidden)
            # Input-dependent, DIRECTED adjacency: separate query/key projections so
            # edge weight A_ij can differ from A_ji (a single shared projection would
            # force A_ij == A_ji, collapsing the graph to an undirected similarity).
            self.adj_q = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, n_nodes),
            )
            self.adj_k = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, n_nodes),
            )
            self.attn_layers = nn.ModuleList([_FlashMHA(hidden, heads, dropout=dropout) for _ in range(num_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
            self.norm_out = nn.LayerNorm(hidden * n_nodes)
            self.head = nn.Linear(hidden * n_nodes, num_classes)
            self.drop = nn.Dropout(dropout)
            _kaiming_init_module(self)

        def forward(self, x, adj=None):
            """
            x  : (B, n_nodes, node_features) - one feature vector per node per bar
            adj: optional precomputed (B, n_nodes, n_nodes) edge weights. When None,
                 adjacency is computed from node embeddings via attention (adapts
                 across regimes). Pre-norm message passing (TM-012).
            """
            h = self.node_embed(x)  # (B, N, hidden)
            if adj is not None:
                A = adj
            else:
                # Attention-style dot-product adjacency: A_ij = softmax_j(q_iÂ·k_j)
                # (separate q/k projections => directed edges A_ij != A_ji).
                q = self.adj_q(h)  # (B, N, N)
                k = self.adj_k(h)
                A = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (self.n_nodes**0.5), dim=-1)
            for attn, norm in zip(self.attn_layers, self.norms, strict=False):
                h_n = norm(h)
                h_mix = torch.einsum("bnm,bmh->bnh", A, h_n)
                out = attn(h_mix)
                h = h + self.drop(out)
            o = self.head(self.norm_out(h.reshape(h.shape[0], -1)))
            # MultiTaskWrapper sets head to Identity to expose (B, D). Never squeeze in that
            # case: num_classes==1 + squeeze(-1) would turn (B, 1) into (B,) and breaks BYOL.
            if isinstance(self.head, nn.Identity):
                return o
            return o.squeeze(-1) if self.num_classes == 1 else o

    class GNNFromSequence(nn.Module):
        """
        Adapts (B, T, F) sequence batches to GNNCrossAsset (B, n_nodes, node_features).
        Time axis is mean-pooled; features are projected into n_nodes x chunk tokens.
        """

        def __init__(self, input_size=64, hidden=64, num_layers=3, dropout=0.1, n_nodes=6, num_classes=1, nhead=4):
            super().__init__()
            chunk = max(8, (input_size + n_nodes - 1) // n_nodes)
            self.n_nodes = n_nodes
            self.chunk = chunk
            self.proj = nn.Linear(input_size, n_nodes * chunk)
            self.temporal_attn = nn.Linear(input_size, 1)
            self.gnn = GNNCrossAsset(
                node_features=chunk,
                hidden=hidden,
                num_layers=num_layers,
                heads=nhead,
                n_nodes=n_nodes,
                dropout=dropout,
                num_classes=num_classes,
            )

        @property
        def head(self) -> "nn.Module":
            """Proxy to inner GNN head - enables MultiTaskWrapper compatibility."""
            return self.gnn.head

        def __setattr__(self, name: str, value: "Any") -> None:
            if name == "head":
                self.gnn.head = value
            else:
                super().__setattr__(name, value)

        def forward(self, x, adj=None):
            attn_w = torch.softmax(self.temporal_attn(x), dim=1)
            z = (x * attn_w).sum(dim=1)
            h = self.proj(z).view(-1, self.n_nodes, self.chunk)
            return self.gnn(h, adj=adj)

    # â”€â”€ 6. EXPERT Encoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    class ConvFFN(nn.Module):
        """1D conv feedforward - captures local temporal patterns better than MLP."""

        def __init__(self, d_model, d_ff, kernel=3, dropout=0.1):
            super().__init__()
            # A8/A9 fix (2026-08-07): asymmetric LEFT-ONLY padding so both
            # conv1 and conv2 are genuinely causal. The previous symmetric
            # `padding=kernel-1` + `h[:, :, :T]` truncation leaked up to
            # `2*(kernel-1)` future bars (since two stacked convs compound).
            # Note: Conv1d only accepts symmetric padding; we manually pad in forward()
            self.conv1 = nn.Conv1d(d_model, d_ff, kernel, padding=0)
            self.conv2 = nn.Conv1d(d_ff, d_model, kernel, padding=0)
            self.conv1_pad = kernel - 1
            self.conv2_pad = kernel - 1
            self.norm = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)
            self.act = nn.GELU()

        def forward(self, x):
            # x: (B, T, D) - pre-norm residual (TM-012)
            x_n = self.norm(x)
            h = x_n.permute(0, 2, 1).contiguous()  # (B, D, T)
            # Manual causal padding for conv1
            h = F.pad(h, (self.conv1_pad, 0))
            h = self.act(self.conv1(h))  # (B, d_ff, T)
            # Manual causal padding for conv2
            h = F.pad(h, (self.conv2_pad, 0))
            h = self.drop(self.conv2(h))  # (B, d_model, T)
            # A8/A9 fix: with manual asymmetric left padding + padding=0, output length
            # equals input length T exactly; no post-hoc slice needed.
            return x + h.permute(0, 2, 1).contiguous()

    class EXPERTEncoder(nn.Module):
        """
        EXPERT: EXchange-Rate Prediction using Encoder Representation from Transformers.
        Key differences from standard Transformer:
          - A4 fix (2026-08-07): positional encoding via learnable
            ``nn.Embedding(max_seq_len, d_model)``. The previous "order is
            inherent in time series" docstring was wrong - attention is
            permutation-equivariant and cannot tell bar 0 from bar 79 without
            position info. Especially with ConvFFN (causal-only) providing
            local temporal structure, the attention layer still needs absolute
            position info to distinguish distant timesteps.
          - 1D convolutional feedforward layers (local temporal patterns)
          - Encoder-only (no decoder needed for regression)
        Focused architecture makes it more data-efficient than general Transformers.
        """

        def __init__(
            self,
            input_size=64,
            d_model=128,
            nhead=8,
            num_layers=4,
            dropout=0.1,
            num_classes=1,
            use_gradient_checkpointing: bool = True,
            max_seq_len: int = 240,
            use_conv_ffn: bool = True,
            no_pos_encoding: bool = False,
            seq_len: int = 120,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.input_size = input_size
            self.seq_len = seq_len
            self.d_model = d_model
            self.hidden_size = d_model
            self.use_conv_ffn = bool(use_conv_ffn)
            self.no_pos_encoding = bool(no_pos_encoding)
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.proj = nn.Linear(input_size, d_model)
            # A4: learnable positional embedding (replaces old "no positional encoding")
            self.max_seq_len = int(max(max_seq_len, seq_len, 240))
            if not self.no_pos_encoding:
                self.pos_emb = nn.Embedding(self.max_seq_len, d_model)
                nn.init.normal_(self.pos_emb.weight, std=0.02)
            else:
                self.pos_emb = None
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "attn": _FlashMHA(d_model, nhead, dropout=dropout),
                            "norm1": nn.LayerNorm(d_model),
                            "ffn": (
                                ConvFFN(d_model, d_model * 4, dropout=dropout)
                                if self.use_conv_ffn
                                else nn.Sequential(
                                    nn.LayerNorm(d_model),
                                    nn.Linear(d_model, d_model * 4),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_model * 4, d_model),
                                    nn.Dropout(dropout),
                                )
                            ),
                        }
                    )
                    for _ in range(num_layers)
                ]
            )
            self.pool = None  # replaced by last-timestep pooling for streaming
            self.norm_out = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)
            _kaiming_init_module(self)

        def _layer_forward(self, layer, h):
            # Pre-norm attention + FFN
            h = h + layer["attn"](layer["norm1"](h))
            if self.use_conv_ffn:
                return layer["ffn"](h)
            return h + layer["ffn"](h)

        def _add_pos(self, h):
            """Add positional embedding to (B, T, d_model)."""
            if self.no_pos_encoding or self.pos_emb is None:
                return h
            T = h.size(1)
            if self.max_seq_len >= T:
                pos = self.pos_emb.weight[:T]
            else:
                idx = torch.arange(T, device=h.device) % self.max_seq_len
                pos = self.pos_emb.weight[idx]
            return h + pos.unsqueeze(0)

        def forward(self, x):
            h = self.proj(x)
            h = self._add_pos(h)  # A4: inject positional information
            for layer in self.layers:
                # Capture layer in default-arg closure for checkpoint safety.
                def _run(t, _layer=layer):
                    return self._layer_forward(_layer, t)

                h = _maybe_checkpoint(
                    _run,
                    h,
                    enabled=self.use_gradient_checkpointing,
                )
            h = h[:, -1, :]  # last-timestep for streaming/online inference
            o = self.head(self.norm_out(h))
            if isinstance(self.head, nn.Identity):
                return o
            return o.squeeze(-1) if self.num_classes == 1 else o

    class PatchTSTScalper(nn.Module):
        """
        PatchTST: Patch Time Series Transformer.
        Groups adjacent timesteps into patches to capture local semantic information 
        and drastically reduce the effective sequence length for the Transformer.
        Employs Channel Independence (treats features as independent sequences).
        """
        def __init__(
            self,
            input_size: int = 64,
            seq_len: int = 120,
            patch_len: int = 12,
            stride: int = 12,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 3,
            dropout: float = 0.1,
            num_classes: int = 1,
            use_gradient_checkpointing: bool = True
        ):
            super().__init__()
            self.num_classes = num_classes
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.input_size = input_size
            self.seq_len = seq_len
            self.d_model = d_model
            self.hidden_size = d_model
            
            # Patching configuration
            self.patch_len = patch_len
            self.stride = stride
            self.patch_num = int((seq_len - patch_len) / stride + 1)
            
            # If sequence cannot be cleanly patched, we will pad it in forward
            self.padding = stride - (seq_len - patch_len) % stride if (seq_len - patch_len) % stride != 0 else 0
            if self.padding != 0 and self.padding != stride:
                self.patch_num += 1
            else:
                self.padding = 0

            # Linear embedding for the patch
            self.value_embedding = nn.Linear(patch_len, d_model)
            
            # Positional embedding
            self.position_embedding = nn.Embedding(self.patch_num, d_model)
            
            # Channel-independent Transformer blocks
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "attn": _FlashMHA(d_model, nhead, dropout=dropout),
                            "norm1": nn.LayerNorm(d_model),
                            "norm2": nn.LayerNorm(d_model),
                            "ffn": nn.Sequential(
                                nn.Linear(d_model, d_model * 4),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_model * 4, d_model),
                                nn.Dropout(dropout),
                            ),
                        }
                    )
                    for _ in range(num_layers)
                ]
            )
            
            self.head_norm = nn.LayerNorm(d_model)
            
            self.flatten = nn.Flatten(start_dim=1)
            self.head = nn.Linear(self.input_size * self.patch_num * d_model, num_classes)
            
            _kaiming_init_module(self)
            
        def _layer_forward(self, layer, h):
            # h: (B * input_size, patch_num, d_model)
            h = h + layer["attn"](layer["norm1"](h))
            h = h + layer["ffn"](layer["norm2"](h))
            return h

        def forward(self, x):
            # x: (B, T, F_in)
            B, T, F_in = x.size()

            # Curriculum may slice T below build-time seq_len; resample to match self.seq_len.
            if self.seq_len != T:
                x_perm = x.permute(0, 2, 1)  # (B, F_in, T)
                x_perm = F.interpolate(
                    x_perm,
                    size=self.seq_len,
                    mode="linear",
                    align_corners=False,
                )
                x = x_perm.permute(0, 2, 1)  # (B, self.seq_len, F_in)
                T = self.seq_len

            if self.padding > 0:
                x = F.pad(x, (0, 0, self.padding, 0), mode='replicate')
                T += self.padding
            
            # Channel independence: we treat the feature dimension as batch dimension
            # (B, T, F_in) -> (B, F_in, T)
            x = x.transpose(1, 2)
            
            # Create patches: (B, F_in, T) -> (B, F_in, patch_num, patch_len)
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            patch_num = x.size(-2)
            
            # Reshape to (B * F_in, patch_num, patch_len)
            x = x.reshape(B * F_in, patch_num, self.patch_len)
            
            # Embed patches: (B * F_in, patch_num, d_model)
            h = self.value_embedding(x)
            
            # Add positional embedding
            pos = self.position_embedding(torch.arange(patch_num, device=x.device))
            h = h + pos.unsqueeze(0)
            
            for layer in self.layers:
                def _run(t, _layer=layer):
                    return self._layer_forward(_layer, t)
                h = _maybe_checkpoint(_run, h, enabled=self.use_gradient_checkpointing)
            
            h = self.head_norm(h)
            
            # Reshape back: (B, F_in, patch_num, d_model)
            h = h.reshape(B, F_in, patch_num, -1)
            
            if isinstance(self.head, nn.Identity):
                # When head is Identity (MultiTaskWrapper), pool across channels and patches
                # to (B, d_model) matching standard transformer pre-head representation
                return self.head_norm(h.mean(dim=(1, 2)))

            # Flatten to (B, F_in * patch_num * d_model)
            h_flat = self.flatten(h)
            
            o = self.head(h_flat)
            return o.squeeze(-1) if self.num_classes == 1 else o

    class TimesNetScalper(nn.Module):
        """
        TimesNet: Temporal 2D-Variation Modeling (Wu et al. 2023).
        FFT detects dominant periods -> reshapes 1D series into 2D (period × cycle)
        -> Inception 2D convolutions capture intra-period (local) + inter-period (global)
        dependencies. Multi-scale period ensembling gives data-efficient long/short
        decomposition ideal for forex regime shifts.
        """

        class _Inception(nn.Module):
            def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
                super().__init__()
                # Three parallel 2D branches (1x1, 3x3, 5x5) as in original paper
                self.conv_1 = nn.Conv2d(d_model, d_ff, kernel_size=1)
                self.conv_3 = nn.Conv2d(d_model, d_ff, kernel_size=3, padding=1)
                self.conv_5 = nn.Conv2d(d_model, d_ff, kernel_size=5, padding=2)
                self.proj = nn.Conv2d(3 * d_ff, d_model, kernel_size=1)
                self.norm = nn.LayerNorm(d_model)
                self.drop = nn.Dropout(dropout)
                self.act = nn.GELU()

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                # x: (B, D, H, W)  D=channels
                y1 = self.act(self.conv_1(x))
                y3 = self.act(self.conv_3(x))
                y5 = self.act(self.conv_5(x))
                y = torch.cat([y1, y3, y5], dim=1)
                y = self.proj(y)
                # LayerNorm over channel dim
                B, D, H, W = y.shape
                y = y.permute(0, 2, 3, 1).contiguous()
                y = self.norm(y)
                y = y.permute(0, 3, 1, 2).contiguous()
                return self.drop(y)

        class _TimesBlock(nn.Module):
            def __init__(self, d_model: int, d_ff: int, top_k: int = 3, dropout: float = 0.1):
                super().__init__()
                self.top_k = top_k
                self.inception = TimesNetScalper._Inception(d_model, d_ff, dropout)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                # x: (B, T, D)
                B, T, D = x.shape
                # FFT over time, average over batch and channel for period detection
                xf = torch.fft.rfft(x, dim=1)  # (B, T//2+1, D)
                amp = xf.abs().mean(dim=(0, 2))  # (F,)
                # Exclude DC (0-freq), pick top_k
                k = min(self.top_k, max(1, amp.shape[0] - 1))
                _, idx = torch.topk(amp[1:], k)
                freqs = idx + 1  # 1 .. T//2
                # Convert frequency to period: p = T // f, clamp to [4, T]
                periods = [max(4, min(T, T // int(f.item()) if int(f.item()) > 0 else T)) for f in freqs]
                outs = []
                for p in periods:
                    # Pad to multiple of p
                    if T % p != 0:
                        pad = p - T % p
                        x_pad = F.pad(x.permute(0, 2, 1), (0, pad)).permute(0, 2, 1)  # (B, T+pad, D)
                    else:
                        x_pad = x
                        pad = 0
                    Tp = x_pad.shape[1]
                    n_cycles = Tp // p
                    # Reshape to 2D: (B, D, n_cycles, p)
                    x_2d = x_pad.reshape(B, n_cycles, p, D).permute(0, 3, 1, 2).contiguous()
                    y_2d = self.inception(x_2d)
                    y = y_2d.permute(0, 2, 3, 1).reshape(B, Tp, D)[:, :T, :]
                    outs.append(y)
                # Amplitude-weighted ensemble (use 1/p as proxy for frequency importance)
                if len(outs) == 1:
                    return x + outs[0]
                weights = torch.tensor([1.0 / max(1, p) for p in periods], device=x.device)
                weights = weights / weights.sum()
                agg = sum(w * o for w, o in zip(weights, outs))
                return x + agg

        def __init__(
            self,
            input_size: int = 64,
            seq_len: int = 120,
            d_model: int = 64,
            d_ff: int = 128,
            top_k: int = 3,
            num_layers: int = 2,
            dropout: float = 0.1,
            num_classes: int = 1,
            use_gradient_checkpointing: bool = True,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.seq_len = seq_len
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.embed = nn.Linear(input_size, d_model)
            self.layers = nn.ModuleList([self._TimesBlock(d_model, d_ff, top_k, dropout) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)
            self.d_model = d_model
            self.hidden_size = d_model
            _kaiming_init_module(self)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, F)
            h = self.embed(x)
            for layer in self.layers:
                h = _maybe_checkpoint(layer, h, enabled=self.use_gradient_checkpointing)
            o = self.head(self.norm(h[:, -1, :]))
            if isinstance(self.head, nn.Identity):
                return o
            return o.squeeze(-1) if self.num_classes == 1 else o

    class TimeMixerScalper(nn.Module):
        """
        TimeMixer: Decomposable Multiscale Mixing (Wang et al. 2024).
        Decomposes series into trend/seasonal via moving average, then
        applies Past-Decomposable-Mixing (PDM): seasonal and trend branches
        are mixed separately across time (channel-independent) and channel
        (time-independent), then merged. Future-Multipredictor-Mixing is
        simplified to last-step pooling for trading (scalping horizon).
        Captures long-term trend + short-term seasonal deformation.
        """

        class _SeriesDecomp(nn.Module):
            def __init__(self, kernel_size: int = 25):
                super().__init__()
                self.kernel = kernel_size
                self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

            def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
                # x: (B, T, D) -> trend/seasonal
                if self.kernel > x.shape[1]:
                    trend = x.mean(dim=1, keepdim=True).expand_as(x)
                else:
                    # AvgPool1d expects (B*D, T)
                    B, T, D = x.shape
                    y = x.permute(0, 2, 1).reshape(B * D, T)  # (B*D, T) not (B,D,T) - handled via reshape
                    # Pad to keep length: replicate edges
                    pad = self.kernel // 2
                    y_pad = F.pad(y.unsqueeze(1), (pad, pad), mode="replicate").squeeze(1)
                    trend_1d = self.avg(y_pad)  # (B*D, T)
                    trend = trend_1d.reshape(B, D, T).permute(0, 2, 1)  # (B, T, D)
                seasonal = x - trend
                return seasonal, trend

        class _MixerBlock(nn.Module):
            def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
                super().__init__()
                self.time_mix = nn.Sequential(
                    nn.Linear(seq_len, seq_len),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.feat_mix = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                # x: (B, T, D)
                # Time mixing (T dimension) - residual
                y = x + self.time_mix(x.transpose(1, 2)).transpose(1, 2)
                # Feature mixing (D dimension) - residual
                y = y + self.feat_mix(y)
                return self.norm(y)

        def __init__(
            self,
            input_size: int = 64,
            seq_len: int = 120,
            d_model: int = 64,
            dropout: float = 0.1,
            decomp_kernel: int = 25,
            num_layers: int = 2,
            down_sampling_layers: int = 2,
            down_sampling_window: int = 2,
            num_classes: int = 1,
            use_gradient_checkpointing: bool = True,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.seq_len = seq_len
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.embed = nn.Linear(input_size, d_model)
            self.decomp = self._SeriesDecomp(kernel_size=decomp_kernel)
            self.seasonal_blocks = nn.ModuleList(
                [self._MixerBlock(seq_len, d_model, dropout) for _ in range(num_layers)]
            )
            self.trend_blocks = nn.ModuleList(
                [self._MixerBlock(seq_len, d_model, dropout) for _ in range(num_layers)]
            )
            # Downsampled scales for multiscale: simple AvgPool1d downsampling
            self.down_sampler = nn.AvgPool1d(kernel_size=down_sampling_window, stride=down_sampling_window)
            self.down_layers = int(down_sampling_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)
            self.d_model = d_model
            self.hidden_size = d_model
            _kaiming_init_module(self)

        def _mix_scale(self, x: "torch.Tensor", blocks: "nn.ModuleList") -> "torch.Tensor":
            h = x
            for blk in blocks:
                h = _maybe_checkpoint(blk, h, enabled=self.use_gradient_checkpointing)
            return h

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, F)
            h = self.embed(x)  # (B, T, D)
            seasonal, trend = self.decomp(h)
            # Multiscale seasonal/trend via downsampling pyramid
            # Level 0 is original, levels 1..down_layers are pooled then interpolated back
            seasonal_agg = self._mix_scale(seasonal, self.seasonal_blocks)
            trend_agg = self._mix_scale(trend, self.trend_blocks)
            # Simple multiscale augmentation: downsample and remix (if seq_len divisible)
            for _ in range(self.down_layers):
                if seasonal.shape[1] < 4:
                    break
                # Downsample time: (B, T, D) -> (B, T//2, D)
                s_down = self.down_sampler(seasonal_agg.permute(0, 2, 1)).permute(0, 2, 1)
                t_down = self.down_sampler(trend_agg.permute(0, 2, 1)).permute(0, 2, 1)
                # Interpolate back to original length for residual addition
                s_up = F.interpolate(s_down.permute(0, 2, 1), size=seasonal_agg.shape[1], mode="linear", align_corners=False).permute(0, 2, 1)
                t_up = F.interpolate(t_down.permute(0, 2, 1), size=trend_agg.shape[1], mode="linear", align_corners=False).permute(0, 2, 1)
                seasonal_agg = seasonal_agg + 0.5 * s_up
                trend_agg = trend_agg + 0.5 * t_up
            h_out = seasonal_agg + trend_agg
            o = self.head(self.norm(h_out[:, -1, :]))
            if isinstance(self.head, nn.Identity):
                return o
            return o.squeeze(-1) if self.num_classes == 1 else o

    class GLMBaseline(nn.Module):
        """Generalized Linear Model (GLM) baseline.

        Flattens the sequence and applies a single linear projection.
        Serves as an ultra-fast, lightweight baseline against complex deep learning models.
        """

        def __init__(self, input_size: int, num_classes: int = 1, seq_len: int = 16):
            super().__init__()
            self.seq_len = seq_len
            self.num_classes = num_classes
            self.d_model = input_size * seq_len
            self.input_norm = nn.LayerNorm(input_size)
            self.flatten = nn.Flatten(start_dim=1)
            # Use LazyLinear to elegantly handle the flattened dimension (seq_len * input_size)
            self.head = nn.LazyLinear(num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, seq_len, input_size)"""
            x = self.input_norm(x)
            x_flat = self.flatten(x)
            o = self.head(x_flat)
            return o.squeeze(-1) if self.num_classes == 1 else o

    # â”€â”€ C: Model role separation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Each architecture has an explicit role in the ensemble pipeline.
    # Used by DiversityLoss to compute role-conditioned diversity penalties
    # and by the ensemble router to weight predictions appropriately.
    MODEL_ROLES = {
        "mamba": "fast_reaction",  # Sub-100 ms latency; micro-structure signals
        "tft": "context",  # Interprets multi-scale regime context
        "haelt": "confirmation",  # LSTM+Transformer confirmation gate
        "gnn": "risk_modulation",  # Cross-asset correlation; reduces position on systemic risk
        "transformer": "context",  # iTransformer - variate-level attention
        "expert": "confirmation",  # EXPERT encoder - conv-based local confirmation
        "glm": "baseline",  # Generalized Linear Model baseline
        "patchtst": "context",  # PatchTST captures local semantics for context
        "timesnet": "context",  # TimesNet 2D-variation multiscale decomposition
        "timemixer": "context",  # TimeMixer decomposable multiscale mixing
    }

    class DiversityLoss(nn.Module):
        """
        C: Diversity loss - penalizes high pairwise correlation between model outputs.

        During ensemble training, if two models produce nearly identical predictions
        they offer no complementary information.  This loss encourages the ensemble
        to maintain diverse predictions across models with different roles.

        Usage:
            div_loss = DiversityLoss(weight=0.1)
            # preds: list of (B,) tensors, one per model
            loss = task_loss + div_loss(preds)

        Role-aware mode: models with the SAME role receive a larger diversity weight
        (they should be most distinct if we have role duplicates), while models with
        DIFFERENT roles receive a smaller weight (they are expected to be somewhat
        correlated - they predict the same asset after all).
        """

        def __init__(
            self,
            weight: float = 0.10,
            same_role_mult: float = 2.0,  # extra penalty for same-role pairs
            roles: list | None = None,  # list of role strings, one per model
            freq_weight: float = 0.05,  # role-specialized frequency diversity
            freq_roles: dict | None = None,  # e.g. {"fast_reaction": "high", "risk_modulation": "low"}
        ):
            super().__init__()
            self.weight = float(weight)
            self.same_role_mult = float(same_role_mult)
            self.roles = roles  # None -> uniform weighting
            self.freq_weight = float(freq_weight)
            self.freq_roles = freq_roles or {
                "fast_reaction": "high",
                "context": "mid",
                "confirmation": "mid",
                "risk_modulation": "low",
                "baseline": "low",
            }

        @staticmethod
        def _freq_ratio(pred: "torch.Tensor") -> "torch.Tensor":
            """High-frequency ratio: std(diff) / (std(pred)+eps) in [0,1]."""
            if pred.numel() < 2:
                return torch.tensor(0.5, device=pred.device)
            diff = pred[1:] - pred[:-1]
            return (diff.std() + 1e-8) / (pred.std() + 1e-8 + 1e-8).clamp(min=1e-8)

        def forward(self, preds: list) -> "torch.Tensor":
            """
            preds: list of tensors, each shape (B,) - one scalar prediction per model.
            Returns a scalar diversity penalty (minimise this).
            Includes role-conditioned correlation penalty + frequency specialization.
            """
            if len(preds) < 2:
                # Still apply freq penalty for single model (encourage correct band)
                if self.freq_weight > 0 and self.roles is not None and len(self.roles) >= 1:
                    freq_pen = torch.tensor(0.0, device=preds[0].device)
                    for idx, p in enumerate(preds):
                        role = self.roles[idx] if idx < len(self.roles) else None
                        target = self.freq_roles.get(role, "mid") if role else "mid"
                        ratio = self._freq_ratio(p.float())
                        # fast_reaction -> high ratio (~0.7-1.0), risk_modulation -> low (~0.0-0.3)
                        if target == "high":
                            freq_pen = freq_pen + (1.0 - ratio).clamp(min=0)
                        elif target == "low":
                            freq_pen = freq_pen + ratio.clamp(min=0)
                    return self.freq_weight * freq_pen / len(preds)
                return torch.tensor(0.0, device=preds[0].device)
            penalty = torch.tensor(0.0, device=preds[0].device)
            n_pairs = 0
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    p_i = preds[i].float()
                    p_j = preds[j].float()
                    # Pearson correlation
                    p_i_c = p_i - p_i.mean()
                    p_j_c = p_j - p_j.mean()
                    var_i = (p_i_c ** 2).sum().clamp(min=1e-8)
                    var_j = (p_j_c ** 2).sum().clamp(min=1e-8)
                    denom = torch.sqrt(var_i * var_j)
                    corr = (p_i_c * p_j_c).sum() / denom
                    # Role multiplier
                    mult = 1.0
                    if self.roles is not None and len(self.roles) > max(i, j):  # noqa: SIM102
                        if self.roles[i] == self.roles[j]:
                            mult = self.same_role_mult
                    penalty = penalty + mult * corr.abs()
                    n_pairs += 1
            base = self.weight * penalty / max(n_pairs, 1)
            # Role-specialized frequency diversity: encourage fast_reaction=high-freq, risk_modulation=low-freq
            if self.freq_weight > 0 and self.roles is not None:
                freq_pen = torch.tensor(0.0, device=preds[0].device)
                for idx, p in enumerate(preds):
                    role = self.roles[idx] if idx < len(self.roles) else None
                    target = self.freq_roles.get(role, "mid") if role else "mid"
                    ratio = self._freq_ratio(p.float())
                    if target == "high":
                        freq_pen = freq_pen + (1.0 - ratio).clamp(min=0)
                    elif target == "low":
                        freq_pen = freq_pen + ratio.clamp(min=0)
                    # mid -> no penalty (any frequency allowed)
                base = base + self.freq_weight * freq_pen / len(preds)
            return base

    # â”€â”€ D: Model confidence calibration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    class TemperatureScaler(nn.Module):
        """
        D: Post-hoc temperature scaling for confidence calibration.

        Divides logits (or regression outputs) by a learnable scalar T.
        For classifiers: calibrates P(class) = softmax(logits / T).
        For regressors: scales the output magnitude.

        Fit after training with a small calibration set (val split).
        T > 1 -> less confident (softer), T < 1 -> more confident (sharper).

        Usage:
            scaler = TemperatureScaler(model)
            scaler.calibrate(val_loader, device)   # fits T via NLL minimisation
            logits_cal = scaler(x)
        """

        def __init__(self, model: "nn.Module"):
            super().__init__()
            self.model = model
            self.temperature = nn.Parameter(torch.ones(1))

        def forward(self, x: "torch.Tensor"):
            output = self.model(x)
            if isinstance(output, tuple):
                # Multi-task output: (direction_logits, ret_pred, conf_pred, ...)
                scaled_logits = output[0] / self.temperature.clamp(min=0.05)
                return (scaled_logits, *output[1:])
            return output / self.temperature.clamp(min=0.05)

        def calibrate(
            self,
            val_loader: "Any",
            device: "torch.device",
            classification: bool = True,
            lr: float = 0.01,
            max_iter: int = 100,
        ) -> float:
            """
            Fit temperature T on the validation set.
            Returns the calibrated temperature value.
            """
            self.model.eval()
            logits_list, labels_list = [], []
            with torch.no_grad():
                for batch in val_loader:
                    X_b = batch[0]
                    y_b = batch[2] if classification and len(batch) > 2 else batch[1]
                    if y_b.dim() > 1:
                        y_b = y_b.squeeze(-1)
                    X_b = X_b.to(device)
                    out = self.model(X_b)
                    if isinstance(out, tuple):
                        out = out[0] if classification else out[1]
                    logits_list.append(out.cpu())
                    labels_list.append(y_b.cpu())
            all_logits = torch.cat(logits_list).to(self.temperature.device)
            all_labels = torch.cat(labels_list).to(self.temperature.device)

            optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
            if classification:
                # Map {-1,0,+1} direction labels to CE indices {0,1,2}.
                # Already-indexed labels are accepted for plain classifiers.
                y_flat = all_labels.reshape(-1).float()
                if y_flat.numel() and (y_flat.min() < 0 or y_flat.max() <= 1):
                    y_cls = (y_flat + 1.0).round().long().clamp(0, 2)
                else:
                    y_cls = y_flat.round().long().clamp(0, 2)

                def cal_step():
                    optimizer.zero_grad()
                    loss = F.cross_entropy(all_logits / self.temperature.clamp(min=0.05), y_cls)
                    loss.backward()
                    return loss
            else:
                all_labels = all_labels.reshape_as(all_logits)

                def cal_step():
                    optimizer.zero_grad()
                    loss = F.mse_loss(all_logits / self.temperature.clamp(min=0.05), all_labels)
                    loss.backward()
                    return loss

            optimizer.step(cal_step)
            t_val = float(self.temperature.item())
            print(f"[Calibration] Optimal temperature T = {t_val:.4f}")
            return t_val

    class OverconfidencePenalty(nn.Module):
        """
        D: Penalise overconfident wrong predictions during training.

        Adds an extra term to the loss whenever the model is confident
        (|pred| > threshold) but the prediction is in the wrong direction.
        This discourages brittle over-fitting to training noise.

        Usage:
            base_loss = HuberLoss()(pred, target)
            penalty   = OverconfidencePenalty()(pred, target)
            loss      = base_loss + penalty
        """

        def __init__(self, conf_threshold: float = 0.6, weight: float = 0.5):
            super().__init__()
            self.conf_threshold = float(conf_threshold)
            self.weight = float(weight)

        def forward(self, pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
            confident = pred.abs() > self.conf_threshold
            wrong_dir = (pred * target) < 0
            overconf_bad = (confident & wrong_dir).float()
            penalty = overconf_bad * pred.abs()
            return self.weight * penalty.mean()

    # â”€â”€ Model factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    MODEL_REGISTRY = {
        "tft": TFTScalper,
        "transformer": iTransformerScalper,
        "itransformer": iTransformerScalper,
        "haelt": HAELTHybrid,
        "mamba": MambaScalper,
        "gnn": GNNFromSequence,
        "expert": EXPERTEncoder,
        "expertencoder": EXPERTEncoder,
        "glm": GLMBaseline,
        "patchtst": PatchTSTScalper,
        "patchtstscalper": PatchTSTScalper,
        "timesnet": TimesNetScalper,
        "timemixer": TimeMixerScalper,
    }

    # ModelZoo provides a stable import for external code
    class _ModelZooMeta(type):
        def __getattr__(cls, name: str):
            """Dynamically expose models as class attributes on ModelZoo class."""
            key = name.lower()
            if key.startswith("baseline_"):
                key = key.replace("baseline_", "", 1)
            if hasattr(cls, "_registry"):
                if key in cls._registry:
                    return cls._registry[key]
                aliases = {
                    "tftscalper": "tft",
                    "haelthybrid": "haelt",
                    "mambascalper": "mamba",
                    "gnncrossasset": "gnn",
                    "gnnfromsequence": "gnn",
                    "glmbaseline": "glm",
                    "timesnetscalper": "timesnet",
                    "timemixerscalper": "timemixer",
                }
                if key in aliases and aliases[key] in cls._registry:
                    return cls._registry[aliases[key]]
            raise AttributeError(f"ModelZoo has no attribute '{name}'")

    class ModelZoo(metaclass=_ModelZooMeta):
        """Expose the model registry for external imports.
        Allows attribute access like ModelZoo.TFTScalper.
        """
        _registry = MODEL_REGISTRY

        @classmethod
        def get(cls, name: str):
            key = name.lower()
            if key.startswith("baseline_"):
                key = key.replace("baseline_", "", 1)
            return cls._registry.get(key)

        @classmethod
        def list_models(cls):
            return list(cls._registry.keys())

    # Export symbols for wildcard imports
    __all__ = [
        "build_model",
        "ModelZoo",
        "MultiTaskHead",
        "MultiTaskLoss",
        "MultiTaskWrapper",
        "MultiPairMultiTaskHead",
        "MultiPairMultiTaskLoss",
        "MultiPairMultiTaskWrapper",
        "MultiPairWrapper",
        "TFTScalper",
    ]


else:
    # CPU stubs when torch unavailable. Define a single runtime stub and alias it
    # to the expected public names so Pyright does not see duplicate declarations
    # in the same module when the torch-enabled branch is present.
    class _TorchUnavailableStub:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    TFTScalper = cast(Any, _TorchUnavailableStub)
    iTransformerScalper = cast(Any, _TorchUnavailableStub)
    HAELTHybrid = cast(Any, _TorchUnavailableStub)
    MambaScalper = cast(Any, _TorchUnavailableStub)
    GNNCrossAsset = cast(Any, _TorchUnavailableStub)
    GNNFromSequence = cast(Any, _TorchUnavailableStub)
    EXPERTEncoder = cast(Any, _TorchUnavailableStub)
    PatchTSTScalper = cast(Any, _TorchUnavailableStub)
    TimesNetScalper = cast(Any, _TorchUnavailableStub)
    TimeMixerScalper = cast(Any, _TorchUnavailableStub)
    MultiTaskHead = cast(Any, _TorchUnavailableStub)
    MultiTaskLoss = cast(Any, _TorchUnavailableStub)
    MultiTaskWrapper = cast(Any, _TorchUnavailableStub)
    MultiPairMultiTaskHead = cast(Any, _TorchUnavailableStub)
    MultiPairMultiTaskLoss = cast(Any, _TorchUnavailableStub)
    MultiPairMultiTaskWrapper = cast(Any, _TorchUnavailableStub)
    MultiPairWrapper = cast(Any, _TorchUnavailableStub)
    DiversityLoss = cast(Any, _TorchUnavailableStub)
    TemperatureScaler = cast(Any, _TorchUnavailableStub)
    OverconfidencePenalty = cast(Any, _TorchUnavailableStub)
    AsymmetricDirectionalLoss = cast(Any, _TorchUnavailableStub)

    # Added ModelZoo placeholder for compatibility with tests
    class ModelZoo:
        """Placeholder class exposing model registry for external imports."""
        _registry = {}

        @classmethod
        def get(cls, name: str):
            """Retrieve model class by name (caseâ€‘insensitive)."""
            key = name.lower()
            if key.startswith("baseline_"):
                key = key.replace("baseline_", "", 1)
            return cls._registry.get(key)

        @classmethod
        def list_models(cls):
            """Return a list of available model names."""
            return list(cls._registry.keys())

    # Export symbols for wildcard imports
    __all__ = [
        "build_model",
        "ModelZoo",
        "MultiTaskHead",
        "MultiTaskLoss",
        "MultiTaskWrapper",
        "MultiPairMultiTaskHead",
        "MultiPairMultiTaskLoss",
        "MultiPairMultiTaskWrapper",
        "MultiPairWrapper",
        "TFTScalper",
    ]

    MODEL_REGISTRY = {}
    ModelZoo._registry = MODEL_REGISTRY
    MODEL_ROLES: dict = {}


if __name__ == "__main__" and TORCH:
    import torch

    B, T, F_IN = 8, 60, 48
    x = torch.randn(B, T, F_IN)

    for name, Cls in [
        ("TFT", TFTScalper),
        ("iTransformer", iTransformerScalper),
        ("HAELT", HAELTHybrid),
        ("Mamba", MambaScalper),
        ("EXPERT", EXPERTEncoder),
        ("PatchTST", PatchTSTScalper),
        ("TimesNet", TimesNetScalper),
        ("TimeMixer", TimeMixerScalper),
    ]:
        try:
            m = Cls(input_size=F_IN)
            out = m(x)
            print(f"  {name:16s}: in {tuple(x.shape)} -> out {tuple(out.shape)}")
        except Exception as e:
            print(f"  {name:16s}: ERROR - {e}")

    x_seq = torch.randn(B, T, F_IN)
    gnn = GNNFromSequence(input_size=F_IN, hidden=64, num_layers=2, dropout=0.1)
    out = gnn(x_seq)
    print(f"  {'GNN-seq':16s}: in {tuple(x_seq.shape)} -> out {tuple(out.shape)}")


# Register the model factory with config (dependency inversion: config never
# imports models; models registers itself here at import time).
from config.model_training_profile import register_build_model

register_build_model(build_model)


