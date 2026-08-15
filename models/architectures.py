"""
models/architectures.py
========================
All six model architectures specified:
  1. TFT          — Temporal Fusion Transformer
  2. iTransformer — Variate-dimension attention
  3. HAELTHybrid  — LSTM + Transformer parallel
  4. MambaScalper — State Space Model (low latency)
  5. GNNCrossAsset— Graph Neural Network for cross-asset correlations
  6. EXPERTEncoder— Exchange-Rate Transformer (conv FFN, no positional enc)

Shared interface: forward(x) -> (batch,) scalars if num_classes==1, else (batch, num_classes) logits.
"""

import inspect
import warnings
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False
    warnings.warn("PyTorch not installed. pip install torch")


if TORCH:

    # ── Shared building blocks ─────────────────────────────────────────────

    def _kaiming_init_module(mod: "nn.Module") -> None:
        """Xavier/Kaiming-stable init; moderate head gain keeps outputs ~O(1)."""
        for name, m in mod.named_modules():
            if isinstance(m, nn.Linear):
                # Final prediction heads: gain=0.1 (was 0.01 — too vanishing for deep nets).
                is_head = (
                    name == "head"
                    or name.startswith("head.")
                    or name.endswith(".head")
                    or ".head." in name
                )
                gain = 0.1 if is_head else 1.0
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _maybe_checkpoint(fn, *args, enabled: bool = True):
        """Gradient checkpointing when training; no-op at eval / when disabled."""
        if enabled and torch.is_grad_enabled() and any(
            isinstance(a, torch.Tensor) and a.requires_grad for a in args
        ):
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    class _FlashMHA(nn.Module):
        """
        Multi-head self-attention that calls F.scaled_dot_product_attention directly.

        Unlike nn.MultiheadAttention, this bypasses PyTorch's internal dispatch
        guards and unconditionally routes to the Flash Attention / mem-efficient
        SDPA kernel when tensors are in FP16 or BF16 (autocast context).

        Requirements for Flash Attention kernel (PyTorch >= 2.0, CUDA):
          • Tensors must be FP16 or BF16 — satisfied by AMP autocast.
          • No custom attn_mask — self-attention only, no causal mask needed.
          • Head dim should be 16, 32, 64, or 128 for best performance.
        """
        def __init__(self, d_model: int, heads: int, dropout: float = 0.0):
            super().__init__()
            assert d_model % heads == 0, f"d_model ({d_model}) must be divisible by heads ({heads})"
            self.heads    = heads
            self.head_dim = d_model // heads
            self.scale    = self.head_dim ** -0.5
            self.dropout  = dropout
            self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, d_model)
            B, T, _ = x.shape
            qkv = self.qkv(x)                                         # (B, T, 3*d)
            q, k, v = qkv.chunk(3, dim=-1)                            # each (B, T, d)
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
        def __init__(self, delta=1.0, sign_weight=0.0):
            super().__init__()
            self.delta = delta
            self.sign_weight = sign_weight
        def forward(self, p, t, weight=None):
            e = p - t; a = e.abs()
            base = torch.where(a <= self.delta, 0.5 * e**2, self.delta * (a - 0.5 * self.delta))
            dir_penalty = (torch.sign(p) != torch.sign(t)).float() * self.sign_weight * a
            loss = base + dir_penalty
            if weight is not None:
                loss = loss * weight
            return loss.mean()

    class AsymmetricDirectionalLoss(nn.Module):
        """
        Huber on residuals plus extra penalty when prediction and target disagree in
        sign (direction), as discussed for asymmetric economic risk in directional
        forecasting. Targets are typically {-1,0,+1} bar labels.
        """
        def __init__(self, delta=1.0, sign_weight=2.0):
            super().__init__()
            self.delta = delta
            self.sign_weight = sign_weight

        def forward(self, pred, target, weight=None):
            e = pred - target
            a = e.abs()
            huber = torch.where(
                a <= self.delta,
                0.5 * e ** 2,
                self.delta * (a - 0.5 * self.delta),
            )
            tnz = target.abs() > 0.05
            wrong = tnz & (torch.sign(pred) != torch.sign(target))
            extra = wrong.float() * target.abs().clamp(min=0.1)
            loss = huber + self.sign_weight * extra
            if weight is not None:
                loss = loss * weight
            return loss.mean()

    # ── Multi-task head, loss, and backbone wrapper ───────────────────────

    class MultiTaskHead(nn.Module):
        """
        Three-output prediction head for multi-task supervision:
          direction  — 3-class logits {sell=0, hold=1, buy=2}   (cross-entropy)
          return_hat — scalar magnitude regression               (Huber)
          confidence — predicted |return|, clipped to [0,1]     (BCE)

        Training on all three signals simultaneously prevents the backbone from
        learning 'correct direction / wrong magnitude' solutions and gives a
        natural confidence signal for downstream position sizing.
        """

        def __init__(self, in_features: int, hidden: int = 64, dropout: float = 0.1,
                     return_aux: bool = False, recon_out_features: int | None = None):
            super().__init__()
            self.return_aux = bool(return_aux)
            h2 = max(hidden // 2, 16)
            self.direction = nn.Sequential(
                nn.Linear(in_features, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 3),
            )
            self.return_hat = nn.Sequential(
                nn.Linear(in_features, h2), nn.GELU(),
                nn.Linear(h2, 1),
            )
            self.confidence = nn.Sequential(
                nn.Linear(in_features, h2), nn.GELU(),
                nn.Linear(h2, 1),
                # Sigmoid removed — BCEWithLogitsLoss in MultiTaskLoss fuses it
                # safely under AMP. Do NOT add Sigmoid back here.
            )
            if self.return_aux:
                recon_dim = int(recon_out_features) if recon_out_features is not None else in_features
                self.recon = nn.Sequential(
                    nn.Linear(in_features, hidden), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden, recon_dim),
                )
                self.vol_hat = nn.Sequential(
                    nn.Linear(in_features, h2), nn.GELU(),
                    nn.Linear(h2, 1),
                )

        def forward(self, h: torch.Tensor):
            """h: (B, in_features) — backbone hidden state BEFORE any prediction head."""
            ret  = self.return_hat(h)   # (B, 1)
            conf = self.confidence(h)   # (B, 1)
            # reshape(-1) instead of squeeze(-1): safe when B=1 (squeeze would
            # collapse both dims -> scalar, breaking loss computation)
            outs = (
                self.direction(h),   # (B, 3)
                ret.reshape(-1),     # (B,)
                conf.reshape(-1),    # (B,)
            )
            if self.return_aux:
                return outs + (self.recon(h), self.vol_hat(h).reshape(-1))
            return outs


    class MultiTaskLoss(nn.Module):
        """
        Weighted combination of three supervised objectives:
          L = w_dir  * CrossEntropy(direction, y_cls)
            + w_ret  * Huber(return_hat, y_cont)
            + w_conf * BCE(confidence, |y_cont|)

        Typical weights: w_dir=1.0, w_ret=0.5, w_conf=0.3
        Class weights can be passed to CrossEntropyLoss to handle {-1,0,+1} imbalance.
        """

        def __init__(
            self,
            class_weights: torch.Tensor | None = None,
            w_dir:       float = 1.0,
            w_ret:       float = 0.5,
            w_conf:      float = 0.3,
            huber_delta: float = 1.0,
            class_balance_weight: float = 0.0,
            entropy_weight: float = 0.0,
            direction_weight_floor: float = 0.0,
            focal_gamma: float = 0.0,
            class_prior: torch.Tensor | None = None,
            w_sharpe:    float = 0.0,
            sharpe_ann:  float = 1.0,
            sharpe_eps:  float = 1e-8,
            label_smoothing: float = 0.05,
            class_floor_frac: float = 0.35,
            recall_margin: float = 0.35,
            dist_penalty_w: float = 4.0,
            balanced_ce_w: float = 0.15,
            aux_bce_w: float = 0.15,
            recon_w: float = 0.1,
            vol_w: float = 0.05,
        ):
            super().__init__()
            self.ce    = nn.CrossEntropyLoss(
                weight=class_weights,
                reduction="none",
                label_smoothing=float(max(0.0, label_smoothing)),
            )
            self.hub   = nn.HuberLoss(delta=huber_delta, reduction="none")
            self.bce   = nn.BCEWithLogitsLoss()   # AMP-safe; sigmoid is fused internally
            self.w_dir  = w_dir
            self.w_ret  = w_ret
            self.w_conf = w_conf
            self.w_sharpe = w_sharpe
            # sharpe_ann is already the sqrt-style annualization factor (e.g. 325).
            self.sharpe_sqrt = float(max(1.0, sharpe_ann))
            self.sharpe_eps = sharpe_eps
            self.class_balance_weight = float(class_balance_weight)
            self.entropy_weight = float(entropy_weight)
            self.direction_weight_floor = float(direction_weight_floor)
            self.focal_gamma = float(focal_gamma)
            self.class_floor_frac = float(class_floor_frac)
            self.recall_margin = float(recall_margin)
            self.dist_penalty_w = float(dist_penalty_w)
            self.balanced_ce_w = float(balanced_ce_w)
            self.aux_bce_w = float(aux_bce_w)
            self.recon_w = float(recon_w)
            self.vol_w = float(vol_w)
            prior = class_prior.float() if class_prior is not None else torch.ones(3) / 3.0
            prior = prior.reshape(-1).clamp_min(1e-6)
            prior = prior / prior.sum().clamp_min(1e-6)
            self.register_buffer("class_prior", prior)

        def forward(
            self,
            logits:  torch.Tensor,  # (B, 3)
            ret_hat: torch.Tensor,  # (B,)
            conf:    torch.Tensor,  # (B,)
            y_cls:   torch.Tensor,  # (B,) long {0,1,2}
            y_cont:  torch.Tensor,  # (B,) float continuous reward
            y_conf:  torch.Tensor | None = None,  # path_quality / confidence target / trade weight
            recon_hat: torch.Tensor | None = None,
            recon_tgt: torch.Tensor | None = None,
            vol_hat: torch.Tensor | None = None,
            vol_tgt: torch.Tensor | None = None,
        ) -> torch.Tensor:
            y_flat = y_cls.reshape(-1).clamp(0, 2)
            l_dir  = self.ce(logits, y_flat)
            if self.focal_gamma > 0.0:
                with torch.no_grad():
                    p_t = torch.softmax(logits, dim=-1).gather(1, y_flat.view(-1, 1)).squeeze(1)
                    focal = (1.0 - p_t.clamp(1e-6, 1.0)).pow(self.focal_gamma)
                l_dir = l_dir * focal
            l_ret  = self.hub(ret_hat, y_cont)
            l_dir_per_sample = l_dir

            if y_conf is not None:
                # y_conf is the confidence / path-quality *target* for BCE.
                # It is also used as a soft sample weight for dir/ret when provided
                # (path_quality convention). Callers that only want a BCE target
                # without reweighting should pass y_conf=None and rely on |y_cont|.
                weight = y_conf.clamp(0.0, 1.0)
                dir_weight = weight.clamp_min(self.direction_weight_floor)
                weighted_l_dir = l_dir_per_sample * dir_weight
                l_dir = weighted_l_dir.mean()
                l_ret = (l_ret * weight).mean()
                conf_tgt = weight
            else:
                dir_weight = torch.ones_like(l_dir_per_sample)
                weighted_l_dir = l_dir_per_sample
                l_dir = weighted_l_dir.mean()
                l_ret = l_ret.mean()
                conf_tgt = y_cont.abs().clamp(0.0, 1.0)

            l_conf = self.bce(conf, conf_tgt)
            loss = self.w_dir * l_dir + self.w_ret * l_ret + self.w_conf * l_conf
            if self.w_sharpe > 0.0:
                # Softsign: avoids tanh vanishing grads on confident ret_hat.
                direction = ret_hat / (1.0 + ret_hat.abs())
                returns = (direction * y_cont).flatten()
                mean = returns.mean()
                var = returns.var(unbiased=False)
                std = torch.sqrt(var + self.sharpe_eps)
                sharpe = mean / std * self.sharpe_sqrt
                loss = loss - self.w_sharpe * sharpe
            if self.class_balance_weight:
                probs = torch.softmax(logits, dim=-1)
                pred_dist = probs.mean(dim=0).clamp_min(1e-6)
                true_dist = self.class_prior.to(dtype=probs.dtype, device=probs.device)
                dist_mse = F.mse_loss(pred_dist, true_dist)
                dist_kl = F.kl_div(pred_dist.log(), true_dist, reduction="sum")
                class_floor = true_dist * self.class_floor_frac
                missing_penalty = F.relu(class_floor - pred_dist).pow(2).sum()
                recall_margin_t = logits.new_tensor(self.recall_margin)
                recall_penalty = logits.new_tensor(0.0)
                balanced_ce_parts = []
                aux_bce_parts = []
                target_oh = F.one_hot(y_flat, num_classes=logits.shape[-1]).to(
                    dtype=logits.dtype, device=logits.device,
                )
                for cls_idx in range(logits.shape[-1]):
                    mask = y_flat == cls_idx
                    if not bool(mask.any()):
                        continue
                    balanced_ce_parts.append(weighted_l_dir[mask].mean())
                    cls_logits = logits[mask]
                    aux_bce_parts.append(
                        F.binary_cross_entropy_with_logits(
                            cls_logits,
                            target_oh[mask],
                            reduction="mean",
                        )
                    )
                    target_logit = cls_logits[:, cls_idx]
                    other_logits = torch.cat(
                        [cls_logits[:, :cls_idx], cls_logits[:, cls_idx + 1:]],
                        dim=1,
                    )
                    other_best = other_logits.max(dim=1).values
                    recall_penalty = recall_penalty + F.relu(
                        recall_margin_t - (target_logit - other_best)
                    ).mean()
                balanced_ce = (
                    torch.stack(balanced_ce_parts).mean()
                    if balanced_ce_parts else logits.new_tensor(0.0)
                )
                aux_bce = (
                    torch.stack(aux_bce_parts).mean()
                    if aux_bce_parts else logits.new_tensor(0.0)
                )
                loss = loss + self.class_balance_weight * (
                    self.dist_penalty_w * (dist_mse + dist_kl + missing_penalty)
                    + recall_penalty
                    + self.balanced_ce_w * balanced_ce
                    + self.aux_bce_w * aux_bce
                )
            if self.entropy_weight:
                probs = torch.softmax(logits, dim=-1).clamp_min(1e-6)
                entropy = -(probs * probs.log()).sum(dim=-1).mean()
                loss = loss - self.entropy_weight * entropy
            if (
                recon_hat is not None
                and recon_tgt is not None
                and recon_hat.shape[-1] == recon_tgt.shape[-1]
            ):
                loss = loss + self.recon_w * F.mse_loss(recon_hat, recon_tgt)
            if vol_hat is not None and vol_tgt is not None:
                loss = loss + self.vol_w * F.mse_loss(vol_hat.reshape(-1), vol_tgt.reshape(-1))
            return loss


    class MultiTaskWrapper(nn.Module):
        """
        Wraps any of the 6 backbone architectures, replacing its .head with
        nn.Identity() to expose the pre-head hidden state, then routing that
        state through a MultiTaskHead.

        After wrapping, forward() returns (direction_logits, return_hat, confidence)
        instead of the backbone's scalar/logit prediction.

        When the backbone's pre-head dimension exceeds proj_threshold (e.g.
        iTransformer whose head_in = d_model × n_features ≈ 18k), an extra
        Linear+GELU projection to proj_to=256 is inserted automatically.

        Usage:
            base  = HAELTHybrid(input_size=73, num_classes=1)
            model = MultiTaskWrapper(base, head_in=256)
            logits, ret, conf = model(x)   # x: (B, T, F)
        """

        def __init__(
            self,
            backbone:       "nn.Module",
            head_in:        int,
            hidden:         int   = 64,
            dropout:        float = 0.1,
            proj_threshold: int   = 1024,
            proj_to:        int   = 256,
            force_project:  bool  = False,
            return_aux:     bool  = False,
            recon_out_features: int | None = None,
        ):
            super().__init__()
            self.backbone = backbone

            if force_project:
                # MultiPairWrapper and iTransformer can change the flattened
                # pre-head width after build-time estimates. LazyLinear binds
                # to the actual tensor width on the first forward pass.
                self.proj  = nn.Sequential(nn.LazyLinear(proj_to), nn.GELU())
                actual_in  = proj_to
            elif head_in > proj_threshold:
                self.proj  = nn.Sequential(nn.Linear(head_in, proj_to), nn.GELU())
                actual_in  = proj_to
            else:
                self.proj  = nn.Identity()
                actual_in  = head_in

            self.mt_head = MultiTaskHead(
                actual_in,
                hidden,
                dropout,
                return_aux=return_aux,
                recon_out_features=recon_out_features,
            )
            # Walk through wrapper chain to find the actual model with a head
            target = backbone
            while hasattr(target, 'backbone'):
                target = target.backbone
            target.head = nn.Identity()

        def forward(self, x: torch.Tensor):
            h = self.backbone(x)   # (B, head_in) — features from backbone
            h = self.proj(h)
            return self.mt_head(h)


    class MultiPairWrapper(nn.Module):
        """
        Structured multi-pair wrapper with cross-pair interaction.

        Improvements over naive concatenation:
          A. Cross-pair interaction channels:
             - Rolling Pearson correlation at two timescales (short + long window)
               for all i<j pairs — explicit co-movement signal the model need not infer.
             - Windowed relative momentum r_i(W) − r_j(W) over momentum_window bars.
             - Volatility share per pair: ATR_i / basket_ATR.
             - Cross-pair return dispersion (std across pairs at each bar).
          B. Regime-conditioned pair weighting — a small attention network weights
             pairs based on the last-bar regime context (vol, OFI-Z, session).
          C. Per-pair LayerNorm — normalises each pair's features independently
             before concatenation, preventing high-vol pairs from dominating.
          D. Alignment confidence gate — tracks how many pairs are present (non-zero)
             and feeds this as an extra scalar feature; missing data = regime signal.

        Input shape:  (B, T, n_pairs * f_per_pair)  — pairs concatenated on feat axis.
        Output: backbone prediction with enriched cross-pair features.

        New feature count fed to backbone:
            n_pairs * (f_per_pair + embed_dim)      [pair features + learned embedding]
          + n_pairs * (n_pairs - 1) // 2            [pairwise windowed relative momentum]
          + n_pairs * (n_pairs - 1) // 2            [pairwise short rolling correlation]
          + n_pairs * (n_pairs - 1) // 2            [pairwise long rolling correlation]
          + n_pairs                                  [vol share per pair]
          + 1                                        [cross-pair dispersion]
          + 1                                        [alignment confidence]
        """

        def __init__(
            self,
            backbone:        "nn.Module",
            n_pairs:         int,
            f_per_pair:      int,
            embed_dim:       int   = 16,
            regime_ctx_dim:  int   = 8,   # regime context features from last bar
            corr_window:     int   = 20,  # short rolling correlation window (bars)
            corr_window_long: int  = 60,  # long rolling correlation window (bars)
            momentum_window: int   = 20,  # windowed relative momentum lookback
            return_idx:      int   = 0,   # per-pair return proxy column (fallback 0)
            atr_idx:         int   = 1,   # per-pair ATR proxy column (fallback 1)
            corr_dropout_p:  float = 0.3,
        ):
            super().__init__()
            self.backbone         = backbone
            self.n_pairs          = n_pairs
            self.f_per_pair       = f_per_pair
            self.embed_dim        = embed_dim
            self.corr_window      = corr_window
            self.corr_window_long = corr_window_long
            self.momentum_window  = momentum_window
            self.return_idx       = int(return_idx)
            self.atr_idx          = int(atr_idx)
            self.corr_dropout_p   = corr_dropout_p

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
        def _rolling_corr(
            x: "torch.Tensor", y: "torch.Tensor", window: int
        ) -> "torch.Tensor":
            """
            Pearson correlation of x and y computed over a rolling window.

            x, y : (B, T) — one time series each
            Returns (B, T) — clipped to [-1, 1]; first (window-1) bars are 0.

            Implemented as differentiable 1-D convolutions so it runs on-GPU
            and participates in autograd for regime_attn gradients.
            """
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y.float(), nan=0.0, posinf=0.0, neginf=0.0)
            B, T = x.shape
            kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / window
            pad    = window - 1

            def _rmean(t: "torch.Tensor") -> "torch.Tensor":
                return F.conv1d(
                    F.pad(t.unsqueeze(1), (pad, 0)),
                    kernel,
                ).squeeze(1)  # (B, T)

            mx  = _rmean(x);    my  = _rmean(y)
            mxy = _rmean(x * y)
            mx2 = _rmean(x * x); my2 = _rmean(y * y)

            cov   = mxy - mx * my
            var_x = (mx2 - mx * mx).clamp(min=1e-8)
            var_y = (my2 - my * my).clamp(min=1e-8)
            corr = cov / (var_x.sqrt() * var_y.sqrt())
            return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

        def _cross_pair_features(self, xp: "torch.Tensor") -> "torch.Tensor":
            """
            Compute cross-pair interaction features from (B, T, P, F).

            Returns (B, T, n_interaction) containing:
              - RelMom_{i,j}     : windowed r_i(W) − r_j(W) for all i<j pairs
              - ShortCorr_{i,j}  : rolling Pearson corr (corr_window) for all i<j
              - LongCorr_{i,j}   : rolling Pearson corr (corr_window_long) for all i<j
              - VolShare_i       : ATR_i / sum(ATR_j) for each pair
              - Dispersion_t     : StdDev of cross-pair returns at each bar
              - AlignConf_t      : fraction of pairs with non-zero features
            """
            xp = torch.nan_to_num(xp.float(), nan=0.0, posinf=0.0, neginf=0.0)
            B, T, P, Fp = xp.shape   # Fp = features per pair (avoid shadowing F=functional)

            ri, ai = self.return_idx, self.atr_idx
            ri = min(max(ri, 0), Fp - 1)
            ai = min(max(ai, 0), Fp - 1)
            close_feat = xp[..., ri]         # (B, T, P) return proxy
            atr_feat   = xp[..., ai].abs() + 1e-8

            # D. Alignment confidence
            has_data = (xp.abs().sum(-1) > 1e-6).float()  # (B, T, P)
            conf     = has_data.mean(-1, keepdim=True)     # (B, T, 1)

            # A. Windowed relative momentum — rolling mean return difference
            W = self.momentum_window
            kernel_m = torch.ones(1, 1, W, device=xp.device, dtype=xp.dtype) / W
            pad_m    = W - 1
            # rolling mean return per pair: (B, T, P)
            cf_t = close_feat.permute(0, 2, 1)  # (B, P, T)
            cf_padded = F.pad(cf_t.reshape(B * P, 1, T), (pad_m, 0))
            mom_all = F.conv1d(cf_padded, kernel_m).reshape(B, P, T).permute(0, 2, 1)  # (B,T,P)

            rel_mom_parts = []
            for i in range(P):
                for j in range(i + 1, P):
                    rel_mom_parts.append((mom_all[..., i] - mom_all[..., j]).unsqueeze(-1))
            rel_mom = (torch.cat(rel_mom_parts, dim=-1)
                       if rel_mom_parts
                       else torch.zeros(B, T, 0, device=xp.device))  # (B, T, n_cross)

            # A. Rolling Pearson correlation — short and long windows
            short_corr_parts = []
            long_corr_parts  = []
            for i in range(P):
                for j in range(i + 1, P):
                    xi = close_feat[..., i]   # (B, T)
                    xj = close_feat[..., j]
                    sc = self._rolling_corr(xi, xj, self.corr_window).unsqueeze(-1)
                    lc = self._rolling_corr(xi, xj, self.corr_window_long).unsqueeze(-1)

                    if self.training and self.corr_dropout_p > 0.0:
                        xi_c = xi - xi.mean(dim=-1, keepdim=True)
                        xj_c = xj - xj.mean(dim=-1, keepdim=True)
                        pair_corr = (xi_c * xj_c).sum(dim=-1) / (xi_c.norm(dim=-1) * xj_c.norm(dim=-1)).clamp(min=1e-8)

                        drop = (pair_corr.abs() > 0.90) & (torch.rand_like(pair_corr) < self.corr_dropout_p)
                        drop = drop.view(-1, 1, 1).expand_as(sc)

                        sc = sc.masked_fill(drop, 0.0)
                        lc = lc.masked_fill(drop, 0.0)

                    short_corr_parts.append(sc)
                    long_corr_parts.append(lc)
            if short_corr_parts:
                short_corr = torch.cat(short_corr_parts, dim=-1)   # (B, T, n_cross)
                long_corr  = torch.cat(long_corr_parts,  dim=-1)
            else:
                short_corr = torch.zeros(B, T, 0, device=xp.device)
                long_corr  = torch.zeros(B, T, 0, device=xp.device)

            # A. Volatility share per pair: ATR_i / basket_ATR
            atr_sum   = atr_feat.sum(-1, keepdim=True) + 1e-8
            vol_share = atr_feat / atr_sum                         # (B, T, P)

            # A. Cross-pair return dispersion
            dispersion = (close_feat.var(dim=-1, keepdim=True, unbiased=False) + 1e-8).sqrt()     # (B, T, 1)

            cross = torch.cat(
                [rel_mom, short_corr, long_corr, vol_share, dispersion, conf], dim=-1
            )  # (B, T, 3*n_cross + P + 2)
            return torch.nan_to_num(cross, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
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
            last_bar_feat = normed[:, -1, 0, :]  # (B, F) — primary pair, last bar
            last_cross    = cross[:, -1, :]       # (B, n_interaction)
            ctx = torch.nan_to_num(
                torch.cat([last_bar_feat, last_cross], dim=-1),
                nan=0.0, posinf=0.0, neginf=0.0,
            )  # (B, F + n_inter)
            attn_logits = torch.nan_to_num(self.regime_attn(ctx), nan=0.0, posinf=0.0, neginf=0.0)
            pair_weights = torch.softmax(attn_logits.float(), dim=-1).to(normed.dtype)  # (B, P)
            pair_weights = torch.nan_to_num(pair_weights, nan=1.0 / self.n_pairs, posinf=0.0, neginf=0.0)

            # Apply pair weights: scale each pair's features by learned attention
            pw = pair_weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, P, 1)
            normed = normed * pw                           # (B, T, P, F)

            # Append static pair embeddings
            ids = torch.arange(self.n_pairs, device=x.device)
            emb = self.pair_embeds(ids).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
            normed = torch.cat([normed, emb], dim=-1)      # (B, T, P, F+E)

            # Flatten pairs + append cross-pair interaction
            flat   = normed.reshape(B, T, self.n_pairs * (self.f_per_pair + self.embed_dim))
            full   = torch.nan_to_num(
                torch.cat([flat, cross], dim=-1),
                nan=0.0, posinf=0.0, neginf=0.0,
            ).clamp(-10.0, 10.0)      # (B, T, pairs_flat + n_inter)

            return self.backbone(full)


    # ── 1. Temporal Fusion Transformer (simplified) ────────────────────────

    class VariableSelectionNetwork(nn.Module):
        """Learns which features matter at each timestep."""
        def __init__(self, input_size, hidden, dropout=0.1):
            super().__init__()
            self.grn = nn.Sequential(
                nn.Linear(input_size, hidden), nn.ELU(),
                nn.Dropout(dropout), nn.Linear(hidden, input_size),
            )
            self.softmax = nn.Softmax(dim=-1)
        def forward(self, x):
            weights = self.softmax(self.grn(x))
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
        encodings — we use a learnable absolute positional embedding, which
        is the simplest equivalent that restores temporal ordering awareness
        for the attention layer.
        """
        def __init__(self, input_size=64, hidden=128, heads=4,
                     lstm_layers=2, dropout=0.1, num_classes=1,
                     use_gradient_checkpointing: bool = True,
                     max_seq_len: int = 240):
            super().__init__()
            self.num_classes = num_classes
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.vsn     = VariableSelectionNetwork(input_size, hidden, dropout)
            self.lstm    = nn.LSTM(input_size, hidden, lstm_layers,
                                   batch_first=True, dropout=dropout)
            # A4: positional embedding for the post-LSTM self-attention block.
            # max_seq_len=240 covers typical daily/hourly seq lengths; the
            # forward gracefully slices or cycles if T differs.
            self.pos_emb = nn.Embedding(max_seq_len, hidden)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
            self.max_seq_len = int(max_seq_len)
            self.attn    = _FlashMHA(hidden, heads, dropout=dropout)
            self.norm1   = nn.LayerNorm(hidden)
            self.ffn     = nn.Sequential(nn.Linear(hidden,hidden*2),nn.GELU(),
                                         nn.Dropout(dropout),nn.Linear(hidden*2,hidden))
            self.norm2   = nn.LayerNorm(hidden)
            self.norm_out= nn.LayerNorm(hidden)
            self.head    = nn.Linear(hidden, num_classes)
            _kaiming_init_module(self)

        def _attn_block(self, h):
            return h + self.attn(self.norm1(h))

        def _ffn_block(self, h):
            return h + self.ffn(self.norm2(h))

        def _add_pos(self, lstm_out):
            """Add positional embedding to (B, T, hidden) tensor.
            Handles T == max_seq_len (exact), T < max_seq_len (forward slice),
            and T > max_seq_len (cyclic reuse — fallback for longer inputs).
            """
            T = lstm_out.size(1)
            if T <= self.max_seq_len:
                pos = self.pos_emb.weight[:T]  # (T, hidden)
            else:
                idx = torch.arange(T, device=lstm_out.device) % self.max_seq_len
                pos = self.pos_emb.weight[idx]
            return lstm_out + pos.unsqueeze(0)

        def forward(self, x):
            # x: (B, T, F) — pre-norm: x + f(norm(x))
            x_sel, _ = self.vsn(x)
            lstm_out, _ = self.lstm(x_sel)
            # A4: inject positional embedding before permutation-equivariant attention
            lstm_out = self._add_pos(lstm_out)
            h = _maybe_checkpoint(
                self._attn_block, lstm_out, enabled=self.use_gradient_checkpointing,
            )
            h = _maybe_checkpoint(
                self._ffn_block, h, enabled=self.use_gradient_checkpointing,
            )
            out = self.head(self.norm_out(h[:, -1, :]))
            if isinstance(self.head, nn.Identity):
                return out
            return out.squeeze(-1) if self.num_classes == 1 else out

    # ── 2. iTransformer (variate-dimension attention) ──────────────────────

    class iTransformerScalper(nn.Module):
        """
        iTransformer: applies attention across the feature (variate) dimension.
        Treats EUR/USD price and US 10Y yield as different 'tokens',
        learning their interactions as a differentiable map.
        Outperforms standard time-dimension Transformers on multivariate series.
        """
        def __init__(self, input_size=64, seq_len=60, d_model=128,
                     nhead=8, num_layers=3, dim_ff=256, dropout=0.1, num_classes=1,
                     use_gradient_checkpointing: bool = True):
            super().__init__()
            self.num_classes = num_classes
            self.seq_len = seq_len
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            # Project each variate's time-series into d_model token
            self.variate_proj = nn.Linear(seq_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True, norm_first=True)
            self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)
            self.norm_out = nn.LayerNorm(d_model * input_size)
            # Separate norm for Identity head path (returns B, d_model)
            self.norm_out_identity = nn.LayerNorm(d_model)
            self.head     = nn.Linear(d_model * input_size, num_classes)
            self.input_size = input_size
            _kaiming_init_module(self)

        def forward(self, x):
            # x: (B, T, F)  ->  treat F as sequence, T as embedding
            B, T, n_feat = x.shape
            tokens = x.permute(0, 2, 1)          # (B, F, T)
            # Curriculum may slice T below build-time seq_len; resample to match variate_proj.
            if self.seq_len != T:
                tokens = F.interpolate(
                    tokens, size=self.seq_len, mode="linear", align_corners=False,
                )
            tokens = self.variate_proj(tokens)    # (B, F, d_model)
            out = _maybe_checkpoint(
                self.encoder, tokens, enabled=self.use_gradient_checkpointing,
            )
            # When head is Identity (MultiTaskWrapper), mean-pool variates to
            # (B, d_model) instead of materializing (B, F*d_model).
            if isinstance(self.head, nn.Identity):
                return self.norm_out_identity(out.mean(dim=1))  # normalize, matching all other archs
            out = out.reshape(B, -1)           # (B, F*d_model)
            o = self.head(self.norm_out(out))
            return o.squeeze(-1) if self.num_classes == 1 else o
    # ── 3. HAELT Hybrid (LSTM + Transformer in parallel) ──────────────────

    class HAELTHybrid(nn.Module):
        """
        Hybrid Attentive Ensemble Learning Transformer.
        LSTM branch captures local microstructure; Transformer captures
        long-range cross-asset correlations. Both run in parallel and are
        fused with a learned attention gate.

        A4 fix (2026-08-07): the Transformer branch is permutation-equivariant
        over time — without positional information it cannot distinguish
        bar 0 from bar 59, undermining the "long-range cross-asset correlations"
        docstring claim. We add a learnable ``nn.Embedding(seq_len, d_model)``
        positional embedding injected after the input projection.
        """
        def __init__(self, input_size=64, seq_len=60, lstm_hidden=64,
                     d_model=64, nhead=4, n_layers=2, dropout=0.1, num_classes=1,
                     use_gradient_checkpointing: bool = True):
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
            enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4,
                                              dropout=dropout, batch_first=True, norm_first=True)
            self.trf  = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
            self.attn_pool_lstm = nn.Linear(lstm_hidden, 1)
            self.attn_pool_trf = nn.Linear(d_model, 1)
            fused = lstm_hidden + d_model
            self.gate = nn.Sequential(nn.Linear(fused, fused), nn.Sigmoid())
            self.norm_out = nn.LayerNorm(fused)
            self.head = nn.Sequential(nn.Linear(fused,64),nn.GELU(),
                                       nn.Dropout(dropout),nn.Linear(64, num_classes))
            self._init_weights()

        def _init_weights(self):
            _kaiming_init_module(self)

        def forward(self, x):
            # PIPE-006: soft clipping via tanh scaling instead of hard clamp at ±10
            # Preserves signal magnitude for high-impact news events while bounding values
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.where(x.abs() > 10.0, 10.0 * torch.tanh(x / 10.0), x)
            # LSTM branch (sequence-aware — no position needed)
            lout, _ = self.lstm(x)
            attn_w_l = torch.softmax(self.attn_pool_lstm(lout), dim=1)
            lf = (lout * attn_w_l).sum(dim=1)
            # Transformer branch with A4 positional encoding
            h = self.proj(x)
            # Inject positional embedding: shape (T, d_model) broadcast over (B, T, d_model)
            T = h.size(1)
            if T == self.seq_len:
                pos = self.pos_emb.weight  # (seq_len, d_model)
                h = h + pos.unsqueeze(0)
            elif T <= self.seq_len:
                # Forward slicing if input is shorter than the training seq_len
                pos = self.pos_emb.weight[:T]
                h = h + pos.unsqueeze(0)
            else:
                # Longer input than training seq_len — pad by reusing positions cyclically.
                # This is a fallback; the canonical use-case has T == seq_len at training time.
                idx = torch.arange(T, device=h.device) % self.seq_len
                pos = self.pos_emb.weight[idx]  # (T, d_model)
                h = h + pos.unsqueeze(0)
            tout = _maybe_checkpoint(
                self.trf, h, enabled=self.use_gradient_checkpointing,
            )
            attn_w_t = torch.softmax(self.attn_pool_trf(tout), dim=1)
            tf = (tout * attn_w_t).sum(dim=1)
            c  = torch.cat([lf, tf], dim=-1)
            c  = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            g  = self.gate(c)
            o = self.head(self.norm_out(c + g * c))   # residual through gate (TM-016)
            if isinstance(self.head, nn.Identity):
                return o
            return o.squeeze(-1) if self.num_classes == 1 else o

    # ── 4. Mamba State Space Model ─────────────────────────────────────────

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
            self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
            # A8/A9 fix (2026-08-07): use asymmetric LEFT-ONLY padding so the
            # 1D conv is genuinely causal. The previous symmetric
            # `padding=d_conv-1` + `[:, :, :T]` truncation leaked `d_conv-1`
            # future bars into every output position (clearest in MambaScalper).
            # Note: Conv1d only accepts symmetric padding; we manually pad in forward()
            self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv,
                                       padding=0, groups=d_inner, bias=True)
            self.conv1d_pad = d_conv - 1
            self.act      = nn.SiLU()
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
            self.norm     = nn.LayerNorm(d_model)
            self.drop     = nn.Dropout(dropout)
            self.dt_proj  = nn.Linear(d_inner, d_inner, bias=True)

        def forward(self, x):
            # x: (B, T, d_model) — pre-norm residual (TM-012)
            B, T, D = x.shape
            res   = x
            x_n   = self.norm(x)
            xz    = self.in_proj(x_n)         # (B, T, d_inner*2)
            x2, z = xz.chunk(2, dim=-1)       # each (B, T, d_inner)
            # 1D conv along time (causal — asymmetric LEFT pad, no future leakage)
            # A8/A9 fix: manually pad left with zeros, then conv with padding=0
            # This avoids the 2-tuple padding issue in Conv1d
            x2_perm = x2.permute(0,2,1).contiguous()  # (B, d_inner, T)
            x2_pad = F.pad(x2_perm, (self.conv1d_pad, 0))  # (B, d_inner, T + pad)
            x2c = self.conv1d(x2_pad)  # (B, d_inner, T)
            x2c = x2c.permute(0,2,1).contiguous()
            x2c   = self.act(x2c)
            # Softplus-dt feature gate (not an SSM recurrence)
            y     = x2c * F.softplus(self.dt_proj(x2c))
            y     = y * torch.sigmoid(z)      # gating
            out   = self.out_proj(y)
            return res + self.drop(out)

    class MambaScalper(nn.Module):
        """
        Stack of MambaBlocks for low-latency HFT inference.
        Handles long sequences with O(L) cost vs transformer O(L²).
        Not a true Mamba SSM — see MambaBlock docstring.
        """
        arch_tag = "mamba_gated_v2"

        def __init__(self, input_size=64, d_model=128, d_conv=4,
                     expand=2, num_layers=4, dropout=0.1, num_classes=1,
                     use_gradient_checkpointing: bool = True):
            super().__init__()
            self.num_classes = num_classes
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.embed = nn.Linear(input_size, d_model)
            self.layers = nn.ModuleList([
                MambaBlock(d_model, d_conv, expand, dropout)
                for _ in range(num_layers)
            ])
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

    # ── 5. GNN Cross-Asset ────────────────────────────────────────────────

    class GNNCrossAsset(nn.Module):
        """
        Graph Neural Network for cross-asset modelling.

        Treats features (per node) as graph nodes. Message passing uses a
        learned dense adjacency (sigmoid(adj_logits)), not a fixed correlation
        graph.

        This is a GAT-style implementation without torch_geometric.
        """
        def __init__(self, node_features=32, hidden=64, num_layers=3,
                     heads=4, n_nodes=6, dropout=0.1, num_classes=1):
            super().__init__()
            self.n_nodes  = n_nodes
            self.num_classes = num_classes
            self.node_embed = nn.Linear(node_features, hidden)
            # Input-dependent adjacency: edge weights are a function of node
            # features so the graph structure can adapt across market regimes.
            self.adj_net = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, n_nodes),
            )
            self.attn_layers = nn.ModuleList([
                _FlashMHA(hidden, heads, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
            self.norm_out = nn.LayerNorm(hidden * n_nodes)
            self.head  = nn.Linear(hidden * n_nodes, num_classes)
            self.drop  = nn.Dropout(dropout)
            _kaiming_init_module(self)

        def forward(self, x, adj=None):
            """
            x  : (B, n_nodes, node_features) — one feature vector per node per bar
            adj: optional precomputed (B, n_nodes, n_nodes) edge weights. When None,
                 adjacency is computed from node embeddings via attention (adapts
                 across regimes). Pre-norm message passing (TM-012).
            """
            h = self.node_embed(x)          # (B, N, hidden)
            if adj is not None:
                A = adj
            else:
                # Attention-style dot-product adjacency: A_ij = softmax_j(q_i·k_j)
                q = self.adj_net(h)         # (B, N, N)
                k = self.adj_net(h)
                A = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (self.n_nodes ** 0.5), dim=-1)
            for attn, norm in zip(self.attn_layers, self.norms):
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
        Time axis is mean-pooled; features are projected into n_nodes × chunk tokens.
        """
        def __init__(self, input_size, hidden, num_layers, dropout, n_nodes=6,
                     num_classes=1, nhead=4):
            super().__init__()
            chunk = max(8, (input_size + n_nodes - 1) // n_nodes)
            self.n_nodes = n_nodes
            self.chunk = chunk
            self.proj = nn.Linear(input_size, n_nodes * chunk)
            self.temporal_attn = nn.Linear(input_size, 1)
            self.gnn = GNNCrossAsset(
                node_features=chunk, hidden=hidden, num_layers=num_layers,
                heads=nhead, n_nodes=n_nodes, dropout=dropout, num_classes=num_classes,
            )

        @property
        def head(self) -> "nn.Module":
            """Proxy to inner GNN head — enables MultiTaskWrapper compatibility."""
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

    # ── 6. EXPERT Encoder ─────────────────────────────────────────────────

    class ConvFFN(nn.Module):
        """1D conv feedforward — captures local temporal patterns better than MLP."""
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
            self.norm  = nn.LayerNorm(d_model)
            self.drop  = nn.Dropout(dropout)
            self.act   = nn.GELU()

        def forward(self, x):
            # x: (B, T, D) — pre-norm residual (TM-012)
            x_n = self.norm(x)
            h = x_n.permute(0,2,1).contiguous()  # (B, D, T)
            # Manual causal padding for conv1
            h = F.pad(h, (self.conv1_pad, 0))
            h = self.act(self.conv1(h))  # (B, d_ff, T)
            # Manual causal padding for conv2
            h = F.pad(h, (self.conv2_pad, 0))
            h = self.drop(self.conv2(h))  # (B, d_model, T)
            # A8/A9 fix: with manual asymmetric left padding + padding=0, output length
            # equals input length T exactly; no post-hoc slice needed.
            return x + h.permute(0,2,1).contiguous()

    class EXPERTEncoder(nn.Module):
        """
        EXPERT: EXchange-Rate Prediction using Encoder Representation from Transformers.
        Key differences from standard Transformer:
          - A4 fix (2026-08-07): positional encoding via learnable
            ``nn.Embedding(max_seq_len, d_model)``. The previous "order is
            inherent in time series" docstring was wrong — attention is
            permutation-equivariant and cannot tell bar 0 from bar 79 without
            position info. Especially with ConvFFN (causal-only) providing
            local temporal structure, the attention layer still needs absolute
            position info to distinguish distant timesteps.
          - 1D convolutional feedforward layers (local temporal patterns)
          - Encoder-only (no decoder needed for regression)
        Focused architecture makes it more data-efficient than general Transformers.
        """
        def __init__(self, input_size=64, d_model=128, nhead=8,
                     num_layers=4, dropout=0.1, num_classes=1,
                     use_gradient_checkpointing: bool = True,
                     max_seq_len: int = 240):
            super().__init__()
            self.num_classes = num_classes
            self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
            self.proj   = nn.Linear(input_size, d_model)
            # A4: learnable positional embedding (replaces old "no positional encoding")
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
            self.max_seq_len = int(max_seq_len)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "attn": _FlashMHA(d_model, nhead, dropout=dropout),
                    "norm1": nn.LayerNorm(d_model),
                    "ffn":  ConvFFN(d_model, d_model*4, dropout=dropout),
                })
                for _ in range(num_layers)
            ])
            self.pool = None  # replaced by last-timestep pooling for streaming
            self.norm_out = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)
            _kaiming_init_module(self)

        def _layer_forward(self, layer, h):
            # Pre-norm attention + ConvFFN (which is itself pre-norm)
            h = h + layer["attn"](layer["norm1"](h))
            return layer["ffn"](h)

        def _add_pos(self, h):
            """Add positional embedding to (B, T, d_model)."""
            T = h.size(1)
            if T <= self.max_seq_len:
                pos = self.pos_emb.weight[:T]
            else:
                idx = torch.arange(T, device=h.device) % self.max_seq_len
                pos = self.pos_emb.weight[idx]
            return h + pos.unsqueeze(0)

        def forward(self, x):
            h = self.proj(x)
            h = self._add_pos(h)   # A4: inject positional information
            for layer in self.layers:
                # Capture layer in default-arg closure for checkpoint safety.
                def _run(t, _layer=layer):
                    return self._layer_forward(_layer, t)
                h = _maybe_checkpoint(
                    _run, h, enabled=self.use_gradient_checkpointing,
                )
            h = h[:, -1, :]   # last-timestep for streaming/online inference
            o = self.head(self.norm_out(h))
            if isinstance(self.head, nn.Identity):
                return o
            return o.squeeze(-1) if self.num_classes == 1 else o

    class GLMBaseline(nn.Module):
        """Generalized Linear Model (GLM) baseline.

        Flattens the sequence and applies a single linear projection.
        Serves as an ultra-fast, lightweight baseline against complex deep learning models.
        """
        def __init__(self, input_size: int, num_classes: int, seq_len: int = 16):
            super().__init__()
            self.seq_len = seq_len
            self.input_norm = nn.LayerNorm(input_size)
            self.flatten = nn.Flatten(start_dim=1)
            # Use LazyLinear to elegantly handle the flattened dimension (seq_len * input_size)
            self.head = nn.LazyLinear(num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, seq_len, input_size)"""
            x = self.input_norm(x)
            x_flat = self.flatten(x)
            return self.head(x_flat)


    # ── C: Model role separation ───────────────────────────────────────────
    # Each architecture has an explicit role in the ensemble pipeline.
    # Used by DiversityLoss to compute role-conditioned diversity penalties
    # and by the ensemble router to weight predictions appropriately.
    MODEL_ROLES = {
        "mamba":       "fast_reaction",   # Sub-100 ms latency; micro-structure signals
        "tft":         "context",         # Interprets multi-scale regime context
        "haelt":       "confirmation",    # LSTM+Transformer confirmation gate
        "gnn":         "risk_modulation", # Cross-asset correlation; reduces position on systemic risk
        "transformer": "context",         # iTransformer — variate-level attention
        "expert":      "confirmation",    # EXPERT encoder — conv-based local confirmation
        "glm":         "baseline",        # Generalized Linear Model baseline
    }

    class DiversityLoss(nn.Module):
        """
        C: Diversity loss — penalizes high pairwise correlation between model outputs.

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
        correlated — they predict the same asset after all).
        """

        def __init__(
            self,
            weight:         float = 0.10,
            same_role_mult: float = 2.0,   # extra penalty for same-role pairs
            roles:          list | None = None,   # list of role strings, one per model
        ):
            super().__init__()
            self.weight         = float(weight)
            self.same_role_mult = float(same_role_mult)
            self.roles          = roles   # None -> uniform weighting

        def forward(self, preds: list) -> "torch.Tensor":
            """
            preds: list of tensors, each shape (B,) — one scalar prediction per model.
            Returns a scalar diversity penalty (minimise this).
            """
            if len(preds) < 2:
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
                    denom  = (p_i_c.norm() * p_j_c.norm()).clamp(min=1e-8)
                    corr   = (p_i_c * p_j_c).sum() / denom
                    # Role multiplier
                    mult = 1.0
                    if self.roles is not None and len(self.roles) > max(i, j):
                        if self.roles[i] == self.roles[j]:
                            mult = self.same_role_mult
                    penalty = penalty + mult * corr.abs()
                    n_pairs += 1
            return self.weight * penalty / max(n_pairs, 1)


    # ── D: Model confidence calibration ───────────────────────────────────

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
                return (scaled_logits,) + output[1:]
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
                    if y_b.dim() > 1: y_b = y_b.squeeze(-1)
                    X_b = X_b.to(device)
                    out  = self.model(X_b)
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
            self.weight         = float(weight)

        def forward(self, pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
            confident    = pred.abs() > self.conf_threshold
            wrong_dir    = (pred * target) < 0
            overconf_bad = (confident & wrong_dir).float()
            penalty      = overconf_bad * pred.abs()
            return self.weight * penalty.mean()


    # ── Model factory ──────────────────────────────────────────────────────

    MODEL_REGISTRY = {
        "tft":         TFTScalper,
        "transformer": iTransformerScalper,
        "haelt":       HAELTHybrid,
        "mamba":       MambaScalper,
        "gnn":         GNNFromSequence,
        "expert":      EXPERTEncoder,
        "glm":         GLMBaseline,
    }

    def build_model(name: str, input_size: int, seq_len: Any | None = 60, **kwargs) -> nn.Module:
        """
        Model factory with automatic hyperparameter filtering.
        Handles both explicit kwargs and argparse.Namespace objects.
        """
        import argparse
        cls = MODEL_REGISTRY.get(name.lower())
        if cls is None:
            raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")

        # 1. Expand Namespace if passed as seq_len or in kwargs
        if isinstance(seq_len, argparse.Namespace):
            for k, v in vars(seq_len).items():
                if k not in kwargs: kwargs[k] = v
            seq_len = getattr(seq_len, "seq_len", 60)

        # 2. Extract valid hyperparameters from the model's signature
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        # Merge input_size and seq_len into kwargs if expected but not present
        if "input_size" in params and "input_size" not in kwargs:
            kwargs["input_size"] = input_size
        if "seq_len" in params and "seq_len" not in kwargs:
            kwargs["seq_len"] = seq_len if isinstance(seq_len, int) else 60

        # Handle common aliases across architectures
        _ALIASES = {
            "hidden": "hidden_size",
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

        # HAELT: halve d_model/nhead to balance LSTM and Transformer branches
        # (matches the train_gpu.py builder behavior)
        if name.lower() == "haelt":
            if "d_model" in kwargs:
                kwargs["d_model"] = kwargs["d_model"] // 2
            if "nhead" in kwargs:
                kwargs["nhead"] = max(2, kwargs["nhead"] // 2)
            if "lstm_hidden" in kwargs and "hidden_size" in kwargs:
                kwargs["lstm_hidden"] = kwargs["hidden_size"] // 2

        # Filter kwargs to only include what the constructor accepts
        valid_kwargs = {k: v for k, v in kwargs.items() if k in params}

        try:
            model = cls(**valid_kwargs)
        except TypeError as e:
            # Fallback for models with non-standard signatures (e.g. GNN wrappers)
            if "unexpected keyword argument" in str(e):
                print(f"[Model] WARN: {name} init failed with filtered kwargs, retrying minimal...")
                model = cls(input_size=input_size)
            else:
                raise e

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] {name.upper()} | {n_params:,} parameters | "
              f"applied_params={list(valid_kwargs.keys())}")
        return model


else:
    # CPU stubs when torch unavailable
    class TFTScalper:
        def __init__(self, **kw): pass
    class iTransformerScalper:
        def __init__(self, **kw): pass
    class HAELTHybrid:
        def __init__(self, **kw): pass
    class MambaScalper:
        def __init__(self, **kw): pass
    class GNNCrossAsset:
        def __init__(self, **kw): pass
    class GNNFromSequence:
        def __init__(self, **kw): pass
    class EXPERTEncoder:
        def __init__(self, **kw): pass
    class MultiTaskHead:
        def __init__(self, **kw): pass
    class MultiTaskLoss:
        def __init__(self, **kw): pass
    class MultiTaskWrapper:
        def __init__(self, **kw): pass
    class MultiPairWrapper:
        def __init__(self, **kw): pass
    class DiversityLoss:
        def __init__(self, **kw): pass
    class TemperatureScaler:
        def __init__(self, model, **kw): pass
    class OverconfidencePenalty:
        def __init__(self, **kw): pass
    MODEL_REGISTRY = {}
    MODEL_ROLES: dict = {}

    def build_model(name, input_size, seq_len=60, **kw):
        print(f"[Model] Stub for {name} (torch not installed)")
        return None

    class HuberLoss:
        def __init__(self, delta=1.0): pass

    class AsymmetricDirectionalLoss:
        def __init__(self, delta=1.0, sign_weight=2.0): pass


if __name__ == "__main__" and TORCH:
    import torch
    B, T, F_IN = 8, 60, 48
    x = torch.randn(B, T, F_IN)

    for name, Cls in [
        ("TFT",         TFTScalper),
        ("iTransformer",iTransformerScalper),
        ("HAELT",       HAELTHybrid),
        ("Mamba",       MambaScalper),
        ("EXPERT",      EXPERTEncoder),
    ]:
        try:
            m = Cls(input_size=F_IN)
            out = m(x)
            print(f"  {name:16s}: in {tuple(x.shape)} -> out {tuple(out.shape)}")
        except Exception as e:
            print(f"  {name:16s}: ERROR - {e}")

    x_seq = torch.randn(B, T, F_IN)
    gnn   = GNNFromSequence(input_size=F_IN, hidden=64, num_layers=2, dropout=0.1)
    out   = gnn(x_seq)
    print(f"  {'GNN-seq':16s}: in {tuple(x_seq.shape)} -> out {tuple(out.shape)}")
