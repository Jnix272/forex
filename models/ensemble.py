"""
models/ensemble.py
==================
Model upgrades:
  1. EnsembleMetaLearner   - weighted average of all 6 architectures
  2. UncertaintyQuantifier - MC Dropout + deep ensemble confidence intervals
  3. MultiTimeframeAttn    - hierarchical attention across 1m/5m/15m bars
  4. CausalityGNN          - Granger-causality-rewired graph network
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from common.math_utils import safe_corrcoef as _safe_corrcoef
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH = True
except ImportError:
    TORCH = False


if TORCH:

    def _base_pred_to_batch_vector(raw: object) -> torch.Tensor:
        """
        Base checkpoints are often MultiTaskWrapper, whose forward returns
        (direction_logits, return_hat, confidence). The meta-learner stacks
        scalar predictions per model - use return_hat for multitask models,
        and a signed direction score for CE-only (B, 3) logits.
        """
        if isinstance(raw, (tuple, list)):
            if len(raw) == 0:
                raise TypeError(f"EnsembleMetaLearner: empty tensor tuple from base model, got {type(raw)}")
            first = raw[0]
            multitask = (
                isinstance(first, torch.Tensor)
                and (
                    (first.dim() >= 2 and first.shape[-1] == 3)  # old 3-class CE head
                    or first.dim() == 1  # CPAR: direction_logit is already (B,) scalar
                )
                and len(raw) >= 3
                and isinstance(raw[1], torch.Tensor)
            )
            t = raw[1] if multitask else first

            # Recursive unwrap for nested tuples (e.g. Ensemble inside Ensemble or custom wrappers)
            if isinstance(t, (tuple, list)):
                return _base_pred_to_batch_vector(t)
        elif isinstance(raw, torch.Tensor):
            t = raw
        else:
            # Final fallback: attempt to convert to tensor
            try:
                t = torch.as_tensor(raw)
            except Exception:
                raise TypeError(
                    f"EnsembleMetaLearner: expected Tensor or tuple of tensors from base model, got {type(raw)}"
                )

        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)

        if t.dim() > 1 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        # CE direction logits (B, 3): sell/hold/buy → signed score (buy − sell)  # noqa: RUF003
        # Never flatten to (B*3,) - that breaks meta stacking.
        elif t.dim() == 2 and t.shape[-1] == 3:
            t = t[:, -1] - t[:, 0]
        elif t.dim() > 1:
            # Unexpected multi-dim head: reduce last axis (mean) to keep batch vector
            t = t.mean(dim=tuple(range(1, t.dim())))
        return t.reshape(-1)

    class TemporalAttentionPooling(nn.Module):
        """
        Temporal attention pooling over sequence (B, S, F).
        Computes normalized attention weights over all S bars, pools features,
        and concatenates with the current (last) bar to preserve both temporal
        context and immediate microstructure.
        """

        def __init__(self, hidden: int = 64, out_dim: int = 32):
            super().__init__()
            self.proj = nn.LazyLinear(hidden)
            self.query = nn.Parameter(torch.randn(hidden))
            self.out_proj = nn.LazyLinear(out_dim)
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor, return_attention: bool = False):
            # Normalize feature dimension so macro features (e.g. volume ~ 10^5) don't dominate
            x_norm = F.layer_norm(x, (x.shape[-1],))
            if x.dim() == 2:
                out = self.out_proj(self.act(self.proj(x_norm)))
                if return_attention:
                    return out, torch.ones(x.shape[0], 1, device=x.device)
                return out
            # x: (B, S, F)
            h = torch.tanh(self.proj(x_norm))  # (B, S, hidden)
            scale = max(float(h.shape[-1]) ** 0.5, 1e-4)
            scores = torch.matmul(h, self.query) / scale  # (B, S)
            weights = torch.softmax(scores, dim=1)  # (B, S)
            pooled = (weights.unsqueeze(-1) * h).sum(dim=1)  # (B, hidden)
            last_bar = x_norm[:, -1, :]  # (B, F)
            combined = torch.cat([pooled, last_bar], dim=-1)  # (B, hidden + F)
            out = self.out_proj(combined)
            if return_attention:
                return out, weights
            return out

    # ── 1. ENSEMBLE META-LEARNER ──────────────────────────────────────────────

    class EnsembleMetaLearner(nn.Module):
        """
        Learned ensemble of the 6 base architectures.
        Each base model produces a scalar prediction; the meta-learner
        learns dynamic weights conditioned on the current market features,
        so it can down-weight models that underperform in the current regime.

        Stacking strategy:
          - Base model outputs are concatenated with a context vector
          - A small attention network assigns weights per model per bar
          - Final output = softmax-weighted sum of base predictions
        """

        def __init__(
            self,
            base_models: list[nn.Module],
            context_dim: int = 32,
            hidden: int = 64,
            base_names: list[str] | None = None,
            base_seq_lens: list[int] | None = None,
        ):
            super().__init__()
            self.bases = nn.ModuleList(base_models)
            self.n_models = len(base_models)
            self._base_names = tuple(base_names) if base_names else None
            self._base_seq_lens = (
                tuple(int(s) for s in base_seq_lens)
                if base_seq_lens and len(base_seq_lens) == len(base_models)
                else None
            )

            # Context encoder: temporal attention pooling across sequence
            self.context_enc = TemporalAttentionPooling(hidden=hidden, out_dim=context_dim)
            # Meta-network: normalized context + n_model predictions -> weights
            self.meta_norm = nn.LayerNorm(context_dim + self.n_models)
            self.meta = nn.Sequential(
                nn.Linear(context_dim + self.n_models, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.n_models),
            )
            # Initialize output layer to zero so initial ensemble weights start uniform (1/N)
            nn.init.zeros_(self.meta[2].weight)
            nn.init.zeros_(self.meta[2].bias)

        def _base_input(self, x: torch.Tensor, idx: int) -> torch.Tensor:
            if self._base_seq_lens is None:
                return x
            seq_len = self._base_seq_lens[idx]
            if seq_len > 0 and x.dim() >= 3 and x.shape[1] > seq_len:
                return x[:, -seq_len:, :]
            return x

        def unfreeze_base_heads(self, unfreeze: bool = True) -> list[nn.Parameter]:
            """Unfreeze only the final projection head of each base model for diversity fine-tuning."""
            self._base_heads_unfrozen = bool(unfreeze)
            unfrozen_params = []
            for m in self.bases:
                for p in m.parameters():
                    p.requires_grad = False
                if unfreeze:
                    head_found = False
                    if hasattr(m, "task_head"):
                        for p in m.task_head.parameters():
                            p.requires_grad = True
                            unfrozen_params.append(p)
                        head_found = True
                    elif hasattr(m, "backbone") and hasattr(m.backbone, "task_head"):
                        for p in m.backbone.task_head.parameters():
                            p.requires_grad = True
                            unfrozen_params.append(p)
                        head_found = True
                    elif hasattr(m, "head"):
                        for p in m.head.parameters():
                            p.requires_grad = True
                            unfrozen_params.append(p)
                        head_found = True
                    if not head_found:
                        linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
                        if linears:
                            for p in linears[-1].parameters():
                                p.requires_grad = True
                                unfrozen_params.append(p)
            return unfrozen_params

        def _get_base_preds(self, x: torch.Tensor) -> torch.Tensor:
            if getattr(self, "_base_heads_unfrozen", False) and torch.is_grad_enabled():
                return torch.stack(
                    [_base_pred_to_batch_vector(m(self._base_input(x, i))) for i, m in enumerate(self.bases)], dim=1
                )
            with torch.no_grad():
                return torch.stack(
                    [_base_pred_to_batch_vector(m(self._base_input(x, i))) for i, m in enumerate(self.bases)], dim=1
                )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            x: (B, seq_len, n_features)
            Returns: (prediction, weights) where weights.shape = (B, n_models)
            """
            preds = self._get_base_preds(x)
            context = self.context_enc(x)  # Temporal attention pooling across full sequence
            meta_in = getattr(self, "meta_norm", nn.Identity())(torch.cat([context, preds], dim=1))
            weights = torch.softmax(self.meta(meta_in), dim=1)  # (B, n_models)
            output = (weights * preds).sum(dim=1)  # (B,)
            return output, weights

        def forward_with_preds(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Returns (output, weights, preds) for computing diversity loss during fine-tuning."""
            preds = self._get_base_preds(x)
            context = self.context_enc(x)
            meta_in = getattr(self, "meta_norm", nn.Identity())(torch.cat([context, preds], dim=1))
            weights = torch.softmax(self.meta(meta_in), dim=1)
            output = (weights * preds).sum(dim=1)
            return output, weights, preds

        def model_weights_summary(self, x: torch.Tensor) -> dict[str, float]:
            """Return avg weight per model - useful for monitoring which models dominate."""
            _, w = self.forward(x)
            w_avg = w.mean(0).detach().cpu().numpy()
            if self._base_names is not None and len(self._base_names) == len(self.bases):
                names = list(self._base_names)
            else:
                _def = ("tft", "transformer", "haelt", "mamba", "gnn", "expert")
                names = list(_def[: len(self.bases)])
            return {names[i]: float(w_avg[i]) for i in range(len(self.bases))}

        def diversity_loss(self, preds: torch.Tensor) -> torch.Tensor:
            """
            Mean pairwise Pearson correlation of base-model outputs.
            Returns a scalar in [-1, +1]; lower = more diverse ensemble.

            Minimizing this in the meta-learner's training loss discourages the
            base models from making identical predictions (which would make the
            ensemble no better than a single model).

            preds: (B, n_models) - stacked raw base model outputs (before softmax)
            """
            # Standardise each model's column to zero-mean unit-variance
            p = preds - preds.mean(0, keepdim=True)
            p = p / (p.std(0, unbiased=True, keepdim=True) + 1e-8)  # (B, n_models)
            # Pearson correlation matrix via inner product (sample correlation)
            corr = (p.T @ p) / max(p.shape[0] - 1, 1)  # (n_models, n_models)
            # Average of upper-triangle (off-diagonal) elements only
            n = corr.shape[0]
            mask = torch.triu(torch.ones(n, n, device=preds.device), diagonal=1).bool()
            return corr[mask].mean()

        def predict_with_disagreement(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Returns (weighted_output, disagreement_score).
            disagreement_score is calculated as the weighted standard deviation of the base predictions.
            """
            with torch.no_grad():
                preds = torch.stack(
                    [_base_pred_to_batch_vector(m(self._base_input(x, i))) for i, m in enumerate(self.bases)], dim=1
                )  # (B, n_models)

            context = self.context_enc(x)
            meta_in = getattr(self, "meta_norm", nn.Identity())(torch.cat([context, preds], dim=1))
            weights = torch.softmax(self.meta(meta_in), dim=1)  # (B, n_models)
            output = (weights * preds).sum(dim=1)  # (B,)

            # Weighted variance: sum(w_i * (x_i - mean)^2)
            variance = (weights * (preds - output.unsqueeze(1)) ** 2).sum(dim=1)
            disagreement_score = torch.sqrt(variance + 1e-8)

            return output, disagreement_score

    class TemporalFoldEnsemble(nn.Module):
        """
        An ensemble that averages predictions across multiple identical models
        trained on different time slices (Cross-Validation Folds).

        Unlike the EnsembleMetaLearner which learns dynamic weights for different
        model architectures, the TemporalFoldEnsemble treats each fold model equally,
        using a simple uniform average (or median) to dramatically reduce variance.
        """

        def __init__(self, fold_models: list[nn.Module], use_median: bool = False):
            super().__init__()
            self.bases = nn.ModuleList(fold_models)
            self.n_models = len(fold_models)
            self.use_median = use_median

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            x: (B, seq_len, n_features)
            Returns: (prediction, standard_deviation)
            prediction shape = (B,)
            standard_deviation shape = (B,) - useful for uncertainty estimation
            """
            with torch.no_grad():
                preds = torch.stack([_base_pred_to_batch_vector(m(x)) for m in self.bases], dim=1)  # (B, n_models)

            if self.use_median:
                output, _ = preds.median(dim=1)
            else:
                output = preds.mean(dim=1)

            # Disagreement/Uncertainty (Standard Deviation across folds)
            uncertainty = preds.std(dim=1, unbiased=False)

            return output, uncertainty

    class EnsembleRiskFilter:
        """
        Deterministic policy for trade sizing based on ensemble disagreement.
        """

        def __init__(self, low_threshold: float = 0.5, high_threshold: float = 1.0):
            self.low_threshold = low_threshold
            self.high_threshold = high_threshold

        def compute_size_multiplier(self, disagreement_score: float) -> float:
            if disagreement_score < self.low_threshold:
                return 1.0
            elif disagreement_score < self.high_threshold:
                return 0.5
            else:
                return 0.0

    # ── Meta-learner training utility ────────────────────────────────────────

    def train_meta_learner(
        meta: EnsembleMetaLearner,
        loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        lr: float = 1e-3,
        diversity_weight: float = 0.1,
        unfreeze_base_heads: bool = False,
        device: str = "cpu",
        verbose: bool = True,
        checkpoint_path: str | None = None,
        checkpoint_meta: dict[str, object] | None = None,
    ) -> list[float]:
        """
        Train EnsembleMetaLearner's context encoder, meta-network, and optionally fine-tune
        unfrozen base model projection heads with diversity regularization.

        Objective when unfreeze_base_heads=True:
          L = MSE(weighted_ensemble_output, target)
            + diversity_weight x Corr(base_predictions)  # penalise base model co-linearity
            - 0.05 x H(weights)                          # maximise weight entropy
        """
        dev = torch.device(device)
        meta = meta.to(dev)
        best_loss = float("inf")
        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        latest_path = ckpt_path.with_name(f"{ckpt_path.stem}_latest{ckpt_path.suffix}") if ckpt_path else None

        def _save_meta_checkpoint(path: Path, epoch: int, loss_value: float, best: bool) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(meta.state_dict(), path)
            path.with_suffix(path.suffix + ".json").write_text(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "loss": float(loss_value),
                        "best_loss": float(best_loss),
                        "history": list(history),
                        "is_best": bool(best),
                        "meta": dict(checkpoint_meta or {}),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        # Base model head unfreezing
        unfrozen_base_params = meta.unfreeze_base_heads(unfreeze_base_heads)

        meta_norm_params = list(meta.meta_norm.parameters()) if hasattr(meta, "meta_norm") else []
        trainable = (
            list(meta.context_enc.parameters())
            + meta_norm_params
            + list(meta.meta.parameters())
            + unfrozen_base_params
        )
        opt = torch.optim.Adam(trainable, lr=lr)
        criterion = nn.MSELoss()
        history: list[float] = []

        for ep in range(epochs):
            ep_loss = 0.0
            n_batches = 0
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError(
                        f"train_meta_learner loader must yield at least (X, y); got {type(batch).__name__}"
                    )
                xb, yb = batch[0], batch[1]
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True).float()
                opt.zero_grad(set_to_none=True)

                output, weights, preds = meta.forward_with_preds(xb)

                # Task loss
                task_loss = criterion(output, yb)

                # Weight entropy (avoid collapse to single model)
                entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()

                if unfreeze_base_heads and len(meta.bases) > 1:
                    corr_loss = meta.diversity_loss(preds)
                    loss = task_loss + diversity_weight * corr_loss - 0.05 * entropy
                else:
                    loss = task_loss - diversity_weight * entropy
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()

                ep_loss += loss.item()
                n_batches += 1

            avg = ep_loss / max(n_batches, 1)
            history.append(avg)

            if ckpt_path is not None and latest_path is not None:
                improved = avg < best_loss
                if improved:
                    best_loss = avg
                _save_meta_checkpoint(latest_path, ep + 1, avg, best=False)
                if improved:
                    _save_meta_checkpoint(ckpt_path, ep + 1, avg, best=True)

            if verbose and (ep + 1) % max(1, epochs // 5) == 0:
                print(f"  [MetaTrain] Epoch {ep + 1:3d}/{epochs} | Loss: {avg:.6f}")

        return history

    # ── 2. UNCERTAINTY QUANTIFIER ─────────────────────────────────────────────

    class MCDropoutWrapper(nn.Module):
        """
        Wraps any model to enable MC Dropout inference.
        Keeps dropout active at inference time and runs N forward passes
        to estimate prediction variance (= epistemic uncertainty).

        High uncertainty -> reduce position size / skip signal.
        Low uncertainty  -> trade at full size.
        """

        def __init__(self, model: nn.Module, n_passes: int = 30):
            super().__init__()
            self.model = model
            self.n_passes = n_passes

        def _enable_dropout(self):
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        def _disable_dropout(self):
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        @torch.no_grad()
        def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Returns (mean_pred, std_pred, confidence_score).
            confidence_score = 1 - normalized_std ∈ [0, 1]
            """
            self._enable_dropout()
            try:
                preds = torch.stack(
                    [_base_pred_to_batch_vector(self.model(x)) for _ in range(self.n_passes)], dim=0
                )  # (n_passes, B)
                mean = preds.mean(0)
                std = preds.std(0, unbiased=False)
                # Use a fixed normaliser (not batch-max) so single-sample inference
                # returns a valid confidence score instead of always 0.0.
                _CONF_MAX_STD = 0.1  # empirical cap; adjust via calibration
                conf = (1.0 - (std / _CONF_MAX_STD)).clamp(0.0, 1.0)
                return mean, std, conf
            finally:
                self._disable_dropout()

    class DeepEnsembleUQ:
        """
        Deep ensemble uncertainty: train N independent models from different
        random seeds. Disagreement between models = uncertainty.
        More reliable than MC Dropout but requires N x training time.
        """

        def __init__(self, models: list[nn.Module], device: str = "cpu"):
            self.models = models
            self.device = torch.device(device)

        @torch.no_grad()
        def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            preds = torch.stack([_base_pred_to_batch_vector(m(x.to(self.device))) for m in self.models], dim=0)
            mean = preds.mean(0)
            std = preds.std(0, unbiased=False)
            # Use a fixed normaliser (not batch-max) so single-sample inference
            # returns a valid confidence score instead of always 0.0.
            _CONF_MAX_STD = 0.1  # empirical cap; adjust via calibration
            conf = (1.0 - (std / _CONF_MAX_STD)).clamp(0.0, 1.0)
            return mean, std, conf

        def confidence_filter(
            self,
            x: torch.Tensor,
            threshold: float = 0.5,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return (signal, mask) where mask=1 for high-confidence predictions."""
            mean, _, conf = self.predict(x)
            mask = (conf > threshold).float()
            return mean * mask, mask

    # ── 3. MULTI-TIMEFRAME ATTENTION ──────────────────────────────────────────────

    from common.model_utils import build_multitf_dataset, build_multitf_tensors  # noqa: F401

    class MultiTimeframeAttention(nn.Module):
        """
        Hierarchical attention across 1-min, 5-min, and 15-min bar streams.

        Each timeframe captures different signal frequencies:
          1-min  -> microstructure, order flow, scalping signal
          5-min  -> intraday momentum, MACD crossovers
          15-min -> structural support/resistance, session bias

        Architecture:
          1. Separate encoder per timeframe (shared weights to save params)
          2. Cross-timeframe attention: each timeframe attends to the others
          3. Fusion layer: concatenate attended representations -> prediction

        Data pipeline
        -------------
        Use ``common.model_utils.build_multitf_tensors`` (also re-exported
        from this module as ``build_multitf_tensors``) to produce the input list
        from a standard 1-min ``X_seq`` array without any lookahead bias::

            from common.model_utils import build_multitf_tensors
            tf_views = build_multitf_tensors(X_seq)   # [x_1m, x_5m, x_15m]
            x_list = [torch.from_numpy(v) for v in tf_views]
            pred = model(x_list)                      # (B,)
        """

        def __init__(
            self,
            input_size: int,
            d_model: int = 128,
            nhead: int = 4,
            n_tf_layers: int = 2,
            dropout: float = 0.1,
            timeframes: list[int] | None = None,  # in minutes
        ):
            if timeframes is None:
                timeframes = [1, 5, 15]
            super().__init__()
            self.tfs = timeframes

            # Shared encoder applied to each timeframe
            self.proj = nn.Linear(input_size, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, n_tf_layers, enable_nested_tensor=False)

            # Cross-timeframe attention: 1m attends to 5m and 15m context
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(d_model)

            # Fusion
            n_tf = len(timeframes)
            self.fuse = nn.Sequential(
                nn.Linear(d_model * n_tf, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
            """
            x_list: list of (B, T_i, input_size) tensors, one per timeframe.
            All T_i can differ (15-min bars will have fewer rows).
            """
            encoded = []
            for x in x_list:
                h = self.encoder(self.proj(x))
                encoded.append(h[:, -1, :])  # Take last bar embedding

            # Cross-timeframe attention: fine (1m) queries, coarse (5m, 15m) as K/V
            if len(encoded) > 1:
                query = encoded[0].unsqueeze(1)  # (B, 1, d)
                context = torch.stack(encoded[1:], dim=1)  # (B, n_tf-1, d)
                attn_out, _ = self.cross_attn(query, context, context)
                fine = self.cross_norm(encoded[0] + attn_out.squeeze(1))
                encoded[0] = fine

            fused = torch.cat(encoded, dim=-1)
            return self.fuse(fused).squeeze(-1)

    # ── 4. GRANGER CAUSALITY GNN ─────────────────────────────────────────────

    class GrangerCausalityGraph:
        """
        Compute Granger causality p-values between asset pairs and build
        a directed adjacency matrix. More principled than correlation:
        tests whether X's past *predicts* Y beyond Y's own past.

        Used to dynamically rewire the GNN cross-asset graph.
        Update every N bars (not every tick - costly computation).
        """

        def __init__(self, max_lag: int = 5, significance: float = 0.05):
            self.max_lag = max_lag
            self.alpha = significance

        def _granger_pvalue(self, y: np.ndarray, x: np.ndarray, lag: int) -> float:
            """
            Simplified Granger test using OLS F-test.
            H0: x does not Granger-cause y.
            """
            n = len(y)
            if n < lag * 3:
                return 1.0
            try:
                from scipy.stats import f as f_dist

                # Restricted model (y regressed on own lags)
                Y = y[lag:]
                Xr = np.column_stack([y[lag - k - 1 : n - k - 1] for k in range(lag)])
                # Unrestricted model (add x lags)
                Xu = np.column_stack([Xr, *[x[lag - k - 1 : n - k - 1] for k in range(lag)]])

                def rss(X, y):
                    try:
                        b = np.linalg.lstsq(X, y, rcond=None)[0]
                        return float(((y - X @ b) ** 2).sum())
                    except (np.linalg.LinAlgError, ValueError) as e:
                        import logging

                        logging.getLogger("Ensemble").warning(f"Granger lstsq failed: {e}")
                        return -1.0

                r_rss = rss(np.column_stack([np.ones(len(Xr)), Xr]), Y)
                u_rss = rss(np.column_stack([np.ones(len(Xu)), Xu]), Y)
                df1 = lag
                df2 = len(Y) - 2 * lag - 1
                if df2 <= 0 or u_rss <= 0:
                    return 1.0
                F = ((r_rss - u_rss) / df1) / (u_rss / df2)
                return float(1 - f_dist.cdf(F, df1, df2))
            except ImportError:
                # Fallback: simple correlation-based p-value proxy
                corr = float(_safe_corrcoef(x[:-lag], y[lag:])[0, 1])
                return float(1 - abs(corr))
            except Exception:
                return 1.0

        def compute_adjacency(
            self,
            returns_df: pd.DataFrame,
            window: int = 120,
        ) -> np.ndarray:
            """
            Compute directed adjacency matrix from Granger causality tests.
            adj[i, j] = 1  if asset i Granger-causes asset j (p < alpha)
            adj[i, j] = 0  otherwise
            """
            assets = returns_df.columns.tolist()
            n = len(assets)
            adj = np.zeros((n, n), dtype=np.float32)
            data = returns_df.tail(window).fillna(0).values

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    p = self._granger_pvalue(data[:, j], data[:, i], self.max_lag)
                    if p < self.alpha:
                        adj[i, j] = 1.0

            return adj

        def to_torch(self, adj: np.ndarray, device: str = "cpu") -> torch.Tensor:
            return torch.tensor(adj, dtype=torch.float32, device=torch.device(device))

    class CausalGNNCrossAsset(nn.Module):
        """
        GNN where edges are determined by Granger causality tests rather
        than static correlation thresholds. Updated every N bars.

        Detects when asset A is about to move asset B - before price shows it.
        """

        def __init__(
            self,
            node_features: int = 32,
            hidden: int = 64,
            num_layers: int = 3,
            heads: int = 4,
            n_nodes: int = 6,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.embed = nn.Linear(node_features, hidden)
            self.layers = nn.ModuleList(
                [nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True) for _ in range(num_layers)]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
            self.head = nn.Linear(hidden * n_nodes, 1)
            self.drop = nn.Dropout(dropout)
            self.causal = GrangerCausalityGraph()
            self._adj: torch.Tensor | None = None
            self._adj_update_count = 0

        def update_adjacency(
            self,
            returns_df: pd.DataFrame,
            device: str = "cpu",
            every: int = 500,  # Update every N calls
        ):
            self._adj_update_count += 1
            if self._adj_update_count % every != 0 and self._adj is not None:
                return
            adj = self.causal.compute_adjacency(returns_df)
            self._adj = self.causal.to_torch(adj, device)

        def forward(
            self,
            x: torch.Tensor,  # (B, n_nodes, node_features)
            adj: torch.Tensor | None = None,
        ) -> torch.Tensor:
            adj = adj if adj is not None else self._adj
            h = self.embed(x)
            attn_mask = None
            if adj is not None:
                adj = adj.to(device=h.device, dtype=torch.float32)
                n = int(adj.shape[0])
                # Block i→j attention when there is no Granger edge; keep self-loops.
                attn_mask = torch.where(
                    adj > 0,
                    torch.zeros_like(adj, dtype=torch.float32),
                    torch.full_like(adj, float("-inf"), dtype=torch.float32),
                )
                eye = torch.eye(n, device=h.device, dtype=torch.bool)
                attn_mask = attn_mask.masked_fill(eye, 0.0)
            for attn, norm in zip(self.layers, self.norms, strict=False):
                out, _ = attn(h, h, h, attn_mask=attn_mask, need_weights=False)
                h = norm(h + self.drop(out))
            o = self.head(h.reshape(h.shape[0], -1))
            return o.squeeze(-1)

else:

    class _FallbackEnsembleMetaLearner:
        def __init__(self, **kw):
            pass

        def predict_with_disagreement(self, *a, **kw):
            return None, None

    class _FallbackEnsembleRiskFilter:
        def __init__(self, **kw):
            pass

        def compute_size_multiplier(self, *a, **kw):
            return 1.0

    class _FallbackMCDropoutWrapper:
        def __init__(self, **kw):
            pass

    class _FallbackDeepEnsembleUQ:
        def __init__(self, **kw):
            pass

    class _FallbackMultiTimeframeAttention:
        def __init__(self, **kw):
            pass

    class _FallbackCausalGNNCrossAsset:
        def __init__(self, **kw):
            pass

    class _FallbackGrangerCausalityGraph:
        def __init__(self, **kw):
            pass

        def compute_adjacency(self, *a, **kw):
            return np.zeros((6, 6))

    class _FallbackTemporalAttentionPooling:
        def __init__(self, **kw):
            pass

    EnsembleMetaLearner = _FallbackEnsembleMetaLearner
    TemporalAttentionPooling = _FallbackTemporalAttentionPooling
    EnsembleRiskFilter = _FallbackEnsembleRiskFilter
    MCDropoutWrapper = _FallbackMCDropoutWrapper
    DeepEnsembleUQ = _FallbackDeepEnsembleUQ
    MultiTimeframeAttention = _FallbackMultiTimeframeAttention
    CausalGNNCrossAsset = _FallbackCausalGNNCrossAsset
    GrangerCausalityGraph = _FallbackGrangerCausalityGraph


if __name__ == "__main__" and TORCH:
    import pandas as pd
    import torch

    B, T, F_IN = 4, 60, 48

    # Ensemble test
    from models.architectures import HAELTHybrid, MambaScalper

    bases = [HAELTHybrid(input_size=F_IN), MambaScalper(input_size=F_IN)]
    ens = EnsembleMetaLearner(bases, context_dim=32)
    x = torch.randn(B, T, F_IN)
    out, w = ens(x)
    print(f"Ensemble: {tuple(out.shape)} | weights: {tuple(w.shape)}")

    # MC Dropout
    from models.architectures import HAELTHybrid

    m = MCDropoutWrapper(HAELTHybrid(input_size=F_IN), n_passes=10)
    mean, std, conf = m.predict_with_uncertainty(x)
    print(f"MC Dropout: mean={tuple(mean.shape)} std={tuple(std.shape)} conf={conf.mean():.3f}")

    # Multi-timeframe
    mtf = MultiTimeframeAttention(F_IN, d_model=64, nhead=4)
    x1 = torch.randn(B, 60, F_IN)  # 1-min
    x5 = torch.randn(B, 12, F_IN)  # 5-min (60/5)
    x15 = torch.randn(B, 4, F_IN)  # 15-min (60/15)
    out = mtf([x1, x5, x15])
    print(f"MultiTimeframe: {tuple(out.shape)}")

    # Granger GNN
    gc = GrangerCausalityGraph(max_lag=3, significance=0.1)
    df = pd.DataFrame(np.random.randn(200, 5), columns=["A", "B", "C", "D", "E"])
    adj = gc.compute_adjacency(df, window=100)
    print(f"Granger adj:\n{adj}")
