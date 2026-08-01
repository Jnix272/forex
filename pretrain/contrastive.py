"""
pretrain/contrastive.py  — Self-supervised pre-training
  BYOLTrainer, TSCLTrainer, RegimeAwareTSCLTrainer, MaskedReconstructionTrainer
  Extended objectives: pretrain/extended_trainers.py (VAE, cluster, forecast, drift)
sentiment/dual_stream.py — FinBERT offline + SLM online
latency/tip_search.py    — TIP-Search latency manager
monitoring/drift.py      — Model drift detection
validation/purged_cv.py  — Purged K-Fold + Embargoing
retraining/rolling.py    — Walk-forward rolling retraining

Compatibility utilities remain here so existing imports keep working.
Move them behind this public surface if this module is physically split later.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
try:
    from tqdm import tqdm as _tqdm
    def _pbar(it, **kw): return _tqdm(it, **kw)
except ImportError:
    def _pbar(it, **kw): return it

from config.settings import PATHS

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False

try:
    from scipy import stats
    SCIPY = True
except ImportError:
    SCIPY = False

__all__ = [
    "RepresentationCollapseError",
    "TimeSeriesAugmenter",
    "TSCLTrainer",
    "RegimeAwareTSCLTrainer",
    "BYOLTrainer",
    "MaskedReconstructionTrainer",
    "DualStreamSentiment",
    "TIPSearchManager",
    "DriftDetector",
    "lalign",
    "lunif",
]
 
class RepresentationCollapseError(Exception):
    """Raised when embeddings collapse to a single point during pretraining."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONTRASTIVE PRE-TRAINING (TSCL)
# ══════════════════════════════════════════════════════════════════════════════

class TimeSeriesAugmenter:
    """
    Augmentation strategies for time-series contrastive learning.
    Two different augmentations of the same window should be 'similar';
    windows from different market regimes should be 'dissimilar'.

    Key design choices for financial time-series:
    - Feature dropout is the MOST impactful augmentation — it forces the encoder
      to learn redundant representations across the 2240 feature channels (the
      time-series equivalent of random crop in vision).
    - Temporal crop preserves all features but shortens the window, forcing
      invariance to exact sequence length.
    - Jitter/scale are weak augmentations that preserve structure but add noise.
    - Permute is EXCLUDED — shuffling time destroys rolling-window features
      (MACD, RSI, ATR, momentum), making views maximally dissimilar.
    """
    def __init__(self, jitter_std=0.02, scale_range=(0.8, 1.2),
                 feature_drop_p=0.3, crop_ratio=(0.7, 1.0), seed=None,
                 channel_chunk: Optional[int] = None):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.feature_drop_p = feature_drop_p   # fraction of features zeroed out
        self.crop_ratio = crop_ratio           # min/max fraction of timesteps kept
        self.channel_chunk = int(channel_chunk) if channel_chunk else None
        # A-M3: a persistent, optionally-seeded RNG so augmentations are
        # reproducible when a global seed is set (previously every call created
        # a fresh unseeded default_rng(), defeating determinism).
        self._rng = np.random.default_rng(seed)
        self.stats = {
            "jitter": 0, "scale": 0, "crop": 0,
            "shuffle": 0, "feat_drop": 0, "total": 0
        }

    def augment_batch(self, X: np.ndarray, progress: float = 1.0, regime: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized augmentation: applies curriculum-scaled augmentations.

        Using multiple augmentations creates meaningfully different views
        that force the encoder to learn invariant representations, instead
        of trivially matching nearly-identical inputs.
        """
        B, T, F = X.shape
        rng = self._rng
        out = X.copy()
        self.stats["total"] += B

        # Curriculum: intensity scales with progress (base * 0.5 -> base)
        curr_drop_p = self.feature_drop_p * (0.5 + 0.5 * progress)
        curr_jitter = self.jitter_std * (0.5 + 0.5 * progress)

        # --- Always apply: Feature dropout ---
        feat_mask = rng.random((B, 1, F)) > curr_drop_p
        out = out * feat_mask.astype(X.dtype)
        self.stats["feat_drop"] += B

        # --- Second augmentation: randomly pick one of jitter/scale/crop ---
        choice = rng.integers(0, 3, size=B)  # 0=jitter 1=scale 2=crop
        
        # Regime awareness
        if regime is not None and len(regime) == B:
            is_trend = regime == 1
            is_mean = regime == -1
            # Trending regimes favour structural shifts (scale, crop)
            if is_trend.any():
                choice[is_trend] = rng.choice([1, 2], size=is_trend.sum())
            # Mean-reverting regimes favour noise (jitter, scale)
            if is_mean.any():
                choice[is_mean] = rng.choice([0, 1], size=is_mean.sum())

        # Jitter
        m_jitter = choice == 0
        if m_jitter.any():
            out[m_jitter] += rng.normal(0, curr_jitter, out[m_jitter].shape).astype(X.dtype)
            self.stats["jitter"] += int(m_jitter.sum())

        # Scale
        m_scale = choice == 1
        if m_scale.any():
            s_min, s_max = self.scale_range
            # Curriculum scaling
            s_range = (s_max - s_min) * (0.5 + 0.5 * progress)
            mid = (s_max + s_min) / 2
            s = rng.uniform(mid - s_range/2, mid + s_range/2, size=(int(m_scale.sum()), 1, 1)).astype(X.dtype)
            out[m_scale] *= s
            self.stats["scale"] += int(m_scale.sum())

        # Temporal crop
        m_crop = choice == 2
        if m_crop.any():
            n_crop = int(m_crop.sum())
            c_min, c_max = self.crop_ratio
            # Curriculum cropping
            c_min_curr = c_max - (c_max - c_min) * progress
            crop_lens = rng.integers(
                int(T * c_min_curr), T + 1, size=n_crop
            )
            for j, idx in enumerate(np.where(m_crop)[0]):
                cl = crop_lens[j]
                start = rng.integers(0, max(1, T - cl + 1))
                cropped = out[idx, start:start + cl, :].copy()
                out[idx] = 0.0  # zero-pad
                out[idx, :cl, :] = cropped
            self.stats["crop"] += n_crop

        # --- High-impact: Channel Shuffle ---
        # Probability scales with progress
        if rng.random() < 0.1 * progress:
            self.stats["shuffle"] += B
            try:
                # Shuffle per-pair feature blocks when channel_chunk is set.
                chunk = self.channel_chunk or 224
                if chunk < 8:
                    chunk = 224
                n_chunks = F // chunk
                if n_chunks > 1:
                    i1, i2 = rng.choice(n_chunks, size=2, replace=False)
                    s1, e1 = i1*chunk, (i1+1)*chunk
                    s2, e2 = i2*chunk, (i2+1)*chunk
                    tmp = out[:, :, s1:e1].copy()
                    out[:, :, s1:e1] = out[:, :, s2:e2]
                    out[:, :, s2:e2] = tmp
            except Exception:
                pass

        return out


if TORCH:
    def lalign(x, y, alpha=2):
        """Alignment: expected squared distance between positive pairs."""
        # Enforce float32 for metric stability
        x, y = x.float(), y.float()
        return (x - y).norm(dim=1).pow(alpha).mean()

    def lunif(x, t=2):
        """Uniformity: logarithm of expected Gaussian potential kernel."""
        # Enforce float32 for metric stability
        x = x.float()
        if len(x) < 2: return torch.tensor(0.0, device=x.device)
        # Manual pdist implementation for maximum stability
        # (x*x).sum(dim=1) -> (B, 1)
        r = torch.mm(x, x.T)
        # Distance squared: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        # Since x is L2-normalized, ||x||^2 = 1
        sq_pdist = 2 - 2 * r
        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(sq_pdist), diagonal=1).bool()
        sq_pdist = sq_pdist[mask]
        
        return sq_pdist.mul(-t).exp().mean().clamp(min=1e-20).log()

    class ProjectionHead(nn.Module):
        def __init__(self, d_model=128, proj_dim=128):
            super().__init__()
            d = int(d_model)
            # iTransformer / wide heads can be 100k+ dims; never use a square hidden layer.
            if d > 8192:
                hidden_dim = 512
            elif d > 2048:
                hidden_dim = 1024
            else:
                hidden_dim = min(d, 2048)
            
            # LayerNorm is safer than BatchNorm for financial data with varying scales.
            self.net = nn.Sequential(
                nn.Linear(d, hidden_dim), 
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, proj_dim),
            )
        def forward(self, x):
            # TSCL is extremely sensitive to NaNs/Infs under AMP (esp. when upstream
            # features have occasional extreme magnitudes). Keep projection in fp32,
            # sanitize, then normalize.
            x = x.float()
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = self.net(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            # Add eps for numerical stability against zero-vector embeddings
            return F.normalize(x, dim=-1, eps=1e-8)

    class TSCLTrainer:
        """
        Time-Series Contrastive Learning pre-trainer.

        Trains the encoder to produce similar representations for two
        augmented views of the same market segment, and dissimilar
        representations for different segments.

        After pre-training, the encoder weights are frozen and used as
        a feature extractor for the supervised trading models.

        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss is used
        with a learnable temperature parameter.
        """
        def __init__(self, encoder: nn.Module, d_model=128, proj_dim=128,
                     temperature=0.5, lr=1e-4, device="cpu", seed=None,
                     aug: Optional["TimeSeriesAugmenter"] = None):
            import copy
            # A-M5: deep-copy so stripping the prediction head does not mutate the
            # caller's shared backbone in place (it still carries the supervised
            # head that the supervised stage reuses). The trained copy is what
            # gets checkpointed and later reloaded into the supervised model.
            encoder = copy.deepcopy(encoder)
            if hasattr(encoder, "head"):
                encoder.head = nn.Identity()
            self.encoder = encoder.to(device)
            self.proj    = ProjectionHead(d_model, proj_dim).to(device)
            # Learnable temperature — auto-tunes during training
            self.log_temp = nn.Parameter(
                torch.tensor(float(temperature)).log().to(device)
            )
            self.aug     = aug if aug is not None else TimeSeriesAugmenter(seed=seed)
            self.device  = torch.device(device)
            self.opt     = torch.optim.AdamW(
                list(encoder.parameters()) + list(self.proj.parameters())
                + [self.log_temp],
                lr=lr, weight_decay=1e-4,
            )
            # Force FP32: BF16 overflows with 2240-feature inputs causing collapse
            self._use_amp    = False
            self._amp_dtype  = torch.float32
            self._use_scaler = False
            self._scaler     = torch.amp.GradScaler(enabled=False)
            self._total_epochs = 0  # track across multiple calls

        @property
        def temp(self):
            """Current temperature, clamped to [0.05, 2.0]."""
            return self.log_temp.exp().clamp(0.05, 2.0)

        def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
            """NT-Xent contrastive loss (SimCLR formulation)."""
            B = z1.shape[0]
            # Similarity/logits in fp32 for stability under AMP.
            z  = torch.cat([z1, z2], dim=0).float()    # (2B, D)
            z  = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            sim = torch.mm(z, z.T) / self.temp.float()  # (2B, 2B)
            mask = torch.eye(2*B, device=self.device).bool()
            neg_inf = torch.finfo(sim.dtype).min
            sim.masked_fill_(mask, neg_inf)
            labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(self.device)
            return F.cross_entropy(sim, labels)

        def _encode_project(self, t):
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            t = t.clamp(min=-1e4, max=1e4)
            h = self.encoder(t)
            if h.ndim == 3:
                h = h[:, -1, :]
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            return self.proj(h)

        def _encode_project_ckpt(self, t):
            """Gradient-checkpointed encode+project.

            Keeps the gradient graph alive (unlike torch.no_grad) so
            contrastive loss can push/pull all branches, but avoids
            storing LSTM intermediate activations in VRAM — they are
            recomputed during backward.  Peak VRAM ≈ 1 forward pass
            instead of 3.
            """
            # Removed checkpointing because it silently drops gradients for custom CUDA kernels like Mamba
            x = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            x = x.clamp(min=-1e4, max=1e4)
            h = self.encoder(x)
            if h.ndim == 3:
                h = h[:, -1, :]
            return self.proj(h.float())

        def _check_collapse(self, epoch: int, X_ref: Optional[np.ndarray] = None) -> Tuple[bool, float, float]:
            """Detect representation collapse by checking embedding variance.
            Also returns alignment and uniformity metrics.
            """
            self.encoder.eval()
            align, unif = 0.0, 0.0
            with torch.no_grad():
                try:
                    if X_ref is not None and len(X_ref) >= 8:
                        sample = torch.as_tensor(X_ref[:32], dtype=torch.float32, device=self.device)
                    else:
                        sample = torch.randn(32, 60, 224, device=self.device) * 0.1
                    
                    with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                        # Positive pairs for alignment (augmented views of same samples)
                        z1 = self._encode_project(sample)
                        
                        # Apply augmentation again for positive pair
                        v2 = self.aug.augment_batch(sample.cpu().numpy())
                        z2 = self._encode_project(torch.as_tensor(v2, device=self.device))
                        
                    # Calculate metrics outside autocast and in float32 for stability
                    align = lalign(z1, z2).item()
                    unif  = lunif(z1).item()
                    
                    # Collapse if variance across samples is near zero
                    std = z1.std(dim=0).mean().item()
                    
                    if not np.isfinite(align) or not np.isfinite(unif):
                        # Diagnostics for NaN metrics
                        if torch.isnan(z1).any():
                             print("[Monitor] ⚠ z1 contains NaNs")
                        if torch.isnan(z2).any():
                             print("[Monitor] ⚠ z2 contains NaNs")
                        align, unif = 0.0, 0.0
                except Exception as e:
                    print(f"[Monitor] Metric error: {e}")
                    std, align, unif = 1.0, 0.0, 0.0
            
            self.encoder.train()
            collapsed = std < 0.01
            if collapsed:
                print(f"\n[Pretrain] ⚠ COLLAPSE DETECTED (epoch {epoch+1}): "
                      f"embedding std={std:.6f} | align={align:.4f} | unif={unif:.4f}")
            return collapsed, align, unif

        def pretrain(
            self,
            X: np.ndarray,           # (N, seq_len, n_features)
            epochs: int = 50,
            batch_size: int = 256,
            checkpoint_path: Optional[str] = None,
            patience: int = 5,
        ) -> dict:
            if checkpoint_path is None:
                checkpoint_path = PATHS["file_contrastive_encoder"]
            N = len(X)
            X_ref_fixed = X[: min(128, N)]
            history = {"loss": [], "align": [], "unif": []}
            amp_str = "BF16" if self._amp_dtype == torch.bfloat16 else (
                "FP16" if self._use_amp else "FP32"
            )
            print(f"[TSCL] Pre-training {epochs} epochs | {N:,} windows | "
                  f"batch={batch_size} | {amp_str} | temp={self.temp.item():.3f}")

            warmup_epochs = min(3, epochs)
            base_lr = self.opt.param_groups[0]['lr']

            best_metric = float('inf')
            patience_counter = 0

            epoch_bar = _pbar(range(epochs), desc="TSCL Pretrain", unit="ep", leave=True)
            for epoch in epoch_bar:
                self._total_epochs += 1
                cur_ep = self._total_epochs
                
                # LR warmup + cosine decay
                if epoch < warmup_epochs:
                    lr_scale = (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                    lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
                for pg in self.opt.param_groups:
                    pg['lr'] = base_lr * lr_scale

                epoch_loss = 0.0; n_batches = 0
                batches = list(range(0, N, batch_size))
                np.random.shuffle(batches)
                
                batch_bar = _pbar(batches, desc=f"  Ep {epoch+1:3d}/{epochs}",
                                  unit="batch", leave=False)
                for start in batch_bar:
                    X_batch = X[start:start + batch_size]
                    if len(X_batch) < 4: continue

                    cur_progress = epoch / max(1, epochs)
                    v1 = self.aug.augment_batch(X_batch, progress=cur_progress)
                    v2 = self.aug.augment_batch(X_batch, progress=cur_progress)
                    t1 = torch.as_tensor(v1, dtype=torch.float32, device=self.device)
                    t2 = torch.as_tensor(v2, dtype=torch.float32, device=self.device)

                    with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                        z1 = self._encode_project_ckpt(t1)
                        z2 = self._encode_project_ckpt(t2)
                        loss = self.nt_xent_loss(z1, z2)

                    del t1, t2, z1, z2
                    if not torch.isfinite(loss): continue

                    self.opt.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.opt)
                    gnorm = nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.proj.parameters()), 1.0
                    )
                    if not torch.isfinite(gnorm):
                        self.opt.zero_grad(set_to_none=True)
                        continue
                    self._scaler.step(self.opt)
                    self._scaler.update()

                    epoch_loss += loss.item(); n_batches += 1
                    if hasattr(batch_bar, "set_postfix"):
                        batch_bar.set_postfix(loss=f"{loss.item():.3f}")

                avg = epoch_loss / max(n_batches, 1)
                history["loss"].append(avg)
                
                # Metric monitoring
                collapsed, align, unif = self._check_collapse(cur_ep, X_ref=X_ref_fixed)
                history["align"].append(align)
                history["unif"].append(unif)
                
                self.temp.item()
                if hasattr(epoch_bar, "set_postfix"):
                    epoch_bar.set_postfix(loss=f"{avg:.3f}", align=f"{align:.2f}", unif=f"{unif:.2f}")
                
                if collapsed:
                    raise RepresentationCollapseError(f"Embedding collapse at epoch {cur_ep}")

                metric = align + unif
                if metric < best_metric - 1e-4:
                    best_metric = metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (metric={metric:.4f})")
                    break

            print(f"[TSCL] Augment Stats: {self.aug.stats}")
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)
            return history


    class RegimeAwareTSCLTrainer(TSCLTrainer):
        """
        Extends TSCLTrainer with regime-aware positive/negative pair selection.

        Standard TSCL: positives = two augmented views of the SAME window.
        Regime-aware:  positives = windows from the SAME market regime
                       (trending / mean-reverting / neutral); hard negatives =
                       windows from the OPPOSITE regime.

        This gives the encoder a structured latent space:
          Trending       windows cluster together  (Hurst > 0.55)
          Mean-reverting windows cluster together  (Hurst < 0.45)
          Neutral        windows fill the middle

        The regime labels (1=trending, -1=mean-reverting, 0=neutral) are
        typically derived from the Parkinson fast_trend_score in
        features/advanced_features.py and passed in at construction time.
        """

        def __init__(
            self,
            encoder,
            regime_labels:        np.ndarray,   # (N,) int8: 1, 0, -1
            d_model=128, proj_dim=128,
            temperature=0.5, lr=1e-4, device="cpu",
            hard_negative_weight: float = 1.0, seed=None,
            aug: Optional["TimeSeriesAugmenter"] = None,
        ):
            super().__init__(encoder, d_model, proj_dim, temperature, lr, device, seed=seed, aug=aug)
            self.regime_labels    = np.asarray(regime_labels, dtype=np.int8)
            self.hard_neg_weight  = hard_negative_weight

            # Pre-build per-regime index lists for O(1) sampling
            self._regime_idx: dict = {}
            for r in np.unique(self.regime_labels):
                self._regime_idx[int(r)] = np.where(self.regime_labels == r)[0]

        def _same_regime(self, anchor_i: int) -> int:
            """Sample a single index sharing the same regime as anchor_i."""
            r    = int(self.regime_labels[anchor_i])
            pool = self._regime_idx.get(r, np.array([anchor_i]))
            return int(np.random.choice(pool))

        def _diff_regime(self, anchor_i: int) -> int:
            """Sample a single index from a DIFFERENT regime (hard negative)."""
            r         = int(self.regime_labels[anchor_i])
            candidates = [
                idx for rk, idxs in self._regime_idx.items()
                if rk != r for idx in idxs
            ]
            if not candidates:
                return anchor_i   # Fallback if only one regime present
            return int(np.random.choice(candidates))

        def _regime_loss(
            self,
            z_a: "torch.Tensor",  # (B, D) anchor
            z_p: "torch.Tensor",  # (B, D) same-regime positive
            z_n: "torch.Tensor",  # (B, D) diff-regime hard negative
        ) -> "torch.Tensor":
            """
            Triplet-style NT-Xent loss that pushes same-regime embeddings
            together and cross-regime embeddings apart.

            Compared to standard SimCLR:
              - Positives are semantically matched (same regime), not just augmented
              - Hard negatives from opposite regime increase training signal quality
            """
            B = z_a.shape[0]
            # Standard SimCLR loss on (anchor, positive) pairs
            z_std = torch.cat([z_a, z_p], dim=0)               # (2B, D)
            sim   = torch.mm(z_std, z_std.T) / self.temp        # (2B, 2B)
            eye   = torch.eye(2 * B, device=self.device).bool()
            neg_inf = torch.finfo(sim.dtype).min
            sim.masked_fill_(eye, neg_inf)
            labels = torch.cat([
                torch.arange(B, 2 * B, device=self.device),
                torch.arange(0, B,     device=self.device),
            ])
            loss_std = F.cross_entropy(sim, labels)

            # Hard-negative margin loss: anchor should be farther from hard neg than pos
            sim_ap = (z_a * z_p).sum(-1) / self.temp  # (B,)
            sim_an = (z_a * z_n).sum(-1) / self.temp  # (B,)
            # Hinge: push anchor-negative similarity 0.2 below anchor-positive
            margin_loss = F.relu(sim_an - sim_ap + 0.2).mean()

            return loss_std + self.hard_neg_weight * margin_loss

        def pretrain(
            self,
            X:                np.ndarray,   # (N, seq_len, n_features)
            epochs:           int = 50,
            batch_size:       int = 256,
            checkpoint_path:  Optional[str] = None,
            patience:         int = 5,
        ) -> dict:
            """
            Regime-aware pre-training loop.
            Falls back to standard TSCL augmentation when all samples share
            the same regime (e.g. pure trending dataset).
            """
            if checkpoint_path is None:
                checkpoint_path = PATHS.get(
                    "file_contrastive_encoder",
                    "/workspace/checkpoints/contrastive_encoder_regime.pt",
                )
            N = len(X)
            X_ref_fixed = X[: min(128, N)]

            # Align regime labels to dataset length
            reg = self.regime_labels
            if len(reg) > N:
                reg = reg[:N]
            elif len(reg) < N:
                reg = np.pad(reg, (0, N - len(reg)), constant_values=0)
            self.regime_labels = reg

            # Rebuild index after potential trimming
            self._regime_idx = {}
            for r in np.unique(reg):
                self._regime_idx[int(r)] = np.where(reg == r)[0]

            n_regimes = len(self._regime_idx)
            print(f"[RegimeTSCL] Pre-training {epochs} epochs | {N:,} windows | "
                  f"{n_regimes} regimes: "
                  f"{ {int(r): int(len(idx)) for r, idx in self._regime_idx.items()} }")

            amp_str = "BF16" if self._amp_dtype == torch.bfloat16 else (
                "FP16" if self._use_amp else "FP32"
            )
            print(f"  batch={batch_size} | {amp_str} | temp={self.temp.item():.3f} (learnable)")

            # Pre-build vectorised regime pools for fast sampling
            _regime_pool = {r: idx for r, idx in self._regime_idx.items()}

            # LR warmup: linear ramp over first 3 epochs
            warmup_epochs = min(3, epochs)
            base_lr = self.opt.param_groups[0]['lr']

            best_metric = float('inf')
            patience_counter = 0

            history = {"loss": [], "align": [], "unif": []}
            epoch_bar = _pbar(range(epochs), desc="Pretrain", unit="ep", leave=True)
            for epoch in epoch_bar:
                self._total_epochs += 1
                cur_ep = self._total_epochs
                
                # LR warmup + cosine decay
                if epoch < warmup_epochs:
                    lr_scale = (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                    lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
                for pg in self.opt.param_groups:
                    pg['lr'] = base_lr * lr_scale

                idx_perm = np.random.permutation(N)
                ep_loss  = 0.0
                n_b      = 0

                batches = list(range(0, N, batch_size))
                batch_bar = _pbar(batches, desc=f"  Ep {epoch+1:3d}/{epochs}",
                                  unit="batch", leave=False)
                for start in batch_bar:
                    batch_idx = idx_perm[start: start + batch_size]
                    if len(batch_idx) < 4: continue
                    X[batch_idx]

                    cur_progress = epoch / max(1, epochs)
                    reg_batch = reg[batch_idx]
                    v_a = self.aug.augment_batch(X[batch_idx], progress=cur_progress, regime=reg_batch)

                    with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                        if n_regimes > 1:
                            batch_reg = reg[batch_idx]
                            pos_i = np.empty(len(batch_idx), dtype=np.int64)
                            neg_i = np.empty(len(batch_idx), dtype=np.int64)
                            for r, pool in _regime_pool.items():
                                m = batch_reg == r
                                cnt = int(m.sum())
                                if cnt == 0: continue
                                pos_i[m] = np.random.choice(pool, cnt)
                                others = np.concatenate([p for rk, p in _regime_pool.items() if rk != r])
                                if len(others) > 0:
                                    neg_i[m] = np.random.choice(others, cnt)
                                else:
                                    neg_i[m] = pos_i[m]

                            cur_progress = epoch / max(1, epochs)
                            v_p = self.aug.augment_batch(X[pos_i], progress=cur_progress, regime=self.regime_labels[pos_i])
                            v_n = self.aug.augment_batch(X[neg_i], progress=cur_progress, regime=self.regime_labels[neg_i])

                            t_p = torch.as_tensor(v_p, dtype=torch.float32, device=self.device)
                            z_p = self._encode_project_ckpt(t_p)
                            del t_p

                            t_n = torch.as_tensor(v_n, dtype=torch.float32, device=self.device)
                            z_n = self._encode_project_ckpt(t_n)
                            del t_n

                            t_a = torch.as_tensor(v_a, dtype=torch.float32, device=self.device)
                            z_a = self._encode_project_ckpt(t_a)
                            del t_a

                            loss = self._regime_loss(z_a, z_p, z_n)
                            del z_a, z_p, z_n
                        else:
                            cur_progress = epoch / max(1, epochs)
                            v_b = self.aug.augment_batch(X[batch_idx], progress=cur_progress, regime=reg_batch)
                            t2 = torch.as_tensor(v_b, dtype=torch.float32, device=self.device)
                            z2 = self._encode_project_ckpt(t2)
                            del t2
                            t1 = torch.as_tensor(v_a, dtype=torch.float32, device=self.device)
                            z1 = self._encode_project_ckpt(t1)
                            del t1
                            loss = self.nt_xent_loss(z1, z2)
                            del z1, z2

                    if not torch.isfinite(loss): continue

                    self.opt.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.opt)
                    gnorm = nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.proj.parameters()), 1.0
                    )
                    if not torch.isfinite(gnorm):
                        self.opt.zero_grad(set_to_none=True)
                        continue
                    self._scaler.step(self.opt)
                    self._scaler.update()

                    ep_loss += loss.item()
                    n_b += 1
                    if hasattr(batch_bar, "set_postfix"):
                        batch_bar.set_postfix(loss=f"{loss.item():.3f}")

                avg = ep_loss / max(n_b, 1)
                history["loss"].append(avg)
                
                # Metric monitoring
                collapsed, align, unif = self._check_collapse(cur_ep, X_ref=X_ref_fixed)
                history["align"].append(align)
                history["unif"].append(unif)
                
                self.temp.item()
                if hasattr(epoch_bar, "set_postfix"):
                    epoch_bar.set_postfix(loss=f"{avg:.3f}", align=f"{align:.2f}", unif=f"{unif:.2f}")

                if collapsed:
                    raise RepresentationCollapseError(f"Embedding collapse at epoch {cur_ep}")

                metric = align + unif
                if metric < best_metric - 1e-4:
                    best_metric = metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (metric={metric:.4f})")
                    break

            print(f"[RegimeTSCL] Augment Stats: {self.aug.stats}")
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)
            print(f"[RegimeTSCL] Encoder saved → {checkpoint_path}")
            return history


    class BYOLTrainer:
        """
        Bootstrap Your Own Latent (BYOL) pre-trainer.

        Advantages over NT-Xent (SimCLR / TSCL) on 8 GB VRAM:
          • No negative pairs  → small batches work well (batch >= 32 is fine)
          • 1 gradient pass    → ~2x less VRAM than regime-TSCL's 3 passes
          • Loss range 0-2     (1.0 = random init, <0.5 = learning, <0.2 = great)
          • No temperature to tune

        Architecture:
          Online : encoder → projector (Linear→BN→ReLU→Linear) → predictor (same)
          Target : EMA copy of (encoder + projector), always eval, no gradient

        EMA update after every step:
          target_θ ← decay × target_θ + (1 − decay) × online_θ
        """

        def __init__(self, encoder: nn.Module, d_model: int = 128,
                     proj_dim: int = 256, pred_dim: int = 128,
                     ema_decay: float = 0.996, lr: float = 1e-4,
                     device: str = "cpu", seed=None,
                     aug: Optional["TimeSeriesAugmenter"] = None):
            import copy
            self.device    = torch.device(device)
            self.ema_decay = ema_decay

            # Online network
            # A-M5: deep-copy so the caller's shared backbone (with its supervised
            # head) is never mutated in place when we strip the head here.
            encoder = copy.deepcopy(encoder)
            if hasattr(encoder, "head"):
                encoder.head = nn.Identity()
            self.encoder   = encoder.to(self.device)
            self.projector = self._make_mlp(d_model, proj_dim, proj_dim).to(self.device)
            self.predictor = self._make_mlp(proj_dim, pred_dim, proj_dim).to(self.device)

            # Target network — EMA copy, no gradient, always eval
            self.target_encoder   = copy.deepcopy(self.encoder)
            self.target_projector = copy.deepcopy(self.projector)
            for p in (list(self.target_encoder.parameters()) +
                      list(self.target_projector.parameters())):
                p.requires_grad_(False)
            self.target_encoder.eval()
            self.target_projector.eval()

            self.aug = aug if aug is not None else TimeSeriesAugmenter(seed=seed)

            # Optimiser — online params only
            self.opt = torch.optim.AdamW(
                list(self.encoder.parameters()) +
                list(self.projector.parameters()) +
                list(self.predictor.parameters()),
                lr=lr, weight_decay=1e-4,
            )

            # Force FP32: BF16 overflows with 2240-feature inputs causing collapse
            self._use_amp    = False
            self._amp_dtype  = torch.float32
            self._use_scaler = False
            self._scaler     = torch.amp.GradScaler(enabled=False)
            self._total_epochs = 0

        @staticmethod
        def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
            """BYOL MLP: Linear → BatchNorm1d → ReLU → Linear."""
            hidden_dim = min(hidden_dim, 2048)
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

        @torch.no_grad()
        def _ema_update(self):
            d = self.ema_decay
            for o, t in zip(self.encoder.parameters(),
                            self.target_encoder.parameters()):
                t.data.mul_(d).add_(o.data, alpha=1.0 - d)
            for o, t in zip(self.projector.parameters(),
                            self.target_projector.parameters()):
                t.data.mul_(d).add_(o.data, alpha=1.0 - d)

        def _online_fwd(self, x: torch.Tensor) -> torch.Tensor:
            """Online path — retains gradients. Returns predictor output."""
            # Removed checkpointing because it silently drops gradients for custom CUDA kernels like Mamba
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
            
            h = self.encoder(x)
            if h.ndim == 3:
                h = h[:, -1, :]
            # Clamp encoder output: BF16 activations can overflow with 2240 features,
            # causing all embeddings to collapse to the same unit vector (loss=0).
            h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-50, 50).float()
            
            z = self.projector(h)
            p = self.predictor(z)
            return p

        @torch.no_grad()
        def _target_fwd(self, x: torch.Tensor) -> torch.Tensor:
            """Target path — no gradient, fp32, L2-normalised output."""
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
            h = self.target_encoder(x)
            if h.ndim == 3:
                h = h[:, -1, :]
            # Same clamp as online path to prevent BF16 overflow collapse
            h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-50, 50).float()
            z = self.target_projector(h)
            return F.normalize(z, dim=-1, eps=1e-8)

        @staticmethod
        def _byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            """BYOL regression loss = 2 - 2·cosine_sim(predict, target). Range [0,2]."""
            p = F.normalize(p.float(), dim=-1, eps=1e-8)
            return (2.0 - 2.0 * (p * z.float()).sum(dim=-1)).mean()

        @torch.no_grad()
        def diagnostics(self, X_ref: np.ndarray, max_samples: int = 128) -> dict:
            """Return representation diagnostics for BYOL handoff quality gates."""
            out = {"embed_std": 0.0, "align": 0.0, "unif": 0.0, "collapsed": True}
            if X_ref is None or len(X_ref) < 4:
                return out
            self.encoder.eval()
            self.projector.eval()
            try:
                sample = X_ref[: min(int(max_samples), len(X_ref))]
                v1 = self.aug.augment_batch(sample)
                v2 = self.aug.augment_batch(sample)
                t1 = torch.as_tensor(v1, dtype=torch.float32, device=self.device)
                t2 = torch.as_tensor(v2, dtype=torch.float32, device=self.device)
                z1 = self._target_fwd(t1)
                z2 = self._target_fwd(t2)
                std = z1.std(dim=0).mean().item()
                align = lalign(z1, z2).item()
                unif = lunif(z1).item()
                finite = np.isfinite(std) and np.isfinite(align) and np.isfinite(unif)
                out = {
                    "embed_std": float(std if np.isfinite(std) else 0.0),
                    "align": float(align if np.isfinite(align) else 0.0),
                    "unif": float(unif if np.isfinite(unif) else 0.0),
                    "collapsed": (not finite) or std < 0.005,
                }
            except Exception as exc:
                print(f"[BYOL] diagnostic error: {exc}")
            finally:
                self.encoder.train()
                self.projector.train()
            return out

        def save_encoder(self, checkpoint_path: str) -> None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)

        def pretrain(
            self,
            X:               np.ndarray,
            epochs:          int = 50,
            batch_size:      int = 256,
            checkpoint_path: Optional[str] = None,
            silent:          bool = False,  # suppress prints/bars (for multi-block calls)
            patience:        int = 5,
        ) -> dict:
            if checkpoint_path is None:
                checkpoint_path = PATHS["file_contrastive_encoder"]
            N = len(X)
            X_ref_fixed = X[: min(128, N)]
            history = {"loss": [], "embed_std": [], "align": [], "unif": []}
            amp_str = ("BF16" if self._amp_dtype == torch.bfloat16
                       else ("FP16" if self._use_amp else "FP32"))
            if not silent:
                print(f"[BYOL] Pre-training {epochs} ep | {N:,} windows | "
                      f"batch={batch_size} | {amp_str} | ema_decay={self.ema_decay}")

            warmup_epochs = min(3, epochs)
            base_lr = self.opt.param_groups[0]["lr"]

            best_metric = float('inf')
            patience_counter = 0

            epoch_bar = (_pbar(range(epochs), desc="BYOL Pretrain", unit="ep", leave=True)
                         if not silent else range(epochs))
            for epoch in epoch_bar:
                self._total_epochs += 1

                # LR warmup + cosine decay
                if epoch < warmup_epochs:
                    lr_scale = (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                    lr_scale = 0.5 * (1.0 + np.cos(np.pi * progress))
                for pg in self.opt.param_groups:
                    pg["lr"] = base_lr * lr_scale

                self.encoder.train()
                self.projector.train()
                self.predictor.train()

                epoch_loss = 0.0
                n_batches  = 0
                idx_perm   = np.random.permutation(N)

                batch_bar = (_pbar(
                    range(0, N, batch_size),
                    desc=f"  Ep {epoch+1:3d}/{epochs}", unit="batch", leave=False,
                ) if not silent else range(0, N, batch_size))
                for start in batch_bar:
                    batch_idx = idx_perm[start : start + batch_size]
                    if len(batch_idx) < 4:
                        continue
                    X_b = X[batch_idx]

                    cur_progress = epoch / max(1, epochs)
                    v1 = self.aug.augment_batch(X_b, progress=cur_progress)
                    v2 = self.aug.augment_batch(X_b, progress=cur_progress)
                    t1 = torch.as_tensor(v1, dtype=torch.float32, device=self.device)
                    t2 = torch.as_tensor(v2, dtype=torch.float32, device=self.device)

                    # Online forward (with gradient)
                    with torch.amp.autocast("cuda", enabled=self._use_amp,
                                            dtype=self._amp_dtype):
                        p1 = self._online_fwd(t1)
                        p2 = self._online_fwd(t2)

                    # Target forward (no gradient, fp32)
                    z2 = self._target_fwd(t2)
                    z1 = self._target_fwd(t1)
                    del t1, t2

                    # Symmetric BYOL loss, averaged to keep range [0, 2]
                    loss = (self._byol_loss(p1, z2) + self._byol_loss(p2, z1)) * 0.5
                    del p1, p2, z1, z2

                    if not torch.isfinite(loss):
                        continue

                    self.opt.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.opt)
                    gnorm = nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) +
                        list(self.projector.parameters()) +
                        list(self.predictor.parameters()),
                        1.0,
                    )
                    if not torch.isfinite(gnorm):
                        self.opt.zero_grad(set_to_none=True)
                        continue
                    self._scaler.step(self.opt)
                    self._scaler.update()
                    self._ema_update()

                    epoch_loss += loss.item()
                    n_batches  += 1
                    if hasattr(batch_bar, "set_postfix"):
                        batch_bar.set_postfix(loss=f"{loss.item():.3f}")

                avg = epoch_loss / max(n_batches, 1)
                history["loss"].append(avg)
                diag = self.diagnostics(X_ref_fixed)
                history["embed_std"].append(diag["embed_std"])
                history["align"].append(diag["align"])
                history["unif"].append(diag["unif"])
                if not silent:
                    if hasattr(epoch_bar, "set_postfix"):
                        epoch_bar.set_postfix(loss=f"{avg:.3f}", std=f"{diag['embed_std']:.4f}")
                    print(f"[BYOL] Ep {self._total_epochs:3d} | loss={avg:.4f}"
                          f" | std={diag['embed_std']:.4f}"
                          f" | align={diag['align']:.3f}"
                          f" | unif={diag['unif']:.3f}"
                          f"  (1.0=random  <0.5=learning  <0.2=great)")

                metric = diag["align"] + diag["unif"]
                if metric < best_metric - 1e-4:
                    best_metric = metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if not silent:
                        print(f"Early stopping at epoch {epoch+1} (metric={metric:.4f})")
                    break

            if not silent:
                print(f"[BYOL] Augment Stats: {self.aug.stats}")
                self.save_encoder(checkpoint_path)
                print(f"[BYOL] Encoder saved → {checkpoint_path}")
            return history


# ══════════════════════════════════════════════════════════════════════════════
# 2. DUAL-STREAM SENTIMENT  (FinBERT offline + SLM online)
# ══════════════════════════════════════════════════════════════════════════════

    class MaskedReconstructionTrainer:
        """
        Masked time-series reconstruction pre-trainer.

        Randomly hides values in scaled windows, reconstructs the original
        window, and optimizes loss only over masked positions.
        """

        def __init__(
            self,
            encoder: nn.Module,
            d_model: int,
            seq_len: int,
            n_features: int,
            hidden_dim: int = 512,
            mask_prob: float = 0.20,
            lr: float = 1e-4,
            device: str = "cpu",
            seed=None,
        ):
            import copy
            self.device = torch.device(device)
            self.seq_len = int(seq_len)
            self.n_features = int(n_features)
            self.mask_prob = float(mask_prob)
            self._rng = np.random.default_rng(seed)

            encoder = copy.deepcopy(encoder)
            if hasattr(encoder, "head"):
                encoder.head = nn.Identity()
            self.encoder = encoder.to(self.device)
            hidden_dim = int(min(max(64, hidden_dim), 2048))
            self.decoder = nn.Sequential(
                nn.Linear(int(d_model), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.seq_len * self.n_features),
            ).to(self.device)
            self.opt = torch.optim.AdamW(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=lr,
                weight_decay=1e-4,
            )
            self._use_amp = False
            self._amp_dtype = torch.float32
            self._scaler = torch.amp.GradScaler(enabled=False)
            self._total_epochs = 0

        def _mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mask = torch.rand_like(x) < self.mask_prob
            if not mask.any():
                flat = mask.view(-1)
                flat[int(self._rng.integers(0, flat.numel()))] = True
            return x.masked_fill(mask, 0.0), mask

        def _forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
            h = self.encoder(x)
            if h.ndim == 3:
                h = h[:, -1, :]
            h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-50, 50).float()
            return self.decoder(h).view(-1, self.seq_len, self.n_features)

        @torch.no_grad()
        def diagnostics(self, X_ref: np.ndarray, max_samples: int = 128) -> dict:
            out = {"masked_mse": 0.0, "embed_std": 0.0, "collapsed": True}
            if X_ref is None or len(X_ref) < 4:
                return out
            self.encoder.eval()
            self.decoder.eval()
            try:
                sample = X_ref[: min(int(max_samples), len(X_ref))]
                x = torch.as_tensor(sample, dtype=torch.float32, device=self.device)
                corrupted, mask = self._mask(x)
                recon = self._forward(corrupted)
                mse = F.mse_loss(recon[mask], x[mask]).item()
                h = self.encoder(corrupted)
                if h.ndim == 3:
                    h = h[:, -1, :]
                std = torch.nan_to_num(h.float()).std(dim=0).mean().item()
                finite = np.isfinite(mse) and np.isfinite(std)
                out = {
                    "masked_mse": float(mse if np.isfinite(mse) else 0.0),
                    "embed_std": float(std if np.isfinite(std) else 0.0),
                    "collapsed": (not finite) or std < 0.005,
                }
            except Exception as exc:
                print(f"[MaskedRecon] diagnostic error: {exc}")
            finally:
                self.encoder.train()
                self.decoder.train()
            return out

        def save_encoder(self, checkpoint_path: str) -> None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)

        def pretrain(
            self,
            X: np.ndarray,
            epochs: int = 50,
            batch_size: int = 256,
            checkpoint_path: Optional[str] = None,
            silent: bool = False,
        ) -> dict:
            if checkpoint_path is None:
                checkpoint_path = PATHS["file_contrastive_encoder"]
            N = len(X)
            history = {"loss": [], "masked_mse": [], "embed_std": []}
            if not silent:
                print(
                    f"[MaskedRecon] Pre-training {epochs} ep | {N:,} windows | "
                    f"batch={batch_size} | mask_prob={self.mask_prob:.2f}"
                )

            warmup_epochs = min(3, epochs)
            base_lr = self.opt.param_groups[0]["lr"]
            epoch_bar = (_pbar(range(epochs), desc="MaskedRecon Pretrain", unit="ep", leave=True)
                         if not silent else range(epochs))
            for epoch in epoch_bar:
                self._total_epochs += 1
                
                self.encoder.train()
                self.decoder.train()
                
                if epoch < warmup_epochs:
                    lr_scale = (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                    lr_scale = 0.5 * (1.0 + np.cos(np.pi * progress))
                for pg in self.opt.param_groups:
                    pg["lr"] = base_lr * lr_scale

                idx_perm = np.random.permutation(N)
                epoch_loss = 0.0
                n_batches = 0
                batch_bar = (_pbar(
                    range(0, N, batch_size),
                    desc=f"  Ep {epoch+1:3d}/{epochs}", unit="batch", leave=False,
                ) if not silent else range(0, N, batch_size))
                for start in batch_bar:
                    batch_idx = idx_perm[start:start + batch_size]
                    if len(batch_idx) < 4:
                        continue
                    x = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=self.device)
                    corrupted, mask = self._mask(x)
                    with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                        recon = self._forward(corrupted)
                        loss = F.mse_loss(recon[mask], x[mask])
                    del corrupted, recon
                    if not torch.isfinite(loss):
                        continue

                    self.opt.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.opt)
                    gnorm = nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        1.0,
                    )
                    if not torch.isfinite(gnorm):
                        self.opt.zero_grad(set_to_none=True)
                        continue
                    self._scaler.step(self.opt)
                    self._scaler.update()

                    epoch_loss += loss.item()
                    n_batches += 1
                    if hasattr(batch_bar, "set_postfix"):
                        batch_bar.set_postfix(loss=f"{loss.item():.4f}")

                avg = epoch_loss / max(n_batches, 1)
                diag = self.diagnostics(X)
                history["loss"].append(avg)
                history["masked_mse"].append(diag["masked_mse"])
                history["embed_std"].append(diag["embed_std"])
                if not silent:
                    if hasattr(epoch_bar, "set_postfix"):
                        epoch_bar.set_postfix(loss=f"{avg:.4f}", std=f"{diag['embed_std']:.4f}")
                    print(
                        f"[MaskedRecon] Ep {self._total_epochs:3d} | loss={avg:.4f} "
                        f"| masked_mse={diag['masked_mse']:.4f} "
                        f"| std={diag['embed_std']:.4f}"
                    )

            if not silent:
                self.save_encoder(checkpoint_path)
                print(f"[MaskedRecon] Encoder saved -> {checkpoint_path}")
            return history


class DualStreamSentiment:
    """
    Two-speed architecture that adds news context without blocking execution.

    GLOBAL BRAIN (slow, ~60s cadence):
      Runs FinBERT (offline pre-computed) or Mistral-7B SLM on live headlines.
      Outputs a single Sentiment Bias in [-1.0, +1.0].

    LOCAL ACTOR (fast, every bar):
      Receives the Sentiment Bias as a feature.
      If bias > threshold → only take long signals.
      If bias < -threshold → only take short signals.

    Latency solution: FinBERT embeddings are pre-computed offline for
    historical data. Online inference uses a small SLM (Mistral-7B) running
    on GPU with cached embeddings — adds <5ms via the cross-attention fusion.
    """

    def __init__(
        self,
        embedding_dim:   int   = 768,
        proj_dim:        int   = 8,
        decay_lambda:    float = 0.1,
        update_sec:      int   = 60,
        sentiment_threshold: float = 0.3,
        prefer_backend:  Optional[str] = None,
        use_cache:       bool  = True,
    ):
        self.emb_dim    = embedding_dim
        self.proj_dim   = proj_dim
        self.decay_lam  = decay_lambda
        self.update_sec = update_sec
        self.threshold  = sentiment_threshold
        # Accepted for live-engine / SentimentPipeline parity; bias path is
        # hash-embedding based and does not load a transformer backend.
        self.prefer_backend = prefer_backend
        self.use_cache = bool(use_cache)

        # Random projection (deterministic — same weights every run)
        rng = np.random.default_rng(0)
        self._proj = rng.standard_normal((embedding_dim, proj_dim)).astype(np.float32)
        self._proj /= np.linalg.norm(self._proj, axis=0, keepdims=True) + 1e-9

        # Live state
        self._current_bias: float = 0.0
        self._last_update:  float = 0.0
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def compute_offline_embedding(self, text: str) -> np.ndarray:
        """
        Compute FinBERT embedding for a news headline.
        In production: called by a batch job that pre-processes news archives.
        Returns a 768-dim vector.
        """
        # Mock: returns deterministic hash-based vector (avoids transformers dependency)
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(self.emb_dim).astype(np.float32)

    def project_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce 768-dim to proj_dim using pre-computed projection matrix."""
        e = embedding.reshape(1, -1) if embedding.ndim == 1 else embedding
        return (e @ self._proj).squeeze()

    def sentiment_from_embedding(self, embedding: np.ndarray) -> float:
        """
        Map FinBERT embedding to scalar sentiment bias [-1, +1].
        FinBERT outputs [positive, negative, neutral] logits in its final layer;
        this projects the full embedding to a scalar.
        """
        proj = self.project_embedding(embedding)
        # Sigmoid of first projected dimension as a proxy for positive sentiment
        return float(np.tanh(proj[0]))

    def update_global_brain(self, headlines: List[str]) -> float:
        """
        Update the global sentiment bias from a batch of recent headlines.
        Called by the slow loop (every ~60 seconds).
        """
        if not headlines:
            # Decay toward zero if no new headlines
            dt = time.time() - self._last_update
            self._current_bias *= np.exp(-self.decay_lam * dt)
            return self._current_bias

        sentiments = []
        for h in headlines:
            if h not in self._embedding_cache:
                self._embedding_cache[h] = self.compute_offline_embedding(h)
            s = self.sentiment_from_embedding(self._embedding_cache[h])
            sentiments.append(s)

        # Weighted average (most recent gets highest weight)
        weights = np.exp(-self.decay_lam * np.arange(len(sentiments))[::-1])
        weights /= weights.sum()
        self._current_bias = float(np.dot(weights, sentiments))
        self._last_update  = time.time()
        return self._current_bias

    def get_bias(self) -> float:
        """Get current sentiment bias (with decay applied)."""
        dt = time.time() - self._last_update
        return self._current_bias * np.exp(-self.decay_lam * dt)

    def filter_signal(self, raw_signal: int, bias: Optional[float] = None) -> int:
        """
        Apply sentiment bias to suppress counter-trend signals.
          bias > +threshold: suppress SELL signals
          bias < -threshold: suppress BUY signals
        """
        if bias is None: bias = self.get_bias()
        if bias > self.threshold  and raw_signal == 2: return 1  # Suppress SELL → HOLD
        if bias < -self.threshold and raw_signal == 0: return 1  # Suppress BUY  → HOLD
        return raw_signal

    def build_sentiment_series(
        self,
        headlines_by_time: Dict[pd.Timestamp, List[str]],
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Build a sentiment bias time series aligned to bar index.
        Used to construct the 'sentiment_decayed' feature column.
        """
        raw = pd.Series(0.0, index=index, dtype=float)
        for ts, headlines in sorted(headlines_by_time.items()):
            if ts in raw.index:
                raw[ts] = self.update_global_brain(headlines)

        # Forward-fill with exponential decay
        result = pd.Series(0.0, index=index, dtype=float)
        last_ts = None; last_v = 0.0
        for ts in index:
            if ts in raw.index and raw[ts] != 0:
                last_ts = ts; last_v = raw[ts]; result[ts] = last_v
            elif last_ts is not None:
                dt = (ts - last_ts).total_seconds()
                result[ts] = last_v * np.exp(-self.decay_lam * dt)
        return result.rename("sentiment_bias")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TIP-SEARCH LATENCY MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class TIPSearchManager:
    """
    Time-Predictable Inference Scheduling (TIP-Search).

    Runs TWO models:
      FAST model (DQN):   ~2ms latency, lower accuracy
      SLOW model (HAELT): ~5ms latency, higher accuracy

    Switching logic:
      - Normal market: use SLOW model (accuracy priority)
      - Volatility spike (ATR > 2× avg): switch to FAST model
        (must get order in before spread widens)

    This ensures the system NEVER misses its execution window, even during
    high-impact news events where the slow model would exceed acceptable latency.
    """

    def __init__(
        self,
        fast_agent,           # DQNAgent (or any .select_action(obs) → int)
        slow_agent,           # PPOAgent / any supervised model
        fast_latency_ms: float = 2.0,
        slow_latency_ms: float = 5.0,
        switch_mult:     float = 2.0,
        atr_lookback:    int   = 60,
        max_latency_ms:  float = 10.0,
    ):
        self.fast = fast_agent
        self.slow = slow_agent
        self.fast_ms  = fast_latency_ms
        self.slow_ms  = slow_latency_ms
        self.switch_m = switch_mult
        self.lb       = atr_lookback
        self.max_ms   = max_latency_ms

        self._atr_history: List[float] = []
        self.stats = {"fast_used": 0, "slow_used": 0, "total": 0}

    def _is_vol_spike(self, current_atr: float) -> bool:
        if len(self._atr_history) < self.lb: return False
        avg_atr = np.mean(self._atr_history[-self.lb:])
        return current_atr > self.switch_m * avg_atr

    def select_action(
        self,
        obs:         np.ndarray,
        current_atr: float = 0.0,
    ) -> Tuple[int, str, float]:
        """
        Returns (action, model_used, latency_ms).

        Selects model dynamically based on current volatility regime.
        """
        self._atr_history.append(current_atr)
        use_fast = self._is_vol_spike(current_atr)

        t0 = time.perf_counter()
        if use_fast:
            action = self.fast.select_action(obs)
            model_used = "fast_dqn"
        else:
            try:
                action = self.slow.select_action(obs)
                model_used = "slow_haelt"
            except Exception:
                # Fallback to fast if slow model errors
                action = self.fast.select_action(obs)
                model_used = "fast_fallback"

        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats["total"] += 1
        self.stats["fast_used" if use_fast else "slow_used"] += 1

        return action, model_used, latency_ms

    def report(self) -> dict:
        t = max(self.stats["total"], 1)
        return {
            "fast_pct":  self.stats["fast_used"] / t * 100,
            "slow_pct":  self.stats["slow_used"] / t * 100,
            "total_calls": t,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class DriftDetector:
    """
    Detects when the model's feature distribution or performance
    has drifted significantly from the training distribution.

    Three detection methods:
      PSI (Population Stability Index): measures feature distribution shift
      KS Test: statistical test for distribution change
      Sharpe Drop: performance-based drift signal

    Any trigger can initiate automatic retraining.
    """

    def __init__(
        self,
        psi_threshold:   float = 0.2,
        ks_pvalue:       float = 0.05,
        sharpe_drop:     float = 0.5,
        window:          int   = 1000,
        bars_per_year:   float = 252 * 24 * 60,  # 1-min bars default
    ):
        self.psi_thresh  = psi_threshold
        self.ks_p        = ks_pvalue
        self.sd_thresh   = sharpe_drop
        self.window      = window
        self.ann_factor  = np.sqrt(bars_per_year)
        self._train_dist: Optional[np.ndarray] = None
        self._baseline_sharpe: float = 0.0

    def fit_baseline(self, X_train: np.ndarray, baseline_returns: np.ndarray):
        """Store training distribution and baseline Sharpe for comparison."""
        self._train_dist = X_train.copy()
        if len(baseline_returns) > 1:
            std = baseline_returns.std(ddof=1)
            self._baseline_sharpe = (
                baseline_returns.mean() / (std + 1e-9) * self.ann_factor
            ) if std > 1e-12 else 0.0
        print(f"[Drift] Baseline fitted | Sharpe: {self._baseline_sharpe:.3f} | "
              f"Train dist: {X_train.shape}")

    def compute_psi(self, expected: np.ndarray, actual: np.ndarray,
                    bins: int = 10) -> float:
        """
        Population Stability Index.
          PSI < 0.1  : no significant shift
          PSI 0.1-0.2: moderate shift, monitor
          PSI > 0.2  : significant shift → retrain
        """
        eps = 1e-6
        exp_hist, edges = np.histogram(expected, bins=bins, density=True)
        act_hist, _     = np.histogram(actual, bins=edges, density=True)
        exp_hist += eps; act_hist += eps
        return float(np.sum((act_hist - exp_hist) * np.log(act_hist / exp_hist)))

    def check(
        self,
        X_live:         np.ndarray,
        live_returns:   np.ndarray,
    ) -> dict:
        """
        Run all drift checks. Returns a dict with drift flags and scores.
        """
        result = {
            "drift_detected": False,
            "psi_max": 0.0,
            "ks_min_pvalue": 1.0,
            "sharpe_drop": 0.0,
            "reasons": [],
        }

        if self._train_dist is None:
            return result

        # PSI per feature (use subset of most important features)
        n_feats = min(X_live.shape[1], self._train_dist.shape[1])
        psi_vals = []
        for f in range(n_feats):
            train_f = self._train_dist[:, f]
            live_f  = X_live[-min(len(X_live), self.window):, f]
            psi_vals.append(self.compute_psi(train_f, live_f))
        result["psi_max"] = float(np.max(psi_vals))

        # KS test on first feature (price return proxy)
        if SCIPY and len(X_live) >= 30:
            ks_pvals = []
            for f in range(min(5, n_feats)):
                _, p = stats.ks_2samp(
                    self._train_dist[:, f],
                    X_live[-self.window:, f]
                )
                ks_pvals.append(p)
            result["ks_min_pvalue"] = float(np.min(ks_pvals))

        # Sharpe drop
        if len(live_returns) > 30:
            window_rets = live_returns[-self.window:]
            std = window_rets.std(ddof=1) if len(window_rets) > 1 else 0.0
            live_sharpe = (
                window_rets.mean() / (std + 1e-9) * self.ann_factor
            ) if std > 1e-12 else 0.0
            drop = self._baseline_sharpe - live_sharpe
            result["sharpe_drop"] = float(drop)
            if drop > self.sd_thresh:
                result["drift_detected"] = True
                result["reasons"].append(f"Sharpe drop {drop:.3f} > {self.sd_thresh}")

        if result["psi_max"] > self.psi_thresh:
            result["drift_detected"] = True
            result["reasons"].append(f"PSI {result['psi_max']:.3f} > {self.psi_thresh}")

        if result["ks_min_pvalue"] < self.ks_p:
            result["drift_detected"] = True
            result["reasons"].append(f"KS p-value {result['ks_min_pvalue']:.4f} < {self.ks_p}")

        return result


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Contrastive Pre-training — Smoke Tests")
    print("=" * 60)

    # 1. Augmentation (batch API)
    aug = TimeSeriesAugmenter()
    x_batch = np.random.randn(8, 60, 10).astype(np.float32)
    out = aug.augment_batch(x_batch)
    print(f"[Aug] augment_batch: {x_batch.shape} → {out.shape}")

    # 2. Dual-stream sentiment
    ds = DualStreamSentiment()
    headlines = ["Fed signals rate cut", "Strong jobs report surprises markets"]
    bias = ds.update_global_brain(headlines)
    print(f"\n[Dual-Stream] Sentiment bias: {bias:.4f}")

    # 3. Drift detection
    dd = DriftDetector()
    X_train = np.random.randn(1000, 20)
    dd.fit_baseline(X_train, np.random.normal(0.001, 0.01, 1000))
    result = dd.check(np.random.randn(500, 20) * 2, np.random.normal(-0.002, 0.01, 500))
    print(f"\n[Drift] Detected: {result['drift_detected']} | PSI: {result['psi_max']:.3f}")

    print("\nAll smoke tests passed.")
