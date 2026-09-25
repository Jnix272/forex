"""
pretrain/extended_trainers.py - Additional self-supervised pretrain objectives.

  vae        - variational autoencoder (reconstruction + KL)
  cluster    - k-means regime discovery + contrastive same-cluster positives
  forecast   - predict future timesteps from prefix (causal pretext)
  drift      - push apart clean vs heavily corrupted window embeddings
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config.settings import PATHS

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH = True
except ImportError:
    TORCH = False

try:
    from tqdm import tqdm as _tqdm

    def _pbar(it, **kw):
        return _tqdm(it, **kw)
except ImportError:

    def _pbar(it, **kw):
        return it


if TORCH:
    from pretrain.contrastive import (
        TimeSeriesAugmenter,
        lalign,
        lunif,
    )
    from pretrain.loss_scaling import compute_target_scale, normalized_mse_loss

    def _encode_last(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        h = encoder(x)
        if h.ndim == 3:
            h = h[:, -1, :]
        return torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-50, 50).float()

    def _kmeans_numpy(X: np.ndarray, k: int, n_iter: int = 20, seed: int = 0) -> np.ndarray:
        """Lloyd k-means on (N, D); returns cluster id per row."""
        X = np.asarray(X, dtype=np.float64)
        n, _d = X.shape
        k = max(1, min(int(k), n))
        rng = np.random.default_rng(seed)
        centroids = X[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=np.int64)
        batch = max(1, min(8192, n))
        for _ in range(n_iter):
            for start in range(0, n, batch):
                end = min(start + batch, n)
                dists = ((X[start:end, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
                labels[start:end] = dists.argmin(axis=1)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centroids[j] = X[mask].mean(axis=0)
                else:
                    centroids[j] = X[rng.integers(0, n)]
        return labels.astype(np.int8)

    class VAESeqTrainer:
        """Sequence VAE: encoder → Gaussian latent → decoder reconstructs full window."""

        def __init__(
            self,
            encoder: nn.Module,
            d_model: int,
            seq_len: int,
            n_features: int,
            latent_dim: int = 64,
            hidden_dim: int = 512,
            beta: float = 0.001,
            lr: float = 1e-4,
            device: str = "cpu",
            seed=None,
        ):
            import copy

            self.device = torch.device(device)
            self.seq_len = int(seq_len)
            self.n_features = int(n_features)
            self.beta = float(beta)
            self.latent_dim = int(min(max(8, latent_dim), d_model))

            encoder = copy.deepcopy(encoder)
            if hasattr(encoder, "head"):
                encoder.head = nn.Identity()
            self.encoder = encoder.to(self.device)
            hidden_dim = int(min(max(64, hidden_dim), 2048))
            self.mu_head = nn.Linear(int(d_model), self.latent_dim).to(self.device)
            self.logvar_head = nn.Linear(int(d_model), self.latent_dim).to(self.device)
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.seq_len * self.n_features),
            ).to(self.device)
            self.opt = torch.optim.AdamW(
                list(self.encoder.parameters())
                + list(self.mu_head.parameters())
                + list(self.logvar_head.parameters())
                + list(self.decoder.parameters()),
                lr=lr,
                weight_decay=1e-4,
            )
            self._use_amp = False
            self._amp_dtype = torch.float32
            self._scaler = torch.amp.GradScaler(enabled=False)
            self._total_epochs = 0

        def _forward(self, x: torch.Tensor, scale: torch.Tensor | None = None):
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
            h = _encode_last(self.encoder, x)
            mu = self.mu_head(h)
            logvar = self.logvar_head(h).clamp(-8, 8)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            recon = self.decoder(z).contiguous().view(-1, self.seq_len, self.n_features)
            if scale is not None:
                recon = recon * scale
            return recon, mu, logvar

        @torch.no_grad()
        def diagnostics(self, X_ref: np.ndarray, max_samples: int = 128) -> dict:
            out = {"recon_loss": 0.0, "kl": 0.0, "embed_std": 0.0, "collapsed": True}
            if X_ref is None or len(X_ref) < 4:
                return out
            self.encoder.eval()
            try:
                sample = X_ref[: min(int(max_samples), len(X_ref))]
                x = torch.as_tensor(sample, dtype=torch.float32, device=self.device)
                scale = compute_target_scale(x)
                recon, mu, logvar = self._forward(x, scale=scale)
                recon_loss = normalized_mse_loss(recon, x).item()
                kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()).item()
                std = mu.std(dim=0).mean().item()
                out = {
                    "recon_loss": float(recon_loss),
                    "kl": float(kl),
                    "embed_std": float(std),
                    "collapsed": std < 0.005,
                }
            except Exception as exc:
                print(f"[VAE] diagnostic error: {exc}")
            finally:
                self.encoder.train()
            return out

        def save_encoder(self, checkpoint_path: str) -> None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)

        def pretrain(
            self,
            X: np.ndarray,
            epochs: int = 50,
            batch_size: int = 256,
            checkpoint_path: str | None = None,
            silent: bool = False,
        ) -> dict:
            if checkpoint_path is None:
                checkpoint_path = PATHS["file_contrastive_encoder"]
            N = len(X)
            history = {"loss": [], "recon_loss": [], "kl": [], "embed_std": []}
            if not silent:
                print(
                    f"[VAE] Pre-training {epochs} ep | {N:,} windows | "
                    f"batch={batch_size} | beta={self.beta:g} | latent={self.latent_dim}"
                )
            base_lr = self.opt.param_groups[0]["lr"]
            warmup_epochs = min(3, epochs)
            epoch_bar = (
                _pbar(range(epochs), desc="VAE Pretrain", unit="ep", leave=True) if not silent else range(epochs)
            )
            for epoch in epoch_bar:
                self._total_epochs += 1
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
                for start in range(0, N, batch_size):
                    batch_idx = idx_perm[start : start + batch_size]
                    if len(batch_idx) < 4:
                        continue
                    x = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=self.device)
                    scale = compute_target_scale(x)
                    recon, mu, logvar = self._forward(x, scale=scale)
                    recon_loss = normalized_mse_loss(recon, x, reduction="sum_features_mean_batch")
                    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
                    loss = recon_loss + self.beta * kl
                    if not torch.isfinite(loss):
                        continue
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters())
                        + list(self.mu_head.parameters())
                        + list(self.logvar_head.parameters())
                        + list(self.decoder.parameters()),
                        1.0,
                    )
                    self.opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg = epoch_loss / max(n_batches, 1)
                diag = self.diagnostics(X)
                history["loss"].append(avg)
                history["recon_loss"].append(diag["recon_loss"])
                history["kl"].append(diag["kl"])
                history["embed_std"].append(diag["embed_std"])
                if not silent:
                    print(
                        f"[VAE] Ep {self._total_epochs:3d} | loss={avg:.4f} "
                        f"| recon={diag['recon_loss']:.4f} kl={diag['kl']:.4f}"
                    )
            if not silent:
                self.save_encoder(checkpoint_path)
            return history

    class ClusterContrastiveTrainer:
        """
        Unsupervised k-means on encoder embeddings, then NT-Xent with
        same-cluster positives and different-cluster negatives.
        """

        def __init__(
            self,
            encoder: nn.Module,
            d_model: int = 128,
            proj_dim: int = 128,
            n_clusters: int = 3,
            temperature: float = 0.5,
            lr: float = 1e-4,
            device: str = "cpu",
            seed=None,
            aug: TimeSeriesAugmenter | None = None,
        ):
            import copy

            self.device = torch.device(device)
            self.n_clusters = max(2, int(n_clusters))
            self.cluster_labels = None
            encoder = copy.deepcopy(encoder)
            if hasattr(encoder, "head"):
                encoder.head = nn.Identity()
            self.encoder = encoder.to(self.device)
            self.proj = nn.Sequential(
                nn.Linear(int(d_model), int(proj_dim)),
                nn.ReLU(),
                nn.Linear(int(proj_dim), int(proj_dim)),
            ).to(self.device)
            self.log_temp = nn.Parameter(torch.tensor(float(temperature)).log().to(self.device))
            self.aug = aug if aug is not None else TimeSeriesAugmenter(seed=seed)
            self.opt = torch.optim.AdamW(
                list(self.encoder.parameters()) + list(self.proj.parameters()) + [self.log_temp],
                lr=lr,
                weight_decay=1e-4,
            )
            self._seed = seed
            self._use_amp = False
            self._total_epochs = 0

        @property
        def temp(self):
            return self.log_temp.exp().clamp(0.05, 2.0)

        def _fit_clusters(self, X: np.ndarray, batch_size: int = 512) -> None:
            embs = []
            self.encoder.eval()
            with torch.no_grad():
                for start in range(0, len(X), batch_size):
                    xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=self.device)
                    h = _encode_last(self.encoder, xb).cpu().numpy()
                    embs.append(h)
            self.encoder.train()
            H = np.concatenate(embs, axis=0)
            self.cluster_labels = _kmeans_numpy(H, self.n_clusters, seed=int(self._seed or 0))

        def nt_xent(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """Supervised contrastive: only same-cluster pairs are positives.

            Vectorised O(B²) computation - the previous per-anchor python loop
            was O(B) allocations and ~10x slower on GPU batches.
            """
            B = z1.shape[0]
            z = torch.cat([z1, z2], dim=0).float()
            z = F.normalize(z, dim=-1, eps=1e-8)
            sim = torch.mm(z, z.T) / self.temp.float()
            mask = torch.eye(2 * B, device=self.device, dtype=torch.bool)
            sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)
            lab = torch.cat([labels, labels], dim=0)
            same = lab.unsqueeze(0) == lab.unsqueeze(1)
            pos = same & ~mask

            # Compute positive/negative similarities on RAW cosine sim (not temp-scaled)
            # so the 0.2 margin has consistent geometric meaning regardless of temperature.
            sim_raw = torch.mm(z, z.T)  # raw cosine (z is already L2-normalized)
            sim_raw = sim_raw.masked_fill(mask, torch.finfo(sim_raw.dtype).min)
            pos_cnt = pos.sum(dim=1)
            pos_sim = (sim_raw * pos).sum(dim=1) / pos_cnt.clamp_min(1)
            neg_sim = sim_raw.masked_fill(same, torch.finfo(sim_raw.dtype).min).amax(dim=1)

            losses = F.relu(neg_sim - pos_sim + 0.2)
            valid = (pos_cnt > 0) & torch.isfinite(neg_sim)
            if not valid.any():
                # Keep a differentiable zero so callers can safely .backward().
                return (z1.sum() + z2.sum()) * 0.0
            return losses[valid].mean()

        @torch.no_grad()
        def diagnostics(self, X_ref: np.ndarray, max_samples: int = 128) -> dict:
            out = {"align": 0.0, "unif": 0.0, "embed_std": 0.0, "collapsed": True}
            if X_ref is None or len(X_ref) < 8:
                return out
            self.encoder.eval()
            self.proj.eval()
            try:
                sample = X_ref[: min(int(max_samples), len(X_ref))]
                v1 = self.aug.augment_batch(sample)
                v2 = self.aug.augment_batch(sample)
                t1 = torch.as_tensor(v1, dtype=torch.float32, device=self.device)
                t2 = torch.as_tensor(v2, dtype=torch.float32, device=self.device)
                z1 = F.normalize(self.proj(_encode_last(self.encoder, t1)), dim=-1)
                z2 = F.normalize(self.proj(_encode_last(self.encoder, t2)), dim=-1)
                std = z1.std(dim=0).mean().item()
                std = float(std if np.isfinite(std) else 0.0)
                align = float(lalign(z1, z2).item())
                unif = float(lunif(z1).item())
                out = {
                    "align": align if np.isfinite(align) else 0.0,
                    "unif": unif if np.isfinite(unif) else 0.0,
                    "embed_std": std,
                    "collapsed": std < 0.005,
                }
            except Exception as exc:
                print(f"[ClusterTSCL] diagnostic error: {exc}")
            finally:
                self.encoder.train()
                self.proj.train()
            return out

        def save_encoder(self, checkpoint_path: str) -> None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)

        def pretrain(
            self,
            X: np.ndarray,
            epochs: int = 50,
            batch_size: int = 256,
            checkpoint_path: str | None = None,
            silent: bool = False,
        ) -> dict:
            if checkpoint_path is None:
                checkpoint_path = PATHS["file_contrastive_encoder"]
            N = len(X)
            history = {"loss": [], "align": [], "unif": []}
            if not silent:
                print(f"[ClusterTSCL] {epochs} ep | {N:,} windows | k={self.n_clusters} | batch={batch_size}")
            for _epoch in range(epochs):
                self._total_epochs += 1
                self._fit_clusters(X, batch_size=min(batch_size * 4, 1024))
                idx_perm = np.random.permutation(N)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, N, batch_size):
                    batch_idx = idx_perm[start : start + batch_size]
                    if len(batch_idx) < 4:
                        continue
                    labels = torch.as_tensor(self.cluster_labels[batch_idx], dtype=torch.long, device=self.device)
                    x = X[batch_idx]
                    v1 = self.aug.augment_batch(x)
                    v2 = self.aug.augment_batch(x)
                    t1 = torch.as_tensor(v1, dtype=torch.float32, device=self.device)
                    t2 = torch.as_tensor(v2, dtype=torch.float32, device=self.device)
                    z1 = self.proj(_encode_last(self.encoder, t1))
                    z2 = self.proj(_encode_last(self.encoder, t2))
                    loss = self.nt_xent(z1, z2, labels)
                    if (not torch.isfinite(loss)) or (not loss.requires_grad):
                        continue
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.proj.parameters()),
                        1.0,
                    )
                    self.opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg = epoch_loss / max(n_batches, 1)
                diag = self.diagnostics(X)
                history["loss"].append(avg)
                history["align"].append(diag["align"])
                history["unif"].append(diag["unif"])
                if not silent:
                    print(
                        f"[ClusterTSCL] Ep {self._total_epochs:3d} | loss={avg:.4f} "
                        f"| align={diag['align']:.3f} unif={diag['unif']:.3f}"
                    )
                if diag["collapsed"] and not silent:
                    print(
                        f"[ClusterTSCL] WARN: embedding std collapsed "
                        f"(std={diag['embed_std']:.5f}) at epoch {self._total_epochs}"
                    )
            if not silent:
                self.save_encoder(checkpoint_path)
            return history

    class ForecastPretextTrainer:
        """Causal prefix → predict suffix timesteps (next-window pretext)."""

        def __init__(
            self,
            encoder: nn.Module,
            d_model: int,
            seq_len: int,
            n_features: int,
            horizon: int = 5,
            hidden_dim: int = 512,
            lr: float = 1e-4,
            device: str = "cpu",
            seed=None,
        ):
            import copy

            self.device = torch.device(device)
            self.seq_len = int(seq_len)
            self.n_features = int(n_features)
            self.horizon = max(1, min(int(horizon), self.seq_len - 1))
            self.prefix_len = self.seq_len - self.horizon

            encoder = copy.deepcopy(encoder)
            if hasattr(encoder, "head"):
                encoder.head = nn.Identity()
            self.encoder = encoder.to(self.device)
            hidden_dim = int(min(max(64, hidden_dim), 2048))
            self.head = nn.Sequential(
                nn.Linear(int(d_model), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.horizon * self.n_features),
            ).to(self.device)
            self.opt = torch.optim.AdamW(
                list(self.encoder.parameters()) + list(self.head.parameters()),
                lr=lr,
                weight_decay=1e-4,
            )
            self._total_epochs = 0

        def _split(self, x: torch.Tensor):
            return x[:, : self.prefix_len, :], x[:, self.prefix_len :, :]

        @torch.no_grad()
        def diagnostics(self, X_ref: np.ndarray, max_samples: int = 128) -> dict:
            out = {"forecast_mse": 0.0, "embed_std": 0.0, "collapsed": True}
            if X_ref is None or len(X_ref) < 4:
                return out
            self.encoder.eval()
            try:
                sample = X_ref[: min(int(max_samples), len(X_ref))]
                x = torch.as_tensor(sample, dtype=torch.float32, device=self.device)
                prefix, target = self._split(x)
                scale = compute_target_scale(target)
                h = _encode_last(self.encoder, prefix)
                pred = self.head(h).contiguous().view(-1, self.horizon, self.n_features) * scale
                mse = normalized_mse_loss(pred, target).item()
                std = h.std(dim=0).mean().item()
                out = {
                    "forecast_mse": float(mse),
                    "embed_std": float(std),
                    "collapsed": std < 0.005,
                }
            except Exception as exc:
                print(f"[Forecast] diagnostic error: {exc}")
            finally:
                self.encoder.train()
            return out

        def save_encoder(self, checkpoint_path: str) -> None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)

        def pretrain(
            self,
            X: np.ndarray,
            epochs: int = 50,
            batch_size: int = 256,
            checkpoint_path: str | None = None,
            silent: bool = False,
        ) -> dict:
            if checkpoint_path is None:
                checkpoint_path = PATHS["file_contrastive_encoder"]
            N = len(X)
            history = {"loss": [], "forecast_mse": [], "embed_std": []}
            if not silent:
                print(f"[Forecast] {epochs} ep | {N:,} windows | horizon={self.horizon} bars | batch={batch_size}")
            for _epoch in range(epochs):
                self._total_epochs += 1
                idx_perm = np.random.permutation(N)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, N, batch_size):
                    batch_idx = idx_perm[start : start + batch_size]
                    if len(batch_idx) < 4:
                        continue
                    x = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=self.device)
                    prefix, target = self._split(x)
                    scale = compute_target_scale(target)
                    h = _encode_last(self.encoder, prefix)
                    pred = self.head(h).contiguous().view(-1, self.horizon, self.n_features) * scale
                    loss = normalized_mse_loss(pred, target)
                    if not torch.isfinite(loss):
                        continue
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.head.parameters()),
                        1.0,
                    )
                    self.opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                avg = epoch_loss / max(n_batches, 1)
                diag = self.diagnostics(X)
                history["loss"].append(avg)
                history["forecast_mse"].append(diag["forecast_mse"])
                history["embed_std"].append(diag["embed_std"])
                if not silent:
                    print(f"[Forecast] Ep {self._total_epochs:3d} | loss={avg:.4f} | mse={diag['forecast_mse']:.4f}")
            if not silent:
                self.save_encoder(checkpoint_path)
            return history

    class DriftContrastiveTrainer:
        """
        Unsupervised drift objective: maximize embedding distance between
        clean windows and heavily corrupted (simulated drift) views.
        """

        def __init__(
            self,
            encoder: nn.Module,
            d_model: int = 128,
            margin: float = 1.0,
            lr: float = 1e-4,
            device: str = "cpu",
            seed=None,
        ):
            import copy

            self.device = torch.device(device)
            self.margin = float(margin)
            encoder = copy.deepcopy(encoder)
            if hasattr(encoder, "head"):
                encoder.head = nn.Identity()
            self.encoder = encoder.to(self.device)
            self.aug = TimeSeriesAugmenter(
                jitter_std=0.08,
                scale_range=(0.5, 1.5),
                feature_drop_p=0.55,
                crop_ratio=(0.4, 0.85),
                seed=seed,
            )
            self.opt = torch.optim.AdamW(self.encoder.parameters(), lr=lr, weight_decay=1e-4)
            self._total_epochs = 0

        @torch.no_grad()
        def diagnostics(self, X_ref: np.ndarray, max_samples: int = 128) -> dict:
            out = {"drift_margin": 0.0, "embed_std": 0.0, "collapsed": True}
            if X_ref is None or len(X_ref) < 4:
                return out
            self.encoder.eval()
            try:
                sample = X_ref[: min(int(max_samples), len(X_ref))]
                x = torch.as_tensor(sample, dtype=torch.float32, device=self.device)
                clean = F.normalize(_encode_last(self.encoder, x), dim=-1)
                drift = F.normalize(
                    _encode_last(
                        self.encoder,
                        torch.as_tensor(self.aug.augment_batch(sample), dtype=torch.float32, device=self.device),
                    ),
                    dim=-1,
                )
                dist = (clean - drift).norm(dim=1).mean().item()
                std = clean.std(dim=0).mean().item()
                out = {
                    "drift_margin": float(dist),
                    "embed_std": float(std),
                    "collapsed": std < 0.005,
                }
            except Exception as exc:
                print(f"[DriftPretrain] diagnostic error: {exc}")
            finally:
                self.encoder.train()
            return out

        def save_encoder(self, checkpoint_path: str) -> None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)

        def pretrain(
            self,
            X: np.ndarray,
            epochs: int = 50,
            batch_size: int = 256,
            checkpoint_path: str | None = None,
            silent: bool = False,
        ) -> dict:
            if checkpoint_path is None:
                checkpoint_path = PATHS["file_contrastive_encoder"]
            N = len(X)
            history = {"loss": [], "drift_margin": [], "embed_std": []}
            if not silent:
                print(f"[DriftPretrain] {epochs} ep | {N:,} windows | margin={self.margin} | batch={batch_size}")
            for _epoch in range(epochs):
                self._total_epochs += 1
                idx_perm = np.random.permutation(N)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, N, batch_size):
                    batch_idx = idx_perm[start : start + batch_size]
                    if len(batch_idx) < 4:
                        continue
                    x_np = X[batch_idx]
                    x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
                    drift_x = torch.as_tensor(self.aug.augment_batch(x_np), dtype=torch.float32, device=self.device)
                    clean = F.normalize(_encode_last(self.encoder, x), dim=-1)
                    drift = F.normalize(_encode_last(self.encoder, drift_x), dim=-1)
                    dist = (clean - drift).norm(dim=1)
                    loss = F.relu(self.margin - dist).mean()
                    if not torch.isfinite(loss):
                        continue
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    self.opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                avg = epoch_loss / max(n_batches, 1)
                diag = self.diagnostics(X)
                history["loss"].append(avg)
                history["drift_margin"].append(diag["drift_margin"])
                history["embed_std"].append(diag["embed_std"])
                if not silent:
                    print(
                        f"[DriftPretrain] Ep {self._total_epochs:3d} | loss={avg:.4f} "
                        f"| margin_dist={diag['drift_margin']:.4f}"
                    )
            if not silent:
                self.save_encoder(checkpoint_path)
            return history


    class JEPATrainer:
        """Time-Series Joint Embedding Predictive Architecture (JEPA)."""
        def __init__(
            self,
            encoder,
            d_model: int,
            seq_len: int,
            n_features: int,
            horizon: int = 5,
            proj_dim: int = 256,
            pred_dim: int = 128,
            ema_decay: float = 0.996,
            lr: float = 1e-4,
            device: str = "cpu",
            seed=None,
        ):
            import copy
            self.device = torch.device(device)
            self.seq_len = int(seq_len)
            self.n_features = int(n_features)
            self.horizon = max(1, min(int(horizon), self.seq_len - 1))
            self.prefix_len = self.seq_len - self.horizon
            self.ema_decay = ema_decay

            encoder = copy.deepcopy(encoder)
            if hasattr(encoder, "head"):
                encoder.head = nn.Identity()
            self.encoder = encoder.to(self.device)
            self.projector = nn.Sequential(
                nn.Linear(int(d_model), proj_dim),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim)
            ).to(self.device)
            self.predictor = nn.Sequential(
                nn.Linear(proj_dim, pred_dim),
                nn.BatchNorm1d(pred_dim),
                nn.ReLU(inplace=True),
                nn.Linear(pred_dim, proj_dim)
            ).to(self.device)

            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_projector = copy.deepcopy(self.projector)
            for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
                p.requires_grad_(False)
            self.target_encoder.eval()
            self.target_projector.eval()

            self.opt = torch.optim.AdamW(
                list(self.encoder.parameters()) + list(self.projector.parameters()) + list(self.predictor.parameters()),
                lr=lr,
                weight_decay=1e-4,
            )
            self._total_epochs = 0

        @torch.no_grad()
        def _ema_update(self):
            d = self.ema_decay
            for o, t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                t.data.mul_(d).add_(o.data, alpha=1.0 - d)
            for o, t in zip(self.projector.parameters(), self.target_projector.parameters()):
                t.data.mul_(d).add_(o.data, alpha=1.0 - d)

        def _split(self, x: torch.Tensor):
            return x[:, : self.prefix_len, :], x[:, self.prefix_len :, :]

        def save_encoder(self, checkpoint_path: str) -> None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)

        def pretrain(
            self,
            X: np.ndarray,
            epochs: int = 50,
            batch_size: int = 256,
            checkpoint_path: str | None = None,
            silent: bool = False,
        ) -> dict:
            if checkpoint_path is None:
                from config.settings import PATHS
                checkpoint_path = PATHS.get("file_contrastive_encoder", "encoder.pt")
            N = len(X)
            history = {"loss": []}
            if not silent:
                print(f"[JEPA] {epochs} ep | {N:,} windows | horizon={self.horizon}")
            for _epoch in range(epochs):
                self._total_epochs += 1
                idx_perm = np.random.permutation(N)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, N, batch_size):
                    batch_idx = idx_perm[start : start + batch_size]
                    if len(batch_idx) < 4:
                        continue
                    x = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=self.device)
                    prefix, target = self._split(x)
                    
                    # Online path
                    h_on = _encode_last(self.encoder, prefix)
                    z_on = self.projector(h_on)
                    p_on = self.predictor(z_on)
                    
                    # Target path
                    with torch.no_grad():
                        h_tgt = _encode_last(self.target_encoder, target)
                        z_tgt = self.target_projector(h_tgt)
                        z_tgt = F.normalize(z_tgt, dim=-1)

                    p_on = F.normalize(p_on, dim=-1)
                    # Cosine loss
                    loss = (2.0 - 2.0 * (p_on * z_tgt).sum(dim=-1)).mean()

                    if not torch.isfinite(loss):
                        continue
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.projector.parameters()) + list(self.predictor.parameters()),
                        1.0,
                    )
                    self.opt.step()
                    self._ema_update()
                    epoch_loss += loss.item()
                    n_batches += 1
                avg = epoch_loss / max(n_batches, 1)
                history["loss"].append(avg)
                if not silent:
                    print(f"[JEPA] Ep {self._total_epochs:3d} | loss={avg:.4f}")
            if not silent:
                self.save_encoder(checkpoint_path)
            return history

    class PatchMaskedTrainer:
        """Patch-Based Masking reconstruction pretext task."""
        def __init__(
            self,
            encoder,
            d_model: int,
            seq_len: int,
            n_features: int,
            patch_size: int = 5,
            mask_prob: float = 0.40,
            hidden_dim: int = 512,
            lr: float = 1e-4,
            device: str = "cpu",
        ):
            import copy
            self.device = torch.device(device)
            self.seq_len = int(seq_len)
            self.n_features = int(n_features)
            self.patch_size = int(patch_size)
            self.mask_prob = float(mask_prob)
            self.n_patches = self.seq_len // self.patch_size
            self.effective_seq_len = self.n_patches * self.patch_size

            encoder = copy.deepcopy(encoder)
            if hasattr(encoder, "head"):
                encoder.head = nn.Identity()
            self.encoder = encoder.to(self.device)
            self.decoder = nn.Sequential(
                nn.Linear(int(d_model), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.effective_seq_len * self.n_features),
            ).to(self.device)
            self.opt = torch.optim.AdamW(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=lr,
                weight_decay=1e-4,
            )
            self._total_epochs = 0

        def save_encoder(self, checkpoint_path: str) -> None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)

        def pretrain(
            self,
            X: np.ndarray,
            epochs: int = 50,
            batch_size: int = 256,
            checkpoint_path: str | None = None,
            silent: bool = False,
        ) -> dict:
            if checkpoint_path is None:
                from config.settings import PATHS
                checkpoint_path = PATHS.get("file_contrastive_encoder", "encoder.pt")
            N = len(X)
            history = {"loss": []}
            if not silent:
                print(f"[PatchMasked] {epochs} ep | {N:,} windows | patch_size={self.patch_size} | mask={self.mask_prob}")
            for _epoch in range(epochs):
                self._total_epochs += 1
                idx_perm = np.random.permutation(N)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, N, batch_size):
                    batch_idx = idx_perm[start : start + batch_size]
                    if len(batch_idx) < 4:
                        continue
                    x = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=self.device)
                    # Truncate to effective length
                    x = x[:, :self.effective_seq_len, :]
                    
                    B = x.shape[0]
                    # Reshape to patches: (B, n_patches, patch_size, F)
                    x_patched = x.view(B, self.n_patches, self.patch_size, self.n_features)
                    
                    # Random masking at patch level
                    mask_prob_tensor = torch.full((B, self.n_patches), self.mask_prob, device=self.device)
                    mask = torch.bernoulli(mask_prob_tensor).bool() # True means masked
                    
                    # Broadcast mask to (B, n_patches, patch_size, F)
                    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(x_patched)
                    
                    x_corrupted = x_patched.masked_fill(mask_expanded, 0.0).view(B, self.effective_seq_len, self.n_features)
                    
                    scale = compute_target_scale(x_patched)
                    h = _encode_last(self.encoder, x_corrupted)
                    recon = (self.decoder(h).view(B, self.n_patches, self.patch_size, self.n_features)) * scale
                    
                    if not mask.any():
                        continue
                        
                    # Calculate MSE only on masked patches
                    loss = normalized_mse_loss(recon, x_patched, mask=mask_expanded)
                    
                    if not torch.isfinite(loss):
                        continue
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        1.0,
                    )
                    self.opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                avg = epoch_loss / max(n_batches, 1)
                history["loss"].append(avg)
                if not silent:
                    print(f"[PatchMasked] Ep {self._total_epochs:3d} | loss={avg:.4f}")
            if not silent:
                self.save_encoder(checkpoint_path)
            return history

else:

    class VAESeqTrainer:
        def __init__(self, **kw):
            pass

        def pretrain(self, *a, **kw):
            return {"loss": [0.0]}

        def diagnostics(self, *a, **kw):
            return {}

        def save_encoder(self, *a, **kw):
            pass

    ClusterContrastiveTrainer = VAESeqTrainer
    ForecastPretextTrainer = VAESeqTrainer
    DriftContrastiveTrainer = VAESeqTrainer

    JEPATrainer = VAESeqTrainer
    PatchMaskedTrainer = VAESeqTrainer
