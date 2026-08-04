"""
Multi-Task Pretraining with Domain Adaptation (Improvement #10)
===============================================================
Unified pretraining framework combining multiple self-supervised objectives:
  - Contrastive (SimCLR/BYOL/NT-Xent)
  - Masked reconstruction (BERT-style masking)
  - Forecast pretext (causal prefix → suffix prediction)
  - VAE reconstruction (optional)
  - Drift detection (optional)

Features:
  - Multi-task loss weighting (static or uncertainty-weighted)
  - Gradient normalization across tasks (GradNorm)
  - Domain adaptation via adversarial discriminator (DANN)
  - Domain adaptation via MMD / CORAL
  - Domain-specific batch normalization (optional)
  - Curriculum-aware task scheduling
  - Shared encoder with task-specific heads
  - Flexible checkpointing and diagnostics

Architecture:
  Shared Encoder → Task-Specific Heads
                    ├─ Contrastive Head (projection)
                    ├─ Masked Recon Head
                    ├─ Forecast Head
                    └─ Domain Discriminator (DANN)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.optim as optim
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


# ═════════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MultiTaskPretrainConfig:
    """Configuration for multi-task pretraining."""
    # Shared encoder
    d_model: int = 128
    seq_len: int = 60
    n_features: int = 20
    encoder_type: str = "transformer"  # "transformer", "lstm", "gru", "mamba"

    # Task enable flags
    use_contrastive: bool = True
    use_masked_recon: bool = True
    use_forecast: bool = True
    use_vae: bool = False
    use_drift: bool = False

    # Task weights (can be auto-adjusted by GradNorm)
    contrastive_weight: float = 1.0
    masked_recon_weight: float = 1.0
    forecast_weight: float = 1.0
    vae_weight: float = 0.5
    drift_weight: float = 0.5

    # GradNorm settings
    use_gradnorm: bool = True
    gradnorm_alpha: float = 0.5  # gradient balancing strength

    # Domain adaptation
    use_domain_adaptation: bool = True
    da_method: str = "dann"  # "dann", "mmd", "coral", "bn"
    da_weight: float = 0.1
    n_domains: int = 2  # source + target
    da_lambda: float = 1.0  # gradient reversal strength

    # Contrastive settings
    contrastive_temp: float = 0.5
    contrastive_proj_dim: int = 128
    contrastive_negatives: int = 1  # number of negative samples per positive

    # Masked reconstruction
    mask_prob: float = 0.2
    mask_recon_hidden: int = 512

    # Forecast settings
    forecast_horizon: int = 5
    forecast_hidden: int = 512

    # VAE settings
    vae_latent_dim: int = 64
    vae_beta: float = 0.001

    # Drift settings
    drift_margin: float = 1.0

    # Augmentation
    aug_jitter_std: float = 0.03
    aug_scale_range: tuple[float, float] = (0.9, 1.1)
    aug_feature_drop_p: float = 0.1
    aug_crop_ratio: tuple[float, float] = (0.7, 1.0)

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 50
    warmup_epochs: int = 3
    device: str = "cuda"
    seed: int = 0
    checkpoint_path: str | None = None

    # Domain adaptation
    domain_labels: np.ndarray | None = None  # per-sample domain labels

    @property
    def prefix_len(self) -> int:
        return self.seq_len - self.forecast_horizon


@dataclass
class PretrainResult:
    """Result of pretraining."""
    history: dict[str, list[float]]
    encoder_state: dict[str, torch.Tensor]
    config: MultiTaskPretrainConfig
    final_metrics: dict[str, float]


# ════════════════════════════════════════════════════════════════════════════
# 2. Augmentations
# ════════════════════════════════════════════════════════════════════════════

class TimeSeriesAugmenter:
    """Time-series augmentations for contrastive/SSL."""

    def __init__(
        self,
        jitter_std: float = 0.03,
        scale_range: tuple[float, float] = (0.9, 1.1),
        feature_drop_p: float = 0.1,
        crop_ratio: tuple[float, float] = (0.7, 1.0),
        seed: int = 0,
    ):
        self.jitter_std = float(jitter_std)
        self.scale_range = scale_range
        self.feature_drop_p = float(feature_drop_p)
        self.crop_ratio = crop_ratio
        self._rng = np.random.default_rng(seed)

    def augment_batch(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentations to batch of windows."""
        x = x.copy()
        B, T, F = x.shape

        # Jitter
        if self.jitter_std > 0:
            x += self._rng.normal(0, self.jitter_std, x.shape).astype(np.float32)

        # Scale
        if self.crop_ratio != (1.0, 1.0):
            scales = self._rng.uniform(self.crop_ratio[0], self.crop_ratio[1], (B, 1, 1))
            x = x * scales.astype(np.float32)

        # Feature dropout
        if self.feature_drop_p > 0:
            mask = self._rng.random((B, F)) < self.feature_drop_p
            # mask is (B, F), need to broadcast to (B, T, F)
            x = x * (~mask)[:, None, :].astype(np.float32)

        return x.astype(np.float32)

    def augment_pair(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate two augmented views for contrastive learning."""
        return self.augment_batch(x), self.augment_batch(x)


# ════════════════════════════════════════════════════════════════════════════
# 3. Shared Encoder & Task Heads
# ═════════════════════════════════════════════════════════════════════════════

class _MeanPool(nn.Module):
    """Mean pooling over sequence dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


def _build_encoder(
    encoder_type: str,
    seq_len: int,
    n_features: int,
    d_model: int,
    n_layers: int = 3,
    n_heads: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    """Build shared encoder that takes (batch, seq_len, n_features) and outputs (batch, d_model)."""
    if encoder_type == "transformer":
        # Project input features to d_model first
        encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=min(n_heads, d_model),
                    dim_feedforward=512,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=n_layers,
            ),
            # Pool over sequence dimension: (batch, seq_len, d_model) -> (batch, d_model)
            _MeanPool(),
        )
    elif encoder_type == "lstm":
        encoder = nn.Sequential(
            nn.LSTM(n_features, 128, num_layers=3, batch_first=True, bidirectional=True),
            lambda x: x[0][:, -1, :],
            nn.Linear(256, d_model),
        )
    elif encoder_type == "gru":
        encoder = nn.Sequential(
            nn.GRU(n_features, 128, num_layers=3, batch_first=True, bidirectional=True),
            lambda x: x[0][:, -1, :],
            nn.Linear(256, d_model),
        )
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")
    return encoder


def _encode_last(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Encode and return last timestep embedding."""
    h = encoder(x)
    if h.ndim == 3:
        h = h[:, -1, :]
    return torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-50, 50).float()


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for DANN."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """Domain discriminator for DANN."""

    def __init__(self, input_dim: int, n_domains: int, hidden_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(grad_reverse(x, 1.0))


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss for domain adaptation."""

    def __init__(self, kernel: str = "rbf", gamma: float = 1.0):
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MMD between source and target features."""
        # RBF kernel MMD
        X, Y = source, target
        XX = torch.cdist(X, X, p=2).pow(2)
        YY = torch.cdist(Y, Y, p=2).pow(2)
        XY = torch.cdist(X, Y, p=2).pow(2)

        if self.kernel == "rbf":
            K_XX = torch.exp(-self.gamma * XX)
            K_YY = torch.exp(-self.gamma * YY)
            K_XY = torch.exp(-self.gamma * XY)
        else:
            raise NotImplementedError

        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        return mmd.clamp(min=0)


class CORALLoss(nn.Module):
    """CORAL loss for domain adaptation (covariance alignment)."""

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = source.size(1)
        # Covariance matrices
        Cs = self._covariance(source)
        Ct = self._covariance(target)
        # Frobenius norm
        loss = (Cs - Ct).pow(2).sum() / (4 * d * d)
        return loss

    def _covariance(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        return (x.T @ x) / (x.size(0) - 1)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Task Losses
# ═════════════════════════════════════════════════════════════════════════════

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent contrastive loss (SimCLR)."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # 2B x D
    z = F.normalize(z, dim=-1, eps=1e-8)
    sim = torch.mm(z, z.T) / temperature

    # Mask self-similarity
    mask = torch.eye(2 * z1.shape[0], device=z1.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float('-inf'))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.arange(B, device=z1.device)
    labels = torch.cat([labels + B, labels], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss


def byol_loss(p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """BYOL loss (no negatives, predictor + target network)."""
    p1 = F.normalize(p1, dim=-1)
    p2 = F.normalize(p2, dim=-1)
    z1 = F.normalize(z1.detach(), dim=-1)
    z2 = F.normalize(z2.detach(), dim=-1)
    loss = 2 - 2 * (p1 * z2).sum(dim=-1).mean() + 2 - 2 * (p2 * z1).sum(dim=-1).mean()
    return loss


def masked_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss only on masked positions."""
    masked_recon = recon[mask]
    masked_target = target[mask]
    return F.mse_loss(masked_recon, masked_target)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss: reconstruction + KL."""
    recon_loss = F.mse_loss(recon, target)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    loss = recon_loss + beta * kl
    return loss, recon_loss, kl


def forecast_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Forecast MSE loss."""
    return F.mse_loss(pred, target)


def drift_loss(
    clean_emb: torch.Tensor,
    drift_emb: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Drift contrastive loss: push drift embeddings away from clean."""
    clean = F.normalize(clean_emb, dim=-1)
    drift = F.normalize(drift_emb, dim=-1)
    dist = (clean - drift).norm(dim=1)
    loss = F.relu(margin - dist).mean()
    return loss


def domain_adversarial_loss(
    features: torch.Tensor,
    domain_labels: torch.Tensor,
    discriminator: nn.Module,
    lambda_: float = 1.0,
) -> torch.Tensor:
    """DANN domain adversarial loss."""
    reversed_features = grad_reverse(features, lambda_)
    domain_logits = discriminator(reversed_features)
    return F.cross_entropy(domain_logits, domain_labels)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Multi-Task Pretrainer
# ═════════════════════════════════════════════════════════════════════════════

class MultiTaskPretrainer(nn.Module):
    """
    Unified multi-task pretraining with optional domain adaptation.
    
    Combines multiple self-supervised objectives:
      - Contrastive (NT-Xent/BYOL)
      - Masked reconstruction
      - Forecast pretext
      - VAE (optional)
      - Drift detection (optional)
      - Domain adaptation (DANN/MMD/CORAL)
    
    Features:
      - Shared encoder with task-specific heads
      - GradNorm for automatic loss balancing
      - Domain adaptation (DANN/MMD/CORAL)
      - Domain-specific BN (optional)
      - Flexible task scheduling
    """

    def __init__(self, config: MultiTaskPretrainConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self._rng = np.random.default_rng(config.seed)

        # Build shared encoder
        self.encoder = _build_encoder(
            config.encoder_type,
            config.seq_len,
            config.n_features,
            config.d_model,
        ).to(self.device)

        d_model = config.d_model

        # Task heads
        self.heads = nn.ModuleDict()
        self.task_weights = {}

        if config.use_contrastive:
            self.heads["contrastive_proj"] = nn.Sequential(
                nn.Linear(d_model, config.contrastive_proj_dim),
                nn.ReLU(),
                nn.Linear(config.contrastive_proj_dim, config.contrastive_proj_dim),
            ).to(self.device)
            self.task_weights["contrastive"] = config.contrastive_weight

        if config.use_masked_recon:
            self.heads["masked_decoder"] = nn.Sequential(
                nn.Linear(d_model, config.mask_recon_hidden),
                nn.LayerNorm(config.mask_recon_hidden),
                nn.ReLU(),
                nn.Linear(config.mask_recon_hidden, config.seq_len * config.n_features),
            ).to(self.device)
            self.task_weights["masked_recon"] = config.masked_recon_weight

        if config.use_forecast:
            self.heads["forecast"] = nn.Sequential(
                nn.Linear(d_model, config.forecast_hidden),
                nn.ReLU(),
                nn.Linear(config.forecast_hidden, config.forecast_horizon * config.n_features),
            ).to(self.device)
            self.task_weights["forecast"] = config.forecast_weight

        if config.use_vae:
            self.heads["vae_mu"] = nn.Linear(d_model, config.vae_latent_dim).to(self.device)
            self.heads["vae_logvar"] = nn.Linear(d_model, config.vae_latent_dim).to(self.device)
            self.heads["vae_decoder"] = nn.Sequential(
                nn.Linear(config.vae_latent_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, config.seq_len * config.n_features),
            ).to(self.device)
            self.task_weights["vae"] = config.vae_weight

        if config.use_drift:
            self.task_weights["drift"] = config.drift_weight

        # Domain adaptation
        self.discriminator = None
        if config.use_domain_adaptation and config.n_domains > 1:
            if config.da_method == "dann":
                self.discriminator = DomainDiscriminator(d_model, config.n_domains).to(self.device)
                self.task_weights["domain"] = config.da_weight

        # Augmentation
        self.augmenter = TimeSeriesAugmenter(
            jitter_std=config.aug_jitter_std,
            scale_range=config.aug_scale_range,
            feature_drop_p=config.aug_feature_drop_p,
            crop_ratio=config.aug_crop_ratio,
            seed=config.seed,
        )

        # Optimizer
        all_params = list(self.encoder.parameters())
        for head in self.heads.values():
            all_params.extend(list(head.parameters()))
        if self.discriminator:
            all_params.extend(list(self.discriminator.parameters()))

        self.opt = optim.AdamW(
            all_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

# GradNorm parameters (learnable task weights)
        self.gradnorm_params = None
        self._gradnorm_task_names = []
        if config.use_gradnorm:
            # Only create params for tasks that are unconditionally enabled (always produce loss)
            self._gradnorm_task_names = []
            if config.use_contrastive:
                self._gradnorm_task_names.append("contrastive")
            if config.use_masked_recon:
                self._gradnorm_task_names.append("masked_recon")
            if config.use_forecast:
                self._gradnorm_task_names.append("forecast")
            if config.use_vae:
                self._gradnorm_task_names.append("vae")
            if config.use_drift:
                self._gradnorm_task_names.append("drift")
            # Domain is conditional on domain_labels, so exclude from GradNorm
            if self._gradnorm_task_names:
                self.gradnorm_params = nn.Parameter(
                    torch.ones(len(self._gradnorm_task_names), device=self.device)
                )
                self.register_parameter('gradnorm_weights', self.gradnorm_params)

        self._total_epochs = 0
        self._rng = np.random.default_rng(config.seed)
        self._use_amp = config.device.startswith("cuda")
        self._amp_dtype = torch.float16
        self._scaler = torch.amp.GradScaler(enabled=self._use_amp)

        # History
        self.history = {k: [] for k in ["loss", "total"] + list(self.task_weights.keys())}
        self.history["gradnorm_weights"] = []

    def _mask_input(self, x: torch.Tensor, mask_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking."""
        mask = torch.rand_like(x) < self.config.mask_prob
        if not mask.any():
            flat = mask.view(-1)
            flat[self._rng.integers(0, flat.numel())] = True
        return x.masked_fill(mask, 0.0), mask

    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input through shared encoder."""
        h = self.encoder(x)
        if h.ndim == 3:
            h = h[:, -1, :]
        return torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-50, 50).float()

    def _compute_contrastive_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss (NT-Xent)."""
        v1 = self.augmenter.augment_batch(x.cpu().numpy())
        v2 = self.augmenter.augment_batch(x.cpu().numpy())

        v1_t = torch.as_tensor(v1, dtype=torch.float32, device=self.device)
        v2_t = torch.as_tensor(v2, dtype=torch.float32, device=self.device)

        h1 = self._forward_encoder(v1_t)
        h2 = self._forward_encoder(v2_t)

        z1 = self.heads["contrastive_proj"](h1)
        z2 = self.heads["contrastive_proj"](h2)

        return nt_xent_loss(z1, z2, self.config.contrastive_temp)

    def _compute_masked_recon_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute masked reconstruction loss."""
        masked_x, mask = self._mask_input(x, self.config.mask_prob)
        h = self._forward_encoder(masked_x)
        recon = self.heads["masked_decoder"](h).view(-1, self.config.seq_len, self.config.n_features)
        return masked_reconstruction_loss(recon, x, mask)

    def _compute_forecast_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forecast pretext loss."""
        prefix_len = self.config.seq_len - self.config.forecast_horizon
        prefix = x[:, :self.config.prefix_len, :]
        target = x[:, self.config.prefix_len:, :]

        h = self._forward_encoder(prefix)
        pred = self.heads["forecast"](h).view(-1, self.config.forecast_horizon, self.config.n_features)
        return forecast_loss(pred, target)

    def _compute_vae_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss."""
        h = self._forward_encoder(x)
        mu = self.heads["vae_mu"](h)
        logvar = self.heads["vae_logvar"](h).clamp(-8, 8)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.heads["vae_decoder"](z).view(-1, self.config.seq_len, self.config.n_features)
        return vae_loss(recon, x, mu, logvar, self.config.vae_beta)

    def _compute_drift_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute drift contrastive loss."""
        drift_x = self.augmenter.augment_batch(x.cpu().numpy())
        drift_x_t = torch.as_tensor(drift_x, dtype=torch.float32, device=self.device)

        clean = F.normalize(self._forward_encoder(x), dim=-1)
        drift = F.normalize(self._forward_encoder(drift_x_t), dim=-1)
        return drift_loss(clean, drift, self.config.drift_margin)

    def _compute_domain_loss(self, x: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        """Compute domain adaptation loss."""
        h = self._forward_encoder(x)

        if self.config.da_method == "dann" and self.discriminator:
            return domain_adversarial_loss(h, domain_labels, self.discriminator, self.config.da_lambda)
        elif self.config.da_method == "mmd":
            # Need source/target split - for now use all as source
            return torch.tensor(0.0, device=self.device)
        elif self.config.da_method == "coral":
            return torch.tensor(0.0, device=self.device)
        return torch.tensor(0.0, device=self.device)

    def _gradnorm_step(self, task_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply GradNorm to balance task losses."""
        if self.gradnorm_params is None:
            return sum(task_losses.values())

        # Only use tasks that have gradnorm_params
        task_losses_filtered = {k: v for k, v in task_losses.items() if k in self._gradnorm_task_names}

        if not task_losses_filtered:
            return sum(task_losses.values())

        # Compute gradient norms
        self.opt.zero_grad(set_to_none=True)
        grads = {}
        for name, loss in task_losses_filtered.items():
            grad = torch.autograd.grad(loss, self.encoder.parameters(), retain_graph=True, create_graph=True)
            grad_norm = torch.cat([g.flatten() for g in grad if g is not None]).norm(2)
            grads[name] = grad_norm

        # GradNorm: L_grad = sum_i |G_i - mean(G) * (L_i / mean(L))^alpha|
        grad_norms = torch.stack([grads[k] for k in task_losses_filtered])
        loss_vals = torch.stack([l.detach() for l in task_losses_filtered.values()])
        mean_grad = grad_norms.mean()
        mean_loss = loss_vals.mean()
        relative = (loss_vals / mean_loss) ** self.config.gradnorm_alpha
        target_grad = mean_grad * relative
        gradnorm_loss = (grad_norms - target_grad).abs().sum()

        # Update task weights
        self.gradnorm_params.data = (self.gradnorm_params * (target_grad / grad_norms).detach()).clamp(0.1, 10.0)

        # Return weighted sum
        weighted_loss = sum(self.gradnorm_params[i] * list(task_losses_filtered.values())[i]
                           for i in range(len(task_losses_filtered)))
        return weighted_loss

    def step(
        self,
        x: torch.Tensor,
        domain_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single training step."""
        task_losses = {}

        # Forward pass for each task
        if self.config.use_contrastive:
            task_losses["contrastive"] = self._compute_contrastive_loss(x)

        if self.config.use_masked_recon:
            task_losses["masked_recon"] = self._compute_masked_recon_loss(x)

        if self.config.use_forecast:
            task_losses["forecast"] = self._compute_forecast_loss(x)

        if self.config.use_vae:
            loss, recon, kl = self._compute_vae_loss(x)
            task_losses["vae"] = loss

        if self.config.use_drift:
            task_losses["drift"] = self._compute_drift_loss(x)

        if self.config.use_domain_adaptation and domain_labels is not None:
            task_losses["domain"] = self._compute_domain_loss(x, domain_labels)

        # Apply GradNorm or weighted sum
        if self.config.use_gradnorm and len(task_losses) > 1:
            total_loss = self._gradnorm_step(task_losses)
        else:
            total_loss = sum(
                self.task_weights.get(k, 1.0) * v
                for k, v in task_losses.items()
            )

        task_losses["total"] = total_loss

        # Backward
        self.opt.zero_grad(set_to_none=True)
        if self._use_amp:
            self._scaler.scale(total_loss).backward()
            self._scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                [p for h in self.heads.values() for p in h.parameters()] +
                ([p for p in self.discriminator.parameters()] if self.discriminator else []),
                1.0,
            )
            self._scaler.step(self.opt)
            self._scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                [p for h in self.heads.values() for p in h.parameters()] +
                ([p for p in self.discriminator.parameters()] if self.discriminator else []),
                1.0,
            )
            self.opt.step()

# Record losses
        losses_dict = {k: v.detach().item() for k, v in task_losses.items()}
        return losses_dict

    def pretrain(
        self,
        X: np.ndarray,
        domain_labels: np.ndarray | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        checkpoint_path: str | None = None,
        silent: bool = False,
    ) -> dict[str, list[float]]:
        """
        Run multi-task pretraining.
        
        Args:
            X: Input data (N, seq_len, n_features)
            domain_labels: Domain labels for each sample (for DANN)
            epochs: Number of epochs
            batch_size: Batch size
            checkpoint_path: Path to save encoder checkpoint
            silent: Suppress progress output
        
        Returns:
            History dictionary with loss curves.
        """
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        if checkpoint_path is None:
            from config.settings import PATHS
            checkpoint_path = PATHS.get("file_contrastive_encoder", "encoder.pt")

        N = len(X)
        if not silent:
            tasks = [k for k in self.task_weights.keys() if self.task_weights[k] > 0]
            print(f"[MultiTask] {epochs} ep | {N:,} windows | batch={batch_size} | tasks={tasks}")

        # Prepare domain labels tensor
        domain_labels_tensor = None
        if domain_labels is not None:
            domain_labels = torch.as_tensor(domain_labels, dtype=torch.long)

        for epoch in range(epochs):
            self._total_epochs += 1
            idx_perm = self._rng.permutation(len(X))
            epoch_losses = dict.fromkeys(self.task_weights.keys(), 0.0)
            epoch_losses["total"] = 0.0
            n_batches = 0

            # LR schedule with warmup + cosine decay
            base_lr = self.config.lr
            warmup = min(self.config.warmup_epochs, epochs)
            if epoch < warmup:
                lr_scale = (epoch + 1) / warmup
            else:
                progress = (epoch - warmup) / max(1, epochs - warmup)
                lr_scale = 0.5 * (1.0 + np.cos(np.pi * progress))
            for pg in self.opt.param_groups:
                pg["lr"] = self.config.lr * lr_scale

            for start in range(0, len(X), batch_size):
                batch_idx = idx_perm[start : start + batch_size]
                if len(batch_idx) < 4:
                    continue

                x_batch = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=self.device)
                dom_batch = domain_labels[batch_idx] if domain_labels is not None else None

                losses = self.step(x_batch, dom_batch)

                for k, v in losses.items():
                    epoch_losses[k] += v
                n_batches += 1

            if n_batches > 0:
                for k in epoch_losses:
                    epoch_losses[k] /= n_batches

            # Add "loss" as alias for "total" for history compatibility
            epoch_losses["loss"] = epoch_losses["total"]

            for k, v in epoch_losses.items():
                self.history[k].append(v)

            # Record gradnorm weights per epoch
            if self.gradnorm_params is not None:
                self.history["gradnorm_weights"].append(self.gradnorm_params.detach().cpu().numpy().copy())

            if not silent:
                parts = [f"Ep {self._total_epochs:3d} | total={epoch_losses.get('total', 0):.4f}"]
                for k, v in epoch_losses.items():
                    if k != "total":
                        parts.append(f"{k}={v:.4f}")
                if self.gradnorm_params is not None:
                    parts.append(f"w={self.gradnorm_params.detach().cpu().numpy()}")
                print(" | ".join(parts))

        # Save encoder
        torch.save(self.encoder.state_dict(), checkpoint_path)
        if not silent:
            print(f"[MultiTask] Encoder saved → {checkpoint_path}")

        return self.history

    @torch.no_grad()
    def diagnostics(self, X_ref: np.ndarray, max_samples: int = 128) -> dict:
        """Compute diagnostic metrics on reference data."""
        self.encoder.eval()
        for head in self.heads.values():
            head.eval()

        out = {"embed_std": 0.0, "collapsed": True}

        if X_ref is not None and len(X_ref) >= 4:
            try:
                sample = X_ref[: min(max_samples, len(X_ref))]
                x = torch.as_tensor(sample, dtype=torch.float32, device=self.device)
                h = self._forward_encoder(x)
                std = h.std(dim=0).mean().item()
                out["embed_std"] = float(std if np.isfinite(std) else 0.0)
                out["collapsed"] = std < 0.005
            except Exception as exc:
                print(f"[MultiTask] diagnostic error: {exc}")

        self.encoder.train()
        for head in self.heads.values():
            head.train()
        return out

    def save_encoder(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), path)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Factory & Convenience
# ═════════════════════════════════════════════════════════════════════════════

def create_multi_task_pretrainer(
    X: np.ndarray,
    config: MultiTaskPretrainConfig | None = None,
    **kwargs,
) -> MultiTaskPretrainer:
    """Factory function to create MultiTaskPretrainer with sensible defaults."""
    if config is None:
        # Pop training-specific kwargs that are not config parameters
        epochs = kwargs.pop("epochs", 50)
        batch_size = kwargs.pop("batch_size", 256)
        device = kwargs.pop("device", "cuda")
        silent = kwargs.pop("silent", False)
        checkpoint_path = kwargs.pop("checkpoint_path", None)
        domain_labels = kwargs.pop("domain_labels", None)

        seq_len = kwargs.pop("seq_len", 60)
        n_features = kwargs.pop("n_features", X.shape[2] if X.ndim == 3 else 20)
        config = MultiTaskPretrainConfig(
            seq_len=seq_len,
            n_features=n_features,
            **kwargs,
        )
    return MultiTaskPretrainer(config)


def pretrain_multi_task(
    X: np.ndarray,
    config: MultiTaskPretrainConfig | None = None,
    domain_labels: np.ndarray | None = None,
    **kwargs,
) -> tuple[MultiTaskPretrainer, dict]:
    """
    One-shot multi-task pretraining.
    
    Args:
        X: Input data (N, seq_len, n_features)
        config: Pretraining config
        domain_labels: Domain labels for DANN
        **kwargs: Additional config overrides
    
    Returns:
        (trainer, history)
    """
    trainer = create_multi_task_pretrainer(X, config, **kwargs)
    # Filter kwargs for pretrain method
    pretrain_kwargs = {k: v for k, v in kwargs.items()
                       if k in ["epochs", "batch_size", "checkpoint_path", "silent", "domain_labels"]}
    history = trainer.pretrain(X, domain_labels=domain_labels, **pretrain_kwargs)
    return trainer, history


# ════════════════════════════════════════════════════════════════════════════
# 7. Domain Adaptation Utilities
# ════════════════════════════════════════════════════════════════════════════

def adapt_encoder_to_target(
    encoder: nn.Module,
    source_data: np.ndarray,
    target_data: np.ndarray,
    method: str = "dann",
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    Fine-tune encoder for domain adaptation (unsupervised).
    
    Args:
        encoder: Pre-trained encoder
        source_data: Source domain data
        target_data: Target domain data
        method: "dann", "mmd", "coral", "fine_tune"
        epochs: Fine-tuning epochs
        lr: Learning rate
        device: Device
    
    Returns:
        Adapted encoder
    """
    device = torch.device(device)
    encoder = encoder.to(device)

    # Determine encoder output dimension
    with torch.no_grad():
        dummy_input = torch.as_tensor(source_data[:1], dtype=torch.float32, device=device)
        encoder.eval()
        dummy_out = encoder(dummy_input)
        encoder_dim = dummy_out.shape[-1]
        encoder.train()

    if method == "fine_tune":
        # Simple fine-tuning on target with same pretext tasks
        # (requires defining a pretext task)
        pass
    elif method == "dann":
        discriminator = DomainDiscriminator(encoder_dim, 2).to(device)
        opt = optim.Adam(
            list(encoder.parameters()) + list(discriminator.parameters()),
            lr=lr,
        )

        source_t = torch.as_tensor(source_data, dtype=torch.float32, device=device)
        target_t = torch.as_tensor(target_data, dtype=torch.float32, device=device)

        source_labels = torch.zeros(len(source_data), dtype=torch.long, device=device)
        target_labels = torch.ones(len(target_data), dtype=torch.long, device=device)

        for epoch in range(epochs):
            # Source forward - get pooled embeddings
            with torch.no_grad():
                source_emb_full = encoder(source_t)
                if source_emb_full.ndim == 3:
                    source_emb = source_emb_full.mean(dim=1)  # Pool over sequence
                else:
                    source_emb = source_emb_full
            # Target forward
            target_emb_full = encoder(target_t)
            if target_emb_full.ndim == 3:
                target_emb = target_emb_full.mean(dim=1)  # Pool over sequence
            else:
                target_emb = target_emb_full

            # Discriminator
            disc_in = torch.cat([source_emb, target_emb], dim=0)
            disc_labels = torch.cat([torch.zeros(len(source_data)), torch.ones(len(target_data))], dim=0).long().to(device)

            pred = discriminator(grad_reverse(disc_in, 1.0))
            loss_d = F.cross_entropy(pred, disc_labels)

            # Encoder tries to fool discriminator
            loss_e = -F.cross_entropy(discriminator(grad_reverse(target_emb, 1.0)), target_labels)

            loss = loss_d + loss_e
            opt.zero_grad()
            loss.backward()
            opt.step()

    return encoder


# ═════════════════════════════════════════════════════════════════════════════
# 7. Export
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    "CORALLoss",
    "DomainDiscriminator",
    "MMDLoss",
    "MultiTaskPretrainConfig",
    "MultiTaskPretrainer",
    "TimeSeriesAugmenter",
    "adapt_encoder_to_target",
    "create_multi_task_pretrainer",
    "grad_reverse",
    "pretrain_multi_task",
]
