"""
Target normalization utilities for self-supervised and pretext pretraining losses.

Prevents feature magnitude disparities (e.g. tick volume ~ 10^5 vs price return ~ 10^-4)
from causing loss explosion or starving representation learning via gradient clipping.
"""

from __future__ import annotations

import torch


def compute_target_scale(
    target: torch.Tensor,
    min_scale: float = 1e-5,
    fallback_scale: float = 1.0,
) -> torch.Tensor:
    """Compute per-feature standard deviation scale across batch and sequence dimensions.

    Parameters
    ----------
    target : torch.Tensor
        Target tensor of shape (..., n_features).
    min_scale : float, optional
        Standard deviation threshold to differentiate active features from
        inactive/constant features (e.g., zero-padded cross-pair channels).
        Defaults to 1e-5.
    fallback_scale : float, optional
        Scale to assign to inactive/constant channels (where std < min_scale).
        Default is 1.0 (leaves raw error unscaled and non-exploding).

    Returns
    -------
    torch.Tensor
        Scale tensor broadcastable to target shape (with 1s on non-channel dimensions).
    """
    if target.ndim <= 1:
        return torch.ones_like(target)

    dims = tuple(range(target.ndim - 1))
    var = torch.var(target, dim=dims, keepdim=True, unbiased=False)
    std = torch.sqrt(torch.clamp(var, min=0.0) + 1e-12).detach()

    # Active channels (std >= min_scale) are scaled by their standard deviation.
    # Inactive/constant channels (std < min_scale) fallback to 1.0.
    scale = torch.where(std >= min_scale, std, torch.full_like(std, fallback_scale))
    return scale


def normalized_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
    min_scale: float = 1e-5,
    fallback_scale: float = 1.0,
) -> torch.Tensor:
    """Variance-normalized MSE loss across feature channels.

    Normalizes prediction error by the target feature's empirical standard deviation:
        e_norm = (pred - target) / scale

    This ensures that predicting a 1-sigma deviation on a micro-scale feature
    (e.g., price return with std ~ 0.0025) incurs the exact same penalty as a
    1-sigma deviation on a macro-scale feature (e.g., tick volume with std ~ 500,000).

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions matching target shape (or broadcastable).
    target : torch.Tensor
        Ground-truth target tensor.
    mask : torch.Tensor, optional
        Boolean mask where True denotes elements to include in the loss.
    reduction : str, optional
        Loss reduction mode:
        - "mean": average over all selected elements.
        - "sum": sum over all selected elements.
        - "none": element-wise unreduced normalized squared errors.
        - "sum_features_mean_batch": sum over feature and sequence dimensions,
          mean over batch dimension (standard for VAE ELBO reconstruction terms).
    min_scale : float, optional
        Minimum std threshold for active channels.
    fallback_scale : float, optional
        Scale for inactive/constant channels.

    Returns
    -------
    torch.Tensor
        Computed normalized loss.
    """
    scale = compute_target_scale(target, min_scale=min_scale, fallback_scale=fallback_scale)
    norm_diff = (pred - target) / scale

    if mask is not None:
        masked_diff = norm_diff[mask]
        if masked_diff.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)

        sq = masked_diff ** 2
        if reduction == "mean":
            return sq.mean()
        if reduction == "sum":
            return sq.sum()
        if reduction == "none":
            return sq
        return sq.mean()

    sq = norm_diff ** 2
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    if reduction == "sum_features_mean_batch":
        if sq.ndim > 1:
            return sq.sum(dim=list(range(1, sq.ndim))).mean()
        return sq.mean()
    return sq.mean()
