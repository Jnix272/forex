"""
Synaptic Intelligence (SI) for Continuous Learning.
Prevents catastrophic forgetting by calculating parameter importance online.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DynamicSILambdaConfig:
    """Auto-scale SI λ per batch as a function of the primary task loss.

    Behaviour (self-calibrating, monotone in task loss):

      - When the model has *mastered* the current epoch's data
        (task loss low), λ rises toward ``lambda_max`` so SI firmly
        anchors the parameters to last epoch's θ* and prevents
        overfit-driven drift on already-learned territory.

      - When the model is *struggling* (task loss high, e.g. a regime
        shock), λ falls toward ``lambda_min`` so the model is free to
        re-fit - exactly when adaptation matters most.

    The mapping is a smooth sigmoid:

        λ_t = λ_min + (λ_max − λ_min) · 1 / (1 + L_task / κ)

    where ``κ`` is the task-loss level at which λ sits at its midpoint.
    ``κ`` self-calibrates per run via an exponential moving average of
    ``L_task`` (so the schedule adapts to whatever loss magnitude the
    active criterion - Sharpe-Proxy / Huber / CE - produces, and the
    schedule is smooth across batches, not bang-bang).

    This composes cleanly with the per-epoch θ* re-anchoring in
    :meth:`SynapticIntelligence.update_omega`: θ* rolls per epoch, so
    "protect continuity against this epoch's drift" is the right
    semantics for the penalty this dynamic λ scales.

    Backward-compat: when ``enabled=False`` (default), callers should
    pass ``lambda_max`` straight through as the static ``lambda_si``
    used by ``apply_si_loss`` - behaviour is identical to pre-dynamic-SI.
    """  # noqa: RUF002

    enabled: bool = False
    target_ratio: float = 0.1  # legacy field kept for the loss-ratio variant; unused by the sigmoid schedule
    lambda_min: float = 0.0  # floor (allow 0 = fully relax SI during regime shocks)
    lambda_max: float = 1.0  # ceiling = static si_lambda when disabled
    ema_alpha: float = 0.99  # 1-α: EMA weight for κ; 0.99 → slow-tracking scale  # noqa: RUF003
    warmup_batches: int = 0  # constant λ=lambda_max for first N batches of each epoch
    eps: float = 1e-8

    def as_static_lambda(self) -> float:
        """λ to use when dynamic scaling is disabled (or before warmup ends)."""
        return self.lambda_max


def compute_dynamic_si_lambda(
    base_loss: torch.Tensor,
    *,
    kappa: float | None,
    cfg: DynamicSILambdaConfig,
) -> tuple[float, float]:
    """Auto-balance SI λ per batch.

    Returns ``(lambda_t, kappa_next)`` so the caller can thread the
    updated κ EMA state back in for the next batch.

    Args:
        base_loss: the primary task loss (already detached-friendly; we call .detach()).
        kappa: current EMA of |L_task| used as the sigmoid midpoint; ``None``
            initialises κ to |L_task| on the first observed batch (so the
            first batch uses λ ≈ midpoint of the schedule).
        cfg: the dynamic-λ configuration.

    Notes:
        - Uses ``abs(L_task)`` for κ so the schedule is sign-correct under
          Sharpe-Proxy / Huber / CE losses alike (Sharpe-Proxy can be negative).
        - Output λ is a Python float, **detached** from autograd - the caller
          must not re-inject it as a differentiable tensor or it would bias
          the gradient of the regularised objective.
    """
    bl = float(base_loss.detach())
    abs_bl = abs(bl)

    # κ EMA update (in-place state threaded through the return value).
    if kappa is None or not (kappa > 0.0):
        kappa_next = max(abs_bl, cfg.eps)
    else:
        kappa_next = cfg.ema_alpha * kappa + (1.0 - cfg.ema_alpha) * max(abs_bl, cfg.eps)

    # Sigmoid schedule in L_task / κ.  Guarded against κ ≈ 0.
    ratio = abs_bl / max(kappa_next, cfg.eps)
    sig = 1.0 / (1.0 + ratio)
    lam = cfg.lambda_min + (cfg.lambda_max - cfg.lambda_min) * sig
    # Clamp for safety against FP blowups on pathologically small κ.
    lam = max(cfg.lambda_min, min(lam, cfg.lambda_max))
    return lam, kappa_next


class SynapticIntelligence(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1e-3,
    ):
        """
        Initialize Synaptic Intelligence.

        Args:
            model: The PyTorch model to track.
            epsilon: Small constant to avoid division by zero in importance calculation.
        """
        super().__init__()
        self.model = model
        self.epsilon = epsilon

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        # Initialize state buffers
        for n, p in self.params.items():
            name = n.replace(".", "_")
            # Importance weights (\Omega)
            self.register_buffer(f"omega_{name}", torch.zeros_like(p, requires_grad=False))
            # Accumulated path integral of gradients
            self.register_buffer(f"path_integral_{name}", torch.zeros_like(p, requires_grad=False))
            # Reference parameters at the start of the task (\theta^*)
            self.register_buffer(f"cached_params_{name}", p.clone().detach().requires_grad_(False))
            # Temporary storage for pre-update parameters (for delta_theta per step)
            self.register_buffer(f"saved_params_{name}", p.clone().detach().requires_grad_(False))
            # Raw gradient snapshot from backward(), captured before the training
            # loop clips / centralizes p.grad in place (see pre_step below).
            self.register_buffer(f"last_grad_{name}", torch.zeros_like(p, requires_grad=False))

    def pre_step(self):
        r"""
        Call this BEFORE optimizer.step() but AFTER backward() has populated
        ``p.grad``.

        Snapshots the current parameters (to later compute \Delta \theta) and
        the RAW gradients from ``backward()``. The training loop must call this
        before any in-place modification of ``p.grad`` (gradient clipping,
        centralization) so the path integral reflects the true loss gradients.
        """
        for n, p in self.params.items():
            name = n.replace(".", "_")
            saved_p = getattr(self, f"saved_params_{name}")
            saved_p.copy_(p.data)
            last_grad = getattr(self, f"last_grad_{name}")
            if p.grad is not None:
                last_grad.copy_(p.grad.data)
            else:
                last_grad.zero_()

    def post_step(self):
        r"""
        Call this AFTER optimizer.step().
        Accumulates the path integral: \int g(\theta) d\theta
        Approximated as: -grad * (theta_new - theta_old), using the raw
        gradient snapshot from ``pre_step`` (before any in-place clipping /
        centralization of ``p.grad``).
        """
        for n, p in self.params.items():
            name = n.replace(".", "_")
            saved_p = getattr(self, f"saved_params_{name}")
            path_int = getattr(self, f"path_integral_{name}")
            last_grad = getattr(self, f"last_grad_{name}")

            delta_theta = p.data - saved_p
            # Add the negative inner product (since gradient points in direction of steepest ascent)
            path_int.add_(-last_grad * delta_theta)

    def update_omega(self):
        """
        Call this at the end of a task (e.g., end of a fold or dataset).
        Finalizes the importance weights for the task and resets the tracking state.
        """
        for n, p in self.params.items():
            name = n.replace(".", "_")
            omega = getattr(self, f"omega_{name}")
            path_int = getattr(self, f"path_integral_{name}")
            cached_p = getattr(self, f"cached_params_{name}")

            # Total parameter change during the task
            delta_theta_total = p.data - cached_p

            # \Omega_k = \Omega_{k, old} + (path_integral_k) / (\Delta\theta_{k, total}^2 + \epsilon)
            # Use absolute value for path integral to ensure positive importance,
            # as non-convexity can sometimes result in small negative values locally.
            importance = torch.abs(path_int) / (delta_theta_total**2 + self.epsilon)
            omega.add_(importance)

            # Reset for the next task
            cached_p.copy_(p.data)
            path_int.zero_()

    def penalty(self) -> torch.Tensor:
        r"""
        Calculate the Synaptic Intelligence penalty term.
        \sum \Omega_k (\theta_k - \theta_k^*)^2
        """
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.params.items():
            name = n.replace(".", "_")
            cached_p = getattr(self, f"cached_params_{name}")
            omega = getattr(self, f"omega_{name}")
            loss = loss + (omega * (p - cached_p) ** 2).sum()
        return loss


def apply_si_loss(
    base_loss: torch.Tensor,
    si_module: SynapticIntelligence | None,
    lambda_si: float = 1.0,
) -> torch.Tensor:
    """Add Synaptic Intelligence penalty to the base loss."""
    if si_module is None or lambda_si == 0.0:
        return base_loss
    return base_loss + lambda_si * si_module.penalty()
