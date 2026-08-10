"""
Adversarial Training (PGD / FGSM / FreeLB)
Provides gradient-based adversarial perturbations for robust training.
Replaces the previous random-market-corruption approach with proper
adversarial robustness methods.

Usage:
    from training.adversarial_generator import PGDAttack, FGSMAttack, FreeLBAttack
    
    # PGD (Projected Gradient Descent) - strongest, multi-step
    adv_attack = PGDAttack(eps=0.3, alpha=0.01, steps=7)
    
    # FGSM (Fast Gradient Sign Method) - single step, fast
    adv_attack = FGSMAttack(eps=0.3)
    
    # FreeLB (Free Large-Batch) - accumulates gradients over multiple steps
    adv_attack = FreeLBAttack(eps=0.3, alpha=0.01, steps=3)
    
    # In training loop:
    for xb, yb in train_loader:
        xb_adv = adv_attack(model, xb, yb, criterion)
        loss = criterion(model(xb_adv), yb)
        loss.backward()
        ...
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Union
import random


class AdversarialAttack(nn.Module):
    """Base class for adversarial attacks."""
    
    def __init__(
        self,
        eps: float = 0.3,
        probability: float = 1.0,
        feature_names: Optional[list[str]] = None,
    ):
        super().__init__()
        self.eps = eps
        self.probability = probability
        self.feature_names = list(feature_names) if feature_names else []
        # Per-dimension epsilon multipliers (for Task 2: pretrain vulnerability)
        self.feature_eps_multipliers: Optional[torch.Tensor] = None
    
    def set_feature_names(self, feature_names: list[str] | None) -> None:
        self.feature_names = list(feature_names) if feature_names else []
    
    def set_eps(self, eps: float) -> None:
        """Update epsilon (used for curriculum-scaled adversarial)."""
        self.eps = eps
    
    def set_feature_eps_multipliers(self, multipliers: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        """Set per-dimension epsilon multipliers from pretrain vulnerability scores."""
        if multipliers is not None:
            if isinstance(multipliers, np.ndarray):
                multipliers = torch.from_numpy(multipliers.astype(np.float32))
            self.feature_eps_multipliers = multipliers
        else:
            self.feature_eps_multipliers = None
    
    def _get_effective_eps(self, x: torch.Tensor) -> torch.Tensor:
        """Compute effective epsilon with per-dimension multipliers if available."""
        base_eps = self.eps
        if self.feature_eps_multipliers is not None:
            # Broadcast multipliers to match input shape
            mult = self.feature_eps_multipliers.to(x.device)
            # Handle different shapes: (T, F) or (B, T, F) or (B, N, F)
            if mult.dim() == 1:
                mult = mult.view(*([1] * (x.dim() - 1)), -1)
            base_eps = base_eps * mult
        return base_eps
    
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: Callable,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate adversarial examples. To be implemented by subclasses."""
        raise NotImplementedError


class FGSMAttack(AdversarialAttack):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).
    Single-step attack: x_adv = x + eps * sign(grad_x loss)
    """
    
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: Callable,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.training or random.random() > self.probability:
            return x
        
        x_adv = x.clone().detach().requires_grad_(True)
        output = model(x_adv)
        if isinstance(output, (tuple, list)):
            output = output[0]
        
        loss = criterion(output, y)
        if mask is not None:
            loss = loss * mask
            loss = loss.mean()
        
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv + self.eps * grad.sign()
        x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)
        
        return x_adv.detach()


class PGDAttack(AdversarialAttack):
    """
    Projected Gradient Descent (Madry et al., 2017).
    Multi-step iterative attack with projection to L-infinity ball.
    """
    
    def __init__(
        self,
        eps: float = 0.3,
        alpha: float = 0.01,
        steps: int = 7,
        probability: float = 1.0,
        feature_names: Optional[list[str]] = None,
        random_start: bool = True,
        normalize_grad: bool = False,  # L2 gradient normalization (Madry best practice)
        warmup_steps: int = 0,  # Gradually increase attack steps over training
        feature_eps_multipliers: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__(eps, probability, feature_names)
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.normalize_grad = normalize_grad
        self.warmup_steps = warmup_steps
        self._current_warmup_step = 0
        if feature_eps_multipliers is not None:
            self.set_feature_eps_multipliers(feature_eps_multipliers)
    
    def set_eps(self, eps: float) -> None:
        """Update epsilon (used for curriculum-scaled adversarial)."""
        super().set_eps(eps)
    
    def set_warmup_step(self, step: int) -> None:
        """Update warmup step for gradual attack strength increase."""
        self._current_warmup_step = step
    
    def _effective_steps(self) -> int:
        """Compute effective steps considering warmup."""
        if self.warmup_steps > 0 and self._current_warmup_step < self.warmup_steps:
            return max(1, int(self.steps * (self._current_warmup_step / self.warmup_steps)))
        return self.steps
    
    def _normalize_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """L2 normalize gradient per sample (Madry et al. best practice)."""
        # Flatten all but batch dim, normalize, then reshape
        grad_flat = grad.view(grad.shape[0], -1)
        grad_norm = grad_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        grad_normalized = grad_flat / grad_norm
        return grad_normalized.view_as(grad)
    
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: Callable,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.training or random.random() > self.probability:
            return x
        
        x_adv = x.clone().detach()
        
        # Get effective epsilon with per-dimension multipliers
        effective_eps = self._get_effective_eps(x)
        
        if self.random_start:
            # Random initialization within eps-ball
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-effective_eps, effective_eps)
            x_adv = torch.clamp(x_adv, x - effective_eps, x + effective_eps)
        
        effective_steps = self._effective_steps()
        
        for _ in range(effective_steps):
            x_adv.requires_grad_(True)
            output = model(x_adv)
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            loss = criterion(output, y)
            if mask is not None:
                loss = loss * mask
                loss = loss.mean()
            
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            
            # Optional L2 gradient normalization
            if self.normalize_grad:
                grad = self._normalize_gradient(grad)
            else:
                grad = grad.sign()
            
            x_adv = x_adv + self.alpha * grad
            x_adv = torch.clamp(x_adv, x - effective_eps, x + effective_eps)
        
        return x_adv.detach()


class FreeLBAttack(AdversarialAttack):
    """
    Free Large-Batch Adversarial Training (Zhu et al., 2019).
    Accumulates gradients over multiple steps within a single batch,
    more efficient than PGD for large-batch training.
    """
    
    def __init__(
        self,
        eps: float = 0.3,
        alpha: float = 0.01,
        steps: int = 3,
        probability: float = 1.0,
        feature_names: Optional[list[str]] = None,
    ):
        super().__init__(eps, probability, feature_names)
        self.alpha = alpha
        self.steps = steps
    
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: Callable,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.training or random.random() > self.probability:
            return x
        
        batch_size = x.size(0)
        # Initialize perturbation
        delta = torch.zeros_like(x, requires_grad=True)
        
        # Accumulate adversarial steps
        for i in range(self.steps):
            # Forward with perturbed input
            x_adv = x + delta
            output = model(x_adv)
            if isinstance(output, (tuple, list)):
                output = output[0]
            loss = criterion(output, y)
            if mask is not None:
                loss = loss * mask
                loss = loss.mean()
            loss.backward()
            
            # Update perturbation
            delta.data = delta.data + self.alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.grad.zero_()
        
        # Final adversarial example
        x_adv = x + delta.detach()
        x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)
        
        return x_adv


class MarketShockGenerator(nn.Module):
    """
    Legacy random market corruption (whipsaw, spread blowout, sentiment shock).
    Kept for backward compatibility but NOT adversarial training.
    Use PGDAttack/FGSMAttack/FreeLBAttack for actual adversarial robustness.
    """
    
    def __init__(
        self,
        whipsaw_prob: float = 0.05,
        spread_blowout_prob: float = 0.05,
        sentiment_shock_prob: float = 0.05,
        whipsaw_magnitude: float = 5.0,
        spread_multiplier: float = 10.0,
        probability: float | None = None,
        feature_names: list[str] | None = None,
    ):
        super().__init__()
        if probability is not None:
            p = float(probability)
            whipsaw_prob = p
            spread_blowout_prob = p
            sentiment_shock_prob = p
        self.whipsaw_prob = whipsaw_prob
        self.spread_blowout_prob = spread_blowout_prob
        self.sentiment_shock_prob = sentiment_shock_prob
        self.whipsaw_magnitude = whipsaw_magnitude
        self.spread_multiplier = spread_multiplier
        self.feature_names = list(feature_names) if feature_names else []
        self._warned_no_schema = False

    def set_feature_names(self, feature_names: list[str] | None) -> None:
        self.feature_names = list(feature_names) if feature_names else []
        self._warned_no_schema = False

    def forward(
        self,
        features: torch.Tensor,
        feature_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Inject random market shocks (NOT gradient-based adversarial)."""
        if not self.training:
            return features

        names = feature_names if feature_names is not None else self.feature_names
        if not names:
            if not self._warned_no_schema:
                print("[MarketShockGenerator] No feature_names provided; skipping injections")
                self._warned_no_schema = True
            return features

        adv_features = features.clone()
        batch_size, seq_len, _num_features = adv_features.shape

        close_idx = names.index("close") if "close" in names else -1
        spread_idx = names.index("spread_mean") if "spread_mean" in names else -1

        if close_idx >= 0 and random.random() < self.whipsaw_prob and seq_len >= 2:
            shock_seq_idx = random.randint(max(0, seq_len - 10), seq_len - 1)
            n_hit = max(1, batch_size // 4)
            batch_indices = torch.randint(0, batch_size, (n_hit,), device=features.device)
            adv_features[batch_indices, shock_seq_idx, close_idx] -= self.whipsaw_magnitude
            if shock_seq_idx + 1 < seq_len:
                adv_features[batch_indices, shock_seq_idx + 1, close_idx] += self.whipsaw_magnitude

        if spread_idx >= 0 and random.random() < self.spread_blowout_prob:
            n_hit = max(1, batch_size // 2)
            batch_indices = torch.randint(0, batch_size, (n_hit,), device=features.device)
            adv_features[batch_indices, -5:, spread_idx] *= self.spread_multiplier

        if random.random() < self.sentiment_shock_prob:
            fb_indices = [i for i, name in enumerate(names) if name.startswith("fb_")]
            if fb_indices:
                n_hit = max(1, batch_size // 4)
                batch_indices = torch.randint(0, batch_size, (n_hit,), device=features.device)
                for idx in fb_indices:
                    adv_features[batch_indices, -1, idx] *= -1.0

        return adv_features


class GraphAdversarialAttack(AdversarialAttack):
    """
    Graph-aware adversarial attack for GNN models.
    
    GNN models receive input as (B, n_nodes, node_features) after temporal pooling.
    This attack perturbs the node features and/or the adjacency structure.
    
    Supports two modes:
    - "node_features": Perturb node feature vectors directly
    - "edge_dropout": Randomly drop edges in the adjacency matrix
    - "combined": Both node feature perturbation and edge dropout
    """
    
    def __init__(
        self,
        eps: float = 0.3,
        alpha: float = 0.01,
        steps: int = 7,
        probability: float = 1.0,
        feature_names: Optional[list[str]] = None,
        attack_mode: str = "node_features",
        edge_dropout_rate: float = 0.1,
        random_start: bool = True,
        normalize_grad: bool = False,
        warmup_steps: int = 0,
        feature_eps_multipliers: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__(eps, probability, feature_names)
        self.alpha = alpha
        self.steps = steps
        self.attack_mode = attack_mode
        self.edge_dropout_rate = edge_dropout_rate
        self.random_start = random_start
        self.normalize_grad = normalize_grad
        self.warmup_steps = warmup_steps
        self._current_warmup_step = 0
        if feature_eps_multipliers is not None:
            self.set_feature_eps_multipliers(feature_eps_multipliers)
    
    def set_eps(self, eps: float) -> None:
        """Update epsilon (used for curriculum-scaled adversarial)."""
        super().set_eps(eps)
    
    def set_edge_eps(self, eps: float) -> None:
        """Update edge dropout rate (conceptually different from feature eps)."""
        self.edge_dropout_rate = eps
    
    def set_warmup_step(self, step: int) -> None:
        """Update warmup step for gradual attack strength increase."""
        self._current_warmup_step = step
    
    def _effective_steps(self) -> int:
        """Compute effective steps considering warmup."""
        if self.warmup_steps > 0 and self._current_warmup_step < self.warmup_steps:
            return max(1, int(self.steps * (self._current_warmup_step / self.warmup_steps)))
        return self.steps
    
    def _normalize_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """L2 normalize gradient per sample."""
        grad_flat = grad.view(grad.shape[0], -1)
        grad_norm = grad_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        grad_normalized = grad_flat / grad_norm
        return grad_normalized.view_as(grad)
    
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: Callable,
        mask: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate adversarial examples for GNN.
        
        Args:
            model: GNN model (expects forward(x, adj))
            x: Node features (B, n_nodes, node_features)
            y: Targets
            criterion: Loss function
            mask: Optional sample mask
            adj: Optional adjacency matrix (B, n_nodes, n_nodes)
        
        Returns:
            Tuple of (perturbed_x, perturbed_adj)
        """
        if not self.training or random.random() > self.probability:
            return x, adj
        
        B, N, F = x.shape
        device = x.device
        
        # Initialize perturbed features
        x_adv = x.clone().detach()
        
        # Get effective epsilon with per-dimension multipliers
        effective_eps = self._get_effective_eps(x)
        
        if self.random_start and "node_features" in self.attack_mode:
            # Random initialization within eps-ball
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-effective_eps, effective_eps)
            x_adv = torch.clamp(x_adv, x - effective_eps, x + effective_eps)
        
        # Initialize perturbed adjacency
        adj_adv = adj.clone().detach() if adj is not None else None
        
        effective_steps = self._effective_steps()
        
        for _ in range(effective_steps):
            x_adv.requires_grad_(True)
            if adj_adv is not None:
                adj_adv.requires_grad_(True)
            
            output = model(x_adv, adj_adv)
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            loss = criterion(output, y)
            if mask is not None:
                loss = loss * mask
                loss = loss.mean()
            
            grads = torch.autograd.grad(
                loss, 
                [x_adv] + ([adj_adv] if adj_adv is not None else []),
                retain_graph=False, 
                create_graph=False
            )
            
            # Perturb node features
            if "node_features" in self.attack_mode:
                grad_x = grads[0]
                if self.normalize_grad:
                    grad_x = self._normalize_gradient(grad_x)
                else:
                    grad_x = grad_x.sign()
                x_adv = x_adv + self.alpha * grad_x
                x_adv = torch.clamp(x_adv, x - effective_eps, x + effective_eps)
            
            # Perturb adjacency (edge dropout - not gradient based, structural)
            if "edge_dropout" in self.attack_mode and adj_adv is not None:
                # Randomly drop edges during training
                dropout_mask = torch.rand_like(adj_adv) > self.edge_dropout_rate
                adj_adv = adj_adv * dropout_mask.float()
                # Re-normalize rows
                row_sum = adj_adv.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                adj_adv = adj_adv / row_sum
        
        return x_adv.detach(), adj_adv.detach() if adj_adv is not None else None


def create_adversarial_attack(
    method: str = "pgd",
    eps: float = 0.3,
    alpha: float = 0.01,
    steps: int = 7,
    probability: float = 1.0,
    feature_names: Optional[list[str]] = None,
    attack_mode: str = "node_features",
    edge_dropout_rate: float = 0.1,
    normalize_grad: bool = False,
    warmup_steps: int = 0,
    feature_eps_multipliers: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> AdversarialAttack:
    """
    Create an adversarial attack instance.
    
    Args:
        method: "pgd" | "fgsm" | "freelb" | "market_shock" | "graph_pgd"
        eps: L-infinity perturbation budget
        alpha: Step size (PGD, FreeLB)
        steps: Number of steps (PGD, FreeLB)
        probability: Probability of applying attack per batch
        feature_names: Feature names for column targeting
        attack_mode: For graph attacks - "node_features" | "edge_dropout" | "combined"
        edge_dropout_rate: Edge dropout rate for graph attacks
        normalize_grad: L2 normalize gradients (PGD, Graph PGD)
        warmup_steps: Gradually increase attack steps over this many steps
        feature_eps_multipliers: Per-dimension epsilon multipliers from pretrain vulnerability
    """
    method = method.lower()
    if method == "pgd":
        return PGDAttack(
            eps=eps, alpha=alpha, steps=steps, probability=probability, 
            feature_names=feature_names, normalize_grad=normalize_grad,
            warmup_steps=warmup_steps, feature_eps_multipliers=feature_eps_multipliers
        )
    elif method == "fgsm":
        return FGSMAttack(eps=eps, probability=probability, feature_names=feature_names)
    elif method == "freelb":
        return FreeLBAttack(eps=eps, alpha=alpha, steps=steps, probability=probability, feature_names=feature_names)
    elif method == "market_shock":
        return MarketShockGenerator(probability=probability, feature_names=feature_names)
    elif method == "graph_pgd":
        return GraphAdversarialAttack(
            eps=eps, alpha=alpha, steps=steps, probability=probability,
            feature_names=feature_names, attack_mode=attack_mode,
            edge_dropout_rate=edge_dropout_rate, normalize_grad=normalize_grad,
            warmup_steps=warmup_steps, feature_eps_multipliers=feature_eps_multipliers
        )
    else:
        raise ValueError(f"Unknown adversarial method: {method}. Choose from: pgd, fgsm, freelb, market_shock, graph_pgd")
