"""
Elastic Weight Consolidation (EWC) for Continuous Learning.
Prevents catastrophic forgetting when training on new regimes.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class ElasticWeightConsolidation:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        device: torch.device,
        max_samples: int = 1000,
        *,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        classification: bool = False,
    ):
        """
        Initialize EWC and compute Fisher Information Matrix diagonal.

        ``loss_fn(outputs, labels)`` should match the supervised training loss.
        When omitted, classification uses cross-entropy and regression uses MSE.
        """
        self.model = model
        self.device = device
        self.dataset = dataset
        self.max_samples = max_samples
        self.classification = bool(classification)
        self.loss_fn = loss_fn

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.saved_params = {n: p.clone().detach() for n, p in self.params.items()}
        self.fisher = self._compute_fisher_diagonal()

    def _task_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_fn is not None:
            return self.loss_fn(outputs, labels)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if self.classification or (
            outputs.ndim >= 2 and outputs.shape[-1] > 1
            and labels.dtype in (torch.long, torch.int64, torch.int32)
        ):
            if labels.dtype.is_floating_point:
                # Signed {-1,0,+1} → class index {0,1,2}
                y = (labels.reshape(-1).clamp(-1, 1) + 1).round().long()
            else:
                y = labels.reshape(-1).long()
            y = y.clamp(0, outputs.shape[-1] - 1)
            return nn.functional.cross_entropy(outputs, y)

        pred = outputs.reshape(-1)
        tgt = labels.reshape(-1).float()
        n = min(pred.numel(), tgt.numel())
        return nn.functional.mse_loss(pred[:n], tgt[:n])

    def _compute_fisher_diagonal(self) -> dict[str, torch.Tensor]:
        """Compute the diagonal of the Fisher Information Matrix."""
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()

        from torch.utils.data import DataLoader
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)

        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= self.max_samples:
                break

            self.model.zero_grad()

            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                features, labels = batch[0].to(self.device), batch[1].to(self.device)
            elif isinstance(batch, dict):
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
            else:
                continue

            outputs = self.model(features)
            if isinstance(outputs, tuple):
                # Use primary head for Fisher estimate
                loss = self._task_loss(outputs[0], labels)
            else:
                loss = self._task_loss(outputs, labels)
            loss.backward()

            for n, p in self.params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / self.max_samples

            samples_processed += features.size(0)

        fisher = {n: p.clone().detach() for n, p in fisher.items()}
        self.model.train()
        return fisher

    def penalty(self) -> torch.Tensor:
        """Calculate the EWC penalty term."""
        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.params.items():
            _loss = self.fisher[n] * (p - self.saved_params[n]) ** 2
            loss += _loss.sum()
        return loss


def apply_ewc_loss(
    base_loss: torch.Tensor,
    ewc: ElasticWeightConsolidation | None,
    lambda_ewc: float = 1000.0,
) -> torch.Tensor:
    """Add EWC penalty to the base loss."""
    if ewc is None or lambda_ewc == 0.0:
        return base_loss
    return base_loss + (lambda_ewc / 2) * ewc.penalty()
