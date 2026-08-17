"""
Elastic Weight Consolidation (EWC) for Continuous Learning.
Prevents catastrophic forgetting when training on new regimes.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class ElasticWeightConsolidation(nn.Module):
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
        super().__init__()
        self.model = model
        self.device = device
        self.dataset = dataset
        self.max_samples = max_samples
        self.classification = bool(classification)
        self.loss_fn = loss_fn

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        for n, p in self.params.items():
            self.register_buffer(f"saved_{n.replace('.', '_')}", p.clone().detach().requires_grad_(False))

        fisher = self._compute_fisher_diagonal()
        for n, f in fisher.items():
            self.register_buffer(f"fisher_{n.replace('.', '_')}", f.clone().detach().requires_grad_(False))

    def _task_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_fn is not None:
            return self.loss_fn(outputs, labels)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        is_multiclass_output = outputs.ndim >= 2 and outputs.shape[-1] > 1
        if self.classification or is_multiclass_output:
            if labels.dtype.is_floating_point:
                # Signed {-1, 0, +1} float labels → class index {0, 1, 2}
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
        """Compute the diagonal of the Fisher Information Matrix.

        EWC fix (audit `docs/AUDIT_2026-08-07_metrics_models_pretrain.md`,
        remediation #4): the previous implementation divided each batch's
        per-sample-mean ``grad²`` by ``self.max_samples`` *inside* the
        loop. This had two defects:

        1. **Batch-size inflation**: grad² of a per-sample-mean loss scales
           with `1/batch_size`, so updating `fisher += grad² / max_samples`
           over `~max_samples/batch_size` iterations produces a Fisher
           diagonal whose magnitude still couples to batch size.
        2. **Underweighting when dataset < max_samples**: when the actual
           dataset is smaller than `max_samples`, the loop iterates fewer
           times (samples_processed << max_samples) but the divisor is the
           flat `max_samples` constant - the Fisher diagonal gets diluted
           by the unfilled `(max_samples - samples_processed)` budget.
           Audit example: a 65-sample dataset gave a Fisher diagonal ~15x
           too small relative to the same diagonal on max_samples=1000.

        The fix accumulates raw per-sample-mean `grad²` over each batch
        (no per-step divisor) and normalises ONCE after the loop by the
        actual `samples_processed`. The Fisher diagonal is now a true
        per-sample mean of grad², invariant to both batch size and dataset
        size (matching the per-sample-mean convention used by the loss
        function throughout the rest of training).
        """
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
                    # A4/EWC fix: raw accumulation; normalise once after the loop.
                    fisher[n] += p.grad.data**2

            samples_processed += features.size(0)

        # Fisher = (1 / N) * sum_i grad²  (per-sample mean)
        # Use max(1) to avoid divide-by-zero when no batches ran (empty ds).
        norm = max(samples_processed, 1)
        fisher = {n: (p / norm).clone().detach() for n, p in fisher.items()}
        # Persist on the instance for diagnostics/telemetry
        self.last_samples_processed = samples_processed
        # Clear gradients to prevent leaking into the next optimizer step.
        self.model.zero_grad()
        self.model.train()
        return fisher

    def penalty(self) -> torch.Tensor:
        """Calculate the EWC penalty term."""
        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.params.items():
            name = n.replace(".", "_")
            saved_p = getattr(self, f"saved_{name}")
            fisher = getattr(self, f"fisher_{name}")
            _loss = fisher * (p - saved_p) ** 2
            loss = loss + _loss.sum()
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
