"""
Elastic Weight Consolidation (EWC) for Continuous Learning.
Prevents catastrophic forgetting when training on new regimes.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from copy import deepcopy

class ElasticWeightConsolidation:
    def __init__(self, model: nn.Module, dataset, device: torch.device, max_samples: int = 1000):
        """
        Initialize EWC and compute Fisher Information Matrix diagonal.
        """
        self.model = model
        self.device = device
        self.dataset = dataset
        self.max_samples = max_samples
        
        # Store original weights
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.saved_params = {n: p.clone().detach() for n, p in self.params.items()}
        self.fisher = self._compute_fisher_diagonal()

    def _compute_fisher_diagonal(self) -> Dict[str, torch.Tensor]:
        """Compute the diagonal of the Fisher Information Matrix."""
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()
        
        # We need a dataloader to sample from the baseline dataset
        from torch.utils.data import DataLoader
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= self.max_samples:
                break
                
            self.model.zero_grad()
            
            # Assuming batch is a tuple: (features, labels)
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                features, labels = batch[0].to(self.device), batch[1].to(self.device)
            elif isinstance(batch, dict):
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
            else:
                continue

            outputs = self.model(features)
            
            # Log-likelihood (Negative Cross Entropy is standard for classification EWC)
            # If regression, use MSE
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            for n, p in self.params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / self.max_samples
                    
            samples_processed += features.size(0)

        # Freeze fisher components
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

def apply_ewc_loss(base_loss: torch.Tensor, ewc: Optional[ElasticWeightConsolidation], lambda_ewc: float = 1000.0) -> torch.Tensor:
    """Add EWC penalty to the base loss."""
    if ewc is None or lambda_ewc == 0.0:
        return base_loss
    return base_loss + (lambda_ewc / 2) * ewc.penalty()
