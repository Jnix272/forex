"""
tests/test_ensemble_risk.py
===========================
Verifies that EnsembleMetaLearner computes disagreement and EnsembleRiskFilter
determines the correct trade size multipliers.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from models.ensemble import EnsembleMetaLearner, EnsembleRiskFilter


class DummyBaseModel(torch.nn.Module):
    def __init__(self, fixed_pred):
        super().__init__()
        self.fixed_pred = fixed_pred

    def forward(self, x):
        # Return a tensor of shape (B, 1) filled with fixed_pred
        return torch.full((x.shape[0], 1), self.fixed_pred, dtype=torch.float32, device=x.device)


def test_ensemble_disagreement():
    B = 2
    T = 5
    F = 10
    x = torch.randn(B, T, F)

    # 1. Strong agreement
    bases_agree = [DummyBaseModel(1.0), DummyBaseModel(1.0), DummyBaseModel(1.0)]
    ens_agree = EnsembleMetaLearner(bases_agree, context_dim=8, hidden=16)
    _out_agree, dis_agree = ens_agree.predict_with_disagreement(x)

    assert torch.all(dis_agree < 0.1), "Disagreement should be near zero for identical models."

    # 2. Strong disagreement
    bases_disagree = [DummyBaseModel(-1.0), DummyBaseModel(0.0), DummyBaseModel(1.0)]
    ens_disagree = EnsembleMetaLearner(bases_disagree, context_dim=8, hidden=16)
    _out_disagree, dis_disagree = ens_disagree.predict_with_disagreement(x)

    assert torch.all(dis_disagree > 0.5), "Disagreement should be high for opposing models."

    # 3. Risk Filter Policy
    filter = EnsembleRiskFilter(low_threshold=0.3, high_threshold=0.7)

    # Agreement -> full size
    assert filter.compute_size_multiplier(float(dis_agree[0])) == 1.0

    # Disagreement -> hold
    assert filter.compute_size_multiplier(float(dis_disagree[0])) == 0.0

    print("test_ensemble_disagreement passed!")


if __name__ == "__main__":
    test_ensemble_disagreement()
