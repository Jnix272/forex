import torch
import torch.nn as nn

from models.ensemble import EnsembleMetaLearner, EnsembleRiskFilter, _base_pred_to_batch_vector, train_meta_learner


class MockModelSingle(nn.Module):
    def forward(self, x):
        # x is (B, T, F)
        # return (B, 1)
        return x.mean(dim=(1, 2), keepdim=True)

class MockModelTuple(nn.Module):
    def forward(self, x):
        # return (direction, return_hat, confidence)
        B = x.shape[0]
        direction = torch.zeros(B, 3)
        return_hat = torch.ones(B, 1) * 0.5
        confidence = torch.ones(B, 1)
        return (direction, return_hat, confidence)

class MockModelMultitask(nn.Module):
    def forward(self, x):
        # mock multitask where first is (B, 3)
        B = x.shape[0]
        first = torch.zeros(B, 3)
        second = x.mean(dim=(1, 2), keepdim=True)
        third = torch.zeros(B, 1)
        return (first, second, third)

def test_base_pred_to_batch_vector():
    B = 4
    # Test tensor (B, 1)
    t1 = torch.randn(B, 1)
    out1 = _base_pred_to_batch_vector(t1)
    assert out1.shape == (B,)
    assert torch.allclose(out1, t1.squeeze(-1))

    # Test tensor (B,)
    t2 = torch.randn(B)
    out2 = _base_pred_to_batch_vector(t2)
    assert out2.shape == (B,)
    assert torch.allclose(out2, t2)

    # Test tuple with multitask pattern
    first = torch.randn(B, 3)
    second = torch.randn(B, 1)
    third = torch.randn(B, 1)
    out3 = _base_pred_to_batch_vector((first, second, third))
    assert out3.shape == (B,)
    assert torch.allclose(out3, second.squeeze(-1))

    # Test normal tuple fallback
    out4 = _base_pred_to_batch_vector((t1, t2))
    assert out4.shape == (B,)
    assert torch.allclose(out4, t1.squeeze(-1))

    # CE-only (B, 3) logits → buy − sell signed score (not B*3 flatten)
    logits = torch.tensor([[1.0, 0.0, 3.0], [2.0, 0.0, 0.5]])
    out5 = _base_pred_to_batch_vector(logits)
    assert out5.shape == (2,)
    assert torch.allclose(out5, torch.tensor([2.0, -1.5]))

def test_ensemble_meta_learner_initialization():
    models = [MockModelSingle(), MockModelSingle()]
    ensemble = EnsembleMetaLearner(base_models=models, context_dim=16, hidden=32)
    assert len(ensemble.bases) == 2
    assert ensemble.n_models == 2

def test_ensemble_meta_learner_forward():
    B, T, F = 8, 10, 5
    x = torch.randn(B, T, F)
    models = [MockModelSingle(), MockModelTuple(), MockModelMultitask()]
    ensemble = EnsembleMetaLearner(base_models=models, context_dim=16, hidden=32)

    output, weights = ensemble(x)
    assert output.shape == (B,)
    assert weights.shape == (B, 3)
    # Weights should sum to 1 over the models
    assert torch.allclose(weights.sum(dim=1), torch.ones(B))

def test_ensemble_model_weights_summary():
    B, T, F = 4, 5, 2
    x = torch.randn(B, T, F)
    models = [MockModelSingle(), MockModelTuple()]
    ensemble = EnsembleMetaLearner(base_models=models, base_names=["m1", "m2"])

    summary = ensemble.model_weights_summary(x)
    assert isinstance(summary, dict)
    assert "m1" in summary and "m2" in summary
    assert len(summary) == 2
    assert abs(sum(summary.values()) - 1.0) < 1e-5

def test_ensemble_diversity_loss():
    models = [MockModelSingle(), MockModelSingle()]
    ensemble = EnsembleMetaLearner(base_models=models)

    # Create artificial predictions
    B = 10
    # Perfect positive correlation
    preds_pos = torch.randn(B, 1).repeat(1, 2)
    div_pos = ensemble.diversity_loss(preds_pos)
    assert torch.allclose(div_pos, torch.tensor(1.0), atol=1e-4)

    # Perfect negative correlation
    preds_neg = torch.cat([preds_pos[:, 0:1], -preds_pos[:, 0:1]], dim=1)
    div_neg = ensemble.diversity_loss(preds_neg)
    assert torch.allclose(div_neg, torch.tensor(-1.0), atol=1e-4)

    # Zero correlation (orthogonal) - random large vectors
    B_large = 10000
    preds_rand = torch.randn(B_large, 2)
    div_rand = ensemble.diversity_loss(preds_rand)
    assert abs(div_rand.item()) < 0.1

def test_ensemble_predict_with_disagreement():
    B, T, F = 4, 5, 2
    x = torch.randn(B, T, F)
    models = [MockModelSingle(), MockModelSingle()]
    ensemble = EnsembleMetaLearner(base_models=models)

    output, disagreement = ensemble.predict_with_disagreement(x)
    assert output.shape == (B,)
    assert disagreement.shape == (B,)
    assert (disagreement >= 0).all()

def test_train_meta_learner(tmp_path):
    B, T, F = 16, 5, 4
    n_batches = 3

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return B * n_batches
        def __getitem__(self, idx):
            return torch.randn(T, F), torch.randn(1).squeeze()

    loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=B)

    models = [MockModelSingle(), MockModelSingle()]
    ensemble = EnsembleMetaLearner(base_models=models)

    ckpt_path = tmp_path / "meta_ckpt.pt"

    history = train_meta_learner(
        meta=ensemble,
        loader=loader,
        epochs=2,
        lr=1e-3,
        diversity_weight=0.1,
        device="cpu",
        verbose=False,
        checkpoint_path=str(ckpt_path)
    )

    assert len(history) == 2
    assert ckpt_path.exists()
    assert ckpt_path.with_name("meta_ckpt_latest.pt").exists()
    assert ckpt_path.with_name(ckpt_path.name + ".json").exists()

def test_ensemble_risk_filter():
    filter = EnsembleRiskFilter(low_threshold=0.5, high_threshold=1.0)
    assert filter.compute_size_multiplier(0.1) == 1.0
    assert filter.compute_size_multiplier(0.7) == 0.5
    assert filter.compute_size_multiplier(1.2) == 0.0
