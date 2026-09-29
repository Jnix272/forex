import numpy as np
import torch

from pretrain.contrastive import BYOLTrainer


class DummyEncoder(torch.nn.Module):
    def __init__(self, seq_len=32, num_features=10, d_model=128):
        super().__init__()
        self.linear = torch.nn.Linear(seq_len * num_features, d_model)

    def forward(self, x):
        # x shape: (B, T, F)
        return self.linear(x.reshape(x.shape[0], -1))


def test_byol_trainer_runs_one_epoch(tmp_path):
    X = np.random.default_rng(42).standard_normal((100, 32, 10)).astype(np.float32)
    trainer = BYOLTrainer(encoder=DummyEncoder(), d_model=128, proj_dim=32, pred_dim=16, lr=1e-3, device="cpu", seed=0)
    ckpt = tmp_path / "byol_encoder.pt"
    history = trainer.pretrain(X, epochs=1, batch_size=8, checkpoint_path=str(ckpt))
    assert len(history["loss"]) == 1
    assert np.isfinite(history["loss"][0])
    assert ckpt.exists()
