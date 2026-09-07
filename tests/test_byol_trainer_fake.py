import pytest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pretrain.contrastive import BYOLTrainer

class DummyEncoder(torch.nn.Module):
    def __init__(self, seq_len=32, num_features=10, d_model=128):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.linear = torch.nn.Linear(seq_len * num_features, d_model)
    def forward(self, x):
        # x shape: (B, T, F)
        B = x.shape[0]
        return self.linear(x.view(B, -1))

class FakeTimeSeriesDataset(Dataset):
    """Simple dataset returning random time‑series tensors."""
    def __init__(self, length: int = 100, seq_len: int = 32, num_features: int = 10):
        self.length = length
        self.seq_len = seq_len
        self.num_features = num_features
        self.rng = np.random.default_rng(42)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        arr = self.rng.standard_normal((self.seq_len, self.num_features)).astype(np.float32)
        return torch.from_numpy(arr)

@pytest.fixture
def fake_loader():
    dataset = FakeTimeSeriesDataset()
    return DataLoader(dataset, batch_size=8, shuffle=True)

def test_byol_trainer_runs_one_epoch(fake_loader):
    encoder = DummyEncoder()
    trainer = BYOLTrainer(
        encoder=encoder,
        epochs=1,
        batch_size=8,
        device="cpu",
        lr=1e-3,
        use_amp=False,
    )
    trainer.fit(fake_loader)
    # Verify training history
    assert hasattr(trainer, "history")
    assert "loss" in trainer.history
    assert len(trainer.history["loss"]) == 1
    loss_val = trainer.history["loss"][0]
    assert np.isfinite(loss_val)
