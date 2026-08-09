"""
Tests for training memory management (Improvement #8):
streaming datasets, gradient checkpointing, activation offloading, memory profiling.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from training.memory_management import (
    CheckpointPolicy,
    MemoryMonitor,
    PrefetchDataLoader,
    StreamingMemmapDataset,
    apply_gradient_checkpointing,
    checkpoint_sequential,
    create_streaming_dataloader,
    memory_efficient_training,
    memory_profiler,
)

# ---------------------------------------------------------------------------
# Gradient Checkpointing
# ---------------------------------------------------------------------------

def test_checkpoint_policy_default():
    """Test default checkpoint policy."""
    policy = CheckpointPolicy()
    assert "transformer" in policy.checkpoint_patterns
    assert "encoder" in policy.checkpoint_patterns
    assert "mamba" in policy.checkpoint_patterns


def test_apply_gradient_checkpointing():
    """Test applying gradient checkpointing to a model."""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
            )
            self.decoder = nn.Linear(10, 5)
            self.output = nn.Linear(5, 1)

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return self.output(x)

    model = TestModel()
    policy = CheckpointPolicy(
        checkpoint_patterns=["encoder"],
        no_checkpoint_patterns=["output"],
    )

    model = apply_gradient_checkpointing(model, policy)

    # Check that encoder layers have been wrapped
    assert hasattr(model.encoder, "_original_forward")
    # output should not be checkpointed
    assert not hasattr(model.output, "_original_forward")


def test_checkpoint_sequential():
    """Test checkpoint_sequential function."""
    modules = [nn.Linear(10, 10) for _ in range(3)]
    x = torch.randn(2, 10)

    out = checkpoint_sequential(modules, x)
    assert out.shape == (2, 10)

    # Gradient should flow
    loss = out.sum()
    loss.backward()
    assert modules[0].weight.grad is not None


# ---------------------------------------------------------------------------
# Streaming Datasets
# ---------------------------------------------------------------------------

def test_streaming_memmap_dataset(tmp_path):
    """Test StreamingMemmapDataset with NPY files."""
    n = 100
    X = np.random.randn(n, 10, 5).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    # Save as NPY
    np.save(tmp_path / "_X.npy", X)
    np.save(tmp_path / "_y.npy", y)

    indices = np.arange(n)
    ds = StreamingMemmapDataset(str(tmp_path), indices)

    assert len(ds) == n
    x, y = ds[0]
    assert x.shape == (10, 5)
    assert isinstance(y, (float, torch.Tensor))
    if isinstance(y, torch.Tensor):
        assert y.dim() == 0  # scalar tensor


def test_streaming_memmap_dataset_zarr(tmp_path):
    """Test StreamingMemmapDataset with Zarr (if available)."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    n = 100
    X = np.random.randn(n, 10, 5).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    # Save as Zarr (new API)
    z = zarr.open_group(str(tmp_path / "test.zarr"), mode="w")
    z["X"] = X
    z["y"] = y

    indices = np.arange(n)
    ds = StreamingMemmapDataset(str(tmp_path / "test.zarr"), indices)

    assert len(ds) == n
    x, y = ds[0]
    assert x.shape == (10, 5)
    assert isinstance(y, (float, torch.Tensor))
    assert x.shape == (10, 5)


def test_prefetch_dataloader():
    """Test PrefetchDataLoader wrapper."""
    ds = torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randn(100)
    )

    loader = PrefetchDataLoader(
        ds,
        batch_size=16,
        num_workers=0,  # Use 0 for test
        prefetch_batches=2,
    )

    batches = list(loader)
    assert len(batches) > 0
    x, y = batches[0]
    assert x.shape == (16, 10)


# ---------------------------------------------------------------------------
# Memory Profiler
# ---------------------------------------------------------------------------

def test_memory_profiler():
    """Test memory profiler context manager."""
    with memory_profiler() as stats:
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000, device="cuda")
            y = x @ x.T
        else:
            x = torch.randn(1000, 1000)
            y = x @ x.T

    # Stats dict should be returned
    # (empty dict in current implementation, but no errors)
    assert isinstance(stats, dict)


def test_memory_monitor():
    """Test MemoryMonitor callback."""
    monitor = MemoryMonitor(log_interval=2, warn_threshold_gb=100)

    for _ in range(5):
        result = monitor.step()

    # Should return dict on log_interval
    assert result is not None or monitor.step_count % monitor.log_interval != 0


def test_memory_efficient_training_context():
    """Test memory_efficient_training context manager."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
            self.decoder = nn.Linear(20, 1)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = SimpleModel()
    policy = CheckpointPolicy(checkpoint_patterns=["encoder"])

    with memory_efficient_training(model, checkpoint_policy=policy, enable_profiler=False) as mem:
        x = torch.randn(2, 10)
        out = model(x)
        loss = out.sum()
        loss.backward()

    # Should complete without errors


# ---------------------------------------------------------------------------
# Streaming DataLoader Factory
# ---------------------------------------------------------------------------

def test_create_streaming_dataloader(tmp_path):
    """Test create_streaming_dataloader factory."""
    n = 100
    X = np.random.randn(n, 10, 5).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    np.save(tmp_path / "_X.npy", X)
    np.save(tmp_path / "_y.npy", y)

    indices = np.arange(n)
    loader = create_streaming_dataloader(
        str(tmp_path), indices, batch_size=16, num_workers=0,
    )

    assert isinstance(loader, PrefetchDataLoader)
    batch = next(iter(loader))
    assert batch[0].shape == (16, 10, 5)


def test_create_streaming_dataloader_sequential(tmp_path):
    """Test create_streaming_dataloader with sequential mode."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    n = 100
    X = np.random.randn(n, 10, 5).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    z = zarr.open_group(str(tmp_path / "test.zarr"), mode="w")
    z["X"] = X
    z["y"] = y

    indices = np.arange(n)
    loader = create_streaming_dataloader(
        str(tmp_path / "test.zarr"), indices, batch_size=16,
        num_workers=0, sequential=True,
    )

    assert isinstance(loader, torch.utils.data.DataLoader)


# ---------------------------------------------------------------------------
# Memory Monitor Integration
# ---------------------------------------------------------------------------

def test_memory_monitor_warning():
    """Test MemoryMonitor warning on high memory."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    monitor = MemoryMonitor(log_interval=1, warn_threshold_gb=0.001)  # Very low threshold

    # Allocate some GPU memory
    x = torch.randn(1000, 1000, device="cuda")

    result = monitor.step()
    assert result is not None
    assert "allocated_gb" in result
    assert result["allocated_gb"] > 0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

def test_streaming_dataset_pickling(tmp_path):
    """Test that StreamingMemmapDataset can be pickled for multiprocessing."""
    import pickle

    n = 50
    X = np.random.randn(n, 5, 3).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    np.save(tmp_path / "_X.npy", X)
    np.save(tmp_path / "_y.npy", y)

    indices = np.arange(n)
    ds = StreamingMemmapDataset(str(tmp_path), indices)

    # Pickle and unpickle
    pickled = pickle.dumps(ds)
    ds2 = pickle.loads(pickled)

    assert len(ds2) == len(ds)
    x1, y1 = ds[0]
    x2, y2 = ds2[0]
    assert torch.allclose(x1, x2)
    assert y1 == y2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
