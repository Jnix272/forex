"""
Tests for curriculum learning (Improvement #9):
Difficulty curriculum, self-paced learning, loss-based weighting, integrated manager.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from training.curriculum import (
    CurriculumDataLoader,
    DifficultyCurriculum,
    DifficultyCurriculumConfig,
    LossBasedWeighting,
    LossWeightingConfig,
    SelfPacedConfig,
    SelfPacedLearning,
    compute_difficulty_scores,
    create_curriculum_manager,
)

# ---------------------------------------------------------------------------
# Difficulty Curriculum
# ---------------------------------------------------------------------------

def test_difficulty_curriculum_basic():
    """Test basic difficulty curriculum progression."""
    n = 1000
    difficulty = np.linspace(0, 1, n)
    config = DifficultyCurriculumConfig(
        n_levels=10,
        advance_rate=0.1,
        pace_function="linear",
        start_level=0.0,
        max_level=1.0,
    )
    curriculum = DifficultyCurriculum(config, difficulty)

    # Epoch 0: start level
    mask0 = curriculum.get_inclusion_mask(0)
    assert mask0.sum() >= 1  # At least some samples

    # Advance epochs
    mask50 = curriculum.get_inclusion_mask(50)
    assert mask50.sum() > mask0.sum()

    mask100 = curriculum.get_inclusion_mask(100)
    assert mask100.sum() == 1000  # All included


def test_difficulty_curriculum_pace_functions():
    """Test different pace functions."""
    difficulty = np.linspace(0, 1, 1000)

    for pace in ["linear", "exp", "sqrt", "step"]:
        config = DifficultyCurriculumConfig(
            pace_function=pace,
            advance_rate=0.1,
            start_level=0.0,
            max_level=1.0,
        )
        curr = DifficultyCurriculum(config, np.linspace(0, 1, 1000))
        level = curr.update(50)
        assert 0 <= level <= 1.0


def test_difficulty_curriculum_weights():
    """Test difficulty-based sample weights."""
    difficulty = np.linspace(0, 1, 100)
    config = DifficultyCurriculumConfig(advance_rate=0.1)
    curr = DifficultyCurriculum(config, difficulty)

    curr.update(0)
    w0 = curr.get_difficulty_weights()
    assert (w0 >= 0.1).all() and (w0 <= 1.0).all()

    curr.update(50)
    w50 = curr.get_difficulty_weights()
    assert (w50 <= w0).all()  # Weights should decrease as level increases


def test_difficulty_curriculum_sorted_indices():
    """Test sorted indices by difficulty."""
    difficulty = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
    config = DifficultyCurriculumConfig(advance_rate=0.5)
    curr = DifficultyCurriculum(config, difficulty)

    curr.update(0)
    idx = curr.get_sorted_indices()
    assert np.array_equal(difficulty[idx], np.sort(difficulty)[:len(idx)])


# ---------------------------------------------------------------------------
# Self-Paced Learning
# ---------------------------------------------------------------------------

def test_self_paced_learning_basic():
    """Test basic self-paced learning."""
    config = SelfPacedConfig(
        pace="linear",
        lambda_pace=1.0,
        total_epochs=100,
        min_fraction=0.1,
    )
    spl = SelfPacedLearning(config, n_samples=100)

    # Initial weights at min_fraction
    w0 = spl.get_weights(0)
    assert np.isclose(w0.mean(), 0.1, atol=0.05)

    # Advance epoch
    w50 = spl.get_weights(50)
    assert w50.mean() > w0.mean()


def test_self_paced_loss_update():
    """Test self-paced weight update from losses."""
    config = SelfPacedConfig(
        pace="linear",
        lambda_pace=1.0,
        total_epochs=100,
        min_fraction=0.1,
        use_loss_weighting=True,
        loss_temp=1.0,
    )
    spl = SelfPacedLearning(config, n_samples=100)

    # Easy samples (low loss) should get higher weights
    losses = np.linspace(0, 10, 100)
    weights = spl.update_weights(losses)

    assert weights[0] > weights[-1]  # Lower loss -> higher weight


def test_self_paced_pace_functions():
    """Test different pacing functions."""
    for pace in ["linear", "log", "root"]:
        config = SelfPacedConfig(pace=pace, total_epochs=100)
        spl = SelfPacedLearning(config, 100)

        p0 = spl.get_pace(0)
        p100 = spl.get_pace(100)
        assert p0 < p100


# ---------------------------------------------------------------------------
# Loss-Based Weighting
# ---------------------------------------------------------------------------

def test_loss_weighting_schemes():
    """Test different loss weighting schemes."""
    losses = np.linspace(0, 5, 100)

    for scheme in ["inverse", "focal", "threshold", "softmax", "curriculum"]:
        config = LossWeightingConfig(scheme=scheme)
        weighting = LossBasedWeighting(config)
        weights = weighting.compute_weights(losses, epoch=50)

        assert len(weights) == 100
        assert weights.min() >= 0.01
        assert weights.max() <= 10.0
        assert np.isclose(weights.mean(), 1.0, atol=0.1)


def test_loss_weighting_ema():
    """Test EMA smoothing of losses."""
    config = LossWeightingConfig(scheme="inverse", ema_decay=0.9)
    weighting = LossBasedWeighting(config)

    # First batch
    losses1 = np.full(10, 5.0)
    w1 = weighting.compute_weights(losses1, epoch=0)

    # Second batch - different losses
    losses2 = np.full(10, 1.0)
    w2 = weighting.compute_weights(losses2, epoch=1)

    # EMA should smooth
    assert not np.array_equal(w1, w2)


def test_loss_weighting_focal():
    """Test focal loss weighting."""
    config = LossWeightingConfig(scheme="focal", focal_gamma=2.0)
    weighting = LossBasedWeighting(config)

    losses = np.array([0.1, 1.0, 3.0, 5.0])
    weights = weighting.compute_weights(losses)

    # Higher loss -> higher weight for focal
    assert weights[-1] > weights[0]


# ---------------------------------------------------------------------------
# Curriculum Manager
# ---------------------------------------------------------------------------

def test_curriculum_manager_difficulty_mode():
    """Test CurriculumManager in difficulty mode."""
    n = 1000
    diff = np.linspace(0, 1, n)

    manager = create_curriculum_manager(
        mode="difficulty",
        n_samples=n,
        difficulty_scores=diff,
        advance_rate=0.1,
    )

    info = manager.update(0)
    assert info["epoch"] == 0
    assert "weights" in info
    assert len(info["weights"]) == n

    info100 = manager.update(100)
    assert info100["difficulty_level"] > 0


def test_curriculum_manager_self_paced_mode():
    """Test CurriculumManager in self-paced mode."""
    n = 100
    manager = create_curriculum_manager(
        mode="self_paced",
        n_samples=n,
        total_epochs=100,
    )

    losses = np.random.rand(n) * 5
    info = manager.update(50, losses=losses)

    assert "weights" in info
    assert "self_paced_pace" in info
    assert info["self_paced_pace"] > 0


def test_curriculum_manager_loss_weighting_mode():
    """Test CurriculumManager in loss-weighting mode."""
    n = 100
    manager = create_curriculum_manager(
        mode="loss_weighting",
        n_samples=n,
        lw_scheme="focal",
    )

    losses = np.random.rand(n) * 5
    info = manager.update(10, losses=losses)

    assert "weights" in info
    assert "loss_weighting_stats" in info


def test_curriculum_manager_adaptive_mode():
    """Test CurriculumManager in adaptive mode."""
    n = 100
    manager = create_curriculum_manager(
        mode="adaptive",
        n_samples=n,
    )

    info = manager.update(10, val_metrics={"val_sharpe": 0.8, "val_loss": 0.5})

    assert "weights" in info
    assert "adaptive_actions" in info


def test_curriculum_manager_combined_mode():
    """Test CurriculumManager in combined mode."""
    n = 1000
    diff = np.linspace(0, 1, n)

    manager = create_curriculum_manager(
        mode="combined",
        n_samples=n,
        difficulty_scores=diff,
        total_epochs=100,
        lw_scheme="focal",
    )

    losses = np.random.rand(n) * 5
    val_metrics = {"val_accuracy": 0.8, "val_sharpe": 1.2, "val_loss": 0.3}

    info = manager.update(50, val_metrics=val_metrics, losses=losses)

    assert "weights" in info
    assert "difficulty_level" in info
    assert "self_paced_pace" in info
    assert "loss_weighting_stats" in info
    assert "adaptive_actions" in info


def test_curriculum_manager_state_dict():
    """Test state dict serialization."""
    n = 100
    manager = create_curriculum_manager(
        mode="combined",
        n_samples=n,
        difficulty_scores=np.linspace(0, 1, n),
        total_epochs=100,
    )

    manager.update(50, losses=np.random.rand(n) * 5)
    state = manager.state_dict()

    assert "epoch" in state
    assert "config_mode" in state
    assert state["epoch"] == 50


# ---------------------------------------------------------------------------
# Difficulty Scoring
# ---------------------------------------------------------------------------

def test_compute_difficulty_scores_margin():
    """Test margin-based difficulty scoring."""
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)
        def forward(self, x):
            return self.fc(x)

    model = SimpleClassifier()
    features = np.random.randn(100, 10).astype(np.float32)
    labels = np.random.randint(0, 3, 100)

    scores = compute_difficulty_scores(features, labels, method="margin", model=model)
    assert len(scores) == 100
    assert (scores >= 0).all() and (scores <= 1).all()


def test_compute_difficulty_scores_loss():
    """Test loss-based difficulty scoring."""
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)
        def forward(self, x):
            return self.fc(x)

    model = SimpleClassifier()
    features = np.random.randn(100, 10).astype(np.float32)
    labels = np.random.randint(0, 3, 100)

    scores = compute_difficulty_scores(features, labels, method="loss", model=model)
    assert len(scores) == 100
    assert (scores >= 0).all() and (scores <= 1).all()


def test_compute_difficulty_scores_entropy():
    """Test entropy-based difficulty scoring."""
    labels = np.random.randint(0, 3, 100)
    scores = compute_difficulty_scores(np.zeros((100, 10)), labels, method="entropy")
    assert len(scores) == 100
    assert (scores >= 0).all() and (scores <= 1).all()


# ---------------------------------------------------------------------------
# CurriculumDataLoader
# ---------------------------------------------------------------------------

def test_curriculum_dataloader():
    """Test CurriculumDataLoader integration."""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, n=1000):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return torch.randn(10), torch.randint(0, 2, (1,)).item()

    n = 1000
    diff = np.linspace(0, 1, n)
    manager = create_curriculum_manager(
        mode="difficulty",
        n_samples=n,
        difficulty_scores=np.linspace(0, 1, n),
        advance_rate=0.1,
    )

    dataset = DummyDataset(n)
    loader = CurriculumDataLoader(dataset, manager, batch_size=32)

    manager.update(50)
    loader.set_epoch(50)

    batches = list(loader)
    assert len(batches) > 0
    x, y = batches[0]
    assert x.shape[0] <= 32


def test_curriculum_manager_inclusion_mask_flow():
    """C4 wiring: update() + inclusion mask yields a usable sample filter."""
    import numpy as np

    from training.curriculum import create_curriculum_manager

    diff = np.linspace(0.0, 1.0, 1000)
    mgr = create_curriculum_manager(
        mode="combined",
        n_samples=1000,
        difficulty_scores=diff,
        total_epochs=10,
    )
    info = mgr.update(epoch=1, losses=None)
    mask = mgr.get_inclusion_mask()
    assert len(mask) == 1000
    assert mask.dtype == bool
    assert 0 < mask.sum() <= 1000
    assert "weights" in info
    assert float(np.asarray(info["weights"]).mean()) > 0


def test_curriculum_manager_trains_gpu_arg(monkeypatch):
    """C4 wiring: --curriculum-manager / --curriculum-manager-mode args parse."""
    import sys

    from training.train_gpu import parse_args
    monkeypatch.setattr(
        sys, "argv",
        ["train_gpu", "--model", "tft", "--curriculum-manager",
         "--curriculum-manager-mode", "self_paced"],
    )
    args = parse_args()
    assert args.curriculum_manager is True
    assert args.curriculum_manager_mode == "self_paced"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
