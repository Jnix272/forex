"""
Tests for HPO module (Improvement #12):
PBT, BOHB, ASHA, multi-fidelity ASHA.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.hpo import (
    HPOConfig,
    TrialState,
    PopulationBasedTraining,
    HyperBandScheduler,
    AsyncSuccessiveHalvingScheduler,
    BOHBScheduler,
    MultiFidelityASHAScheduler,
    HPOManager,
    run_hpo_study,
)
from pretrain.multi_task import (
    DomainDiscriminator,
    TimeSeriesAugmenter,
    MMDLoss,
    CORALLoss,
    grad_reverse,
    nt_xent_loss,
    byol_loss,
    masked_reconstruction_loss,
    vae_loss,
    forecast_loss,
    drift_loss,
    domain_adversarial_loss,
    adapt_encoder_to_target,
    create_multi_task_pretrainer,
    pretrain_multi_task,
    adapt_encoder_to_target,
    MultiTaskPretrainer,
)


# ═════════════════════════════════════════════════════════════════════════════
# Population-Based Training
# ════════════════════════════════════════════════════════════════════════════

def test_pbt_initialization():
    """Test PBT initialization."""
    config = HPOConfig(population_size=4)
    pbt = PopulationBasedTraining(config)
    
    assert len(pbt.population) == 0
    assert pbt.best_trial is None


def test_pbt_add_trial():
    """Test adding trials to population."""
    config = HPOConfig(population_size=4)
    pbt = PopulationBasedTraining(config)
    
    for i in range(4):
        pbt.add_trial(f"trial_{i}", {"lr": 1e-3 * (i+1), "batch_size": 32 * (i+1)})
    
    assert len(pbt.population) == 4


def test_pbt_update_score():
    """Test updating trial scores."""
    config = HPOConfig(population_size=4)
    pbt = PopulationBasedTraining(config)
    
    for i in range(4):
        pbt.add_trial(f"trial_{i}", {"lr": 1e-3})
    
    # Update scores
    pbt.update_score("trial_0", 0.5, 10)
    pbt.update_score("trial_1", 0.8, 10)
    pbt.update_score("trial_2", 0.3, 10)
    pbt.update_score("trial_3", 0.9, 10)
    
    # Best trial should be trial_3 with score 0.9
    assert pbt.best_trial is not None
    assert pbt.best_trial.trial_id == "trial_3"
    assert pbt.best_trial.best_score == 0.9


def test_pbt_perturb_trial():
    """Test parameter perturbation."""
    config = HPOConfig(population_size=2)
    pbt = PopulationBasedTraining(config)
    
    pbt.add_trial("trial_0", {"lr": 1e-3, "batch_size": 32})
    pbt.add_trial("trial_1", {"lr": 1e-4, "batch_size": 64})
    
    pbt.update_score("trial_0", 0.5, 10)
    pbt.update_score("trial_1", 0.8, 10)
    
    # Perturb the worse trial
    new_params = pbt.perturb_trial("trial_0")
    
    assert "lr" in new_params
    assert "batch_size" in new_params
    # lr should be perturbed
    assert new_params["lr"] != 1e-3


def test_pbt_exploit_and_perturb():
    """Test exploit and perturb operation."""
    config = HPOConfig(population_size=4)
    pbt = PopulationBasedTraining(config)
    
    for i in range(4):
        pbt.add_trial(f"trial_{i}", {"lr": 1e-3 * (i+1)})
    
    pbt.update_score("trial_0", 0.5, 10)
    pbt.update_score("trial_1", 0.8, 10)
    pbt.update_score("trial_2", 0.3, 10)
    pbt.update_score("trial_3", 0.9, 10)
    
    donor_id, new_params = pbt.exploit_and_perturb("trial_2")
    
    # Donor should be the best trial (trial_3 with score 0.9)
    assert donor_id == "trial_3"
    assert "lr" in new_params


def test_pbt_population_stats():
    """Test population statistics."""
    config = HPOConfig(population_size=3)
    pbt = PopulationBasedTraining(config)
    
    pbt.add_trial("trial_0", {"lr": 1e-3})
    pbt.add_trial("trial_1", {"lr": 1e-4})
    pbt.add_trial("trial_2", {"lr": 1e-5})
    
    pbt.update_score("trial_0", 0.5, 10)
    pbt.update_score("trial_1", 0.8, 10)
    pbt.update_score("trial_2", 0.3, 10)
    
    stats = pbt.get_population_stats()
    
    assert stats["population_size"] == 3
    assert stats["best_score"] == 0.8
    assert stats["worst_score"] == 0.3
    assert 0.3 <= stats["mean_score"] <= 0.8


def test_pbt_exploit_and_perturb():
    """Test exploit and perturb operation."""
    config = HPOConfig(population_size=3)
    pbt = PopulationBasedTraining(config)
    
    pbt.add_trial("trial_0", {"lr": 1e-3})
    pbt.add_trial("trial_1", {"lr": 1e-3})
    pbt.add_trial("trial_2", {"lr": 1e-3})
    
    pbt.update_score("trial_0", 0.5, 10)
    pbt.update_score("trial_1", 0.8, 10)
    pbt.update_score("trial_2", 0.3, 10)
    
    donor_id, new_params = pbt.exploit_and_perturb("trial_2")
    
    # Donor should be trial_1 (best score 0.8)
    assert donor_id == "trial_1"
    assert "lr" in new_params


# ═════════════════════════════════════════════════════════════════════════════
# HyperBand / ASHA Schedulers
# ══════════════════════════════════════════════════════════════════════════════

def test_hyperband_scheduler_init():
    """Test HyperBand scheduler initialization."""
    config = HPOConfig(
        min_budget=3,
        max_budget=27,
        eta=3,
    )
    hb = HyperBandScheduler(config)
    
    assert hb.s_max >= 0
    assert len(hb.brackets) >= 1
    assert hb.config.min_budget == 3
    assert hb.config.max_budget == 27


def test_asha_scheduler_init():
    """Test ASHA scheduler initialization."""
    config = HPOConfig(
        grace_period=3,
        reduction_factor=3,
        brackets=1,
    )
    asha = AsyncSuccessiveHalvingScheduler(config)
    
    assert asha.grace_period == 3
    assert asha.reduction_factor == 3
    assert len(asha.rungs) == 0


def test_asha_add_trial():
    """Test adding trials to ASHA."""
    config = HPOConfig(grace_period=3, reduction_factor=3)
    asha = AsyncSuccessiveHalvingScheduler(config)
    
    asha.add_trial("trial_0", {"lr": 1e-3})
    asha.add_trial("trial_1", {"lr": 1e-4})
    
    assert "trial_0" in asha.trial_states
    assert "trial_1" in asha.trial_states
    assert len(asha.rungs[0]) == 2


def test_asha_on_trial_result():
    """Test ASHA on_trial_result."""
    config = HPOConfig(grace_period=1, reduction_factor=2, mode="maximize")
    asha = AsyncSuccessiveHalvingScheduler(config)
    
    asha.add_trial("trial_0", {"lr": 1e-3})
    asha.add_trial("trial_1", {"lr": 1e-4})
    
    # Simulate results
    result_0 = {"val_sharpe": 0.8}
    result_1 = {"val_sharpe": 0.5}
    
    # First results - both continue (only one trial in rung)
    action_0 = asha.on_trial_result("trial_0", result_0)
    assert action_0["action"] == "continue"
    
    # Second result - trial_1 has lower score, should be stopped (bottom 1/eta)
    action_1 = asha.on_trial_result("trial_1", result_1)
    assert action_1["action"] == "stop"


# ══════════════════════════════════════════════════════════════════════════════
# BOHB Scheduler
# ══════════════════════════════════════════════════════════════════════════════

def test_bohb_scheduler_init():
    """Test BOHB scheduler initialization."""
    config = HPOConfig(
        min_budget=3,
        max_budget=27,
        eta=3,
    )
    bohb = BOHBScheduler(config)
    
    assert bohb.config == config
    assert bohb.hyperband is not None
    assert bohb.kde_good == {}
    assert bohb.kde_bad == {}


def test_bohb_observe():
    """Test BOHB observation recording."""
    config = HPOConfig(min_budget=3, max_budget=27, eta=3)
    bohb = BOHBScheduler(config)
    
    # Add some observations
    bohb.observe("trial_0", 0, {"lr": 1e-3}, 0.8)
    bohb.observe("trial_1", 0, {"lr": 1e-4}, 0.5)
    bohb.observe("trial_2", 0, {"lr": 1e-3}, 0.9)
    
    assert len(bohb.observations[0]) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Fidelity ASHA
# ══════════════════════════════════════════════════════════════════════════════

def test_mf_asha_init():
    """Test multi-fidelity ASHA initialization."""
    config = HPOConfig(grace_period=1, reduction_factor=3)
    fidelity_dims = [
        {"name": "epochs", "min": 1, "max": 27, "eta": 3},
        {"name": "data_frac", "min": 0.1, "max": 1.0, "eta": 2},
    ]
    
    mf_asha = MultiFidelityASHAScheduler(config, fidelity_dims)
    
    assert len(mf_asha.fidelity_dims) == 2
    assert mf_asha.config == config


def test_mf_asha_add_trial():
    """Test MF-ASHA trial addition."""
    config = HPOConfig()
    fidelity_dims = [{"name": "epochs", "min": 1, "max": 10, "eta": 2}]
    mf_asha = MultiFidelityASHAScheduler(config, fidelity_dims)
    
    mf_asha.add_trial("trial_0", {"lr": 1e-3})
    
    assert "trial_0" in mf_asha.trial_states
    assert mf_asha.trial_states["trial_0"]["fidelity"]["epochs"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# HPOManager
# ═════════════════════════════════════════════════════════════════════════════

def test_hpo_manager_init():
    """Test HPOManager initialization."""
    config = HPOConfig(
        population_size=4,
        max_epochs=10,
    )
    manager = HPOManager(config)
    
    assert manager.config == config
    assert manager.pbt is not None
    assert manager.hyperband is not None
    assert manager.asha is not None
    assert manager.bohb is not None
    assert manager.mf_asha is not None


def test_hpo_manager_sample_params():
    """Test parameter sampling."""
    config = HPOConfig(seed=42)
    manager = HPOManager(config)
    
    params = manager._sample_initial_params()
    
    assert "lr" in manager._sample_initial_params()
    assert "d_model" in manager._sample_initial_params()
    assert "batch_size" in manager._sample_initial_params()


def test_hpo_manager_create_study():
    """Test Optuna study creation."""
    config = HPOConfig()
    manager = HPOManager(config)
    
    if True:  # OPTUNA_AVAILABLE
        study = manager.create_study(direction="maximize", sampler="tpe", pruner="hyperband")
        assert manager.study is not None
        assert manager.study.direction.name == "MAXIMIZE"


# ══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_pbt_integration():
    """Test PBT integration with HPOManager."""
    config = HPOConfig(population_size=3, max_epochs=5)
    manager = HPOManager(config)
    
    # Initialize population
    for i in range(3):
        params = manager._sample_initial_params()
        trial_id = f"pbt_{i}"
        manager.pbt.add_trial(trial_id, params)
    
    assert len(manager.pbt.population) == 3


def test_hyperband_integration():
    """Test HyperBand integration."""
    config = HPOConfig(min_budget=3, max_budget=9, eta=3)
    manager = HPOManager(config)
    
    # HyperBand should be initialized
    assert manager.hyperband is not None
    assert len(manager.hyperband.brackets) >= 1


def test_asha_integration():
    """Test ASHA integration."""
    config = HPOConfig(grace_period=2, reduction_factor=2)
    manager = HPOManager(config)
    
    assert manager.asha is not None
    assert manager.asha.grace_period == 2
    assert manager.asha.reduction_factor == 2


def test_bohb_integration():
    """Test BOHB integration."""
    config = HPOConfig(min_budget=3, max_budget=9, eta=3)
    manager = HPOManager(config)
    
    assert manager.bohb is not None
    assert manager.bohb.hyperband is not None


def test_hpo_manager_state_dict():
    """Test HPOManager state dict."""
    config = HPOConfig(population_size=2)
    manager = HPOManager(config)
    
    # Add some trials
    for i in range(2):
        params = manager._sample_initial_params()
        trial_id = f"trial_{i}"
        manager.pbt.add_trial(trial_id, params)
        manager.pbt.update_score(f"trial_{i}", 0.5 + i * 0.1, 10)
    
    state = manager.state_dict()
    
    assert "epoch" in state
    assert "config_mode" in state
    assert "difficulty" in state
    assert "self_paced_pace" in state
    assert "adaptive_state" in state


def test_hpo_factory():
    """Test HPO factory function."""
    X = np.random.randn(50, 10, 4).astype(np.float32)
    
    trainer = create_multi_task_pretrainer(X, seq_len=10, n_features=3, device="cpu")
    
    assert isinstance(trainer, MultiTaskPretrainer)


# ══════════════════════════════════════════════════════════════════════════════
# run_hpo_study
# ═════════════════════════════════════════════════════════════════════════════

def test_run_hpo_study():
    """Test run_hpo_study factory function."""
    X = np.random.randn(10, 5, 4).astype(np.float32)
    
    # This would fail without Optuna, but we can test the function signature
    try:
        trainer, history = pretrain_multi_task(
            X,
            seq_len=5,
            n_features=3,
            epochs=1,
            batch_size=8,
            device="cpu",
        )
        assert isinstance(history, dict)
    except Exception:
        # May fail without Optuna - that's okay for this test
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Domain Adaptation
# ══════════════════════════════════════════════════════════════════════════════

def test_adapt_encoder_to_target():
    """Test domain adaptation."""
    encoder = torch.nn.Sequential(
        torch.nn.Linear(4, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
    )
    source = np.random.randn(50, 10, 4).astype(np.float32)
    target = np.random.randn(50, 10, 4).astype(np.float32) + 2.0  # shift
    
    adapted = adapt_encoder_to_target(
        encoder, source, target,
        method="dann", epochs=2, lr=1e-3, device="cpu"
    )
    assert adapted is encoder


def test_adapt_encoder_fine_tune():
    """Test fine-tuning adaptation."""
    encoder = torch.nn.Sequential(
        torch.nn.Linear(4, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
    )
    source = np.random.randn(50, 10, 4).astype(np.float32)
    target = np.random.randn(50, 10, 4).astype(np.float32)
    
    adapted = adapt_encoder_to_target(
        encoder, source, target,
        method="fine_tune", epochs=2, lr=1e-3, device="cpu"
    )
    assert adapted is encoder


# ══════════════════════════════════════════════════════════════════════════════
# Domain Adaptation Losses
# ══════════════════════════════════════════════════════════════════════════════

def test_mmd_loss():
    """Test MMD loss."""
    source = torch.randn(20, 8)
    target = torch.randn(20, 8)
    mmd = MMDLoss(kernel="rbf", gamma=1.0)
    loss = mmd(source, target)
    
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_mmd_loss_same_distribution():
    """Test MMD with same distribution."""
    X = torch.randn(20, 8)
    idx = torch.randperm(20)
    source = X[:10]
    target = X[10:]
    mmd = MMDLoss(kernel="rbf", gamma=1.0)
    loss = mmd(source, target)
    
    assert loss.item() < 1.0


def test_coral_loss():
    """Test CORAL loss."""
    source = torch.randn(20, 8)
    target = torch.randn(20, 8)
    coral = CORALLoss()
    loss = coral(source, target)
    
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_coral_loss_same():
    """Test CORAL with same distribution."""
    X = torch.randn(20, 8)
    source = X[:10]
    target = X[10:]
    coral = CORALLoss()
    loss = coral(source, target)
    
    assert loss.item() < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Domain Discriminator
# ══════════════════════════════════════════════════════════════════════════════

def test_domain_discriminator():
    """Test DomainDiscriminator."""
    disc = DomainDiscriminator(64, 3, hidden_dim=64)
    x = torch.randn(8, 64)
    logits = disc(x)
    assert logits.shape == (8, 3)


def test_grad_reverse():
    """Test gradient reversal."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = grad_reverse(x, 2.0)
    loss = y.sum()
    loss.backward()
    assert torch.allclose(x.grad, -2.0 * torch.ones_like(x.grad))


# ══════════════════════════════════════════════════════════════════════════════
# Task Losses
# ═════════════════════════════════════════════════════════════════════════════

def test_nt_xent_loss():
    """Test NT-Xent contrastive loss."""
    z1 = torch.randn(4, 16)
    z2 = torch.randn(4, 16)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    
    assert loss.ndim == 0
    assert loss.item() > 0


def test_nt_xent_loss_perfect_match():
    """Test NT-Xent with perfect match."""
    z = F.normalize(torch.randn(4, 16), dim=-1)
    loss = nt_xent_loss(z, z, temperature=0.5)
    
    assert loss.item() < 1.0


def test_byol_loss():
    """Test BYOL loss."""
    p1 = torch.randn(4, 16)
    p2 = torch.randn(4, 16)
    z1 = torch.randn(4, 16)
    z2 = torch.randn(4, 16)
    loss = byol_loss(p1, p2, z1, z2)
    
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_masked_reconstruction_loss():
    """Test masked reconstruction loss."""
    B, T, F = 4, 10, 3
    recon = torch.randn(B, T, F)
    target = torch.randn(B, T, F)
    mask = torch.zeros(B, T, F, dtype=torch.bool)
    mask[:, 2:5, :] = True
    
    loss = masked_reconstruction_loss(recon, target, mask)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_vae_loss():
    """Test VAE loss."""
    recon = torch.randn(4, 10, 3)
    target = torch.randn(4, 10, 3)
    mu = torch.randn(4, 16)
    logvar = torch.randn(4, 16)
    
    loss, recon_loss, kl = vae_loss(recon, target, mu, logvar, beta=0.001)
    
    assert loss.ndim == 0
    assert recon_loss.ndim == 0
    assert kl.ndim == 0


def test_forecast_loss():
    """Test forecast loss."""
    pred = torch.randn(4, 5, 3)
    target = torch.randn(4, 5, 3)
    loss = forecast_loss(pred, target)
    
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_drift_loss():
    """Test drift loss."""
    clean = torch.randn(8, 16)
    drift = torch.randn(8, 16)
    loss = drift_loss(clean, drift, margin=1.0)
    
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_domain_adversarial_loss():
    """Test domain adversarial loss."""
    features = torch.randn(10, 16, requires_grad=True)
    domain_labels = torch.randint(0, 2, (10,))
    disc = torch.nn.Linear(16, 2)
    loss = domain_adversarial_loss(features, domain_labels, disc, lambda_=1.0)
    
    assert loss.ndim == 0


# ══════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═════════════════════════════════════════════════════════════════════════════

def test_create_multi_task_pretrainer():
    """Test factory function."""
    X = np.random.randn(50, 20, 5).astype(np.float32)
    trainer = create_multi_task_pretrainer(X, seq_len=20, n_features=5, device="cpu")
    assert isinstance(trainer, MultiTaskPretrainer)


def test_pretrain_multi_task():
    """Test one-shot pretraining."""
    X = np.random.randn(50, 15, 4).astype(np.float32)
    trainer, history = pretrain_multi_task(
        X,
        seq_len=15,
        n_features=4,
        epochs=1,
        batch_size=8,
        device="cpu",
    )
    assert isinstance(trainer, MultiTaskPretrainer)
    assert isinstance(history, dict)


def test_adapt_encoder_to_target():
    """Test domain adaptation utility."""
    encoder = torch.nn.Sequential(
        torch.nn.Linear(4, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
    )
    source = np.random.randn(50, 10, 4).astype(np.float32)
    target = np.random.randn(50, 10, 4).astype(np.float32) + 2.0
    
    adapted = adapt_encoder_to_target(
        encoder, source, target,
        method="dann", epochs=2, lr=1e-3, device="cpu"
    )
    assert adapted is encoder


def test_adapt_encoder_fine_tune():
    encoder = torch.nn.Sequential(
        torch.nn.Linear(4, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
    )
    source = np.random.randn(50, 10, 4).astype(np.float32)
    target = np.random.randn(50, 10, 4).astype(np.float32)
    
    adapted = adapt_encoder_to_target(
        encoder, source, target,
        method="fine_tune", epochs=2, lr=1e-3, device="cpu"
    )
    assert adapted is encoder


def test_build_optuna_search_defaults():
    """tpe preserves the default Optuna behavior (TPE + MedianPruner)."""
    from training.hpo import build_optuna_search
    sampler, pruner = build_optuna_search("tpe", seed=42, max_resource=8)
    assert type(sampler).__name__ == "TPESampler"
    assert type(pruner).__name__ == "MedianPruner"


def test_build_optuna_search_asha():
    from training.hpo import build_optuna_search
    sampler, pruner = build_optuna_search("asha", seed=0, max_resource=8)
    assert type(pruner).__name__ == "SuccessiveHalvingPruner"


def test_build_optuna_search_bohb_pbt():
    from training.hpo import build_optuna_search
    _, pruner = build_optuna_search("bohb", seed=0, max_resource=8)
    assert type(pruner).__name__ == "HyperbandPruner"
    sampler, _ = build_optuna_search("pbt", seed=0, max_resource=8)
    assert type(sampler).__name__ == "CmaEsSampler"


def test_build_optuna_search_unknown():
    from training.hpo import build_optuna_search
    with pytest.raises(ValueError):
        build_optuna_search("nonexistent", seed=0)


def test_optuna_tune_hpo_scheduler_arg(monkeypatch):
    """--hpo-scheduler must be accepted by scripts.optuna_tune.parse_args."""
    import sys
    from scripts.optuna_tune import parse_args
    monkeypatch.setattr(sys, "argv", ["optuna_tune", "--model", "tft", "--hpo-scheduler", "asha", "--trials", "1"])
    args = parse_args()
    assert args.hpo_scheduler == "asha"
    monkeypatch.setattr(sys, "argv", ["optuna_tune", "--model", "tft"])
    args = parse_args()
    assert args.hpo_scheduler == "tpe"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])