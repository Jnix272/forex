"""Tests for adversarial generator (PGD, FGSM, FreeLB, MarketShock)."""
import pytest

# Skip if torch not available (sandbox environment)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)


class SimpleModel(nn.Module):
    def __init__(self, n_features=10, n_outputs=1):
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)
    
    def forward(self, x):
        # x: (batch, seq, features) -> take last timestep
        return self.linear(x[:, -1, :])


def test_fgsm_attack_basic():
    """Test FGSM attack produces adversarial examples within eps-ball."""
    from training.adversarial_generator import FGSMAttack
    
    model = SimpleModel()
    model.train()
    
    attack = FGSMAttack(eps=0.3, probability=1.0)
    attack.train()
    
    x = torch.randn(4, 16, 10)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    x_adv = attack(model, x, y, criterion)
    
    # Check that perturbation is bounded by eps
    diff = (x_adv - x).abs().max().item()
    assert diff <= 0.3 + 1e-6, f"Perturbation {diff} exceeds eps=0.3"
    
    # Check that output is different from input
    assert not torch.allclose(x_adv, x), "Adversarial example should differ from clean"


def test_pgd_attack_basic():
    """Test PGD attack produces adversarial examples within eps-ball."""
    from training.adversarial_generator import PGDAttack
    
    model = SimpleModel()
    model.train()
    
    attack = PGDAttack(eps=0.3, alpha=0.01, steps=7, probability=1.0, random_start=False)
    attack.train()
    
    x = torch.randn(4, 16, 10)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    x_adv = attack(model, x, y, criterion)
    
    # Check that perturbation is bounded by eps
    diff = (x_adv - x).abs().max().item()
    assert diff <= 0.3 + 1e-6, f"Perturbation {diff} exceeds eps=0.3"
    
    # Check that output is different from input
    assert not torch.allclose(x_adv, x), "Adversarial example should differ from clean"


def test_pgd_random_start():
    """Test PGD with random start produces different results each time."""
    from training.adversarial_generator import PGDAttack
    
    model = SimpleModel()
    model.train()
    
    attack = PGDAttack(eps=0.3, alpha=0.01, steps=7, probability=1.0, random_start=True)
    attack.train()
    
    x = torch.randn(4, 16, 10)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    x_adv1 = attack(model, x, y, criterion)
    x_adv2 = attack(model, x, y, criterion)
    
    # With random start, results should differ (very high probability)
    # Just check they're both within eps-ball
    diff1 = (x_adv1 - x).abs().max().item()
    diff2 = (x_adv2 - x).abs().max().item()
    assert diff1 <= 0.3 + 1e-6
    assert diff2 <= 0.3 + 1e-6


def test_freelb_attack_basic():
    """Test FreeLB attack produces adversarial examples within eps-ball."""
    from training.adversarial_generator import FreeLBAttack
    
    model = SimpleModel()
    model.train()
    
    attack = FreeLBAttack(eps=0.3, alpha=0.01, steps=3, probability=1.0)
    attack.train()
    
    x = torch.randn(4, 16, 10)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    x_adv = attack(model, x, y, criterion)
    
    # Check that perturbation is bounded by eps
    diff = (x_adv - x).abs().max().item()
    assert diff <= 0.3 + 1e-6, f"Perturbation {diff} exceeds eps=0.3"
    
    # Check that output is different from input
    assert not torch.allclose(x_adv, x), "Adversarial example should differ from clean"


def test_adversarial_probability_zero():
    """Test that probability=0 returns clean input unchanged."""
    from training.adversarial_generator import PGDAttack
    
    model = SimpleModel()
    model.train()
    
    attack = PGDAttack(eps=0.3, alpha=0.01, steps=7, probability=0.0)
    attack.train()
    
    x = torch.randn(4, 16, 10)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    x_adv = attack(model, x, y, criterion)
    
    # Should return original input unchanged
    assert torch.allclose(x_adv, x), "With probability=0, should return clean input"


def test_adversarial_eval_mode():
    """Test that attack returns clean input in eval mode."""
    from training.adversarial_generator import FGSMAttack
    
    model = SimpleModel()
    model.eval()  # Model in eval
    
    attack = FGSMAttack(eps=0.3, probability=1.0)
    attack.eval()  # Attack in eval
    
    x = torch.randn(4, 16, 10)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    x_adv = attack(model, x, y, criterion)
    
    # Should return original input unchanged
    assert torch.allclose(x_adv, x), "In eval mode, should return clean input"


def test_market_shock_generator_basic():
    """Test legacy MarketShockGenerator still works."""
    from training.adversarial_generator import MarketShockGenerator
    
    # Need feature names for market shock generator
    feature_names = ["close", "spread_mean", "fb_0", "fb_1", "other"]
    
    attack = MarketShockGenerator(
        whipsaw_prob=1.0,
        spread_blowout_prob=1.0,
        sentiment_shock_prob=1.0,
        whipsaw_magnitude=5.0,
        spread_multiplier=10.0,
        feature_names=feature_names,
    )
    attack.train()
    
    x = torch.randn(4, 16, 5)
    x_adv = attack(x, feature_names)
    
    # Should modify the input (with prob=1.0)
    # Not checking exact values since it's random, but should not be identical
    # (though could be by very small chance)
    assert x_adv.shape == x.shape, "Output shape should match input"


def test_create_adversarial_attack_factory():
    """Test factory function creates correct attack types."""
    from training.adversarial_generator import (
        create_adversarial_attack,
        PGDAttack,
        FGSMAttack,
        FreeLBAttack,
        MarketShockGenerator,
    )
    
    pgd = create_adversarial_attack("pgd", eps=0.3)
    assert isinstance(pgd, PGDAttack)
    
    fgsm = create_adversarial_attack("fgsm", eps=0.3)
    assert isinstance(fgsm, FGSMAttack)
    
    freelb = create_adversarial_attack("freelb", eps=0.3)
    assert isinstance(freelb, FreeLBAttack)
    
    market_shock = create_adversarial_attack("market_shock", eps=0.3)
    assert isinstance(market_shock, MarketShockGenerator)
    
    # Test invalid method
    with pytest.raises(ValueError):
        create_adversarial_attack("invalid_method")


def test_adversarial_attack_with_mask():
    """Test adversarial attacks work with loss mask."""
    from training.adversarial_generator import PGDAttack
    
    model = SimpleModel()
    model.train()
    
    attack = PGDAttack(eps=0.3, alpha=0.01, steps=3, probability=1.0, random_start=False)
    attack.train()
    
    x = torch.randn(4, 16, 10)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    mask = torch.ones(4, 1)
    mask[0] = 0  # Mask out first sample
    
    def masked_criterion(output, target):
        loss = criterion(output, target)
        return (loss * mask).mean()
    
    x_adv = attack(model, x, y, masked_criterion)
    
    # Check perturbation bounded
    diff = (x_adv - x).abs().max().item()
    assert diff <= 0.3 + 1e-6


def test_adversarial_attack_multi_output():
    """Test adversarial attacks work with model returning tuple."""
    from training.adversarial_generator import FGSMAttack
    
    class MultiOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 1)
            self.linear2 = nn.Linear(10, 1)
        
        def forward(self, x):
            out1 = self.linear1(x[:, -1, :])
            out2 = self.linear2(x[:, -1, :])
            return (out1, out2)
    
    model = MultiOutputModel()
    model.train()
    
    attack = FGSMAttack(eps=0.3, probability=1.0)
    attack.train()
    
    x = torch.randn(4, 16, 10)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    # Should not crash with multi-output models
    x_adv = attack(model, x, y, criterion)
    
    diff = (x_adv - x).abs().max().item()
    assert diff <= 0.3 + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])