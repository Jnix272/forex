import torch
import torch.nn as nn

from training.synaptic_intelligence import (
    DynamicSILambdaConfig,
    SynapticIntelligence,
    compute_dynamic_si_lambda,
)


def test_synaptic_intelligence_tracking():
    # Simple model
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    # Init weights to ones
    model[0].weight.data.fill_(1.0)

    si = SynapticIntelligence(model, epsilon=1e-3)

    # Simulate step 1: backward() has filled grads; pre_step snapshots the RAW
    # gradient, so it must be set before pre_step.
    model[0].weight.grad = torch.tensor([[-0.1, -0.1], [-0.1, -0.1]])
    si.pre_step()
    # Simulate optimizer moving weight by 0.5
    model[0].weight.data.fill_(1.5)
    si.post_step()

    # Path integral = -grad * delta = -(-0.1) * 0.5 = +0.05
    path_int = si.path_integral_0_weight
    assert torch.allclose(path_int, torch.tensor(0.05)), f"Path integral was {path_int}"

    # Simulate step 2
    model[0].weight.grad = torch.tensor([[0.2, 0.2], [0.2, 0.2]])
    si.pre_step()
    model[0].weight.data.fill_(1.0)  # moving back
    si.post_step()

    # Path integral addition = -0.2 * -0.5 = +0.10. Total = 0.15
    path_int = si.path_integral_0_weight
    assert torch.allclose(path_int, torch.tensor(0.15)), f"Path integral was {path_int}"

    # Update omega
    si.update_omega()
    omega = si.omega_0_weight

    # Delta total from initial (1.0) to current (1.0) is 0.
    # omega should be abs(path_int) / (0 + 1e-3) = 0.15 / 1e-3 = 150
    assert torch.allclose(omega, torch.tensor(150.0)), f"Omega was {omega}"

    # Simulate some drift to get a penalty
    model[0].weight.data.fill_(2.0)
    # Penalty = sum(omega * (2.0 - 1.0)^2) = 150 * 1 * 4 elements = 600
    penalty = si.penalty()
    assert torch.allclose(penalty, torch.tensor(600.0)), f"Penalty was {penalty}"


def test_si_uses_raw_gradients_not_clipped_or_centralized():
    model = nn.Sequential(nn.Linear(1, 1, bias=False))
    si = SynapticIntelligence(model)
    w = model[0].weight

    w.data.fill_(0.5)
    w.grad = torch.full_like(w, 0.2)  # raw backward() gradient
    si.pre_step()
    # Simulate in-place mutation by the training loop (grad clip / GC) AFTER
    # pre_step: the path integral must still use the raw gradient.
    w.grad.copy_(torch.full_like(w, 5.0))
    w.data.fill_(0.6)  # delta_theta = +0.1
    si.post_step()

    # -raw_grad * delta = -0.2 * 0.1 = -0.02 (NOT -5.0 * 0.1 = -0.5)
    path_int = si.path_integral_0_weight
    assert torch.allclose(path_int, torch.tensor(-0.02)), f"Path integral was {path_int}"


def test_compute_dynamic_si_lambda_schedule():
    cfg = DynamicSILambdaConfig(enabled=True, lambda_min=0.0, lambda_max=1.0, ema_alpha=0.99)
    # Low task loss (epoch mastered) -> lambda near the ceiling.
    lam_low, kappa_low = compute_dynamic_si_lambda(torch.tensor(0.01), kappa=1.0, cfg=cfg)
    assert lam_low > 0.9, f"lambda on low loss was {lam_low}"
    # High task loss (regime shock) -> lambda near the floor.
    lam_high, kappa_high = compute_dynamic_si_lambda(torch.tensor(100.0), kappa=1.0, cfg=cfg)
    assert lam_high < 0.1, f"lambda on high loss was {lam_high}"
    assert lam_low > lam_high
    # kappa EMA tracks the absolute task loss and rises with the shock.
    assert kappa_high > kappa_low
    assert 0.0 <= lam_low <= 1.0 and 0.0 <= lam_high <= 1.0

    # First-batch kappa initializes to the task loss magnitude.
    _, kappa_first = compute_dynamic_si_lambda(torch.tensor(-2.5), kappa=None, cfg=cfg)
    assert abs(kappa_first - 2.5) < 1e-6, f"kappa init was {kappa_first}"
