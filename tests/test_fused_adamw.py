"""Fused AdamW builder — CUDA fused kernel with eager / apex fallbacks."""
from __future__ import annotations

import torch
import torch.nn as nn

from training.gpu_device import build_adamw


def test_build_adamw_eager_on_cpu():
    m = nn.Linear(4, 2)
    opt = build_adamw(m.parameters(), lr=1e-3, weight_decay=1e-4, fused=False)
    assert isinstance(opt, torch.optim.AdamW)
    assert not bool(opt.defaults.get("fused"))


def test_build_adamw_fused_when_cuda_available():
    if not torch.cuda.is_available():
        return
    m = nn.Linear(8, 4).cuda()
    opt = build_adamw(m.parameters(), lr=1e-3, weight_decay=1e-4)
    assert isinstance(opt, torch.optim.Optimizer)
    # Native fused AdamW sets defaults['fused']=True; apex uses a different class.
    is_native_fused = isinstance(opt, torch.optim.AdamW) and bool(opt.defaults.get("fused"))
    is_apex = type(opt).__name__ in ("FusedAdam", "FusedAdamW")
    assert is_native_fused or is_apex

    # One step must succeed under fused kernels
    x = torch.randn(2, 8, device="cuda")
    opt.zero_grad(set_to_none=True)
    m(x).sum().backward()
    opt.step()
