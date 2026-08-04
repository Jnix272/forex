"""Tests for AMP dtype selection (Ampere+ → BF16)."""
from __future__ import annotations

import torch

from training.gpu_device import resolve_amp_dtype


def test_resolve_amp_dtype_explicit_fp32_fp16():
    assert resolve_amp_dtype("fp32") is torch.float32
    assert resolve_amp_dtype("fp16") is torch.float16


def test_resolve_amp_dtype_auto_forces_bf16_on_ampere(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (8, 9))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_amp_dtype("auto") is torch.bfloat16
    assert resolve_amp_dtype("bf16") is torch.bfloat16


def test_resolve_amp_dtype_auto_fp16_pre_ampere(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_amp_dtype("auto") is torch.float16


def test_resolve_amp_dtype_bf16_falls_back_pre_ampere(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_amp_dtype("bf16") is torch.float16
    assert "falling back to FP16" in capsys.readouterr().out


def test_settings_amp_dtype_auto_documents_ampere_bf16():
    from config.settings import GPU

    assert GPU.get("amp_dtype") == "auto"
