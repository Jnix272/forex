"""Tests for torch.compile RNN eager-skip helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.gpu_device import disable_compile_on_rnn_modules, maybe_torch_compile


def test_disable_compile_on_rnn_modules_marks_lstm_only():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.lstm = nn.LSTM(4, 4, batch_first=True)
            self.gru = nn.GRU(4, 4, batch_first=True)

        def forward(self, x):
            x = self.lin(x)
            y, _ = self.lstm(x)
            z, _ = self.gru(y)
            return z

    m = Tiny()
    n = disable_compile_on_rnn_modules(m)
    assert n == 2
    assert getattr(m.lstm.forward, "_forex_compiler_disabled", False) is True
    assert getattr(m.gru.forward, "_forex_compiler_disabled", False) is True
    # Linear forward is untouched (no marker)
    assert getattr(m.lin.forward, "_forex_compiler_disabled", False) is False
    # Idempotent
    assert disable_compile_on_rnn_modules(m) == 0


def test_maybe_torch_compile_respects_disabled_flag_on_cpu():
    m = nn.Linear(2, 2)
    out = maybe_torch_compile(m, torch.device("cpu"), {"torch_compile": True})
    assert out is m


def test_maybe_torch_compile_honours_explicit_false(monkeypatch):
    m = nn.Linear(2, 2)

    # Force CUDA path checks without needing a GPU: patch device type via fake
    class _Dev:
        type = "cuda"

    monkeypatch.setattr(
        "training.gpu_device._ensure_bound",
        lambda: None,
    )
    monkeypatch.setattr("training.gpu_device._GPU_CFG", {"torch_compile": False}, raising=False)
    monkeypatch.setattr("training.gpu_device._log_info", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("training.gpu_device._log_warn", lambda *a, **k: None, raising=False)

    out = maybe_torch_compile(m, _Dev(), {"torch_compile": False})
    assert out is m


def test_settings_torch_compile_default_true():
    from config.settings import GPU

    assert GPU.get("torch_compile", False) is True
    assert GPU.get("torch_compile_mode") == "reduce-overhead"
