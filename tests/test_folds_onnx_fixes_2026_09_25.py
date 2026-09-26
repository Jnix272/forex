"""Regression tests for docs/AUDIT_2026-09-25_training_folds_onnx.md fixes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_refit_epochs_is_median_selected_epoch():
    from training.post_train import _cv_refit_epochs

    hist = [
        {"history": {"val_loss": [3, 2, 1, 2, 3, 4]}},           # best epoch 3
        {"history": {"val_loss": [3, 1, 2, 3, 4, 5]}},           # best epoch 2
        {"history": {"honest_sharpe_ci_low": [0, 0.1, 0.2, 0.5, 0.1, 0]}},  # best epoch 4
    ]
    assert _cv_refit_epochs(hist, 40) == 4  # median(3, 2, 4) = 3 -> floor of 4
    hist.append({"history": {"val_loss": [9] * 9 + [1]}})       # best epoch 10
    assert _cv_refit_epochs(hist, 40) == 4
    assert _cv_refit_epochs([], 12) == 12


def test_onnx_parity_and_sidecar(tmp_path):
    pytest.importorskip("onnxruntime")
    from inference.onnx_inference import (
        _verify_onnx_parity,
        _write_onnx_sidecar,
        onnx_matches_checkpoint,
    )

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4 * 3, 2)).eval()
    dummy = torch.randn(1, 4, 3)
    path = tmp_path / "m.onnx"
    torch.onnx.export(model, dummy, str(path), input_names=["features"], output_names=["logits"],
                      dynamic_axes={"features": {0: "b"}, "logits": {0: "b"}}, dynamo=False)
    assert _verify_onnx_parity(model, path, dummy) < 1e-4

    ckpt = tmp_path / "m_best.pt"
    torch.save(model.state_dict(), ckpt)
    _write_onnx_sidecar(path, ckpt)
    assert onnx_matches_checkpoint(path, ckpt)[0]
    torch.save({"changed": torch.ones(1)}, ckpt)  # retrained checkpoint
    ok, why = onnx_matches_checkpoint(path, ckpt)
    assert not ok and "stale" in why


def test_onnx_parity_rejects_mismatch(tmp_path):
    pytest.importorskip("onnxruntime")
    from inference.onnx_inference import _verify_onnx_parity

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2)).eval()
    other = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2)).eval()
    dummy = torch.randn(1, 4, 3)
    path = tmp_path / "m.onnx"
    torch.onnx.export(other, dummy, str(path), input_names=["features"], output_names=["logits"], dynamo=False)
    with pytest.raises(RuntimeError):
        _verify_onnx_parity(model, path, dummy)
    assert not path.exists()  # a failing export is deleted
