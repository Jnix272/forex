"""Regression tests for docs/AUDIT_2026-09-25_pretraining.md fixes."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_discard_renames_encoder_and_epoch_files(tmp_path):
    from training.pretrain_runner import _discard_encoder_files

    for name in ("contrastive_encoder.pt", "contrastive_encoder_ep1.pt", "contrastive_encoder_ep4.pt"):
        (tmp_path / name).write_bytes(b"x")
    moved = _discard_encoder_files(str(tmp_path / "contrastive_encoder.pt"))
    assert len(moved) == 3
    assert not list(tmp_path.glob("*.pt"))


def _loader_setup(tmp_path, report: dict):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    torch.save(model.state_dict(), tmp_path / "contrastive_encoder.pt")
    if report is not None:
        (tmp_path / "pretrain_report.json").write_text(json.dumps(report))
    args = types.SimpleNamespace(checkpoint_dir=str(tmp_path), model="m", pretrain_method="masked",
                                 pretrain_regime=False, pretrain_framework="custom")
    return model, args


@pytest.mark.parametrize(
    "report",
    [
        None,
        {"status": "discarded", "quality_gate_result": "failed_low_uniformity"},
        {"status": "completed", "quality_gate_result": "passed", "ablation_verdict": "pretrain_hurt", "scaled_inputs": True},
        {"status": "completed", "quality_gate_result": "passed"},  # pre-fix: unscaled inputs
    ],
)
def test_loader_refuses_unaccepted_encoders(tmp_path, report):
    from training.supervised_loop import _load_pretrained_encoder

    model, args = _loader_setup(tmp_path, report)
    assert _load_pretrained_encoder(model, args, torch.device("cpu")) is False


def test_loader_checks_hash(tmp_path):
    import hashlib

    from training.supervised_loop import _load_pretrained_encoder

    model, args = _loader_setup(tmp_path, {})
    good = hashlib.sha256((tmp_path / "contrastive_encoder.pt").read_bytes()).hexdigest()
    rep = {"status": "completed", "quality_gate_result": "passed", "scaled_inputs": True, "checkpoint_sha256": good}
    (tmp_path / "pretrain_report.json").write_text(json.dumps(rep))
    assert _load_pretrained_encoder(model, args, torch.device("cpu")) is not False
    rep["checkpoint_sha256"] = "0" * 64
    (tmp_path / "pretrain_report.json").write_text(json.dumps(rep))
    assert _load_pretrained_encoder(model, args, torch.device("cpu")) is False


def test_loss_floor_does_not_amplify_near_constant_channels():
    from pretrain.loss_scaling import normalized_mse_loss

    target = torch.zeros(64, 10, 2)
    target[..., 0] = torch.randn(64, 10)          # active channel, std ~1
    target[..., 1] = 1.0 + 1e-4 * torch.randn(64, 10)  # near-constant channel
    pred = target.clone()
    pred[..., 1] += 0.01  # tiny absolute error on the flat channel
    assert normalized_mse_loss(pred, target).item() < 1.0  # was ~0.5e4 with a 1e-5 floor


def test_pretrain_disabled_in_run_yaml():
    import yaml

    cfg = yaml.safe_load(Path("config/run.yaml").read_text(encoding="utf-8"))
    assert cfg["pretrain"]["enabled"] is False
