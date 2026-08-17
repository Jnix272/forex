"""
Tests for audit package (Improvement #6): data lineage, model registry records,
decision trail, and reproducibility manifests.
"""

from __future__ import annotations

import json
import os

import pytest

from audit.lineage import (
    DataLineage,
    DecisionRecord,
    LineageStep,
    ModelRegistryRecord,
    decision_trail,
)
from audit.manifest import (
    capture_env,
    compute_dir_hash,
    compute_file_hash,
    generate_manifest,
    git_info,
    hash_inputs,
    regenerate_manifest,
    verify_manifest,
    verify_manifests_in_tree,
    write_manifest,
)


@pytest.fixture
def ckpt_dir(tmp_path):
    d = tmp_path / "run_ckpt"
    d.mkdir()
    (d / "production_best.pt").write_bytes(b"model-weights-v1")
    (d / "train_summary.json").write_text('{"calibration": {"ece": 0.02}}')
    return d


# ═════════════════════════════════════════════════════════════════════════════
# Data lineage
# ═════════════════════════════════════════════════════════════════════════════


def test_lineage_chain():
    lineage = DataLineage(dataset="EURUSD_1m", dataset_version="2026.07")
    lineage.add_step("preprocess", "clean_v2", params={"outliers": "mad"})
    lineage.add_step("feature_set", "features_v3", data_hash="abc")
    run = lineage.record_training_run(run_id="r1", params={"lr": 1e-4}, seed=42)
    assert lineage.dataset == "EURUSD_1m"
    assert len(lineage.steps) == 2
    assert run["run_id"] == "r1"
    assert run["seed"] == 42


def test_lineage_roundtrip(tmp_path):
    lineage = DataLineage(dataset="X", dataset_version="1")
    lineage.add_step("label", "v4")
    lineage.record_training_run(run_id="r2", model="haelt")
    path = tmp_path / "lineage.json"
    lineage.save(str(path))
    loaded = DataLineage.load(str(path))
    assert loaded.dataset == "X"
    assert len(loaded.steps) == 1
    assert loaded.training_runs[0]["run_id"] == "r2"


def test_lineage_step_to_dict():
    s = LineageStep(step="feature_set", name="v3", params={"n": 5}, data_hash="h")
    d = s.to_dict()
    assert d["step"] == "feature_set"
    assert d["data_hash"] == "h"
    assert "timestamp" in d


def test_model_registry_record():
    rec = ModelRegistryRecord(model="mamba", run_id="r1", seed=7, data_hash="d1", code_commit="c1", dataset_hash="ds1")
    assert rec["model"] == "mamba"
    assert rec["seed"] == 7
    assert rec["code_commit"] == "c1"


def test_decision_trail_appends():
    trail = decision_trail(model="haelt", decision="promote", decision_made=True)
    assert trail["decision_made"] is True
    assert len(trail["trail"]) == 1
    trail2 = decision_trail(
        model="haelt", decision="rollback", decision_made=True, history=[DecisionRecord("promote", "haelt", True)]
    )
    assert len(trail2["trail"]) == 2
    assert trail2["trail"][0]["decision"] == "promote"


def test_decision_trail_persists(tmp_path):
    path = tmp_path / "trail.json"
    decision_trail(model="m", decision="risk_block", decision_made=True, path=str(path))
    data = json.loads(path.read_text())
    assert data["decision"] == "risk_block"


# ═════════════════════════════════════════════════════════════════════════════
# Manifests
# ═════════════════════════════════════════════════════════════════════════════


def test_compute_file_hash_deterministic():
    h1 = compute_file_hash("audit/manifest.py")
    h2 = compute_file_hash("audit/manifest.py")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_compute_dir_hash(ckpt_dir):
    h = compute_dir_hash(str(ckpt_dir))
    assert len(h) == 64
    # order-independent
    assert h == compute_dir_hash(str(ckpt_dir))


def test_hash_inputs_deterministic_and_order_independent():
    assert hash_inputs({"a": 1, "b": [2, 3]}) == hash_inputs({"b": [2, 3], "a": 1})
    assert hash_inputs({"a": 1}) != hash_inputs({"a": 2})


def test_capture_env_shape():
    env = capture_env(include_gpu=False)
    assert "python" in env
    assert "platform" in env
    assert "libs" in env


def test_git_info_runs():
    info = git_info()
    assert "commit" in info
    assert "branch" in info


def test_generate_manifest_shape(ckpt_dir):
    m = generate_manifest(
        run_dir=str(ckpt_dir),
        run_id="r1",
        model="haelt",
        params={"lr": 1e-4},
        seed=42,
        dataset={"name": "EURUSD", "version": "1.0"},
        artifacts={"production_best.pt": str(ckpt_dir / "production_best.pt")},
    )
    assert m["manifest_version"] == 1
    assert m["model"] == "haelt"
    assert m["seed"] == 42
    assert m["artifacts"]["production_best.pt"]["sha256"] == compute_file_hash(str(ckpt_dir / "production_best.pt"))
    assert "hash" in m


def test_write_and_verify_manifest(ckpt_dir):
    m = generate_manifest(
        run_dir=str(ckpt_dir), run_id="r1", model="haelt", artifacts={"production_best.pt": "production_best.pt"}
    )
    path = write_manifest(m, str(ckpt_dir))
    assert os.path.exists(path)
    report = verify_manifest(m, run_dir=str(ckpt_dir))
    assert report["reproducible"] is True
    assert report["gates"]["hash_ok"] is True
    assert report["gates"]["artifacts_ok"] is True


def test_verify_manifest_catches_tampering(ckpt_dir):
    m = generate_manifest(
        run_dir=str(ckpt_dir), run_id="r1", model="haelt", artifacts={"production_best.pt": "production_best.pt"}
    )
    # modify the artifact on disk
    (ckpt_dir / "production_best.pt").write_bytes(b"tampered")
    report = verify_manifest(m, run_dir=str(ckpt_dir))
    assert report["reproducible"] is False
    assert report["gates"]["artifacts_ok"] is False


def test_verify_manifest_catches_hash_change():
    m = generate_manifest(run_dir=".", run_id="r1", model="m")
    m["params"]["evil"] = True
    report = verify_manifest(m)
    assert report["reproducible"] is False
    assert report["gates"]["hash_ok"] is False


def test_regenerate_manifest_preserves_lineage(ckpt_dir):
    first = generate_manifest(
        run_dir=str(ckpt_dir),
        run_id="orig",
        model="haelt",
        params={"lr": 1e-4},
        artifacts={"production_best.pt": "production_best.pt"},
    )
    write_manifest(first, str(ckpt_dir))
    rebuilt = regenerate_manifest(str(ckpt_dir))
    assert rebuilt["run_id"] == "orig"
    assert rebuilt["model"] == "haelt"
    assert rebuilt["params"]["lr"] == 1e-4
    assert rebuilt["artifacts"]["production_best.pt"]["sha256"] == compute_file_hash(
        str(ckpt_dir / "production_best.pt")
    )


def test_verify_manifests_in_tree(ckpt_dir, tmp_path):
    root = tmp_path / "models"
    d1 = root / "run_a"
    d2 = root / "run_b"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    (d1 / "model.pt").write_bytes(b"a")
    (d2 / "model.pt").write_bytes(b"b")
    write_manifest(
        generate_manifest(run_dir=str(d1), run_id="a", model="m", artifacts={"model.pt": "model.pt"}), str(d1)
    )
    write_manifest(
        generate_manifest(run_dir=str(d2), run_id="b", model="m", artifacts={"model.pt": "model.pt"}), str(d2)
    )
    reports = verify_manifests_in_tree(str(root))
    assert len(reports) == 2
    assert all(r["reproducible"] for r in reports)
