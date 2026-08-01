"""Unit tests for config.settings checkpoint resolution."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import resolve_checkpoint_paths, PRODUCTION_CHECKPOINT


def test_resolve_prefers_production(tmp_path):
    ckpt = tmp_path / "run_ckpt"
    ckpt.mkdir()
    prod = ckpt / PRODUCTION_CHECKPOINT
    prod.write_text("fake", encoding="utf-8")
    model_pt = ckpt / "haelt_best.pt"
    model_pt.write_text("fake", encoding="utf-8")

    paths = resolve_checkpoint_paths("haelt", checkpoint_dir=ckpt)
    assert paths.pt_path == prod
    assert paths.source == "production"


def test_resolve_model_best_when_no_production(tmp_path):
    ckpt = tmp_path / "run_ckpt"
    ckpt.mkdir()
    model_pt = ckpt / "mamba_best.pt"
    model_pt.write_text("fake", encoding="utf-8")

    paths = resolve_checkpoint_paths("mamba", checkpoint_dir=ckpt)
    assert paths.pt_path == model_pt
    assert paths.source == "model_best"


def test_resolve_none_when_missing(tmp_path):
    ckpt = tmp_path / "empty"
    ckpt.mkdir()
    paths = resolve_checkpoint_paths("tft", checkpoint_dir=ckpt)
    assert paths.pt_path is None
    assert paths.source == "none"
