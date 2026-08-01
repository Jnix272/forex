from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from training.config_validate import build_config_preflight_report
from training.window_loading_plan import (
    WindowLoadingConfig,
    build_finbert_prefetch_batches,
    build_window_loading_report,
    group_date_windows,
    write_window_loading_report,
)


def _make_temp_dir() -> Path:
    root = Path(__file__).resolve().parent / "_tmp_preflight_window"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


def _args(**overrides):
    data = {
        "all_models": False,
        "auto_tune": False,
        "batch_size": 64,
        "checkpoint_dir": "checkpoints",
        "config": "config/run.yaml",
        "curriculum": {"seq_schedule": [{"epoch_start": 0, "seq_len": 60}]},
        "data_cache": "data/processed",
        "data_end": "2026-01-01",
        "data_start": "2025-01-01",
        "distill_weight": 0.0,
        "drift_gate": False,
        "dry_tune": True,
        "epochs": 10,
        "grad_accum_steps": 1,
        "lr": 0.0001,
        "lr_warmup_epochs": 2,
        "model": "haelt",
        "models": "",
        "pair": "EURUSD",
        "pairs": "EURUSD",
        "patience": 4,
        "pretrain": False,
        "pretrain_ablation": "false",
        "pretrain_ablation_models": "",
        "pretrain_epochs": 0,
        "quick_mode": True,
        "resume": False,
        "rl_train": False,
        "seq_len": 60,
        "strategy_mode": "scalping",
        "teacher_model": "",
        "train_ensemble": False,
        "training_memory": False,
        "walk_forward_cv": False,
        "walk_forward_folds": 1,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class TestConfigPreflightAndWindowLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = _make_temp_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_preflight_accepts_diverse_recipe_losses(self) -> None:
        for loss in ("cross_entropy", "sharpe_huber", "directional_huber", "asymmetric", "rmse"):
            report = build_config_preflight_report(_args(loss=loss, checkpoint_dir=str(self.temp_dir)))
            self.assertTrue(report["ok"], (loss, report["errors"]))
            self.assertIn(loss, report["supported_losses"])

    def test_preflight_rejects_unknown_model_and_bad_ranges(self) -> None:
        report = build_config_preflight_report(
            _args(
                model="unknown_model",
                loss="made_up_loss",
                epochs=0,
                lr=2.0,
                checkpoint_dir=str(self.temp_dir),
            )
        )

        self.assertFalse(report["ok"])
        joined = " ".join(report["errors"])
        self.assertIn("Unknown model", joined)
        self.assertIn("training.epochs", joined)
        self.assertIn("training.lr", joined)

    def test_finbert_prefetch_batches_are_unique_and_bounded(self) -> None:
        batches = build_finbert_prefetch_batches(
            ["ECB raises rates", "ECB raises rates", "Dollar slips", "", "Yen rallies"],
            batch_size=2,
        )

        self.assertEqual([len(batch) for batch in batches], [2, 1])
        self.assertEqual(sum(len(batch) for batch in batches), 3)

    def test_window_groups_preserve_order(self) -> None:
        groups = group_date_windows(["w1", "w2", "w3", "w4", "w5"], window_batch_days=2)

        self.assertEqual(groups, [["w1", "w2"], ["w3", "w4"], ["w5"]])

    def test_window_loading_report_flags_unsafe_workers(self) -> None:
        report = build_window_loading_report(
            ["w1", "w2"],
            ["h1", "h2"],
            WindowLoadingConfig(dataset_build_workers=8, parallel_window_workers=0),
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["gates"]["dataset_build_workers_ok"])
        self.assertFalse(report["gates"]["parallel_window_workers_ok"])

    def test_window_loading_report_writes_json(self) -> None:
        out = self.temp_dir / "window_loading_report.json"
        report = write_window_loading_report(
            ["w1", "w2", "w3"],
            ["h1", "h2", "h1"],
            out,
            WindowLoadingConfig(window_batch_days=2, finbert_batch_size=1),
        )

        self.assertTrue(out.exists())
        self.assertTrue(report["ok"])
        self.assertEqual(report["windows"]["group_count"], 2)
        self.assertEqual(report["finbert_prefetch"]["batch_count"], 2)


if __name__ == "__main__":
    unittest.main()
