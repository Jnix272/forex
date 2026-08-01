from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from training.model_training_audit import (
    ModelTrainingAuditConfig,
    validate_model_training_package,
    write_model_training_audit_report,
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_temp_dir() -> Path:
    root = Path(__file__).resolve().parent / "_tmp_priority5"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


class TestPriority5ModelTrainingAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = _make_temp_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _write_complete_training_package(self) -> None:
        _write_json(
            self.temp_dir / "model_card.json",
            {
                "model_name": "haelt",
                "architecture": "haelt",
                "training_data": {
                    "window": "2024-01-01/2026-01-01",
                    "pairs": ["EURUSD"],
                    "features_count": 128,
                    "label_method": "rl_reward",
                },
                "performance": {
                    "validation_sharpe": 1.72,
                    "validation_loss": 0.41,
                    "forward_holdout_pnl": 2500.0,
                },
                "known_weaknesses": [],
                "promotion_status": "candidate",
            },
        )
        _write_json(
            self.temp_dir / "training_control_report.json",
            {
                "model_recipe_used": "haelt",
                "memory_suggestions": [],
                "overfitting_signals_detected": [],
                "curriculum_changes": ["easy_to_medium"],
                "lr_changes": [],
                "best_epoch": 8,
                "restore_decision": True,
                "final_promotion_recommendation": "candidate",
            },
        )
        _write_json(
            self.temp_dir / "train_summary.json",
            {
                "model": "haelt",
                "recipe": "haelt",
                "best_val_sharpe": 1.72,
                "train_val_gap": 0.08,
            },
        )
        _write_json(
            self.temp_dir / "pretrain_ablation.json",
            {
                "verdict": "pretrain_helped",
                "metric_deltas": {"val_sharpe": 0.12},
            },
        )

    def test_priority5_accepts_clean_training_package(self) -> None:
        self._write_complete_training_package()

        report = validate_model_training_package(
            self.temp_dir,
            ModelTrainingAuditConfig(min_best_val_sharpe=1.0),
        )

        self.assertTrue(report["ready_for_validation"], report["reasons"])
        self.assertEqual(report["model"], "haelt")
        self.assertEqual(report["recipe_info"]["role"], "primary")

    def test_priority5_rejects_unknown_recipe_overfit_and_bad_pretrain(self) -> None:
        self._write_complete_training_package()
        _write_json(
            self.temp_dir / "training_control_report.json",
            {
                "model_recipe_used": "unknown_recipe",
                "memory_suggestions": "not-a-list",
                "overfitting_signals_detected": ["gap", "collapse", "confidence_miss"],
                "best_epoch": 3,
                "restore_decision": False,
            },
        )
        _write_json(
            self.temp_dir / "train_summary.json",
            {
                "model": "unknown_recipe",
                "best_val_sharpe": 0.2,
                "train_val_gap": 0.7,
            },
        )
        _write_json(self.temp_dir / "pretrain_ablation.json", {"verdict": "pretrain_hurt"})

        report = validate_model_training_package(
            self.temp_dir,
            ModelTrainingAuditConfig(min_best_val_sharpe=1.0),
        )

        self.assertFalse(report["ready_for_validation"])
        self.assertFalse(report["gates"]["model_identity_consistent"])
        self.assertFalse(report["gates"]["best_epoch_restored"])
        self.assertFalse(report["gates"]["overfit_signal_count_ok"])
        self.assertFalse(report["gates"]["train_val_gap_ok"])
        self.assertFalse(report["gates"]["pretrain_ablation_ok"])
        self.assertFalse(report["gates"]["memory_suggestions_audited"])

    def test_priority5_writes_report(self) -> None:
        self._write_complete_training_package()

        report = write_model_training_audit_report(self.temp_dir)

        self.assertTrue((self.temp_dir / "priority5_model_training_report.json").exists())
        self.assertTrue(report["ready_for_validation"])


if __name__ == "__main__":
    unittest.main()
