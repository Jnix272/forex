from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from data.feature_readiness import (
    DataFeatureGateConfig,
    feature_schema_hash,
    validate_data_feature_readiness,
    write_data_feature_readiness_report,
)
from validation.promotion_audit import (
    CalibrationGateConfig,
    PromotionAuditConfig,
    validate_priority2_promotion,
    write_priority2_promotion_report,
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_temp_dir() -> Path:
    root = Path(__file__).resolve().parent / "_tmp_priority_gates"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


class TestPriority2PromotionAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = _make_temp_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _write_complete_promotion_package(self) -> None:
        _write_json(
            self.temp_dir / "train_summary.json",
            {
                "model": "haelt",
                "calibration": {
                    "ece": 0.03,
                    "nll": 0.72,
                    "accuracy": 0.58,
                    "avg_confidence": 0.61,
                },
            },
        )
        _write_json(self.temp_dir / "fold_selection.json", {"model": "haelt", "selected_fold": 1})
        _write_json(
            self.temp_dir / "promotion_gate.json",
            {
                "model": "haelt",
                "promoted": True,
                "gate_input_type": "execution_backtest",
                "gates": {"sharpe_ok": True, "psr_ok": True, "dsr_ok": True},
            },
        )
        _write_json(self.temp_dir / "deployment.json", {"status": "success", "model": "haelt"})
        _write_json(self.temp_dir / "manifest.json", {"model_name": "haelt"})
        _write_json(
            self.temp_dir / "model_diagnostics_report.json",
            {"leaderboard": [{"model": "haelt", "rank": 1}], "calibration": {}},
        )

    def test_priority2_accepts_complete_execution_backed_package(self) -> None:
        self._write_complete_promotion_package()

        report = validate_priority2_promotion(
            self.temp_dir,
            PromotionAuditConfig(require_leaderboard_rank=True),
        )

        self.assertTrue(report["ready_for_production"], report["reasons"])
        self.assertEqual(report["model"], "haelt")
        self.assertEqual(report["leaderboard_rank"], 1)

    def test_priority2_rejects_proxy_gate_and_bad_calibration(self) -> None:
        self._write_complete_promotion_package()
        _write_json(
            self.temp_dir / "promotion_gate.json",
            {"model": "haelt", "promoted": True, "gate_input_type": "proxy"},
        )
        _write_json(
            self.temp_dir / "train_summary.json",
            {"model": "haelt", "calibration": {"ece": 0.4, "nll": 2.5, "accuracy": 0.4, "avg_confidence": 0.9}},
        )

        report = validate_priority2_promotion(
            self.temp_dir,
            PromotionAuditConfig(
                calibration=CalibrationGateConfig(
                    max_ece=0.08,
                    max_nll=1.25,
                    max_confidence_accuracy_gap=0.10,
                )
            ),
        )

        self.assertFalse(report["ready_for_production"])
        self.assertFalse(report["gates"]["gate_input_not_proxy"])
        self.assertFalse(report["gates"]["calibration_ece_ok"])

    def test_priority2_writes_report(self) -> None:
        self._write_complete_promotion_package()

        report = write_priority2_promotion_report(self.temp_dir)

        self.assertTrue((self.temp_dir / "priority2_promotion_report.json").exists())
        self.assertTrue(report["ready_for_production"])


class TestPriority4DataFeatureGate(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = _make_temp_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _write_complete_data_package(self) -> str:
        features = ["open", "high", "low", "close", "spread"]
        schema_hash = feature_schema_hash(features)
        _write_json(
            self.temp_dir / "dataset_manifest.json",
            {
                "source": "dukascopy",
                "pairs": ["EURUSD"],
                "sequence_length": 60,
                "feature_count": len(features),
                "schema_hash": schema_hash,
            },
        )
        _write_json(
            self.temp_dir / "feature_schema.json",
            {
                "feature_count": len(features),
                "schema_hash": schema_hash,
                "ordered_features": features,
            },
        )
        _write_json(
            self.temp_dir / "data_quality_report.json",
            {
                "missing_bars_by_pair": {"EURUSD": 0},
                "feature_nan_rates": {"open": 0.0, "spread": 0.01},
                "label_class_balance": {"down": 0.34, "flat": 0.31, "up": 0.35},
            },
        )
        _write_json(
            self.temp_dir / "pair_readiness_report.json",
            {
                "EURUSD": {
                    "status": "pass",
                    "valid_sequence_count": 1000,
                    "fail_reason": "",
                }
            },
        )
        return schema_hash

    def test_priority4_accepts_clean_data_feature_package(self) -> None:
        schema_hash = self._write_complete_data_package()

        report = validate_data_feature_readiness(
            self.temp_dir,
            DataFeatureGateConfig(
                expected_seq_len=60,
                expected_schema_hash=schema_hash,
                expected_feature_count=5,
            ),
        )

        self.assertTrue(report["ready_for_training"], report["reasons"])
        self.assertEqual(report["pairs"]["count"], 1)

    def test_priority4_rejects_schema_mismatch_and_failed_pair(self) -> None:
        self._write_complete_data_package()
        _write_json(
            self.temp_dir / "pair_readiness_report.json",
            {"EURUSD": {"status": "fail", "fail_reason": "Zero usable sequences"}},
        )

        report = validate_data_feature_readiness(
            self.temp_dir,
            DataFeatureGateConfig(expected_schema_hash="wrong"),
        )

        self.assertFalse(report["ready_for_training"])
        self.assertFalse(report["gates"]["schema_hash_ok"])
        self.assertFalse(report["gates"]["pair_readiness_ok"])

    def test_priority4_writes_report(self) -> None:
        self._write_complete_data_package()

        report = write_data_feature_readiness_report(self.temp_dir)

        self.assertTrue((self.temp_dir / "priority4_data_feature_report.json").exists())
        self.assertTrue(report["ready_for_training"])


if __name__ == "__main__":
    unittest.main()
