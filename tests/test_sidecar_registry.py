from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from data.sidecar_registry import (
    SidecarSpec,
    file_sha256,
    validate_sidecar_index,
    write_sidecar_index,
)


def _make_temp_dir() -> Path:
    root = Path(__file__).resolve().parent / "_tmp_sidecars"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


class TestSidecarRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = _make_temp_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_writes_and_validates_sidecar_index(self) -> None:
        (self.temp_dir / "dataset_manifest.json").write_text('{"seq_len": 60}', encoding="utf-8")
        (self.temp_dir / "feature_schema.json").write_text('{"feature_count": 5}', encoding="utf-8")

        index = write_sidecar_index(
            self.temp_dir,
            [
                SidecarSpec("manifest", "dataset_manifest.json", kind="dataset_manifest"),
                SidecarSpec("schema", "feature_schema.json", kind="feature_schema"),
            ],
            package_name="training_cache",
            metadata={"pair": "EURUSD"},
        )

        self.assertEqual(index["package_name"], "training_cache")
        self.assertTrue((self.temp_dir / "sidecar_index.json").exists())

        report = validate_sidecar_index(self.temp_dir / "sidecar_index.json")

        self.assertTrue(report["valid"], report["reasons"])
        self.assertTrue(report["gates"]["manifest_present"])
        self.assertTrue(report["gates"]["schema_hash_ok"])

    def test_optional_missing_sidecar_is_allowed(self) -> None:
        (self.temp_dir / "dataset_manifest.json").write_text("{}", encoding="utf-8")

        write_sidecar_index(
            self.temp_dir,
            [
                SidecarSpec("manifest", "dataset_manifest.json"),
                SidecarSpec("oanda_report", "oanda_sentiment_report.json", required=False),
            ],
        )

        report = validate_sidecar_index(self.temp_dir / "sidecar_index.json")

        self.assertTrue(report["valid"], report["reasons"])
        self.assertTrue(report["gates"]["oanda_report_present"])

    def test_detects_hash_drift(self) -> None:
        target = self.temp_dir / "promotion_gate.json"
        target.write_text('{"promoted": true}', encoding="utf-8")
        original_hash = file_sha256(target)

        write_sidecar_index(
            self.temp_dir,
            [SidecarSpec("promotion_gate", "promotion_gate.json", kind="promotion")],
        )
        target.write_text('{"promoted": false}', encoding="utf-8")

        report = validate_sidecar_index(self.temp_dir / "sidecar_index.json")

        self.assertNotEqual(original_hash, file_sha256(target))
        self.assertFalse(report["valid"])
        self.assertFalse(report["gates"]["promotion_gate_hash_ok"])
        self.assertIn("sha256 mismatch", " ".join(report["reasons"]))

    def test_index_is_json_serializable(self) -> None:
        (self.temp_dir / "training_control_report.json").write_text("{}", encoding="utf-8")

        write_sidecar_index(
            self.temp_dir,
            [SidecarSpec("training_control", "training_control_report.json", kind="training")],
        )
        loaded = json.loads((self.temp_dir / "sidecar_index.json").read_text(encoding="utf-8"))

        self.assertEqual(loaded["sidecars"][0]["name"], "training_control")
        self.assertEqual(loaded["sidecars"][0]["kind"], "training")


if __name__ == "__main__":
    unittest.main()
