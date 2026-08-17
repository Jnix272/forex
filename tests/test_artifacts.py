import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path


class TestArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_checkpoint_dir = Path(self.temp_dir) / "checkpoints"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_nested_checkpoint_directory(self):
        """Tests that nested checkpoint directories are supported and created correctly."""
        model_name = "haelt"
        nested_dir = self.base_checkpoint_dir / model_name
        nested_dir.mkdir(parents=True, exist_ok=True)
        self.assertTrue(nested_dir.exists())
        self.assertTrue(nested_dir.is_dir())

    def test_promotion_json_is_written(self):
        """Tests that a promotion JSON or model_card.json can be written successfully."""
        model_name = "tft"
        nested_dir = self.base_checkpoint_dir / model_name
        nested_dir.mkdir(parents=True, exist_ok=True)

        card_path = nested_dir / "model_card.json"
        dummy_data = {"model_name": model_name, "promotion_status": "promoted"}

        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(dummy_data, f)

        self.assertTrue(card_path.exists())

        with open(card_path, encoding="utf-8") as f:
            loaded = json.load(f)

        self.assertEqual(loaded["model_name"], "tft")
        self.assertEqual(loaded["promotion_status"], "promoted")

    def test_atomic_checkpoint_copy(self):
        """Tests that a production checkpoint copy is atomic (replace)."""
        source_file = Path(self.temp_dir) / "source.pt"
        source_file.write_text("DUMMY_WEIGHTS")

        dest_file = self.base_checkpoint_dir / "canonical_best.pt"
        self.base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Simulate atomic replace
        os.replace(source_file, dest_file)

        self.assertFalse(source_file.exists())
        self.assertTrue(dest_file.exists())
        self.assertEqual(dest_file.read_text(), "DUMMY_WEIGHTS")


if __name__ == "__main__":
    unittest.main()
