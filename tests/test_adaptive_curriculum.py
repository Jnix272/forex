from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from training.curriculum_controller import AdaptiveCurriculumConfig, CurriculumController


def _make_temp_dir() -> Path:
    root = Path(__file__).resolve().parent / "_tmp_adaptive_curriculum"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


class TestAdaptiveCurriculumController(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = _make_temp_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_advances_difficulty_only_after_rolling_stability(self) -> None:
        cfg = AdaptiveCurriculumConfig(
            seq_lens=(30, 60),
            stable_epochs_required=2,
            stability_window=2,
            min_stable_sharpe=0.8,
            max_sharpe_std=0.05,
        )
        controller = CurriculumController(start_stage="easy", start_seq_len=30, config=cfg)

        first = controller.evaluate_epoch(1, val_sharpe=0.82, val_loss=0.4)
        second = controller.evaluate_epoch(2, val_sharpe=0.84, val_loss=0.39)
        third = controller.evaluate_epoch(3, val_sharpe=0.83, val_loss=0.38)

        self.assertFalse(first["advance_difficulty"])
        self.assertFalse(second["advance_difficulty"])
        self.assertTrue(third["advance_difficulty"])
        self.assertTrue(third["reduce_lr"])
        self.assertEqual(third["new_stage"], "medium")
        self.assertEqual(controller.event_log()[-1]["event"], "difficulty_advance")

    def test_reverts_sequence_length_and_reduces_lr_on_sharpe_collapse(self) -> None:
        cfg = AdaptiveCurriculumConfig(
            seq_lens=(30, 60, 90),
            collapse_ratio=0.5,
            collapse_floor=0.2,
            recovery_epochs=2,
            lr_reduction_mult=0.8,
        )
        controller = CurriculumController(start_stage="medium", start_seq_len=90, config=cfg)

        controller.evaluate_epoch(1, val_sharpe=1.2, val_loss=0.4)
        action = controller.evaluate_epoch(2, val_sharpe=0.4, val_loss=0.6)

        self.assertTrue(action["delay_seq_len_increase"])
        self.assertTrue(action["revert_seq_len"])
        self.assertTrue(action["reduce_lr"])
        self.assertEqual(action["lr_multiplier"], 0.8)
        self.assertEqual(action["new_seq_len"], 60)
        self.assertEqual(controller.current_seq_len, 60)
        self.assertEqual(controller.event_log()[-1]["event"], "seq_len_revert")

    def test_recovery_holds_block_immediate_advance(self) -> None:
        cfg = AdaptiveCurriculumConfig(
            seq_lens=(30, 60),
            stable_epochs_required=1,
            stability_window=1,
            min_stable_sharpe=0.5,
            max_sharpe_std=0.1,
            recovery_epochs=1,
        )
        controller = CurriculumController(start_stage="easy", start_seq_len=60, config=cfg)

        controller.evaluate_epoch(1, val_sharpe=1.0, val_loss=0.4)
        collapse = controller.evaluate_epoch(2, val_sharpe=0.1, val_loss=0.7)
        recovery = controller.evaluate_epoch(3, val_sharpe=1.1, val_loss=0.35)

        self.assertTrue(collapse["revert_seq_len"])
        self.assertTrue(recovery["delay_seq_len_increase"])
        self.assertFalse(recovery["advance_difficulty"])
        self.assertEqual(recovery["event"], "recovery_hold")

    def test_writes_event_report(self) -> None:
        cfg = AdaptiveCurriculumConfig(
            stable_epochs_required=1,
            stability_window=1,
            min_stable_sharpe=0.5,
            max_sharpe_std=0.1,
        )
        controller = CurriculumController(config=cfg)
        controller.evaluate_epoch(1, val_sharpe=0.8, val_loss=0.4)

        out = self.temp_dir / "adaptive_curriculum_report.json"
        report = controller.write_report(out)

        self.assertTrue(out.exists())
        loaded = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(loaded["events"][0]["event"], "difficulty_advance")
        self.assertEqual(report["current_stage"], "medium")


if __name__ == "__main__":
    unittest.main()
