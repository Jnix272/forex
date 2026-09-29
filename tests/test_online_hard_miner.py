from pathlib import Path

import numpy as np
import torch

from training.hard_example_miner import OnlineHardExampleMiner
from training.loop_losses import _apply_online_miner, _per_sample_direction_loss

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_multi_pair_logits_give_one_finite_loss_per_sample():
    logits = torch.tensor([[2.0, -2.0], [0.0, 0.0], [1.0, 1.0]])
    y_idx = torch.tensor([[2, 0], [1, 1], [0, 1]])

    loss = _per_sample_direction_loss(logits, y_idx)

    assert loss.shape == (3,)
    assert loss[0] < 0.2
    assert torch.isnan(loss[1])
    assert loss[2] > 1.0


def test_three_class_logits_use_cross_entropy():
    logits = torch.tensor([[0.0, 0.0, 5.0], [5.0, 0.0, 0.0]])
    loss = _per_sample_direction_loss(logits, torch.tensor([2, 2]))
    assert loss[0] < 0.1 < 4.0 < loss[1]


def test_apply_online_miner_records_multi_pair_batch():
    batch, pairs = 8, 4
    miner = OnlineHardExampleMiner(n_samples=batch)
    miner.begin_epoch()
    logits = torch.randn(batch, pairs)
    y_cls = torch.tensor([[1.0, -1.0, 1.0, -1.0]] * batch)
    pred = (logits, torch.randn(batch, pairs), torch.rand(batch, pairs))

    _apply_online_miner(miner, pred, torch.randn(batch, pairs), y_cls, torch.arange(batch), False, True)

    recorded = miner._loss_buffer[-1]
    assert np.isfinite(recorded).all()
    assert len(np.unique(recorded)) > 1


def test_row_ids_map_dataset_rows_for_non_zero_based_split():
    rows = np.arange(1000, 1100, dtype=np.int64)
    miner = OnlineHardExampleMiner(n_samples=len(rows), row_ids=rows)
    for _ in range(3):
        miner.begin_epoch()
        losses = np.full(len(rows), 0.1, dtype=np.float32)
        losses[-10:] = 5.0
        miner.update_batch(rows, losses)
        miner.update_batch(np.array([5, 50]), np.array([9.0, 9.0]))
        miner.end_epoch()

    hard = miner.get_hard_indices()
    assert set(hard.tolist()) >= set(range(1090, 1100))
    assert hard.min() >= 1000

    out = miner.get_oversampled_indices(rows)
    assert set(out.tolist()) <= set(rows.tolist())


def test_oversampling_never_adds_rows_outside_base():
    miner = OnlineHardExampleMiner(n_samples=100)
    for _ in range(3):
        miner.begin_epoch()
        losses = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        miner.update_batch(np.arange(100), losses)
        miner.end_epoch()

    base = np.arange(0, 50)
    out = miner.get_oversampled_indices(base)
    assert set(out.tolist()) <= set(base.tolist())


def test_non_finite_losses_are_ignored_and_epoch_losses_fill_gaps():
    miner = OnlineHardExampleMiner(n_samples=4)
    miner.begin_epoch()
    miner.update_batch(np.arange(4), np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32))

    assert np.isnan(miner._loss_buffer[-1, 1])
    filled = miner.epoch_losses()
    np.testing.assert_allclose(filled, [1.0, 2.0, 3.0, 2.0])


def test_epoch_losses_none_when_nothing_observed():
    miner = OnlineHardExampleMiner(n_samples=4)
    miner.begin_epoch()
    assert miner.epoch_losses() is None


def test_supervised_loop_finalises_miner_once_per_epoch():
    src = (_REPO_ROOT / "training" / "supervised_loop.py").read_text(encoding="utf-8")
    assert src.count("_online_miner.end_epoch()") == 1
