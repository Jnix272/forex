"""Sanitize / match_target_shape must not silently zero non-finite targets."""
from __future__ import annotations

import torch

from training.gpu_losses import match_target_shape
from training.supervised_loop import (
    _sanitize_batch_tensors,
    reset_sanitize_stats,
    sanitize_stats,
)


def test_sanitize_drops_bad_targets_with_counter_and_warning(capsys):
    reset_sanitize_stats()
    xb = torch.tensor([[[float("nan"), 1.0], [2.0, 3.0]]])  # (1, 2, 2) — one bad feature
    # batch of 2: first target NaN, second fine
    xb = torch.stack(
        [
            torch.tensor([[float("nan"), 1.0], [2.0, 3.0]]),
            torch.ones(2, 2),
        ]
    )
    yb = torch.tensor([float("nan"), 0.5])
    y_cls = torch.tensor([0.0, 1.0])
    y_conf = torch.tensor([0.5, 0.5])

    xb_o, yb_o, y_cls_o, y_conf_o, keep = _sanitize_batch_tensors(xb, yb, y_cls, y_conf)

    assert keep is not None
    assert keep.tolist() == [False, True]
    # Target NaN left intact (not zeroed)
    assert not torch.isfinite(yb_o[0])
    assert float(yb_o[1]) == 0.5
    stats = sanitize_stats()
    assert stats["target_rows_dropped"] == 1
    assert stats["batches_with_target_drops"] == 1
    assert stats["feature_nonfinite"] >= 1
    out = capsys.readouterr().out
    assert "WARN" in out
    assert "not zeroed" in out or "non-finite" in out


def test_sanitize_fail_closed_when_skip_disabled():
    reset_sanitize_stats()
    xb = torch.ones(1, 2, 2)
    yb = torch.tensor([float("nan")])
    try:
        _sanitize_batch_tensors(xb, yb, None, None, skip_bad_targets=False)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not zeroed" in str(exc) or "non-finite" in str(exc)


def test_match_target_shape_does_not_zero_nan(capsys):
    pred = torch.zeros(2, 1)
    target = torch.tensor([float("nan"), 1.0])
    out = match_target_shape(pred, target)
    assert out.shape == (2, 1)
    assert not torch.isfinite(out[0])
    assert float(out[1]) == 1.0
    assert "WARN" in capsys.readouterr().out
