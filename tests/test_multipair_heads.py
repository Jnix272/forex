"""Unit tests for Track B: Multi-Task Per-Pair Heads Architecture.

Tests:
  - MultiPairMultiTaskHead output shapes (B, 4), indexing, quantile heads
  - MultiPairMultiTaskWrapper with backbones (HAELT, Mamba, TFT)
  - MultiPairMultiTaskLoss computation and gradient backprop to all heads and backbone
  - MultiPairMultiTaskLoss with custom pair weights
  - build_model integration with per_pair_heads=True
  - MultiPairChunk backward-compatible unpacking (9-item and 11-item access)
  - compute_batch_loss dispatch with MultiPairMultiTaskLoss and (B, P) shapes
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.architectures import (
    HAELTHybrid,
    MambaScalper,
    MultiPairMultiTaskHead,
    MultiPairMultiTaskLoss,
    MultiPairMultiTaskWrapper,
    MultiTaskHead,
    MultiTaskLoss,
    MultiTaskWrapper,
    TFTScalper,
    build_model,
)
from training.dataset_builder import MultiPairChunk
from training.loop_losses import compute_batch_loss


@pytest.fixture
def sample_batch():
    """Batch of shape (B=8, T=30, F=32)."""
    torch.manual_seed(42)
    return torch.randn(8, 30, 32)


class TestMultiPairMultiTaskHead:
    """Tests for MultiPairMultiTaskHead shape, indexing, and quantiles."""

    def test_output_shapes_4_pairs_with_quantiles(self):
        B, in_feat, P = 8, 64, 4
        h = torch.randn(B, in_feat)
        pairs = ["EURUSD", "GBPUSD", "USDCAD", "USDJPY"]
        head = MultiPairMultiTaskHead(
            in_features=in_feat,
            pairs=pairs,
            hidden=32,
            quantile_enabled=True,
        )
        assert len(head) == 4
        outs = head(h)
        assert len(outs) == 5, f"Expected 5 output tensors (dir, ret, conf, q_low, q_high), got {len(outs)}"
        logits, ret_hat, conf, q_low, q_high = outs
        assert logits.shape == (B, P), f"logits: {logits.shape}"
        assert ret_hat.shape == (B, P), f"ret_hat: {ret_hat.shape}"
        assert conf.shape == (B, P), f"conf: {conf.shape}"
        assert q_low.shape == (B, P), f"q_low: {q_low.shape}"
        assert q_high.shape == (B, P), f"q_high: {q_high.shape}"

    def test_output_shapes_without_quantiles(self):
        B, in_feat, P = 4, 48, 3
        h = torch.randn(B, in_feat)
        head = MultiPairMultiTaskHead(
            in_features=in_feat,
            pairs=P,
            hidden=32,
            quantile_enabled=False,
        )
        outs = head(h)
        assert len(outs) == 3
        logits, ret_hat, conf = outs
        assert logits.shape == (B, P)
        assert ret_hat.shape == (B, P)
        assert conf.shape == (B, P)

    def test_head_indexing_by_int_and_name(self):
        pairs = ["EURUSD", "GBPUSD"]
        head = MultiPairMultiTaskHead(in_features=32, pairs=pairs, hidden=16)
        assert isinstance(head[0], MultiTaskHead)
        assert isinstance(head["EURUSD"], MultiTaskHead)
        assert isinstance(head["GBPUSD"], MultiTaskHead)
        assert head[0] is head["EURUSD"]


class TestMultiPairMultiTaskWrapper:
    """Tests for MultiPairMultiTaskWrapper with backbones."""

    @pytest.mark.parametrize("model_name", ["haelt", "mamba", "tft"])
    def test_wrapper_with_backbones(self, model_name, sample_batch):
        B, T, F = sample_batch.shape
        n_pairs = 4

        base = build_model(
            model_name,
            F,
            seq_len=T,
            hidden_size=64,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.0,
        )
        head_in = 64
        wrapped = MultiPairMultiTaskWrapper(
            base,
            head_in=head_in,
            pairs=n_pairs,
            hidden=32,
            force_project=True,
            quantile_enabled=True,
        )
        wrapped.eval()
        outs = wrapped(sample_batch)
        assert len(outs) == 5
        logits, ret_hat, conf, q_low, q_high = outs
        assert logits.shape == (B, n_pairs)
        assert ret_hat.shape == (B, n_pairs)
        assert conf.shape == (B, n_pairs)
        assert q_low.shape == (B, n_pairs)
        assert q_high.shape == (B, n_pairs)

    def test_build_model_factory_per_pair_heads(self, sample_batch):
        B, T, F = sample_batch.shape
        model = build_model(
            "haelt",
            F,
            seq_len=T,
            per_pair_heads=True,
            n_pair_heads=4,
            d_model=64,
        )
        assert isinstance(model, MultiPairMultiTaskWrapper)
        outs = model(sample_batch)
        assert outs[0].shape == (B, 4)
        assert outs[1].shape == (B, 4)
        assert outs[2].shape == (B, 4)


class TestMultiPairMultiTaskLoss:
    """Tests for MultiPairMultiTaskLoss and gradient backprop."""

    def test_loss_scalar_and_gradients_propagate_to_all_heads_and_backbone(self, sample_batch):
        B, T, F = sample_batch.shape
        P = 4
        pairs = ["EURUSD", "GBPUSD", "USDCAD", "USDJPY"]

        base = HAELTHybrid(input_size=F, seq_len=T, lstm_hidden=32, d_model=32)
        model = MultiPairMultiTaskWrapper(base, head_in=64, pairs=pairs, hidden=32, force_project=True)
        crit = MultiPairMultiTaskLoss(w_dir=1.0, w_ret=0.5, w_conf=0.3, w_quantile=0.2)

        outs = model(sample_batch)
        logits, ret_hat, conf, q_low, q_high = outs

        y_cls = torch.randint(0, 3, (B, P)).long()
        y_cont = torch.randn(B, P)

        loss = crit(
            logits=logits,
            ret_hat=ret_hat,
            conf=conf,
            y_cls=y_cls,
            y_cont=y_cont,
            q_low=q_low,
            q_high=q_high,
        )

        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert torch.isfinite(loss)
        assert loss.item() > 0

        loss.backward()

        # Verify gradients reached the shared backbone
        backbone_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.backbone.parameters()
        )
        assert backbone_grad, "No gradient propagated to shared backbone parameters"

        # Verify gradients reached each of the 4 independent pair heads
        for p_name in pairs:
            head_p = model.mt_head[p_name]
            head_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in head_p.parameters()
            )
            assert head_grad, f"No gradient reached head for pair {p_name}"

    def test_pair_weights_influence_loss(self):
        B, P = 4, 2
        logits = torch.randn(B, P, requires_grad=True)
        ret_hat = torch.randn(B, P, requires_grad=True)
        conf = torch.randn(B, P, requires_grad=True)
        y_cls = torch.zeros(B, P, dtype=torch.long)
        y_cont = torch.ones(B, P)

        # Equal weighting
        crit_eq = MultiPairMultiTaskLoss(w_quantile=0.0, pair_weights=[0.5, 0.5])
        loss_eq = crit_eq(logits, ret_hat, conf, y_cls, y_cont)

        # Skewed weighting (100% on pair 0)
        crit_skew = MultiPairMultiTaskLoss(w_quantile=0.0, pair_weights=[1.0, 0.0])
        loss_skew = crit_skew(logits, ret_hat, conf, y_cls, y_cont)

        assert torch.isfinite(loss_eq)
        assert torch.isfinite(loss_skew)
        assert not torch.allclose(loss_eq, loss_skew)


class TestMultiPairChunkCompatibility:
    """Tests for MultiPairChunk backward-compatible 9-item and 11-item unpacking."""

    def test_nine_value_unpacking_and_attribute_access(self):
        N, T, F, P = 10, 20, 16, 4
        X = np.zeros((N, T, F), dtype=np.float32)
        y = np.zeros(N, dtype=np.float32)
        y_cls = np.zeros(N, dtype=np.float32)
        pq = np.zeros(N, dtype=np.float32)
        diff = np.zeros(N, dtype=np.uint8)
        close = np.zeros(N, dtype=np.float32)
        atr = np.zeros(N, dtype=np.float32)
        spread = np.zeros(N, dtype=np.float32)
        n_feat = F

        y_pairs = np.ones((N, P), dtype=np.float32)
        ycls_pairs = np.ones((N, P), dtype=np.int64)

        chunk = MultiPairChunk(
            (X, y, y_cls, pq, diff, close, atr, spread, n_feat),
            y_pairs=y_pairs,
            ycls_pairs=ycls_pairs,
        )

        # Unpack as 9 items (existing callers)
        c_X, c_y, c_ycls, c_pq, c_diff, c_close, c_atr, c_spread, c_nfeat = chunk
        assert c_X.shape == (N, T, F)
        assert c_y.shape == (N,)
        assert c_nfeat == F

        # Access per-pair arrays via attributes
        assert chunk.y_pairs.shape == (N, P)
        assert chunk.ycls_pairs.shape == (N, P)

        # Access per-pair arrays via 10th and 11th index
        assert chunk[9].shape == (N, P)
        assert chunk[10].shape == (N, P)

        # 11-tuple export
        t11 = chunk.to_11_tuple()
        assert len(t11) == 11
        assert t11[9].shape == (N, P)
        assert t11[10].shape == (N, P)


class TestComputeBatchLossDispatch:
    """Tests for compute_batch_loss dispatching MultiPairMultiTaskLoss and (B, P) shapes."""

    def test_compute_batch_loss_with_multipair_loss(self):
        B, P = 4, 3
        logits = torch.randn(B, P)
        ret_hat = torch.randn(B, P)
        conf = torch.randn(B, P)
        q_low = torch.randn(B, P)
        q_high = torch.randn(B, P)
        model_out = (logits, ret_hat, conf, q_low, q_high)

        yb = torch.randn(B, P)
        y_cls = torch.randint(0, 3, (B, P)).long()
        crit = MultiPairMultiTaskLoss()

        loss = compute_batch_loss(
            model_out=model_out,
            crit=crit,
            yb=yb,
            classification=False,
            y_cls=y_cls,
            multitask=True,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_compute_batch_loss_with_multitask_loss_and_2d_targets(self):
        B, P = 4, 3
        logits = torch.randn(B, P)
        ret_hat = torch.randn(B, P)
        conf = torch.randn(B, P)
        model_out = (logits, ret_hat, conf)

        yb = torch.randn(B, P)
        y_cls = torch.randint(0, 3, (B, P)).long()
        crit = MultiTaskLoss(w_quantile=0.0)

        loss = compute_batch_loss(
            model_out=model_out,
            crit=crit,
            yb=yb,
            classification=False,
            y_cls=y_cls,
            multitask=True,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0
