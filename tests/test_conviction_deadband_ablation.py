"""Tests for Conviction Deadband Hurdle Rate and Feature Ablation.

Verifies:
1. decide() filters out micro-noise predictions below deadband hurdle.
2. decide() rejects predictions where direction contradicts return forecast.
3. Feature ablation mask properly retains top 60-80 features and masks noise.
"""

import numpy as np
import pytest
import torch

from training.decision import decide
from training.feature_ablation import _build_feature_ablation_mask


def test_decide_deadband_tensor_regression():
    # Signal in [-deadband, +deadband] should become 0 (HOLD)
    # Signal > +deadband becomes +1 (BUY)
    # Signal < -deadband becomes -1 (SELL)
    deadband = 0.15
    pred = torch.tensor([-0.50, -0.15, -0.05, 0.0, 0.10, 0.15, 0.40])
    d = decide(pred, deadband=deadband)
    expected = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert d.tolist() == expected


def test_decide_deadband_tuple_multitask():
    deadband = 0.15
    # Row 0: Strong BUY, ret_hat +0.30 > 0.15 -> BUY (+1)
    # Row 1: Strong BUY logit, but ret_hat +0.05 <= 0.15 -> HOLD (0)
    # Row 2: Strong BUY logit, but ret_hat -0.25 contradicts direction -> HOLD (0)
    # Row 3: Strong SELL logit, ret_hat -0.30 < -0.15 -> SELL (-1)
    # Row 4: Low confidence -> HOLD (0)
    logits = torch.tensor([3.0, 3.0, 3.0, -3.0, 3.0])
    ret_hat = torch.tensor([0.30, 0.05, -0.25, -0.30, 0.50])
    conf = torch.tensor([3.0, 3.0, 3.0, 3.0, -3.0])  # row 4 has negative conf

    d = decide((logits, ret_hat, conf), threshold=0.45, deadband=deadband)
    assert d.tolist() == [1.0, 0.0, 0.0, -1.0, 0.0]


def test_decide_backward_compatibility():
    # With deadband=0.0, standard behavior is preserved
    logits = torch.tensor([3.0, -3.0, 3.0])
    conf = torch.tensor([3.0, 3.0, -3.0])
    d = decide((logits, torch.zeros(3), conf), deadband=0.0)
    assert d.tolist() == [1.0, -1.0, 0.0]


def test_feature_ablation_top_80():
    # Test that 3 core groups + drop_ungrouped yields exactly 80 active features on the schema
    feature_groups = {
        "momentum": {"features": ["rsi_14", "macd", "macd_sig", "macd_hist", "ret_5", "ret_20", "ret_60"]},
        "microstructure": {"features": ["ofi", "ofi_z", "ofi_l2", "kyles_lambda", "amihud_illiq", "realized_spread", "vpin"]},
        "execution_cost": {"features": ["spread_pips", "spread_zscore", "spread_percentile", "spread_widening_5m", "spread_widening_20m", "cost_to_atr"]},
        "noise_group": {"features": ["dummy_1", "dummy_2"]},
    }
    schema = []
    pairs = ["EURUSD", "GBPUSD", "USDCAD", "USDJPY"]
    # 20 features per pair across the 3 kept groups = 80 features
    for p in pairs:
        for g in ["momentum", "microstructure", "execution_cost"]:
            for f in feature_groups[g]["features"]:
                schema.append(f"{p}::{f}")
        for f in feature_groups["noise_group"]["features"]:
            schema.append(f"{p}::{f}")
        # Ungrouped features
        schema.append(f"{p}::open")
        schema.append(f"{p}::close")

    # Schema has 80 (kept) + 8 (noise) + 8 (ungrouped) = 96 features
    cfg = {
        "enabled": True,
        "name": "test_top_80",
        "keep_groups": ["momentum", "microstructure", "execution_cost"],
        "drop_ungrouped": True,
    }
    mask, report = _build_feature_ablation_mask(schema, feature_groups, cfg, len(schema))
    assert report["active_count"] == 80
    assert report["masked_count"] == 16
    assert (mask == 1.0).sum() == 80
    assert (mask == 0.0).sum() == 16


def test_feature_ablation_explicit_keep_features():
    schema = ["EURUSD::rsi_14", "EURUSD::close", "GBPUSD::rsi_14", "GBPUSD::close"]
    cfg = {
        "enabled": True,
        "keep_features": ["rsi_14"],
    }
    mask, report = _build_feature_ablation_mask(schema, {}, cfg, len(schema))
    assert report["active_count"] == 2
    assert report["masked_count"] == 2
    assert mask[0] == 1.0 and mask[2] == 1.0
    assert mask[1] == 0.0 and mask[3] == 0.0


def test_feature_ablation_stationary_curated():
    # Test that stationary_curated keeps economic signals and drops dead fourier and raw prices
    feature_groups = {
        "cross_asset": {"features": ["spread_us_de", "carry_eur"]},
        "session": {"features": ["time_cos", "london_ny"]},
        "momentum": {"features": ["rsi_14"]},
        "higher_timeframe": {"features": ["ret_1h", "atr_1h"]},
        "execution_cost": {"features": ["spread_pips"]},
        "microstructure": {"features": ["ofi_z"]},
        "macro": {"features": ["fb_0", "fb_1"]},
    }
    schema = [
        "EURUSD::spread_us_de", "EURUSD::carry_eur", "EURUSD::time_cos",
        "EURUSD::london_ny", "EURUSD::rsi_14", "EURUSD::ret_1h",
        "EURUSD::atr_1h", "EURUSD::spread_pips", "EURUSD::ofi_z",
        "EURUSD::fb_0", "EURUSD::fb_1",  # dead fourier in macro
        "EURUSD::close",  # ungrouped raw price
    ]
    cfg = {
        "enabled": True,
        "name": "stationary_curated",
        "keep_groups": ["cross_asset", "session", "momentum", "higher_timeframe", "execution_cost", "microstructure"],
        "drop_ungrouped": True,
        "drop_features": ["fb_0", "fb_1"],
    }
    mask, report = _build_feature_ablation_mask(schema, feature_groups, cfg, len(schema))
    assert report["active_count"] == 9
    assert report["masked_count"] == 3
    assert mask[schema.index("EURUSD::close")] == 0.0
    assert mask[schema.index("EURUSD::fb_0")] == 0.0
    assert mask[schema.index("EURUSD::fb_1")] == 0.0
    assert mask[schema.index("EURUSD::spread_us_de")] == 1.0
    assert mask[schema.index("EURUSD::time_cos")] == 1.0

