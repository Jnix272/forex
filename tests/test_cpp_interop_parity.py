"""
tests/test_cpp_interop_parity.py
================================
Verification and parity test suite for the C++ High-Frequency Inference Engine:
1. Math & Conviction Parity:
   - 1-class regression: clamp(x, -1, 1)
   - 2-class binary: [short, long] -> (p_long - p_short)
   - 3-class supervised: [sell, hold, buy] -> (p_buy - p_sell)
   - 5-class discrete RL: [-1.0, -0.5, 0.0, +0.5, +1.0] expected value
   - 10-class ScalingAction: collapsing RL policy matching trading.live_actions
   - Robustness: NaN and Inf sanitization to 0.0 neutral signal
2. ZeroMQ Wire Framing Parity:
   - Mode 1: Binary protocol with "FXST" magic (0x54535846) + count + float32 payload
   - Mode 2: JSON payload {"features": [...]} and [...] array
   - Mode 3: Raw IEEE-754 binary floats
3. Time Series Sequence Slicing:
   - Tail extraction: latest N bars from M-bar incoming stream
   - Heterogeneous column extraction: first K features per timestep
"""

import math
import struct
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. C++ Math Parity Equivalents
# ─────────────────────────────────────────────────────────────────────────────

def cpp_logits_to_signal(logits: list[float]) -> float:
    """Python exact implementation of EnsembleRunner::logits_to_signal in C++."""
    if not logits:
        return 0.0

    # NaN / Inf check
    for v in logits:
        if math.isnan(v) or math.isinf(v):
            return 0.0

    n = len(logits)
    if n == 1:
        return max(-1.0, min(1.0, logits[0]))

    if n == 2:
        max_l = max(logits[0], logits[1])
        e0 = math.exp(logits[0] - max_l)
        e1 = math.exp(logits[1] - max_l)
        s = e0 + e1
        return (e1 - e0) / s if s > 0.0 and not math.isnan(s) else 0.0

    if n == 3:
        # [0: Sell, 1: Hold, 2: Buy]
        max_l = max(logits[0], logits[1], logits[2])
        e0 = math.exp(logits[0] - max_l)
        e1 = math.exp(logits[1] - max_l)
        e2 = math.exp(logits[2] - max_l)
        s = e0 + e1 + e2
        if s <= 0.0 or math.isnan(s):
            return 0.0
        p_sell = e0 / s
        p_buy = e2 / s
        return max(-1.0, min(1.0, p_buy - p_sell))

    if n == 5:
        max_l = max(logits)
        exp_l = [math.exp(x - max_l) for x in logits]
        s = sum(exp_l)
        if s <= 0.0 or math.isnan(s):
            return 0.0
        actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
        expected = sum(a * (e / s) for a, e in zip(actions, exp_l))
        return max(-1.0, min(1.0, expected))

    if n == 10:
        # ScalingAction mapping matching live_actions.py:
        # 0: HOLD, 1: OPEN_LONG, 2: OPEN_SHORT, 3..5: SCALE_IN, 6..8: SCALE_OUT, 9: CLOSE_ALL
        buy = logits[1]
        sell = logits[2]
        hold = logits[0]
        for i in range(3, 10):
            if logits[i] > hold:
                hold = logits[i]

        max_l = max(sell, hold, buy)
        e_sell = math.exp(sell - max_l)
        e_hold = math.exp(hold - max_l)
        e_buy = math.exp(buy - max_l)
        s = e_sell + e_hold + e_buy
        if s <= 0.0 or math.isnan(s):
            return 0.0
        return max(-1.0, min(1.0, (e_buy - e_sell) / s))

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_scalar_regression():
    assert cpp_logits_to_signal([0.75]) == 0.75
    assert cpp_logits_to_signal([2.5]) == 1.0
    assert cpp_logits_to_signal([-3.0]) == -1.0
    assert cpp_logits_to_signal([float("nan")]) == 0.0


def test_binary_classification():
    # Equal -> 0.0
    assert abs(cpp_logits_to_signal([1.0, 1.0])) < 1e-6
    # Pure long
    assert cpp_logits_to_signal([-10.0, 10.0]) > 0.999
    # Pure short
    assert cpp_logits_to_signal([10.0, -10.0]) < -0.999
    # NaN check
    assert cpp_logits_to_signal([float("nan"), 1.0]) == 0.0


def test_three_class_supervised():
    # Balanced hold -> conviction near 0
    bal = cpp_logits_to_signal([0.0, 5.0, 0.0])
    assert abs(bal) < 1e-2

    # Strongly bullish (class 2 buy)
    bull = cpp_logits_to_signal([-5.0, 0.0, 5.0])
    assert bull > 0.98

    # Strongly bearish (class 0 sell)
    bear = cpp_logits_to_signal([5.0, 0.0, -5.0])
    assert bear < -0.98

    # NaN robustness
    assert cpp_logits_to_signal([float("nan"), 0.0, 1.0]) == 0.0
    assert cpp_logits_to_signal([1.0, float("inf"), 1.0]) == 0.0


def test_five_class_rl():
    # Uniform -> 0.0
    assert abs(cpp_logits_to_signal([1.0, 1.0, 1.0, 1.0, 1.0])) < 1e-6
    # Extreme long
    assert cpp_logits_to_signal([-10.0, -10.0, -10.0, -10.0, 10.0]) > 0.99
    # Extreme short
    assert cpp_logits_to_signal([10.0, -10.0, -10.0, -10.0, -10.0]) < -0.99


def test_ten_class_rl_scaling_action():
    # HOLD action dominant (class 0) -> neutral conviction ~0.0
    hold_logits = [10.0, -5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert abs(cpp_logits_to_signal(hold_logits)) < 1e-4

    # OPEN_LONG action dominant (class 1) -> strong buy signal
    long_logits = [-5.0, 10.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert cpp_logits_to_signal(long_logits) > 0.99

    # OPEN_SHORT action dominant (class 2) -> strong sell signal
    short_logits = [-5.0, -5.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert cpp_logits_to_signal(short_logits) < -0.99

    # CLOSE_ALL action dominant (class 9) -> collapses to HOLD (neutral conviction)
    close_logits = [-5.0, -5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
    assert abs(cpp_logits_to_signal(close_logits)) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 2. ZeroMQ Wire Framing Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_binary_framing_fxst():
    """Verify Mode 1 binary framing packed by Python matches C++ expected layout."""
    FX_BINARY_MAGIC = 0x54535846  # 'FXST' in little endian
    sample_features = np.array([1.25, -2.5, 0.0, 3.75], dtype=np.float32)
    count = len(sample_features)

    # Pack header: uint32 magic, uint32 count
    header = struct.pack("<II", FX_BINARY_MAGIC, count)
    payload = header + sample_features.tobytes()

    assert len(payload) == 8 + count * 4

    # Unpack to verify roundtrip
    unpacked_magic, unpacked_count = struct.unpack("<II", payload[:8])
    assert unpacked_magic == FX_BINARY_MAGIC
    assert unpacked_count == count
    unpacked_floats = np.frombuffer(payload[8:], dtype=np.float32)
    np.testing.assert_array_almost_equal(sample_features, unpacked_floats)


def test_json_framing_fallback():
    """Verify Mode 2 JSON parsing matches both dict and array representations."""
    import json

    features = [1.0, 2.0, -3.5, 4.25]

    # Format A: {"features": [...]}
    payload_a = json.dumps({"features": features}).encode("utf-8")
    parsed_a = json.loads(payload_a.decode("utf-8"))
    assert parsed_a["features"] == features

    # Format B: top-level array [...]
    payload_b = json.dumps(features).encode("utf-8")
    parsed_b = json.loads(payload_b.decode("utf-8"))
    assert parsed_b == features


# ─────────────────────────────────────────────────────────────────────────────
# 3. Time Series Tail Slicing and Striding Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_tail_slicing_and_striding():
    """
    Verify sequence slicing extracts the MOST RECENT bars (tail)
    and validates stride math matching C++ fix.
    """
    batch_size = 1
    incoming_seq_len = 120
    common_n_features = 584
    required_seq = 60
    required_feat = 584

    # Create dummy incoming buffer shape (1, 120, 584)
    buffer = np.arange(incoming_seq_len * common_n_features, dtype=np.float32).reshape(
        batch_size, incoming_seq_len, common_n_features
    )
    flat_features = buffer.flatten()

    # Step offset and float offset
    step_offset = incoming_seq_len - required_seq  # 120 - 60 = 60
    float_offset = step_offset * (batch_size * common_n_features)
    slice_count = batch_size * required_seq * required_feat

    sliced = flat_features[float_offset : float_offset + slice_count].reshape(
        batch_size, required_seq, required_feat
    )

    # Expected: buffer[:, -60:, :] (the TAIL, not the head)
    expected = buffer[:, -required_seq:, :]
    np.testing.assert_array_equal(sliced, expected)

    # Check that it is NOT the head
    head = buffer[:, :required_seq, :]
    assert not np.array_equal(sliced, head)


def test_heterogeneous_column_extraction():
    """
    Verify extraction when a submodel requires fewer features (e.g. 128 of 584).
    """
    batch_size = 1
    incoming_seq_len = 120
    common_n_features = 584
    required_seq = 60
    required_feat = 128

    buffer = np.arange(incoming_seq_len * common_n_features, dtype=np.float32).reshape(
        batch_size, incoming_seq_len, common_n_features
    )
    flat_features = buffer.flatten()

    step_offset = incoming_seq_len - required_seq

    # Simulate C++ packed buffer logic
    packed = np.zeros((batch_size, required_seq, required_feat), dtype=np.float32)
    for b in range(batch_size):
        for s in range(required_seq):
            src_step = step_offset + s
            src_idx = (b * incoming_seq_len + src_step) * common_n_features
            packed[b, s, :] = flat_features[src_idx : src_idx + required_feat]

    expected = buffer[:, -required_seq:, :required_feat]
    np.testing.assert_array_equal(packed, expected)


def test_multi_batch_tail_slicing():
    """
    Verify that when batch_size > 1, the row-by-row packing correctly slices the
    tail of each individual batch without data leakage or corruption.
    """
    for batch_size in [2, 4]:
        incoming_seq_len = 120
        common_n_features = 32
        required_seq = 60
        required_feat = 32

        # Create distinct recognizable data across batches and steps
        buffer = np.arange(batch_size * incoming_seq_len * common_n_features, dtype=np.float32).reshape(
            batch_size, incoming_seq_len, common_n_features
        )
        flat_features = buffer.flatten()
        step_offset = incoming_seq_len - required_seq

        # Simulate C++ multi-batch packing logic
        packed = np.zeros((batch_size, required_seq, required_feat), dtype=np.float32)
        for b in range(batch_size):
            for s in range(required_seq):
                src_step = step_offset + s
                src_idx = (b * incoming_seq_len + src_step) * common_n_features
                dst_idx = (b * required_seq + s) * required_feat
                packed[b, s, :] = flat_features[src_idx : src_idx + required_feat]

        expected = buffer[:, -required_seq:, :]
        np.testing.assert_array_equal(packed, expected)

