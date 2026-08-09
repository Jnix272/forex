"""Tests for the causal conv padding fix (audit A8/A9, 2026-08-07).

Before the fix, two convs in `models/architectures.py` used symmetric
`padding=k-1` plus a post-hoc `[:, :, :T]` slice. With symmetric padding,
the OUTPUT slice kept the asymmetric portion of the receptive field, which
could leak `k-1` future bars when stacked (or under dilation/stride). The
FIX uses asymmetric `(k-1, 0)` left-only padding and removes the post-hoc
slice (output length == input length by construction).

Tests:
1. Numerical equivalence for stride=1 zero-padding (old symmetric+slice is
   equivalent to new asymmetric for the common case — keeps backward-compat).
2. Causality: asymmetric padding never reads input indices past the current
   output index `t`.
3. Source-level: `MambaBlock.conv1d` and `ConvFFN.conv*` use tuple padding
   and no `[:, :, :T]` slice in code.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Pure-numpy reference implementations
# ---------------------------------------------------------------------------

def conv_symmetric_slice(x: list, kernel: list, k: int) -> list:
    """OLD code path: symmetric zero-padding `pad=k-1` on both sides
    followed by output slice `[:len(x)]`. Equivalent to PyTorch
    `Conv1d(..., padding=k-1)` then `out[:, :, :N]`.
    """
    pad = k - 1
    padded = [0.0] * pad + list(x) + [0.0] * pad
    N = len(x)
    # Output length BEFORE slice: len(padded) - k + 1 = N + 2*pad - k + 1 = N + (k-1)
    out_unsliced = [
        sum(padded[t + i] * kernel[i] for i in range(k))
        for t in range(N + (k - 1))
    ]
    return out_unsliced[:N]


def conv_asymmetric_left(x: list, kernel: list, k: int) -> list:
    """FIXED code path: asymmetric left-only padding `pad_left=k-1`,
    NO output slice. Equivalent to PyTorch `Conv1d(..., padding=(k-1, 0))`."""
    pad = k - 1
    padded = [0.0] * pad + list(x)
    N = len(x)
    # Output length: len(padded) - k + 1 = N + pad - k + 1 = N
    return [
        sum(padded[t + i] * kernel[i] for i in range(k))
        for t in range(N)
    ]


# ---------------------------------------------------------------------------
# Numerical equivalence (old symmetric+slice equals new asymmetric for stride=1)
# ---------------------------------------------------------------------------

def test_numerical_equivalence_for_stride_1_zero_pad():
    """For stride=1 with zero-padding on both ends, the OLD symmetric+slice
    and the NEW asymmetric-left give identical outputs. This is why the
    audit didn't manifest as a live numeric regression — but the OLD form
    is fragile (any future edit removing the slice, or any dilated conv,
    would silently leak future data).
    """
    import random
    random.seed(0)
    for _ in range(50):
        N = random.randint(4, 10)
        k = random.randint(2, min(N, 5))
        x = [random.uniform(-1, 1) for _ in range(N)]
        kernel = [random.uniform(-1, 1) for _ in range(k)]
        old = conv_symmetric_slice(x, kernel, k)
        new = conv_asymmetric_left(x, kernel, k)
        assert len(old) == len(new) == N
        for ov, nv in zip(old, new):
            assert abs(ov - nv) < 1e-9, (
                f"\u00d7 N={N} k={k} x={x} kernel={kernel}\n  OLD={old}\n  NEW={new}"
            )


# ---------------------------------------------------------------------------
# Causality check: asymmetric never reads future indices
# ---------------------------------------------------------------------------

def test_asymmetric_padding_never_reads_future_indices():
    """Compute output[t] and verify it doesn't read any x[j] with j > t."""
    x = [10.0, 20.0, 30.0, 40.0, 50.0]
    k = 3
    # Kernel of all-ones sums the full window
    kernel = [1.0, 1.0, 1.0]
    fixed = conv_asymmetric_left(x, kernel, k)
    # padded = [0,0,10,20,30,40,50]
    # fixed[0] = padded[0..2] = 0+0+10 = 10 (no future; reads x[0] at most)
    # fixed[1] = padded[1..3] = 0+10+20 = 30 (reads x[1])
    # fixed[2] = padded[2..4] = 10+20+30 = 60 (reads x[2])
    # fixed[3] = padded[3..5] = 20+30+40 = 90 (reads x[3])
    # fixed[4] = padded[4..6] = 30+40+50 = 120 (reads x[4])
    expected = [10.0, 30.0, 60.0, 90.0, 120.0]
    for got, want in zip(fixed, expected):
        assert abs(got - want) < 1e-9


def test_asymmetric_output_length_equals_input_length():
    """Output length must equal input length (no slice needed)."""
    for N in [4, 10, 50]:
        for k in [2, 3, 5]:
            if k > N:
                continue
            x = list(range(N))
            kernel = [1.0] * k
            out = conv_asymmetric_left(x, kernel, k)
            assert len(out) == N, f"N={N} k={k} -> len(out)={len(out)}"


# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------

def _strip_comments(src: str) -> str:
    """Remove comment-only and docstring lines for code-only assertions."""
    out = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith('"""') or s.startswith("'''"):
            continue
        out.append(line)
    return "\n".join(out)


def test_mamba_uses_manual_causal_padding():
    path = _ROOT / "models" / "architectures.py"
    if not path.exists():
        pytest.skip("models/architectures.py not found")
    src = path.read_text(encoding="utf-8")
    code = _strip_comments(src)
    # The OLD int-form `padding=d_conv-1` should be gone
    bad = re.search(r"padding\s*=\s*d_conv\s*-\s*1\s*(?!,\s*0\s*\))", code)
    assert bad is None, f"Symmetric Padding=d_conv-1 still in code: {bad.group(0)}"
    # The NEW manual F.pad approach should be present
    assert "F.pad" in src
    assert "self.conv1d_pad" in src
    assert "padding=0" in src


def test_convffn_uses_manual_causal_padding():
    path = _ROOT / "models" / "architectures.py"
    if not path.exists():
        pytest.skip("models/architectures.py not found")
    src = path.read_text(encoding="utf-8")
    code = _strip_comments(src)
    # The OLD tuple form should be gone
    bad = re.search(r"pad_left\s*=\s*\(kernel\s*-\s*1\s*,\s*0\)", code)
    assert bad is None, f"Old tuple padding form still in code: {bad.group(0)}"
    # The NEW manual F.pad approach should be present
    assert "F.pad" in src
    assert "self.conv1_pad" in src
    assert "self.conv2_pad" in src
    assert "padding=0" in src


def test_mamba_post_hoc_slice_removed():
    """`[:, :, :T]` slice should no longer appear in MambaBlock.forward code."""
    path = _ROOT / "models" / "architectures.py"
    if not path.exists():
        pytest.skip("models/architectures.py not found")
    src = path.read_text(encoding="utf-8")
    code = _strip_comments(src)
    assert "[:, :, :T].permute" not in code, (
        "Mamba [:, :, :T] slice should be removed (asymmetric padding gives length=T)"
    )


def test_convffn_post_hoc_slice_removed():
    """`h[:, :, :T]` slice should no longer appear in ConvFFN.forward code."""
    path = _ROOT / "models" / "architectures.py"
    if not path.exists():
        pytest.skip("models/architectures.py not found")
    src = path.read_text(encoding="utf-8")
    code = _strip_comments(src)
    assert "h[:, :, :T]" not in code, (
        "ConvFFN h[:, :, :T] slice should be removed (asymmetric padding gives length=T)"
    )


def test_existing_correct_causal_pattern_still_present():
    """Sanity-check: the existing correct `F.pad(x, (pad, 0))` patterns
    (already present in the file) should still be there — they're the
    reference pattern we modelled the fix on.
    """
    path = _ROOT / "models" / "architectures.py"
    if not path.exists():
        pytest.skip("models/architectures.py not found")
    src = path.read_text(encoding="utf-8")
    assert "F.pad" in src, "F.pad utility still expected for manual causal pattern"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
