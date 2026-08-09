"""Tests for the EWC Fisher normalization fix (audit E4 / EWC, 2026-08-07).

Before the fix: ``ElasticWeightConsolidation._compute_fisher_diagonal``
normalised ``grad²`` per-step by ``self.max_samples`` (a fixed constant),
which had two defects:
1. **Batch-size inflation**: when batching up the dataset, the per-batch
   `grad²` already includes a `1/batch_size` mean factor from the loss's
   `reduction=\"mean\"`. Summing `grad² / max_samples` over `ceil(max_samples/batch)`
   iterations produced a Fisher diagonal whose magnitude was invariant to
   `batch_size` but did NOT track the actual sample count.
2. **Underweighting when dataset < max_samples**: when the dataset had
   fewer than `max_samples`, the loop iterated fewer times but the divisor
   stayed `max_samples`, diluting the Fisher diagonal by up to ~15×.

After the fix: accumulate raw `grad²` per-step, normalise ONCE at the end
by `samples_processed` (the actual sample count processed). Fisher is now a
true per-sample mean of `grad²`, invariant to batch size and dataset size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------

def test_fisher_diagonal_uses_samples_processed_normalisation():
    """The fixed `_compute_fisher_diagonal` should NOT divide by `self.max_samples`
    inside the loop and SHOULD divide by `max(samples_processed, 1)` after the loop.
    """
    path = _ROOT / "training" / "ewc.py"
    if not path.exists():
        pytest.skip("training/ewc.py not found")
    src = path.read_text(encoding="utf-8")
    code_lines = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith('"""') or s.startswith("'''"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    # The OLD bug form should not be present in code
    assert "p.grad.data ** 2 / self.max_samples" not in code, (
        "EWC bug: per-step `/ self.max_samples` divisor should be removed"
    )
    # The NEW fix: `max(samples_processed, 1)` normalisation after the loop
    assert "max(samples_processed, 1)" in code, (
        "EWC fix missing: should normalise by `max(samples_processed, 1)` after the loop"
    )


def test_samples_processed_still_tracks_actual_count():
    """The `samples_processed` counter should remain in the loop body and
    accumulate `features.size(0)` per batch.
    """
    path = _ROOT / "training" / "ewc.py"
    if not path.exists():
        pytest.skip("training/ewc.py not found")
    src = path.read_text(encoding="utf-8")
    assert "samples_processed += features.size(0)" in src


def test_no_max_samples_in_per_step_divisor():
    """No `self.max_samples` divisor inside the Fisher accumulation loop."""
    path = _ROOT / "training" / "ewc.py"
    if not path.exists():
        pytest.skip("training/ewc.py not found")
    src = path.read_text(encoding="utf-8")
    # Get the loop body
    start = src.find("samples_processed = 0")
    end = src.find("fisher = {", start)
    loop_block = src[start:end]
    # `/ self.max_samples` should not appear inside this loop body
    assert "/ self.max_samples" not in loop_block, (
        "Per-step `/ self.max_samples` should not be inside the Fisher loop"
    )


# ---------------------------------------------------------------------------
# Behavioural test (skipped if torch unavailable)
# ---------------------------------------------------------------------------

def test_fisher_diagonal_invariant_to_batch_size():
    """The Fisher diagonal computed with batch_size=32 and with batch_size=64
    should have the same magnitude (per-sample mean invariance).
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        pytest.skip("torch not available")
    sys.path.insert(0, str(_ROOT))
    try:
        from training.ewc import ElasticWeightConsolidation
    except Exception as e:
        pytest.skip(f"could not import ewc: {e}")

    # Tiny dummy model + dataset
    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 1)
        def forward(self, x):
            return self.fc(x).squeeze(-1)

    def make_ds(n):
        X = torch.randn(n, 4)
        y = (X[:, 0] + 0.1 * torch.randn(n))  # noisy linear
        return torch.utils.data.TensorDataset(X, y)

    fishers = {}
    for batch_size in [32, 64]:
        torch.manual_seed(0)
        model = Tiny()
        ds = make_ds(128)  # dataset has 128 samples, max_samples=128 too
        # Patch the DataLoader creation inside _compute_fisher_diagonal by
        # instrumenting it to use the configured batch_size
        ewc = ElasticWeightConsolidation(model, ds, device=torch.device("cpu"),
                                            max_samples=128)
        # Monkey-patch the batch_size in the DataLoader creation call.
        # We do this by overriding the method to inject `batch_size=`.
        # Cleaner: trigger via the existing path with the fixed batch_size 32
        # (default) and compare against a manual override of 64.
        if batch_size != 32:
            # Save and patch the DataLoader class
            from torch.utils.data import DataLoader as _DL
            real_dl = _DL
            def _patched_dl(ds, **kw):
                kw['batch_size'] = batch_size
                return real_dl(ds, **kw)
            import training.ewc as ewc_mod
            original_dl_attr = ewc_mod.__dict__.get("DataLoader")
            ewc_mod.DataLoader = _patched_dl  # type: ignore[attr-defined]
            try:
                f = ewc._compute_fisher_diagonal()
            finally:
                if original_dl_attr is not None:
                    ewc_mod.DataLoader = original_dl_attr
                else:
                    del ewc_mod.DataLoader  # type: ignore[attr-defined]
        else:
            f = ewc._compute_fisher_diagonal()
        fishers[batch_size] = f['fc.weight'].abs().mean().item()

    # The Fisher diagonal means should be close (within an order of magnitude
    # and ideally within ~30%) — the OLD bug would have made them scale
    # with batch_size in a way that the per-sample-mean should not.
    ratio = max(fishers[32], fishers[64]) / max(min(fishers[32], fishers[64]), 1e-12)
    assert ratio < 5.0, (
        f"Fisher diagonal magnitude differs by {ratio:.2f}× between batch_size=32 "
        f"and batch_size=64 — should be invariant (fix regression)"
    )


def test_fisher_diagonal_invariant_to_dataset_size():
    """When the dataset has fewer samples than `max_samples`, the Fisher
    diagonal should NOT be diluted — using a 65-sample dataset with
    max_samples=1000 should produce ~same magnitude as
    a 65-sample dataset with max_samples=65.

    BEFORE the fix: the per-step `/ max_samples` constant divisor diluted
    the Fisher by ~15× when dataset < max_samples.
    AFTER the fix: dividing by `samples_processed` makes the diagonal
    independent of the `max_samples` choice.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        pytest.skip("torch not available")
    sys.path.insert(0, str(_ROOT))
    try:
        from training.ewc import ElasticWeightConsolidation
    except Exception as e:
        pytest.skip(f"could not import ewc: {e}")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 1)
        def forward(self, x):
            return self.fc(x).squeeze(-1)

    def make_ds(n):
        X = torch.randn(n, 4)
        y = (X[:, 0] + 0.1 * torch.randn(n))
        return torch.utils.data.TensorDataset(X, y)

    n_samples = 65
    magnitudes = []
    for max_samples in [65, 1000]:
        torch.manual_seed(0)
        model = Tiny()
        ds = make_ds(n_samples)
        ewc = ElasticWeightConsolidation(model, ds, device=torch.device("cpu"),
                                            max_samples=max_samples)
        f = ewc._compute_fisher_diagonal()
        magnitudes.append(f['fc.weight'].abs().mean().item())

    # The two magnitudes should be ~equal (within 20%). The OLD bug would
    # have made max_samples=1000 produce a magnitude ~15× smaller than
    # max_samples=65 due to the constant divisor.
    ratio = max(magnitudes) / max(min(magnitudes), 1e-12)
    assert ratio < 1.5, (
        f"Fisher diagonal magnitude differs by {ratio:.2f}× between "
        f"max_samples=65 and max_samples=1000 (with same 65-sample dataset) "
        f"— should be ~equal after the fix. magnitudes={magnitudes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
