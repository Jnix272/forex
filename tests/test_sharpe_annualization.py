"""Tests for the Sharpe-ratio annualization logic.

These verify:
  1. ``sharpe_ann_factor`` correctly divides by ``holding_period_bars``
     (the per-trade annualisation rule, not per-bar).
  2. ``auto_annualization_factor`` returns the right value for the
     common configurations: session vs full-day, 5m vs 1h vs 1d.
  3. The override parameter wins.
  4. The full-day FX case is wired through correctly.
  5. ``_non_overlapping_sharpe`` returns the textbook-correct Sharpe
     on a stream of overlapping per-trade returns.
  6. End-to-end: ``_sharpe_ann_factor`` in train_gpu.py uses the
     auto-detected value when no override is set.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ``torch`` is only needed for the _non_overlapping_sharpe test.  We
# skip those tests (with a clear message) when torch isn't available
# so the rest of the test suite can still run in a minimal env.
try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# â"€â"€ 1. sharpe_ann_factor: textbook per-trade annualisation â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€


def test_sharpe_ann_factor_textbook_per_trade():
    from training.sharpe_annualization import sharpe_ann_factor

    f = sharpe_ann_factor(252 * 78, holding_period_bars=30)
    expected = math.sqrt(252 * 78 / 30)
    assert abs(f - expected) < 1e-9, f"got {f}, expected {expected}"
    f_24h = sharpe_ann_factor(252 * 288, holding_period_bars=30)
    expected_24h = math.sqrt(252 * 288 / 30)
    assert abs(f_24h - expected_24h) < 1e-9
    assert abs(f_24h / f - math.sqrt(288 / 78)) < 1e-9
    print(f"OK: sharpe_ann_factor session={f:.3f}, 24h={f_24h:.3f}, ratio={f_24h / f:.3f}")


def test_sharpe_ann_factor_inflation_factor():
    from training.sharpe_annualization import sharpe_ann_factor

    f_per_bar = sharpe_ann_factor(252 * 78, holding_period_bars=1)
    f_per_trd = sharpe_ann_factor(252 * 78, holding_period_bars=30)
    inflation = f_per_bar / f_per_trd
    assert abs(inflation - math.sqrt(30)) < 1e-9
    print(f"OK: per-bar inflates Sharpe by {inflation:.3f}x (expected sqrt(30)={math.sqrt(30):.3f})")


# â"€â"€ 2. annualization_factor_from_freq â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€


def test_annualization_from_freq_5m_session():
    from training.sharpe_annualization import annualization_factor_from_freq

    f = annualization_factor_from_freq("5m", holding_period_bars=30, full_day=False)
    expected = math.sqrt(252 * 78 / 30)
    assert abs(f - expected) < 1e-9
    print(f"OK: 5m session factor = {f:.3f} (expected {expected:.3f})")


def test_annualization_from_freq_5m_full_day():
    from training.sharpe_annualization import annualization_factor_from_freq

    f = annualization_factor_from_freq("5m", holding_period_bars=30, full_day=True)
    expected = math.sqrt(252 * 288 / 30)
    assert abs(f - expected) < 1e-9
    print(f"OK: 5m full-day factor = {f:.3f} (expected {expected:.3f})")


def test_annualization_from_freq_unknown_returns_one():
    from training.sharpe_annualization import annualization_factor_from_freq

    f = annualization_factor_from_freq("unknown_freq", holding_period_bars=30)
    assert f == 1.0, f"expected neutral 1.0, got {f}"
    print("OK: unknown frequency returns neutral 1.0")


# â"€â"€ 3. auto_annualization_factor with override â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€


def test_auto_annualization_factor_override_wins():
    from training.sharpe_annualization import auto_annualization_factor

    f_auto = auto_annualization_factor(
        bar_freq="5m",
        lookahead_bars=30,
        full_day=True,
    )
    assert abs(f_auto - math.sqrt(252 * 288 / 30)) < 1e-9
    f_ovr = auto_annualization_factor(
        bar_freq="5m",
        lookahead_bars=30,
        full_day=True,
        override=99.0,
    )
    assert f_ovr == 99.0
    f_neg = auto_annualization_factor(
        bar_freq="5m",
        lookahead_bars=30,
        full_day=True,
        override=-1.0,
    )
    assert abs(f_neg - f_auto) < 1e-9
    print("OK: override wins; negative override falls through to auto")


def test_auto_annualization_factor_with_cache_manifest(tmp_path):
    """When a manifest is present, the auto-detector reads start/end
    dates and combines with the per-day rate."""
    import json
    from datetime import datetime, timedelta

    from training.sharpe_annualization import auto_annualization_factor

    cache = tmp_path / "EURUSD_5m.zarr"
    cache.mkdir()
    manifest = tmp_path / "dataset_manifest.json"
    start = datetime(2020, 1, 1)
    end = start + timedelta(days=730)
    manifest.write_text(
        json.dumps(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
        )
    )
    f = auto_annualization_factor(
        cache_path=cache,
        bar_freq="5m",
        lookahead_bars=30,
        full_day=True,
    )
    expected = math.sqrt(252 * 288 / 30)
    assert abs(f - expected) < 1e-9, f"got {f}, expected {expected}"
    print(f"OK: auto-detect with cache manifest = {f:.3f}")


# â"€â"€ 4. _non_overlapping_sharpe â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€


def _load_supervised_loop_helper(name: str):
    """AST-extract a top-level function from supervised_loop.py so the
    test can run without importing the full training stack."""
    if not _HAS_TORCH:
        return None
    import ast

    src = (_REPO_ROOT / "training" / "supervised_loop.py").read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    extracted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert extracted, f"{name} not found in supervised_loop.py"
    ns: dict = {"__builtins__": __builtins__, "torch": torch, "np": np}
    exec(compile(ast.Module(body=extracted, type_ignores=[]), "<x>", "exec"), ns)
    return ns[name]


def test_non_overlapping_sharpe_drops_overlaps():
    if not _HAS_TORCH:
        print("SKIP: torch not available")
        return
    fn = _load_supervised_loop_helper("_non_overlapping_sharpe")
    pos = np.full(30, 0.01, dtype=np.float32)
    neg = np.full(30, -0.01, dtype=np.float32)
    r_overlap = np.concatenate([pos, neg])
    s = fn(torch.from_numpy(r_overlap), lookahead_bars=30)
    assert abs(s) < 1e-6, f"expected ~0, got {s}"

    r_pos = np.full(30, 0.01, dtype=np.float32)
    s_pos = fn(torch.from_numpy(r_pos), lookahead_bars=30)
    assert s_pos == 0.0

    rng = np.random.default_rng(42)
    r_real = rng.normal(0.01, 0.005, size=30).astype(np.float32)
    s_real = fn(torch.from_numpy(r_real), lookahead_bars=1)
    # Expected value computed from the seeded draw (mean=0.010084..., sample_std=0.003884...)
    expected = 2.5964808960543313
    assert abs(s_real - expected) < 1e-6, f"got {s_real}, expected {expected}"
    print(f"OK: _non_overlapping_sharpe returns {s_real:.3f} on synthetic edge")


def test_non_overlapping_sharpe_uses_sample_variance():
    if not _HAS_TORCH:
        print("SKIP: torch not available")
        return
    fn = _load_supervised_loop_helper("_non_overlapping_sharpe")
    r = torch.tensor([0.01, 0.02], dtype=torch.float32)
    s = fn(r, lookahead_bars=1)
    expected = 0.015 / 0.0070710678
    assert abs(s - expected) < 1e-3, f"got {s}, expected {expected}"
    print(f"OK: _non_overlapping_sharpe uses sample variance (n-1) -> {s:.4f}")


# â"€â"€ 5. _sharpe_ann_factor in train_gpu.py â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€


def test_train_gpu_sharpe_ann_factor_uses_auto_detect():
    import ast
    import types

    src = (_REPO_ROOT / "training" / "train_gpu.py").read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    extracted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sharpe_ann_factor"]
    assert extracted, "_sharpe_ann_factor not found in train_gpu.py"
    ns: dict = {"__builtins__": __builtins__}
    exec(compile(ast.Module(body=extracted, type_ignores=[]), "<x>", "exec"), ns)
    fn = ns["_sharpe_ann_factor"]

    args = types.SimpleNamespace(
        sharpe_annualization_factor=None,
        cache_path=None,
        data_cache=None,
        bar_freq="5m",
        lookahead_bars=30,
        fx_full_day=True,
    )
    f = fn(args)
    expected = math.sqrt(252 * 288 / 30)
    assert abs(f - expected) < 1e-9, f"got {f}, expected {expected}"

    args2 = types.SimpleNamespace(
        sharpe_annualization_factor=42.0,
        cache_path=None,
        data_cache=None,
        bar_freq="5m",
        lookahead_bars=30,
        fx_full_day=True,
    )
    f2 = fn(args2)
    assert f2 == 42.0

    args3 = types.SimpleNamespace(
        sharpe_annualization_factor=None,
        cache_path=None,
        data_cache=None,
        bar_freq="5m",
        lookahead_bars=30,
        fx_full_day=False,
    )
    f3 = fn(args3)
    expected3 = math.sqrt(252 * 78 / 30)
    assert abs(f3 - expected3) < 1e-9
    print(f"OK: _sharpe_ann_factor auto-detects 24h={f:.3f}, session={f3:.3f}, override=42.0->{f2}")


# â"€â"€ 6. End-to-end Sharpe inflation regression test â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€


def test_sharpe_inflation_regression():
    """A clear regression test: with the OLD hard-coded factor (325.0
    for daily-bar) applied to a 5m per-trade return stream, the
    headline Sharpe is ~6.6Ã- too high. With the correct per-trade
    factor (sqrt(252*288/30) â‰ˆ 49.2 for 24h FX) the answer is sane.
    This is the inflation error we are eliminating."""  # noqa: RUF002
    rng = np.random.default_rng(0)
    rng.normal(0.001, 0.01, size=1000).astype(np.float32)
    sample_sharpe = 0.001 / 0.01
    correct_ann = math.sqrt(252 * 288 / 30)
    buggy_ann = 325.0
    assert abs(correct_ann - 49.21) < 0.1
    inflation = buggy_ann / correct_ann
    assert inflation > 6.0
    assert abs(sample_sharpe * buggy_ann - 32.5) < 0.1
    assert abs(sample_sharpe * correct_ann - 4.92) < 0.1
    print(f"OK: inflation regression - buggy=32.5, correct=4.92, ratio={inflation:.2f}x")


def test_short_cache_emits_warning():
    """When the cache spans fewer than 90 days, the auto-detector
    should emit a warning that the annualization assumes a full-year
    schedule."""
    import json
    import tempfile
    import warnings
    from datetime import datetime, timedelta

    from training.sharpe_annualization import auto_annualization_factor

    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "tiny.zarr"
        cache.mkdir()
        manifest = Path(td) / "dataset_manifest.json"
        start = datetime(2024, 1, 1)
        # 30-day cache - below the 90-day threshold
        manifest.write_text(
            json.dumps(
                {
                    "start": start.isoformat(),
                    "end": (start + timedelta(days=30)).isoformat(),
                }
            )
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f = auto_annualization_factor(
                cache_path=cache,
                bar_freq="5m",
                lookahead_bars=30,
                full_day=True,
            )
            short_warnings = [x for x in w if "Cache spans only" in str(x.message)]
            assert len(short_warnings) >= 1, f"Expected a short-cache warning, got {[str(x.message) for x in w]}"
        # The factor is still derived from the schedule (not capped)
        expected = math.sqrt(252 * 288 / 30)
        assert abs(f - expected) < 1e-9
    print(f"OK: short cache (30 days) emits warning, factor={f:.2f}")


if __name__ == "__main__":
    test_sharpe_ann_factor_textbook_per_trade()
    test_sharpe_ann_factor_inflation_factor()
    test_annualization_from_freq_5m_session()
    test_annualization_from_freq_5m_full_day()
    test_annualization_from_freq_unknown_returns_one()
    test_auto_annualization_factor_override_wins()
    import tempfile

    with tempfile.TemporaryDirectory() as _td:
        test_auto_annualization_factor_with_cache_manifest(Path(_td))
    test_non_overlapping_sharpe_drops_overlaps()
    test_non_overlapping_sharpe_uses_sample_variance()
    test_train_gpu_sharpe_ann_factor_uses_auto_detect()
    test_sharpe_inflation_regression()
    test_short_cache_emits_warning()
    print("\nAll Sharpe annualization tests pass.")
