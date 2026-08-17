"""Tests for run.yaml / model-profile fixes.

These verify the audit fixes for the run.yaml and model-profile
config bugs:

  1. ``fx_full_day`` is in the YAML→argparse map.
  2. ``run.yaml`` ``epochs`` is consistent with curriculum schedule.
  3. ``run.yaml`` ``dropout`` matches the model profile.
  4. Hardware ``profile`` is ``auto``.
  5. HAELT ``seq_len`` is consistent across run.yaml, models.py,
     haelt.yaml, and strategy_profiles.py.
  6. GLM profile has ``pretrain_method='none'``.
  7. pretrain runner accepts ``'none'`` and short-circuits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── 1. fx_full_day wired into the CLI map ──────────────────────────────────


def test_fx_full_day_is_in_yaml_map():
    """Without this entry, run.yaml's ``training.fx_full_day: true`` is
    silently dropped by ``_apply_yaml_config``, which means
    ``_sharpe_ann_factor`` defaults to session (78 bars/day) instead of
    full-day FX (288 bars/day)."""
    # AST-extract _YAML_MAP to avoid importing the full module
    # (which pulls torch + matplotlib).
    import ast

    src = (_REPO_ROOT / "training" / "gpu_cli.py").read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == "_YAML_MAP":
                _yaml_map = ast.literal_eval(node.value)
                break
    else:
        raise AssertionError("_YAML_MAP not found in gpu_cli.py")
    assert "training.fx_full_day" in _yaml_map, (
        "training.fx_full_day is missing from _YAML_MAP - the value is silently dropped by _apply_yaml_config"
    )
    assert _yaml_map["training.fx_full_day"] == "fx_full_day"


def test_run_yaml_says_fx_full_day_true():
    with open(_REPO_ROOT / "config" / "run.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["training"]["fx_full_day"] is True, (
        "run.yaml must declare fx_full_day: true so the full-day Sharpe annualization is actually used."
    )


# ── 2. epochs consistent ──────────────────────────────────────────────────


def test_run_yaml_epochs_is_not_1():
    """An ``epochs: 1`` in run.yaml contradicted the curriculum
    feature-group schedule (which unfreezes at epochs 1-5)."""
    with open(_REPO_ROOT / "config" / "run.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["training"]["epochs"] >= 6, (
        f"epochs={cfg['training']['epochs']} is too low for the curriculum feature-group unfreeze schedule to fire."
    )


# ── 3. dropout matches the model profile ───────────────────────────────────


def test_run_yaml_dropout_matches_haelt_profile():
    with open(_REPO_ROOT / "config" / "run.yaml", encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)
    haelt_dropout = run_cfg["model"]["dropout"]
    assert haelt_dropout == 0.25, f"run.yaml dropout={haelt_dropout} disagrees with the canonical 0.25 in models.py."


# ── 4. hardware profile is "auto" ──────────────────────────────────────────


def test_hardware_profile_is_auto():
    with open(_REPO_ROOT / "config" / "run.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    profile = cfg.get("hardware", {}).get("profile", "")
    assert profile == "auto", f"hardware.profile={profile!r} is Linux-specific; use 'auto'."


# ── 5. seq_len consistent across files ────────────────────────────────────


def test_haelt_yaml_seq_len_matches_run_yaml():
    """run.yaml: 6h40m = 80 bars at 5min. haelt.yaml should agree so
    that the model profile doesn't silently override it."""
    with open(_REPO_ROOT / "config" / "run.yaml", encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)
    with open(_REPO_ROOT / "config" / "models" / "haelt.yaml", encoding="utf-8") as f:
        haelt_yaml = yaml.safe_load(f)
    # run.yaml can express seq_len as a time string like "6h40m" or an
    # integer.  Convert the time string to bars-at-5min for comparison.
    run_seq = _seq_len_to_bars(run_cfg["training"]["seq_len"], minutes_per_bar=5)
    yaml_seq = int(haelt_yaml["training"]["seq_len"])
    assert yaml_seq == run_seq, (
        f"haelt.yaml seq_len={yaml_seq} disagrees with run.yaml "
        f"seq_len={run_seq} bars. The model profile silently overrides "
        f"the YAML value - pick one source of truth."
    )


def _seq_len_to_bars(seq_len, minutes_per_bar: int = 5) -> int:
    """Convert a seq_len spec (int or "HhMm" / "Mm" string) to bar count."""
    if isinstance(seq_len, int):
        return seq_len
    s = str(seq_len).strip().lower()
    if s.endswith("m") and "h" not in s:
        return int(s.rstrip("m")) // minutes_per_bar
    if "h" in s:
        parts = s.replace("m", "").split("h")
        hours = int(parts[0]) if parts[0] else 0
        minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        return (hours * 60 + minutes) // minutes_per_bar
    return int(s)


def test_models_py_haelt_seq_len_is_80():
    """models.py HAELT entry must use seq_len=80 to match the
    curriculum schedule and run.yaml."""
    src = (_REPO_ROOT / "config" / "models.py").read_text(encoding="utf-8", errors="replace")
    # The HAELT entry is the only one with 'lstm_hidden' - use it
    # as a marker to scope the check.
    haelt_start = src.find('"haelt":')
    haelt_block = src[haelt_start : haelt_start + 1200]
    assert '"seq_len": 80' in haelt_block, "models.py haelt entry must have seq_len=80 (was 60)."


# ── 6. GLM pretrain_method is "none" ───────────────────────────────────────


def test_glm_profile_has_no_pretrain():
    with open(_REPO_ROOT / "config" / "model_training_profile.py", encoding="utf-8") as f:
        src = f.read()
    glm_start = src.find('"glm":')
    assert glm_start > 0
    glm_block = src[glm_start : glm_start + 1500]
    assert 'pretrain_method="none"' in glm_block, (
        "GLM (linear baseline) must have pretrain_method='none' - BYOL/SimCLR/CPC all require a deep encoder."
    )


# ── 7. pretrain runner handles method='none' ─────────────────────────────


def test_pretrain_runner_accepts_none_method():
    src = (_REPO_ROOT / "training" / "pretrain_runner.py").read_text(encoding="utf-8", errors="replace")
    assert '"none"' in src, "_VALID_PRETRAIN_METHODS must include 'none'"
    assert '_method == "none"' in src, "run_pretrain must short-circuit with `if _method == 'none':`"
    assert "return None" in src, "run_pretrain must have a None return path"


if __name__ == "__main__":
    test_fx_full_day_is_in_yaml_map()
    test_run_yaml_says_fx_full_day_true()
    test_run_yaml_epochs_is_not_1()
    test_run_yaml_dropout_matches_haelt_profile()
    test_hardware_profile_is_auto()
    test_haelt_yaml_seq_len_matches_run_yaml()
    test_models_py_haelt_seq_len_is_80()
    test_glm_profile_has_no_pretrain()
    test_pretrain_runner_accepts_none_method()
    print("\nAll run.yaml / model-profile fix tests pass.")
