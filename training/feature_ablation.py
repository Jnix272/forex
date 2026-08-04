"""Feature ablation config and mask builders.\n\nSee docs/CONTINUE.md."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

_HOST = None
_BOUND = False
_HOST_DEPS = (
    '_log_warn',
    '_log_info',
    '_load_feature_schema',
    'PATHS',
)


def bind_host(host_mod) -> None:
    global _HOST, _BOUND
    _HOST = host_mod
    g = globals()
    for name in _HOST_DEPS:
        if hasattr(host_mod, name):
            g[name] = getattr(host_mod, name)
    _BOUND = True


def _ensure_bound() -> None:
    import training.train_gpu as tg
    bind_host(tg)

def _atomic_copy(src, dst) -> None:
    """B-M2: copy `src` onto `dst` atomically (temp file in dst's dir + os.replace)
    so a concurrent reader (live inference) never sees a half-written checkpoint."""
    import tempfile
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def _csv_set(value) -> set[str]:

    if value is None:

        return set()

    if isinstance(value, str):

        return {part.strip().lower() for part in value.split(",") if part.strip()}

    if isinstance(value, (list, tuple, set)):

        return {str(part).strip().lower() for part in value if str(part).strip()}

    return set()

def _feature_base_name(name: str) -> str:

    return str(name).split("::")[-1].split(":", 1)[-1]

def _feature_ablation_config(args) -> dict:

    _ensure_bound()
    cfg = getattr(args, "feature_ablation", None)

    cfg = dict(cfg) if isinstance(cfg, dict) else {}



    cli_name = str(getattr(args, "feature_ablation_name", "") or "").strip()

    cli_drop_groups = _csv_set(getattr(args, "feature_ablation_drop_groups", ""))

    cli_keep_groups = _csv_set(getattr(args, "feature_ablation_keep_groups", ""))

    cli_drop_features = _csv_set(getattr(args, "feature_ablation_drop_features", ""))



    if cli_name:

        cfg["name"] = cli_name

        cfg["enabled"] = True

    if cli_drop_groups:

        cfg["drop_groups"] = sorted(cli_drop_groups)

        cfg["enabled"] = True

    if cli_keep_groups:

        cfg["keep_groups"] = sorted(cli_keep_groups)

        cfg["enabled"] = True

    if cli_drop_features:

        cfg["drop_features"] = sorted(cli_drop_features)

        cfg["enabled"] = True



    cfg.setdefault("enabled", False)

    cfg.setdefault("name", "full_features")

    cfg.setdefault("drop_groups", [])

    cfg.setdefault("keep_groups", [])

    cfg.setdefault("drop_features", [])

    return cfg

def _build_feature_ablation_mask(schema: list, feature_groups: dict, cfg: dict, n_features: int) -> tuple[np.ndarray | None, dict]:

    """Build a static feature mask and a JSON-ready report for feature ablation runs."""

    _ensure_bound()
    enabled = bool(cfg.get("enabled", False))

    report = {

        "enabled": enabled,

        "name": str(cfg.get("name", "full_features") or "full_features"),

        "drop_groups": sorted(_csv_set(cfg.get("drop_groups", []))),

        "keep_groups": sorted(_csv_set(cfg.get("keep_groups", []))),

        "drop_features": sorted(_csv_set(cfg.get("drop_features", []))),

        "n_features": int(n_features),

        "masked_count": 0,

        "active_count": int(n_features),

        "masked_by_group": {},

        "masked_features_sample": [],

    }

    if not enabled:

        return None, report

    if len(schema) != n_features:

        report["warning"] = f"schema length {len(schema)} != n_features {n_features}; ablation mask skipped"

        return None, report



    drop_groups = set(report["drop_groups"])

    keep_groups = set(report["keep_groups"])

    drop_features = set(report["drop_features"])

    group_feature_map: dict[str, set[str]] = {}

    for g_name, g_cfg in (feature_groups or {}).items():

        group_feature_map[str(g_name).lower()] = {

            str(f).lower() for f in (g_cfg or {}).get("features", []) if str(f).strip()

        }

    if keep_groups:

        drop_groups |= {g for g in group_feature_map if g not in keep_groups}



    mask = np.ones(n_features, dtype=np.float32)

    masked_names: list[str] = []

    for idx, raw_name in enumerate(schema):

        base = _feature_base_name(str(raw_name)).lower()

        reason_group = None

        if base in drop_features:

            reason_group = "__explicit_features__"

        else:

            for g_name in drop_groups:

                if base in group_feature_map.get(g_name, set()):

                    reason_group = g_name

                    break

        if reason_group is not None:

            mask[idx] = 0.0

            report["masked_by_group"][reason_group] = int(report["masked_by_group"].get(reason_group, 0)) + 1

            if len(masked_names) < 40:

                masked_names.append(str(raw_name))



    report["masked_count"] = int((mask == 0.0).sum())

    report["active_count"] = int((mask != 0.0).sum())

    report["masked_features_sample"] = masked_names

    return mask, report
