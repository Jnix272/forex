"""
Curriculum / FEATURE_MASK consistency audits.

Catches:
  - feature-group names missing from the active schema (silent curriculum no-ops)
  - FEATURE_MASK columns enabled but not staged in any group (always-on orphans)
  - columns listed in multiple curriculum groups (config smell)
  - required market columns for RL/backtest not covered by FEATURE_MASK
  - settings.CURRICULUM vs active YAML schedule drift (epoch_unfreeze / scalars)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

# Scalars compared between settings.CURRICULUM and YAML curriculum.
_SCHEDULE_SCALAR_KEYS: tuple[str, ...] = (
    "chunk_early_stop_patience",
    "chunk_early_stop_min_batches",
    "difficulty_spread_threshold",
    "difficulty_spread_threshold_hard",
)

# List schedules compared for structural equality (normalized).
_SCHEDULE_LIST_KEYS: tuple[str, ...] = (
    "seq_schedule",
    "difficulty_schedule",
)


# Columns required (or strongly expected) by RL cache / market-bar helpers / backtest.
REQUIRED_MARKET_FEATURE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "close": ("mid_close", "close"),
    "atr": ("atr_6", "atr_20", "atr_60", "atr"),
    "spread": ("spread_pips", "spread_avg"),
}


def _base_name(name: str) -> str:
    """Strip pair:: prefixes used in multipair schemas."""
    text = str(name or "").strip()
    if "::" in text:
        text = text.split("::")[-1]
    return text


def iter_group_features(feature_groups: dict | None) -> dict[str, list[str]]:
    """Return {group_name: [feature, ...]} for non-empty group feature lists."""
    out: dict[str, list[str]] = {}
    for g_name, g_cfg in (feature_groups or {}).items():
        if not isinstance(g_cfg, dict):
            continue
        feats = [
            str(f).strip()
            for f in (g_cfg.get("features") or [])
            if str(f).strip()
        ]
        if feats:
            out[str(g_name)] = feats
    return out


def audit_curriculum_feature_groups(
    *,
    schema: list[str] | None,
    feature_groups: dict | None,
    feature_mask: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """
    Compare curriculum feature_groups against schema / FEATURE_MASK.

    Returns a structured report with warnings lists (never raises).
    """
    groups = iter_group_features(feature_groups)
    schema_list = [str(s) for s in (schema or [])]
    schema_bases = {_base_name(s) for s in schema_list}
    schema_set = set(schema_list) | schema_bases

    mask = feature_mask or {}
    enabled_mask = {str(k) for k, v in mask.items() if v}

    membership: dict[str, list[str]] = defaultdict(list)
    missing_from_schema: dict[str, list[str]] = defaultdict(list)
    intra_group_dupes: dict[str, list[str]] = defaultdict(list)
    for g_name, feats in groups.items():
        seen_in_group: set[str] = set()
        for f_name in feats:
            if f_name in seen_in_group:
                intra_group_dupes[g_name].append(f_name)
            seen_in_group.add(f_name)
            if g_name not in membership[f_name]:
                membership[f_name].append(g_name)
            if schema_set and f_name not in schema_set and _base_name(f_name) not in schema_set:
                if f_name not in missing_from_schema[g_name]:
                    missing_from_schema[g_name].append(f_name)

    overlapping = {
        feat: groups_for
        for feat, groups_for in membership.items()
        if len(groups_for) > 1
    }

    staged = set(membership.keys())
    # Orphans: enabled in FEATURE_MASK, never referenced by any curriculum group.
    orphans = sorted(enabled_mask - staged) if enabled_mask and groups else []

    # Masked-off but still staged (won't appear in schema built from FEATURE_MASK).
    staged_but_masked_off = sorted(
        f for f in staged if f in mask and not mask.get(f, True)
    )

    warnings: list[str] = []
    if missing_from_schema:
        total_miss = sum(len(v) for v in missing_from_schema.values())
        sample = []
        for g, feats in missing_from_schema.items():
            sample.extend(f"{g}:{f}" for f in feats[:3])
            if len(sample) >= 8:
                break
        warnings.append(
            f"{total_miss} curriculum feature(s) missing from schema "
            f"(silently skipped when freezing groups); e.g. {', '.join(sample)}"
        )
    if overlapping:
        sample = [f"{f}∈{gs}" for f, gs in list(overlapping.items())[:5]]
        warnings.append(
            f"{len(overlapping)} feature(s) listed in multiple curriculum groups: "
            + "; ".join(sample)
        )
    if intra_group_dupes:
        sample = [f"{g}:{sorted(set(fs))[:3]}" for g, fs in list(intra_group_dupes.items())[:4]]
        warnings.append(
            f"Duplicate feature entries inside curriculum group(s): {'; '.join(str(s) for s in sample)}"
        )
    if orphans:
        warnings.append(
            f"{len(orphans)} FEATURE_MASK-enabled column(s) are not in any "
            f"curriculum group (always-on / never staged); e.g. {', '.join(orphans[:8])}"
        )
    if staged_but_masked_off:
        warnings.append(
            f"{len(staged_but_masked_off)} curriculum feature(s) are False in "
            f"FEATURE_MASK (will never appear in schema): {', '.join(staged_but_masked_off[:8])}"
        )

    return {
        "groups": list(groups.keys()),
        "missing_from_schema": {k: list(v) for k, v in missing_from_schema.items()},
        "overlapping": overlapping,
        "orphans_always_on": orphans,
        "staged_but_masked_off": staged_but_masked_off,
        "warnings": warnings,
    }


def audit_required_market_columns(
    *,
    feature_mask: dict[str, bool] | None = None,
    available_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Check RL/backtest market-column contract against FEATURE_MASK and/or a built frame.
    """
    mask = feature_mask or {}
    cols = {_base_name(c) for c in (available_columns or [])}
    warnings: list[str] = []
    errors: list[str] = []
    coverage: dict[str, str | None] = {}

    for role, candidates in REQUIRED_MARKET_FEATURE_CANDIDATES.items():
        hit = next((c for c in candidates if c in cols), None)
        if hit is None and cols:
            # Frame provided but none of the candidates exist.
            if role == "spread":
                warnings.append(
                    f"No {candidates} column in feature frame — "
                    "RL market cache will synthesize a default spread."
                )
            else:
                errors.append(
                    f"Required market role '{role}' missing from feature frame "
                    f"(tried {candidates})"
                )
            coverage[role] = None
            continue

        if hit is not None:
            coverage[role] = hit
            continue

        # No frame — audit FEATURE_MASK only.
        mask_hit = next((c for c in candidates if mask.get(c, False)), None)
        coverage[role] = mask_hit
        if mask_hit is None and mask:
            # mid_close/close often aren't in FEATURE_MASK (price columns stay unmasked).
            if role == "close":
                warnings.append(
                    f"FEATURE_MASK has no explicit {candidates} entry "
                    "(price columns are usually kept outside the mask — OK if build() emits them)."
                )
            elif role == "spread":
                warnings.append(
                    f"FEATURE_MASK enables none of {candidates} — "
                    "spread may be synthesized at RL cache build time."
                )
            else:
                warnings.append(
                    f"FEATURE_MASK enables none of {candidates} for market role '{role}'."
                )

    return {"coverage": coverage, "warnings": warnings, "errors": errors}


def format_audit_warnings(report: dict[str, Any], *, prefix: str = "[CurriculumAudit]") -> list[str]:
    """Turn report['warnings'] into printable log lines."""
    return [f"{prefix} {w}" for w in (report.get("warnings") or [])]


# Columns commonly present in FeatureEngineer output but outside FEATURE_MASK.
_BUILT_SCHEMA_ALLOWLIST: frozenset[str] = frozenset({
    "open", "high", "low", "close", "mid_close", "volume",
    "bid_close", "ask_close", "bid_open", "ask_open",
    "timestamp_utc", "session", "session_label", "regime", "regime_label", "regime_class",
    "no_trade_score", "latency_ms", "expected_latency_ms", "pair",
    # DST overlap aux (labeling-only; not required in X)
    "asia_london",
})

# Aux columns required for regime-conditional RL labeling (not necessarily in X).
_LABELING_AUX_CANDIDATES: dict[str, tuple[str, ...]] = {
    "session": ("session_label", "asia_london", "london_ny"),
    "regime": ("regime_class", "regime_label"),
    "latency": ("expected_latency_ms",),
    "no_trade": ("no_trade_score",),
}


def audit_built_dataset_schema(
    *,
    feature_names: list[str] | None,
    feature_groups: dict | None = None,
    feature_mask: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """
    Gate for dataset builds: built X columns must cover curriculum + market needs.

    Errors (fail closed when integrity_gate is on):
      - curriculum group features absent from the built schema
      - required market roles (close / atr) absent from the built frame

    Warnings:
      - FEATURE_MASK-enabled columns not produced by the build
      - overlapping / orphan curriculum issues
      - missing spread (may be synthesized later)
    """
    names = [str(n) for n in (feature_names or []) if str(n).strip()]
    bases = {_base_name(n) for n in names}
    name_set = set(names) | bases

    group_audit = audit_curriculum_feature_groups(
        schema=names,
        feature_groups=feature_groups,
        feature_mask=feature_mask,
    )
    mkt = audit_required_market_columns(
        feature_mask=feature_mask,
        available_columns=names,
    )

    errors: list[str] = []
    warnings: list[str] = []

    missing = group_audit.get("missing_from_schema") or {}
    if missing:
        total = sum(len(v) for v in missing.values())
        sample: list[str] = []
        for g_name, feats in missing.items():
            sample.extend(f"{g_name}:{f}" for f in feats[:3])
            if len(sample) >= 8:
                break
        errors.append(
            f"{total} curriculum feature(s) missing from built dataset schema "
            f"(freeze/unfreeze would silently no-op); e.g. {', '.join(sample)}"
        )

    # Keep non-missing curriculum warnings (overlaps, orphans, masked-off).
    for w in group_audit.get("warnings") or []:
        if "missing from schema" in w:
            continue
        warnings.append(w)

    errors.extend(mkt.get("errors") or [])
    warnings.extend(mkt.get("warnings") or [])

    # Labeling aux: regime_class should be in the built frame for regime-conditional
    # barriers; session_label may live only on bars (not in X) — warn if neither
    # regime_class nor regime_label is present.
    for role, candidates in _LABELING_AUX_CANDIDATES.items():
        if role == "session":
            continue  # attached on bars in dataset_builder, not required in X
        if not any(c in name_set for c in candidates):
            warnings.append(
                f"Labeling aux '{role}' missing from built schema "
                f"(tried {candidates}) — regime/latency labeling may degrade"
            )

    mask = feature_mask or {}
    enabled = {str(k) for k, v in mask.items() if v}
    mask_missing = sorted(f for f in enabled if f not in name_set)
    if mask_missing:
        warnings.append(
            f"{len(mask_missing)} FEATURE_MASK-enabled feature(s) absent from built "
            f"schema; e.g. {', '.join(mask_missing[:10])}"
        )

    extras = sorted(
        b for b in bases
        if b not in enabled
        and b not in _BUILT_SCHEMA_ALLOWLIST
        and not b.startswith(("embed_", "fb_", "factor_", "granger_", "leadlag_",
                              "vol_regime_", "hurst_", "sent_", "topic_", "ner_"))
    )
    if extras:
        warnings.append(
            f"{len(extras)} built column(s) not listed in FEATURE_MASK "
            f"(still stored in X); e.g. {', '.join(extras[:10])}"
        )

    return {
        "n_features": len(names),
        "missing_from_schema": {k: list(v) for k, v in missing.items()},
        "mask_enabled_missing": mask_missing,
        "extras_not_in_mask": extras,
        "market": mkt,
        "errors": errors,
        "warnings": warnings,
    }


def _norm_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _norm_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _norm_schedule_list(entries: Any) -> list[dict[str, Any]]:
    """Normalize schedule list entries for equality (ignore unknown keys order)."""
    if not isinstance(entries, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item: dict[str, Any] = {}
        if "epoch_start" in entry:
            item["epoch_start"] = _norm_int(entry.get("epoch_start"), 0)
        if "epoch_end" in entry:
            item["epoch_end"] = _norm_int(entry.get("epoch_end"), 0)
        if "seq_len" in entry:
            item["seq_len"] = _norm_int(entry.get("seq_len"), 0)
        if "max_difficulty" in entry:
            item["max_difficulty"] = _norm_int(entry.get("max_difficulty"), 0)
        out.append(item)
    return out


def _group_schedule(feature_groups: dict | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, cfg in (feature_groups or {}).items():
        if not isinstance(cfg, dict):
            continue
        out[str(name)] = {
            "always_on": _norm_bool(cfg.get("always_on", True), True),
            "epoch_unfreeze": _norm_int(cfg.get("epoch_unfreeze", 0), 0),
        }
    return out


def audit_settings_yaml_curriculum_drift(
    settings_curriculum: dict | None,
    yaml_curriculum: dict | None,
    *,
    yaml_path: str = "config/run.yaml",
) -> dict[str, Any]:
    """
    Compare settings.CURRICULUM schedule stubs against the active YAML curriculum.

    Feature name lists are YAML-authoritative and are not compared here.
    Mismatches on shared group schedules / list schedules / shared scalars are
    returned as errors (validate_run_config should fail closed).
    """
    settings = settings_curriculum if isinstance(settings_curriculum, dict) else {}
    yaml_cur = yaml_curriculum if isinstance(yaml_curriculum, dict) else {}
    errors: list[str] = []
    warnings: list[str] = []
    mismatches: list[dict[str, Any]] = []

    s_groups = _group_schedule(settings.get("feature_groups"))
    y_groups = _group_schedule(yaml_cur.get("feature_groups"))
    only_settings = sorted(set(s_groups) - set(y_groups))
    only_yaml = sorted(set(y_groups) - set(s_groups))
    shared = sorted(set(s_groups) & set(y_groups))

    for name in shared:
        s, y = s_groups[name], y_groups[name]
        if s["always_on"] != y["always_on"] or s["epoch_unfreeze"] != y["epoch_unfreeze"]:
            mismatches.append(
                {
                    "kind": "feature_group",
                    "group": name,
                    "settings": s,
                    "yaml": y,
                }
            )
            errors.append(
                f"curriculum.feature_groups.{name}: settings "
                f"always_on={s['always_on']}/epoch_unfreeze={s['epoch_unfreeze']} "
                f"!= {yaml_path} always_on={y['always_on']}/epoch_unfreeze={y['epoch_unfreeze']}"
            )

    if only_settings:
        warnings.append(
            f"settings.CURRICULUM feature_groups not in {yaml_path}: {', '.join(only_settings)} "
            "(dead when YAML curriculum is loaded)"
        )
    if only_yaml:
        warnings.append(
            f"{yaml_path} feature_groups missing from settings.CURRICULUM stubs: "
            f"{', '.join(only_yaml)} (settings fallback will not freeze these groups)"
        )

    for key in _SCHEDULE_LIST_KEYS:
        s_list = _norm_schedule_list(settings.get(key))
        y_list = _norm_schedule_list(yaml_cur.get(key))
        if s_list != y_list:
            mismatches.append(
                {"kind": "schedule_list", "key": key, "settings": s_list, "yaml": y_list}
            )
            errors.append(
                f"curriculum.{key}: settings != {yaml_path} "
                f"(settings={s_list!r}, yaml={y_list!r})"
            )

    for key in _SCHEDULE_SCALAR_KEYS:
        s_has = key in settings
        y_has = key in yaml_cur
        if not s_has and not y_has:
            continue
        if s_has and not y_has:
            warnings.append(
                f"curriculum.{key} present in settings ({settings.get(key)!r}) "
                f"but missing from {yaml_path}"
            )
            continue
        if y_has and not s_has:
            warnings.append(
                f"curriculum.{key} present in {yaml_path} ({yaml_cur.get(key)!r}) "
                f"but missing from settings.CURRICULUM"
            )
            continue
        s_val, y_val = settings.get(key), yaml_cur.get(key)
        try:
            same = float(s_val) == float(y_val)
        except (TypeError, ValueError):
            same = s_val == y_val
        if not same:
            mismatches.append(
                {"kind": "scalar", "key": key, "settings": s_val, "yaml": y_val}
            )
            errors.append(
                f"curriculum.{key}: settings={s_val!r} != {yaml_path}={y_val!r}"
            )

    return {
        "shared_groups": shared,
        "only_settings_groups": only_settings,
        "only_yaml_groups": only_yaml,
        "mismatches": mismatches,
        "errors": errors,
        "warnings": warnings,
    }
