"""Priority 5 model-training audit and readiness report.

The helpers here are standard-library only. They validate the artifacts that
prove a training run used a known recipe, tracked control decisions, produced a
model card, and did not end in an obvious overfit/collapse state.

``ARCHITECTURE_RECIPES`` is derived from ``config.models`` (plus tabular
baselines), not a hardcoded parallel catalogue.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.models import BENCHMARK_BASELINES, MODELS, SUPPORTED_SUPERVISED

# Loss preferences are training-policy overlays (not architecture hyperparams).
_RECIPE_LOSS_OVERLAY: dict[str, tuple[str, ...]] = {
    "haelt": ("sharpe_huber", "directional_huber"),
    "mamba": ("directional_huber", "sharpe_huber"),
    "tft": ("cross_entropy", "multitask", "directional_huber"),
    "gnn": ("sharpe_huber", "directional_huber"),
    "expert": ("directional_huber", "cross_entropy"),
    "transformer": ("sharpe_huber", "cross_entropy"),
    "xgboost": ("tabular",),
    "catboost": ("tabular",),
}

_ROLE_FROM_DECISION: dict[str, str] = {
    "flagship_production_candidate": "primary",
    "long_sequence_efficiency": "low_latency",
    "interpretability_exogenous": "context_interpretable",
    "cross_asset_structure": "cross_asset",
    "specialist_ablation": "specialist",
    "generic_long_range_baseline": "long_range_baseline",
    "non_deep_baseline": "tabular_baseline",
}


def _seq_lens_from_cfg(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    base = int(cfg.get("seq_len", 60) or 60)
    # Offer a short / base / long triad around the configured seq_len.
    short = max(1, base // 2)
    long = max(base, int(base * 1.5))
    return tuple(sorted({short, base, long}))


def _build_architecture_recipes() -> dict[str, dict[str, Any]]:
    recipes: dict[str, dict[str, Any]] = {}
    for name, cfg in MODELS.items():
        if name not in SUPPORTED_SUPERVISED and name not in MODELS:
            continue
        decision = str(cfg.get("decision_role", "unknown"))
        recipes[name] = {
            "role": _ROLE_FROM_DECISION.get(decision, decision),
            "preferred_losses": _RECIPE_LOSS_OVERLAY.get(
                name, ("directional_huber", "sharpe_huber"),
            ),
            "preferred_seq_lens": _seq_lens_from_cfg(cfg),
            "notes": str(cfg.get("default_use") or cfg.get("use_when") or ""),
            "source": "config.models.MODELS",
        }
    for name, cfg in BENCHMARK_BASELINES.items():
        decision = str(cfg.get("decision_role", "non_deep_baseline"))
        recipes[name] = {
            "role": _ROLE_FROM_DECISION.get(decision, "tabular_baseline"),
            "preferred_losses": _RECIPE_LOSS_OVERLAY.get(name, ("tabular",)),
            "preferred_seq_lens": (1, int(cfg.get("seq_len", 60) or 60)),
            "notes": str(cfg.get("default_use") or cfg.get("use_when") or ""),
            "source": "config.models.BENCHMARK_BASELINES",
        }
    # CatBoost is shelled alongside XGB but may not live in BENCHMARK_BASELINES.
    if "catboost" not in recipes:
        recipes["catboost"] = {
            "role": "tabular_baseline",
            "preferred_losses": ("tabular",),
            "preferred_seq_lens": (1, 60),
            "notes": "CatBoost tabular baseline; trained via train_catboost.py shell.",
            "source": "derived",
        }
    return recipes


ARCHITECTURE_RECIPES: dict[str, dict[str, Any]] = _build_architecture_recipes()


@dataclass(frozen=True)
class ModelTrainingAuditConfig:
    require_model_card: bool = True
    require_training_control: bool = True
    require_recipe_known: bool = True
    require_best_epoch_restored: bool = True
    max_overfit_signals: int = 2
    max_train_val_gap: float = 0.35
    min_best_val_sharpe: float | None = None
    require_pretrain_ablation_when_present: bool = True
    allowed_pretrain_verdicts: tuple[str, ...] = ("pretrain_helped", "mixed", "unknown")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _nested_get(data: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _model_name(*docs: Mapping[str, Any]) -> str:
    for doc in docs:
        for key in ("model", "model_name", "architecture"):
            value = doc.get(key)
            if value:
                return str(value).lower()
    return "unknown"


def _all_model_names(*docs: Mapping[str, Any]) -> list[str]:
    names = []
    for doc in docs:
        for key in ("model", "model_name", "architecture"):
            value = doc.get(key)
            if value:
                names.append(str(value).lower())
                break
    return names


def _control_overfit_signals(control: Mapping[str, Any]) -> list:
    for key in ("overfitting_signals_detected", "overfitting_warnings", "warnings"):
        value = control.get(key)
        if isinstance(value, list):
            return value
    return []


def validate_model_training_package(
    artifact_dir: str | Path,
    config: ModelTrainingAuditConfig | None = None,
) -> dict[str, Any]:
    """Validate Priority 5 training artifacts for one model/run directory."""
    cfg = config or ModelTrainingAuditConfig()
    root = Path(artifact_dir)
    reasons = []
    gates: dict[str, bool] = {}
    artifacts: dict[str, str] = {}

    model_card_path = _first_existing(root, ("model_card.json", f"{root.name}_model_card.json"))
    control_path = _first_existing(root, ("training_control_report.json", f"{root.name}_training_control_report.json"))
    train_summary_path = root / "train_summary.json"
    pretrain_ablation_path = root / "pretrain_ablation.json"

    paths = {
        "model_card": model_card_path,
        "training_control_report": control_path,
        "train_summary": train_summary_path if train_summary_path.exists() else None,
        "pretrain_ablation": pretrain_ablation_path if pretrain_ablation_path.exists() else None,
    }
    docs: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        artifacts[f"{name}.json"] = str(path) if path else ""
        if path is None:
            docs[name] = {}
            continue
        try:
            docs[name] = _read_json(path)
        except Exception as exc:
            docs[name] = {}
            reasons.append(f"{path.name}: invalid JSON ({exc})")

    model_card = docs["model_card"]
    control = docs["training_control_report"]
    train_summary = docs["train_summary"]
    pretrain_ablation = docs["pretrain_ablation"]

    gates["model_card_present"] = bool(model_card) or not cfg.require_model_card
    if cfg.require_model_card and not model_card:
        reasons.append("model_card.json: missing")

    gates["training_control_present"] = bool(control) or not cfg.require_training_control
    if cfg.require_training_control and not control:
        reasons.append("training_control_report.json: missing")

    model = _model_name(model_card, train_summary, control)
    model_names = _all_model_names(model_card, train_summary, control)
    unique_models = {name for name in model_names if name and name != "unknown"}
    gates["model_identity_consistent"] = len(unique_models) <= 1
    if not gates["model_identity_consistent"]:
        reasons.append(f"model identity mismatch across artifacts: {sorted(unique_models)}")

    recipe = str(
        control.get("model_recipe_used")
        or train_summary.get("recipe")
        or model_card.get("recipe")
        or model
    ).lower()
    known_recipe = model in ARCHITECTURE_RECIPES or recipe in ARCHITECTURE_RECIPES
    gates["recipe_known"] = known_recipe or not cfg.require_recipe_known
    if cfg.require_recipe_known and not known_recipe:
        reasons.append(f"recipe: unknown model/recipe '{recipe}'")

    restore_decision = bool(
        control.get("restore_decision")
        or _nested_get(train_summary, "training_control", "restore_decision")
    )
    gates["best_epoch_restored"] = restore_decision or not cfg.require_best_epoch_restored
    if cfg.require_best_epoch_restored and not restore_decision:
        reasons.append("training_control_report.json: best epoch restore decision not recorded")

    overfit_signals = _control_overfit_signals(control)
    gates["overfit_signal_count_ok"] = len(overfit_signals) <= cfg.max_overfit_signals
    if not gates["overfit_signal_count_ok"]:
        reasons.append(f"training_control_report.json: {len(overfit_signals)} overfit signals > {cfg.max_overfit_signals}")

    train_val_gap = _as_float(
        train_summary.get("train_val_gap", _nested_get(control, "final_metrics", "train_val_gap")),
        default=0.0,
    )
    gates["train_val_gap_ok"] = train_val_gap <= cfg.max_train_val_gap
    if not gates["train_val_gap_ok"]:
        reasons.append(f"train_summary.json: train_val_gap {train_val_gap:.6g} > {cfg.max_train_val_gap:.6g}")

    best_val_sharpe = _as_float(
        train_summary.get(
            "best_val_sharpe",
            train_summary.get("best_val_sharpe_proxy", _nested_get(model_card, "performance", "validation_sharpe")),
        ),
        default=0.0,
    )
    gates["best_val_sharpe_ok"] = cfg.min_best_val_sharpe is None or best_val_sharpe >= cfg.min_best_val_sharpe
    if not gates["best_val_sharpe_ok"]:
        reasons.append(f"train_summary.json: best_val_sharpe {best_val_sharpe:.6g} < {cfg.min_best_val_sharpe:.6g}")

    verdict = str(pretrain_ablation.get("verdict", pretrain_ablation.get("ablation_verdict", "missing"))).lower()
    gates["pretrain_ablation_ok"] = (
        not pretrain_ablation
        or not cfg.require_pretrain_ablation_when_present
        or verdict in cfg.allowed_pretrain_verdicts
    )
    if not gates["pretrain_ablation_ok"]:
        reasons.append(f"pretrain_ablation.json: verdict '{verdict}' is not allowed")

    memory_suggestions = control.get("memory_suggestions", [])
    gates["memory_suggestions_audited"] = isinstance(memory_suggestions, list)
    if not gates["memory_suggestions_audited"]:
        reasons.append("training_control_report.json: memory_suggestions must be a list")

    ready = all(gates.values())
    recipe_info = ARCHITECTURE_RECIPES.get(model, ARCHITECTURE_RECIPES.get(recipe, {}))
    return {
        "ready_for_validation": ready,
        "model": model,
        "recipe": recipe,
        "artifact_dir": str(root),
        "gates": gates,
        "reasons": reasons,
        "training": {
            "best_val_sharpe": best_val_sharpe,
            "train_val_gap": train_val_gap,
            "overfit_signal_count": len(overfit_signals),
            "restore_decision": restore_decision,
            "pretrain_verdict": verdict,
        },
        "recipe_info": recipe_info,
        "artifacts": artifacts,
    }


def write_model_training_audit_report(
    artifact_dir: str | Path,
    output_path: str | Path | None = None,
    config: ModelTrainingAuditConfig | None = None,
) -> dict[str, Any]:
    """Validate training artifacts and write `priority5_model_training_report.json`."""
    report = validate_model_training_package(artifact_dir, config=config)
    out = Path(output_path) if output_path else Path(artifact_dir) / "priority5_model_training_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
