"""
Model diagnostics for live-use promotion.

These helpers intentionally avoid torch imports so they can run in reporting,
promotion, and CI contexts. Inputs can be Python lists or array-like objects.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ARCHITECTURE_PRIORITY: list[dict[str, Any]] = [
    {
        "rank": 1,
        "model": "haelt",
        "role": "primary",
        "reason": "Best default production candidate; balances temporal context and confirmation.",
    },
    {
        "rank": 2,
        "model": "mamba",
        "role": "low_latency",
        "reason": "Promote when latency and turnover are the binding constraints.",
    },
    {
        "rank": 3,
        "model": "expert",
        "role": "regime_robustness",
        "reason": "Use as a robustness candidate across market regimes.",
    },
    {
        "rank": 4,
        "model": "ensemble",
        "role": "challenger",
        "reason": "Deploy only when it beats the best base model after costs and disagreement gating.",
    },
    {
        "rank": 5,
        "model": "tft",
        "role": "context_research",
        "reason": "Keep as context-heavy challenger until live-use metrics justify promotion.",
    },
    {
        "rank": 6,
        "model": "transformer",
        "role": "context_research",
        "reason": "Useful comparison model; promote only if calibration and net returns lead.",
    },
    {
        "rank": 7,
        "model": "gnn",
        "role": "cross_asset_research",
        "reason": "Promote only after cross-asset leakage and ablation checks pass.",
    },
]


DEFAULT_ABLATION_PLAN: list[dict[str, Any]] = [
    {"name": "full_features", "drop_groups": [], "purpose": "Reference run."},
    {"name": "no_sentiment", "drop_groups": ["sentiment", "news"], "purpose": "Verify news/sentiment edge."},
    {"name": "no_cross_asset", "drop_groups": ["cross_asset", "intermarket"], "purpose": "Verify external asset edge."},
    {"name": "no_orderbook_proxy", "drop_groups": ["orderbook", "microstructure"], "purpose": "Verify synthetic/orderbook proxy value."},
    {"name": "no_macro", "drop_groups": ["macro", "calendar", "fred"], "purpose": "Verify macro feature value."},
    {"name": "price_only", "keep_groups": ["price", "returns", "volatility"], "purpose": "Baseline against simple price dynamics."},
]


def _as_list(values: Iterable[Any]) -> list[Any]:
    if hasattr(values, "tolist"):
        return list(values.tolist())
    if hasattr(values, "to_list"):
        return values.to_list()
    return list(values)


def _clip_prob(p: float, eps: float = 1e-12) -> float:
    return min(max(float(p), eps), 1.0 - eps)


def _softmax_row(row: Sequence[float]) -> list[float]:
    vals = [float(x) for x in row]
    if not vals:
        return []
    m = max(vals)
    exps = [math.exp(x - m) for x in vals]
    s = sum(exps)
    return [x / s for x in exps] if s > 0 else [1.0 / len(vals)] * len(vals)


def as_probabilities(predictions: Iterable[Any]) -> list[list[float]]:
    """
    Convert class probabilities or logits into normalized probability rows.

    One-dimensional inputs are treated as confidence for class 1 in a binary
    problem. Two-dimensional rows are normalized if they do not already sum to 1.
    """
    rows = _as_list(predictions)
    if not rows:
        return []
    if not isinstance(rows[0], (list, tuple)) and not hasattr(rows[0], "tolist"):
        out = []
        for p in rows:
            q = _clip_prob(float(p))
            out.append([1.0 - q, q])
        return out

    out: list[list[float]] = []
    for row in rows:
        vals = _as_list(row)
        vals = [float(x) for x in vals]
        total = sum(vals)
        looks_like_probs = all(0.0 <= x <= 1.0 for x in vals) and abs(total - 1.0) <= 1e-4
        if looks_like_probs:
            out.append([_clip_prob(x) for x in vals])
        else:
            out.append(_softmax_row(vals))
    return out


def classification_calibration_metrics(
    y_true: Iterable[int],
    predictions: Iterable[Any],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return ECE, MCE, Brier score, NLL, accuracy, and bin-level calibration."""
    labels = [int(x) for x in _as_list(y_true)]
    probs = as_probabilities(predictions)
    if len(labels) != len(probs):
        raise ValueError(f"labels/probability length mismatch: {len(labels)} != {len(probs)}")
    if not labels:
        return {
            "n": 0,
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "brier": 0.0,
            "nll": 0.0,
            "bins": [],
        }

    n_classes = max(len(p) for p in probs)
    bins = [
        {"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0, "conf_sum": 0.0, "correct": 0}
        for i in range(n_bins)
    ]
    correct = 0
    conf_sum = 0.0
    brier_sum = 0.0
    nll_sum = 0.0

    for label, row in zip(labels, probs):
        pred = max(range(len(row)), key=lambda i: row[i])
        conf = float(row[pred])
        hit = int(pred == label)
        correct += hit
        conf_sum += conf
        idx = min(n_bins - 1, int(conf * n_bins))
        bins[idx]["n"] += 1
        bins[idx]["conf_sum"] += conf
        bins[idx]["correct"] += hit
        for k in range(n_classes):
            target = 1.0 if k == label else 0.0
            pk = row[k] if k < len(row) else 0.0
            brier_sum += (pk - target) ** 2
        p_true = row[label] if 0 <= label < len(row) else 1e-12
        nll_sum -= math.log(_clip_prob(p_true))

    ece = 0.0
    mce = 0.0
    bin_rows = []
    total_n = len(labels)
    for b in bins:
        if b["n"] == 0:
            bin_rows.append({**b, "accuracy": None, "confidence": None, "gap": None})
            continue
        acc = b["correct"] / b["n"]
        avg_conf = b["conf_sum"] / b["n"]
        gap = abs(acc - avg_conf)
        ece += (b["n"] / total_n) * gap
        mce = max(mce, gap)
        bin_rows.append({**b, "accuracy": acc, "confidence": avg_conf, "gap": gap})

    return {
        "n": total_n,
        "accuracy": correct / total_n,
        "avg_confidence": conf_sum / total_n,
        "ece": ece,
        "mce": mce,
        "brier": brier_sum / total_n,
        "nll": nll_sum / total_n,
        "bins": bin_rows,
    }


def ensemble_safety_decision(
    disagreement: float,
    low_threshold: float = 0.5,
    high_threshold: float = 1.0,
    min_multiplier: float = 0.0,
) -> dict[str, Any]:
    """Map ensemble disagreement to deterministic size multiplier and action."""
    try:
        d = float(disagreement)
    except Exception:
        return {"decision": "block", "size_multiplier": 0.0, "reason": "invalid_disagreement"}
    if not math.isfinite(d) or d < 0:
        return {"decision": "block", "size_multiplier": 0.0, "reason": "invalid_disagreement"}
    if d < low_threshold:
        return {"decision": "full_size", "size_multiplier": 1.0, "reason": "low_disagreement"}
    if d < high_threshold:
        return {"decision": "half_size", "size_multiplier": max(0.5, min_multiplier), "reason": "moderate_disagreement"}
    return {"decision": "block", "size_multiplier": 0.0, "reason": "high_disagreement"}


def live_use_score(row: Mapping[str, Any]) -> float:
    """
    Composite score for model leaderboard sorting.

    Higher is better. Missing metrics are treated conservatively but do not crash
    report generation.
    """
    sharpe = float(row.get("net_sharpe", row.get("sharpe", 0.0)) or 0.0)
    drawdown = abs(float(row.get("max_drawdown", row.get("drawdown", 0.0)) or 0.0))
    turnover = abs(float(row.get("turnover_rate", 0.0) or 0.0))
    latency = abs(float(row.get("latency_ms", row.get("avg_latency_ms", 0.0)) or 0.0))
    ece = abs(float(row.get("ece", 0.25) or 0.0))
    disagreement = abs(float(row.get("disagreement", 0.0) or 0.0))
    promoted = 1.0 if row.get("promoted") else 0.0
    return (
        3.0 * sharpe
        - 2.0 * drawdown
        - 0.02 * turnover
        - 0.001 * latency
        - 2.0 * ece
        - 0.5 * disagreement
        + promoted
    )


def model_leaderboard(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return rows sorted by live-use score with rank and architecture priority."""
    priority = {item["model"]: item["rank"] for item in ARCHITECTURE_PRIORITY}
    scored = []
    for row in rows:
        item = dict(row)
        name = str(item.get("model", item.get("model_name", "unknown"))).lower()
        item["model"] = name
        item["architecture_priority"] = priority.get(name, 999)
        item["live_use_score"] = live_use_score(item)
        scored.append(item)
    scored.sort(key=lambda x: (-x["live_use_score"], x["architecture_priority"], x["model"]))
    for i, item in enumerate(scored, start=1):
        item["rank"] = i
    return scored


def write_model_diagnostics_report(
    path: str | Path,
    leaderboard_rows: Iterable[Mapping[str, Any]],
    calibration: Mapping[str, Any] | None = None,
    ablation_plan: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a JSON report containing leaderboard, calibration, ablation, and priority."""
    report = {
        "leaderboard": model_leaderboard(leaderboard_rows),
        "calibration": dict(calibration or {}),
        "ablation_plan": list(ablation_plan or DEFAULT_ABLATION_PLAN),
        "architecture_priority": ARCHITECTURE_PRIORITY,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
