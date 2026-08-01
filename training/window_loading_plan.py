"""Batch FinBERT prefetch and parallel window loading planner.

This module is standard-library only. It does not score sentiment or load tick
data; it validates and reports the plan used by those heavier paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class WindowLoadingConfig:
    real_data_window_days: int = 7
    window_batch_days: int = 1
    dataset_build_workers: int = 1
    parallel_window_workers: int = 1
    finbert_batch_size: int = 256
    max_safe_workers: int = 4


def unique_headlines(headlines: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for headline in headlines:
        text = str(headline).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_finbert_prefetch_batches(
    headlines: Iterable[str],
    batch_size: int = 256,
) -> List[List[str]]:
    """Return unique headline batches for upfront sentiment prefetch."""
    if batch_size <= 0:
        raise ValueError("finbert batch_size must be > 0")
    uniq = unique_headlines(headlines)
    return [uniq[i : i + batch_size] for i in range(0, len(uniq), batch_size)]


def group_date_windows(
    windows: Sequence[Any],
    window_batch_days: int = 1,
) -> List[List[Any]]:
    """Group consecutive date windows while preserving chronological order."""
    if window_batch_days <= 0:
        raise ValueError("window_batch_days must be > 0")
    return [list(windows[i : i + window_batch_days]) for i in range(0, len(windows), window_batch_days)]


def build_window_loading_report(
    windows: Sequence[Any],
    headlines: Iterable[str],
    config: WindowLoadingConfig | None = None,
) -> Dict[str, Any]:
    """Build a validation/report object for prefetch and parallel window loading."""
    cfg = config or WindowLoadingConfig()
    reasons = []
    gates: Dict[str, bool] = {}

    gates["real_data_window_days_ok"] = cfg.real_data_window_days > 0
    if not gates["real_data_window_days_ok"]:
        reasons.append("real_data_window_days must be > 0")

    gates["window_batch_days_ok"] = cfg.window_batch_days > 0
    if not gates["window_batch_days_ok"]:
        reasons.append("window_batch_days must be > 0")

    gates["dataset_build_workers_ok"] = 1 <= cfg.dataset_build_workers <= cfg.max_safe_workers
    if not gates["dataset_build_workers_ok"]:
        reasons.append(
            f"dataset_build_workers {cfg.dataset_build_workers} outside safe range 1..{cfg.max_safe_workers}"
        )

    gates["parallel_window_workers_ok"] = 1 <= cfg.parallel_window_workers <= cfg.max_safe_workers
    if not gates["parallel_window_workers_ok"]:
        reasons.append(
            f"parallel_window_workers {cfg.parallel_window_workers} outside safe range 1..{cfg.max_safe_workers}"
        )

    gates["finbert_batch_size_ok"] = cfg.finbert_batch_size > 0
    if not gates["finbert_batch_size_ok"]:
        reasons.append("finbert_batch_size must be > 0")

    prefetch_batches = (
        build_finbert_prefetch_batches(headlines, cfg.finbert_batch_size)
        if gates["finbert_batch_size_ok"]
        else []
    )
    window_groups = (
        group_date_windows(windows, cfg.window_batch_days)
        if gates["window_batch_days_ok"]
        else []
    )

    return {
        "ok": all(gates.values()),
        "gates": gates,
        "reasons": reasons,
        "config": {
            "real_data_window_days": cfg.real_data_window_days,
            "window_batch_days": cfg.window_batch_days,
            "dataset_build_workers": cfg.dataset_build_workers,
            "parallel_window_workers": cfg.parallel_window_workers,
            "finbert_batch_size": cfg.finbert_batch_size,
            "max_safe_workers": cfg.max_safe_workers,
        },
        "windows": {
            "count": len(windows),
            "group_count": len(window_groups),
            "groups": window_groups,
        },
        "finbert_prefetch": {
            "unique_headline_count": len(unique_headlines(headlines)),
            "batch_count": len(prefetch_batches),
            "batch_sizes": [len(batch) for batch in prefetch_batches],
        },
    }


def write_window_loading_report(
    windows: Sequence[Any],
    headlines: Iterable[str],
    output_path: str | Path,
    config: WindowLoadingConfig | None = None,
) -> Dict[str, Any]:
    report = build_window_loading_report(windows, headlines, config=config)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
