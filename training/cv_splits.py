"""Train/val split helpers (embargo, walk-forward, CV strategies).\n\nSee docs/CONTINUE.md."""
from __future__ import annotations

import numpy as np

from config.settings import LABELING, TRAINING

_HOST = None
_BOUND = False
_HOST_DEPS = (

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

# -----------------------------------------------------------------------------
# SPLITS + LABEL UTILITIES
# -----------------------------------------------------------------------------

def _embargo_bars(args) -> int:
    """A-H3: embargo gap (in samples) that must separate train from val so a
    training sample's forward-looking label cannot peek into the validation set.

    Dynamic floor is always ``seq_len + lookahead_bars + execution_delay_bars``.
    If ``validation.embargo_bars`` is set, use ``max(yaml, dynamic)`` so a short
    static YAML value cannot under-gap label horizons.
    """
    seq_len   = int(getattr(args, "seq_len", 60) or 60)
    lookahead = int(getattr(args, "lookahead_bars", LABELING.get("lookahead_bars", 15)))
    delay     = int(getattr(args, "execution_delay_bars", 1) or 0)
    dynamic = max(1, seq_len + lookahead + delay)
    cfg_embargo = getattr(args, "validation_embargo_bars", None)
    if cfg_embargo is not None:
        return max(dynamic, int(cfg_embargo))
    return dynamic


def _purge_bars(args) -> int:
    """Purge zone: training samples within this window of validation are dropped
    to prevent feature overlap (rolling windows extending into val period)."""
    cfg_purge = getattr(args, "validation_purge_bars", None)
    if cfg_purge is not None:
        return max(0, int(cfg_purge))
    # Default: use seq_len as purge if not configured
    seq_len = int(getattr(args, "seq_len", 60) or 60)
    return max(0, seq_len)


def _validation_method(args) -> str:
    """Read validation.method from config (default: purged_embargo)."""
    method = getattr(args, "validation_method", None)
    if method:
        return str(method).lower()
    return "purged_embargo"


def embargo_purge_from_config(cfg: dict | None = None) -> tuple[int, int, str]:
    """Resolve (embargo, purge, method) from a YAML dict for tabular baselines."""
    from types import SimpleNamespace

    cfg = cfg or {}
    training = cfg.get("training") or {}
    labeling = cfg.get("labeling") or {}
    strategy = cfg.get("strategy") or {}
    validation = cfg.get("validation") or {}
    execution = cfg.get("execution") or {}
    ns = SimpleNamespace(
        seq_len=int(training.get("seq_len", TRAINING.get("seq_len", 80)) or 80),
        lookahead_bars=int(
            strategy.get(
                "lookahead_bars",
                labeling.get("lookahead_bars", LABELING.get("lookahead_bars", 15)),
            )
            or 15
        ),
        execution_delay_bars=int(
            execution.get("delay_bars", execution.get("execution_delay_bars", 1)) or 0
        ),
        validation_embargo_bars=validation.get("embargo_bars"),
        validation_purge_bars=validation.get("purge_bars"),
        validation_method=validation.get("method", "purged_embargo"),
    )
    return _embargo_bars(ns), _purge_bars(ns), _validation_method(ns)


def _embargo_split(n_samples: int, val_split: float, embargo: int, purge: int = 0,
                   method: str = "purged_embargo") -> tuple[np.ndarray, np.ndarray]:
    """A-H3: chronological train/val split with embargo and optional purge gap.

    Val is the most-recent `val_split` fraction; the `embargo` samples immediately
    before the val window are DROPPED from train to prevent label leakage.
    If method == "purged_embargo", an additional `purge` samples are dropped to
    prevent feature-window overlap (rolling features extending into val period).
    """
    _ensure_bound()
    val_split = min(max(float(val_split), 0.0), 0.9)
    val_n     = int(n_samples * val_split)
    val_start = max(0, n_samples - val_n)
    if method == "purged_embargo":
        train_end = max(0, val_start - int(embargo) - int(purge))
    else:
        train_end = max(0, val_start - int(embargo))
    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx   = np.arange(val_start, n_samples, dtype=np.int64)
    return train_idx, val_idx


def _three_way_split(n_samples: int, val_split: float, tune_split: float,
                     embargo: int, purge: int = 0,
                     method: str = "purged_embargo") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-way chronological split: train | embargo | val (early stop) | embargo | tune_eval.

    Prevents data leakage between:
    - val set (used ONLY for early stopping)
    - tune_eval set (used ONLY for auto-tune hyperparameter decisions)

    Returns (train_idx, val_idx, tune_idx).
    """
    _ensure_bound()
    val_split = min(max(float(val_split), 0.0), 0.5)
    tune_split = min(max(float(tune_split), 0.0), 0.3)

    tune_n = int(n_samples * tune_split)
    val_n = int(n_samples * val_split)

    # tune_eval is the most recent chunk
    tune_start = max(0, n_samples - tune_n)
    # val is the chunk before tune_eval (with embargo between)
    val_end = max(0, tune_start - int(embargo))
    val_start = max(0, val_end - val_n)
    # train is everything before val (with embargo+purge between)
    if method == "purged_embargo":
        train_end = max(0, val_start - int(embargo) - int(purge))
    else:
        train_end = max(0, val_start - int(embargo))

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(val_start, val_end, dtype=np.int64)
    tune_idx = np.arange(tune_start, n_samples, dtype=np.int64)
    return train_idx, val_idx, tune_idx


def walk_forward_splits(n_samples: int, n_folds: int, embargo: int, purge: int = 0,
                        method: str = "purged_embargo") -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward: each fold trains on [0, val_start - embargo - purge),
    validates on [val_start, val_end). Prevents overlap leakage via embargo + purge.
    """
    _ensure_bound()
    if n_samples < max(embargo + purge + n_folds * 2, 500):
        # A-H3: small-data fallback must still embargo+purge the train/val boundary
        return [_embargo_split(n_samples, float(TRAINING.get("val_split", 0.2)), embargo, purge, method)]
    edges = np.linspace(0, n_samples, n_folds + 2, dtype=np.int64)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        va, vb = int(edges[k + 1]), int(edges[k + 2])
        if method == "purged_embargo":
            train_end = max(0, va - int(embargo) - int(purge))
        else:
            train_end = max(0, va - int(embargo))
        tr = np.arange(0, train_end, dtype=np.int64)
        va_idx = np.arange(va, vb, dtype=np.int64)
        if len(tr) < 100 or len(va_idx) < 10:
            continue
        out.append((tr, va_idx))
    if not out:
        # A-H3: embargoed+purge fallback (was a plain 80/20 split with no embargo).
        return [_embargo_split(n_samples, float(TRAINING.get("val_split", 0.2)), embargo, purge, method)]
    return out


def _build_cv_splits(model_args, n_samples: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """
    Build CV splits for supervised training (Improvement #11 wiring).

    ``--cv-strategy``:
      legacy       -> original walk_forward_splits (default, unchanged behavior)
      walk_forward -> validation.cv.WalkForwardCV (purged/embargoed expanding window)
      comb         -> validation.cv.CombCV (combinatorial purged CV)
      online       -> validation.cv.OnlineCV (rolling window)

    Returns (splits, strategy_label) where each split is (train_idx, val_idx).
    Any failure falls back to the legacy walk-forward splits.
    """
    _ensure_bound()
    strategy = str(getattr(model_args, "cv_strategy", "legacy") or "legacy").lower()
    _embargo = _embargo_bars(model_args)
    _purge = _purge_bars(model_args)
    _method = _validation_method(model_args)
    _n_folds = max(1, int(getattr(model_args, "walk_forward_folds", 1) or 1))

    def _legacy():
        return walk_forward_splits(n_samples, _n_folds, _embargo, _purge, _method)

    if strategy == "legacy":
        return _legacy(), "legacy"

    try:
        from validation.cv import CombCV, OnlineCV, WalkForwardCV
        X = np.zeros((max(n_samples, 1), 1))
        if strategy == "walk_forward":
            cv = WalkForwardCV(
                n_splits=_n_folds,
                initial_train_size=0.6,
                purge=_purge,
                embargo=_embargo,
            )
        elif strategy == "comb":
            n_groups = max(4, min(int(getattr(model_args, "cv_n_groups", 10)), max(4, n_samples // 500)))
            cv = CombCV(n_groups=n_groups, test_groups=2, purge=_purge, embargo=_embargo)
        elif strategy == "online":
            cv = OnlineCV(
                initial_train=0.6,
                window=max(50, n_samples // 20),
                step=max(50, n_samples // 20),
                purge=_purge,
            )
        else:
            raise ValueError(f"Unknown cv strategy: {strategy}")
        splits = [
            (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
            for tr, va in cv.split(X)
            if len(tr) >= 100 and len(va) >= 10
        ]
        if not splits:
            raise RuntimeError(f"{strategy} produced no usable folds")
        return splits, strategy
    except Exception as exc:
        print(f"[CV] strategy={strategy} unavailable ({exc}); falling back to legacy walk-forward")
        return _legacy(), "legacy"
