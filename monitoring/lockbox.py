"""Lockbox evaluator — hold-out final test set evaluated exactly once."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import PATHS

__all__ = ["LockboxEvaluator"]


class LockboxEvaluator:
    """
    Holds out a final test set that is NEVER touched during development.
    Only evaluated once before going live.

    Prevents: "I tried 30 model variants, picked the one with best test Sharpe"
    — which inflates test performance by 0.2-0.5 Sharpe units.
    """

    def __init__(
        self,
        lockbox_start: str = "2024-01-01",
        lockbox_end: str = "2024-12-31",
        lock_file: str | None = None,
    ):
        if lock_file is None:
            lock_file = PATHS["file_lockbox_used"]
        self.start = pd.Timestamp(lockbox_start, tz="UTC")
        self.end = pd.Timestamp(lockbox_end, tz="UTC")
        self.lock_file = Path(lock_file)
        self.audit_file = self.lock_file.with_suffix(self.lock_file.suffix + ".audit.jsonl")
        self._evaluated = self.lock_file.exists() or self.audit_file.exists()

    @property
    def is_locked(self) -> bool:
        return self._evaluated

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split into (train_val, lockbox) without looking at lockbox."""
        mask = (df.index >= self.start) & (df.index <= self.end)
        lockbox = df[mask]
        train_val = df[~mask]
        print(
            f"[Lockbox] Train/val: {len(train_val):,} rows | "
            f"Lockbox ({self.start.date()} -> {self.end.date()}): {len(lockbox):,} rows  [SEALED]"
        )
        return train_val, lockbox

    def evaluate(self, model_fn, X_lockbox: np.ndarray, y_lockbox: np.ndarray, model_name: str = "final") -> dict:
        """Evaluate on the lockbox. Can only be called ONCE."""
        if self._evaluated:
            prev = self._load_previous_result()
            print(f"[Lockbox] WARNING: Already evaluated on {prev.get('date')}! Previous: {prev}")
            return prev

        print(f"\n[Lockbox] *** FINAL EVALUATION *** | Model: {model_name} | Samples: {len(X_lockbox):,}")
        signals = model_fn(X_lockbox)
        returns = y_lockbox * np.where(signals == 0, 1, np.where(signals == 2, -1, 0))
        sr = float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 1440))
        cum = np.cumprod(1 + returns)
        mdd = float((cum / np.maximum.accumulate(cum) - 1).min())
        wr = float((returns > 0).mean())

        result = {
            "model": model_name,
            "date": datetime.now(UTC).isoformat(),
            "lockbox_start": str(self.start.date()),
            "lockbox_end": str(self.end.date()),
            "n_samples": len(X_lockbox),
            "sharpe": round(sr, 4),
            "max_drawdown": round(mdd, 4),
            "win_rate": round(wr, 4),
            "final": True,
        }
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._write_result(result)
        self._evaluated = True
        print(f"[Lockbox] Results SEALED | Sharpe={sr:.4f} | MaxDD={mdd:.2%} | WinRate={wr:.2%}")
        return result

    def _load_previous_result(self) -> dict:
        if self.audit_file.exists():
            last = ""
            with self.audit_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        last = line
            if last:
                return json.loads(last)
        if self.lock_file.exists():
            with self.lock_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"date": "unknown", "error": "lockbox already evaluated but result file is missing"}

    def _write_result(self, result: dict) -> None:
        tmp_file = self.lock_file.with_suffix(self.lock_file.suffix + ".tmp")
        with tmp_file.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        tmp_file.replace(self.lock_file)
        with self.audit_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, sort_keys=True) + "\n")
