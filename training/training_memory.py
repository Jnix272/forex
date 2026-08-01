"""
training/training_memory.py
============================
Persistent training memory: accumulates lessons from every run and provides
conservative hyperparameter nudges for the next run.

Saved to ``logs/training_memory.json`` by default.

Schema (all fields optional/nullable):
--------------------------------------
{
  "updated_at":              "2026-06-11T21:00:00Z",
  "total_runs":              7,
  "best_model":              "haelt",
  "best_run":                "haelt_0611_2000",
  "best_sharpe":             0.5608,
  "best_val_loss":           0.2314,
  "best_epoch":              8,
  "best_epoch_pattern":      "early_peak",   # early_peak | late_peak | plateau
  "common_failure":          "sharpe_collapse_after_lr_peak",
  "recommended_lr":          0.00005,
  "recommended_dropout":     0.30,
  "recommended_patience":    6,
  "recommended_max_epochs":  24,
  "sharpe_history":          [0.41, 0.51, 0.56],
  "failure_counts":          {"gate_failed": 3, "sharpe_collapse": 2},
  "per_model":               {"haelt": {...}, "mamba": {...}}
}
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


_DEFAULT_PATH = Path("logs/training_memory.json")

# Maximum ratio by which a single run may nudge the recommended LR
_MAX_LR_CHANGE = 0.5          # can halve or double in one step, but clamped below
_LR_FLOOR      = 1e-6
_LR_CEIL       = 1e-2
_DO_FLOOR      = 0.05
_DO_CEIL       = 0.60


class TrainingMemory:
    """Persistent cross-run training memory.

    Usage
    -----
    memory = TrainingMemory()          # load from default path
    memory.apply_to_args(args)         # nudge args at startup
    ...train...
    memory.update(run_result)          # record outcome
    memory.save()                      # persist to disk
    """

    def __init__(self, path: str | Path = _DEFAULT_PATH):
        self.path = Path(path)
        self._data: Dict[str, Any] = {}
        self.load()

    # ── persistence ─────────────────────────────────────────────────────────

    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                print(f"[TrainingMemory] Loaded from {self.path}  "
                      f"(runs={self._data.get('total_runs', 0)}, "
                      f"best_sharpe={self._data.get('best_sharpe', 'n/a')})")
            except Exception as e:
                print(f"[TrainingMemory] Could not load {self.path}: {e} — starting fresh")
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        fd, tmp = tempfile.mkstemp(
            prefix=".training_memory.", suffix=".tmp", dir=str(self.path.parent)
        )
        os.close(fd)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
            os.replace(tmp, self.path)
            print(f"[TrainingMemory] Saved -> {self.path}")
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    # ── update ───────────────────────────────────────────────────────────────

    def update(self, run_result: Dict[str, Any]) -> None:
        """Record the outcome of a completed training run.

        Parameters
        ----------
        run_result : dict with keys:
            model_name      str
            run_name        str
            best_sharpe     float | None
            best_val_loss   float | None
            best_epoch      int | None
            total_epochs    int | None
            history         dict  (keys: train_loss, val_loss, val_sharpe, ...)
            gate_result     dict  (keys: promoted, reasons, ...)
            failure_reason  str | None
        """
        model  = run_result.get("model_name", "unknown")
        run    = run_result.get("run_name",   "unknown")
        sharpe = run_result.get("best_sharpe")
        vloss  = run_result.get("best_val_loss")
        b_ep   = run_result.get("best_epoch")
        t_ep   = run_result.get("total_epochs")
        hist   = run_result.get("history") or {}
        gate   = run_result.get("gate_result") or {}
        failure = run_result.get("failure_reason", "")

        # total runs
        self._data["total_runs"] = self._data.get("total_runs", 0) + 1

        # best sharpe tracking
        if sharpe is not None:
            prev_best = self._data.get("best_sharpe")
            if prev_best is None or sharpe > prev_best:
                self._data["best_sharpe"]  = round(float(sharpe), 6)
                self._data["best_val_loss"] = round(float(vloss), 6) if vloss is not None else None
                self._data["best_model"]   = model
                self._data["best_run"]     = run
                self._data["best_epoch"]   = b_ep

            sharpe_hist: List[float] = self._data.get("sharpe_history", [])
            sharpe_hist.append(round(float(sharpe), 6))
            self._data["sharpe_history"] = sharpe_hist[-50:]   # keep last 50

        # epoch pattern
        val_sharpe_curve: List[float] = hist.get("val_sharpe", [])
        if val_sharpe_curve and b_ep is not None and t_ep and t_ep > 0:
            peak_frac = b_ep / t_ep
            if peak_frac < 0.35:
                pattern = "early_peak"
            elif peak_frac > 0.75:
                pattern = "late_peak"
            else:
                pattern = "plateau"
            self._data["best_epoch_pattern"] = pattern

        # Sharpe collapse detection
        if val_sharpe_curve and len(val_sharpe_curve) >= 4:
            peak = max(val_sharpe_curve)
            final = val_sharpe_curve[-1]
            if peak > 0 and (peak - final) / max(abs(peak), 1e-9) > 0.2:
                self._record_failure("sharpe_collapse_after_lr_peak")
                self._data["common_failure"] = "sharpe_collapse_after_lr_peak"

        # gate failures
        if not gate.get("promoted", True):
            for reason in gate.get("reasons", []):
                reason_text = str(reason).strip()
                reason_lower = reason_text.lower()
                passed_marker = "✓" in reason_text or reason_lower.endswith(":_true")
                failed_marker = "✗" in reason_text or "fail" in reason_lower or "reject" in reason_lower
                if passed_marker and not failed_marker:
                    continue
                reason_key = reason_lower.replace(" ", "_")[:40]
                self._record_failure(f"gate_{reason_key}")
            if not self._data.get("common_failure"):
                self._data["common_failure"] = "gate_failed"

        if failure:
            self._record_failure(failure)

        # update per-model block
        per_model = self._data.get("per_model", {})
        m_block = per_model.get(model, {})
        m_block["runs"] = m_block.get("runs", 0) + 1
        if sharpe is not None:
            prev = m_block.get("best_sharpe")
            if prev is None or sharpe > prev:
                m_block["best_sharpe"] = round(float(sharpe), 6)
                m_block["best_epoch"]  = b_ep
                m_block["best_run"]    = run
        per_model[model] = m_block
        self._data["per_model"] = per_model

        # update recommended hyperparams
        self._update_recommendations(run_result)

    def _record_failure(self, key: str) -> None:
        fc = self._data.get("failure_counts", {})
        fc[key] = fc.get(key, 0) + 1
        self._data["failure_counts"] = fc

    def _update_recommendations(self, run_result: Dict[str, Any]) -> None:
        """Conservative hyperparameter nudges based on run outcome."""
        hist   = run_result.get("history") or {}
        b_ep   = run_result.get("best_epoch")
        t_ep   = run_result.get("total_epochs") or 0
        gate   = run_result.get("gate_result") or {}
        args   = run_result.get("args_snapshot") or {}

        current_lr      = float(args.get("lr",      self._data.get("recommended_lr",      5e-5)))
        current_do      = float(args.get("dropout",  self._data.get("recommended_dropout", 0.25)))
        current_pat     = int(  args.get("patience", self._data.get("recommended_patience", 6)))
        current_maxep   = int(  args.get("epochs",   self._data.get("recommended_max_epochs", 24)))

        new_lr    = current_lr
        new_do    = current_do
        new_pat   = current_pat
        new_maxep = current_maxep

        # Overfitting: val_loss >> train_loss
        tl_curve = hist.get("train_loss", [])
        vl_curve = hist.get("val_loss",   [])
        if tl_curve and vl_curve:
            gen_gap = vl_curve[-1] - tl_curve[-1]
            if gen_gap > tl_curve[-1] * 0.15:
                new_do = min(_DO_CEIL, current_do + 0.03)

        # Sharpe collapse: peaked early, degraded
        vs_curve = hist.get("val_sharpe", [])
        if vs_curve and len(vs_curve) >= 4:
            peak = max(vs_curve)
            final = vs_curve[-1]
            if peak > 0 and (peak - final) / max(abs(peak), 1e-9) > 0.15:
                new_lr    = max(_LR_FLOOR, current_lr * 0.5)
                new_do    = min(_DO_CEIL, current_do + 0.05)
                new_pat   = max(3, current_pat - 1)
                new_maxep = max(12, b_ep + 4) if b_ep is not None else current_maxep

        # Early stopping: best epoch was very early
        if t_ep > 0 and b_ep is not None and b_ep < t_ep * 0.25:
            new_lr  = max(_LR_FLOOR, current_lr * 0.5)

        # Gate failure on drawdown → keep patience, nudge dropout
        reasons = gate.get("reasons", [])
        if any("drawdown" in str(r).lower() for r in reasons):
            new_do = min(_DO_CEIL, new_do + 0.03)

        # Promotion passed → slightly relax dropout (model is ok)
        if gate.get("promoted"):
            new_do = max(_DO_FLOOR, new_do - 0.01)

        self._data["recommended_lr"]          = round(float(new_lr),    8)
        self._data["recommended_dropout"]     = round(float(new_do),    4)
        self._data["recommended_patience"]    = int(new_pat)
        self._data["recommended_max_epochs"]  = int(new_maxep)

    # ── apply nudges ─────────────────────────────────────────────────────────

    def apply_to_args(self, args: Any) -> None:
        """Conservatively nudge training args based on historical lessons.

        Only adjusts if a recommendation exists *and* the current value is
        notably different (>10% for continuous params).  Never overrides
        high-risk fields (data range, label method, checkpoint dir, etc.).
        """
        if not self._data:
            return   # no history yet

        rec_lr   = self._data.get("recommended_lr")
        rec_do   = self._data.get("recommended_dropout")
        rec_pat  = self._data.get("recommended_patience")
        rec_ep   = self._data.get("recommended_max_epochs")
        pattern  = self._data.get("best_epoch_pattern")
        runs     = self._data.get("total_runs", 0)

        # Need at least 1 completed run before nudging
        if runs < 1:
            print("[TrainingMemory] Not enough runs to apply nudges (< 1).")
            return

        applied = []

        if rec_lr is not None and hasattr(args, "lr"):
            cur = float(getattr(args, "lr", rec_lr))
            if abs(cur - rec_lr) / max(abs(rec_lr), 1e-9) > 0.10:
                # Blend: move 50% toward recommendation
                blended = cur + 0.50 * (rec_lr - cur)
                blended = max(_LR_FLOOR, min(_LR_CEIL, blended))
                setattr(args, "lr", blended)
                applied.append(f"lr {cur:.2e} -> {blended:.2e} (rec={rec_lr:.2e})")

        if rec_do is not None and hasattr(args, "dropout"):
            cur = float(getattr(args, "dropout", rec_do))
            if abs(cur - rec_do) > 0.02:
                blended = cur + 0.50 * (rec_do - cur)
                blended = max(_DO_FLOOR, min(_DO_CEIL, blended))
                setattr(args, "dropout", round(blended, 4))
                applied.append(f"dropout {cur:.3f} -> {blended:.3f} (rec={rec_do:.3f})")

        if rec_pat is not None and hasattr(args, "patience"):
            cur = int(getattr(args, "patience", rec_pat))
            if cur != rec_pat:
                setattr(args, "patience", rec_pat)
                applied.append(f"patience {cur} -> {rec_pat}")

        # Only lower max epochs if pattern is early_peak (cap at best+4)
        if rec_ep is not None and hasattr(args, "epochs") and pattern == "early_peak":
            cur_ep = int(getattr(args, "epochs", rec_ep))
            if cur_ep > rec_ep:
                setattr(args, "epochs", rec_ep)
                applied.append(f"epochs {cur_ep} -> {rec_ep} (early_peak pattern)")

        if applied:
            print(f"[TrainingMemory] Applied {len(applied)} nudge(s) from {runs} historical runs:")
            for a in applied:
                print(f"  - {a}")
        else:
            print(f"[TrainingMemory] No nudges needed ({runs} historical runs).")

    def apply_to_model_args(
        self,
        model_args: Any,
        model_name: str,
        base_args: Any = None,
    ) -> None:
        """Per-architecture nudge applied to a model-local args copy.

        Mutates ``model_args`` only. ``base_args`` is left untouched so the
        global trainer config stays stable across the model queue.
        """
        _ = model_name  # reserved for future per-model memory tables
        self.apply_to_args(model_args)

    # ── convenience ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        d = self._data
        return (
            f"TrainingMemory(runs={d.get('total_runs',0)}, "
            f"best_model={d.get('best_model','?')}, "
            f"best_sharpe={d.get('best_sharpe','?')}, "
            f"pattern={d.get('best_epoch_pattern','?')}, "
            f"common_failure={d.get('common_failure','?')})"
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
