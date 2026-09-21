"""
training/continuous_learner.py
==============================
Autonomous continuous-learning scheduler.

Runs as a long-lived background process alongside the live engine.
Periodically checks whether enough new data has accumulated since the last
retrain, then:
  1. Spawns a full retrain with --live-retrain (live-data embargo enforced)
  2. Waits for the retrain process to finish
  3. Reads the promotion-gate result from the checkpoint directory
  4. If the new model passes, atomically swaps the checkpoint symlink so the
     live engine picks it up on the next bar without a restart

Usage
-----
# Standalone (background):
python -m training.continuous_learner --model haelt --pair EUR_USD

# All models:
python -m training.continuous_learner --all-models

# One-shot dry run (shows what would happen, no retrain):
python -m training.continuous_learner --dry-run

Config (all have env-var overrides):
  --min-new-bars      Minimum new bars since last retrain before triggering  [default: 2016 = 1 week of 5-min]
  --check-interval    Seconds between dataset-size checks                    [default: 3600 = 1 hour]
  --cooldown          Seconds between retrains regardless of new bars        [default: 86400 = 24 hours]
  --model             Model name (default: haelt)
  --checkpoint-dir    Checkpoint root (default: from config/settings.py)
  --data-cache        Dataset cache path (default: from config/settings.py)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

from config.settings import PATHS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[ContinuousLearner {ts}] {msg}", flush=True)


def _on_disk_n_samples(cache_path: str) -> int:
    """Return the current number of samples in the dataset cache."""
    try:
        from training.cache_integrity import _on_disk_sequence_count
        n = _on_disk_sequence_count(cache_path)
        return int(n) if n else 0
    except Exception as e:
        _log(f"WARN: could not count samples in {cache_path}: {e}")
        return 0


def _read_last_retrain_record(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_state(state_path: Path, data: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _promotion_passed(checkpoint_dir: Path, model_name: str) -> bool | None:
    """Return True/False/None (unknown) from the latest promotion gate JSON."""
    gate_paths = [
        checkpoint_dir / model_name / f"{model_name}_promotion_gate.json",
        checkpoint_dir / f"{model_name}_promotion_gate.json",
    ]
    for p in gate_paths:
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                return bool(d.get("promoted", False))
            except Exception:
                pass
    return None


def _atomic_promote(checkpoint_dir: Path, model_name: str) -> bool:
    """Swap <model>_live.pt symlink to the new best checkpoint atomically."""
    best = None
    for candidate in [
        checkpoint_dir / model_name / f"{model_name}_best.pt",
        checkpoint_dir / f"{model_name}_best.pt",
    ]:
        if candidate.exists():
            best = candidate
            break
    if best is None:
        _log(f"WARN: no best checkpoint found for {model_name} — skipping promotion")
        return False

    live_link = checkpoint_dir / f"{model_name}_live.pt"
    tmp_link = checkpoint_dir / f"{model_name}_live.pt.tmp"
    try:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        tmp_link.symlink_to(best.resolve())
        tmp_link.replace(live_link)  # atomic on POSIX; best-effort on Windows
        _log(f"Promoted {model_name}: {best} -> {live_link}")
        return True
    except Exception as e:
        _log(f"ERROR: promotion symlink failed: {e}")
        try:
            if tmp_link.exists() or tmp_link.is_symlink():
                tmp_link.unlink()
        except Exception:
            pass
        return False


def _run_retrain(
    model_name: str,
    checkpoint_dir: str,
    data_cache: str,
    extra_args: list[str],
    dry_run: bool,
) -> int:
    """Spawn training/train_gpu.py --live-retrain and wait for completion."""
    script = Path(__file__).resolve().parent / "train_gpu.py"
    cmd = [
        sys.executable, str(script),
        "--model", model_name,
        "--resume",
        "--live-retrain",
        "--checkpoint-dir", checkpoint_dir,
        "--data-cache", data_cache,
    ] + extra_args

    _log(f"Launching retrain: {' '.join(cmd)}")
    if dry_run:
        _log("DRY RUN — skipping subprocess")
        return 0

    lock = Path(checkpoint_dir) / "retrain_in_progress.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({
        "time": datetime.now(UTC).isoformat(),
        "reason": "continuous_learner",
        "model": model_name,
    }, indent=2), encoding="utf-8")

    try:
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent.parent))
        proc.wait()
        return proc.returncode
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class ContinuousLearner:
    def __init__(
        self,
        models: list[str],
        checkpoint_dir: str,
        data_cache: str,
        min_new_bars: int = 2016,
        check_interval: int = 3600,
        cooldown: int = 86400,
        dry_run: bool = False,
        extra_train_args: list[str] | None = None,
    ):
        self.models = models
        self.checkpoint_dir = Path(checkpoint_dir)
        self.data_cache = data_cache
        self.min_new_bars = min_new_bars
        self.check_interval = check_interval
        self.cooldown = cooldown
        self.dry_run = dry_run
        self.extra_train_args = extra_train_args or []
        self._running = True
        self._state_path = self.checkpoint_dir / "continuous_learner_state.json"

        signal.signal(signal.SIGINT,  self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, *_) -> None:
        _log("Shutdown signal received.")
        self._running = False

    def run(self) -> None:
        _log(f"Started. models={self.models} min_new_bars={self.min_new_bars} "
             f"check_interval={self.check_interval}s cooldown={self.cooldown}s")
        while self._running:
            self._tick()
            for _ in range(self.check_interval):
                if not self._running:
                    break
                time.sleep(1)
        _log("Stopped.")

    def _tick(self) -> None:
        state = _read_last_retrain_record(self._state_path)
        n_now = _on_disk_n_samples(self.data_cache)
        if n_now == 0:
            _log(f"Dataset empty or unreadable at {self.data_cache} — skipping")
            return

        for model in self.models:
            model_state = state.get(model, {})
            last_n = int(model_state.get("n_samples_at_retrain", 0))
            last_ts = float(model_state.get("retrain_ts", 0.0))
            new_bars = n_now - last_n
            age_s = time.time() - last_ts

            _log(f"{model}: n={n_now} last_n={last_n} new_bars={new_bars} "
                 f"cooldown_remaining={max(0, self.cooldown - age_s):.0f}s")

            if new_bars < self.min_new_bars:
                _log(f"{model}: {new_bars} new bars < min_new_bars={self.min_new_bars} — waiting")
                continue
            if age_s < self.cooldown:
                _log(f"{model}: cooldown not expired ({age_s:.0f}s < {self.cooldown}s) — waiting")
                continue

            _log(f"{model}: triggering retrain ({new_bars} new bars, cooldown OK)")
            rc = _run_retrain(
                model,
                str(self.checkpoint_dir),
                self.data_cache,
                self.extra_train_args,
                self.dry_run,
            )
            _log(f"{model}: retrain finished (rc={rc})")

            if rc == 0:
                promoted = _promotion_passed(self.checkpoint_dir, model)
                if promoted is True:
                    ok = _atomic_promote(self.checkpoint_dir, model)
                    _log(f"{model}: promotion gate PASSED — live checkpoint {'updated' if ok else 'FAILED to update'}")
                elif promoted is False:
                    _log(f"{model}: promotion gate FAILED — keeping existing live model")
                else:
                    _log(f"{model}: promotion gate result unknown — checkpoint not swapped")
            else:
                _log(f"{model}: retrain exited with non-zero rc={rc} — skipping promotion")

            # Update state regardless of outcome so cooldown resets
            model_state["n_samples_at_retrain"] = n_now
            model_state["retrain_ts"] = time.time()
            model_state["last_rc"] = rc
            model_state["last_time"] = datetime.now(UTC).isoformat()
            state[model] = model_state
            _write_state(self._state_path, state)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuous auto-learning scheduler")
    p.add_argument("--model", default="haelt", help="Model to retrain (default: haelt)")
    p.add_argument("--all-models", action="store_true", help="Retrain all configured models")
    p.add_argument("--models", nargs="+", default=None, help="Explicit list of models")
    p.add_argument("--checkpoint-dir", default=PATHS.get("checkpoints", "checkpoints"))
    p.add_argument("--data-cache", default=PATHS.get("data_processed", "data/processed"))
    p.add_argument("--min-new-bars", type=int,
                   default=int(os.environ.get("CL_MIN_NEW_BARS", 2016)),
                   help="New bars required to trigger retrain (default 2016 = 1 week of 5-min)")
    p.add_argument("--check-interval", type=int,
                   default=int(os.environ.get("CL_CHECK_INTERVAL", 3600)),
                   help="Seconds between dataset-size checks (default 3600)")
    p.add_argument("--cooldown", type=int,
                   default=int(os.environ.get("CL_COOLDOWN", 86400)),
                   help="Minimum seconds between retrains (default 86400 = 24h)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would happen but don't spawn training")
    p.add_argument("--once", action="store_true",
                   help="Run one check tick then exit (useful for cron)")
    return p.parse_args()


def main() -> None:
    args = _parse()

    if args.models:
        models = args.models
    elif args.all_models:
        from config.settings import TRAINING
        models = list(TRAINING.get("models", {}).keys()) or [args.model]
    else:
        models = [args.model]

    learner = ContinuousLearner(
        models=models,
        checkpoint_dir=args.checkpoint_dir,
        data_cache=args.data_cache,
        min_new_bars=args.min_new_bars,
        check_interval=args.check_interval,
        cooldown=args.cooldown,
        dry_run=args.dry_run,
    )

    if args.once:
        learner._tick()
    else:
        learner.run()


if __name__ == "__main__":
    main()
