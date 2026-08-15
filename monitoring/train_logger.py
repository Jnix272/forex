"""
monitoring/train_logger.py
===========================
Comprehensive training logger for the Forex Scaling Model.

Features
--------
  • Human-readable rotating log  -> logs/train_<run>.log   (10 MB × 5 backups)
  • Structured JSONL event stream -> logs/train_<run>.jsonl  (machine-parseable)
  • Session summary on completion  -> logs/train_<run>_summary.txt
  • Discord alerts for CRITICAL events (OOM storm, NaN storm, epoch crash,
    training complete, training crash)
  • GPU temperature + memory tracking every epoch
  • NaN gradient detection
  • Frozen-training heartbeat watchdog (background thread)
  • Optional sidecar process for async log shipping (monitoring/sidecar.py)
  • Zero hard dependencies beyond the stdlib + optional discord_alerts

Public API
----------
    from monitoring.train_logger import TrainingLogger

    log = TrainingLogger(
        log_dir      = "logs",
        run_name     = "haelt_0405_1430",
        model_name   = "haelt",
    )
    log.setup()

    log.on_epoch_start(0, total_epochs=50)
    log.on_batch_oom(batch_idx=12, epoch=0, oom_count=1)
    log.on_batch_nan(batch_idx=34, epoch=0, nan_count=1)
    log.on_epoch_end(0, {"train_loss": 0.42, "val_loss": 0.38,
                          "dir_acc": 0.58, "val_sharpe": 0.91,
                          "lr": 1e-4, "gpu_mb": 4800, "elapsed_s": 47.2})
    log.on_training_complete(best_epoch=22, best_metric=1.34, total_s=3600)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False



# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY THRESHOLDS — tune without touching code
# ─────────────────────────────────────────────────────────────────────────────

# Send a Discord alert after this many OOM-skipped batches in one epoch
OOM_DISCORD_THRESHOLD = 5
# Send a Discord alert after this many NaN-skipped batches in one epoch
NAN_DISCORD_THRESHOLD = 3
# If no heartbeat tick arrives within this many seconds, log a WARNING.
# Make it configurable so slow I/O (network disks, zarr decode) doesn't spam warnings.
HEARTBEAT_TIMEOUT_S   = float(os.getenv("TRAIN_HEARTBEAT_TIMEOUT_S", "1800"))  # default: 30 minutes


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _gpu_stats() -> dict[str, Any]:
    """Return GPU 0 util/temp/memory stats when NVML is available."""
    stats: dict[str, Any] = {
        "gpu_temp_c": -1,
        "gpu_mem_mb": -1,
        "gpu_util_pct": -1,
        "gpu_mem_util_pct": -1,
    }
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        stats["gpu_temp_c"] = int(pynvml.nvmlDeviceGetTemperature(
            h, pynvml.NVML_TEMPERATURE_GPU))
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        stats["gpu_util_pct"] = int(util.gpu)
        stats["gpu_mem_util_pct"] = int(util.memory)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        stats["gpu_mem_mb"] = int(mem.used / 1_000_000)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and stats["gpu_mem_mb"] < 0:
            stats["gpu_mem_mb"] = int(
                torch.cuda.max_memory_allocated() / 1_000_000)
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    return stats


def _fmt_metric(value: Any, fmt: str, suffix: str = "") -> str:
    """Format a metric for Discord without crashing on missing values."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):{fmt}}{suffix}"
    except Exception:
        return str(value)


def _nan_in_gradients(model: Any) -> list[str]:
    """Return names of parameters whose .grad contains NaN values."""
    bad: list[str] = []
    try:
        import torch  # type: ignore
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                bad.append(name)
    except Exception:
        pass
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# HEARTBEAT WATCHDOG
# ─────────────────────────────────────────────────────────────────────────────

class _HeartbeatWatchdog(threading.Thread):
    """
    Background daemon thread.
    Fires a WARNING into the logger if no heartbeat tick is received within
    HEARTBEAT_TIMEOUT_S seconds — signals that the training loop may be frozen.
    """

    def __init__(self, callback, timeout_s: float = HEARTBEAT_TIMEOUT_S):
        super().__init__(daemon=True, name="TrainHeartbeatWatchdog")
        self._callback  = callback
        self._timeout   = timeout_s
        self._last_tick = time.monotonic()
        self._stop_evt  = threading.Event()

    def tick(self) -> None:
        self._last_tick = time.monotonic()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._stop_evt.wait(timeout=30)
            if self._stop_evt.is_set():
                break
            gap = time.monotonic() - self._last_tick
            if gap > self._timeout:
                self._callback(gap)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING UTILS (Color, Stream Capture)
# ─────────────────────────────────────────────────────────────────────────────

class ColorFormatter(logging.Formatter):
    """Custom Formatter for console output with ANSI colors."""
    COLORS = {
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[1;31m" # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        msg = super().format(record)
        if color:
            return f"{color}{msg}{self.RESET}"
        return msg

class StreamToLogger:
    """Fake file-like stream object that redirects writes to a logger instance."""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, buf):
        # Bypass logger for tqdm progress bars to avoid log file spam and red terminal coloring
        if "\r" in buf or ("%|" in buf and ("it/s" in buf or "batch/s" in buf or "s/it" in buf)):
            sys.__stderr__.write(buf)
            sys.__stderr__.flush()
            return

        for line in buf.rstrip().splitlines():
            if line.strip():
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        sys.__stderr__.flush()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TrainingLogger:
    """
    Comprehensive training logger.

    Parameters
    ----------
    log_dir      : Directory where all log files are written.
    run_name     : Unique identifier for this training run
                   (auto-generated from model + timestamp if blank).
    model_name   : Architecture name shown in log headers.
    discord      : Optional pre-built DiscordAlerter instance.
                   If None, one is created automatically from env vars.
    heartbeat    : Enable frozen-training watchdog thread.
    verbose      : Mirror WARNING+ to console in addition to file.
    """

    def __init__(
        self,
        log_dir:    str  = "logs",
        run_name:   str  = "",
        model_name: str  = "model",
        discord:    Any  = None,
        heartbeat:  bool = True,
        verbose:    bool = True,
        sidecar:    Any  = None,
    ):
        self.log_dir    = Path(log_dir)
        self.model_name = model_name
        self.run_name   = run_name or f"{model_name}_{datetime.now().strftime('%m%d_%H%M')}"
        self.verbose    = verbose
        self.sidecar    = sidecar

        # Epoch-level accumulators (reset each epoch)
        self._ep_oom_count  = 0
        self._ep_nan_count  = 0
        self._ep_nan_grads: list[str] = []

        # Session-level history (for summary)
        self._epoch_history: list[dict[str, Any]] = []
        self._errors:        list[str] = []
        self._warnings:      list[str] = []
        self._start_ts       = time.monotonic()
        self._current_epoch  = 0
        self._total_epochs   = 0

        # Internal logger (populated by setup())
        self._log:    logging.Logger | None = None
        self._jlog:   Any | None            = None   # JSONL file handle
        self._log_path: Path | None         = None

        # Discord
        self._discord = discord
        self._discord_ready = False

        # Heartbeat watchdog
        self._watchdog: _HeartbeatWatchdog | None = None
        self._heartbeat_enabled = heartbeat

    # ─────────────────────────────────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────────────────────────────────

    def setup(self) -> Path:
        """
        Initialise log files and background watchdog.
        Call once before training starts.
        Returns the path to the human-readable log file.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

        log_path  = self.log_dir / f"train_{self.run_name}.log"
        jsonl_path = self.log_dir / f"train_{self.run_name}.jsonl"
        self._log_path = log_path

        # ── Python logger ────────────────────────────────────────────────────
        logger = logging.getLogger(f"forex.train.{self.run_name}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if not logger.handlers:
            # Rotating file handler (10 MB × 5 backups)
            fh = RotatingFileHandler(
                log_path, maxBytes=10 * 1024 * 1024, backupCount=5,
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(fh)

            if self.verbose:
                ch = logging.StreamHandler(sys.__stdout__)
                ch.setLevel(logging.INFO)
                ch.setFormatter(ColorFormatter("%(message)s"))
                logger.addHandler(ch)

        self._log = logger

        # ── JSONL event stream ───────────────────────────────────────────────
        self._jlog = open(jsonl_path, "a", encoding="utf-8")

        # ── Discord ──────────────────────────────────────────────────────────
        self._init_discord()

        # ── Heartbeat watchdog ───────────────────────────────────────────────
        if self._heartbeat_enabled:
            self._watchdog = _HeartbeatWatchdog(self._on_watchdog_timeout)
            self._watchdog.start()

        # ── Session header ───────────────────────────────────────────────────
        sep = "═" * 72
        header = (
            f"\n{sep}\n"
            f"  TRAINING SESSION  |  model={self.model_name}  |  run={self.run_name}\n"
            f"  Started : {_utcnow()} UTC\n"
            f"  Log     : {log_path}\n"
            f"{sep}"
        )
        self._log.info(header)
        self._write_event("session_start", {
            "model": self.model_name, "run": self.run_name,
        })
        print(f"[TrainingLogger] Logging -> {log_path}")
        return log_path

    def close(self) -> None:
        """Flush and close all handlers. Called automatically by on_training_complete / on_training_crash."""
        if self._watchdog:
            self._watchdog.stop()
            self._watchdog.join(timeout=5.0)
            self._watchdog = None
        if self._jlog:
            try:
                self._jlog.flush()
                self._jlog.close()
            except Exception:
                pass
        if self._log:
            for h in self._log.handlers[:]:
                h.flush()
                h.close()
                self._log.removeHandler(h)

        # Flush sidecar remaining events
        if self.sidecar is not None:
            try:
                self.sidecar.flush()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # CORE LOG METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._safe_log(logging.INFO, msg)

    def warning(self, msg: str) -> None:
        self._warnings.append(msg)
        self._safe_log(logging.WARNING, msg)

    def error(self, msg: str, exc: Exception | None = None) -> None:
        full = self._format_exc(msg, exc)
        self._errors.append(full)
        self._safe_log(logging.ERROR, full)
        self._write_event("error", {"msg": msg, "exc": _exc_summary(exc)})

    def critical(self, msg: str, exc: Exception | None = None,
                 discord_fields: dict[str, str] | None = None) -> None:
        """Log at CRITICAL level and optionally send a Discord alert."""
        full = self._format_exc(msg, exc)
        self._errors.append(full)
        self._safe_log(logging.CRITICAL, full)
        self._write_event("critical", {"msg": msg, "exc": _exc_summary(exc)})
        if discord_fields is not None:
            self._discord_send("training_warning", discord_fields)

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LIFECYCLE EVENTS
    # ─────────────────────────────────────────────────────────────────────────

    def on_epoch_start(self, epoch: int, total_epochs: int,
                       seq_len: int | None = None) -> None:
        """Call at the beginning of each epoch."""
        self._current_epoch = epoch
        self._total_epochs  = total_epochs
        self._ep_oom_count  = 0
        self._ep_nan_count  = 0
        self._ep_nan_grads  = []

        if self._watchdog:
            self._watchdog.tick()

        extra = f" | seq_len={seq_len}" if seq_len else ""
        self.info(f"[Epoch {epoch+1}/{total_epochs}] START{extra}")
        self._write_event("epoch_start", {
            "epoch": epoch + 1, "total": total_epochs, "seq_len": seq_len,
        })

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        """
        Call at the end of each epoch with a metrics dict.

        Expected keys (all optional):
            train_loss, val_loss, dir_acc, val_sharpe, lr, gpu_mb, elapsed_s
        """
        if self._watchdog:
            self._watchdog.tick()

        gpu = _gpu_stats()
        rec: dict[str, Any] = {
            "epoch":      epoch + 1,
            "train_loss": metrics.get("train_loss"),
            "val_loss":   metrics.get("val_loss"),
            "dir_acc":    metrics.get("dir_acc"),
            "val_sharpe": metrics.get("val_sharpe"),
            "lr":         metrics.get("lr"),
            "val_pred_counts": metrics.get("val_pred_counts"),
            "val_true_counts": metrics.get("val_true_counts"),
            "gpu_mb":     metrics.get("gpu_mb", gpu["gpu_mem_mb"]),
            "gpu_temp_c": gpu["gpu_temp_c"],
            "elapsed_s":  metrics.get("elapsed_s"),
            "oom_skips":  self._ep_oom_count,
            "nan_skips":  self._ep_nan_count,
        }
        self._epoch_history.append(rec)

        msg = (
            f"[Epoch {epoch+1}/{self._total_epochs}] END  "
            f"train={rec['train_loss']:.6f}  val={rec['val_loss']:.6f}  "
            f"dir_acc={rec['dir_acc']:.4f}  sharpe={rec['val_sharpe']:.4f}  "
            f"lr={rec['lr']:.2e}  {rec['elapsed_s']:.1f}s  "
            f"gpu={rec['gpu_mb']}MB  temp={rec['gpu_temp_c']}°C"
        )
        if self._ep_oom_count:
            msg += f"  ⚠ OOM×{self._ep_oom_count}"
        if self._ep_nan_count:
            msg += f"  ⚠ NaN×{self._ep_nan_count}"

        self.info(msg)
        self._write_event("epoch_end", rec)
        _pred_counts = rec.get("val_pred_counts") or [0, 0, 0]
        _true_counts = rec.get("val_true_counts") or [0, 0, 0]
        self._discord_send("training_epoch", {
            "Model | Epoch": f"{self.model_name.upper()} | {rec['epoch']}/{self._total_epochs}",
            "Loss (Train | Val)": f"{_fmt_metric(rec['train_loss'], '.5f')} | {_fmt_metric(rec['val_loss'], '.5f')}",
            "Metrics": f"Acc: {_fmt_metric(rec['dir_acc'], '.4f')} | Sharpe: {_fmt_metric(rec['val_sharpe'], '.4f')} | LR: {_fmt_metric(rec['lr'], '.2e')}",
            "Pred S/H/B": f"{_pred_counts[0]} / {_pred_counts[1]} / {_pred_counts[2]}",
            "True S/H/B": f"{_true_counts[0]} / {_true_counts[1]} / {_true_counts[2]}",
            "System": f"{rec['gpu_mb']}MB ({_fmt_metric(rec['gpu_temp_c'], '.0f')}C) | {_fmt_metric(rec['elapsed_s'], '.1f')}s",
        }, force=True)

    def on_batch_progress(self, epoch: int, batch_idx: int, total_batches: int, loss: float, elapsed_s: float) -> None:
        """Call periodically during an epoch to log mid-epoch progress."""
        if self._watchdog:
            self._watchdog.tick()

        pct = (batch_idx / max(1, total_batches)) * 100
        msg = f"[Epoch {epoch+1}] Batch {batch_idx}/{total_batches} ({pct:.1f}%) | Loss: {loss:.6f} | {elapsed_s:.1f}s"
        self.info(msg)
        self._write_event("batch_progress", {
            "epoch": epoch + 1, "batch": batch_idx, "total": total_batches, "loss": loss, "elapsed_s": elapsed_s
        })

        # Only ping discord if we are exactly at 50% or 25/50/75% etc. We don't want to spam.
        # Let's say we send one at 50% exactly, or roughly 50%.
        if batch_idx > 0 and batch_idx == total_batches // 2:
            self._discord_send("training_epoch", {
                "Event":       "Mid-Epoch Progress",
                "Model":       self.model_name,
                "Epoch":       f"{epoch+1}/{self._total_epochs}",
                "Progress":    f"{batch_idx}/{total_batches} ({pct:.1f}%)",
                "Train Loss":  f"{loss:.6f}",
            })

    def on_batch_oom(self, batch_idx: int, epoch: int, oom_count: int) -> None:
        """Call every time a batch is skipped due to CUDA OOM."""
        self._ep_oom_count += 1
        msg = (f"[Epoch {epoch+1}] OOM at batch {batch_idx} "
               f"(total this epoch: {self._ep_oom_count})")
        self.warning(msg)
        self._write_event("oom", {
            "epoch": epoch + 1, "batch": batch_idx, "ep_total": self._ep_oom_count,
        })

        if self._ep_oom_count == OOM_DISCORD_THRESHOLD:
            self._discord_send("training_warning", {
                "Event":   "CUDA OOM Storm",
                "Model":   self.model_name,
                "Epoch":   str(epoch + 1),
                "OOM Count": str(self._ep_oom_count),
                "Action":  "Reduce batch_size or seq_len",
            })

    def on_batch_nan(self, batch_idx: int, epoch: int, nan_count: int) -> None:
        """Call every time a batch is skipped due to NaN loss."""
        self._ep_nan_count += 1
        msg = (f"[Epoch {epoch+1}] NaN loss at batch {batch_idx} "
               f"(total this epoch: {self._ep_nan_count})")
        self.warning(msg)
        self._write_event("nan_loss", {
            "epoch": epoch + 1, "batch": batch_idx, "ep_total": self._ep_nan_count,
        })

        if self._ep_nan_count == NAN_DISCORD_THRESHOLD:
            self._discord_send("training_warning", {
                "Event":   "NaN Loss Storm",
                "Model":   self.model_name,
                "Epoch":   str(epoch + 1),
                "NaN Count": str(self._ep_nan_count),
                "Action":  "Check LR, grad_clip, AMP dtype",
            }, ping_user=True)

    def on_nan_gradient(self, epoch: int, param_names: list[str]) -> None:
        """Call after detecting NaN values in parameter gradients."""
        if not param_names:
            return
        self._ep_nan_grads.extend(param_names)
        msg = (f"[Epoch {epoch+1}] NaN gradients in: "
               f"{', '.join(param_names[:5])}"
               + (" …" if len(param_names) > 5 else ""))
        self.error(msg)
        self._write_event("nan_gradient", {
            "epoch": epoch + 1, "params": param_names[:10],
        })

    def on_grad_norm(self, epoch: int, batch_idx: int, grad_norm: float,
                     threshold: float = 50.0) -> None:
        """Emit a JSONL event when the pre-clip gradient norm exceeds threshold.

        Called from training loops BEFORE ``clip_grad_norm_`` so the value
        reflects the true gradient magnitude (not the post-clip value which
        is always ≤ grad_clip by construction).
        """
        self._write_event("grad_norm", {
            "epoch": epoch + 1, "batch": batch_idx,
            "grad_norm": float(grad_norm), "threshold": float(threshold),
        })
        if grad_norm > threshold * 2.0:
            self.warning(
                f"[Epoch {epoch+1}] High grad norm ({grad_norm:.2f}) at "
                f"batch {batch_idx} (threshold {threshold})"
            )

    def on_epoch_failure(self, epoch: int, exc: Exception) -> None:
        """Call when an entire epoch raises an unhandled exception."""
        tb  = traceback.format_exc()
        msg = f"[Epoch {epoch+1}] FAILED — {type(exc).__name__}: {exc}"
        self._safe_log(logging.ERROR, f"{msg}\n{tb}")
        self._errors.append(msg)
        self._write_event("epoch_failure", {
            "epoch": epoch + 1,
            "exc_type": type(exc).__name__,
            "exc_msg":  str(exc),
            "traceback": tb[-2000:],
        })
        self._discord_send("training_warning", {
            "Event":     "Epoch Failure",
            "Model":     self.model_name,
            "Epoch":     str(epoch + 1),
            "Exception": f"{type(exc).__name__}: {str(exc)[:120]}",
        }, ping_user=True)

    def on_thermal_throttle(self, temp_c: int, limit_c: int) -> None:
        """Call when GPU temperature exceeds the throttle limit."""
        msg = (f"[GPU] Thermal throttle — {temp_c}°C ≥ limit {limit_c}°C. "
               "Pausing training.")
        self.warning(msg)
        self._write_event("thermal_throttle", {"temp_c": temp_c, "limit_c": limit_c})

    def on_promotion_decision(
        self,
        model_name: str,
        promoted: bool,
        metric_name: str,
        metric_value: float | None,
        gate_summary: str,
        gate_reasons: list[str] | None = None,
        gate_details: dict | None = None,
        challenger_vs_prod: dict | None = None,
    ) -> None:
        """Emit a JSONL event when a promotion gate decision is made.

        Parameters
        ----------
        model_name         : Name of the model being evaluated.
        promoted          : True if model was promoted to production.
        metric_name       : 'val_sharpe' or 'val_loss' (the metric direction).
        metric_value      : The challenger's metric value.
        gate_summary      : PromotionGate.evaluate()['summary'] string.
        gate_reasons      : Per-gate pass/fail list.
        gate_details      : Per-gate numeric details dict.
        challenger_vs_prod: If applicable, dict with prod_metric, min_delta,
                            accepted (bool), and direction ('sharpe'|'loss').
        """
        evt = {
            "model": model_name,
            "promoted": bool(promoted),
            "metric_name": str(metric_name),
            "metric_value": metric_value,
            "gate_summary": str(gate_summary),
            "gate_reasons": list(gate_reasons) if gate_reasons else [],
            "gate_details": gate_details or {},
            "challenger_vs_prod": challenger_vs_prod or {},
        }
        self._write_event("promotion_decision", evt)
        verdict = "PROMOTE" if promoted else "REJECT"
        self.info(
            f"[PromotionGate] {model_name}: {verdict} | "
            f"{metric_name}={metric_value} | {gate_summary}"
        )
        # Append to CSV log for audit trail
        try:
            csv_path = self.log_dir / "promotion_decisions.csv"
            import csv as _csv
            write_header = not csv_path.exists()
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                w = _csv.writer(f)
                if write_header:
                    w.writerow([
                        "ts", "model", "promoted", "metric_name",
                        "metric_value", "verdict_summary",
                    ])
                w.writerow([
                    _utcnow(), model_name, bool(promoted),
                    metric_name, metric_value, gate_summary,
                ])
        except Exception:
            pass
        # Discord alert on important transitions
        if promoted:
            self._discord_send("training_warning", {
                "Event": "Model Promoted",
                "Model": model_name,
                "Metric": f"{metric_name}={metric_value}",
                "Summary": gate_summary[:200] if isinstance(gate_summary, str) else str(gate_summary),
            })
        elif challenger_vs_prod and challenger_vs_prod.get("rejected"):
            self._discord_send("training_warning", {
                "Event": "Challenger Rejected",
                "Model": model_name,
                "Metric": f"{metric_name}={metric_value}",
                "Prod": str(challenger_vs_prod.get("prod_metric")),
                "Reason": str(challenger_vs_prod.get("reason", "")),
            })

    def on_training_complete(
        self,
        best_epoch:   int,
        best_metric:  float,
        total_s:      float,
        metric_name:  str = "val_sharpe",
    ) -> None:
        """Call once after all epochs finish successfully."""
        self._write_summary(crashed=False)
        sep  = "═" * 72
        mins = int(total_s // 60); secs = int(total_s % 60)
        msg  = (
            f"\n{sep}\n"
            f"  TRAINING COMPLETE  |  model={self.model_name}\n"
            f"  Best epoch : {best_epoch + 1}\n"
            f"  Best {metric_name:12s}: {best_metric:.4f}\n"
            f"  Total time : {mins}m {secs}s\n"
            f"  Errors     : {len(self._errors)}\n"
            f"  Warnings   : {len(self._warnings)}\n"
            f"{sep}"
        )
        self._safe_log(logging.INFO, msg)
        self._write_event("session_end", {
            "status":     "complete",
            "best_epoch": best_epoch + 1,
            metric_name:  best_metric,
            "total_s":    total_s,
            "n_errors":   len(self._errors),
            "n_warnings": len(self._warnings),
        })
        # Plot loss curve
        image_path = None
        if MATPLOTLIB_AVAILABLE and self._epoch_history:
            try:
                plt.figure(figsize=(10, 6))
                epochs = [r["epoch"] for r in self._epoch_history]
                train_loss = [r.get("train_loss") for r in self._epoch_history]
                val_loss = [r.get("val_loss") for r in self._epoch_history]
                if any(v is not None for v in train_loss):
                    plt.plot(epochs, train_loss, label="Train Loss", color="blue")
                if any(v is not None for v in val_loss):
                    plt.plot(epochs, val_loss, label="Val Loss", color="orange")
                plt.title(f"Training Curve: {self.model_name} ({self.run_name})")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True, alpha=0.3)
                image_path = str(self.log_dir / f"train_{self.run_name}_curve.png")
                plt.savefig(image_path, bbox_inches="tight", dpi=100)
                plt.close()
            except Exception as e:
                self._safe_log(logging.WARNING, f"[Plot] Failed to generate loss curve: {e}")
        self.close()

    def on_rl_complete(self, stats: dict[str, Any]) -> None:
        """Log RL episode summary metrics after train_agent finishes."""
        self._write_event("rl_complete", stats)
        self._safe_log(
            logging.INFO,
            "[RL] stats — "
            + ", ".join(f"{k.split('/')[-1]}={v}" for k, v in stats.items() if k.startswith("rl/")),
        )

    def on_training_crash(self, exc: Exception) -> None:
        """Call when training aborts due to an unhandled exception."""
        tb  = traceback.format_exc()
        msg = f"TRAINING CRASHED — {type(exc).__name__}: {exc}"
        self._safe_log(logging.CRITICAL, f"{msg}\n{tb}")
        self._errors.append(msg)
        self._write_event("session_end", {
            "status":   "crashed",
            "exc_type": type(exc).__name__,
            "exc_msg":  str(exc),
            "traceback": tb[-2000:],
            "n_errors": len(self._errors),
        })
        self._discord_send("training_warning", {
            "Event":     "Training Crash",
            "Model":     self.model_name,
            "Exception": f"{type(exc).__name__}: {str(exc)[:150]}",
            "Errors":    str(len(self._errors)),
        }, ping_user=True)
        self._write_summary(crashed=True)
        self.close()

    # ─────────────────────────────────────────────────────────────────────────
    # GRADIENT NaN CHECK — call after loss.backward() each batch
    # ─────────────────────────────────────────────────────────────────────────

    def check_gradients(self, model: Any, epoch: int) -> bool:
        """
        Inspect model gradients for NaN values.
        Returns True if any NaN gradients were found.
        Lightweight — skips if torch is not available.
        """
        bad = _nan_in_gradients(model)
        if bad:
            self.on_nan_gradient(epoch, bad)
            return True
        return False

    def heartbeat(self) -> None:
        """Tick the watchdog so long-running phases don't trigger frozen warnings."""
        if self._watchdog is not None:
            self._watchdog.tick()

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────────────────────────────────

    def _safe_log(self, level: int, msg: str) -> None:
        if self._log is not None:
            try:
                self._log.log(level, msg)
            except UnicodeEncodeError:
                safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
                self._log.log(level, safe_msg)

    def _write_event(self, event: str, data: dict[str, Any]) -> None:
        if self._jlog is None:
            return
        record = {"ts": _utcnow(), "event": event, "run": self.run_name,
                  "model": self.model_name, **data}
        try:
            self._jlog.write(json.dumps(record, default=str) + "\n")
            self._jlog.flush()
        except Exception:
            pass

        # Also forward to sidecar for async shipping
        if self.sidecar is not None:
            try:
                self.sidecar.metric(event, {**data, "run": self.run_name})
            except Exception:
                pass

    def _format_exc(self, msg: str, exc: Exception | None) -> str:
        if exc is None:
            return msg
        tb = traceback.format_exc()
        return f"{msg}\n{tb}"

    def _on_watchdog_timeout(self, gap_s: float) -> None:
        msg = (f"[Watchdog] No training progress for {gap_s/60:.1f} min — "
               "training may be frozen or waiting on I/O.")
        self.warning(msg)
        self._discord_send("drift_detected", {
            "Event":    "Training Heartbeat Timeout",
            "Model":    self.model_name,
            "Epoch":    str(self._current_epoch + 1),
            "Idle For": f"{gap_s/60:.1f} min",
        })

    def _init_discord(self) -> None:
        if self._discord is not None:
            self._discord_ready = True
            return
        try:
            from monitoring.discord_alerts import DiscordAlerter  # type: ignore
            url = os.getenv("DISCORD_WEBHOOK_URL", "")
            if url:
                self._discord = DiscordAlerter(verbose=False)
                self._discord_ready = True
            else:
                self._discord_ready = False
        except Exception:
            self._discord_ready = False

    def _discord_send(
        self,
        alert_type: str,
        fields: dict[str, str],
        force: bool = False,
        ping_user: bool = False,
        image_path: str | None = None,
    ) -> None:
        if not self._discord_ready or self._discord is None:
            return
        try:
            self._discord.send(alert_type, fields, force=force, ping_user=ping_user, image_path=image_path)
        except Exception as e:
            self._safe_log(logging.WARNING, f"[Discord] send failed: {e}")

    def _write_summary(self, crashed: bool = False) -> None:
        """Write a human-readable session summary file."""
        if not self._epoch_history:
            return
        summary_path = self.log_dir / f"train_{self.run_name}_summary.txt"
        elapsed = time.monotonic() - self._start_ts
        mins, secs = int(elapsed // 60), int(elapsed % 60)

        lines = [
            "═" * 72,
            "  TRAINING SESSION SUMMARY",
            f"  Model   : {self.model_name}",
            f"  Run     : {self.run_name}",
            f"  Status  : {'CRASHED' if crashed else 'COMPLETE'}",
            f"  Finished: {_utcnow()} UTC",
            f"  Duration: {mins}m {secs}s",
            "═" * 72,
            "",
            f"  Epochs completed : {len(self._epoch_history)}",
            f"  Errors logged    : {len(self._errors)}",
            f"  Warnings logged  : {len(self._warnings)}",
            "",
        ]

        if self._epoch_history:
            lines += [
                f"  {'Ep':>4}  {'TrainLoss':>10}  {'ValLoss':>10}  "
                f"{'DirAcc':>7}  {'Sharpe':>8}  {'LR':>10}  "
                f"{'GPU°C':>6}  {'OOM':>4}  {'NaN':>4}",
                "  " + "─" * 68,
            ]
            for r in self._epoch_history:
                lines.append(
                    f"  {r['epoch']:>4}  "
                    f"{_fmt(r.get('train_loss')):>10}  "
                    f"{_fmt(r.get('val_loss')):>10}  "
                    f"{_fmt(r.get('dir_acc'), 4):>7}  "
                    f"{_fmt(r.get('val_sharpe'), 4):>8}  "
                    f"{_fmt_lr(r.get('lr')):>10}  "
                    f"{r.get('gpu_temp_c', -1):>6}  "
                    f"{r.get('oom_skips', 0):>4}  "
                    f"{r.get('nan_skips', 0):>4}"
                )

        if self._errors:
            lines += ["", "  ERRORS:", "  " + "─" * 68]
            for i, e in enumerate(self._errors[:20], 1):
                lines.append(f"  [{i:02d}] {e[:120]}")
            if len(self._errors) > 20:
                lines.append(f"  … and {len(self._errors) - 20} more (see .log file)")

        lines += ["", "═" * 72]

        try:
            summary_path.write_text("\n".join(lines), encoding="utf-8")
            self._safe_log(logging.INFO, f"[Summary] Written -> {summary_path}")
        except Exception as e:
            self._safe_log(logging.WARNING, f"[Summary] Could not write: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: Any, decimals: int = 6) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


def _fmt_lr(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2e}"
    except Exception:
        return str(v)


def _exc_summary(exc: Exception | None) -> str:
    if exc is None:
        return ""
    return f"{type(exc).__name__}: {str(exc)[:200]}"


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("TrainingLogger — smoke test")
    with tempfile.TemporaryDirectory() as tmp:
        log = TrainingLogger(
            log_dir    = tmp,
            run_name   = "smoke_test",
            model_name = "haelt",
            heartbeat  = False,
            verbose    = True,
        )
        log.setup()

        for ep in range(3):
            log.on_epoch_start(ep, total_epochs=3, seq_len=60)
            log.on_batch_oom(batch_idx=5,  epoch=ep, oom_count=ep + 1)
            log.on_batch_nan(batch_idx=12, epoch=ep, nan_count=ep + 1)
            log.on_epoch_end(ep, {
                "train_loss": 0.42 - ep * 0.05,
                "val_loss":   0.38 - ep * 0.04,
                "dir_acc":    0.55 + ep * 0.01,
                "val_sharpe": 0.80 + ep * 0.15,
                "lr":         1e-4,
                "gpu_mb":     4800,
                "elapsed_s":  45.0,
            })

        log.on_training_complete(best_epoch=2, best_metric=1.10, total_s=135.0)

        log_file = Path(tmp) / "train_smoke_test.log"
        summary  = Path(tmp) / "train_smoke_test_summary.txt"
        jsonl    = Path(tmp) / "train_smoke_test.jsonl"

        print(f"\n── Log ({log_file.stat().st_size} bytes) ──")
        print(log_file.read_text(encoding="utf-8")[:1200])
        print("\n── Summary ──")
        print(summary.read_text(encoding="utf-8"))
        print(f"\n── JSONL events ({sum(1 for _ in jsonl.open())} lines) ──")
        for line in jsonl.open():
            print(" ", line.rstrip())

    print("\nOK ✓")
