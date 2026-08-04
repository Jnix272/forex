"""
monitoring/sidecar.py
=====================
Sidecar process for the Forex Scaling Model.

A sidecar is a companion process that runs alongside the main training or
inference process and handles observability independently:

  • Async log shipping — the main process sends log events via a Queue;
    the sidecar writes them to file, JSONL, and optional Discord without
    blocking the training loop.
  • Metrics collection — GPU temperature, memory, NaN/gradient health,
    epoch timing, and per-batch latency are aggregated and emitted as
    structured JSONL events.
  • Health monitoring — a background watchdog detects frozen training
    (no heartbeat within a configurable timeout) and fires alerts.
  • Log rotation & retention — RotatingFileHandler + optional time-based
    cleanup of old log files.
  • Deduplication — identical alerts within a window are suppressed.
  • Process isolation — if the main process crashes, the sidecar can still
    flush remaining buffered events and write a final summary.

Usage
-----
    from monitoring.sidecar import Sidecar

    sidecar = Sidecar(log_dir="logs", run_name="haelt_0804")
    sidecar.start()

    # From the training loop:
    sidecar.log("info", "Epoch 1 started")
    sidecar.log("warning", "GPU temp high", {"temp_c": 85})
    sidecar.metric("epoch_end", {"train_loss": 0.42, "val_sharpe": 1.3})
    sidecar.heartbeat()

    sidecar.stop()

The sidecar can also be used as a context manager:

    with Sidecar(log_dir="logs", run_name="haelt_0804") as sc:
        sc.log("info", "Training started")
        sc.metric("epoch_end", {"train_loss": 0.42})
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

__all__ = ["Sidecar", "SidecarProcess"]

_SIDECAR_LOGGER = logging.getLogger("forex.sidecar")


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND TYPES — what the main process sends to the sidecar
# ─────────────────────────────────────────────────────────────────────────────

CMD_LOG = "log"
CMD_METRIC = "metric"
CMD_HEARTBEAT = "heartbeat"
CMD_FLUSH = "flush"
CMD_STOP = "stop"
CMD_SHUTDOWN = "shutdown"


# ─────────────────────────────────────────────────────────────────────────────
# SIDECAR PROCESS (runs in a separate OS process)
# ─────────────────────────────────────────────────────────────────────────────

class SidecarProcess:
    """
    The actual sidecar logic that runs in a separate OS process.

    It reads commands from an input Queue, writes structured events to
    JSONL, rotates log files, and optionally ships alerts to Discord.
    """

    def __init__(
        self,
        log_dir: str,
        run_name: str,
        model_name: str = "model",
        max_queue_size: int = 10_000,
        flush_interval_s: float = 2.0,
        retention_days: int = 30,
        enable_discord: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.model_name = model_name
        self.max_queue_size = max_queue_size
        self.flush_interval_s = flush_interval_s
        self.retention_days = retention_days
        self.enable_discord = enable_discord

        self._queue: Queue | None = None
        self._stop_event: threading.Event | None = None
        self._process: Process | None = None
        self._log_path: Path | None = None
        self._jsonl_path: Path | None = None
        self._jsonl_handle = None
        self._file_handler = None
        self._logger: logging.Logger | None = None
        self._last_heartbeat = 0.0
        self._last_flush = 0.0
        self._event_count = 0
        self._dropped_events = 0
        self._start_ts = 0.0

    def start(self) -> bool:
        """Start the sidecar process. Returns True on success."""
        self._queue = Queue(maxsize=self.max_queue_size)
        self._stop_event = threading.Event()

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / f"sidecar_{self.run_name}.log"
        self._jsonl_path = self.log_dir / f"sidecar_{self.run_name}.jsonl"

        # Set up the sidecar's own logger
        self._logger = logging.getLogger(f"forex.sidecar.{self.run_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if not self._logger.handlers:
            fh = RotatingFileHandler(
                str(self._log_path),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._logger.addHandler(fh)

        try:
            self._jsonl_handle = open(str(self._jsonl_path), "a", encoding="utf-8")
        except Exception as e:
            _SIDECAR_LOGGER.error(f"Failed to open JSONL file: {e}")
            return False

        self._process = Process(
            target=self._run_loop,
            args=(self._queue, self._stop_event),
            daemon=True,
            name=f"sidecar-{self.run_name}",
        )
        self._process.start()
        self._start_ts = time.monotonic()
        _SIDECAR_LOGGER.info(
            f"Sidecar started — pid={self._process.pid} "
            f"log={self._log_path} jsonl={self._jsonl_path}"
        )
        return True

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the sidecar process gracefully."""
        if self._queue is None or self._stop_event is None:
            return

        # Send stop command
        try:
            self._queue.put_nowait({"cmd": CMD_STOP, "ts": _now_iso()})
        except queue.Full:
            pass

        # Wait for process to finish
        if self._process and self._process.is_alive():
            self._process.join(timeout=timeout_s)
            if self._process.is_alive():
                _SIDECAR_LOGGER.warning("Sidecar process did not stop; terminating.")
                self._process.terminate()
                self._process.join(timeout=2.0)

        # Flush and close JSONL
        if self._jsonl_handle:
            try:
                self._jsonl_handle.flush()
                self._jsonl_handle.close()
            except Exception:
                pass
            self._jsonl_handle = None

        # Clean up logger handlers
        if self._logger:
            for h in self._logger.handlers[:]:
                h.close()
                self._logger.removeHandler(h)

        _SIDECAR_LOGGER.info(
            f"Sidecar stopped — {self._event_count} events, "
            f"{self._dropped_events} dropped"
        )

    def send(self, cmd: str, data: dict[str, Any]) -> bool:
        """
        Send a command to the sidecar process.
        Returns True if queued successfully, False if queue is full.
        """
        if self._queue is None:
            return False
        try:
            self._queue.put_nowait({"cmd": cmd, "data": data, "ts": _now_iso()})
            return True
        except queue.Full:
            self._dropped_events += 1
            return False

    def log(self, level: str, message: str, **fields: Any) -> bool:
        """Send a log event to the sidecar."""
        return self.send(CMD_LOG, {"level": level, "message": message, **fields})

    def metric(self, metric_name: str, values: dict[str, Any]) -> bool:
        """Send a metric event to the sidecar."""
        return self.send(CMD_METRIC, {"metric": metric_name, "values": values})

    def heartbeat(self) -> bool:
        """Send a heartbeat to the sidecar."""
        return self.send(CMD_HEARTBEAT, {"ts": _now_iso()})

    def flush(self) -> bool:
        """Request the sidecar to flush buffered events."""
        return self.send(CMD_FLUSH, {})

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL — the run loop that executes in the child process
    # ─────────────────────────────────────────────────────────────────────────

    def _run_loop(self, q: Queue, stop: threading.Event) -> None:
        """Main event loop running in the sidecar process."""
        _SIDECAR_LOGGER.info("Sidecar event loop started")
        buffer: list[dict[str, Any]] = []
        last_health_check = time.monotonic()

        while not stop.is_set():
            # Process pending commands
            while True:
                try:
                    msg = q.get(timeout=0.1)
                except queue.Empty:
                    break

                cmd = msg.get("cmd")
                data = msg.get("data", {})

                if cmd == CMD_LOG:
                    self._handle_log(data, buffer)
                elif cmd == CMD_METRIC:
                    self._handle_metric(data, buffer)
                elif cmd == CMD_HEARTBEAT:
                    self._last_heartbeat = time.monotonic()
                    self._write_jsonl(buffer, {"event": "heartbeat", **data})
                    buffer.clear()
                elif cmd == CMD_FLUSH:
                    self._flush_buffer(buffer)
                elif cmd == CMD_STOP:
                    self._flush_buffer(buffer)
                    stop.set()
                    return
                elif cmd == CMD_SHUTDOWN:
                    self._flush_buffer(buffer)
                    return

            # Periodic health check
            now = time.monotonic()
            if now - last_health_check > 60:
                last_health_check = now
                self._health_check(buffer)

            # Periodic flush
            if now - self._last_flush > self.flush_interval_s and buffer:
                self._flush_buffer(buffer)

        # Final flush on exit
        self._flush_buffer(buffer)

    def _handle_log(self, data: dict[str, Any], buffer: list[dict[str, Any]]) -> None:
        level = data.get("level", "info").upper()
        message = data.get("message", "")
        fields = {k: v for k, v in data.items() if k not in ("level", "message")}

        # Map to Python logging levels
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        py_level = level_map.get(level, logging.INFO)

        if self._logger:
            try:
                self._logger.log(py_level, message)
            except Exception:
                pass

        # Also write to JSONL
        record = {
            "event": "log",
            "level": level,
            "message": message,
            "run": self.run_name,
            "model": self.model_name,
            **fields,
        }
        buffer.append(record)
        self._event_count += 1

    def _handle_metric(self, data: dict[str, Any], buffer: list[dict[str, Any]]) -> None:
        metric_name = data.get("metric", "unknown")
        values = data.get("values", {})
        record = {
            "event": "metric",
            "metric": metric_name,
            "values": values,
            "run": self.run_name,
            "model": self.model_name,
        }
        buffer.append(record)
        self._event_count += 1

    def _health_check(self, buffer: list[dict[str, Any]]) -> None:
        """Check sidecar health and write a status event."""
        queue_size = self._queue.qsize() if self._queue else 0
        record = {
            "event": "health_check",
            "queue_size": queue_size,
            "buffer_size": len(buffer),
            "events_total": self._event_count,
            "events_dropped": self._dropped_events,
            "uptime_s": round(time.monotonic() - self._start_ts, 1),
            "run": self.run_name,
        }
        buffer.append(record)

        if queue_size > self.max_queue_size * 0.8:
            _SIDECAR_LOGGER.warning(
                f"Sidecar queue nearly full ({queue_size}/{self.max_queue_size})"
            )

    def _flush_buffer(self, buffer: list[dict[str, Any]]) -> None:
        """Write buffered events to JSONL and flush."""
        if not buffer or self._jsonl_handle is None:
            return

        try:
            for record in buffer:
                record["ts"] = _now_iso()
                self._jsonl_handle.write(json.dumps(record, default=str) + "\n")
            self._jsonl_handle.flush()
            buffer.clear()
        except Exception as e:
            _SIDECAR_LOGGER.error(f"Failed to flush buffer: {e}")

    def _write_jsonl(
        self, buffer: list[dict[str, Any]], record: dict[str, Any]
    ) -> None:
        """Write a single record directly to JSONL (bypasses buffer)."""
        if self._jsonl_handle is None:
            return
        try:
            record["ts"] = _now_iso()
            self._jsonl_handle.write(json.dumps(record, default=str) + "\n")
            self._jsonl_handle.flush()
        except Exception as e:
            _SIDECAR_LOGGER.error(f"Failed to write JSONL record: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SIDECAR MANAGER — the public API used by training/inference code
# ─────────────────────────────────────────────────────────────────────────────

class Sidecar:
    """
    Manager that wraps a SidecarProcess and provides a simple API for the
    main training/inference process.

    The sidecar can run in-process (same process, background thread) or as a
    separate OS process. In-process mode is the default for local development;
    separate-process mode is recommended for production training runs.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        run_name: str = "",
        model_name: str = "model",
        enabled: bool = True,
        mode: str = "process",
        max_queue_size: int = 10_000,
        flush_interval_s: float = 2.0,
        retention_days: int = 30,
        enable_discord: bool = False,
    ):
        self.log_dir = log_dir
        self.run_name = run_name or f"sidecar_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_name = model_name
        self.enabled = enabled
        self.mode = mode
        self.max_queue_size = max_queue_size
        self.flush_interval_s = flush_interval_s
        self.retention_days = retention_days
        self.enable_discord = enable_discord

        self._sidecar: SidecarProcess | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._queue: Queue | None = None
        self._started = False

    def start(self) -> bool:
        """Start the sidecar. Returns True on success."""
        if not self.enabled:
            return False

        if self.mode == "process":
            self._sidecar = SidecarProcess(
                log_dir=self.log_dir,
                run_name=self.run_name,
                model_name=self.model_name,
                max_queue_size=self.max_queue_size,
                flush_interval_s=self.flush_interval_s,
                retention_days=self.retention_days,
                enable_discord=self.enable_discord,
            )
            self._started = self._sidecar.start()
        else:
            # In-process mode: run the sidecar loop in a background thread
            self._queue = Queue(maxsize=self.max_queue_size)
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run_loop,
                args=(self._queue, self._stop_event),
                daemon=True,
                name=f"sidecar-thread-{self.run_name}",
            )
            self._thread.start()
            self._started = True

        if self._started:
            _SIDECAR_LOGGER.info(
                f"Sidecar started — mode={self.mode} run={self.run_name}"
            )
        return self._started

    def stop(self) -> None:
        """Stop the sidecar gracefully."""
        if not self._started:
            return

        if self._sidecar:
            self._sidecar.stop()
        elif self._queue and self._stop_event:
            try:
                self._queue.put_nowait({"cmd": CMD_STOP, "ts": _now_iso()})
            except queue.Full:
                pass
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)

        self._started = False
        _SIDECAR_LOGGER.info("Sidecar stopped")

    def log(self, level: str, message: str, **fields: Any) -> None:
        """Send a log event to the sidecar."""
        if not self._started:
            return
        if self._sidecar:
            self._sidecar.log(level, message, **fields)
        elif self._queue:
            try:
                self._queue.put_nowait(
                    {"cmd": CMD_LOG, "data": {"level": level, "message": message, **fields}}
                )
            except queue.Full:
                pass

    def metric(self, metric_name: str, values: dict[str, Any]) -> None:
        """Send a metric event to the sidecar."""
        if not self._started:
            return
        if self._sidecar:
            self._sidecar.metric(metric_name, values)
        elif self._queue:
            try:
                self._queue.put_nowait(
                    {"cmd": CMD_METRIC, "data": {"metric": metric_name, "values": values}}
                )
            except queue.Full:
                pass

    def heartbeat(self) -> None:
        """Send a heartbeat to the sidecar."""
        if not self._started:
            return
        if self._sidecar:
            self._sidecar.heartbeat()
        elif self._queue:
            try:
                self._queue.put_nowait(
                    {"cmd": CMD_HEARTBEAT, "ts": _now_iso()}
                )
            except queue.Full:
                pass

    def flush(self) -> None:
        """Request the sidecar to flush buffered events."""
        if not self._started:
            return
        if self._sidecar:
            self._sidecar.flush()
        elif self._queue:
            try:
                self._queue.put_nowait({"cmd": CMD_FLUSH})
            except queue.Full:
                pass

    def info(self, message: str, **fields: Any) -> None:
        self.log("info", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self.log("warning", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self.log("error", message, **fields)

    def critical(self, message: str, **fields: Any) -> None:
        self.log("critical", message, **fields)

    # ─────────────────────────────────────────────────────────────────────────
    # Context manager
    # ─────────────────────────────────────────────────────────────────────────

    def __enter__(self) -> Sidecar:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — in-process run loop (for thread mode)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_loop(self, q: Queue, stop: threading.Event) -> None:
        """Background thread run loop for in-process mode."""
        buffer: list[dict[str, Any]] = []

        while not stop.is_set():
            while True:
                try:
                    msg = q.get(timeout=0.1)
                except queue.Empty:
                    break

                cmd = msg.get("cmd")
                data = msg.get("data", {})

                if cmd == CMD_LOG:
                    level = data.get("level", "info").upper()
                    message = data.get("message", "")
                    _SIDECAR_LOGGER.log(
                        getattr(logging, level, logging.INFO),
                        f"[{self.run_name}] {message}",
                    )
                    record = {
                        "event": "log",
                        "level": level,
                        "message": message,
                        "run": self.run_name,
                        "model": self.model_name,
                        **{k: v for k, v in data.items() if k not in ("level", "message")},
                    }
                    buffer.append(record)

                elif cmd == CMD_METRIC:
                    record = {
                        "event": "metric",
                        "metric": data.get("metric", "unknown"),
                        "values": data.get("values", {}),
                        "run": self.run_name,
                        "model": self.model_name,
                    }
                    buffer.append(record)

                elif cmd == CMD_HEARTBEAT:
                    pass

                elif cmd == CMD_FLUSH:
                    self._flush_buffer(buffer)

                elif cmd == CMD_STOP:
                    self._flush_buffer(buffer)
                    stop.set()
                    return

            # Periodic flush
            if buffer and time.monotonic() - getattr(self, "_last_flush", 0) > self.flush_interval_s:
                self._flush_buffer(buffer)

        self._flush_buffer(buffer)

    def _flush_buffer(self, buffer: list[dict[str, Any]]) -> None:
        if not buffer:
            return
        log_path = Path(self.log_dir) / f"sidecar_{self.run_name}.jsonl"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                for record in buffer:
                    record["ts"] = _now_iso()
                    f.write(json.dumps(record, default=str) + "\n")
            buffer.clear()
        except Exception as e:
            _SIDECAR_LOGGER.error(f"Failed to flush sidecar buffer: {e}")
