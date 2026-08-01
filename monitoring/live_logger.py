from __future__ import annotations

import json
import logging
import sys
import atexit
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class LiveLogger:
    def __init__(
        self,
        log_dir: str = "logs",
        run_id: str = "",
        component: str = "live_engine",
        verbose: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.component = component
        self.verbose = verbose
        self._log: Optional[logging.Logger] = None
        self._jsonl = None

    def setup(self) -> Dict[str, Path]:
        if self._jsonl is not None:
            self.close()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"live_{self.run_id}.log"
        jsonl_path = self.log_dir / f"live_{self.run_id}.jsonl"

        logger = logging.getLogger(f"forex.live.{self.run_id}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if not logger.handlers:
            fh = RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s"))
            logger.addHandler(fh)
            if self.verbose:
                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                ch.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(ch)
        self._log = logger
        self._jsonl = open(jsonl_path, "a", encoding="utf-8")
        atexit.register(self.close)
        self.event("INFO", "startup", "live logger initialized")
        return {"log": log_path, "jsonl": jsonl_path}

    def close(self) -> None:
        if self._jsonl is None:
            return
        self.event("INFO", "shutdown", "live logger closed")
        if self._jsonl is not None:
            try:
                self._jsonl.flush()
                self._jsonl.close()
            except Exception:
                pass
            self._jsonl = None

    def info(self, msg: str) -> None:
        if self._log:
            self._log.info(msg)

    def warn(self, msg: str) -> None:
        if self._log:
            self._log.warning(msg)

    def error(self, msg: str) -> None:
        if self._log:
            self._log.error(msg)

    def critical(self, msg: str) -> None:
        if self._log:
            self._log.critical(msg)

    def event(self, level: str, event_type: str, message: str, **fields: Any) -> None:
        lvl = level.upper()
        if lvl == "WARN":
            self.warn(message)
        elif lvl == "ERROR":
            self.error(message)
        elif lvl == "CRITICAL":
            self.critical(message)
        else:
            self.info(message)
        if self._jsonl is None:
            return
        rec: Dict[str, Any] = {
            "ts": _utc_now(),
            "run_id": self.run_id,
            "component": self.component,
            "severity": lvl,
            "event_type": event_type,
            "message": message,
        }
        rec.update(fields)
        try:
            self._jsonl.write(json.dumps(rec, default=str) + "\n")
            self._jsonl.flush()
        except Exception:
            pass
