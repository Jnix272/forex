"""Shared helpers for the Streamlit GUI: paths, JSON, background jobs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JOBS_DIR = ROOT / "logs" / "gui_jobs"
PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
LIVE_LOGS = ROOT / "logs" / "live"
CONFIGS = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "config").glob("run*.yaml"))
MODELS = ["tft", "haelt", "mamba", "gnn", "transformer", "patchtst", "ensemble"]


def read_json(path) -> dict | list | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []
    return rows[-limit:] if limit else rows


def list_caches() -> list[Path]:
    return sorted(PROCESSED.glob("*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)


def list_run_dirs() -> list[Path]:
    return sorted(
        [p for p in CHECKPOINTS.iterdir() if p.is_dir() and not p.name.startswith("optuna_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if CHECKPOINTS.exists() else []


def short(name: str, n: int = 70) -> str:
    return name if len(name) <= n else name[: n // 2 - 2] + " … " + name[-n // 2 :]


# ── Background jobs ──────────────────────────────────────────────────────────
def start_job(label: str, args: list[str]) -> dict:
    """Run ``python <args>`` detached from the Streamlit session, logging to a file."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    log_path = JOBS_DIR / f"{job_id}.log"
    cmd = [sys.executable, *args]
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            creationflags=flags, env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
        )
    meta = {"id": job_id, "label": label, "cmd": cmd, "pid": proc.pid, "log": str(log_path),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (JOBS_DIR / f"{job_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True)
        return str(int(pid)) in out.stdout
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def list_jobs() -> list[dict]:
    jobs = []
    for p in sorted(JOBS_DIR.glob("*.json"), reverse=True):
        meta = read_json(p)
        if isinstance(meta, dict):
            meta["running"] = pid_alive(int(meta.get("pid", -1)))
            jobs.append(meta)
    return jobs


def stop_job(meta: dict) -> None:
    pid = int(meta.get("pid", -1))
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass


def tail(path, n_lines: int = 200) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n_lines:])
    except OSError:
        return ""
