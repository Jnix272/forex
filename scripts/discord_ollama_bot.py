"""
Discord <-> Ollama training Q&A bot.

Read-only bridge for asking local Ollama about training logs, metrics,
checkpoints, and config from a Discord channel. It does not patch files,
auto-tune config, restart training, or execute shell commands.

Setup:
    set DISCORD_BOT_TOKEN=your_bot_token
    set DISCORD_CHANNEL_ID=your_channel_id
    set OLLAMA_MODEL=gemma4:e2b
    set OLLAMA_URL=http://localhost:11434

Run:
    .venv\\Scripts\\python.exe scripts\\discord_ollama_bot.py

Ask in Discord:
    !train status
    !train result
    !ollama why did validation loss spike?
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import aiohttp


ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
CHECKPOINT_DIR = ROOT / "checkpoints"
CONFIG_PATH = ROOT / "config" / "run.yaml"
DISCORD_API = "https://discord.com/api/v10"
MAX_CONTEXT_CHARS = int(os.getenv("DISCORD_OLLAMA_CONTEXT_CHARS", "18000"))


SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password|webhook|bearer)\s*[:=]\s*['\"]?[^'\"\s]+"),
    re.compile(r"https://discord(?:app)?\.com/api/webhooks/[^\s`'\"]+", re.I),
    re.compile(r"(?i)Authorization:\s*Bearer\s+[^\s`'\"]+"),
)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def redact(text: str) -> str:
    out = text
    for pattern in SECRET_PATTERNS:
        out = pattern.sub(lambda m: m.group(0).split(":", 1)[0].split("=", 1)[0] + ": [REDACTED]", out)
    return out


def latest_file(pattern: str) -> Path | None:
    files = [p for p in LOG_DIR.rglob(pattern) if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def tail_text(path: Path | None, max_chars: int = 8000) -> str:
    if path is None:
        return "No matching file found."
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[could not read {path}: {exc}]"
    return redact(text[-max_chars:])


def parse_latest_jsonl(max_rows: int = 40) -> tuple[Path | None, list[dict[str, Any]]]:
    path = latest_file("*.jsonl")
    if path is None:
        return None, []
    rows: list[dict[str, Any]] = []
    for line in tail_text(path, max_chars=120000).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return path, rows[-max_rows:]


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"message": "No parseable JSONL metrics found."}
    last = rows[-1]
    metric_keys = (
        "epoch", "train_loss", "val_loss", "val_sharpe", "sharpe",
        "best_val_loss", "best_sharpe", "lr", "model", "model_name",
        "train_acc", "val_acc", "profit_factor", "max_drawdown",
    )
    latest = {k: last.get(k) for k in metric_keys if k in last}
    best_sharpe = None
    best_loss = None
    for row in rows:
        for key in ("val_sharpe", "sharpe", "best_sharpe"):
            val = row.get(key)
            if isinstance(val, (int, float)):
                best_sharpe = val if best_sharpe is None else max(best_sharpe, val)
        for key in ("val_loss", "best_val_loss"):
            val = row.get(key)
            if isinstance(val, (int, float)):
                best_loss = val if best_loss is None else min(best_loss, val)
    return {
        "latest": latest,
        "best_seen": {"sharpe": best_sharpe, "val_loss": best_loss},
        "recent_rows": rows[-8:],
    }


def list_checkpoints(limit: int = 12) -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return ["checkpoints directory not found"]
    files = sorted(CHECKPOINT_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    if not files:
        return ["No .pt checkpoints found."]
    out = []
    for path in files:
        try:
            rel = path.relative_to(ROOT)
            size_mb = path.stat().st_size / (1024 * 1024)
            out.append(f"{rel} ({size_mb:.1f} MB)")
        except Exception:
            out.append(str(path))
    return out


def config_excerpt(max_chars: int = 6000) -> str:
    if not CONFIG_PATH.exists():
        return "config/run.yaml not found."
    text = tail_text(CONFIG_PATH, max_chars=max_chars)
    keep_lines = []
    for line in text.splitlines():
        if re.search(r"(?i)(key|token|secret|password|webhook)", line):
            keep_lines.append(re.sub(r":.*$", ": [REDACTED]", line))
        elif line.strip().startswith("#") and len(keep_lines) > 80:
            continue
        else:
            keep_lines.append(line)
    return "\n".join(keep_lines[-180:])


def build_context(question: str) -> str:
    log_path = latest_file("*.log")
    jsonl_path, rows = parse_latest_jsonl()
    context = {
        "user_question": question,
        "latest_log_file": str(log_path.relative_to(ROOT)) if log_path else None,
        "latest_jsonl_file": str(jsonl_path.relative_to(ROOT)) if jsonl_path else None,
        "metrics_summary": summarize_metrics(rows),
        "recent_checkpoints": list_checkpoints(),
        "config_excerpt": config_excerpt(),
        "latest_log_tail": tail_text(log_path, max_chars=6000),
    }
    text = json.dumps(context, indent=2, default=str)
    return text[-MAX_CONTEXT_CHARS:]


async def ask_ollama(session: aiohttp.ClientSession, question: str) -> str:
    base_url = _env("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = _env("OLLAMA_MODEL", "gemma4:e2b")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a read-only training analyst for a forex ML project. "
                    "Answer Discord questions from the provided local context. "
                    "Be concise and practical. Do not claim you changed files, "
                    "restarted jobs, or applied tuning."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nLocal context:\n{build_context(question)}",
            },
        ],
    }
    try:
        async with session.post(f"{base_url}/api/chat", json=payload, timeout=180) as resp:
            if resp.status >= 400:
                return f"Ollama HTTP {resp.status}: {(await resp.text())[:500]}"
            data = await resp.json()
    except Exception as exc:
        return f"Could not reach Ollama at {base_url}: {exc}"
    return redact(str(data.get("message", {}).get("content", "")).strip()) or "Ollama returned an empty response."


async def discord_post(session: aiohttp.ClientSession, token: str, channel_id: str, content: str) -> None:
    headers = {"Authorization": f"Bot {token}"}
    chunks = [content[i : i + 1900] for i in range(0, len(content), 1900)] or [content]
    for chunk in chunks:
        async with session.post(
            f"{DISCORD_API}/channels/{channel_id}/messages",
            headers=headers,
            json={"content": chunk},
            timeout=30,
        ) as resp:
            if resp.status >= 400:
                print(f"[Discord] send failed HTTP {resp.status}: {(await resp.text())[:500]}", flush=True)


def extract_question(message: dict[str, Any], bot_user_id: str) -> str | None:
    author = message.get("author") or {}
    if author.get("bot"):
        return None
    content = str(message.get("content") or "").strip()
    if not content:
        return None
    if content.lower().startswith("!train "):
        return content
    if content.lower().startswith("!ollama "):
        return content.split(" ", 1)[1].strip()
    mentioned = any(str(user.get("id")) == bot_user_id for user in message.get("mentions", []))
    if mentioned:
        return re.sub(rf"<@!?{re.escape(bot_user_id)}>", "", content).strip()
    return None


async def heartbeat(ws: aiohttp.ClientWebSocketResponse, interval_ms: int) -> None:
    while True:
        await asyncio.sleep(interval_ms / 1000)
        await ws.send_json({"op": 1, "d": None})


async def run_bot() -> None:
    token = _env("DISCORD_BOT_TOKEN")
    allowed_channel_id = _env("DISCORD_CHANNEL_ID")
    if not token:
        raise SystemExit("Set DISCORD_BOT_TOKEN first.")
    if not allowed_channel_id:
        raise SystemExit("Set DISCORD_CHANNEL_ID to the one channel this bot may answer in.")

    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bot {token}"}
        async with session.get(f"{DISCORD_API}/users/@me", headers=headers, timeout=30) as resp:
            resp.raise_for_status()
            me = await resp.json()
        bot_user_id = str(me["id"])
        print(f"[Discord] Logged in as {me.get('username')} ({bot_user_id})", flush=True)

        async with session.get(f"{DISCORD_API}/gateway/bot", headers=headers, timeout=30) as resp:
            resp.raise_for_status()
            gateway = await resp.json()
        gateway_url = gateway["url"] + "/?v=10&encoding=json"

        async with session.ws_connect(gateway_url, heartbeat=None, timeout=30) as ws:
            hello = await ws.receive_json()
            asyncio.create_task(heartbeat(ws, int(hello["d"]["heartbeat_interval"])))
            await ws.send_json({
                "op": 2,
                "d": {
                    "token": token,
                    "intents": 1 | 512 | 32768,
                    "properties": {
                        "os": "windows",
                        "browser": "forex-ollama-bot",
                        "device": "forex-ollama-bot",
                    },
                },
            })
            print("[Discord] Ready. Use !train, !ollama, or mention the bot.", flush=True)

            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(msg.data)
                if event.get("t") != "MESSAGE_CREATE":
                    continue
                data = event.get("d") or {}
                channel_id = str(data.get("channel_id") or "")
                if channel_id != allowed_channel_id:
                    continue
                question = extract_question(data, bot_user_id)
                if not question:
                    continue
                print(f"[Discord] Question: {question}", flush=True)
                await discord_post(session, token, channel_id, "Checking local training context with Ollama...")
                answer = await ask_ollama(session, question)
                await discord_post(session, token, channel_id, answer)


if __name__ == "__main__":
    asyncio.run(run_bot())
