"""
scripts/check_apis.py
=====================
Validates every API key and service endpoint configured in .env.
Run this after editing .env to confirm all credentials are working.

Usage:
    python scripts/check_apis.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=False)


# ── helpers ───────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    icon = "PASS" if ok else "FAIL"
    _results.append((name, ok, detail))
    print(f"  [{icon}]  {name:<28} {detail}")


def _get(url: str, headers: dict | None = None, timeout: int = 15) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={**(headers or {}), "User-Agent": "Mozilla/5.0 (forex-scaling-model)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 15) -> bytes:
    import urllib.request

    body = json.dumps(payload).encode()
    h = {"Content-Type": "application/json", **(headers or {}), "User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ── 1. EODHD ─────────────────────────────────────────────────────────────────


def check_eodhd() -> None:
    print("\n  [1] EODHD (forex + cross-asset data)")
    key = os.getenv("EODHD_API_KEY", "")
    if not key:
        _check("EODHD_API_KEY", False, "not set in .env")
        return
    try:
        import urllib.parse

        params = urllib.parse.urlencode(
            {
                "from": "2024-01-02",
                "to": "2024-01-05",
                "api_token": key,
                "fmt": "json",
            }
        )
        raw = _get(f"https://eodhd.com/api/eod/EURUSD.FOREX?{params}")
        data = json.loads(raw)
        ok = isinstance(data, list) and len(data) > 0
        _check("EODHD_API_KEY", ok, f"{len(data)} bars returned (EURUSD)" if ok else f"unexpected: {str(data)[:60]}")
    except Exception as exc:
        _check("EODHD_API_KEY", False, str(exc)[:70])


# ── 2. FRED ───────────────────────────────────────────────────────────────────


def check_fred() -> None:
    print("\n  [2] FRED (Federal Reserve Economic Data)")
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        _check("FRED_API_KEY", False, "not set in .env")
        return
    try:
        from fredapi import Fred

        fred = Fred(api_key=key)
        s = fred.get_series("DGS10", observation_start="2024-01-01", observation_end="2024-01-15")
        ok = s is not None and len(s) > 0
        _check("FRED_API_KEY", ok, f"{len(s)} points (US10Y DGS10)" if ok else "empty response")
    except ImportError:
        _check("FRED_API_KEY", False, "fredapi not installed  ->  pip install fredapi")
    except Exception as exc:
        _check("FRED_API_KEY", False, str(exc)[:70])


# ── 3. Alpha Vantage ──────────────────────────────────────────────────────────


def check_alpha_vantage() -> None:
    print("\n  [3] Alpha Vantage (economic calendar)")
    key = os.getenv("AV_API_KEY", "")
    if not key:
        _check("AV_API_KEY", False, "not set in .env")
        return
    try:
        import urllib.parse

        params = urllib.parse.urlencode(
            {
                "function": "FX_DAILY",
                "from_symbol": "EUR",
                "to_symbol": "USD",
                "apikey": key,
                "outputsize": "compact",
            }
        )
        raw = _get(f"https://www.alphavantage.co/query?{params}")
        data = json.loads(raw)
        if "Time Series FX (Daily)" in data:
            pts = len(data["Time Series FX (Daily)"])
            _check("AV_API_KEY", True, f"{pts} daily bars returned (EURUSD)")
        elif "Note" in data:
            _check("AV_API_KEY", False, "rate limited (free tier: 25 req/day)")
        elif "Information" in data:
            _check("AV_API_KEY", False, data["Information"][:70])
        else:
            _check("AV_API_KEY", False, str(list(data.keys()))[:70])
    except Exception as exc:
        _check("AV_API_KEY", False, str(exc)[:70])


# ── 4. Weights & Biases ───────────────────────────────────────────────────────


def check_wandb() -> None:
    print("\n  [4] Weights & Biases (experiment tracking)")
    key = os.getenv("WANDB_API_KEY", "")
    if not key:
        _check("WANDB_API_KEY", False, "not set in .env")
        return
    try:
        raw = _post_json(
            "https://api.wandb.ai/graphql",
            {"query": "{viewer{id username email}}"},
            headers={"Authorization": f"Bearer {key}"},
        )
        data = json.loads(raw)
        user = data.get("data", {}).get("viewer", {})
        if user and user.get("username"):
            _check("WANDB_API_KEY", True, f"user: {user['username']}  ({user.get('email', '')})")
        else:
            _check("WANDB_API_KEY", False, f"unexpected response: {str(data)[:60]}")
    except Exception as exc:
        _check("WANDB_API_KEY", False, str(exc)[:70])


# ── 5. Discord webhook ────────────────────────────────────────────────────────


def check_discord() -> None:
    print("\n  [5] Discord (alert webhook)")
    url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not url:
        _check("DISCORD_WEBHOOK_URL", False, "not set in .env")
        return
    try:
        raw = _get(url)
        data = json.loads(raw)
        name = data.get("name", "?")
        gid = data.get("guild_id", "?")
        ch = data.get("channel_id", "?")
        _check("DISCORD_WEBHOOK_URL", True, f"webhook: {name!r}  guild: {gid}  channel: {ch}")
    except Exception as exc:
        _check("DISCORD_WEBHOOK_URL", False, str(exc)[:70])


# ── 6. Telegram ───────────────────────────────────────────────────────────────


def check_telegram() -> None:
    print("\n  [6] Telegram (optional alert bot)")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token:
        _check("TELEGRAM_BOT_TOKEN", False, "not set (optional)")
        return
    try:
        raw = _get(f"https://api.telegram.org/bot{token}/getMe")
        data = json.loads(raw)
        bot = data.get("result", {})
        _check(
            "TELEGRAM_BOT_TOKEN",
            data.get("ok", False),
            f"bot: @{bot.get('username', '?')}  chat_id: {chat_id or 'not set'}",
        )
    except Exception as exc:
        _check("TELEGRAM_BOT_TOKEN", False, str(exc)[:70])


# ── 7. Ollama (local LLM) ────────────────────────────────────────────────────


def check_ollama() -> None:
    print("\n  [7] Ollama (local LLM sentiment)")
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "mistral")
    try:
        raw = _get(f"{url}/api/tags", timeout=5)
        data = json.loads(raw)
        models = [m["name"] for m in data.get("models", [])]
        model_loaded = any(model in m for m in models)
        if model_loaded:
            _check("OLLAMA", True, f"{model!r} loaded  |  {len(models)} models available")
        else:
            _check("OLLAMA", False, f"running but {model!r} not loaded  ->  ollama pull {model}")
    except Exception:
        _check("OLLAMA", False, f"not running at {url}  (start with: ollama serve)")


# ── 8. MLflow ────────────────────────────────────────────────────────────────


def check_mlflow() -> None:
    print("\n  [8] MLflow (experiment tracking server)")
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        raw = _get(f"{uri}/api/2.0/mlflow/experiments/list", timeout=5)
        data = json.loads(raw)
        exps = len(data.get("experiments", []))
        _check("MLFLOW", True, f"{exps} experiments at {uri}")
    except Exception:
        _check("MLFLOW", False, f"not running at {uri}  (fallback: logs/mlflow_fallback/)")


# ── 9. Stooq (free, no key) ───────────────────────────────────────────────────


def check_stooq() -> None:
    print("\n  [9] Stooq (free public data - no key needed)")
    try:
        import io

        import pandas as pd

        raw = _get("https://stooq.com/q/d/l/?s=^spx&i=d")
        df = pd.read_csv(io.StringIO(raw.decode("utf-8", "replace")))
        ok = "Close" in df.columns and len(df) > 5
        _check("Stooq", ok, f"{len(df):,} daily bars (SPX)" if ok else "unexpected response")
    except Exception as exc:
        _check("Stooq", False, str(exc)[:70])


# ── 10. TimescaleDB (optional) ───────────────────────────────────────────────


def check_timescale() -> None:
    print("\n  [10] TimescaleDB (optional - only needed for live trading)")
    host = os.getenv("TIMESCALE_HOST", "localhost")
    port = int(os.getenv("TIMESCALE_PORT", "5432"))
    db = os.getenv("TIMESCALE_DB", "forex_ticks")
    user = os.getenv("TIMESCALE_USER", "forex_user")
    pw = os.getenv("TIMESCALE_PASSWORD", "")
    if not pw:
        _check("TimescaleDB", False, "TIMESCALE_PASSWORD not set (optional for live trading)")
        return
    try:
        import psycopg2

        conn = psycopg2.connect(host=host, port=port, dbname=db, user=user, password=pw, connect_timeout=5)
        conn.close()
        _check("TimescaleDB", True, f"connected  {user}@{host}:{port}/{db}")
    except ImportError:
        _check("TimescaleDB", False, "psycopg2 not installed (optional)")
    except Exception as exc:
        _check("TimescaleDB", False, str(exc)[:70])


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("=" * 66)
    print("  Forex Scaling Model -- API Key Checker")
    print("=" * 66)

    check_eodhd()
    check_fred()
    check_alpha_vantage()
    check_wandb()
    check_discord()
    check_telegram()
    check_ollama()
    check_mlflow()
    check_stooq()
    check_timescale()

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [name for name, ok, detail in _results if not ok]

    print()
    print("=" * 66)
    print(f"  {passed}/{len(_results)} APIs operational")
    if failed:
        print(f"  Not available : {', '.join(failed)}")
    print("=" * 66)
    print()


if __name__ == "__main__":
    main()
