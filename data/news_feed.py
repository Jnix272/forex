import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_HEADLINES_CACHE: dict = {"ts": 0.0, "headlines": []}
_CACHE_TTL = 900.0  # 15 minutes cache to respect API rate limits


def _fetch_freenewsapi(api_key: str, limit: int = 20) -> list[str]:
    """Fetch recent business / financial news headlines from FreeNewsApi.io."""
    url = "https://api.freenewsapi.io/v1/news?topic=business&language=en"
    req = Request(
        url,
        headers={
            "x-api-key": api_key.strip(),
            "User-Agent": "forex-scaling-model",
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8", errors="replace"))

    articles = data.get("data") or data.get("articles") or []
    headlines = []
    for a in articles:
        if isinstance(a, dict) and a.get("title"):
            headlines.append(str(a["title"]).strip())
        elif isinstance(a, str) and a.strip():
            headlines.append(a.strip())
    return [h for h in headlines if h][:limit]


def get_latest_headlines(limit: int = 20) -> list[str]:
    """
    Headline loader with multi-tier fallback:
      1. FreeNewsApi.io (via FREENEWS_API_KEY or NEWS_API_KEY env var)
      2. Generic HTTP news feed (via NEWS_FEED_URL env var)
      3. Local JSON file (data/news/latest_headlines.json)
    """
    now = time.time()
    if _HEADLINES_CACHE["headlines"] and (now - _HEADLINES_CACHE["ts"] < _CACHE_TTL):
        return _HEADLINES_CACHE["headlines"][:limit]

    file_path = Path(
        os.getenv(
            "NEWS_FEED_FILE",
            str(Path(__file__).resolve().parent.parent / "data" / "news" / "latest_headlines.json"),
        )
    )

    # Tier 1: FreeNewsApi.io
    freenews_key = (
        os.getenv("FREENEWS_API_KEY")
        or os.getenv("NEWS_API_KEY")
        or os.getenv("FREE_NEWS_API_KEY")
    )
    if freenews_key:
        try:
            headlines = _fetch_freenewsapi(freenews_key, limit=limit)
            if headlines:
                _HEADLINES_CACHE["ts"] = now
                _HEADLINES_CACHE["headlines"] = headlines
                try:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(json.dumps({"headlines": headlines}, indent=2), encoding="utf-8")
                except Exception:
                    pass
                return headlines[:limit]
        except Exception:
            pass

    # Tier 2: Generic HTTP endpoint
    url = str(os.getenv("NEWS_FEED_URL", "")).strip()
    if url:
        try:
            req = Request(url, headers={"User-Agent": "forex-scaling-model"})
            with urlopen(req, timeout=5) as r:
                payload = json.loads(r.read().decode("utf-8", errors="replace"))
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                items = payload.get("headlines", [])
            else:
                items = []
            out = [str(x).strip() for x in items if str(x).strip()]
            if out:
                _HEADLINES_CACHE["ts"] = now
                _HEADLINES_CACHE["headlines"] = out
                return out[:limit]
        except Exception:
            pass

    # Tier 3: Local file cache
    try:
        if file_path.exists():
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                items = payload.get("headlines", [])
            else:
                items = []
            res = [str(x).strip() for x in items if str(x).strip()][:limit]
            if res:
                _HEADLINES_CACHE["ts"] = now
                _HEADLINES_CACHE["headlines"] = res
            return res
    except Exception:
        pass

    return []
