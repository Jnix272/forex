import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def get_latest_headlines(limit: int = 20) -> list[str]:
    """
    HTTP-first, file-fallback headline loader.
    Env:
      NEWS_FEED_URL (optional)
      NEWS_FEED_FILE (optional, default: data/news/latest_headlines.json)
    Expected payload: list[str] OR {"headlines": list[str]}
    """
    url = str(os.getenv("NEWS_FEED_URL", "")).strip()
    file_path = Path(
        os.getenv(
            "NEWS_FEED_FILE",
            str(Path(__file__).resolve().parent.parent / "data" / "news" / "latest_headlines.json"),
        )
    )

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
                return out[:limit]
        except Exception:
            pass

    try:
        if file_path.exists():
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                items = payload.get("headlines", [])
            else:
                items = []
            return [str(x).strip() for x in items if str(x).strip()][:limit]
    except Exception:
        pass

    return []
