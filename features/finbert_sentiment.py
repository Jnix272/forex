"""
features/finbert_sentiment.py
==============================
Three-tier local sentiment pipeline for Forex news headlines.

Tier 1 - Ollama (preferred)
    Calls http://localhost:11434/api/generate with a structured JSON prompt.
    Uses whichever model is set via OLLAMA_MODEL (default: gemma4:e2b).
    Zero API cost, GPU-accelerated, completely offline.

Tier 2 - FinBERT (fallback)
    Uses transformers.pipeline("text-classification", model="ProsusAI/finbert").
    ~500 MB download on first use. Fine-tuned on financial language.

Tier 3 - VADER (zero-dep fallback)
    Rule-based, no download required, always available.
    Less accurate but fast and robust.

Usage:
    from features.finbert_sentiment import SentimentPipeline
    pipe = SentimentPipeline()                         # auto-detects best tier
    scores = pipe.score_headlines_batch(["EUR/USD rises on ECB rate hike"])
    # -> list[float], one score per headline in [-1, +1]
    score  = pipe.score_headlines(["EUR/USD rises on ECB rate hike"])
    # -> float (weighted mean, kept for backward compat)
    print(pipe.active_backend())                       # "ollama" / "finbert" / "vader"

Caching:
    Results cached to data/embeddings/sentiment_cache.pkl keyed by MD5(headline).
    Stale root-level sentiment_cache.pkl is merged in and deleted on first load.
    Cache is flushed to disk every `cache_save_every` new entries (default 50),
    not on every single call.

Parallelism:
    score_headlines_batch() uses ThreadPoolExecutor(max_workers) to fire
    multiple Ollama requests concurrently.  For true parallel inference set
    OLLAMA_NUM_PARALLEL=4 in your Ollama server environment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from infrastructure.logging_utils import log_data_load

_log = logging.getLogger(__name__)

# ── Cache paths ────────────────────────────────────────────────────────────────
# Canonical location.  The stale root-level cache is merged in on first load.
CACHE_DIR = Path(
    os.getenv(
        "SENTIMENT_CACHE_DIR",
        str(Path(__file__).resolve().parent.parent / "data" / "embeddings"),
    )
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "sentiment_cache.pkl"

# Stale root-level cache (written by older code versions) - merged on startup.
_STALE_CACHE_FILE = Path(__file__).resolve().parent.parent / "sentiment_cache.pkl"

# ── Backend config ─────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")

FINBERT_MODEL = "ProsusAI/finbert"
LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


# ── Cache helpers ──────────────────────────────────────────────────────────────


def _load_cache() -> dict:
    """
    Load the canonical cache, merging the stale root-level file if present.
    After merging the stale file is deleted so it won't be read again.
    """
    cache: dict = {}
    _t0 = time.perf_counter()

    # Load canonical cache
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            log_data_load(
                "finbert_cache_load",
                str(CACHE_FILE),
                n_rows=len(cache),
                status="ok",
                t0=_t0,
                note=f"size_mb={CACHE_FILE.stat().st_size / 1e6:.1f}",
            )
        except Exception as _e:
            log_data_load(
                "finbert_cache_load",
                str(CACHE_FILE),
                n_rows=0,
                status="corrupt",
                t0=_t0,
                exc=_e,
                note="cache will be re-scored live",
            )
            cache = {}
    else:
        log_data_load("finbert_cache_load", str(CACHE_FILE), n_rows=0, status="skip_missing")

    # Merge stale root cache if it exists
    if _STALE_CACHE_FILE.exists() and _STALE_CACHE_FILE != CACHE_FILE:
        try:
            with open(_STALE_CACHE_FILE, "rb") as f:
                stale = pickle.load(f)
            before = len(cache)
            cache.update({k: v for k, v in stale.items() if k not in cache})
            merged = len(cache) - before
            if merged:
                print(
                    f"[Sentiment] Merged {merged} entries from stale cache {_STALE_CACHE_FILE}",
                    flush=True,
                )
            _STALE_CACHE_FILE.unlink(missing_ok=True)
            log_data_load(
                "finbert_cache_stale_merge",
                str(_STALE_CACHE_FILE),
                n_rows=merged,
                status="ok",
                note=f"into {len(cache)} total entries",
            )
        except Exception as _e:
            log_data_load("finbert_cache_stale_merge", str(_STALE_CACHE_FILE), n_rows=0, status="error", exc=_e)

    return cache


def _save_cache(cache: dict) -> None:
    _t0 = time.perf_counter()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(cache, f)
        os.replace(tmp, CACHE_FILE)
        log_data_load(
            "finbert_cache_save",
            str(CACHE_FILE),
            n_rows=len(cache),
            status="ok",
            t0=_t0,
            note=f"size_mb={CACHE_FILE.stat().st_size / 1e6:.1f}",
        )
    except Exception as _e:
        log_data_load("finbert_cache_save", str(CACHE_FILE), n_rows=len(cache), status="error", t0=_t0, exc=_e)


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── Tier 1: Ollama ─────────────────────────────────────────────────────────────

_OLLAMA_PROMPT_TMPL = (
    "You are a financial sentiment analyser for Forex markets. "
    "Rate the sentiment of the following headline for a Forex trader. "
    'Reply with ONLY a JSON object in this exact format: {{"score": <float>}} '
    "where <float> is between -1.0 (very bearish) and +1.0 (very bullish). "
    "No other text.\n\nHeadline: {headline}\n\nJSON:"
)


def _parse_ollama_float(raw: str) -> float | None:
    """
    Robustly extract a float from an Ollama response.
    Tries JSON parse first, then regex fallback.
    """
    text = raw.strip()

    # Strategy 1: parse as JSON {"score": X}
    for attempt in (text, text.split("\n")[0]):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                val = obj.get("score", obj.get("value", obj.get("sentiment")))
                if val is not None:
                    return float(np.clip(float(val), -1.0, 1.0))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 2: find first float-like token
    m = re.search(r"-?\d+\.?\d*", text)
    if m:
        try:
            return float(np.clip(float(m.group()), -1.0, 1.0))
        except ValueError:
            pass

    return None


def _ollama_reachable(timeout: float = 5.0) -> bool:
    """Quick check: is the Ollama server up at all? Uses /api/tags (no inference)."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _score_ollama(
    text: str,
    model: str = OLLAMA_MODEL,
    timeout: float = 45.0,
) -> float | None:
    """
    Ask the local Ollama LLM to rate sentiment.
    Returns float in [-1, +1], or None if Ollama is unreachable/fails.
    """
    import urllib.error
    import urllib.request

    prompt = _OLLAMA_PROMPT_TMPL.format(headline=text[:400])
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 32},
        }
    ).encode()

    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        raw = str(data.get("response", "")).strip()
        return _parse_ollama_float(raw)
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        return None  # Ollama not running
    except Exception:
        return None


# ── Tier 2: FinBERT ────────────────────────────────────────────────────────────

_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        _finbert_pipeline = pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            truncation=True,
            max_length=512,
            device=device,
            torch_dtype=torch.float16,
        )
    return _finbert_pipeline


def _score_finbert(text: str) -> float | None:
    try:
        pipe = _get_finbert()
        result = pipe(text[:512])[0]
        label = result["label"].lower()
        score = result["score"]  # confidence in [0, 1]
        return LABEL_MAP.get(label, 0.0) * score
    except Exception:
        return None


# ── Tier 3: VADER ─────────────────────────────────────────────────────────────

_vader_analyzer = None


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            _vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            # Simple lexicon fallback when vaderSentiment is not installed.
            class _Simple:
                BULLISH = {
                    "rise",
                    "rises",
                    "gain",
                    "gains",
                    "up",
                    "rally",
                    "bullish",
                    "strong",
                    "beat",
                    "beats",
                    "surges",
                    "high",
                    "positive",
                    "growth",
                    "hawkish",
                    "recovery",
                    "better",
                    "outperform",
                }
                BEARISH = {
                    "fall",
                    "falls",
                    "drop",
                    "drops",
                    "down",
                    "decline",
                    "bearish",
                    "weak",
                    "miss",
                    "misses",
                    "tumbles",
                    "low",
                    "negative",
                    "recession",
                    "crisis",
                    "dovish",
                    "worse",
                    "underperform",
                }

                def polarity_scores(self, text):
                    words = set(text.lower().split())
                    bull = len(words & self.BULLISH)
                    bear = len(words & self.BEARISH)
                    total = bull + bear
                    compound = (bull - bear) / total if total else 0.0
                    return {"compound": float(np.clip(compound, -1, 1))}

            _vader_analyzer = _Simple()
    return _vader_analyzer


def _score_vader(text: str) -> float:
    return float(_get_vader().polarity_scores(text)["compound"])


# ── Main pipeline ──────────────────────────────────────────────────────────────


class SentimentPipeline:
    """
    Three-tier sentiment pipeline: Ollama -> FinBERT -> VADER.

    Parameters
    ----------
    prefer_backend : str or None
        Force a specific backend ("ollama", "finbert", "vader").
        None = auto-detect best available.
    ollama_model : str
        Ollama model name (default: gemma4:e2b via OLLAMA_MODEL env var).
    use_cache : bool
        Cache results by headline MD5 (recommended for large runs).
    max_workers : int
        Threads for parallel Ollama requests in score_headlines_batch().
        Has no effect on FinBERT or VADER (they are not thread-safe).
    cache_save_every : int
        Flush the cache pickle every N *new* entries (not every call).
    """

    def __init__(
        self,
        prefer_backend: str | None = None,
        ollama_model: str = OLLAMA_MODEL,
        use_cache: bool = True,
        max_workers: int = 4,
        cache_save_every: int = 50,
    ):
        self._prefer = prefer_backend
        self._ollama_model = ollama_model
        self._use_cache = use_cache
        self._max_workers = max(1, int(max_workers))
        self._cache_save_every = max(1, int(cache_save_every))
        self._cache: dict = _load_cache() if use_cache else {}
        self._cache_lock = threading.Lock()
        self._new_entries = 0  # entries added since last save
        self._backend: str | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def active_backend(self) -> str:
        if not self._backend:
            self._detect_backend()
        return self._backend or "unknown"

    def score_headlines_batch(self, headlines: list[str]) -> list[float]:
        """
        Score each headline independently and return a list of floats in [-1, +1].

        Uses ThreadPoolExecutor for parallel Ollama requests (Ollama only;
        FinBERT and VADER run single-threaded for safety).
        Cache hits are served immediately without a thread.

        Parameters
        ----------
        headlines : list of str

        Returns
        -------
        list[float], same length and order as input
        """
        if not headlines:
            return []

        texts = [str(h).strip() for h in headlines]

        # Separate cache hits from misses
        results: dict[int, float] = {}
        misses: list[tuple[int, str]] = []  # (original_index, text)

        for i, text in enumerate(texts):
            key = _cache_key(text)
            with self._cache_lock:
                if self._use_cache and key in self._cache:
                    results[i] = self._cache[key]
                    continue
            misses.append((i, text))

        if misses:
            backend = self._detect_backend()

            if backend == "ollama" and self._max_workers > 1:
                # Parallel Ollama requests
                with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                    future_map = {pool.submit(self._score_with_backend, text, backend): idx for idx, text in misses}
                    for future in as_completed(future_map):
                        idx = future_map[future]
                        try:
                            results[idx] = future.result()
                        except Exception:
                            results[idx] = 0.0
            elif backend == "finbert":
                # Native GPU batching
                texts_to_score = [text for idx, text in misses]
                try:
                    pipe = _get_finbert()
                    short_texts = [t[:512] for t in texts_to_score]
                    batch_res = pipe(short_texts, batch_size=256, truncation=True)

                    with self._cache_lock:
                        for (idx, text), res in zip(misses, batch_res, strict=False):
                            label = res["label"].lower()
                            score = res["score"]
                            final_score = LABEL_MAP.get(label, 0.0) * score
                            results[idx] = final_score

                            if self._use_cache:
                                key = _cache_key(text)
                                if key not in self._cache:
                                    self._cache[key] = final_score
                                    self._new_entries += 1
                except Exception:
                    for idx, text in misses:  # noqa: B007
                        results[idx] = 0.0
            else:
                # Sequential (VADER / single-worker Ollama)
                for idx, text in misses:
                    results[idx] = self._score_with_backend(text, backend)

        # Persist new cache entries
        self._maybe_save_cache()

        return [float(results.get(i, 0.0)) for i in range(len(texts))]

    def score_headlines(self, headlines: list[str]) -> float:
        """
        Score a list of headlines and return a weighted mean in [-1, +1].
        Kept for backward compatibility with all existing call sites.
        """
        if not headlines:
            return 0.0
        scores = np.array(self.score_headlines_batch(headlines), dtype=np.float32)
        weights = np.abs(scores) + 0.01  # stronger signals count more
        return float(np.clip(np.average(scores, weights=weights), -1.0, 1.0))

    def prefetch_headlines(self, headlines: list[str], batch_size: int = 512) -> int:
        """Pre-score all *headlines* so subsequent per-window calls are cache hits.

        Returns the number of cache misses that were scored (0 if all cached).
        """
        unique = list({str(h).strip() for h in headlines if str(h).strip()})
        with self._cache_lock:
            misses = [h for h in unique if _cache_key(h) not in self._cache]
        if not misses:
            return 0
        print(
            f"[Sentiment] Prefetch: {len(misses):,} uncached of {len(unique):,} unique headlines",
            flush=True,
        )
        scored = 0
        backend = self._detect_backend()

        if backend == "finbert":
            pipe = _get_finbert()

            def data_gen():
                for t in misses:
                    yield t[:512]

            texts_for_model = list(data_gen())
            print("[Sentiment] Accelerated GPU FinBERT Inference started...", flush=True)
            results = pipe(texts_for_model, batch_size=batch_size, truncation=True)

            for text, res in zip(misses, results, strict=False):
                label = res["label"].lower()
                score = res["score"]
                final_score = LABEL_MAP.get(label, 0.0) * score

                with self._cache_lock:
                    key = _cache_key(text)
                    self._cache[key] = final_score
                    self._new_entries += 1

                scored += 1
                if scored % (batch_size * 10) == 0:
                    print(f"  [Sentiment] prefetch {scored:,}/{len(misses):,}", flush=True)
                    self.flush_cache()
        else:
            for i in range(0, len(misses), batch_size):
                batch = misses[i : i + batch_size]
                self.score_headlines_batch(batch)
                scored += len(batch)
                if scored % (batch_size * 4) == 0 or scored == len(misses):
                    print(
                        f"  [Sentiment] prefetch {scored:,}/{len(misses):,}",
                        flush=True,
                    )
        self.flush_cache()
        return scored

    def flush_cache(self) -> None:
        """Force-save the cache to disk immediately."""
        with self._cache_lock:
            _save_cache(self._cache)
            self._new_entries = 0

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _detect_backend(self) -> str:
        """
        Detect the best available backend.
        - Checks Ollama reachability via /api/tags first (instant, no inference).
        - Then runs a short inference probe with a generous timeout for cold-loading.
        - Falls through to FinBERT, then VADER.
        """
        if self._backend:
            return self._backend

        if self._prefer in (None, "ollama"):
            if not _ollama_reachable():
                print("[Sentiment] Ollama not reachable -- skipping.", flush=True)
            else:
                print(
                    f"[Sentiment] Ollama reachable. Probing {self._ollama_model} (may take up to 60s on cold start)...",
                    flush=True,
                )
                probe = _score_ollama(
                    "EUR/USD rose after the ECB raised rates.",
                    model=self._ollama_model,
                    timeout=60.0,
                )
                if probe is not None:
                    print(
                        f"[Sentiment] Ollama probe OK ({probe:+.3f}) -- using {self._ollama_model}.",
                        flush=True,
                    )
                    self._backend = "ollama"
                    return "ollama"
                else:
                    print(
                        f"[Sentiment] Ollama probe returned None from {self._ollama_model} "
                        "-- check model response format.",
                        flush=True,
                    )

        if self._prefer in (None, "finbert"):
            try:
                _get_finbert()
                self._backend = "finbert"
                return "finbert"
            except Exception:
                pass

        print("[Sentiment] Falling back to VADER.", flush=True)
        self._backend = "vader"
        return "vader"

    def _score_with_backend(self, text: str, backend: str) -> float:
        """Score a single headline using the specified backend; update cache."""
        key = _cache_key(text)
        score: float | None = None

        if backend == "ollama":
            score = _score_ollama(text, model=self._ollama_model)
            if score is None:
                # Ollama temporarily failed - fall through
                score = _score_vader(text)
        elif backend == "finbert":
            score = _score_finbert(text)
            if score is None:
                score = _score_vader(text)
        else:
            score = _score_vader(text)

        score = float(np.clip(score if score is not None else 0.0, -1.0, 1.0))

        if self._use_cache:
            with self._cache_lock:
                if key not in self._cache:
                    self._cache[key] = score
                    self._new_entries += 1

        return score

    def _maybe_save_cache(self) -> None:
        """Save the cache if enough new entries have accumulated."""
        with self._cache_lock:
            if self._new_entries >= self._cache_save_every:
                _save_cache(self._cache)
                self._new_entries = 0

    # ── score_to_series (unchanged, kept for live engine) ─────────────────────

    def score_to_series(
        self,
        headlines_df,
        bars,
        text_col: str = "headline",
        time_col: str = "timestamp_utc",
        decay_lambda: float = 0.1,
    ):
        """
        Align timestamped headlines to bars.index, apply exponential decay.

        Parameters
        ----------
        headlines_df : DataFrame with `text_col` and `time_col` columns.
        bars         : Bar DataFrame with DatetimeTZIndex.
        decay_lambda : Exponential decay rate (higher = faster decay).

        Returns pd.Series aligned to bars.index in [-1, +1].
        """
        is_pl = False
        if hasattr(bars, "is_empty"):
            bars = bars.to_pandas()
            if "timestamp_utc" in bars.columns:
                bars = bars.set_index("timestamp_utc")
            is_pl = True

        if hasattr(headlines_df, "is_empty"):
            headlines_df = headlines_df.to_pandas()

        if len(headlines_df) == 0:
            res = pd.Series(0.0, index=bars.index, name="sentiment")
            if is_pl:
                import polars as pl

                return pl.Series("sentiment", res.values)
            return res

        raw_times = pd.to_datetime(headlines_df[time_col], utc=True, errors="coerce")
        texts = headlines_df[text_col].astype(str).tolist()
        scores_list = self.score_headlines_batch(texts)
        scored = pd.Series(scores_list, index=raw_times, dtype=float)
        scored = scored[scored.index.notna()].sort_index()
        scored = scored[scored != 0.0]

        if len(scored) == 0:
            res = pd.Series(0.0, index=bars.index, name="sentiment")
        else:
            scored = scored.groupby(level=0).mean().sort_index()
            bar_index = pd.DatetimeIndex(bars.index)
            if bar_index.tz is None:
                bar_index_utc = bar_index.tz_localize("UTC")
            else:
                bar_index_utc = bar_index.tz_convert("UTC")

            event_ns = pd.DatetimeIndex(pd.to_datetime(scored.index, utc=True)).to_numpy(dtype="datetime64[ns]").view("int64")
            bar_ns = pd.DatetimeIndex(pd.to_datetime(bar_index_utc, utc=True)).to_numpy(dtype="datetime64[ns]").view("int64")
            last_idx = np.searchsorted(event_ns, bar_ns, side="right") - 1

            values = np.zeros(len(bar_ns), dtype=float)
            valid = last_idx >= 0
            if valid.any():
                elapsed = (bar_ns[valid] - event_ns[last_idx[valid]]) / 1e9
                values[valid] = scored.to_numpy(dtype=float)[last_idx[valid]] * np.exp(-float(decay_lambda) * elapsed)
            res = pd.Series(np.clip(values, -1.0, 1.0), index=bars.index, name="sentiment")
        if is_pl:
            import polars as pl

            return pl.Series("sentiment", res.values)
        return res


# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SentimentPipeline -- smoke test (gemma4:e2b)")
    pipe = SentimentPipeline(use_cache=False, max_workers=2)

    test_headlines = [
        "EUR/USD rises sharply after ECB surprise rate hike",
        "US Dollar falls amid weak NFP data, recession fears grow",
        "Markets steady as Fed holds rates unchanged",
        "GBP/USD hits 18-month high after strong UK jobs data",
        "Yen weakens as Bank of Japan signals prolonged easy policy",
    ]

    scores = pipe.score_headlines_batch(test_headlines)
    for h, s in zip(test_headlines, scores, strict=False):
        print(f"  [{pipe.active_backend():10s}] {s:+.3f}  {h[:60]}")

    print(f"\n  Active backend: {pipe.active_backend()}")
    print("OK")
