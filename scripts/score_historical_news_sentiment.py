"""
Score historical news headlines and write sentiment_score into the CSV.

The training loader prefers a sentiment_score column when present, so this
script is a resumable post-processing step for data/raw/news/historical_news_combined.parquet.

Improvements over the previous version:
  - True parallel batch scoring via SentimentPipeline.score_headlines_batch()
  - --workers N  passed through to SentimentPipeline (parallel Ollama threads)
  - --stats       print score distribution without running any inference
  - --dry-run     preview what would be scored without API calls
  - Incremental CSV write: only rewrites the file at checkpoint boundaries
    using a temp-file atomic swap (no in-place rewrite per headline)
  - tqdm progress bar with graceful plain-print fallback
  - Per-batch ETA and throughput display
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from features.finbert_sentiment import SentimentPipeline


# ── Optional tqdm ──────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, *, desc="", total=None):
        return _tqdm(iterable, desc=desc, total=total, unit="headline",
                     dynamic_ncols=True)

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class _FallbackBar:
        def __init__(self, it, *, desc="", total=None):
            self._it = iter(it)
            self._desc = desc
            self._total = total
            self._n = 0

        def __iter__(self): return self

        def __next__(self):
            v = next(self._it)
            self._n += 1
            pct = f"{100*self._n/self._total:.0f}%" if self._total else str(self._n)
            print(f"\r[progress] {self._desc} {pct}", end="", flush=True)
            return v

        def close(self): print(flush=True)
        def __enter__(self): return self
        def __exit__(self, *_): self.close()

    def _progress(iterable, *, desc="", total=None):
        return _FallbackBar(iterable, desc=desc, total=total)


DEFAULT_NEWS_FILE = Path("data/raw/news/historical_news_combined.parquet")


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write df to a temp file then atomically replace path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _needs_score(df: pd.DataFrame, *, force: bool) -> pd.Series:
    if force or "sentiment_score" not in df.columns:
        return pd.Series(True, index=df.index)
    scores = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return scores.isna()


# ── Stats helper ───────────────────────────────────────────────────────────────

def print_stats(df: pd.DataFrame) -> None:
    total = len(df)
    if "sentiment_score" not in df.columns:
        print(f"[Stats] {total:,} rows | no sentiment_score column yet")
        return

    scores = pd.to_numeric(df["sentiment_score"], errors="coerce")
    scored = scores.notna().sum()
    unscored = total - scored
    print(f"[Stats] Total rows  : {total:,}")
    print(f"[Stats] Scored      : {scored:,}")
    print(f"[Stats] Unscored    : {unscored:,}")

    if scored:
        s = scores.dropna()
        print(f"[Stats] Score range : {s.min():+.3f} .. {s.max():+.3f}")
        print(f"[Stats] Mean        : {s.mean():+.3f}  |  Std: {s.std():.3f}")
        bins = {"bullish (>0.1)": (s > 0.1).sum(),
                "neutral (-0.1..0.1)": ((s >= -0.1) & (s <= 0.1)).sum(),
                "bearish (<-0.1)": (s < -0.1).sum()}
        for label, count in bins.items():
            pct = 100 * count / scored
            print(f"[Stats]   {label:<22}: {count:>6,}  ({pct:.1f}%)")

    if "sentiment_backend" in df.columns:
        backends = df.loc[scores.notna(), "sentiment_backend"].value_counts()
        print("[Stats] Backends:")
        for backend, count in backends.items():
            print(f"[Stats]   {backend:<12}: {count:,}")


# ── Core scoring function ──────────────────────────────────────────────────────

def score_historical_news(
    path: Path,
    *,
    backend: str,
    batch_size: int,
    checkpoint_every: int,
    force: bool,
    workers: int,
    dry_run: bool,
) -> int:
    if not path.exists():
        raise FileNotFoundError(f"historical news file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    if "headline" not in df.columns:
        raise ValueError(f"{path} must contain a 'headline' column")

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = np.nan
    if "sentiment_backend" not in df.columns:
        df["sentiment_backend"] = ""

    mask      = _needs_score(df, force=force)
    headlines = df.loc[mask, "headline"].fillna("").astype(str).str.strip()
    todo      = headlines[headlines != ""]

    if todo.empty:
        print("[Sentiment] Nothing to score -- all rows already have sentiment_score.")
        return 0

    # De-duplicate: score each unique text once, map back to all duplicate rows
    unique_headlines = list(dict.fromkeys(todo.tolist()))
    print(
        f"[Sentiment] {len(unique_headlines):,} unique headline(s) across "
        f"{len(todo):,} row(s) | backend={backend} | workers={workers}",
        flush=True,
    )

    if dry_run:
        print(
            f"[DRY-RUN] Would score {len(unique_headlines):,} unique headline(s) "
            f"in batches of {batch_size} with {workers} worker(s).",
            flush=True,
        )
        return 0

    # Initialise the pipeline
    prefer = None if backend == "auto" else backend
    pipe   = SentimentPipeline(
        prefer_backend=prefer,
        use_cache=True,
        max_workers=workers,
    )

    # Probe backend before the big loop so we know what we're using
    _ = pipe.score_headlines_batch(["probe"])
    print(f"[Sentiment] Active backend: {pipe.active_backend()}", flush=True)

    # ── Main batch loop ────────────────────────────────────────────────────────
    scored_map: dict[str, float] = {}   # headline text -> score
    t0            = time.perf_counter()
    batch_size    = max(1, int(batch_size))
    checkpoint_n  = max(1, int(checkpoint_every))
    next_ckpt     = checkpoint_n
    total         = len(unique_headlines)
    done          = 0

    batches = [unique_headlines[i:i + batch_size]
               for i in range(0, total, batch_size)]

    with _progress(batches, desc="Scoring", total=len(batches)) as bar:
        for batch in bar:
            scores_list = pipe.score_headlines_batch(batch)
            for text, score in zip(batch, scores_list):
                scored_map[text] = score
            done += len(batch)

            if done >= next_ckpt or done == total:
                # Map scores back to all matching rows in the DataFrame
                for orig_idx in todo.index:
                    h = str(df.at[orig_idx, "headline"]).strip()
                    if h in scored_map:
                        df.at[orig_idx, "sentiment_score"]   = float(scored_map[h])
                        df.at[orig_idx, "sentiment_backend"] = pipe.active_backend()

                _atomic_write_csv(df, path)

                elapsed = time.perf_counter() - t0
                rate    = done / max(elapsed, 0.1)
                eta     = (total - done) / max(rate, 0.01)
                print(
                    f"  [Sentiment] {done:,}/{total:,} scored"
                    f" | {pipe.active_backend()}"
                    f" | {rate:.1f}/s"
                    f" | ETA {eta:.0f}s",
                    flush=True,
                )
                while next_ckpt <= done:
                    next_ckpt += checkpoint_n

    # Final cache flush
    pipe.flush_cache()
    return len(unique_headlines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add sentiment_score to historical_news_combined.parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--news-file", default=str(DEFAULT_NEWS_FILE),
        help="Historical news CSV to enrich",
    )
    p.add_argument(
        "--backend", choices=["auto", "ollama", "finbert", "vader"], default="auto",
        help="Sentiment backend. auto tries Ollama first, then FinBERT, then VADER.",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Unique headlines per scoring batch",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=256,
        help="Rewrite the CSV every N unique headlines scored",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Parallel Ollama request threads (no effect on FinBERT/VADER)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-score rows even if sentiment_score already exists",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print score distribution summary without running inference",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be scored without making any API calls",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.news_file)

    if not path.exists():
        print(f"[Sentiment] ERROR: file not found: {path}")
        return 1

    df = pd.read_csv(path, encoding="utf-8")

    if args.stats:
        print_stats(df)
        return 0

    scored = score_historical_news(
        path,
        backend=args.backend,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        force=args.force,
        workers=args.workers,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"[Sentiment] Done -- scored {scored:,} unique headline(s).")
        print_stats(pd.read_csv(path, encoding="utf-8"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
