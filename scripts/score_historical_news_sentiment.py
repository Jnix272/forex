"""
Score historical news headlines and write sentiment_score into the parquet.

Memory-safe / resumable design (14GB-class hosts):
  1. DuckDB extracts unique unscored headlines -> queue parquet (no full DF load)
  2. Score in batches; append shard files under sentiment_map/ (checkpointed)
  3. DuckDB left-join map shards back into combined (streamed COPY, atomic replace)

Usage examples:
  uv run python scripts/score_historical_news_sentiment.py --dry-run
  uv run python scripts/score_historical_news_sentiment.py --backend finbert --batch-size 64
  uv run python scripts/score_historical_news_sentiment.py --apply-only
  uv run python scripts/score_historical_news_sentiment.py --stats
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from features.finbert_sentiment import SentimentPipeline

try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, *, desc="", total=None):
        return _tqdm(iterable, desc=desc, total=total, unit="batch", dynamic_ncols=True)

except ImportError:

    class _FallbackBar:
        def __init__(self, it, *, desc="", total=None):
            self._it = iter(it)
            self._desc = desc
            self._total = total
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            v = next(self._it)
            self._n += 1
            pct = f"{100 * self._n / self._total:.0f}%" if self._total else str(self._n)
            print(f"\r[progress] {self._desc} {pct}", end="", flush=True)
            return v

        def close(self):
            print(flush=True)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def _progress(iterable, *, desc="", total=None):
        return _FallbackBar(iterable, desc=desc, total=total)


DEFAULT_NEWS_FILE = Path("data/raw/news/historical_news_combined.parquet")


def _duck(mem: str = "3GB", tmp: Path | None = None):
    import duckdb

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{mem}'")
    con.execute("PRAGMA threads=2")
    if tmp is not None:
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")
    return con


def _paths(news: Path) -> dict[str, Path]:
    root = news.parent
    return {
        "queue": root / "sentiment_queue_unscored.parquet",
        "map_dir": root / "sentiment_map",
        "map_all": root / "sentiment_map_all.parquet",
        "merging": news.with_suffix(news.suffix + ".scoring"),
        "backup": news.with_suffix(news.suffix + ".pre_score.bak"),
        "tmp": root / "tmp",
    }


def print_stats_duck(path: Path) -> None:
    con = _duck(tmp=path.parent / "tmp")
    row = con.execute(
        f"""
        SELECT
          count(*) AS total,
          count(*) FILTER (WHERE sentiment_score IS NOT NULL) AS scored,
          count(*) FILTER (WHERE sentiment_score IS NULL) AS unscored,
          min(sentiment_score), max(sentiment_score), avg(sentiment_score)
        FROM read_parquet('{path.as_posix()}')
        """
    ).fetchone()
    total, scored, unscored, smin, smax, smean = row
    print(f"[Stats] Total rows  : {total:,}")
    print(f"[Stats] Scored      : {scored:,}")
    print(f"[Stats] Unscored    : {unscored:,}")
    if scored:
        print(f"[Stats] Score range : {smin:+.3f} .. {smax:+.3f}")
        print(f"[Stats] Mean        : {smean:+.3f}")
    con.close()


def build_queue(news: Path, queue: Path, *, force: bool) -> int:
    """Extract unique unscored (non-URL) headlines to a queue parquet."""
    con = _duck(tmp=news.parent / "tmp")
    partial = queue.with_suffix(".parquet.partial")
    if partial.exists():
        partial.unlink()
    score_filter = "TRUE" if force else "(sentiment_score IS NULL OR isnan(sentiment_score))"
    print("[Sentiment] Building unique-headline queue via DuckDB...", flush=True)
    con.execute(
        f"""
        COPY (
          SELECT DISTINCT trim(CAST(headline AS VARCHAR)) AS headline
          FROM read_parquet('{news.as_posix()}')
          WHERE {score_filter}
            AND headline IS NOT NULL
            AND length(trim(CAST(headline AS VARCHAR))) > 0
            AND lower(trim(CAST(headline AS VARCHAR))) NOT LIKE 'http%'
        ) TO '{partial.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    os.replace(partial, queue)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{queue.as_posix()}')").fetchone()[0]
    print(f"[Sentiment] Queue SAVED: {n:,} unique headlines -> {queue} ({queue.stat().st_size / (1<<20):.1f}M)", flush=True)
    con.close()
    return int(n)


def _already_scored_count(map_dir: Path) -> int:
    if not map_dir.exists():
        return 0
    parts = sorted(map_dir.glob("part_*.parquet"))
    if not parts:
        return 0
    con = _duck(tmp=map_dir.parent / "tmp")
    glob = (map_dir / "part_*.parquet").as_posix()
    n = con.execute(f"SELECT count(DISTINCT headline) FROM read_parquet('{glob}')").fetchone()[0]
    con.close()
    return int(n)


def _next_shard_id(map_dir: Path) -> int:
    map_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for p in map_dir.glob("part_*.parquet"):
        try:
            ids.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return (max(ids) + 1) if ids else 0


def materialize_remaining(queue: Path, map_dir: Path, out: Path) -> int:
    """Write queue ⟕ map into a session todo parquet; return row count."""
    con = _duck(tmp=queue.parent / "tmp")
    partial = out.with_suffix(".parquet.partial")
    if partial.exists():
        partial.unlink()
    parts = sorted(map_dir.glob("part_*.parquet")) if map_dir.exists() else []
    if parts:
        glob = (map_dir / "part_*.parquet").as_posix()
        con.execute(
            f"""
            COPY (
              SELECT q.headline
              FROM read_parquet('{queue.as_posix()}') q
              LEFT JOIN (SELECT DISTINCT headline FROM read_parquet('{glob}')) m
                ON q.headline = m.headline
              WHERE m.headline IS NULL
            ) TO '{partial.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
    else:
        con.execute(
            f"""
            COPY (
              SELECT headline FROM read_parquet('{queue.as_posix()}')
            ) TO '{partial.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
    os.replace(partial, out)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    con.close()
    return int(n)


def iter_todo_batches(todo: Path, *, batch_size: int, limit: int | None):
    """Yield headline lists from the session todo parquet (no OFFSET scans)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(todo)
    emitted = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=["headline"]):
        headlines = [str(x) for x in batch.column(0).to_pylist() if x is not None and str(x).strip()]
        if not headlines:
            continue
        if limit is not None:
            remain = limit - emitted
            if remain <= 0:
                return
            if len(headlines) > remain:
                headlines = headlines[:remain]
        emitted += len(headlines)
        yield headlines


def count_remaining(queue: Path, map_dir: Path) -> int:
    con = _duck(tmp=queue.parent / "tmp")
    parts = sorted(map_dir.glob("part_*.parquet")) if map_dir.exists() else []
    if not parts:
        n = con.execute(f"SELECT count(*) FROM read_parquet('{queue.as_posix()}')").fetchone()[0]
    else:
        glob = (map_dir / "part_*.parquet").as_posix()
        n = con.execute(
            f"""
            SELECT count(*)
            FROM read_parquet('{queue.as_posix()}') q
            LEFT JOIN (SELECT DISTINCT headline FROM read_parquet('{glob}')) m
              ON q.headline = m.headline
            WHERE m.headline IS NULL
            """
        ).fetchone()[0]
    con.close()
    return int(n)


def write_shard(map_dir: Path, shard_id: int, headlines: list[str], scores: list[float], backend: str) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    map_dir.mkdir(parents=True, exist_ok=True)
    path = map_dir / f"part_{shard_id:05d}.parquet"
    table = pa.table(
        {
            "headline": headlines,
            "sentiment_score": scores,
            "sentiment_backend": [backend] * len(headlines),
        }
    )
    tmp = path.with_suffix(".parquet.partial")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def apply_scores(news: Path, map_dir: Path, *, paths: dict[str, Path]) -> int:
    """Join map shards into news parquet via DuckDB; atomic replace."""
    parts = sorted(map_dir.glob("part_*.parquet"))
    if not parts:
        print("[Sentiment] No map shards to apply.", flush=True)
        return 0

    con = _duck(tmp=paths["tmp"])
    glob = (map_dir / "part_*.parquet").as_posix()
    merging = paths["merging"]
    if merging.exists():
        merging.unlink()

    print(f"[Sentiment] Applying {len(parts)} map shard(s) -> {merging} ...", flush=True)
    con.execute(
        f"""
        COPY (
          SELECT
            n.timestamp_utc, n.event_type, n.currency, n.impact, n.headline,
            n.actual, n.forecast, n.source, n.url,
            coalesce(m.sentiment_score, n.sentiment_score) AS sentiment_score,
            n.event_category AS event_category
          FROM read_parquet('{news.as_posix()}') n
          LEFT JOIN (
            SELECT headline, arbitrary(sentiment_score) AS sentiment_score
            FROM read_parquet('{glob}')
            GROUP BY headline
          ) m ON trim(CAST(n.headline AS VARCHAR)) = m.headline
        ) TO '{merging.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
        """
    )
    stats = con.execute(
        f"""
        SELECT count(*),
               count(*) FILTER (WHERE sentiment_score IS NOT NULL),
               count(*) FILTER (WHERE sentiment_score IS NULL)
        FROM read_parquet('{merging.as_posix()}')
        """
    ).fetchone()
    print(
        f"[Sentiment] merging rows={stats[0]:,} scored={stats[1]:,} unscored={stats[2]:,} "
        f"size={merging.stat().st_size / (1<<30):.2f}G",
        flush=True,
    )
    if stats[0] < 20_000_000:
        raise SystemExit(f"ABORT: unexpected row count {stats[0]}; original untouched")

    backup = paths["backup"]
    if backup.exists():
        backup.unlink()
    os.replace(news, backup)
    os.replace(merging, news)
    print(f"[Sentiment] Applied. combined={news} backup={backup}", flush=True)
    con.close()
    return int(stats[1])


def score_historical_news(
    path: Path,
    *,
    backend: str,
    batch_size: int,
    checkpoint_every: int,
    force: bool,
    workers: int,
    dry_run: bool,
    limit: int | None,
    apply: bool,
    rebuild_queue: bool,
) -> int:
    paths = _paths(path)
    queue = paths["queue"]
    map_dir = paths["map_dir"]

    if not path.exists():
        raise FileNotFoundError(f"historical news file not found: {path}")

    if rebuild_queue or not queue.exists() or queue.stat().st_size < 1000:
        build_queue(path, queue, force=force)
    else:
        con = _duck(tmp=paths["tmp"])
        nq = con.execute(f"SELECT count(*) FROM read_parquet('{queue.as_posix()}')").fetchone()[0]
        con.close()
        print(f"[Sentiment] Reusing queue: {nq:,} headlines ({queue})", flush=True)

    already = _already_scored_count(map_dir)
    todo_path = path.parent / "sentiment_todo_session.parquet"
    remaining = materialize_remaining(queue, map_dir, todo_path)
    print(
        f"[Sentiment] Remaining to score: {remaining:,} | already mapped: {already:,}",
        flush=True,
    )

    if dry_run:
        print(
            f"[DRY-RUN] Would score up to {remaining:,} unique headline(s) "
            f"in batches of {batch_size} (backend={backend}).",
            flush=True,
        )
        return 0

    if remaining == 0:
        print("[Sentiment] Nothing left to score in queue/map.", flush=True)
        if apply:
            apply_scores(path, map_dir, paths=paths)
        return 0

    prefer = None if backend == "auto" else backend
    pipe = SentimentPipeline(
        prefer_backend=prefer,
        use_cache=True,
        max_workers=workers,
        cache_save_every=max(50, checkpoint_every),
    )
    _ = pipe.score_headlines_batch(["probe"])
    active = pipe.active_backend()
    print(f"[Sentiment] Active backend: {active}", flush=True)

    batch_size = max(1, int(batch_size))
    checkpoint_every = max(batch_size, int(checkpoint_every))
    target = remaining if limit is None else min(remaining, max(0, int(limit)))
    print(
        f"[Sentiment] Scoring {target:,} headline(s) | batch={batch_size} | "
        f"checkpoint_every={checkpoint_every}",
        flush=True,
    )

    buf_h: list[str] = []
    buf_s: list[float] = []
    shard_id = _next_shard_id(map_dir)
    done = 0
    t0 = time.perf_counter()
    n_batches = (target + batch_size - 1) // batch_size
    with _progress(
        iter_todo_batches(todo_path, batch_size=batch_size, limit=target),
        desc="Scoring",
        total=n_batches,
    ) as bar:
        for batch in bar:
            scores = pipe.score_headlines_batch(batch)
            buf_h.extend(batch)
            buf_s.extend(float(x) for x in scores)
            done += len(batch)

            if len(buf_h) >= checkpoint_every:
                path_s = write_shard(map_dir, shard_id, buf_h, buf_s, active)
                print(
                    f"  [ckpt] {path_s.name} +{len(buf_h):,} | done={done:,}/{target:,} "
                    f"| {done / max(time.perf_counter() - t0, 0.1):.1f}/s",
                    flush=True,
                )
                shard_id += 1
                buf_h, buf_s = [], []
                pipe.flush_cache()

    if buf_h:
        path_s = write_shard(map_dir, shard_id, buf_h, buf_s, active)
        print(f"  [ckpt] {path_s.name} +{len(buf_h):,} (final flush)", flush=True)
        pipe.flush_cache()

    elapsed = time.perf_counter() - t0
    print(
        f"[Sentiment] Scored {done:,} headlines in {elapsed:.1f}s "
        f"({done / max(elapsed, 0.1):.1f}/s)",
        flush=True,
    )

    if apply:
        apply_scores(path, map_dir, paths=paths)

    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add sentiment_score to historical_news_combined.parquet (streaming).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--news-file", default=str(DEFAULT_NEWS_FILE))
    p.add_argument(
        "--backend",
        choices=["auto", "ollama", "finbert", "vader"],
        default="finbert",
        help="Sentiment backend (default finbert for GPU quality).",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=2048,
        help="Write a map shard every N scored headlines",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true", help="Rebuild queue including already-scored rows")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Score at most N headlines this run")
    p.add_argument(
        "--apply",
        action="store_true",
        help="After scoring (or alone with --apply-only), join map into combined.parquet",
    )
    p.add_argument(
        "--apply-only",
        action="store_true",
        help="Only join existing map shards into combined.parquet",
    )
    p.add_argument(
        "--rebuild-queue",
        action="store_true",
        help="Force rebuild of the unique-headline queue parquet",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.news_file)
    if not path.exists():
        print(f"[Sentiment] ERROR: file not found: {path}")
        return 1

    if args.stats:
        print_stats_duck(path)
        return 0

    paths = _paths(path)
    if args.apply_only:
        apply_scores(path, paths["map_dir"], paths=paths)
        print_stats_duck(path)
        return 0

    scored = score_historical_news(
        path,
        backend=args.backend,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        force=args.force,
        workers=args.workers,
        dry_run=args.dry_run,
        limit=args.limit,
        apply=args.apply,
        rebuild_queue=args.rebuild_queue,
    )
    if not args.dry_run:
        print(f"[Sentiment] Done -- scored {scored:,} unique headline(s) this run.")
        rem = count_remaining(paths["queue"], paths["map_dir"]) if paths["queue"].exists() else "?"
        print(f"[Sentiment] Remaining in queue (unmapped): {rem}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
