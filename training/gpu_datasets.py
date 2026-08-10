"""Memory-mapped / Zarr streaming datasets for GPU training."""
from __future__ import annotations

import queue as _queue
import threading
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from training.gpu_cache_io import (
    ZARR,
    _pq_path,
    _x_path,
    _y_cls_path,
    _y_path,
    _zarr_open_group,
)


class _ThreadPrefetchLoader:
    """Wraps any DataLoader to prefetch batches in a daemon background thread.

    Usage:
        loader = DataLoader(ds, ...)
        loader = _ThreadPrefetchLoader(loader, prefetch=8)

    The background thread decompresses / loads the next ``prefetch`` batches
    while the training loop is busy doing a forward+backward pass. Deeper
    queues (6–8) hide Zarr/zstd decompress + H2D under GPU compute better than
    the historical depth of 2.
    """
    def __init__(self, loader, prefetch: int = 8):
        self._loader  = loader
        self._prefetch = max(1, int(prefetch))

    # Forward attribute access to the underlying loader (len, dataset, etc.)
    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        _sentinel = object()
        q = _queue.Queue(maxsize=self._prefetch)

        def _producer():
            try:
                for batch in self._loader:
                    q.put(batch)
            except Exception as exc:          # propagate exceptions to consumer
                q.put(exc)
            finally:
                q.put(_sentinel)

        t = threading.Thread(target=_producer, daemon=True, name="ThreadPrefetchLoader")
        t.start()
        try:
            while True:
                item = q.get()
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Drain the queue so the producer thread can exit cleanly if the
            # consumer breaks early (e.g. chunk-level early stopping).
            t.join(timeout=0)
            while not q.empty():
                try:
                    q.get_nowait()
                except _queue.Empty:
                    break


def wrap_loader_prefetch(loader, args=None, *, prefetch: int | None = None):
    """Wrap ``loader`` with :class:`_ThreadPrefetchLoader` for I/O/GPU overlap.

    With ``num_workers == 0`` (single-process), the prefetch thread is the
    *only* way to overlap Zarr decompress + H2D with the GPU step — wrap
    unconditionally.

    With ``num_workers > 0`` the DataLoader already overlaps decompress with
    GPU compute via its worker processes (each holds up to
    ``prefetch_factor`` batches). Layering a daemon-thread queue of depth 8
    on top of that adds 8 extra pinned batches (~50+ GB on large
    ``batch_size`` × seq-len × feature-dim) contending the GIL for nothing —
    the worker processes are already the buffering layer. To fall back to
    the legacy "always wrap" behaviour (e.g. to hide GPU-step jitter on slow
    disks), pass ``force_thread_prefetch=True`` or set
    ``hardware.force_thread_prefetch: true`` in YAML.

    Depth defaults to ``thread_prefetch_batches`` (YAML default 8) or 8 when
    unset. Pass ``prefetch=`` to force it explicitly.
    """
    if isinstance(loader, _ThreadPrefetchLoader):
        return loader
    force = False
    if isinstance(args, dict):
        force = bool(args.get("force_thread_prefetch", False))
    elif args is not None:
        force = bool(getattr(args, "force_thread_prefetch", False))
    nw = getattr(loader, "num_workers", 0) or 0
    if nw > 0 and not force:
        return loader
    if prefetch is not None:
        depth = max(1, int(prefetch))
    elif args is not None:
        depth = max(1, int(getattr(args, "thread_prefetch_batches", 8) or 8))
    else:
        depth = 8
    return _ThreadPrefetchLoader(loader, prefetch=depth)


class MemmapSequenceDataset(Dataset):
    """
    Reads pre-built sequences directly from Zarr / NPY memmap on disk.
    Never loads the full dataset into RAM — workers stream batches asynchronously.

    Read priority
    -------------
    1. Zarr directory store (.zarr)  — concurrent reads, no locking, LZ4 compressed
    2. NPY memory-maps (_X.npy / _y.npy) — O(1) random access, zero compression overhead

    Why this is fast:
      - Zarr/NPY: OS page cache pre-fetches adjacent chunks while GPU trains.
      - Blosc+lz4@1 (Linux) / zstd@3 compresses training caches for fast sequential decompress.
      - num_workers=4-8 parallel DataLoader workers — no SWMR or retry loops needed.
      - pin_memory=True eliminates CPU->GPU copy latency.
      - persistent_workers=True avoids worker restart overhead per epoch.
      - wrap_loader_prefetch overlaps decompress/H2D with GPU compute.

    Scaler
    ------
    Pass the fitted sklearn StandardScaler (or any object with .transform()) via the
    ``scaler`` argument.  It is applied identically to ZarrStreamDataset so that
    models see the same input distribution regardless of the storage backend.
    Without a scaler the raw (unscaled) arrays are returned — which is WRONG for
    models trained with a scaler.  Always pass the scaler fitted on train split.

    Inf handling
    ------------
    Both backends normalize infinities to ±1e6 before scaling (matching
    ZarrStreamDataset) so the distribution is consistent across backends.
    """

    def __init__(self, cache_path: str, indices: np.ndarray, scaler=None):
        self.cache_path = cache_path
        # Contiguous copy: avoids pickling a view whose base is the full index array.
        self.indices = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
        # Fix A: scaler must be applied to match ZarrStreamDataset behaviour
        self.scaler = scaler

        # -- Detect storage backend --------------------------------------------
        npy_x = Path(_x_path(cache_path))
        npy_y = Path(_y_path(cache_path))

        self.use_zarr = (
            ZARR
            and Path(cache_path).is_dir()
            and (Path(cache_path) / ".zgroup").exists()
        )

        if self.use_zarr:
            _z = _zarr_open_group(cache_path, mode="r")
            self.X_zarr = _z["X"]
            self.y_zarr = _z["y"]
        else:
            self.X_mmap = np.load(str(npy_x), mmap_mode="r")
            self.y_mmap = np.load(str(npy_y), mmap_mode="r")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        if self.use_zarr:
            X = np.array(self.X_zarr[real_idx], dtype=np.float32)
            y = float(self.y_zarr[real_idx])
        else:
            X = np.array(self.X_mmap[real_idx], dtype=np.float32)
            y = float(self.y_mmap[real_idx])
        # Fix B: unify inf handling to ±1e6 (matches ZarrStreamDataset._decompress_block)
        # so models see identical value ranges regardless of storage backend.
        np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        # Fix A: apply scaler identically to ZarrStreamDataset
        if self.scaler is not None:
            orig_shape = X.shape
            if X.ndim == 3:
                X = self.scaler.transform(X.reshape(-1, orig_shape[-1])).astype(np.float32)
                X = X.reshape(orig_shape)
            else:
                X = self.scaler.transform(X).astype(np.float32)
        y = float(np.nan_to_num(np.float32(y), nan=0.0, posinf=0.0, neginf=0.0))
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __getstate__(self):
        # Never pickle memmaps/zarr arrays into worker processes —
        # pickling materialises the full array on Windows (MemoryError).
        return {"cache_path": self.cache_path, "indices": self.indices,
                "use_zarr": self.use_zarr, "scaler": self.scaler}

    def __setstate__(self, state):
        self.cache_path = state["cache_path"]
        self.indices    = state["indices"]
        self.use_zarr   = state.get("use_zarr", False)
        self.scaler     = state.get("scaler", None)  # Fix A: restore scaler in workers
        if self.use_zarr:
            _z = _zarr_open_group(self.cache_path, mode="r")
            self.X_zarr = _z["X"]
            self.y_zarr = _z["y"]
        else:
            npy_x = Path(_x_path(self.cache_path))
            npy_y = Path(_y_path(self.cache_path))
            self.X_mmap = np.load(str(npy_x), mmap_mode="r")
            self.y_mmap = np.load(str(npy_y), mmap_mode="r")


class ZarrStreamDataset(IterableDataset):
    """
    Sequential-read IterableDataset for zarr-backed training data.

    Problem solved
    --------------
    MemmapSequenceDataset.__getitem__ reads one random row at a time.  For
    zarr with 512-row chunks (~262 MB each), every row access decompresses an
    entire 262 MB block.  With batch_size=256 and random shuffle, each batch
    triggers ~242 separate chunk decompressions — measured at 2–49 s each.

    This class reads zarr in blocks aligned to the zarr chunk boundary so each
    physical chunk on disk is decompressed **exactly once** per epoch visit,
    regardless of batch size.  Rows within each block are shuffled in memory;
    the order in which blocks are visited is also shuffled each epoch.

    Multi-worker safety
    -------------------
    Each DataLoader worker receives a disjoint slice of the sorted index array
    (via :func:`np.array_split`, so the trailing worker never silently yields
    nothing when ``len(indices) < num_workers``). Because the index is sorted,
    each worker naturally owns different zarr chunks — no locking needed.

    Per-worker RNG
    -------------
    All shuffle / permutation uses a private ``np.random.Generator``
    (:func:`np.random.default_rng`) seeded from the worker id plus the rank's
    base ``shuffle_seed``. The legacy code touched the global ``np.random``
    state, which is shared by all forked worker processes and produced
    identical shuffles in every worker.

    Cross-chunk shuffle buffer
    --------------------------
    When ``shuffle_chunks=True`` and ``shuffle_buffer_size > 0`` we maintain a
    reservoir of up to ``shuffle_buffer_size`` shuffled rows that span multiple
    zarr chunks; on every push we emit one random row from the reservoir. This
    breaks the temporal-autocorrelation pattern where rows from the same chunk
    always land in adjacent batches. ``shuffle_buffer_size=0`` disables it and
    restores the legacy within-block-only shuffle.

    Missing ``pq`` handling
    ----------------------
    When the cache lacks a ``pq`` (path-quality) array, multitask batches are
    emitted with ``pq=1.0`` (uniform confidence). This matches the convention
    used at cache-write time in ``dataset_builder.py`` (`np.ones(n_rows)` as
    default ``pq``). The previous behaviour was to fake it as
    ``min(1.0, |y|)``, which conflated path quality with absolute return and
    made the confidence head learn a semantically wrong target.
    """

    def __init__(self, cache_path: str, indices: np.ndarray,
                 shuffle_chunks: bool = True, multitask_targets: bool = False,
                 return_indices: bool = False, scaler = None,
                 shuffle_buffer_size: int | None = None,
                 shuffle_seed: int | None = None):
        self.cache_path     = cache_path
        # Sort once so contiguous positions map to the same zarr chunk.
        self.sorted_idx     = np.ascontiguousarray(
            np.sort(np.asarray(indices, dtype=np.int64)))
        self.shuffle_chunks = shuffle_chunks
        # When on, cross-chunk shuffle buffer; off → within-block-only (legacy).
        if shuffle_buffer_size is None:
            shuffle_buffer_size = 8192 if shuffle_chunks else 0
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        # Base seed mixed with worker id so each worker gets an independent
        # stream; None → entropy from numpy.
        self.shuffle_seed = int(shuffle_seed) if shuffle_seed is not None else None
        self.multitask_targets = bool(multitask_targets)
        self.return_indices = bool(return_indices)
        self.scaler         = scaler
        self.use_zarr       = (
            ZARR
            and Path(cache_path).is_dir()
            and (Path(cache_path) / ".zgroup").exists()
        )
        # Read the zarr row-chunk size directly from metadata (no data I/O).
        self._zarr_row_chunk = 512  # safe default
        if self.use_zarr:
            import json as _json
            meta_file = Path(cache_path) / "X" / ".zarray"
            if meta_file.exists():
                self._zarr_row_chunk = int(
                    _json.loads(meta_file.read_text())["chunks"][0]
                )
        # Precompute the per-zarr-chunk block partition (sorted idx is
        # immutable) so __iter__ only re-shuffles a pre-built list per epoch.
        if len(self.sorted_idx):
            chunk_nums = self.sorted_idx // self._zarr_row_chunk
            split_pts  = np.where(np.diff(chunk_nums))[0] + 1
            self._blocks = list(np.split(self.sorted_idx, split_pts))
        else:
            self._blocks = []

    def __len__(self) -> int:
        return len(self.sorted_idx)

    def _worker_rng(self, worker_id: int) -> np.random.Generator:
        """Per-worker independent RNG stream.

        ``np.random`` is process-global and fork-inherited → every worker
        gets the same shuffle. We derive a SeedSequence from the base
        ``shuffle_seed`` (if set) and the worker id so each worker sees a
        different stream while still being reproducible.
        """
        if self.shuffle_seed is None:
            entropy = None
            extra = (worker_id,)
        else:
            entropy = self.shuffle_seed
            extra = (worker_id,)
        try:
            ss = np.random.SeedSequence(entropy=entropy, spawn_key=extra)
        except TypeError:  # very old numpy fallback
            ss = np.random.SeedSequence(entropy + worker_id if entropy is not None else None)
        return np.random.default_rng(ss)

    def _open_arrays(self, worker_id: int):
        """Open X/y/y_cls/pq and cache the handles for this worker process.

        Zarr handle leak risk: with ``persistent_workers=True`` the worker
        process is reused across epochs and ``self._opened_arrays`` would
        forever point at the first cache opened. The dict is now keyed by
        ``(worker_id, cache_path)`` so swapping the cache between epochs on
        the same worker (e.g. the per-epoch index rebuild in
        ``supervised_loop.py``) opens a fresh handle instead of leaking the
        old one.
        """
        if getattr(self, '_opened_arrays', None) is None:
            self._opened_arrays = {}
        key = (worker_id, str(self.cache_path))
        if key not in self._opened_arrays:
            if self.use_zarr:
                z = _zarr_open_group(self.cache_path, mode="r")
                y_cls = z["y_cls"] if "y_cls" in z else None
                pq = z["pq"] if "pq" in z else None
                self._opened_arrays[key] = (z["X"], z["y"], y_cls, pq, True)
            else:
                X = np.load(_x_path(self.cache_path), mmap_mode="r")
                y = np.load(_y_path(self.cache_path), mmap_mode="r")
                y_cls_p, pq_p = Path(_y_cls_path(self.cache_path)), Path(_pq_path(self.cache_path))
                y_cls = np.load(str(y_cls_p), mmap_mode="r") if y_cls_p.exists() else None
                pq = np.load(str(pq_p), mmap_mode="r") if pq_p.exists() else None
                # NPY sidecars are now legacy — y_cls/pq live in the zarr
                # group when ``use_zarr=True``. The NPY branch is retained as
                # a fallback for old caches that predate that move.
                self._opened_arrays[key] = (X, y, y_cls, pq, False)
        return self._opened_arrays[key]

    def _decompress_block(self, block_idx, X_arr, y_arr, y_cls_arr, pq_arr, is_zarr):
        """Decompress one zarr chunk's worth of rows into float32 numpy arrays.

        Uses contiguous-slice reads when the block is contiguous (it always
        is, because ``sorted_idx`` is sorted and blocks are split on chunk
        boundaries) — zarr's fast path. The previous ``oindex[block_idx]``
        call went through the slow fancy-indexing dispatch on every chunk.
        """
        # Handle both single integer and array
        if np.isscalar(block_idx) or (hasattr(block_idx, 'ndim') and block_idx.ndim == 0):
            # Single integer index
            block_idx = np.array([block_idx])
        
        if len(block_idx) == 0:
            return None
        start = int(block_idx[0])
        end   = int(block_idx[-1]) + 1
        if is_zarr and (end - start) == len(block_idx):
            # Contiguous slice — zarr fast path.
            X_blk  = np.array(X_arr[start:end],  dtype=np.float32)
            y_blk  = np.array(y_arr[start:end],  dtype=np.float32)
            yc_blk = (np.array(y_cls_arr[start:end], dtype=np.float32)
                      if y_cls_arr is not None else None)
            pq_blk = (np.array(pq_arr[start:end], dtype=np.float32)
                      if pq_arr is not None else None)
            # block_idx is sorted ascending; the chunk is decompressed in the
            # same order, so the local-to-global offset is just `start`.
            offset = start
        else:
            # Fancy-index fallback (only hit on pathological indices or NPY).
            if is_zarr:
                X_blk  = np.array(X_arr.oindex[block_idx], dtype=np.float32)
                y_blk  = np.array(y_arr.oindex[block_idx], dtype=np.float32)
                yc_blk = (np.array(y_cls_arr.oindex[block_idx], dtype=np.float32)
                          if y_cls_arr is not None else None)
                pq_blk = (np.array(pq_arr.oindex[block_idx], dtype=np.float32)
                          if pq_arr is not None else None)
            else:
                X_blk  = np.array(X_arr[block_idx], dtype=np.float32)
                y_blk  = np.array(y_arr[block_idx], dtype=np.float32)
                yc_blk = (np.array(y_cls_arr[block_idx], dtype=np.float32)
                          if y_cls_arr is not None else None)
                pq_blk = (np.array(pq_arr[block_idx], dtype=np.float32)
                          if pq_arr is not None else None)
            offset = None  # not contiguous; indices are not predictable
        np.nan_to_num(X_blk, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        if self.scaler is not None:
            # Validate scaler shape matches data at load time (not just build time)
            _n_features = getattr(self.scaler, "n_features_in_", None)
            if _n_features is not None:
                _data_n = X_blk.shape[-1]
                if _data_n != _n_features:
                    raise ValueError(
                        f"Scaler/data shape mismatch at load time: "
                        f"scaler.n_features_in_={_n_features}, data has {_data_n} features. "
                        f"Rebuild the Zarr cache or fix the scaler."
                    )
            # StandardScaler expects 2D (samples, features), reshape 3D -> 2D -> 3D
            orig_shape = X_blk.shape
            if X_blk.ndim == 3:
                X_blk = X_blk.reshape(-1, orig_shape[-1])
                X_blk = self.scaler.transform(X_blk).astype(np.float32)
                X_blk = X_blk.reshape(orig_shape)
            else:
                X_blk = self.scaler.transform(X_blk).astype(np.float32)
        np.nan_to_num(y_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if yc_blk is not None:
            np.nan_to_num(yc_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if pq_blk is not None:
            np.nan_to_num(pq_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return X_blk, y_blk, yc_blk, pq_blk, offset

    def _yield_block_rows(self, X_blk, y_blk, yc_blk, pq_blk,
                         block_idx, rng):
        """Yield per-row samples from a single decompressed+sanitised block.

        Pre-permutes the block once and walks the permuted view, so we emit
        ``len(block_idx)`` samples with exactly one tensor construction per
        field per row — no per-row ``float(...)`` wrapping of the scipy-
        style ``torch.tensor(float(y_blk[j]))`` (which forces a scalar copy
        every row). ``block_idx`` is the per-row global sample index, aligned
        to the rows of ``X_blk`` (caller guarantees this — `_decompress_block`
        only emits a contiguous block whose rows map 1:1 to ``block_idx``).
        """
        n = len(X_blk)
        perm = rng.permutation(n)
        for j in perm:
            smp_idx = int(block_idx[j])
            yield self._make_sample(
                X_blk[j], float(y_blk[j]),
                None if yc_blk is None else float(yc_blk[j]),
                None if pq_blk is None else float(pq_blk[j]),
                smp_idx,
            )

    def _make_sample(self, X_row, y_val, yc_val, pq_val, smp_idx):
        """Build a yield-tuple matching the active mode.

        ``pq_val`` is forwarded only when the cache actually published a
        ``pq`` array. When the cache lacks ``pq`` we fall back to ``1.0`` —
        matching the convention used by ``dataset_builder.py`` (which writes
        ``np.ones(n_rows)`` as the default ``pq`` when none is supplied) so
        the multitask BCE head trains against a uniform-confidence target
        instead of being skipped. The previous code faked it as
        ``min(1.0, |y|)``, which conflates "path quality" with "absolute
        return" — the multitask loss then learns |y| as a confidence target,
        a semantically wrong signal.
        """
        X_t = torch.from_numpy(np.ascontiguousarray(X_row)).to(torch.float32)
        if self.return_indices:
            if self.multitask_targets:
                yc_t = (torch.tensor(yc_val, dtype=torch.float32)
                        if yc_val is not None else torch.tensor(float(y_val), dtype=torch.float32))
                pq_t = torch.tensor(1.0 if pq_val is None else float(pq_val),
                                    dtype=torch.float32)
                y_t  = torch.tensor(float(y_val), dtype=torch.float32)
                idx_t = torch.tensor(smp_idx, dtype=torch.long)
                return (X_t, y_t, yc_t, pq_t, idx_t)
            return (X_t, torch.tensor(float(y_val), dtype=torch.float32),
                    torch.tensor(smp_idx, dtype=torch.long))
        if self.multitask_targets:
            yc_t = (torch.tensor(yc_val, dtype=torch.float32)
                    if yc_val is not None else torch.tensor(float(y_val), dtype=torch.float32))
            pq_t = torch.tensor(1.0 if pq_val is None else float(pq_val),
                                dtype=torch.float32)
            y_t  = torch.tensor(float(y_val), dtype=torch.float32)
            return (X_t, y_t, yc_t, pq_t)
        return (X_t, torch.tensor(float(y_val), dtype=torch.float32))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        idx = self.sorted_idx

        # np.array_split (vs ceil-division slicing) so that the trailing
        # workers get empty arrays instead of swallowing rows silently and
        # making ``len(loader)`` over-count. The previous code divided the
        # sorted index with ``(len+n-1)//n`` and silently dropped the tail.
        if worker_info is not None:
            n, wid = worker_info.num_workers, worker_info.id
            # Convert blocks to a flat array for array_split
            flat_blocks = np.concatenate(self._blocks)
            worker_blocks = np.array_split(flat_blocks, n)
            blocks = [worker_blocks[wid]]  # Keep as array in a list
        else:
            # Single process: keep as a list with one array (not a list of scalars)
            blocks = [np.concatenate(self._blocks)]

        # Fast path: this worker owns no chunks.
        # (IterableDataset __iter__ must still return an iterator.)
        if len(blocks) == 0 or len(blocks[0]) == 0:
            return iter(())

        worker_id = worker_info.id if worker_info is not None else 0
        rng = self._worker_rng(worker_id)

        if self.shuffle_chunks:
            rng.shuffle(blocks)

        X_arr, y_arr, y_cls_arr, pq_arr, is_zarr = self._open_arrays(worker_id)

        # Cross-chunk shuffle buffer: pull rows from one block at a time and
        # hold up to ``shuffle_buffer_size`` of them in a reservoir; emit one
        # random reservoir row per push. This decouples emitted batch order
        # from the zarr chunk structure, breaking the autocorrelation
        # pattern where rows from the same 512-row chunk always appear in
        # adjacent batches (a known overfitting source on time-series FX).
        if self.shuffle_chunks and self.shuffle_buffer_size > 0:
            return self._iter_buffered(blocks, X_arr, y_arr, y_cls_arr,
                                       pq_arr, is_zarr, rng)
        return self._iter_unbuffered(blocks, X_arr, y_arr, y_cls_arr,
                                    pq_arr, is_zarr, rng)

    def _iter_unbuffered(self, blocks, X_arr, y_arr, y_cls_arr, pq_arr,
                         is_zarr, rng):
        for block_idx in blocks:
            blk = self._decompress_block(block_idx, X_arr, y_arr,
                                         y_cls_arr, pq_arr, is_zarr)
            if blk is None:
                continue
            X_blk, y_blk, yc_blk, pq_blk, _offset = blk
            yield from self._yield_block_rows(
                X_blk, y_blk, yc_blk, pq_blk, block_idx, rng,
            )

    def _iter_buffered(self, blocks, X_arr, y_arr, y_cls_arr, pq_arr,
                       is_zarr, rng):
        # Reservoir of (X_row, y_val, yc_val, pq_val, smp_idx) tuples.
        buf_X: list = []
        buf_y: list = []
        buf_yc: list = []
        buf_pq: list = []
        buf_idx: list = []
        cap = self.shuffle_buffer_size

        def _push(X_row, y_val, yc_val, pq_val, smp_idx):
            """Append one row to the reservoir; if full, return one random
            sample to emit (else return None). Classic reservoir shuffle:
            the emitted row is a uniformly-random member of the buffer, so
            the emitted batch order is decoupled from the chunk read order."""
            buf_X.append(X_row); buf_y.append(y_val)
            buf_yc.append(yc_val); buf_pq.append(pq_val); buf_idx.append(smp_idx)
            if len(buf_X) > cap:
                k = int(rng.integers(0, len(buf_X)))
                return self._make_sample(
                    buf_X.pop(k), buf_y.pop(k), buf_yc.pop(k),
                    buf_pq.pop(k), buf_idx.pop(k),
                )
            return None

        for block_idx in blocks:
            blk = self._decompress_block(block_idx, X_arr, y_arr,
                                         y_cls_arr, pq_arr, is_zarr)
            if blk is None:
                continue
            X_blk, y_blk, yc_blk, pq_blk, offset = blk
            n = len(X_blk)
            perm = rng.permutation(n)
            for j in perm:
                smp_idx = int(block_idx[j])
                yc_val  = None if yc_blk is None else float(yc_blk[j])
                pq_val  = None if pq_blk is None else float(pq_blk[j])
                sample = _push(X_blk[j], float(y_blk[j]), yc_val, pq_val, smp_idx)
                if sample is not None:
                    yield sample

        # Drain remaining buffer in random order. ``rng.permutation`` is
        # computed BEFORE any pop — popping mid-loop shifts indices, so we
        # sort the reservoir into a fresh random order and pop from a fixed
        # end (cheap O(1) on a list).
        while buf_X:
            k = int(rng.integers(0, len(buf_X)))
            yield self._make_sample(
                buf_X.pop(k), buf_y.pop(k), buf_yc.pop(k),
                buf_pq.pop(k), buf_idx.pop(k),
            )

