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
    """Always wrap ``loader`` with :class:`_ThreadPrefetchLoader` for I/O/GPU overlap.

    Applies even when ``num_workers > 0`` so Zarr decompress / pinned H2D can
    run ahead of the training step. Depth defaults to
    ``thread_prefetch_batches`` (YAML default 8) or 8 when unset.
    """
    if isinstance(loader, _ThreadPrefetchLoader):
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
    Never loads the full dataset into RAM ΓÇö workers stream batches asynchronously.

    Read priority
    -------------
    1. Zarr directory store (.zarr)  ΓÇö concurrent reads, no locking, LZ4 compressed
    2. NPY memory-maps (_X.npy / _y.npy) ΓÇö O(1) random access, zero compression overhead

    Why this is fast:
      - Zarr/NPY: OS page cache pre-fetches adjacent chunks while GPU trains.
      - Blosc+lz4@1 (Linux) / zstd@3 compresses training caches for fast sequential decompress.
      - num_workers=4-8 parallel DataLoader workers ΓÇö no SWMR or retry loops needed.
      - pin_memory=True eliminates CPU->GPU copy latency.
      - persistent_workers=True avoids worker restart overhead per epoch.
      - wrap_loader_prefetch overlaps decompress/H2D with GPU compute.
    """

    def __init__(self, cache_path: str, indices: np.ndarray):
        self.cache_path = cache_path
        # Contiguous copy: avoids pickling a view whose base is the full index array.
        self.indices = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))

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
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y = float(np.nan_to_num(np.float32(y), nan=0.0, posinf=0.0, neginf=0.0))
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __getstate__(self):
        # Never pickle memmaps/zarr arrays into worker processes ΓÇö
        # pickling materialises the full array on Windows (MemoryError).
        return {"cache_path": self.cache_path, "indices": self.indices,
                "use_zarr": self.use_zarr}

    def __setstate__(self, state):
        self.cache_path = state["cache_path"]
        self.indices    = state["indices"]
        self.use_zarr   = state.get("use_zarr", False)
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
    triggers ~242 separate chunk decompressions ΓÇö measured at 2ΓÇô49 s each.

    This class reads zarr in blocks aligned to the zarr chunk boundary so each
    physical chunk on disk is decompressed **exactly once** per epoch visit,
    regardless of batch size.  Rows within each block are shuffled in memory;
    the order in which blocks are visited is also shuffled each epoch.

    Multi-worker safety
    -------------------
    Each DataLoader worker receives a disjoint, contiguous slice of the sorted
    index array.  Because the index is sorted, each worker naturally owns
    different zarr chunks ΓÇö no locking or coordination needed.
    """

    def __init__(self, cache_path: str, indices: np.ndarray,
                 shuffle_chunks: bool = True, multitask_targets: bool = False,
                 return_indices: bool = False):
        self.cache_path     = cache_path
        # Sort once so contiguous positions map to the same zarr chunk.
        self.sorted_idx     = np.sort(np.asarray(indices, dtype=np.int64))
        self.shuffle_chunks = shuffle_chunks
        self.multitask_targets = bool(multitask_targets)
        self.return_indices = bool(return_indices)
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

    def __len__(self) -> int:
        return len(self.sorted_idx)

    def _open_arrays(self):
        import torch
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if getattr(self, '_opened_arrays', None) is None:
            self._opened_arrays = {}

        if worker_id not in self._opened_arrays:
            if self.use_zarr:
                z = _zarr_open_group(self.cache_path, mode="r")
                y_cls = z["y_cls"] if "y_cls" in z else None
                pq = z["pq"] if "pq" in z else None
                self._opened_arrays[worker_id] = (z["X"], z["y"], y_cls, pq, True)
            else:
                X = np.load(_x_path(self.cache_path), mmap_mode="r")
                y = np.load(_y_path(self.cache_path), mmap_mode="r")
                y_cls_p, pq_p = Path(_y_cls_path(self.cache_path)), Path(_pq_path(self.cache_path))
                y_cls = np.load(str(y_cls_p), mmap_mode="r") if y_cls_p.exists() else None
                pq = np.load(str(pq_p), mmap_mode="r") if pq_p.exists() else None
                self._opened_arrays[worker_id] = (X, y, y_cls, pq, False)

        return self._opened_arrays[worker_id]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        idx = self.sorted_idx

        # Assign each worker a contiguous slice of the sorted index so workers
        # naturally own different zarr chunks (no duplicated decompression).
        if worker_info is not None:
            n, wid = worker_info.num_workers, worker_info.id
            per    = (len(idx) + n - 1) // n
            idx    = idx[wid * per : (wid + 1) * per]

        if len(idx) == 0:
            return

        X_arr, y_arr, y_cls_arr, pq_arr, is_zarr = self._open_arrays()
        cs = self._zarr_row_chunk

        # Split sorted index into per-zarr-chunk blocks (all indices in a block
        # belong to the same physical chunk ΓåÆ exactly 1 decompression per block).
        chunk_nums   = idx // cs
        split_pts    = np.where(np.diff(chunk_nums))[0] + 1
        blocks       = np.split(idx, split_pts)

        if self.shuffle_chunks:
            np.random.shuffle(blocks)

        for block_idx in blocks:
            if is_zarr:
                X_blk = np.array(X_arr.oindex[block_idx], dtype=np.float32)
                y_blk = np.array(y_arr.oindex[block_idx], dtype=np.float32)
                yc_blk = (np.array(y_cls_arr.oindex[block_idx], dtype=np.float32)
                          if y_cls_arr is not None else None)
                pq_blk = (np.array(pq_arr.oindex[block_idx], dtype=np.float32)
                          if pq_arr is not None else None)
            else:
                X_blk = np.array(X_arr[block_idx], dtype=np.float32)
                y_blk = np.array(y_arr[block_idx], dtype=np.float32)
                yc_blk = (np.array(y_cls_arr[block_idx], dtype=np.float32)
                          if y_cls_arr is not None else None)
                pq_blk = (np.array(pq_arr[block_idx], dtype=np.float32)
                          if pq_arr is not None else None)

            # Sanitise: replace NaN/Inf with 0 so bad chunks can't poison the model
            np.nan_to_num(X_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(y_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if yc_blk is not None:
                np.nan_to_num(yc_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if pq_blk is not None:
                np.nan_to_num(pq_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            perm = np.random.permutation(len(X_blk))
            for j in perm:
                y_t = torch.tensor(float(y_blk[j]), dtype=torch.float32)
                if self.return_indices:
                    smp_idx = int(block_idx[j])
                    if self.multitask_targets:
                        yc = float(yc_blk[j]) if yc_blk is not None else float(y_blk[j])
                        pq = float(pq_blk[j]) if pq_blk is not None else min(1.0, abs(float(y_blk[j])))
                        yield (
                            torch.tensor(X_blk[j], dtype=torch.float32),
                            y_t,
                            torch.tensor(yc, dtype=torch.float32),
                            torch.tensor(pq, dtype=torch.float32),
                            torch.tensor(smp_idx, dtype=torch.long),
                        )
                    else:
                        yield (torch.tensor(X_blk[j], dtype=torch.float32), y_t,
                               torch.tensor(smp_idx, dtype=torch.long))
                else:
                    y_t = torch.tensor(float(y_blk[j]), dtype=torch.float32)
                    if self.multitask_targets:
                        yc = float(yc_blk[j]) if yc_blk is not None else float(y_blk[j])
                        pq = float(pq_blk[j]) if pq_blk is not None else min(1.0, abs(float(y_blk[j])))
                        yield (
                            torch.tensor(X_blk[j], dtype=torch.float32),
                            y_t,
                            torch.tensor(yc, dtype=torch.float32),
                            torch.tensor(pq, dtype=torch.float32),
                        )
                    else:
                        yield (torch.tensor(X_blk[j], dtype=torch.float32), y_t)

