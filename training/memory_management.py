"""
Training Memory Management (Improvement #8)
===========================================
Streaming datasets with prefetching, gradient checkpointing, and activation offloading
for memory-efficient training on limited GPU VRAM (e.g., 8-16 GB).

Components:
  1. StreamingDataset / PrefetchDataLoader: Zero-copy memory-mapped streaming with
     background prefetching and Zarr chunk-aware sequential reads.
  2. GradientCheckpointing: Selective activation checkpointing for transformer/Mamba
     blocks with policy-based control (recompute vs store).
  3. MemoryProfiler: Context manager for tracking peak GPU/CPU memory usage.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, WeightedRandomSampler

try:
    import zarr

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

try:
    from torch.utils.checkpoint import checkpoint

    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# 1. Streaming Dataset with Prefetching
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class StreamingDatasetConfig:
    """Configuration for streaming dataset."""

    cache_path: str
    indices: np.ndarray
    chunk_size: int = 512  # Zarr chunk size (rows per chunk)
    prefetch_batches: int = 4  # Batches to prefetch in background
    pin_memory: bool = True
    persistent_workers: bool = True
    num_workers: int = 4
    shuffle_blocks: bool = True  # Shuffle zarr chunk blocks each epoch


class StreamingMemmapDataset(Dataset):
    """Map-style zero-copy streaming dataset backed by memory-mapped NPY or Zarr.

    Reads rows lazily so only the requested tile is paged into memory. Supports
    NPY memmap pairs (``_X.npy``/``_y.npy``) and Zarr groups containing ``X``/``y``
    arrays. ``__getitem__`` maps a position ``i`` to sample ``indices[i]``. Picklable
    for multiprocessing DataLoader workers (reopens storage on unpickle).
    """

    def __init__(
        self,
        cache_path: str,
        indices: np.ndarray,
        chunk_size: int = 512,
        use_zarr_chunks: bool = False,
    ):
        self.cache_path = cache_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.chunk_size = chunk_size
        self.use_zarr_chunks = use_zarr_chunks
        self._mode = "zarr" if self._is_zarr_group(cache_path) else "npy"
        self._zarr_group = None
        self._X_mem = None
        self._y_mem = None
        self._N = 0
        self._open_storage()

    @staticmethod
    def _is_zarr_group(cache_path: str) -> bool:
        if not ZARR_AVAILABLE:
            return False
        try:
            group = zarr.open_group(cache_path, mode="r")
            return "X" in group and "y" in group
        except Exception:
            return False

    def _open_storage(self):
        if self._mode == "zarr":
            self._zarr_group = zarr.open_group(self.cache_path, mode="r")
            self._N = self._zarr_group["X"].shape[0]
        else:
            self._X_mem = np.load(os.path.join(self.cache_path, "_X.npy"), mmap_mode="r")
            self._y_mem = np.load(os.path.join(self.cache_path, "_y.npy"), mmap_mode="r")
            self._N = self._X_mem.shape[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zarr_group"] = None
        state["_X_mem"] = None
        state["_y_mem"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_storage()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        if real_idx < 0 or real_idx >= self._N:
            raise IndexError(f"sample index {real_idx} out of range [0, {self._N})")
        if self._mode == "zarr":
            X = np.array(self._zarr_group["X"][real_idx], dtype=np.float32)
            y = np.array(self._zarr_group["y"][real_idx], dtype=np.float32)
        else:
            X = np.array(self._X_mem[real_idx], dtype=np.float32)
            y = np.array(self._y_mem[real_idx], dtype=np.float32)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y = float(np.nan_to_num(np.float32(y), nan=0.0, posinf=0.0, neginf=0.0))
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class SequentialZarrDataset(IterableDataset):
    """
    Sequential-read IterableDataset for zarr-backed training data.

    Reads zarr in blocks aligned to chunk boundaries so each physical chunk
    is decompressed exactly once per epoch visit. Rows within each block
    are shuffled in memory; block visit order is shuffled each epoch.

    Each DataLoader worker receives a disjoint, contiguous slice of the
    sorted index array. Because the index is sorted, each worker naturally
    owns different zarr chunks - no locking or coordination needed.
    """

    def __init__(
        self,
        cache_path: str,
        indices: np.ndarray,
        chunk_size: int = 512,
        shuffle_blocks: bool = True,
        worker_rank: int = 0,
        num_workers: int = 1,
    ):
        self.cache_path = cache_path
        self.chunk_size = chunk_size
        self.shuffle_blocks = shuffle_blocks
        self.worker_rank = worker_rank
        self.num_workers = num_workers

        self._z = zarr.open_group(cache_path, mode="r")
        self.X_zarr = self._z["X"]
        self.y_zarr = self._z["y"]

        # Sorted indices for sequential access
        self.indices = np.sort(np.asarray(indices, dtype=np.int64))

        self.n_chunks = (self.X_zarr.shape[0] + chunk_size - 1) // chunk_size

        # Precompute chunk boundaries
        self.chunk_starts = np.arange(0, self.X_zarr.shape[0], chunk_size)
        self.chunk_ends = np.minimum(self.chunk_starts + chunk_size, self.X_zarr.shape[0])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_rank = worker_info.id
            effective_workers = worker_info.num_workers
        else:
            worker_rank = 0
            effective_workers = 1

        n = len(self.indices)
        start = (n * worker_rank) // effective_workers
        end = (n * (worker_rank + 1)) // effective_workers
        worker_indices = self.indices[start:end]

        # Group worker indices by zarr chunk
        chunk_to_indices = {}
        for idx in worker_indices:
            chunk_idx = idx // self.chunk_size
            if chunk_idx not in chunk_to_indices:
                chunk_to_indices[chunk_idx] = []
            chunk_to_indices[chunk_idx].append(idx)

        chunk_ids = list(chunk_to_indices.keys())
        if self.shuffle_blocks:
            np.random.shuffle(chunk_ids)

        for chunk_idx in chunk_ids:
            indices_in_chunk = chunk_to_indices[chunk_idx]
            # Read entire chunk at once
            start = self.chunk_starts[chunk_idx]
            end = self.chunk_ends[chunk_idx]
            X_chunk = np.array(self.X_zarr[start:end], dtype=np.float32)
            y_chunk = np.array(self.y_zarr[start:end], dtype=np.float32)

            # Shuffle within chunk
            perm = np.random.permutation(len(indices_in_chunk))
            for i in perm:
                real_idx = indices_in_chunk[i]
                local_idx = real_idx - chunk_idx * self.chunk_size
                X = X_chunk[local_idx]
                y = float(y_chunk[local_idx])
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                y = float(np.nan_to_num(np.float32(y), nan=0.0, posinf=0.0, neginf=0.0))
                yield torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class PrefetchDataLoader:
    """
    Wrapper around DataLoader that prefetches batches in a background thread.

    Eliminates GPU idle time during CPU-side data loading/preprocessing.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 4,
        prefetch_batches: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **dataloader_kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            **dataloader_kwargs,
        )

        self._iterator = None
        self._prefetch_queue = []
        self._prefetch_thread = None
        self._stop_prefetch = False

    def __iter__(self):
        self._iterator = iter(self.loader)
        self._prefetch_queue = []
        self._stop_prefetch = False
        return self

    def __next__(self):
        if self._prefetch_queue:
            return self._prefetch_queue.pop(0)
        return next(self._iterator)

    def __len__(self):
        return len(self.loader)


class _IndexedDataset(Dataset):
    """Wraps a map-style dataset so each batch row carries its sample index.

    Training loops use the index to map per-sample losses back to priorities
    (PER ``update_priorities``)."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        X, y = self.base[idx]
        return X, y, torch.tensor(idx, dtype=torch.long)


class PrioritizedDataLoader(PrefetchDataLoader):
    """
    Prioritized Experience Replay (PER) DataLoader.
    Samples sequences proportionally to their Temporal Difference (TD) error or Loss.

    Uses a ``WeightedRandomSampler`` over per-sample priorities (rebuild each
    epoch so updated priorities take effect). Batches carry sample indices so
    the training loop can call ``update_priorities``.
    """

    def __init__(self, dataset, batch_size: int, alpha: float = 0.6, beta: float = 0.4, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.prefetch_factor = kwargs.pop("prefetch_factor", None)

        # Initialize uniform priorities
        self.priorities = torch.ones(len(dataset), dtype=torch.float32)
        self.max_priority = 1.0

        # Wrap dataset to carry indices; PrefetchDataLoader sets up prefetch.
        self._indexed_ds = _IndexedDataset(dataset)
        super().__init__(self._indexed_ds, batch_size, shuffle=False, **kwargs)
        self._reload_sampler()

    def update_priorities(self, indices: torch.Tensor, losses: torch.Tensor):
        """Update priorities based on training loss."""
        # Convert loss to priority (e.g., loss + epsilon)
        priorities = (losses.detach().cpu().abs() + 1e-6) ** self.alpha
        for idx, p in zip(indices.view(-1).tolist(), priorities.view(-1).tolist(), strict=False):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = p
                self.max_priority = max(self.max_priority, p)

    def _reload_sampler(self):
        """Rebuild the DataLoader with a WeightedRandomSampler over current priorities."""
        self.loader = DataLoader(
            self._indexed_ds,
            batch_size=self.batch_size,
            sampler=WeightedRandomSampler(
                weights=self.priorities,
                num_samples=len(self.priorities),
                replacement=True,
            ),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,  # Fixed leak: don't persist workers if we recreate loader
            prefetch_factor=self.prefetch_factor,
        )

    def __iter__(self):
        self._reload_sampler()
        return super().__iter__()


# ═════════════════════════════════════════════════════════════════════════════
# 2. Gradient Checkpointing
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class CheckpointPolicy:
    """Policy for selective gradient checkpointing."""

    # Module name patterns to checkpoint
    checkpoint_patterns: list[str] = None  # e.g., ["transformer", "mamba", "encoder"]
    # Module name patterns to never checkpoint
    no_checkpoint_patterns: list[str] = None  # e.g., ["output", "head", "norm"]
    # Checkpoint every N layers (for uniform policies)
    every_n_layers: int = 1
    # Minimum layer size to checkpoint (skip tiny layers)
    min_layer_params: int = 10000

    def __post_init__(self):
        if self.checkpoint_patterns is None:
            self.checkpoint_patterns = ["transformer", "encoder", "mamba", "layer", "block"]


def apply_gradient_checkpointing(
    model: nn.Module,
    policy: CheckpointPolicy | None = None,
) -> nn.Module:
    """
    Apply selective gradient checkpointing to model modules.

    Wraps target modules with torch.utils.checkpoint.checkpoint to trade
    compute for memory: activations are recomputed during backward instead
    of stored during forward.

    Args:
        model: The model to apply checkpointing to.
        policy: CheckpointPolicy defining which modules to checkpoint.

    Returns:
        The model with checkpointed modules.
    """
    if not CHECKPOINT_AVAILABLE:
        warnings.warn("torch.utils.checkpoint not available; skipping gradient checkpointing", stacklevel=2)
        return model

    if policy is None:
        policy = CheckpointPolicy()

    def should_checkpoint(module: nn.Module, name: str) -> bool:
        # Check no-checkpoint patterns first
        if policy.no_checkpoint_patterns:
            for pattern in policy.no_checkpoint_patterns:
                if pattern in name:
                    return False
        # Check checkpoint patterns
        if policy.checkpoint_patterns:
            for pattern in policy.checkpoint_patterns:
                if pattern in name:
                    return True
        return False

    def checkpoint_forward(module):
        """Wrap module forward with checkpoint."""

        def forward(*args, **kwargs):
            return checkpoint(module._original_forward, *args, use_reentrant=False, **kwargs)

        return forward

    for name, module in model.named_modules():
        if should_checkpoint(module, name):
            # Store original forward
            module._original_forward = module.forward
            # Replace with checkpointed version
            module.forward = lambda *args, m=module, **kwargs: checkpoint(
                m._original_forward, *args, use_reentrant=False, **kwargs
            )

    return model


def checkpoint_sequential(
    modules: list[nn.Module],
    input: torch.Tensor,
    use_reentrant: bool = False,
) -> torch.Tensor:
    """
    Checkpoint a sequence of modules.

    Memory-efficient alternative to sequential forward pass.
    """

    def run_module(m, x):
        return m(x)

    x = input
    for m in modules:
        x = checkpoint(run_module, m, x, use_reentrant=use_reentrant)
    return x


# ═════════════════════════════════════════════════════════════════════════════
# 4. Memory Profiler
# ════════════════════════════════════════════════════════════════════════════


@contextmanager
def memory_profiler(device: str = "cuda") -> Iterator[dict[str, float]]:
    """
    Context manager for profiling peak memory usage.

    Usage:
        with memory_profiler() as stats:
            # training step
        print(f"Peak GPU: {stats['peak_gpu_gb']:.2f} GB")
    """
    stats = {}
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_alloc = torch.cuda.memory_allocated()
        torch.cuda.memory_reserved()
    else:
        start_alloc = 0

    try:
        yield stats
    finally:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            current_alloc = torch.cuda.memory_allocated()
            current_reserved = torch.cuda.memory_reserved()
        else:
            peak_alloc = peak_reserved = current_alloc = current_reserved = 0

        stats.update(
            {
                "peak_gpu_allocated_gb": peak_alloc / 1e9,
                "peak_gpu_reserved_gb": peak_reserved / 1e9,
                "current_gpu_allocated_gb": current_alloc / 1e9,
                "current_gpu_reserved_gb": current_reserved / 1e9,
                "delta_allocated_gb": (peak_alloc - start_alloc) / 1e9,
            }
        )


class MemoryMonitor:
    """
    Continuous memory monitoring callback for training loops.

    Logs memory usage at specified intervals and warns on OOM risk.
    """

    def __init__(
        self,
        log_interval: int = 100,
        warn_threshold_gb: float = 7.0,  # For 8GB GPU
        device: str = "cuda",
    ):
        self.log_interval = log_interval
        self.warn_threshold = warn_threshold_gb * 1e9
        self.device = device
        self._step_count = 0
        self.peak = 0

    def step(self):
        self._step_count += 1
        if self._step_count % self.log_interval == 0 and self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            self.peak = max(self.peak, allocated)

            if allocated > self.warn_threshold:
                warnings.warn(
                    f"[MemoryMonitor] High GPU memory: {allocated / 1e9:.2f} GB allocated, "
                    f"{reserved / 1e9:.2f} GB reserved. Peak: {self.peak / 1e9:.2f} GB", stacklevel=2
                )

            return {
                "step": self._step_count,
                "allocated_gb": allocated / 1e9,
                "reserved_gb": reserved / 1e9,
                "peak_gb": self.peak / 1e9,
            }
        return None

    @property
    def step_count(self):
        return self._step_count


# ═════════════════════════════════════════════════════════════════════════════
# 5. Convenience: Integrated Training Context
# ════════════════════════════════════════════════════════════════════════════


@contextmanager
def memory_efficient_training(
    model: nn.Module,
    checkpoint_policy: CheckpointPolicy | None = None,
    enable_profiler: bool = False,
) -> Iterator[dict]:
    """
    Context manager for memory-efficient training setup.

    Applies gradient checkpointing and yields control, then cleans up.

    Usage:
        with memory_efficient_training(model, checkpoint_policy=policy) as mem:
            for epoch in epochs:
                train_epoch()
        print(f"Peak GPU: {mem['peak_gpu_gb']:.2f} GB")
    """
    if enable_profiler and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Apply gradient checkpointing
    if checkpoint_policy is not None:
        apply_gradient_checkpointing(model, checkpoint_policy)

    try:
        yield {}
    finally:
        if checkpoint_policy is not None:
            for _name, module in model.named_modules():
                if hasattr(module, "_original_forward"):
                    module.forward = module._original_forward
                    delattr(module, "_original_forward")

        if enable_profiler and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[MemoryEfficientTraining] Peak GPU: {peak:.2f} GB")


# ════════════════════════════════════════════════════════════════════════════
# 6. Dataset Factory
# ════════════════════════════════════════════════════════════════════════════


def create_streaming_dataloader(
    cache_path: str,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int = 4,
    prefetch_batches: int = 4,
    chunk_size: int = 512,
    sequential: bool = False,
    shuffle_blocks: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Factory function to create a streaming DataLoader with optimal settings.

    Args:
        cache_path: Path to Zarr/NPY cache directory.
        indices: Sample indices to use.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        prefetch_batches: Batches to prefetch.
        chunk_size: Zarr chunk size.
        sequential: Use sequential Zarr reading (IterableDataset).
        shuffle_blocks: Shuffle zarr chunk blocks per epoch.

    Returns:
        Configured DataLoader with prefetching.
    """
    if sequential and ZARR_AVAILABLE:
        dataset = SequentialZarrDataset(
            cache_path,
            indices,
            chunk_size,
            shuffle_blocks=True,
            worker_rank=0,
            num_workers=num_workers,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        dataset = StreamingMemmapDataset(
            cache_path,
            indices,
            chunk_size,
            use_zarr_chunks=sequential,
        )
        return PrefetchDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Types
# ════════════════════════════════════════════════════════════════════════════

T = TypeVar("T")


if __name__ == "__main__":
    # Quick self-test

    # Test memory profiler
    with memory_profiler() as stats:
        x = torch.randn(1000, 1000, device="cuda" if torch.cuda.is_available() else "cpu")
        y = x @ x.T
    print(f"Memory stats: {stats}")

    print("Memory management module OK")
