"""Neutral multi-timeframe tensor utilities.

Moved from ``training.dataset_builder`` so that the models layer no longer
needs to depend on the training layer: both ``models.ensemble`` and
``training.dataset_builder`` import these helpers from here (dependency
direction: training -> models/common, never models -> training).

The algorithm is **lookahead-free**: each coarser bar uses only the last
1-min bar within its completed window.
"""
from __future__ import annotations

import numpy as np


def build_multitf_tensors(
    X_seq: np.ndarray,
    base_tf_minutes: int = 1,
    target_tfs: list[int] | None = None,
) -> list[np.ndarray]:
    """Downsample a 1-min ``X_seq`` into coarser timeframe views.

    Produces a list of arrays that can be passed directly to
    ``MultiTimeframeAttention.forward(x_list)``.  The algorithm is
    **lookahead-free**: each coarser bar uses only the last 1-min bar within
    its completed window, so a 5-min bar at index *i* corresponds to 1-min
    bars ``[i-4, i]`` -- all of which have already closed by the time the
    sequence ends.

    Parameters
    ----------
    X_seq : np.ndarray
        Shape ``(N, T, F)`` -- the standard sliding-window feature tensor
        produced by ``dataset_builder``.  Each sample ``X_seq[n]`` is a
        sequence of ``T`` 1-min bars ending at time ``t_n``.
    base_tf_minutes : int
        The bar frequency of ``X_seq`` in minutes (default 1).
    target_tfs : list[int]
        Target timeframes in minutes.  Defaults to ``[1, 5, 15]``.
        Values must be multiples of ``base_tf_minutes``.

    Returns
    -------
    list[np.ndarray]
        One array per target timeframe, ordered as ``target_tfs``.
        - ``[0]`` shape ``(N, T,       F)``  -- unchanged 1-min view
        - ``[1]`` shape ``(N, T//5,    F)``  -- 5-min view (last bar of each 5)
        - ``[2]`` shape ``(N, T//15,   F)``  -- 15-min view
    """
    if target_tfs is None:
        target_tfs = [base_tf_minutes, 5, 15]

    _N, T, _F = X_seq.shape
    result = []

    for tf in target_tfs:
        stride = max(1, tf // base_tf_minutes)
        if stride == 1:
            result.append(X_seq.astype(np.float32, copy=False))
            continue

        # Take every stride-th bar starting from the last bar (index T-1)
        # working backwards, then reverse so time is ascending.
        # This picks the LAST bar of each completed coarser window -- no lookahead.
        coarse_indices = list(range(T - 1, -1, -stride))[::-1]
        if not coarse_indices:
            coarse_indices = [T - 1]

        coarse = X_seq[:, coarse_indices, :]  # (N, n_coarse, F)
        result.append(np.ascontiguousarray(coarse, dtype=np.float32))

    return result


def build_multitf_dataset(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    base_tf_minutes: int = 1,
    target_tfs: list[int] | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Convenience wrapper: returns ``(tf_views, labels)`` ready for training.

    The first element of ``tf_views`` is the full-resolution sequence.
    Subsequent elements are downsampled coarser-timeframe views.

    Usage with MultiTimeframeAttention::

        tf_views, y = build_multitf_dataset(X_seq, y_seq)
        import torch
        x_list = [torch.from_numpy(v) for v in tf_views]
        model = MultiTimeframeAttention(input_size=F)
        pred = model(x_list)
    """
    if target_tfs is None:
        target_tfs = [base_tf_minutes, 5, 15]
    views = build_multitf_tensors(X_seq, base_tf_minutes=base_tf_minutes, target_tfs=target_tfs)
    return views, y_seq
