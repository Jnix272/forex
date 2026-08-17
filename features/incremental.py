"""
Incremental Feature Computation
===============================
Stateful incremental feature computation for streaming/online processing.
"""

from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from features.feature_engineering_pl import FeatureEngineer


@dataclass
class FeatureState:
    """State for incremental feature computation"""

    # EMA states
    ema_states: dict[str, float] = field(default_factory=dict)
    # Rolling window buffers
    rolling_buffers: dict[str, list[float]] = field(default_factory=dict)
    # Rolling statistics
    rolling_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    # Last timestamp processed
    last_timestamp: int | None = None
    # Last bar index processed
    last_bar_index: int = -1
    # Pair identifier
    pair: str = ""
    # Version for compatibility
    version: int = 1


class IncrementalFeatureEngine:
    """
    Incremental feature computation engine.

    Maintains state across batches to compute features incrementally
    without reprocessing historical data.
    """

    def __init__(
        self,
        feature_engineer: FeatureEngineer | None = None,
        state_dir: str | Path = "./feature_state",
        max_buffer_size: int = 10000,
    ):
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.max_buffer_size = max_buffer_size
        self.states: dict[str, FeatureState] = {}
        self._lock = threading.Lock()

    def _get_state_path(self, pair: str) -> Path:
        """Get state file path for a pair"""
        return self.state_dir / f"{pair}_feature_state.pkl"

    def load_state(self, pair: str) -> FeatureState:
        """Load feature state for a pair"""
        state_path = self._get_state_path(pair)

        with self._lock:
            if pair in self.states:
                return self.states[pair]

            if state_path.exists():
                try:
                    with open(state_path, "rb") as f:
                        state = pickle.load(f)
                    if not isinstance(state, FeatureState):
                        state = FeatureState(pair=pair)
                except Exception:
                    state = FeatureState(pair=pair)
            else:
                state = FeatureState(pair=pair)

            self.states[pair] = state
            return state

    def save_state(self, pair: str):
        """Save feature state for a pair"""
        state_path = self._get_state_path(pair)

        with self._lock:
            if pair in self.states:
                try:
                    with open(state_path, "wb") as f:
                        pickle.dump(self.states[pair], f)
                except Exception as e:
                    print(f"[IncrementalFeatureEngine] Failed to save state for {pair}: {e}")

    def compute_incremental(
        self,
        bars: pl.DataFrame,
        pair: str,
        warmup_bars: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """
        Compute features incrementally.

        Args:
            bars: New bars to process
            pair: Currency pair
            warmup_bars: Optional warmup bars for cold start

        Returns:
            DataFrame with computed features
        """
        if len(bars) == 0:
            return pl.DataFrame()

        state = self.load_state(pair)

        # Combine warmup + new bars if provided
        if warmup_bars is not None and len(warmup_bars) > 0:
            # Check for overlap
            combined = pl.concat([warmup_bars, bars], how="vertical_relaxed")
            combined = combined.unique(subset=["timestamp_utc"], maintain_order=True)
            combined = combined.sort("timestamp_utc")
            bars = combined

        # Compute features using the feature engineer
        features = self.feature_engineer.build(bars, pair=pair)

        # If we had warmup, slice off the warmup portion
        if warmup_bars is not None and len(warmup_bars) > 0:
            features = features.slice(len(warmup_bars))

        # Update state with new data
        self._update_state(state, bars, features)

        # Save state
        self.save_state(pair)

        return features

    def _update_state(self, state: FeatureState, bars: pl.DataFrame, features: pl.DataFrame):
        """Update incremental state with new data"""
        # Update last timestamp
        if "timestamp_utc" in bars.columns and len(bars) > 0:
            ts = bars["timestamp_utc"].max()
            if ts is not None:
                if isinstance(ts, (int, np.integer)):
                    state.last_timestamp = int(ts)
                elif isinstance(ts, (float, np.floating)):
                    state.last_timestamp = int(float(ts))
                elif isinstance(ts, str):
                    state.last_timestamp = int(pd.Timestamp(ts).value)
                else:
                    try:
                        if isinstance(ts, (pd.Timestamp, np.datetime64)):
                            state.last_timestamp = int(pd.Timestamp(ts).value)
                        elif isinstance(ts, str):
                            state.last_timestamp = int(pd.Timestamp(ts).value)
                        else:
                            state.last_timestamp = None
                    except Exception:
                        state.last_timestamp = None

        # Update rolling buffers for key features
        key_features = [
            "close",
            "high",
            "low",
            "volume",
            "atr_6",
            "atr_20",
            "rsi_14",
            "macd",
            "ofi",
            "vpin",
            "spread_avg",
        ]

        for feat in key_features:
            if feat in features.columns:
                vals = features[feat].drop_nulls().to_list()
                if vals:
                    if feat not in state.rolling_buffers:
                        state.rolling_buffers[feat] = []
                    state.rolling_buffers[feat].extend(vals)
                    # Trim buffer
                    if len(state.rolling_buffers[feat]) > self.max_buffer_size:
                        state.rolling_buffers[feat] = state.rolling_buffers[feat][-self.max_buffer_size :]

                    # Update rolling stats
                    arr = np.array(state.rolling_buffers[feat])
                    state.rolling_stats[feat] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "count": len(arr),
                    }

        # Update EMA states
        ema_features = ["ema_12", "ema_26", "ema_50"]
        for feat in ema_features:
            if feat in features.columns:
                vals = features[feat].drop_nulls().to_list()
                if vals:
                    state.ema_states[feat] = vals[-1]

    def get_state_summary(self, pair: str) -> dict[str, Any]:
        """Get summary of current state"""
        state = self.load_state(pair)

        return {
            "pair": pair,
            "last_timestamp": state.last_timestamp,
            "last_bar_index": state.last_bar_index,
            "buffer_sizes": {k: len(v) for k, v in state.rolling_buffers.items()},
            "rolling_stats": state.rolling_stats,
            "ema_states": state.ema_states,
        }

    def reset_state(self, pair: str):
        """Reset state for a pair"""
        with self._lock:
            self.states[pair] = FeatureState(pair=pair)
            state_path = self._get_state_path(pair)
            if state_path.exists():
                state_path.unlink()


class StreamingFeatureProcessor:
    """
    Streaming feature processor for real-time feature computation.

    Processes bars one at a time or in small batches, maintaining
    all necessary state for incremental computation.
    """

    def __init__(
        self,
        feature_engineer: FeatureEngineer | None = None,
        state_dir: str | Path = "./feature_state",
        warmup_bars: int = 200,  # Bars needed for warmup
    ):
        self.engine = IncrementalFeatureEngine(feature_engineer, state_dir)
        self.warmup_bars = warmup_bars
        self.pending_bars: dict[str, list[pl.DataFrame]] = {}
        self._lock = threading.Lock()

    def process_bar(self, bar: pl.DataFrame, pair: str) -> pl.DataFrame | None:
        """
        Process a single bar (or small batch) and return features.

        Returns None if not enough data for reliable features yet.
        """
        with self._lock:
            if pair not in self.pending_bars:
                self.pending_bars[pair] = []

            self.pending_bars[pair].append(bar)

            # Check if we have enough bars
            total_bars = sum(len(b) for b in self.pending_bars[pair])

            if total_bars < self.warmup_bars:
                return None  # Still warming up

            # Combine pending bars
            combined = pl.concat(self.pending_bars[pair], how="vertical_relaxed")
            combined = combined.unique(subset=["timestamp_utc"], maintain_order=True)
            combined = combined.sort("timestamp_utc")

            # Keep only recent bars (warmup + new)
            if len(combined) > self.warmup_bars * 2:
                combined = combined.tail(self.warmup_bars * 2)

            # Split into warmup and new
            warmup = combined.head(self.warmup_bars)
            new_bars = combined.tail(len(combined) - self.warmup_bars)

            if len(new_bars) == 0:
                return None

            # Compute incremental features
            features = self.engine.compute_incremental(new_bars, pair, warmup_bars=warmup)

            # Update pending bars - keep warmup + any unprocessed
            self.pending_bars[pair] = [combined.tail(self.warmup_bars)]

            return features

    def flush(self, pair: str) -> pl.DataFrame | None:
        """Flush remaining bars for a pair"""
        with self._lock:
            if pair not in self.pending_bars or len(self.pending_bars[pair]) == 0:
                return None

            combined = pl.concat(self.pending_bars[pair], how="vertical_relaxed")
            combined = combined.unique(subset=["timestamp_utc"], maintain_order=True)
            combined = combined.sort("timestamp_utc")

            if len(combined) < self.warmup_bars:
                return None

            warmup = combined.head(self.warmup_bars)
            new_bars = combined.tail(len(combined) - self.warmup_bars)

            features = self.engine.compute_incremental(new_bars, pair, warmup_bars=warmup)

            self.pending_bars[pair] = []

            return features

    def get_pending_count(self, pair: str) -> int:
        """Get number of pending bars"""
        with self._lock:
            if pair not in self.pending_bars:
                return 0
            return sum(len(b) for b in self.pending_bars[pair])


class FeatureStateStore:
    """
    Redis-backed feature state store for distributed processing.

    Provides persistent, shared state across multiple workers.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "forex:feature_state:",
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None
        self._connect()

    def _connect(self):
        """Connect to Redis"""
        try:
            import redis  # pyright: ignore[reportMissingImports]

            self._redis = redis.from_url(self.redis_url, decode_responses=False)
            self._redis.ping()
        except Exception as e:
            print(f"[FeatureStateStore] Redis connection failed: {e}")
            self._redis = None

    def _get_key(self, pair: str) -> str:
        return f"{self.key_prefix}{pair}"

    def save(self, pair: str, state: FeatureState):
        """Save state to Redis"""
        if self._redis is None:
            return

        try:
            data = pickle.dumps(state)
            self._redis.set(self._get_key(pair), data)
        except Exception as e:
            print(f"[FeatureStateStore] Failed to save state for {pair}: {e}")

    def load(self, pair: str) -> FeatureState | None:
        """Load state from Redis"""
        if self._redis is None:
            return None

        try:
            data = self._redis.get(self._get_key(pair))
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"[FeatureStateStore] Failed to load state for {pair}: {e}")
        return None

    def delete(self, pair: str):
        """Delete state from Redis"""
        if self._redis is None:
            return

        try:
            self._redis.delete(self._get_key(pair))
        except Exception as e:
            print(f"[FeatureStateStore] Failed to delete state for {pair}: {e}")

    def exists(self, pair: str) -> bool:
        """Check if state exists"""
        if self._redis is None:
            return False

        try:
            return self._redis.exists(self._get_key(pair)) > 0
        except Exception:
            return False


# Convenience function for creating incremental processor
def create_incremental_processor(
    feature_engineer: FeatureEngineer | None = None,
    state_dir: str | Path = "./feature_state",
    warmup_bars: int = 200,
) -> StreamingFeatureProcessor:
    """Create a streaming feature processor"""
    return StreamingFeatureProcessor(
        feature_engineer=feature_engineer,
        state_dir=state_dir,
        warmup_bars=warmup_bars,
    )


# Integration with existing FeatureEngineer
def add_incremental_support(feature_engineer: FeatureEngineer) -> IncrementalFeatureEngine:
    """Add incremental support to existing FeatureEngineer"""
    return IncrementalFeatureEngine(feature_engineer=feature_engineer)
