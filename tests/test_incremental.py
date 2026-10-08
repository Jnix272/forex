"""
Tests for Incremental Feature Engine and FeatureStateStore JSON serialization.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from features.incremental import (
    FeatureState,
    FeatureStateStore,
    IncrementalFeatureEngine,
)


class MockRedis:
    def __init__(self):
        self.store = {}

    def ping(self):
        return True

    def set(self, key, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        self.store.pop(key, None)

    def exists(self, key):
        return 1 if key in self.store else 0


class TestFeatureStateSerialization:
    def test_to_dict_and_from_dict(self):
        state = FeatureState(
            ema_states={"ema_12": 1.1234},
            rolling_buffers={"close": [1.1, 1.2, 1.3]},
            rolling_stats={"close": {"mean": 1.2}},
            last_timestamp=1700000000,
            last_bar_index=10,
            pair="EURUSD",
            version=2,
        )

        as_dict = state.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["pair"] == "EURUSD"
        assert as_dict["ema_states"] == {"ema_12": 1.1234}

        restored = FeatureState.from_dict(as_dict)
        assert restored == state

    def test_from_dict_ignores_extra_fields(self):
        data = {
            "pair": "GBPUSD",
            "version": 2,
            "future_unknown_field": "some_value",
        }
        state = FeatureState.from_dict(data)
        assert state.pair == "GBPUSD"
        assert state.version == 2
        assert not hasattr(state, "future_unknown_field")


class TestIncrementalFeatureEnginePersistence:
    def test_save_and_load_state_json(self, tmp_path):
        engine = IncrementalFeatureEngine(state_dir=tmp_path)
        pair = "EURUSD"

        state = engine.load_state(pair)
        state.ema_states["ema_12"] = 1.0850
        state.last_timestamp = 1600000000
        engine.save_state(pair)

        # Check JSON file written
        json_path = tmp_path / f"{pair}_feature_state.json"
        assert json_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        assert raw_data["pair"] == pair
        assert raw_data["ema_states"]["ema_12"] == 1.0850

        # Create new engine instance and reload state
        engine2 = IncrementalFeatureEngine(state_dir=tmp_path)
        loaded_state = engine2.load_state(pair)

        assert loaded_state.ema_states["ema_12"] == 1.0850
        assert loaded_state.last_timestamp == 1600000000

    def test_reset_state(self, tmp_path):
        engine = IncrementalFeatureEngine(state_dir=tmp_path)
        pair = "EURUSD"

        state = engine.load_state(pair)
        state.last_timestamp = 100
        engine.save_state(pair)

        engine.reset_state(pair)
        json_path = tmp_path / f"{pair}_feature_state.json"
        assert not json_path.exists()

        new_state = engine.load_state(pair)
        assert new_state.last_timestamp is None


class TestFeatureStateStoreRedisJSON:
    def test_save_and_load_state_with_mock_redis(self):
        mock_redis = MockRedis()
        store = FeatureStateStore()
        store._redis = mock_redis

        pair = "EURUSD"
        state = FeatureState(
            ema_states={"ema_12": 1.0850},
            rolling_buffers={"close": [1.08, 1.085]},
            last_timestamp=1700000000,
            pair=pair,
            version=2,
        )

        store.save(pair, state)

        # Verify Redis value is JSON string/bytes, not pickled bytes
        key = store._get_key(pair)
        raw_val = mock_redis.get(key)
        assert raw_val is not None

        # Should be valid JSON
        if isinstance(raw_val, bytes):
            raw_val = raw_val.decode("utf-8")
        parsed = json.loads(raw_val)
        assert parsed["pair"] == pair
        assert parsed["ema_states"]["ema_12"] == 1.0850

        # Load back via FeatureStateStore
        loaded = store.load(pair)
        assert loaded is not None
        assert loaded.pair == pair
        assert loaded.ema_states["ema_12"] == 1.0850
        assert loaded.rolling_buffers["close"] == [1.08, 1.085]

    def test_load_bytes_from_redis(self):
        mock_redis = MockRedis()
        store = FeatureStateStore()
        store._redis = mock_redis

        pair = "USDJPY"
        state = FeatureState(pair=pair, last_timestamp=12345)
        raw_json_bytes = json.dumps(state.to_dict()).encode("utf-8")
        mock_redis.set(store._get_key(pair), raw_json_bytes)

        loaded = store.load(pair)
        assert loaded is not None
        assert loaded.pair == pair
        assert loaded.last_timestamp == 12345

    def test_exists_and_delete(self):
        mock_redis = MockRedis()
        store = FeatureStateStore()
        store._redis = mock_redis

        pair = "AUDUSD"
        assert not store.exists(pair)

        store.save(pair, FeatureState(pair=pair))
        assert store.exists(pair)

        store.delete(pair)
        assert not store.exists(pair)
