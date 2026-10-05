import json
import pickle
from pathlib import Path
import pytest

import features.finbert_sentiment as fs
from features.finbert_sentiment import SentimentPipeline, _load_cache, _save_cache, _cache_key


@pytest.fixture(autouse=True)
def reset_cache_singleton(tmp_path, monkeypatch):
    """Reset module level shared cache and point cache paths to temp directory."""
    monkeypatch.setattr(fs, "_SHARED_CACHE", None)
    monkeypatch.setattr(fs, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fs, "CACHE_FILE", tmp_path / "sentiment_cache.json")
    monkeypatch.setattr(fs, "_LEGACY_PKL_CACHE", tmp_path / "sentiment_cache.pkl")
    monkeypatch.setattr(fs, "_STALE_CACHE_FILE", tmp_path / "stale_sentiment_cache.pkl")
    monkeypatch.setattr(fs, "_STALE_JSON_CACHE_FILE", tmp_path / "stale_sentiment_cache.json")


def test_json_cache_save_and_load(tmp_path, monkeypatch):
    test_cache = {_cache_key("test headline"): 0.75}
    _save_cache(test_cache)

    assert fs.CACHE_FILE.exists()
    assert not fs.CACHE_FILE.name.endswith(".pkl")

    # Read raw content to ensure it is valid JSON
    with open(fs.CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == test_cache

    # Reset singleton and load via _load_cache
    monkeypatch.setattr(fs, "_SHARED_CACHE", None)
    loaded = _load_cache()
    assert loaded == test_cache


def test_legacy_pickle_migration(tmp_path, monkeypatch):
    legacy_pkl = fs._LEGACY_PKL_CACHE
    legacy_data = {_cache_key("legacy headline"): -0.5}

    with open(legacy_pkl, "wb") as f:
        pickle.dump(legacy_data, f)

    assert legacy_pkl.exists()

    # Load cache should auto-migrate legacy pkl
    loaded = _load_cache()
    assert loaded.get(_cache_key("legacy headline")) == -0.5
    assert not legacy_pkl.exists()


def test_sentiment_pipeline_caching(tmp_path):
    pipe = SentimentPipeline(prefer_backend="vader", use_cache=True, cache_save_every=1)
    scores = pipe.score_headlines_batch(["EUR/USD reaches new high"])
    assert len(scores) == 1

    # Force save cache and verify file
    pipe.flush_cache()
    assert fs.CACHE_FILE.exists()

    with open(fs.CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    key = _cache_key("EUR/USD reaches new high")
    assert key in data
    assert isinstance(data[key], float)
