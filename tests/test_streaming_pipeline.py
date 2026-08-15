"""Tests for the Bytewax streaming feature pipeline (current API)."""
from __future__ import annotations

import time

import pytest

from features.streaming_pipeline import (
    BYTEWAX_AVAILABLE,
    ComputedFeatures,
    MarketTick,
    StreamConfig,
    StreamingFeaturePipeline,
    _accumulate_ticks,
    _finalize_window_features,
    build_feature_pipeline,
    create_test_dataflow,
    parse_tick,
    tick_to_keyed,
)


@pytest.fixture
def sample_tick():
    return MarketTick(
        symbol="EURUSD",
        timestamp=int(time.time_ns()),
        bid=1.0800,
        ask=1.0802,
        bid_size=1.0,
        ask_size=1.0,
        trade_price=1.0801,
        trade_size=100000,
        trade_side="buy",
        exchange="test",
    )


@pytest.fixture
def stream_config():
    return StreamConfig(
        bootstrap_servers="localhost:9092",
        input_topic="market.ticks",
        consumer_group="test-group",
        window_size_sec=60,
        slide_sec=10,
        feature_store_type="console",
        num_partitions=1,
        worker_threads=1,
    )


def test_market_tick_creation(sample_tick):
    assert sample_tick.symbol == "EURUSD"
    assert sample_tick.mid_price == pytest.approx(1.0801)
    assert sample_tick.spread == pytest.approx(0.0002)
    d = sample_tick.to_dict()
    assert d["symbol"] == "EURUSD"
    assert d["bid"] == 1.0800


def test_computed_features_creation():
    features = ComputedFeatures(
        symbol="EURUSD",
        window_start=1000000000000000000,
        window_end=1000000060000000000,
        sma_20=1.0800,
        rsi_14=55.5,
        spread_mean=0.0002,
    )
    assert features.symbol == "EURUSD"
    assert features.sma_20 == 1.0800
    d = features.to_dict()
    assert d["symbol"] == "EURUSD"
    assert d["sma_20"] == 1.0800


def test_stream_config_defaults():
    config = StreamConfig()
    assert config.bootstrap_servers == "localhost:9092"
    assert config.window_size_sec == 60
    assert config.feature_store_type == "redis"


def test_parse_tick():
    event = {
        "symbol": "EURUSD",
        "timestamp": 1234567890,
        "bid": 1.08,
        "ask": 1.0802,
        "bid_size": 1.0,
        "ask_size": 1.5,
    }
    tick = parse_tick(event)
    assert tick.symbol == "EURUSD"
    assert tick.bid == 1.08
    assert tick.ask_size == 1.5


def test_tick_to_keyed(sample_tick):
    key, tick = tick_to_keyed(sample_tick)
    assert key == "EURUSD"
    assert tick is sample_tick


def test_accumulate_ticks(sample_tick):
    state = _accumulate_ticks(None, sample_tick)
    assert len(state) == 1
    state = _accumulate_ticks(state, sample_tick)
    assert len(state) == 2


def test_finalize_window_features(sample_tick):
    ticks = [sample_tick, sample_tick]
    item = ("EURUSD", (None, ticks))
    feats = _finalize_window_features(item)
    assert feats.symbol == "EURUSD"
    assert feats.tick_count == 2
    assert feats.sma_20 == pytest.approx(sample_tick.mid_price)
    assert feats.spread_mean == pytest.approx(sample_tick.spread)


def test_finalize_empty_window():
    feats = _finalize_window_features(("EURUSD", (None, [])))
    assert feats.symbol == "EURUSD"
    assert feats.tick_count == 0
    assert feats.sma_20 is None


@pytest.mark.skipif(not BYTEWAX_AVAILABLE, reason="bytewax not installed")
def test_streaming_pipeline_init(stream_config):
    pipeline = StreamingFeaturePipeline(stream_config)
    assert pipeline.config is stream_config
    assert pipeline.flow is not None


@pytest.mark.skipif(not BYTEWAX_AVAILABLE, reason="bytewax not installed")
def test_build_feature_pipeline(stream_config):
    flow = build_feature_pipeline(stream_config)
    assert flow is not None


@pytest.mark.skipif(not BYTEWAX_AVAILABLE, reason="bytewax not installed")
def test_create_test_dataflow(stream_config):
    flow = create_test_dataflow(stream_config)
    assert flow is not None


def test_config_with_all_options(stream_config):
    assert stream_config.feature_store_type == "console"
    assert stream_config.compute_technical is True
    assert stream_config.num_partitions == 1
