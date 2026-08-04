"""
Tests for streaming feature pipeline (Bytewax-based).
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import streaming pipeline components
from features.streaming_pipeline import (
    StreamConfig,
    MarketTick,
    ComputedFeatures,
    FeatureState,
    SymbolState,
    StreamConfig,
    MarketTick,
    ComputedFeatures,
    FeatureState,
    SymbolState,
    FeatureSink,
    RedisFeatureSink,
    PostgresFeatureSink,
    ConsoleSink,
    create_sink,
    StreamingFeaturePipeline,
    parse_tick,
    tick_to_keyed,
    _accumulate_ticks,
    _finalize_window_features,
    _accumulate_microstructure,
    _finalize_microstructure,
    _detect_regime,
    build_feature_pipeline,
    create_test_dataflow,
    run_test_pipeline,
    stream_features_to_batch,
    register_streaming_features,
    compute_windowed_features,
    BYTEWAX_AVAILABLE,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_tick():
    """Create a sample market tick."""
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
def sample_ticks():
    """Create a list of sample ticks."""
    base_time = int(time.time_ns())
    ticks = []
    for i in range(50):
        tick = MarketTick(
            symbol="EURUSD",
            timestamp=base_time + i * 1_000_000_000,  # 1 second apart
            bid=1.0800 + np.random.normal(0, 0.0001),
            ask=1.0802 + np.random.normal(0, 0.0001),
            bid_size=1.0,
            ask_size=1.0,
            trade_price=1.0801 if i % 5 == 0 else None,
            trade_size=100000 if i % 5 == 0 else None,
            trade_side="buy" if i % 10 == 0 else ("sell" if i % 10 == 5 else None),
            exchange="test",
        )
        ticks.append(tick)
    return ticks


@pytest.fixture
def stream_config():
    """Create a test stream config."""
    return StreamConfig(
        bootstrap_servers="localhost:9092",
        input_topic="market.ticks",
        consumer_group="test-group",
        window_size_sec=60,
        slide_sec=10,
        watermark_delay_sec=30,
        state_backend="memory",
        state_dir="./test_state",
        checkpoint_interval_sec=60,
        num_partitions=1,
        worker_threads=1,
        compute_technical=True,
        compute_cross_asset=False,
        compute_regime=True,
        compute_microstructure=True,
        feature_store_type="console",
        redis_url="redis://localhost:6379/0",
        postgres_dsn="postgresql://user:pass@localhost:5432/features",
        output_topic="features.computed",
        metrics_interval_sec=30,
        enable_metrics=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Core Data Structures
# ════════════════════════════════════════════════════════════════════════════

def test_market_tick_creation():
    """Test MarketTick creation and serialization."""
    tick = MarketTick(
        symbol="EURUSD",
        timestamp=1234567890123456789,
        bid=1.0800,
        ask=1.0802,
        bid_size=1.0,
        ask_size=1.0,
    )
    
    assert tick.symbol == "EURUSD"
    assert tick.timestamp == 1234567890123456789
    assert tick.bid == 1.0800
    assert tick.ask == 1.0802
    assert tick.trade_price is None
    
    d = tick.to_dict()
    assert d["symbol"] == "EURUSD"
    assert d["bid"] == 1.0800


def test_computed_features_creation():
    """Test ComputedFeatures creation and serialization."""
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
    assert features.rsi_14 == 55.5
    
    d = features.to_dict()
    assert d["symbol"] == "EURUSD"
    assert d["sma_20"] == 1.0800
    assert "timestamp" not in d  # Not a field


def test_stream_config():
    """Test StreamConfig creation."""
    config = StreamConfig(
        bootstrap_servers="localhost:9092",
        input_topic="market.ticks",
        window_size_sec=60,
    )
    
    assert config.bootstrap_servers == "localhost:9092"
    assert config.window_size_sec == 60
    assert config.feature_store_type == "console"  # default


# ═════════════════════════════════════════════════════════════════════════════
# FeatureState Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_feature_state_basic():
    """Test FeatureState basic operations."""
    state = FeatureState("EURUSD", 60)
    
    # Add ticks
    base_time = int(time.time_ns())
    for i in range(30):
        tick = MarketTick(
            symbol="EURUSD",
            timestamp=int(time.time_ns()) + i * 1_000_000_000,
            bid=1.0800 + i * 0.0001,
            ask=1.0802 + i * 0.0001,
            bid_size=1.0,
            ask_size=1.0,
        )
        state.add_tick(tick)
    
    assert len(state.ticks) == 30
    
    # Test technical computation
    tech = state.compute_technical()
    assert "sma_5" in tech
    assert "sma_20" in tech
    assert "rsi_14" in tech


def test_feature_state_trade_handling():
    """Test FeatureState trade handling."""
    state = FeatureState("EURUSD", 60)
    
    base_time = int(time.time_ns())
    for i in range(20):
        tick = MarketTick(
            symbol="EURUSD",
            timestamp=int(time.time_ns()) + i * 1_000_000_000,
            bid=1.0800,
            ask=1.0802,
            bid_size=1.0,
            ask_size=1.0,
            trade_price=1.0801 if i % 5 == 0 else None,
            trade_size=100000 if i % 5 == 0 else None,
            trade_side="buy" if i % 10 == 0 else "sell",
        )
        state.add_tick(tick)
        if i % 5 == 0:
            state.add_trade(tick)
    
    micro = state.compute_microstructure()
    assert "spread_mean" in micro
    assert "order_flow_imbalance" in micro
    assert "trade_intensity" in micro


def test_feature_state_window_cleanup():
    """Test that old ticks are cleaned up."""
    state = FeatureState("EURUSD", 60)  # 60 second window
    
    base_time = int(time.time_ns()) - 100_000_000_000  # 100 seconds ago
    for i in range(10):
        tick = MarketTick(
            symbol="EURUSD",
            timestamp=base_time + i * 1_000_000_000,
            bid=1.0800,
            ask=1.0802,
            bid_size=1.0,
            ask_size=1.0,
        )
        state.add_tick(tick)
    
    # Add recent ticks
    recent_time = int(time.time_ns())
    for i in range(5):
        tick = MarketTick(
            symbol="EURUSD",
            timestamp=recent_time + i * 1_000_000_000,
            bid=1.0800,
            ask=1.0802,
            bid_size=1.0,
            ask_size=1.0,
        )
        state.add_tick(tick)
    
    # Old ticks should be cleaned up
    assert len(state.ticks) <= 5 + 10  # Some buffer
    # All remaining ticks should be recent
    for tick in state.ticks:
        assert tick["timestamp"] > int(time.time_ns()) - 70_000_000_000


# ═════════════════════════════════════════════════════════════════════════════
# Technical Indicator Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_ema_calculation():
    """Test EMA calculation."""
    state = FeatureState("TEST", 60)
    
    prices = np.array([1.0, 1.1, 1.2, 1.1, 1.3, 1.2, 1.4, 1.3, 1.5, 1.4])
    ema_5 = state._ema(prices, 5)
    
    # EMA should be between min and max
    assert ema_5 >= 1.0
    assert ema_5 <= 1.5
    
    # Test with insufficient data
    short_prices = np.array([1.0, 1.1])
    assert state._ema(short_prices, 5) is None


def test_rsi_calculation():
    """Test RSI calculation."""
    state = FeatureState("TEST", 60)
    
    # Rising prices -> high RSI
    prices = np.array([1.0 + i * 0.01 for i in range(20)])
    rsi = state._rsi(prices, 14)
    assert rsi > 70  # Strong uptrend
    
    # Falling prices -> low RSI
    prices_down = np.array([1.0 - i * 0.01 for i in range(20)])
    rsi_down = state._rsi(prices_down, 14)
    assert rsi_down < 30


def test_atr_calculation():
    """Test ATR calculation."""
    state = FeatureState("TEST", 60)
    
    # Need at least 14 data points for ATR(14)
    highs = np.array([1.01, 1.02, 1.015, 1.03, 1.025, 1.04, 1.035, 1.05, 1.045, 1.06, 1.055, 1.07, 1.065, 1.08, 1.075])
    lows = np.array([0.99, 0.98, 0.985, 0.97, 0.975, 0.96, 0.965, 0.95, 0.955, 1.04, 1.035, 1.06, 1.055, 1.07, 1.065])
    closes = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    atr = state._atr(highs, lows, closes)
    assert atr is not None
    assert atr > 0
    
    # Test insufficient data
    assert state._atr(np.array([1.0]), np.array([1.0]), np.array([1.0])) is None


# ═════════════════════════════════════════════════════════════════════════════
# SymbolState Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_symbol_state():
    """Test SymbolState."""
    state = SymbolState("EURUSD", 60)
    
    tick = MarketTick(
        symbol="EURUSD",
        timestamp=int(time.time_ns()),
        bid=1.0800,
        ask=1.0802,
        bid_size=1.0,
        ask_size=1.0,
    )
    
    result = state.process_tick(tick)
    assert result is None  # No window closed yet
    
    # Test window computation
    window_end = int(time.time_ns())
    features = state.compute_window(window_end)
    assert features.symbol == "EURUSD"
    assert features.window_end == window_end


# ═════════════════════════════════════════════════════════════════════════════
# Parser and Key Functions
# ═════════════════════════════════════════════════════════════════════════════

def test_parse_tick():
    """Test parse_tick function."""
    event = {
        "symbol": "EURUSD",
        "timestamp": 1234567890123456789,
        "bid": 1.0800,
        "ask": 1.0802,
        "bid_size": 1.0,
        "ask_size": 1.0,
        "trade_price": 1.0801,
        "trade_size": 100000,
        "trade_side": "buy",
        "exchange": "test",
    }
    
    tick = parse_tick(event)
    assert tick.symbol == "EURUSD"
    assert tick.timestamp == 1234567890123456789
    assert tick.bid == 1.0800
    assert tick.trade_price == 1.0801
    assert tick.trade_side == "buy"


def test_tick_to_keyed():
    """Test tick_to_keyed function."""
    tick = MarketTick(
        symbol="EURUSD",
        timestamp=int(time.time_ns()),
        bid=1.0800,
        ask=1.0802,
        bid_size=1.0,
        ask_size=1.0,
    )
    
    key, tick2 = tick_to_keyed(tick)
    assert key == "EURUSD"
    assert tick2.symbol == "EURUSD"


# ═════════════════════════════════════════════════════════════════════════════
# Window Functions
# ═════════════════════════════════════════════════════════════════════════════

def test_accumulate_ticks():
    """Test _accumulate_ticks."""
    tick = MarketTick(
        symbol="EURUSD",
        timestamp=int(time.time_ns()),
        bid=1.0800,
        ask=1.0802,
        bid_size=1.0,
        ask_size=1.0,
    )
    
    state = _accumulate_ticks(None, tick)
    assert len(state) == 1
    assert state[0].symbol == "EURUSD"
    
    # Add another
    tick2 = MarketTick(
        symbol="EURUSD",
        timestamp=int(time.time_ns()) + 1_000_000_000,
        bid=1.0801,
        ask=1.0803,
        bid_size=1.0,
        ask_size=1.0,
    )
    state = _accumulate_ticks(state, tick2)
    assert len(state) == 2


def test_finalize_window_features():
    """Test _finalize_window_features."""
    ticks = []
    base = int(time.time_ns())
    for i in range(20):
        ticks.append(MarketTick(
            symbol="EURUSD",
            timestamp=int(time.time_ns()) + i * 1_000_000_000,
            bid=1.0800 + i * 0.0001,
            ask=1.0802 + i * 0.0001,
            bid_size=1.0,
            ask_size=1.0,
        ))
    
    features = _finalize_window_features(ticks)
    assert features.symbol == "EURUSD"
    assert features.sma_20 is not None
    assert features.rsi_14 is not None


def test_finalize_microstructure():
    """Test _finalize_microstructure."""
    ticks = []
    for i in range(20):
        tick = MarketTick(
            symbol="EURUSD",
            timestamp=int(time.time_ns()) + i * 1_000_000_000,
            bid=1.0800,
            ask=1.0802,
            bid_size=1.0,
            ask_size=1.0,
            trade_price=1.0801 if i % 5 == 0 else None,
            trade_size=100000 if i % 5 == 0 else None,
            trade_side="buy" if i % 10 == 0 else "sell",
        )
        ticks.append(tick)
    
    features = _finalize_microstructure(ticks)
    assert features.symbol == "EURUSD"
    assert features.trade_count > 0


# ═════════════════════════════════════════════════════════════════════════════
# Regime Detection
# ═════════════════════════════════════════════════════════════════════════════

def test_detect_regime():
    """Test _detect_regime function."""
    features = ComputedFeatures(
        symbol="EURUSD",
        window_start=0,
        window_end=0,
        sma_20=1.0800,
        atr_14=0.0003,
        macd_hist=0.0001,
    )
    
    features = _detect_regime(features)
    
    assert features.volatility_regime is not None
    assert features.volatility_regime in [0, 1, 2]
    assert features.regime_prob is not None
    assert 0 <= features.regime_prob <= 1


# ═════════════════════════════════════════════════════════════════════════════
# Sink Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_console_sink():
    """Test ConsoleSink."""
    sink = ConsoleSink()
    features = ComputedFeatures(
        symbol="EURUSD",
        window_start=0,
        window_end=0,
        sma_20=1.0800,
        rsi_14=55.0,
        spread_mean=0.0002,
    )
    
    # Should not raise
    sink.write(features)
    sink.close()


def test_create_sink_console():
    """Test create_sink with console."""
    config = StreamConfig(feature_store_type="console")
    sink = create_sink(config)
    assert isinstance(sink, ConsoleSink)


def test_create_sink_redis_requires_redis():
    """Test RedisFeatureSink requires redis."""
    config = StreamConfig(feature_store_type="redis", redis_url="redis://localhost:6379/0")
    
    try:
        sink = create_sink(config)
        assert isinstance(sink, RedisFeatureSink)
    except ImportError:
        pytest.skip("redis not available")


# ═════════════════════════════════════════════════════════════════════════════
# StreamingFeaturePipeline
# ═════════════════════════════════════════════════════════════════════════════

# StreamingFeaturePipeline
@pytest.mark.skipif(not BYTEWAX_AVAILABLE, reason="Bytewax not available")
def test_streaming_pipeline_init():
    """Test StreamingFeaturePipeline initialization."""
    config = StreamConfig(
        feature_store_type="console",
        window_size_sec=60,
    )
    
    pipeline = StreamingFeaturePipeline(config)
    assert pipeline.config == config
    assert not pipeline._running


@pytest.mark.skipif(not BYTEWAX_AVAILABLE, reason="Bytewax not available")
def test_streaming_pipeline_run_stop():
    """Test pipeline run and stop."""
    config = StreamConfig(
        feature_store_type="console",
        window_size_sec=60,
    )
    
    pipeline = StreamingFeaturePipeline(config)
    pipeline.run(duration_sec=1)
    assert not pipeline._running


# ═════════════════════════════════════════════════════════════════════════════
# Dataflow Building
# ═════════════════════════════════════════════════════════════════════════════

def test_build_feature_pipeline():
    """Test build_feature_pipeline."""
    config = StreamConfig(
        window_size_sec=60,
        slide_sec=10,
    )
    
    flow = build_feature_pipeline(config)
    assert flow is not None


# ═════════════════════════════════════════════════════════════════════════════
# Stream to Batch
# ═════════════════════════════════════════════════════════════════════════════

def test_stream_features_to_batch():
    """Test stream_features_to_batch."""
    features = [
        ComputedFeatures(symbol="EURUSD", window_start=i, window_end=i+60, sma_20=1.08)
        for i in range(1500)
    ]
    
    batches = list(stream_features_to_batch(iter(features), batch_size=100))
    assert len(batches) == 15  # 1500 / 100


# ══════════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═════════════════════════════════════════════════════════════════════════════

def test_empty_window_features():
    """Test features with empty window."""
    features = _finalize_window_features([])
    assert features.symbol == ""
    assert features.window_start == 0
    assert features.window_end == 0


def test_microstructure_no_trades():
    """Test microstructure with no trades."""
    ticks = []
    for i in range(20):
        ticks.append(MarketTick(
            symbol="EURUSD",
            timestamp=int(time.time_ns()) + i * 1_000_000_000,
            bid=1.0800,
            ask=1.0802,
            bid_size=1.0,
            ask_size=1.0,
        ))
    
    features = _finalize_microstructure(ticks)
    assert features.trade_count == 0


def test_config_with_all_options():
    """Test StreamConfig with all options."""
    config = StreamConfig(
        bootstrap_servers="kafka1:9092,kafka2:9092",
        input_topic="market.ticks",
        consumer_group="feature-pipeline",
        window_size_sec=300,
        slide_sec=30,
        watermark_delay_sec=60,
        state_backend="rocksdb",
        state_dir="/mnt/state",
        checkpoint_interval_sec=120,
        num_partitions=8,
        worker_threads=8,
        compute_technical=True,
        compute_cross_asset=True,
        compute_regime=True,
        compute_microstructure=True,
        feature_store_type="redis",
        redis_url="redis://redis:6379/0",
        postgres_dsn="postgresql://user:pass@db:5432/features",
        output_topic="features.computed",
        metrics_interval_sec=60,
        enable_metrics=True,
    )
    
    assert config.bootstrap_servers == "kafka1:9092,kafka2:9092"
    assert config.num_partitions == 8
    assert config.compute_cross_asset is True


# ═════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_full_feature_computation():
    """Test full feature computation pipeline."""
    # Create ticks
    ticks = []
    base_time = int(time.time_ns())
    for i in range(100):
        tick = MarketTick(
            symbol="EURUSD",
            timestamp=int(time.time_ns()) + i * 1_000_000_000,
            bid=1.0800 + np.random.normal(0, 0.0001),
            ask=1.0802 + np.random.normal(0, 0.0001),
            bid_size=1.0,
            ask_size=1.0,
            trade_price=1.0801 if i % 5 == 0 else None,
            trade_size=100000 if i % 5 == 0 else None,
            trade_side="buy" if i % 10 == 0 else ("sell" if i % 10 == 5 else None),
            exchange="test",
        )
        ticks.append(tick)
    
    # Compute all features
    features = compute_windowed_features(ticks, 0, 0, "EURUSD")
    
    assert features.symbol == "EURUSD"
    assert features.tick_count == 100
    assert features.trade_count == 20  # 100/5
    assert features.sma_5 is not None
    assert features.sma_20 is not None
    assert features.rsi_14 is not None
    assert features.spread_mean is not None
    assert features.order_flow_imbalance is not None


def test_regime_detection_high_vol():
    """Test regime detection for high volatility."""
    features = ComputedFeatures(
        symbol="EURUSD",
        window_start=0,
        window_end=0,
        sma_20=1.0800,
        atr_14=0.025,  # High vol: 0.025/1.08 = 2.3% > 2%
    )
    
    features = _detect_regime(features)
    assert features.volatility_regime == 2  # High vol


def test_regime_detection_low_vol():
    """Test regime detection for low volatility."""
    features = ComputedFeatures(
        symbol="EURUSD",
        window_start=0,
        window_end=0,
        sma_20=1.0800,
        atr_14=0.0005,  # Low vol: 0.0005/1.08 = 0.046% < 0.5%
    )
    
    features = _detect_regime(features)
    assert features.volatility_regime == 0  # Low vol


def test_regime_bullish():
    """Test bullish regime detection."""
    features = ComputedFeatures(
        symbol="EURUSD",
        window_start=0,
        window_end=0,
        macd_hist=0.0005,  # Positive
    )
    
    features = _detect_regime(features)
    assert features.regime_prob > 0.5


def test_regime_bearish():
    """Test bearish regime detection."""
    features = ComputedFeatures(
        symbol="EURUSD",
        window_start=0,
        window_end=0,
        macd_hist=-0.0005,  # Negative
    )
    
    features = _detect_regime(features)
    assert features.regime_prob < 0.5


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])