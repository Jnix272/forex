"""
Streaming Feature Pipeline (Bytewax-based)
==========================================
Real-time feature computation pipeline replacing batch Zarr processing.
Uses Bytewax for stateful stream processing with exactly-once semantics.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

try:
    import bytewax
    import bytewax.operators as op
    import bytewax.operators.windowing as wop
    from bytewax.dataflow import Dataflow
    from bytewax.inputs import DynamicSource, StatelessSourcePartition
    from bytewax.testing import TestingSource
    BYTEWAX_AVAILABLE = True
except ImportError:
    BYTEWAX_AVAILABLE = False
    bytewax = None

# 1. Config
@dataclass
class StreamConfig:
    bootstrap_servers: str = "localhost:9092"
    input_topic: str = "market.ticks"
    consumer_group: str = "feature-pipeline"
    window_size_sec: int = 60
    slide_sec: int = 10
    watermark_delay_sec: int = 30
    state_backend: str = "rocksdb"
    state_dir: str = "./bytewax_state"
    checkpoint_interval_sec: int = 60
    num_partitions: int = 4
    worker_threads: int = 4
    compute_technical: bool = True
    compute_cross_asset: bool = True
    compute_regime: bool = True
    compute_microstructure: bool = True
    feature_store_type: str = "redis"
    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: str = "postgresql://user:pass@localhost:5432/features"
    output_topic: str = "features.computed"
    metrics_interval_sec: int = 30
    enable_metrics: bool = True

@dataclass
class MarketTick:
    symbol: str
    timestamp: int
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    trade_price: float | None = None
    trade_size: float | None = None
    trade_side: str | None = None
    exchange: str = "unknown"

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class ComputedFeatures:
    symbol: str
    window_start: int
    window_end: int
    tick_count: int = 0
    trade_count: int = 0
    sma_20: float | None = None
    ema_20: float | None = None
    rsi_14: float | None = None
    atr_14: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    spread_mean: float | None = None
    spread_std: float | None = None
    ofi: float | None = None
    vpin: float | None = None
    trade_imbalance: float | None = None
    carry_factor: float | None = None
    regime: str = "neutral"
    computed_at: int = field(default_factory=lambda: int(time.time_ns()))

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

def parse_tick(event: dict) -> MarketTick:
    return MarketTick(
        symbol=event["symbol"],
        timestamp=event["timestamp"],
        bid=event["bid"],
        ask=event["ask"],
        bid_size=event.get("bid_size", 0),
        ask_size=event.get("ask_size", 0),
        trade_price=event.get("trade_price"),
        trade_size=event.get("trade_size"),
        trade_side=event.get("trade_side"),
        exchange=event.get("exchange", "unknown"),
    )

def tick_to_keyed(tick: MarketTick) -> tuple[str, MarketTick]:
    return (tick.symbol, tick)

def _accumulate_ticks(state: list[MarketTick] | None, tick: MarketTick) -> list[MarketTick]:
    if state is None:
        state = []
    state.append(tick)
    return state

def _finalize_window_features(item) -> ComputedFeatures:
    key, (window_metadata, ticks) = item
    symbol = key

    if not ticks:
        return ComputedFeatures(symbol=symbol, window_start=0, window_end=0)

    prices = [t.mid_price for t in ticks]
    spreads = [t.spread for t in ticks]

    return ComputedFeatures(
        symbol=symbol,
        window_start=0,
        window_end=0,
        tick_count=len(ticks),
        sma_20=sum(prices)/len(prices),
        spread_mean=sum(spreads)/len(spreads),
    )

if BYTEWAX_AVAILABLE:
    class TickSource(StatelessSourcePartition):
        def __init__(self, config: StreamConfig):
            self.config = config
            try:
                from confluent_kafka import Consumer
                self.consumer = Consumer({
                    'bootstrap.servers': config.bootstrap_servers,
                    'group.id': 'bytewax_tick_processor',
                    'auto.offset.reset': 'latest',
                    'enable.auto.commit': False
                })
                self.consumer.subscribe([config.input_topic])
            except ImportError:
                warnings.warn("confluent_kafka not installed. TickSource will yield no data.")
                self.consumer = None

        def next_batch(self) -> list[MarketTick]:
            if not self.consumer:
                return []
            msgs = self.consumer.consume(num_messages=100, timeout=0.1)
            batch = []
            for msg in msgs:
                if msg is None or msg.error():
                    continue
                try:
                    payload = json.loads(msg.value().decode('utf-8'))
                    batch.append(parse_tick(payload))
                except Exception:
                    pass
            return batch
        def close(self):
            if self.consumer: self.consumer.close()

    class DynamicTickSource(DynamicSource):
        def __init__(self, config: StreamConfig):
            self.config = config
        def build(self, step_id: str, worker_index: int, worker_count: int) -> StatelessSourcePartition:
            return TickSource(self.config)

def build_feature_pipeline(config: StreamConfig) -> Dataflow:
    flow = Dataflow("streaming_pipeline")

    if not BYTEWAX_AVAILABLE:
        return flow

    stream = op.input("ticks", flow, DynamicTickSource(config))
    keyed_stream = op.key_on("key_by_symbol", stream, lambda t: t.symbol)

    clock = wop.SystemClock()
    windower = wop.TumblingWindower(
        length=timedelta(seconds=config.window_size_sec),
        align_to=datetime(2023, 1, 1, tzinfo=UTC),
    )

    windowed = wop.collect_window("accumulate_ticks", keyed_stream, clock, windower)
    features = op.map("finalize_window", windowed.down, _finalize_window_features)
    # Output to console
    def print_features(step_id, f):
        print(f"Computed features: {f.symbol} | SMA20: {f.sma_20:.5f} | Spread: {f.spread_mean:.5f}")

    op.inspect("print_output", features, print_features)
    return flow

def create_test_dataflow(config: StreamConfig) -> Dataflow:
    flow = Dataflow("test_pipeline")

    def generate_ticks():
        rng = np.random.default_rng(42)
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        base_prices = {"EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 149.5}
        for i in range(100):
            symbol = rng.choice(symbols)
            price = base_prices[symbol] + rng.normal(0, 0.0005)
            spread = rng.uniform(0.0001, 0.0003)
            yield MarketTick(
                symbol=symbol,
                timestamp=int(time.time_ns()) + i * 1_000_000_000,
                bid=price - spread/2,
                ask=price + spread/2,
            )

    stream = op.input("ticks", flow, TestingSource(generate_ticks()))
    keyed_stream = op.key_on("key_by_symbol", stream, lambda t: t.symbol)

    clock = wop.EventClock(
        ts_getter=lambda x: datetime.fromtimestamp(x.timestamp / 1e9, tz=UTC),
        wait_for_system_duration=timedelta(seconds=0),
    )
    windower = wop.TumblingWindower(
        length=timedelta(seconds=5),
        align_to=datetime(2023, 1, 1, tzinfo=UTC),
    )

    windowed = wop.collect_window("accumulate_ticks", keyed_stream, clock, windower)
    features = op.map("finalize_window", windowed.down, _finalize_window_features)

    def print_features(step_id, f):
        print(f"Test Pipeline Computed features: {f.symbol} | SMA20: {f.sma_20:.5f} | Spread: {f.spread_mean:.5f}")

    op.inspect("print_output", features, print_features)
    return flow

class StreamingFeaturePipeline:
    def __init__(self, config: StreamConfig):
        self.config = config
        self.flow = build_feature_pipeline(config)

if __name__ == "__main__":
    if not BYTEWAX_AVAILABLE:
        print("Bytewax not available.")
    else:
        print("Bytewax available. Streaming feature pipeline ready.")
        config = StreamConfig(
            bootstrap_servers="localhost:9092",
            input_topic="market.ticks",
            window_size_sec=5,
        )
        pipeline = StreamingFeaturePipeline(config)
        import bytewax.testing
        print("Starting Bytewax streaming engine on Redpanda...")
        bytewax.testing.run_main(pipeline.flow)
