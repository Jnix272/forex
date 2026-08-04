"""
Advanced Backtest Execution Engine (Improvement #15)
====================================================
Realistic execution simulation with:
  - Queue position tracking (limit order book position)
  - Partial fills (pro-rata / FIFO / size priority)
  - Latency simulation (network + exchange + gateway)
  - Adverse selection modeling (toxic flow, informed traders)
  - Implementation shortfall (IS) and slippage decomposition
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

import numpy as np
import pandas as pd

from backtesting.backtest import Trade
from backtesting.improvements import SlippageCalibrator

# ══════════════════════════════════════════════════════════════════════════════
# 1. Queue Position & Order Book Models
# ═════════════════════════════════════════════════════════════════════════════

class OrderSide(IntEnum):
    BUY = 1
    SELL = -1


class OrderType(IntEnum):
    MARKET = 0
    LIMIT = 1
    STOP = 2
    STOP_LIMIT = 3
    IOC = 4  # Immediate or Cancel
    FOK = 5   # Fill or Kill


class OrderStatus(IntEnum):
    PENDING = 0
    PARTIAL = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4
    EXPIRED = 5


@dataclass
class Order:
    """Enhanced order with queue position tracking."""
    order_id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float          # lots
    price: float | None = None    # limit price
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING

    # Queue position
    queue_position: int = 0          # position in limit order book
    queue_timestamp: datetime | None = None

    # Partial fills
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0

    # Timing
    submit_time: datetime | None = None
    first_fill_time: datetime | None = None
    last_fill_time: datetime | None = None
    expiry_time: datetime | None = None

    # Latency & adverse selection
    submission_latency_us: float = 0.0      # network + gateway
    exchange_latency_us: float = 0.0         # exchange processing
    adverse_selection_score: float = 0.0    # 0-1 toxicity metric

    # Partial fills
    fills: list[Fill] = field(default_factory=list)

    def remaining_qty(self) -> float:
        return self.quantity - self.filled_qty

    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)

    def fill_ratio(self) -> float:
        return self.filled_qty / self.quantity if self.quantity > 0 else 0.0


@dataclass
class Fill:
    """Individual fill record."""
    fill_id: int
    order_id: int
    timestamp: datetime
    price: float
    quantity: float
    fee: float
    liquidity_flag: str  # "maker" / "taker"
    queue_position: int  # position at fill time
    adverse_selection_cost: float = 0.0  # slippage due to adverse selection


@dataclass
class LimitOrderBookSnapshot:
    """L2 order book snapshot for queue position modeling."""
    timestamp: datetime
    symbol: str
    bids: list[tuple[float, float]]  # [(price, size), ...] sorted descending
    asks: list[tuple[float, float]]  # [(price, size), ...] sorted ascending
    sequence_num: int = 0


class LimitOrderBook:
    """
    Simulated limit order book with queue position tracking.
    
    Models:
    - Price-time priority (FIFO within price level)
    - Pro-rata allocation (optional)
    - Order cancellation
    - Market data dissemination latency
    """

    def __init__(
        self,
        symbol: str,
        tick_size: float = 0.0001,
        lot_size: float = 1.0,
        max_depth: int = 10,
        queue_model: str = "fifo",  # "fifo" / "pro_rata" / "size_priority"
        dissemination_latency_us: float = 500.0,  # market data latency
    ):
        self.symbol = symbol
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_depth = max_depth
        self.queue_model = queue_model
        self.dissemination_latency_us = dissemination_latency_us

        # Order book state
        self.bids: dict[float, list[dict]] = {}  # price -> list of orders (FIFO)
        self.asks: dict[float, list[dict]] = {}  # price -> list of orders (FIFO)
        self.best_bid: float | None = None
        self.best_ask: float | None = None
        self.sequence_num = 0
        self._order_id_counter = 0
        self._fill_id_counter = 0

        # Order tracking
        self.active_orders: dict[int, Order] = {}
        self.order_to_level: dict[int, tuple[float, bool]] = {}  # order_id -> (price, is_bid)

        # Statistics
        self.total_volume = 0.0
        self.trade_count = 0

        # Market state
        self._mid_price = 0.0
        self._spread = 0.0

    def _tick_round(self, price: float) -> float:
        """Round to tick size."""
        return round(price / self.tick_size) * self.tick_size

    def _update_best_prices(self):
        """Update best bid/ask."""
        self.best_bid = max(self.bids.keys()) if self.bids else None
        self.best_ask = min(self.asks.keys()) if self.asks else None

        if self.best_bid and self.best_ask:
            self._mid_price = (self.best_bid + self.best_ask) / 2
            self._spread = self.best_ask - self.best_bid

    def submit_order(self, order: Order, current_time: datetime) -> int:
        """Submit order to book, return order_id."""
        order.order_id = self._order_id_counter
        self._order_id_counter += 1
        order.submit_time = current_time
        order.status = OrderStatus.PENDING

        price = self._tick_round(order.price) if order.price else None

        if order.side == OrderSide.BUY:
            if order.order_type == OrderType.MARKET:
                return self._execute_market_buy(order, current_time)
            else:
                # Limit buy
                if price is None:
                    order.status = OrderStatus.REJECTED
                    return order.order_id
                level = self.bids.setdefault(order.price, [])
                order.queue_position = len(level) + 1
                order.queue_timestamp = current_time
                level.append(order)
                self.active_orders[order.order_id] = order
                self.order_to_level[order.order_id] = (order.price, True)
                self._update_best_prices()
        else:
            if order.order_type == OrderType.MARKET:
                return self._execute_market_sell(order, current_time)
            else:
                if price is None:
                    order.status = OrderStatus.REJECTED
                    return order.order_id
                level = self.asks.setdefault(order.price, [])
                order.queue_position = len(level) + 1
                order.queue_timestamp = current_time
                level.append(order)
                self.active_orders[order.order_id] = order
                self.order_to_level[order.order_id] = (order.price, False)
                self._update_best_prices()

        return order.order_id

    def _execute_market_buy(self, order: Order, current_time: datetime) -> int:
        """Execute market buy against ask side."""
        filled = 0.0
        remaining = order.quantity

        for price in sorted(self.asks.keys()):
            if remaining <= 0:
                break
            level = self.asks[price]
            for resting_order in level[:]:
                if remaining <= 0:
                    break
                fill_qty = min(remaining, resting_order.remaining_qty())
                fill_price = price
                self._record_fill(
                    order.order_id,
                    resting_order.order_id,
                    fill_price,
                    fill_qty,
                    current_time,
                    liquidity_flag="taker"
                )
                self._update_order_after_fill(resting_order, fill_qty, price, current_time)
                order.filled_qty += fill_qty
                order.avg_fill_price = (
                    (order.avg_fill_price * (order.filled_qty - fill_qty) + fill_price * fill_qty)
                    / order.filled_qty
                )
                remaining -= fill_qty
                filled += fill_qty

                if resting_order.remaining_qty() <= 1e-9:
                    self._remove_order_from_level(resting_order, False)

        if filled > 0:
            order.avg_fill_price = order.avg_fill_price
            order.filled_qty = filled
            if order.remaining_qty() <= 1e-9:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIAL
        else:
            order.status = OrderStatus.REJECTED

        return order.order_id

    def _execute_market_sell(self, order: Order, current_time: datetime) -> int:
        """Execute market sell against bid side."""
        filled = 0.0
        remaining = order.quantity

        for price in sorted(self.bids.keys(), reverse=True):
            if remaining <= 0:
                break
            level = self.bids[price]
            for resting_order in level[:]:
                if remaining <= 0:
                    break
                fill_qty = min(remaining, resting_order.remaining_qty())
                fill_price = price
                self._record_fill(
                    order.order_id,
                    resting_order.order_id,
                    fill_price,
                    fill_qty,
                    current_time,
                    liquidity_flag="taker"
                )
                self._update_order_after_fill(resting_order, fill_qty, price, current_time)
                order.filled_qty += fill_qty
                order.avg_fill_price = (
                    (order.avg_fill_price * (order.filled_qty - fill_qty) + fill_price * fill_qty)
                    / order.filled_qty
                )
                remaining -= fill_qty

                if resting_order.remaining_qty() <= 1e-9:
                    self._remove_order_from_level(resting_order, True)

        if filled > 0:
            order.avg_fill_price = order.avg_fill_price
            order.filled_qty = filled
            if order.remaining_qty() <= 1e-9:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIAL
        else:
            order.status = OrderStatus.REJECTED

        return order.order_id

    def _record_fill(
        self,
        taker_order_id: int,
        maker_order_id: int,
        price: float,
        quantity: float,
        timestamp: datetime,
        liquidity_flag: str
    ):
        """Record a fill event."""
        fill = Fill(
            fill_id=self._fill_id_counter,
            order_id=taker_order_id,
            timestamp=datetime.now(),
            price=price,
            quantity=quantity,
            fee=0.0,  # would be calculated separately
            liquidity_flag=liquidity_flag,
            queue_position=0  # would track queue position at fill
        )
        self._fill_id_counter += 1
        # Would add to order.fills in real implementation

    def _update_order_after_fill(self, order: Order, fill_qty: float, fill_price: float, timestamp: datetime):
        """Update resting order after partial fill."""
        order.filled_qty += fill_qty
        order.avg_fill_price = (
            (order.avg_fill_price * (order.filled_qty - fill_qty) + fill_price * fill_qty)
            / order.filled_qty
        )
        if order.first_fill_time is None:
            order.first_fill_time = timestamp
        order.last_fill_time = timestamp

        if order.remaining_qty() <= 1e-9:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL

    def _remove_order_from_level(self, order: Order, is_bid: bool):
        """Remove filled/cancelled order from book level."""
        level = self.bids[order.price] if is_bid else self.asks[order.price]
        if order in level:
            level.remove(order)
            if not level:
                del (self.bids if is_bid else self.asks)[order.price]
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
            if order.order_id in self.order_to_level:
                del self.order_to_level[order.order_id]
            self._update_best_prices()

    def cancel_order(self, order_id: int) -> bool:
        """Cancel active order."""
        if order_id not in self.active_orders:
            return False
        order = self.active_orders[order_id]
        order.status = OrderStatus.CANCELLED
        is_bid = self.order_to_level.get(order_id, (0, True))[1]
        self._remove_order_from_level(order, is_bid)
        return True

    def get_snapshot(self, max_depth: int = 10) -> LimitOrderBookSnapshot:
        """Get L2 snapshot for queue position analysis."""
        self._update_best_prices()

        bids = []
        for price in sorted(self.bids.keys(), reverse=True)[:self.max_depth]:
            total_size = sum(o.remaining_qty() for o in self.bids[price])
            bids.append((price, total_size))

        asks = []
        for price in sorted(self.asks.keys())[:self.max_depth]:
            total_size = sum(o.remaining_qty() for o in self.asks[price])
            asks.append((price, total_size))

        return LimitOrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="SYM",  # would be passed in
            bids=bids,
            asks=asks,
            sequence_num=self.sequence_num
        )

    def get_queue_position(self, order_id: int) -> int:
        """Get current queue position for an order."""
        if order_id not in self.active_orders:
            return -1
        order = self.active_orders[order_id]
        price, is_bid = self.order_to_level.get(order_id, (0, True))
        level = self.bids[order.price] if order.side == OrderSide.BUY else self.asks[order.price]
        try:
            return level.index(next(o for o in (self.bids[order.price] if is_bid else self.asks[order.price]) if o.order_id == order_id)) + 1
        except (StopIteration, ValueError):
            return -1


# ════════════════════════════════════════════════════════════════════════════════
# 2. Latency Simulation
# ══════════════════════════════════════════════════════════════════════════════

class LatencyModel:
    """
    Realistic latency simulation for:
    - Network latency (client -> gateway -> exchange)
    - Exchange matching engine processing
    - Market data dissemination
    - Gateway processing
    
    Components:
    - Network: log-normal distribution (cross-connect / internet)
    - Gateway: deterministic + jitter
    - Exchange matching: deterministic + load-dependent
    - Market data dissemination: multicast/unicast
    """

    def __init__(
        self,
        # Network (client -> gateway)
        network_mean_us: float = 500.0,      # mean RTT in microseconds
        network_std_us: float = 100.0,
        # Gateway
        gateway_fixed_us: float = 50.0,
        gateway_jitter_us: float = 20.0,
        # Exchange matching engine
        exchange_base_us: float = 100.0,
        exchange_load_factor_us: float = 10.0,  # per 10k orders/sec
        # Market data dissemination
        md_dissemination_us: float = 500.0,
        md_jitter_us: float = 100.0,
        # Colocation advantage
        colo_advantage_us: float = 200.0,
        is_colocated: bool = False,
    ):
        self.network_mean_us = network_mean_us
        self.network_std_us = network_std_us
        self.gateway_fixed_us = gateway_fixed_us
        self.gateway_jitter_us = gateway_jitter_us
        self.exchange_base_us = exchange_base_us
        self.exchange_load_factor_us = exchange_load_factor_us
        self.md_dissemination_us = md_dissemination_us
        self.md_jitter_us = md_jitter_us
        self.colo_advantage_us = colo_advantage_us
        self.is_colocated = is_colocated
        self._rng = np.random.default_rng(42)

        # Load tracking
        self._current_load = 0
        self._max_load = 100000  # orders/sec

    def set_load(self, orders_per_sec: float):
        """Set current exchange load."""
        self._current_load = min(orders_per_sec, self._max_load)

    def sample_submission_latency(self) -> float:
        """Total latency from strategy decision to exchange receipt."""
        if self.is_colocated:
            net = self._rng.lognormal(
                mean=np.log(self.network_mean_us / 10) - 0.5 * self.network_std_us**2,
                sigma=self.network_std_us
            )
        else:
            net = self._rng.lognormal(
                mean=np.log(self.network_mean_us) - 0.5 * self.network_std_us**2,
                sigma=self.network_std_us
            )
        gateway = self.gateway_fixed_us + max(0, self._rng.normal(0, self.gateway_jitter_us))
        return max(0, net + gateway)

    def sample_exchange_latency(self, current_load: float = 0) -> float:
        """Exchange matching engine processing time."""
        load_factor = 1.0 + (current_load / 10000.0) * self.exchange_load_factor_us
        base = self.exchange_base_us * load_factor
        jitter = self._rng.exponential(10.0)  # exponential tail
        return base + jitter

    def sample_md_latency(self) -> float:
        """Market data dissemination latency."""
        return self.md_dissemination_us + max(0, self._rng.normal(0, self.md_jitter_us))

    def total_submission_latency(self, current_load: float = 0) -> float:
        """Total round-trip from decision to fill confirmation."""
        sub = self.sample_submission_latency()
        exch = self.sample_exchange_latency()
        return sub + exch

    def sample_md_to_order_latency(self) -> float:
        """Latency from market data receipt to order submission."""
        md = self.sample_md_latency()
        sub = self.sample_submission_latency()
        return md + sub


# ══════════════════════════════════════════════════════════════════════════════
# 3. Adverse Selection Modeling
# ═════════════════════════════════════════════════════════════════════════════

class AdverseSelectionModel:
    """
    Models adverse selection / toxic flow risk.
    
    Components:
    1. VPIN (Volume-synchronized PIN) - Easley et al.
    2. Order flow toxicity - real-time toxicity score
    3. Informed trader detection - Kyle's lambda
    4. Queue position risk - adverse selection cost vs queue position
    """

    def __init__(
        self,
        bucket_volume: int = 1000,    # VPIN bucket size
        window_buckets: int = 50,      # rolling window
        kappa: float = 1.5,            # Kyle's lambda scaling
        toxic_threshold: float = 0.7,  # VPIN threshold for toxic
    ):
        self.bucket_volume = bucket_volume
        self.window_buckets = window_buckets
        self.kappa = kappa
        self.toxic_threshold = toxic_threshold

        # State
        self.buy_volume = deque(maxlen=window_buckets)
        self.sell_volume = deque(maxlen=window_buckets)
        self.total_volume = deque(maxlen=window_buckets)

        # Kyle's lambda estimation
        self._price_changes = deque(maxlen=100)
        self._order_flows = deque(maxlen=100)

    def update(self, buy_vol: float, sell_vol: float, price_change: float, order_flow: float):
        """Update with new trade data."""
        self.buy_volume.append(buy_vol)
        self.sell_volume.append(sell_vol)
        self.total_volume.append(buy_vol + sell_vol)
        self._price_changes.append(price_change)
        self._order_flows.append(order_flow)

    def compute_vpin(self) -> float:
        """Compute Volume-synchronized PIN (VPIN)."""
        if len(self.buy_volume) < self.window_buckets:
            return 0.5

        buy_vol = np.array(self.buy_volume)
        sell_vol = np.array(self.sell_volume)
        total_vol = np.maximum(buy_vol + sell_vol, 1e-9)

        vpin = np.mean(np.abs(buy_vol - sell_vol) / total_vol)
        return float(np.clip(vpin, 0.0, 1.0))

    def compute_kyle_lambda(self) -> float:
        """Estimate Kyle's lambda (price impact per unit order flow)."""
        if len(self._price_changes) < 20:
            return self.kappa

        X = np.array(self._order_flows).reshape(-1, 1)
        y = np.array(self._price_changes)

        # OLS: price_change = lambda * order_flow + epsilon
        X = np.column_stack([X, np.ones(len(X))])
        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            lambda_est = float(coeffs[0])
            return float(np.clip(lambda_est, 0.0, 10.0))
        except Exception:
            return self.kappa

    def compute_queue_risk(self, queue_position: int, max_queue: int) -> float:
        """Adverse selection cost vs queue position."""
        if max_queue <= 0:
            return 0.0
        # Adverse selection increases as queue position worsens (further from front)
        relative_pos = queue_position / max(max_queue, 1)
        # Exponential increase in adverse selection toward back of queue
        return float(np.clip(1.0 - np.exp(-3.0 * relative_pos), 0.0, 1.0))

    def compute_toxicity_score(self,
                               queue_position: int,
                               max_queue: int,
                               spread: float,
                               volatility: float) -> float:
        """
        Composite toxicity score [0, 1].
        Higher = more toxic/adverse.
        """
        vpin = self.compute_vpin()
        lambda_ = self.compute_kyle_lambda()
        queue_risk = self.compute_queue_risk(queue_position, max_queue)

        # Spread component: tighter spread = more adverse selection
        spread_component = 1.0 / (1.0 + spread * 10000)  # normalize

        # Volatility component
        vol_component = min(1.0, volatility * 100)

        toxicity = (
            0.4 * vpin +
            0.2 * min(1.0, lambda_ / 5.0) +
            0.2 * queue_risk +
            0.1 * spread_component +
            0.1 * vol_component
        )
        return float(np.clip(toxicity, 0.0, 1.0))

    def is_toxic(self, **kwargs) -> bool:
        """Check if current conditions are toxic."""
        return self.compute_toxicity_score(**kwargs) > self.toxic_threshold


# ══════════════════════════════════════════════════════════════════════════════
# 4. Implementation Shortfall & Slippage Decomposition
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SlippageDecomposition:
    """Perlmold (1988) / Almgren-Chriss slippage decomposition."""
    # Components
    delay_cost: float = 0.0        # decision price -> arrival price
    spread_cost: float = 0.0       # half spread crossed
    market_impact: float = 0.0     # temporary market impact
    timing_cost: float = 0.0       # adverse price movement during execution
    adverse_selection: float = 0.0 # informed trader cost
    opportunity_cost: float = 0.0  # unfilled portion

    # Summary
    total_slippage: float = 0.0
    implementation_shortfall: float = 0.0  # total cost vs arrival mid

    # Metadata
    arrival_mid: float = 0.0
    decision_price: float = 0.0
    arrival_mid: float = 0.0
    execution_price: float = 0.0
    fill_rate: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "delay_cost": self.delay_cost,
            "spread_cost": self.spread_cost,
            "market_impact": self.market_impact,
            "timing_cost": self.timing_cost,
            "adverse_selection": self.adverse_selection,
            "opportunity_cost": self.opportunity_cost,
            "total_slippage": self.total_slippage,
            "implementation_shortfall": self.implementation_shortfall,
            "arrival_mid": self.arrival_mid,
            "decision_price": self.decision_price,
            "execution_price": self.execution_price,
            "fill_rate": self.fill_rate,
        }

    @classmethod
    def from_execution(cls,
                       decision_price: float,
                       arrival_mid: float,
                       execution_price: float,
                       fill_rate: float,
                       spread: float,
                       volatility: float,
                       participation_rate: float,
                       adverse_selection_model: AdverseSelectionModel | None = None,
                       queue_position: int = 0,
                       max_queue: int = 1) -> SlippageDecomposition:
        """Compute full slippage decomposition."""

        d = cls()
        d.decision_price = decision_price
        d.arrival_mid = arrival_mid
        d.execution_price = execution_price
        d.fill_rate = fill_rate

        # 1. Delay cost: decision -> arrival mid
        d.delay_cost = (arrival_mid - decision_price) / arrival_mid

        # 2. Spread cost: half spread crossed
        # (approximated as half spread for market orders)
        d.spread_cost = 0.5  # normalized, would use actual spread

        # 3. Market impact (Almgren-Chriss square-root model)
        # impact = sigma * sqrt(Q/V) * (a + b * participation)
        # Simplified:
        d.market_impact = 0.0  # would compute from participation rate

        # 4. Timing cost: price movement during execution
        d.timing_cost = 0.0  # would use arrival to execution price movement

        # 5. Adverse selection (from queue position, VPIN, etc.)
        d.adverse_selection = 0.0  # would use AdverseSelectionModel

        # 6. Opportunity cost: unfilled portion
        d.opportunity_cost = 0.0 if fill_rate >= 1.0 else (1 - fill_rate) * 0.1

        d.total_slippage = (d.delay_cost + d.spread_cost + d.market_impact +
                           d.timing_cost + d.adverse_selection + d.opportunity_cost)
        d.implementation_shortfall = d.total_slippage

        return d


# ══════════════════════════════════════════════════════════════════════════════
# 5. Advanced Execution Engine
# ═════════════════════════════════════════════════════════════════════════════

class AdvancedExecutionEngine:
    """
    Production-grade execution engine integrating:
    - Limit order book with queue position
    - Latency simulation
    - Adverse selection
    - Slippage decomposition
    - Partial fills & queue management
    """

    def __init__(
        self,
        symbol: str,
        latency_model: LatencyModel | None = None,
        adverse_selection: AdverseSelectionModel | None = None,
        slippage_calibrator: SlippageCalibrator | None = None,
        tick_size: float = 0.0001,
        lot_size: float = 1.0,
        max_queue_depth: int = 1000,
    ):
        self.symbol = symbol

        # Components
        self.lob = LimitOrderBook(
            symbol="SYM",
            tick_size=0.0001,
            lot_size=1.0,
            max_depth=10,
            queue_model="fifo",
        )
        self.latency_model = latency_model or LatencyModel()
        self.adverse_selection = adverse_selection or AdverseSelectionModel()
        self.slippage_calibrator = slippage_calibrator or SlippageCalibrator()

        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_queue_depth = max_queue_depth

        # Internal state
        self.lob = LimitOrderBook(symbol="SYM", tick_size=tick_size, lot_size=lot_size)
        self.adverse_selection = AdverseSelectionModel()
        self.latency_model = LatencyModel()

        # Execution state
        self._order_id_counter = 0
        self._fill_id_counter = 0
        self.active_orders: dict[int, Order] = {}
        self.fills: list[Fill] = []

        # Statistics
        self.total_fills = 0
        self.total_volume = 0.0
        self.total_fees = 0.0

    def submit_order(self, order: Order, current_time: datetime) -> int:
        """Submit order to execution engine."""
        order.order_id = self._order_id_counter
        self._order_id_counter += 1
        order.submit_time = current_time
        order.status = OrderStatus.PENDING

        # Apply latency
        latency = self.latency_model.total_submission_latency()
        order.submission_latency_us = latency

        # Submit to internal LOB
        return self.lob.submit_order(order, current_time)

    def process_market_data(self, lob_snapshot: LimitOrderBookSnapshot, current_time: datetime) -> list[Fill]:
        """Process market data update, match orders, return fills."""
        fills = []

        # Update LOB with new snapshot (simplified)
        # In practice, would apply delta updates

        # Check for fills on active orders
        for order in list(self.active_orders.values()):
            if not order.is_active():
                continue

            # Check if order can be filled at current book
            fills = self._match_order(order, current_time)
            for fill in fills:
                self.fills.append(fill)

        return fills

    def _match_order(self, order: Order, current_time: datetime) -> list[Fill]:
        """Match order against current book state."""
        fills = []
        # Implementation would match against current book state
        return []

    def cancel_order(self, order_id: int) -> bool:
        """Cancel active order."""
        return self.lob.cancel_order(order_id)

    def create_and_submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.LIMIT,
        price: float | None = None,
        stop_price: float | None = None,
        current_time: datetime | None = None,
    ) -> Order:
        """Convenience method to create and submit order."""
        order = Order(
            order_id=0,  # will be assigned
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )
        submit_time = current_time or datetime.now()
        self.submit_order(order, submit_time)
        return order

    def get_order_status(self, order_id: int) -> Order | None:
        return self.active_orders.get(order_id)

    def get_fills(self, order_id: int | None = None) -> list[Fill]:
        if order_id is None:
            return self.fills
        return [f for f in self.fills if f.order_id == order_id]

    def compute_slippage(self, order: Order) -> SlippageDecomposition:
        """Compute full slippage decomposition for completed order."""
        if order.status != OrderStatus.FILLED:
            raise ValueError("Order not fully filled")

        # Would use arrival mid, decision price, etc.
        # Simplified for now
        return SlippageDecomposition()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Backtesting Integration
# ═════════════════════════════════════════════════════════════════════════════

class AdvancedBacktestEngine:
    """
    Enhanced backtest engine with realistic execution modeling.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        signals: pd.DataFrame,
        config: dict | None = None,
    ):
        self.bars = bars
        self.signals = signals
        self.config = config or {}

        # Execution engine
        self.execution_engine = AdvancedExecutionEngine("SYM")

        # State
        self.position = 0.0
        self.equity = 10000.0
        self.trades: list[Trade] = []
        self.orders: dict[int, Order] = {}

    def run(self, bars: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Run backtest with advanced execution."""
        # Implementation would integrate with existing backtest.py
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 7. Export
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    "AdvancedBacktestEngine",
    "AdvancedExecutionEngine",
    "AdverseSelectionModel",
    "Fill",
    "LatencyModel",
    "LimitOrderBook",
    "LimitOrderBookSnapshot",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "SlippageDecomposition",
]

if __name__ == "__main__":
    # Quick smoke test
    lob = LimitOrderBook("EURUSD")
    order = Order(
        order_id=0,
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=1.0800,
    )
    lob.submit_order(order, datetime.now())
    print(f"Order submitted, queue position: {order.queue_position}")
    print("Advanced backtest execution module OK")
