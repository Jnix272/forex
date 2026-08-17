import json
import random
import time

from confluent_kafka import Producer


def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")


def main():
    print("Starting Redpanda Tick Pump...")
    producer = Producer({"bootstrap.servers": "localhost:9092"})

    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    base_prices = {"EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 149.5}

    try:
        for _i in range(1, 100):
            symbol = random.choice(symbols)
            price = base_prices[symbol] + random.gauss(0, 0.0005)
            spread = random.uniform(0.0001, 0.0003)

            tick = {
                "symbol": symbol,
                "timestamp": int(time.time_ns()),
                "bid": price - spread / 2,
                "ask": price + spread / 2,
                "bid_size": random.uniform(1, 10),
                "ask_size": random.uniform(1, 10),
                "trade_price": price if random.random() < 0.1 else None,
                "trade_size": random.uniform(1, 100) if random.random() < 0.1 else None,
                "trade_side": random.choice(["buy", "sell"]) if random.random() < 0.1 else None,
                "exchange": "synthetic",
            }

            # Remove None values
            tick = {k: v for k, v in tick.items() if v is not None}

            producer.produce(
                "market.ticks",
                key=symbol.encode("utf-8"),
                value=json.dumps(tick).encode("utf-8"),
                callback=delivery_report,
            )
            producer.poll(0)

            print(f"Produced tick for {symbol}: Bid={tick['bid']:.4f} Ask={tick['ask']:.4f}")
            time.sleep(0.5)  # 2 ticks per second

    except KeyboardInterrupt:
        pass
    finally:
        print("Flushing final messages...")
        producer.flush()


if __name__ == "__main__":
    main()
