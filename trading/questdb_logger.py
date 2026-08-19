import time
from questdb.ingress import Sender, TimestampNanos

class QuestDBLiveLogger:
    """
    Streams live ticks into QuestDB via InfluxDB Line Protocol (ILP).
    This protocol is ultra-fast and designed for high-frequency ingestion.
    """
    def __init__(self, host: str = "localhost", port: int = 9009):
        self.conf_string = f"tcp::addr={host}:{port};"
        self.table_name = "live_ticks"

    def log_tick(self, pair: str, bid: float, ask: float, volume: float = 0.0):
        """
        Sends a single tick to QuestDB.
        """
        ts_nanos = TimestampNanos.now()

        try:
            with Sender.from_conf(self.conf_string) as sender:
                sender.row(
                    self.table_name,
                    symbols={"pair": pair},
                    columns={
                        "bid": bid,
                        "ask": ask,
                        "spread": ask - bid,
                        "volume": volume
                    },
                    at=ts_nanos
                )
                sender.flush()
        except Exception as e:
            print(f"QuestDB ILP Error: {e}")

if __name__ == "__main__":
    print("Testing QuestDB Ingestion...")
    logger = QuestDBLiveLogger()
    
    for i in range(5):
        bid = 1.1050 + (i * 0.0001)
        ask = bid + 0.0002
        logger.log_tick("EURUSD", bid, ask, 1.5)
        print(f"Sent tick: EURUSD | Bid: {bid:.4f} | Ask: {ask:.4f}")
        import time
        time.sleep(0.1)
        
    print("\nTicks sent! You can query them at http://localhost:9000")
