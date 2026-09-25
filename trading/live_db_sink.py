"""
trading/live_db_sink.py
=======================
Non-blocking, asynchronous DuckDB sink for live ticks, 5-minute bars, and trades.
Uses a background worker queue to ensure database writes never introduce latency
into the live trading or tick polling loops.
"""

from __future__ import annotations

import http.server
import json
import logging
import queue
import threading
import time
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("forex.live_db_sink")

try:
    import pandas  # Pre-import to avoid lazy import lock in DuckDB background thread
except ImportError:
    pandas = None

try:
    import duckdb

    _DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore
    _DUCKDB_AVAILABLE = False


_ALLOWED_SQL_VERBS = ("SELECT", "PRAGMA", "DESCRIBE", "EXPLAIN")
_DISALLOWED_SQL_KEYWORDS = (
    ";",
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "ALTER",
    "CREATE",
    "ATTACH",
    "DETACH",
    "COPY",
    "INSTALL",
    "LOAD",
    "CALL",
    "READ_CSV",
    "READ_PARQUET",
    "READ_JSON",
    "SCAN_",
)


class _TelemetryHTTPHandler(http.server.BaseHTTPRequestHandler):
    sink: LiveDuckDBSink = None

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logging

    def address_string(self):
        # Override to prevent reverse DNS lookup on Windows (avoids 5-9s latency per request)
        return str(self.client_address[0])

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/summary"):
            data = self.sink.get_summary()
            self._send_json(data)
        elif path == "/query":
            sql = query.get("sql", [""])[0].strip()
            if not sql:
                self._send_json({"status": "error", "error": "Missing sql parameter"}, status=400)
                return

            sql_upper = sql.upper()
            if not any(sql_upper.startswith(verb) for verb in _ALLOWED_SQL_VERBS):
                self._send_json(
                    {
                        "status": "error",
                        "error": "Only read queries (SELECT, PRAGMA, DESCRIBE, EXPLAIN) are allowed",
                    },
                    status=400,
                )
                return

            for kw in _DISALLOWED_SQL_KEYWORDS:
                if kw in sql_upper:
                    self._send_json(
                        {
                            "status": "error",
                            "error": f"Disallowed SQL keyword or symbol: '{kw}' is prohibited",
                        },
                        status=400,
                    )
                    return

            data = self.sink.query_sql(sql)
            status_code = 200 if data.get("status") == "ok" else 400
            self._send_json(data, status=status_code)
        else:
            self._send_json({"status": "error", "error": "Not found"}, status=404)

    def _send_json(self, obj: Any, status: int = 200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class LiveDuckDBSink:
    """Asynchronous DuckDB persistence for live trading telemetry.

    Tables managed:
      - live_ticks: Raw bid/ask quotes and spreads from broker.
      - live_bars: Completed OHLCV candles, model actions, sentiment, and latency.
      - live_trades: Order executions, fills, stop-loss, take-profit, and guard blocks.
    """

    def __init__(
        self,
        db_path: str | Path = "data/store/live_trading.duckdb",
        batch_size: int = 200,
        flush_interval_s: float = 1.0,
        max_queue_size: int = 100_000,
        http_port: int | None = 8002,
        enabled: bool = True,
    ):
        self.enabled = enabled and _DUCKDB_AVAILABLE
        self.db_path = Path(db_path)
        self.batch_size = int(batch_size)
        self.flush_interval_s = float(flush_interval_s)
        self.http_port = int(http_port) if http_port is not None else None

        if not _DUCKDB_AVAILABLE and enabled:
            logger.warning("[LiveDBSink] duckdb package not available; DB sink disabled.")

        self._queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._conn = None
        self._http_server: http.server.ThreadingHTTPServer | None = None
        self._db_lock = threading.Lock()

        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._start_worker()

    def _start_worker(self) -> None:
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="LiveDuckDBWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def _create_tables(self, conn) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_ticks (
                timestamp   TIMESTAMPTZ,
                pair        VARCHAR,
                bid         DOUBLE,
                ask         DOUBLE,
                mid         DOUBLE,
                spread_pips DOUBLE,
                volume      DOUBLE
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_bars (
                timestamp   TIMESTAMPTZ,
                pair        VARCHAR,
                bar_idx     BIGINT,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                action      INTEGER,
                model       VARCHAR,
                lots        DOUBLE,
                equity      DOUBLE,
                sentiment   DOUBLE,
                latency_ms  DOUBLE,
                var_pct     DOUBLE,
                model_weights VARCHAR
            );
            ALTER TABLE live_bars ADD COLUMN IF NOT EXISTS model_weights VARCHAR;
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_trades (
                timestamp   TIMESTAMPTZ,
                pair        VARCHAR,
                event       VARCHAR,
                action      VARCHAR,
                lots        DOUBLE,
                price       DOUBLE,
                order_id    VARCHAR,
                reason      VARCHAR,
                pnl         DOUBLE,
                details     VARCHAR
            );
            """
        )

    def record_tick(
        self,
        pair: str,
        bid: float,
        ask: float,
        mid: float | None = None,
        spread_pips: float | None = None,
        volume: float = 1.0,
        timestamp: datetime | None = None,
    ) -> None:
        if not self.enabled or not self._running:
            return

        try:
            ts = timestamp or datetime.now(UTC)
            mid_val = mid if mid is not None else (float(bid) + float(ask)) / 2.0
            p = str(pair).upper()
            if spread_pips is None:
                mult = 100.0 if "JPY" in p else 10000.0
                spread_pips = (float(ask) - float(bid)) * mult

            item = {
                "timestamp": ts,
                "pair": p,
                "bid": float(bid),
                "ask": float(ask),
                "mid": float(mid_val),
                "spread_pips": float(spread_pips),
                "volume": float(volume),
            }
            self._queue.put_nowait(("tick", item))
        except queue.Full:
            pass
        except Exception as exc:
            logger.debug(f"[LiveDBSink] record_tick error: {exc}")

    def record_bar(
        self,
        pair: str,
        bar_idx: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        action: int,
        model: str,
        lots: float,
        equity: float,
        sentiment: float = 0.0,
        latency_ms: float = 0.0,
        var_pct: float = 0.0,
        model_weights: dict[str, float] | str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        if not self.enabled or not self._running:
            return

        try:
            ts = timestamp or datetime.now(UTC)
            weights_val = (
                json.dumps(model_weights)
                if isinstance(model_weights, dict)
                else (str(model_weights) if model_weights is not None else None)
            )
            item = {
                "timestamp": ts,
                "pair": str(pair).upper(),
                "bar_idx": int(bar_idx),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "action": int(action),
                "model": str(model),
                "lots": float(lots),
                "equity": float(equity),
                "sentiment": float(sentiment),
                "latency_ms": float(latency_ms),
                "var_pct": float(var_pct),
                "model_weights": weights_val,
            }
            self._queue.put_nowait(("bar", item))
        except queue.Full:
            pass
        except Exception as exc:
            logger.debug(f"[LiveDBSink] record_bar error: {exc}")

    def record_trade(
        self,
        pair: str,
        event: str,
        action: str = "",
        lots: float = 0.0,
        price: float = 0.0,
        order_id: str = "",
        reason: str = "",
        pnl: float = 0.0,
        details: Any = None,
        timestamp: datetime | None = None,
    ) -> None:
        if not self.enabled or not self._running:
            return

        try:
            ts = timestamp or datetime.now(UTC)
            details_str = json.dumps(details, default=str) if details is not None and not isinstance(details, str) else str(details or "")

            item = {
                "timestamp": ts,
                "pair": str(pair).upper(),
                "event": str(event),
                "action": str(action),
                "lots": float(lots),
                "price": float(price),
                "order_id": str(order_id),
                "reason": str(reason),
                "pnl": float(pnl),
                "details": details_str,
            }
            self._queue.put_nowait(("trade", item))
        except queue.Full:
            pass
        except Exception as exc:
            logger.debug(f"[LiveDBSink] record_trade error: {exc}")

    def _append_msg(
        self,
        msg: tuple[str, dict[str, Any]],
        tick_batch: list[tuple],
        bar_batch: list[tuple],
        trade_batch: list[tuple],
    ) -> None:
        kind, data = msg
        if kind == "tick":
            tick_batch.append(
                (
                    data["timestamp"],
                    data["pair"],
                    data["bid"],
                    data["ask"],
                    data["mid"],
                    data["spread_pips"],
                    data["volume"],
                )
            )
        elif kind == "bar":
            bar_batch.append(
                (
                    data["timestamp"],
                    data["pair"],
                    data["bar_idx"],
                    data["open"],
                    data["high"],
                    data["low"],
                    data["close"],
                    data["action"],
                    data["model"],
                    data["lots"],
                    data["equity"],
                    data["sentiment"],
                    data["latency_ms"],
                    data["var_pct"],
                    data.get("model_weights"),
                )
            )
        elif kind == "trade":
            trade_batch.append(
                (
                    data["timestamp"],
                    data["pair"],
                    data["event"],
                    data["action"],
                    data["lots"],
                    data["price"],
                    data["order_id"],
                    data["reason"],
                    data["pnl"],
                    data["details"],
                )
            )

    def _worker_loop(self) -> None:
        conn = None
        try:
            conn = duckdb.connect(str(self.db_path))
            self._create_tables(conn)
            self._conn = conn
            logger.info(f"[LiveDBSink] Connected to {self.db_path}")
            if self.http_port:
                self._start_http_server()
        except Exception as exc:
            logger.error(f"[LiveDBSink] Failed to connect/init tables at {self.db_path}: {exc}")
            return

        tick_batch: list[tuple] = []
        bar_batch: list[tuple] = []
        trade_batch: list[tuple] = []
        last_flush = time.time()

        try:
            while self._running or not self._queue.empty():
                try:
                    msg = self._queue.get(timeout=0.1)
                    if msg is None:
                        break

                    self._append_msg(msg, tick_batch, bar_batch, trade_batch)
                    now = time.time()
                    if (
                        len(tick_batch) >= self.batch_size
                        or bar_batch
                        or trade_batch
                        or (now - last_flush >= self.flush_interval_s and tick_batch)
                    ):
                        self._flush_batches(conn, tick_batch, bar_batch, trade_batch)
                        tick_batch.clear()
                        bar_batch.clear()
                        trade_batch.clear()
                        last_flush = now
                except queue.Empty:
                    now = time.time()
                    if tick_batch and (now - last_flush >= self.flush_interval_s):
                        self._flush_batches(conn, tick_batch, bar_batch, trade_batch)
                        tick_batch.clear()
                        bar_batch.clear()
                        trade_batch.clear()
                        last_flush = now

            # Final drain and flush on worker exit
            while True:
                try:
                    remaining_msg = self._queue.get_nowait()
                    if remaining_msg is not None:
                        self._append_msg(remaining_msg, tick_batch, bar_batch, trade_batch)
                except queue.Empty:
                    break

            if tick_batch or bar_batch or trade_batch:
                self._flush_batches(conn, tick_batch, bar_batch, trade_batch)
                tick_batch.clear()
                bar_batch.clear()
                trade_batch.clear()
        except Exception as exc:
            logger.exception(f"[LiveDBSink] Worker loop error: {exc}")
        finally:
            with self._db_lock:
                self._conn = None
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    logger.info(f"[LiveDBSink] Closed {self.db_path} connection cleanly.")

    def _start_http_server(self) -> None:
        if not self.http_port:
            return

        sink_self = self

        class Handler(_TelemetryHTTPHandler):
            sink = sink_self

        try:
            self._http_server = http.server.ThreadingHTTPServer(("127.0.0.1", self.http_port), Handler)
            t = threading.Thread(
                target=self._http_server.serve_forever,
                daemon=True,
                name="LiveDuckDBHTTPServer",
            )
            t.start()
            logger.info(f"[LiveDBSink] Telemetry API serving at http://127.0.0.1:{self.http_port}/")
        except Exception as exc:
            logger.warning(f"[LiveDBSink] Could not bind telemetry HTTP server to port {self.http_port}: {exc}")
            self._http_server = None

    def get_summary(self) -> dict[str, Any]:
        with self._db_lock:
            if self._conn is None:
                return {"status": "connecting", "ticks_count": 0, "bars_count": 0, "trades_count": 0}
            try:
                cur = self._conn.cursor()
                tick_count = cur.execute("SELECT count(*) FROM live_ticks").fetchone()[0]
                bar_count = cur.execute("SELECT count(*) FROM live_bars").fetchone()[0]
                trade_count = cur.execute("SELECT count(*) FROM live_trades").fetchone()[0]
                latest_ticks = cur.execute(
                    "SELECT timestamp, pair, bid, ask, mid, spread_pips FROM live_ticks ORDER BY timestamp DESC LIMIT 10"
                ).fetchall()
                latest_bars = cur.execute(
                    "SELECT timestamp, pair, bar_idx, close, action, model, sentiment, latency_ms, model_weights FROM live_bars ORDER BY timestamp DESC LIMIT 5"
                ).fetchall()
                latest_trades = cur.execute(
                    "SELECT timestamp, pair, event, action, lots, price, reason FROM live_trades ORDER BY timestamp DESC LIMIT 5"
                ).fetchall()
                return {
                    "status": "ok",
                    "ticks_count": tick_count,
                    "bars_count": bar_count,
                    "trades_count": trade_count,
                    "latest_ticks": latest_ticks,
                    "latest_bars": latest_bars,
                    "latest_trades": latest_trades,
                }
            except Exception as exc:
                return {"status": "error", "error": str(exc)}

    def query_sql(self, sql: str) -> dict[str, Any]:
        with self._db_lock:
            if self._conn is None:
                return {"status": "connecting", "columns": [], "rows": []}
            try:
                cur = self._conn.cursor()
                res = cur.execute(sql)
                cols = [desc[0] for desc in res.description] if res.description else []
                rows = res.fetchall()
                return {
                    "status": "ok",
                    "columns": cols,
                    "rows": rows,
                }
            except Exception as exc:
                return {"status": "error", "error": str(exc)}

    def _flush_batches(
        self,
        conn,
        ticks: list[tuple],
        bars: list[tuple],
        trades: list[tuple],
    ) -> None:
        if conn is None:
            return

        with self._db_lock:
            try:
                if ticks:
                    conn.executemany(
                        "INSERT INTO live_ticks VALUES (?, ?, ?, ?, ?, ?, ?)",
                        ticks,
                    )
                if bars:
                    conn.executemany(
                        "INSERT INTO live_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        bars,
                    )
                if trades:
                    conn.executemany(
                        "INSERT INTO live_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        trades,
                    )
            except Exception as exc:
                logger.error(f"[LiveDBSink] Flush error: {exc}")

    def close(self, timeout: float = 5.0) -> None:
        """Flushes remaining items and closes the database connection."""
        if not self.enabled:
            return

        self._running = False

        # 1. Stop HTTP server first to prevent new requests touching a closing connection
        if self._http_server is not None:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
            except Exception:
                pass
            self._http_server = None

        # 2. Put sentinel to worker queue
        try:
            self._queue.put(None, timeout=1.0)
        except Exception:
            pass

        # 3. Wait for worker thread to drain and cleanly close DB connection
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        self.enabled = False
        logger.info("[LiveDBSink] Closed database sink.")
