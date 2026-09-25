"""OANDABroker pure-Python pricing stream against a local fake v20 stream."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class _StreamHandler(BaseHTTPRequestHandler):
    price = {"type": "PRICE", "instrument": "EUR_USD", "tradeable": True,
             "bids": [{"price": "1.10000"}], "asks": [{"price": "1.10012"}]}

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        self.server.paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        try:
            self.wfile.write((json.dumps({"type": "HEARTBEAT"}) + "\n").encode())
            self.wfile.write((json.dumps(self.price) + "\n").encode())
            self.wfile.flush()
            while not self.server.stop.is_set():  # keep the stream open with heartbeats
                time.sleep(0.2)
                self.wfile.write((json.dumps({"type": "HEARTBEAT"}) + "\n").encode())
                self.wfile.flush()
        except Exception:
            pass


@pytest.fixture()
def fake_stream(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StreamHandler)
    srv.paths, srv.stop = [], threading.Event()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("OANDA_API_KEY", "test-token")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "001-test")
    monkeypatch.setenv("OANDA_STREAM_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.delenv("OANDA_ZMQ_ENDPOINT", raising=False)
    monkeypatch.delenv("OANDA_STREAM", raising=False)
    yield srv
    srv.stop.set()
    srv.shutdown()


def test_stream_quotes_served_without_rest(fake_stream):
    from trading.live_engine import OANDABroker

    b = OANDABroker()
    b.ensure_stream("EURUSD")
    deadline = time.time() + 5
    while time.time() < deadline and b._stream_bid_ask("EURUSD") is None:
        time.sleep(0.05)
    assert b.streaming
    assert b.get_bid_ask("EURUSD") == (1.10000, 1.10012)
    assert "/pricing/stream?instruments=EUR_USD" in fake_stream.paths[0]
    b.disconnect()


def test_adding_a_pair_restarts_stream_with_all_instruments(fake_stream):
    from trading.live_engine import OANDABroker

    b = OANDABroker()
    b.ensure_stream("EURUSD")
    b.ensure_stream("USDJPY")
    deadline = time.time() + 5
    while time.time() < deadline and not any("USD_JPY" in p for p in fake_stream.paths):
        time.sleep(0.05)
    assert any("instruments=EUR_USD,USD_JPY" in p for p in fake_stream.paths)
    b.disconnect()


def test_stream_disabled_by_env(fake_stream, monkeypatch):
    from trading.live_engine import OANDABroker

    monkeypatch.setenv("OANDA_STREAM", "0")
    b = OANDABroker()
    b.ensure_stream("EURUSD")
    assert b._stream_thread is None and not b.streaming
