"""
tests/test_live_db_sink.py
==========================
Unit tests for LiveDuckDBSink ensuring non-blocking persistence of ticks, bars, and trades.
"""

import time
from datetime import UTC, datetime
from pathlib import Path
import duckdb
import pytest

from trading.live_db_sink import LiveDuckDBSink


def test_live_duckdb_sink_lifecycle(tmp_path: Path):
    db_file = tmp_path / "test_live.duckdb"
    sink = LiveDuckDBSink(db_path=db_file, batch_size=10, flush_interval_s=0.2)

    assert sink.enabled is True
    time.sleep(0.1)
    assert db_file.exists()

    # 1. Record ticks
    now = datetime.now(UTC)
    for i in range(15):
        sink.record_tick(
            pair="EURUSD",
            bid=1.0850 + i * 0.0001,
            ask=1.0852 + i * 0.0001,
            mid=1.0851 + i * 0.0001,
            spread_pips=2.0,
            volume=1.0,
            timestamp=now,
        )

    # 2. Record bar
    sink.record_bar(
        pair="EURUSD",
        bar_idx=1,
        open_=1.0850,
        high=1.0860,
        low=1.0845,
        close=1.0855,
        action=1,
        model="slow_haelt",
        lots=0.1,
        equity=100000.0,
        sentiment=0.45,
        latency_ms=3.2,
        var_pct=0.015,
        timestamp=now,
    )

    # 3. Record trade
    sink.record_trade(
        pair="EURUSD",
        event="fill",
        action="BUY",
        lots=0.1,
        price=1.0852,
        order_id="TEST-123",
        reason="model_signal",
        pnl=0.0,
        details={"confidence": 0.82},
        timestamp=now,
    )

    # Allow worker to flush or close directly
    sink.close(timeout=5.0)
    assert sink.enabled is False

    # 4. Verify directly with duckdb connection
    conn = duckdb.connect(str(db_file))

    ticks_count = conn.execute("SELECT count(*) FROM live_ticks").fetchone()[0]
    assert ticks_count == 15

    bars = conn.execute("SELECT pair, bar_idx, close, action, model FROM live_bars").fetchall()
    assert len(bars) == 1
    assert bars[0] == ("EURUSD", 1, 1.0855, 1, "slow_haelt")

    trades = conn.execute("SELECT pair, event, action, price, order_id FROM live_trades").fetchall()
    assert len(trades) == 1
    assert trades[0] == ("EURUSD", "fill", "BUY", 1.0852, "TEST-123")

    conn.close()


def test_live_duckdb_sink_http_api(tmp_path: Path):
    import json
    import urllib.request

    db_file = tmp_path / "test_http.duckdb"
    port = 8099
    sink = LiveDuckDBSink(db_path=db_file, batch_size=5, flush_interval_s=0.1, http_port=port)
    time.sleep(0.2)

    try:
        sink.record_tick("EURUSD", 1.0850, 1.0852)
        sink.record_bar(
            pair="EURUSD",
            bar_idx=1,
            open_=1.0850,
            high=1.0860,
            low=1.0845,
            close=1.0855,
            action=1,
            model="haelt",
            lots=0.1,
            equity=10000.0,
        )
        time.sleep(0.3)

        # 1. Query summary
        req = urllib.request.Request(f"http://127.0.0.1:{port}/summary")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["ticks_count"] >= 1
            assert data["bars_count"] >= 1

        # 2. Query custom SQL
        req = urllib.request.Request(f"http://127.0.0.1:{port}/query?sql=SELECT+pair,+close+FROM+live_bars")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert len(data["rows"]) >= 1
            assert data["rows"][0][0] == "EURUSD"

        # 3. Disallowed multi-statement / DROP injection test
        import urllib.error
        req_bad = urllib.request.Request(f"http://127.0.0.1:{port}/query?sql=SELECT+1;+DROP+TABLE+live_ticks")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_bad, timeout=2.0)
        assert exc_info.value.code == 400
    finally:
        sink.close()
