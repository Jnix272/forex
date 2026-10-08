import time

import pandas as pd

from trading.live_guards import EconomicCalendarGuard


def test_economic_calendar_guard_logic():
    guard = EconomicCalendarGuard("EURUSD", calendar_file="")

    # Mock events dataframe
    now = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    events = pd.DataFrame([
        {
            "timestamp_utc": "2026-01-01 12:15:00",
            "headline": "US CPI Inflation YoY",
            "event": "CPI",
            "impact": "high",
            "currency": "USD"
        },
        {
            "timestamp_utc": "2026-01-01 13:00:00",
            "headline": None,
            "event": "Nonfarm Payrolls",
            "impact": "high",
            "currency": "USD"
        },
        {
            "timestamp_utc": "2026-01-01 15:00:00",
            "event": "Low Impact Event",
            "impact": "low",
            "currency": "USD"
        }
    ])

    guard._events = events
    guard._loaded_at = now

    # Check at 12:00: US CPI at 12:15 is high impact (30m before window starts at 11:45, ends at 12:30).
    # So now (12:00) is inside [11:45, 12:30], should return block = True
    res = guard.check(now)
    assert res.blocked is True
    assert res.reason == "economic_calendar_block"
    assert res.details["tier"] in ("high", "critical")

    # Check at 10:00 (outside window)
    res_outside = guard.check(pd.Timestamp("2026-01-01 10:00:00", tz="UTC"))
    assert res_outside.blocked is False


def benchmark_economic_calendar_guard(iterations=500):
    guard = EconomicCalendarGuard("EURUSD", calendar_file="")

    # Generate 500 economic events
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=500, freq="15min", tz="UTC")
    events = pd.DataFrame({
        "timestamp_utc": timestamps,
        "headline": ["US CPI Inflation Rate" if i % 2 == 0 else None for i in range(500)],
        "event": ["CPI" for _ in range(500)],
        "impact": ["high" if i % 3 == 0 else "medium" if i % 3 == 1 else "low" for i in range(500)],
        "currency": ["USD" if i % 2 == 0 else "EUR" for i in range(500)]
    })

    now = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    guard._events = events
    guard._loaded_at = now

    t0 = time.perf_counter()
    for _ in range(iterations):
        guard.check(now)
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000.0
    avg_ms = total_ms / iterations
    print(f"\n[BENCHMARK] {iterations} iterations: Total = {total_ms:.2f} ms | Avg per call = {avg_ms:.4f} ms")
    return avg_ms


if __name__ == "__main__":
    benchmark_economic_calendar_guard()
