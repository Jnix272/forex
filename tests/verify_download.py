"""
tests/verify_download.py
========================
Live download verification for all configured forex pairs.

Downloads exactly 1 hour of tick data (2024-01-02 10:00 UTC) for every pair
in DUKA_PAIR_MAP, validates the data, and prints a per-pair pass/fail report.

Run:
    python tests/verify_download.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── ensure repo root is on path ───────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.sources import DUKA_PAIR_MAP, TICK_COLUMNS, DukascopyLoader

# ─────────────────────────────────────────────────────────────────────────────
# Expected price ranges  (min_bid, max_ask) — wide but realistic for 2024
# ─────────────────────────────────────────────────────────────────────────────
PRICE_BOUNDS: dict[str, tuple[float, float]] = {
    "EURUSD": (1.05,  1.15),
    "GBPUSD": (1.20,  1.35),
    "USDJPY": (130.0, 160.0),
    "AUDUSD": (0.60,  0.72),
    "USDCAD": (1.30,  1.40),
    "USDCHF": (0.83,  0.96),
    "NZDUSD": (0.57,  0.65),
    "EURGBP": (0.84,  0.90),
    "EURJPY": (138.0, 168.0),
}

# Download date/hour — choose a well-known liquid session (London open, Tue)
DATE  = "2024-01-02"
HOURS = [10]          # just 1 hour per pair to keep it fast


# ─────────────────────────────────────────────────────────────────────────────

def check_pair(loader: DukascopyLoader, pair: str) -> dict:
    """Download 1 hour for `pair` and return a result dict."""
    result = {
        "pair":     pair,
        "ok":       False,
        "ticks":    0,
        "elapsed":  0.0,
        "errors":   [],
    }

    t0 = time.perf_counter()
    try:
        df = loader.load(pair, start=DATE, end=DATE, hours=HOURS)
    except Exception as exc:
        result["errors"].append(f"load() raised {type(exc).__name__}: {exc}")
        result["elapsed"] = time.perf_counter() - t0
        return result
    result["elapsed"] = time.perf_counter() - t0

    # ── basic existence ──────────────────────────────────────────────────────
    if df.empty:
        result["errors"].append("DataFrame is empty — no ticks returned")
        return result

    result["ticks"] = len(df)

    # ── schema ───────────────────────────────────────────────────────────────
    missing_cols = [c for c in TICK_COLUMNS if c not in df.columns]
    if missing_cols:
        result["errors"].append(f"Missing columns: {missing_cols}")

    # ── UTC index ────────────────────────────────────────────────────────────
    if df.index.tz is None or str(df.index.tz) != "UTC":
        result["errors"].append(f"Index timezone is {df.index.tz!r}, expected UTC")

    # ── bid < ask (Golden Rule) ───────────────────────────────────────────────
    bad_spread = (df["ask"] <= df["bid"]).sum()
    if bad_spread:
        result["errors"].append(f"{bad_spread} rows where ask <= bid")

    # ── spread is positive ────────────────────────────────────────────────────
    zero_spread = (df["spread"] <= 0).sum()
    if zero_spread:
        result["errors"].append(f"{zero_spread} rows with zero/negative spread")

    # ── price sanity ─────────────────────────────────────────────────────────
    lo, hi = PRICE_BOUNDS.get(pair, (0.0, 999.0))
    if (df["bid"] < lo).any() or (df["ask"] > hi).any():
        actual_lo = df["bid"].min()
        actual_hi = df["ask"].max()
        result["errors"].append(
            f"Prices out of expected range [{lo}, {hi}]: got [{actual_lo:.5f}, {actual_hi:.5f}]"
        )

    # ── pair / source tags ────────────────────────────────────────────────────
    if "pair" in df.columns and (df["pair"] != pair).any():
        result["errors"].append(f"'pair' column contains unexpected values")
    if "source" in df.columns and (df["source"] != "dukascopy").any():
        result["errors"].append(f"'source' column is not 'dukascopy'")

    # ── NaN check ────────────────────────────────────────────────────────────
    for col in ("bid", "ask", "mid"):
        if col in df.columns and df[col].isna().any():
            result["errors"].append(f"NaN values found in '{col}'")

    result["ok"] = len(result["errors"]) == 0
    return result


def run():
    loader = DukascopyLoader(
        concurrency=16,
        max_retries=3,
        verbose=False,   # suppress per-file progress; we print our own
    )

    pairs = list(DUKA_PAIR_MAP.keys())
    print(f"\n{'-'*60}")
    print(f"  Dukascopy Live Download Check  |  {DATE}  |  hour {HOURS[0]:02d}:00 UTC")
    print(f"  Pairs: {', '.join(pairs)}")
    print(f"{'-'*60}\n")

    results = []
    all_ok = True

    for pair in pairs:
        print(f"  Downloading {pair}...", end="", flush=True)
        r = check_pair(loader, pair)
        results.append(r)

        status = "PASS" if r["ok"] else "FAIL"
        if not r["ok"]:
            all_ok = False

        tick_info = f"{r['ticks']:,} ticks" if r["ticks"] else "0 ticks"
        print(f"  [{status}]  {tick_info}  ({r['elapsed']:.1f}s)")
        for err in r["errors"]:
            print(f"            ERROR: {err}")

    # ── summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["ok"])
    failed = len(results) - passed
    total_ticks = sum(r["ticks"] for r in results)

    print(f"\n{'-'*60}")
    print(f"  Result : {'ALL PASSED' if all_ok else f'{failed} FAILED'}")
    print(f"  Pairs  : {passed}/{len(results)} passed")
    print(f"  Ticks  : {total_ticks:,} total ticks downloaded")
    print(f"{'-'*60}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    run()
