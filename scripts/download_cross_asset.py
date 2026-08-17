#!/usr/bin/env python3
"""
Download free cross-asset data from EODHD (indices, commodities, bonds).

Free tier: 20 requests/day, daily OHLCV.
Key symbols for forex: DXY, SPX, GOLD, WTI, US10Y, US2Y, VIX, EUR, JPY, GBP

Usage:
    export EODHD_API_KEY="your_key"
    python scripts/download_cross_asset.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.eodhd import EODHDLoader

# Key cross-asset symbols for forex trading
CROSS_ASSET_SYMBOLS = {
    # USD Dollar Index
    "DXY": "DXY.INDX",
    # Equity Indices
    "SPX": "SPX.INDX",  # S&P 500
    "NDX": "NDX.INDX",  # Nasdaq 100
    "DJI": "DJI.INDX",  # Dow Jones
    "VIX": "VIX.INDX",  # Volatility Index
    # Commodities
    "GOLD": "XAUUSD.FOREX",  # Gold
    "SILVER": "XAGUSD.FOREX",  # Silver
    "WTI": "CL.FUTURES",  # WTI Crude Oil
    "BRENT": "BZ.FUTURES",  # Brent Crude
    "COPPER": "HG.FUTURES",  # Copper
    # Bonds / Yields
    "US10Y": "US10Y.BOND",  # US 10-Year
    "US2Y": "US2Y.BOND",  # US 2-Year
    "US30Y": "US30Y.BOND",  # US 30-Year
    "DE10Y": "DE10Y.BOND",  # German 10-Year (Bund)
    "JP10Y": "JP10Y.BOND",  # Japanese 10-Year (JGB)
    # Currency Indices (if available)
    "EUR_IDX": "EUR.INDX",
    "JPY_IDX": "JPY.INDX",
    "GBP_IDX": "GBP.INDX",
}


def main():
    parser = argparse.ArgumentParser(description="Download cross-asset data from EODHD")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(CROSS_ASSET_SYMBOLS.keys()),
        help="Symbols to download (default: all standard forex cross-assets)",
    )
    parser.add_argument("--start", default="2010-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument(
        "--api-key", default=os.getenv("EODHD_API_KEY"), help="EODHD API key (or set EODHD_API_KEY env var)"
    )
    parser.add_argument("--cache-dir", default="data/raw/cross_asset", help="Cache directory")
    parser.add_argument("--list-symbols", action="store_true", help="List available symbols and exit")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress")

    args = parser.parse_args()

    if args.list_symbols:
        print("Available cross-asset symbols:")
        for name, symbol in CROSS_ASSET_SYMBOLS.items():
            print(f"  {name:10s} -> {symbol}")
        return

    if not args.api_key:
        print("❌ ERROR: EODHD_API_KEY not set!")
        print("   Get free key at: https://eodhistoricaldata.com/")
        print("   Then: export EODHD_API_KEY='your_key'")
        sys.exit(1)

    print(f"{'=' * 60}")
    print("DOWNLOAD CROSS-ASSET DATA (EODHD)")
    print(f"{'=' * 60}")
    print(f"Symbols:   {', '.join(args.symbols)}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Cache dir:  {args.cache_dir}")
    print(f"{'=' * 60}\n")

    loader = EODHDLoader(
        api_key=args.api_key,
        cache_dir=args.cache_dir,
        verbose=not args.quiet,
    )

    failed = []
    success = []

    for name in args.symbols:
        symbol = CROSS_ASSET_SYMBOLS.get(name, name)
        try:
            if not args.quiet:
                print(f"Downloading {name} ({symbol})...", end=" ", flush=True)

            df = loader.load(symbol, start=args.start, end=args.end)

            if df.empty:
                print("❌ NO DATA")
                failed.append((name, "No data returned"))
            else:
                print(f"✅ {len(df):,} rows")
                success.append((name, len(df)))

        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed.append((name, str(e)))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Success: {len(success)}")
    for name, rows in success:
        print(f"  {name}: {rows:,} rows")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for name, error in failed:
            print(f"  {name}: {error}")

    print(f"\nData saved to: {args.cache_dir}/")
    print("Use in training with: --cross-asset-mode real --cross-asset-source eodhd")


if __name__ == "__main__":
    main()
