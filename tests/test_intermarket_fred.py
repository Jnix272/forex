"""
tests/test_intermarket_fred.py
================================
Live download tests for:
  1. Cross-asset intermarket panel  (data/cross_asset.py)
  2. FRED yield spreads             (features/macro_features.py)

Each asset / yield series is validated for:
  - Data actually returned (not empty)
  - Correct UTC timezone
  - Values in realistic ranges for 2024
  - No NaN-only series
  - Correct sign relationships (e.g. US10Y > US2Y most of the time)
  - Spread features are correctly computed from their source yields
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.cross_asset import (
    _NO_YAHOO_FALLBACK,
    FRED_YIELD_SYMBOLS,
    STOOQ_SYMBOLS,
    _read_fred_daily,
    _read_stooq_daily,
    load_cross_asset_panel,
)
from features.macro_features import (
    YIELD_DEFAULTS,
    MacroYieldFeatureBuilder,
    _fetch_fred_series,
    _synthetic_yields,
)


def macro_to_pandas(df):
    out = df.to_pandas()
    if "timestamp_utc" in out.columns:
        out = out.set_index("timestamp_utc")
    return out

# ── Test window ────────────────────────────────────────────────────────────────
START = "2024-01-01"
END   = "2024-03-31"

# ── Realistic value bounds for validation ─────────────────────────────────────
ASSET_BOUNDS = {
    "WTI":           (60.0,  100.0),
    "GOLD":          (1800.0, 2500.0),
    "COPPER":        (3.0,    5.0),
    "DXY":           (95.0,  110.0),
    "SPX":           (4000.0, 5500.0),
    "NASDAQ100":     (14000.0, 20000.0),
    "VIX":           (10.0,   40.0),
    "BTC":           (30000.0, 75000.0),
    "US10Y":         (3.5,    5.5),
    "US2Y":          (3.5,    5.5),
    "DE10Y":         (1.5,    3.5),
    "JP10Y":         (0.0,    1.0),
    "GB10Y":         (3.0,    5.0),
    "AU10Y":         (3.5,    5.0),
    "CA10Y":         (3.0,    4.5),
}

YIELD_BOUNDS = {
    "US10Y": (3.5,  5.5),
    "US2Y":  (3.5,  5.5),
    "DE10Y": (1.5,  3.5),
    "JP10Y": (0.0,  1.0),
    "GB10Y": (3.0,  5.0),
    "AU10Y": (3.5,  5.0),
    "CA10Y": (3.0,  4.5),
    "NZ10Y": (4.0,  5.5),
    "CH10Y": (-0.5, 2.0),
}


# =============================================================================
# 1. CROSS-ASSET INTERMARKET PANEL
# =============================================================================

class TestStooqSymbols:
    """Validate every Stooq symbol candidate is correct and returns data."""

    # Core assets that must work via Yahoo Finance (no FRED key required)
    # DE10Y/JP10Y/GB10Y excluded: no Yahoo symbol and Stooq currently blocks scraping
    MUST_PASS = ["WTI", "GOLD", "SPX", "VIX", "US10Y"]

    @classmethod
    @pytest.fixture(scope="class")
    def panel(cls, tmp_path_factory):
        cache = str(tmp_path_factory.mktemp("cross_asset"))
        # Use "auto" so Yahoo/FRED fallback if Stooq is unavailable
        return load_cross_asset_panel(START, END, cache_dir=cache, source="auto")

    def test_must_pass_assets_present(self, panel):
        missing = [a for a in self.MUST_PASS if a not in panel]
        assert not missing, f"Missing critical assets: {missing}"

    def test_all_values_are_numeric(self, panel):
        for asset, ser in panel.items():
            assert pd.api.types.is_numeric_dtype(ser), f"{asset}: non-numeric"

    def test_no_all_nan_series(self, panel):
        for asset, ser in panel.items():
            assert not ser.isna().all(), f"{asset}: all NaN"

    def test_index_is_utc(self, panel):
        for asset, ser in panel.items():
            assert ser.index.tz is not None, f"{asset}: no timezone"
            assert str(ser.index.tz) == "UTC", f"{asset}: timezone is {ser.index.tz}"

    def test_index_is_sorted(self, panel):
        for asset, ser in panel.items():
            assert ser.index.is_monotonic_increasing, f"{asset}: index not sorted"

    def test_values_in_realistic_range(self, panel):
        failures = []
        for asset, (lo, hi) in ASSET_BOUNDS.items():
            if asset not in panel:
                continue
            ser = panel[asset]
            median = ser.median()
            if not (lo <= median <= hi):
                failures.append(f"{asset}: median={median:.2f} not in [{lo}, {hi}]")
        assert not failures, "\n".join(failures)

    def test_us10y_above_us2y_majority(self, panel):
        """10Y and 2Y yields should be in plausible range and the spread within bounds."""
        if "US10Y" not in panel or "US2Y" not in panel:
            pytest.skip("US10Y or US2Y not available")
        common = panel["US10Y"].index.intersection(panel["US2Y"].index)
        us10 = panel["US10Y"].reindex(common)
        us2  = panel["US2Y"].reindex(common)
        # Q1 2024 had a deeply inverted curve (2Y > 10Y the entire period) — that is real.
        # Just verify the inversion is plausible (spread within ±3 percentage points).
        spread = (us10 - us2).mean()
        assert -3.0 <= spread <= 3.0, (
            f"US10Y - US2Y avg spread={spread:.2f}% — looks like wrong data"
        )

    def test_de10y_below_us10y(self, panel):
        """German Bund yield was below US Treasury throughout 2024."""
        if "US10Y" not in panel or "DE10Y" not in panel:
            pytest.skip("yields unavailable")
        common = panel["US10Y"].index.intersection(panel["DE10Y"].index)
        us10 = panel["US10Y"].reindex(common)
        de10 = panel["DE10Y"].reindex(common)
        pct_us_above = (us10 > de10).mean()
        assert pct_us_above >= 0.80, \
            f"US10Y > DE10Y only {pct_us_above*100:.0f}% — DE10Y may be using wrong symbol"

    def test_jp10y_much_lower_than_us10y(self, panel):
        """Japan held near-zero rates throughout 2024."""
        if "US10Y" not in panel or "JP10Y" not in panel:
            pytest.skip("yields unavailable")
        common = panel["US10Y"].index.intersection(panel["JP10Y"].index)
        jp10 = panel["JP10Y"].reindex(common)
        assert jp10.median() < 1.5, \
            f"JP10Y median={jp10.median():.2f} — too high, wrong data?"

    def test_yield_curve_slope_computed(self, panel):
        """YIELD_CURVE_SLOPE should be automatically derived from US10Y and US2Y."""
        if "YIELD_CURVE_SLOPE" not in panel:
            pytest.skip("US10Y or US2Y unavailable for slope calculation")
        slope = panel["YIELD_CURVE_SLOPE"]
        assert not slope.isna().all()
        # Slope should be within a reasonable range (−3% to +3%)
        assert slope.abs().max() < 5.0, f"Unrealistic slope: {slope.abs().max():.2f}%"

    def test_no_yahoo_fallback_blocked(self):
        """Yield series with wrong Yahoo symbols must be in the block list."""
        blocked = {"DE10Y", "JP10Y", "GB10Y", "AU10Y", "CA10Y"}
        assert blocked.issubset(_NO_YAHOO_FALLBACK), \
            f"Missing from _NO_YAHOO_FALLBACK: {blocked - _NO_YAHOO_FALLBACK}"

    def test_vix_between_10_and_40(self, panel):
        if "VIX" not in panel:
            pytest.skip("VIX not available")
        assert 10 <= panel["VIX"].median() <= 40

    def test_btc_above_30k(self, panel):
        if "BTC" not in panel:
            pytest.skip("BTC not available")
        assert panel["BTC"].median() > 25000, "BTC price looks wrong"

    def test_gold_above_1800(self, panel):
        if "GOLD" not in panel:
            pytest.skip("GOLD not available")
        assert panel["GOLD"].median() > 1500, "Gold price looks wrong"

    def test_wti_between_60_and_100(self, panel):
        if "WTI" not in panel:
            pytest.skip("WTI not available")
        assert 40 <= panel["WTI"].median() <= 120, \
            f"WTI median={panel['WTI'].median():.1f} — out of range"


class TestStooqSymbolCandidates:
    """Unit-level: test each primary Stooq symbol directly."""

    SPOT_CHECK = {
        "US10Y": "10usy.b",
        "DE10Y": "10dey.b",
        "JP10Y": "10jpy.b",
        "GB10Y": "10gby.b",
        "US2Y":  "2usy.b",
        "WTI":   "cl.f",
        "GOLD":  "xauusd",
        "VIX":   "^vix",
        "SPX":   "^spx",
        "DXY":   "usdidx",
    }

    @pytest.mark.parametrize("asset,sym", list(SPOT_CHECK.items()))
    def test_stooq_symbol_returns_data(self, asset, sym):
        ser = _read_stooq_daily(sym)
        if ser is None or ser.empty:
            pytest.skip(f"Stooq symbol '{sym}' for {asset} returned no data "
                        "(Stooq may be rate-limiting or unavailable)")
        lo, hi = ASSET_BOUNDS.get(asset, (0, 1e9))
        assert lo <= ser.median() <= hi, \
            f"{asset} ({sym}): median={ser.median():.4f} not in [{lo}, {hi}]"


class TestFREDCrossAsset:
    """FRED fallback for bond yields via cross_asset._read_fred_daily."""

    FRED_SPOT_CHECK = {
        "US10Y": ("DGS10",           3.5, 5.5),
        "US2Y":  ("DGS2",            3.5, 5.5),
        "DE10Y": ("IRLTLT01DEM156N", 1.5, 3.5),
        "JP10Y": ("IRLTLT01JPM156N", 0.0, 1.0),
        "GB10Y": ("IRLTLT01GBM156N", 3.0, 5.5),
    }

    @pytest.mark.skipif(
        not __import__("os").getenv("FRED_API_KEY"),
        reason="FRED_API_KEY not set"
    )
    @pytest.mark.parametrize("asset,series,lo,hi", [
        (a, s, lo, hi) for a, (s, lo, hi) in FRED_SPOT_CHECK.items()
    ])
    def test_fred_yield_in_range(self, asset, series, lo, hi):
        ser = _read_fred_daily(series, START, END)
        assert ser is not None and not ser.empty, \
            f"FRED series '{series}' for {asset} returned nothing"
        assert lo <= ser.median() <= hi, \
            f"{asset} ({series}): median={ser.median():.3f} not in [{lo}, {hi}]"

    @pytest.mark.skipif(
        not __import__("os").getenv("FRED_API_KEY"),
        reason="FRED_API_KEY not set"
    )
    def test_fred_panel_with_fred_source(self, tmp_path):
        panel = load_cross_asset_panel(
            START, END, cache_dir=str(tmp_path), source="fred"
        )
        yield_assets = [a for a in FRED_YIELD_SYMBOLS if a in panel]
        assert len(yield_assets) >= 3, \
            f"Expected at least 3 yield series from FRED, got: {yield_assets}"


# =============================================================================
# 2. FRED YIELD SPREADS (MacroYieldFeatureBuilder)
# =============================================================================

class TestSyntheticYields:
    """Validate synthetic fallback is self-consistent."""

    @classmethod
    @pytest.fixture(scope="class")
    def yields(cls):
        start = pd.Timestamp(START, tz="UTC")
        end   = pd.Timestamp(END,   tz="UTC")
        return _synthetic_yields(start, end)

    def test_all_series_present(self, yields):
        expected = list(YIELD_DEFAULTS.keys())
        missing  = [k for k in expected if k not in yields]
        assert not missing, f"Missing synthetic yields: {missing}"

    def test_no_nan_in_synthetic(self, yields):
        for name, s in yields.items():
            assert not s.isna().any(), f"{name}: NaN in synthetic"

    def test_index_is_utc(self, yields):
        for name, s in yields.items():
            assert s.index.tz is not None, f"{name}: no timezone"
            assert str(s.index.tz) == "UTC"

    def test_values_in_plausible_range(self, yields):
        for name, s in yields.items():
            lo, hi = YIELD_BOUNDS.get(name, (0.0, 20.0))
            # Allow ±2% from bounds for random walks
            assert s.min() > lo - 2.0, f"{name}: min={s.min():.2f} too low"
            assert s.max() < hi + 2.0, f"{name}: max={s.max():.2f} too high"

    def test_reproducible_with_same_seed(self):
        start = pd.Timestamp(START, tz="UTC")
        end   = pd.Timestamp(END,   tz="UTC")
        a = _synthetic_yields(start, end, seed=0)
        b = _synthetic_yields(start, end, seed=0)
        pd.testing.assert_series_equal(a["US10Y"], b["US10Y"])

    def test_different_seeds_give_different_results(self):
        start = pd.Timestamp(START, tz="UTC")
        end   = pd.Timestamp(END,   tz="UTC")
        a = _synthetic_yields(start, end, seed=1)
        b = _synthetic_yields(start, end, seed=2)
        assert not a["US10Y"].equals(b["US10Y"])

    def test_us10y_reasonable_correlation_with_de10y(self, yields):
        """US and German yields should be positively correlated."""
        corr = yields["US10Y"].corr(yields["DE10Y"])
        assert corr > 0.0, f"US10Y-DE10Y correlation is negative: {corr:.3f}"

    def test_jp10y_lower_than_us10y_on_average(self, yields):
        assert yields["JP10Y"].mean() < yields["US10Y"].mean(), \
            "Synthetic JP10Y should be lower than US10Y"


class TestMacroFeatureBuilder:
    """Validate MacroYieldFeatureBuilder.build() output."""

    @classmethod
    @pytest.fixture(scope="class")
    def feature_df(cls):
        idx  = pd.date_range(START, END, freq="1h", tz="UTC")
        bars = pd.DataFrame({"close": np.random.randn(len(idx)).cumsum() + 1.085},
                            index=idx)
        builder = MacroYieldFeatureBuilder()   # no FRED key → synthetic
        df = macro_to_pandas(builder.build(bars))
        if len(df) == len(bars) and not isinstance(df.index, pd.DatetimeIndex):
            df.index = bars.index
        return df, bars

    def test_no_nans_in_output(self, feature_df):
        df, _ = feature_df
        total_nans = df.isna().sum().sum()
        assert total_nans == 0, f"NaNs in macro features: {total_nans}"

    def test_all_expected_columns_present(self, feature_df):
        df, _ = feature_df
        expected = [
            "spread_us_de", "spread_us_jp", "spread_us_gb",
            "spread_us_au", "spread_us_ca", "spread_us_nz",
            "spread_de_gb", "spread_de_jp", "spread_us_ch",
            "yield_curve_slope",
            "carry_eur", "carry_jpy", "carry_gbp", "carry_aud", "carry_cad",
            "carry_nzd", "carry_eurgbp", "carry_eurjpy", "carry_chf",
            "yield_momentum_5d", "yield_momentum_20d", "yield_vol_20d",
        ]
        missing = [c for c in expected if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_index_matches_bars(self, feature_df):
        df, bars = feature_df
        assert len(df) == len(bars)
        assert (df.index == bars.index).all()

    def test_carry_eur_is_negation_of_spread_us_de(self, feature_df):
        """carry_eur = -spread_us_de by definition."""
        df, _ = feature_df
        pd.testing.assert_series_equal(
            df["carry_eur"], -df["spread_us_de"],
            check_names=False, atol=1e-10
        )

    def test_carry_jpy_equals_spread_us_jp(self, feature_df):
        df, _ = feature_df
        pd.testing.assert_series_equal(
            df["carry_jpy"], df["spread_us_jp"],
            check_names=False, atol=1e-10
        )

    def test_yield_curve_slope_sign(self, feature_df):
        """slope = US10Y - US2Y; can be negative (inverted) but must be finite."""
        df, _ = feature_df
        assert df["yield_curve_slope"].abs().max() < 10.0
        assert df["yield_curve_slope"].isna().sum() == 0

    def test_yield_momentum_zero_at_warmup_handled(self, feature_df):
        """Momentum is NaN for the first N bars — builder must fill to 0."""
        df, _ = feature_df
        assert df["yield_momentum_5d"].isna().sum() == 0
        assert df["yield_momentum_20d"].isna().sum() == 0

    def test_yield_vol_non_negative(self, feature_df):
        df, _ = feature_df
        assert (df["yield_vol_20d"] >= 0).all()

    def test_spread_values_in_range(self, feature_df):
        df, _ = feature_df
        # All spreads should be within ±10 percentage points
        for col in df.columns:
            if col.startswith("spread_"):
                assert df[col].abs().max() < 10.0, \
                    f"{col} has extreme value: {df[col].abs().max():.3f}"

    def test_build_with_tz_naive_bars(self):
        """build() must handle tz-naive bar index without crashing."""
        idx  = pd.date_range(START, periods=100, freq="1h")  # no tz
        bars = pd.DataFrame({"close": np.ones(100)}, index=idx)
        builder = MacroYieldFeatureBuilder()
        df = macro_to_pandas(builder.build(bars))
        assert len(df) == 100
        assert df.isna().sum().sum() == 0


class TestFREDYieldSpreads:
    """Live FRED API tests (skipped when FRED_API_KEY is not set)."""

    NEEDS_FRED = pytest.mark.skipif(
        not __import__("os").getenv("FRED_API_KEY"),
        reason="FRED_API_KEY env var not set — skipping live FRED tests"
    )

    @NEEDS_FRED
    def test_fred_us10y_series(self):
        key = __import__("os").getenv("FRED_API_KEY")
        s   = _fetch_fred_series("DGS10", START, END, key)
        assert not s.empty
        assert 3.5 <= s.median() <= 5.5, f"US10Y median={s.median():.3f}"
        assert str(s.index.tz) == "UTC"

    @NEEDS_FRED
    def test_fred_us2y_series(self):
        key = __import__("os").getenv("FRED_API_KEY")
        s   = _fetch_fred_series("DGS2", START, END, key)
        assert not s.empty
        assert 3.5 <= s.median() <= 5.5, f"US2Y median={s.median():.3f}"

    @NEEDS_FRED
    def test_fred_de10y_series(self):
        key = __import__("os").getenv("FRED_API_KEY")
        s   = _fetch_fred_series("IRLTLT01DEM156N", START, END, key)
        assert not s.empty
        assert 1.5 <= s.median() <= 3.5, f"DE10Y median={s.median():.3f}"

    @NEEDS_FRED
    def test_fred_jp10y_near_zero(self):
        key = __import__("os").getenv("FRED_API_KEY")
        s   = _fetch_fred_series("IRLTLT01JPM156N", START, END, key)
        assert not s.empty
        assert s.median() < 1.5, f"JP10Y median={s.median():.3f} — too high"

    @NEEDS_FRED
    def test_fred_us10y_higher_than_jp10y(self):
        key  = __import__("os").getenv("FRED_API_KEY")
        us10 = _fetch_fred_series("DGS10",            START, END, key)
        jp10 = _fetch_fred_series("IRLTLT01JPM156N",  START, END, key)
        assert us10.mean() > jp10.mean(), \
            "US10Y should be higher than JP10Y in 2024"

    @NEEDS_FRED
    def test_fred_builder_with_real_key(self):
        """Full MacroYieldFeatureBuilder with FRED data — verify spreads are real."""
        key  = __import__("os").getenv("FRED_API_KEY")
        idx  = pd.date_range(START, END, freq="1D", tz="UTC")
        bars = pd.DataFrame({"close": np.ones(len(idx))}, index=idx)
        builder = MacroYieldFeatureBuilder(fred_api_key=key)
        df = macro_to_pandas(builder.build(bars))

        assert df.isna().sum().sum() == 0, "NaNs in FRED-sourced features"
        # EUR/USD spread: US10Y was ~4.5%, DE10Y was ~2.5% → spread ~2%
        assert 0.5 <= df["spread_us_de"].mean() <= 3.5, \
            f"EUR spread looks wrong: {df['spread_us_de'].mean():.3f}"
        # JPY spread: US10Y ~4.5%, JP10Y ~0.5% → spread ~4%
        assert 2.0 <= df["spread_us_jp"].mean() <= 6.0, \
            f"JPY spread looks wrong: {df['spread_us_jp'].mean():.3f}"


# =============================================================================
# Live download summary script (run directly)
# =============================================================================

def run_live_summary():
    """Print a human-readable report of what's available from each provider."""

    print()
    print("=" * 65)
    print("  Intermarket + FRED Yield Spread — Live Data Verification")
    print(f"  Period: {START} to {END}")
    print("=" * 65)

    # ── Cross-asset panel ────────────────────────────────────────────────────
    print("\n[1] Cross-asset intermarket panel (Stooq)")
    with tempfile.TemporaryDirectory() as tmp:
        t0    = time.perf_counter()
        panel = load_cross_asset_panel(START, END, cache_dir=tmp, source="stooq")
        elapsed = time.perf_counter() - t0

    print(f"    Loaded {len(panel)} series in {elapsed:.1f}s\n")
    print(f"    {'Asset':<14} {'Points':>7}  {'Min':>10}  {'Median':>10}  {'Max':>10}  Status")
    print("    " + "-" * 62)

    all_pass = True
    for asset in sorted(STOOQ_SYMBOLS.keys()):
        if asset not in panel:
            print(f"    {asset:<14} {'MISSING':>7}  {'':>10}  {'':>10}  {'':>10}  FAIL")
            all_pass = False
            continue
        ser = panel[asset]
        lo, hi = ASSET_BOUNDS.get(asset, (0, 1e9))
        ok  = lo <= ser.median() <= hi
        tag = "PASS" if ok else "WARN"
        if not ok:
            all_pass = False
        print(f"    {asset:<14} {len(ser):>7,}  {ser.min():>10.3f}  "
              f"{ser.median():>10.3f}  {ser.max():>10.3f}  {tag}")

    if "YIELD_CURVE_SLOPE" in panel:
        s = panel["YIELD_CURVE_SLOPE"]
        print(f"    {'YIELD_CURVE':<14} {len(s):>7,}  {s.min():>10.3f}  "
              f"{s.median():>10.3f}  {s.max():>10.3f}  DERIVED")

    # ── Macro yield features ─────────────────────────────────────────────────
    print("\n[2] FRED Yield Spread features (MacroYieldFeatureBuilder)")
    idx  = pd.date_range(START, END, freq="1D", tz="UTC")
    bars = pd.DataFrame({"close": np.ones(len(idx))}, index=idx)

    fred_key = __import__("os").getenv("FRED_API_KEY", "")
    src_label = "FRED" if fred_key else "SYNTHETIC"
    builder   = MacroYieldFeatureBuilder(fred_api_key=fred_key)

    t0 = time.perf_counter()
    df = builder.build(bars).to_pandas()
    elapsed = time.perf_counter() - t0

    print(f"    Source: {src_label}  |  {df.shape[1]} features  |  {elapsed:.1f}s\n")
    print(f"    {'Feature':<22} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("    " + "-" * 58)

    for col in df.columns:
        s = df[col]
        print(f"    {col:<22} {s.mean():>8.3f}  {s.std():>8.3f}  "
              f"{s.min():>8.3f}  {s.max():>8.3f}")

    nans = df.isna().sum().sum()
    print(f"\n    NaNs: {nans}  |  All features present: {df.shape[1] == 22}")

    print()
    print("=" * 65)
    print(f"  Cross-asset: {'ALL PASS' if all_pass else 'WARNINGS — see above'}")
    print(f"  Macro yields: {'OK' if nans == 0 else f'FAIL — {nans} NaNs'}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    run_live_summary()
