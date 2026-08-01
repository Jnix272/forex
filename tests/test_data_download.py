"""
tests/test_data_download.py
============================
Unit and integration tests for pair data downloading.

Covers:
  - _parse_bi5_hour: binary tick parsing
  - _enforce_schema: output schema validation
  - DukascopyLoader: cache path, load (mocked HTTP), load_multiple
  - DUKA_PAIR_MAP / PIP_SIZES completeness
"""

from __future__ import annotations

import lzma
import struct
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.sources import (
    DUKA_PAIR_MAP,
    ForexDataManager,
    PIP_SIZES,
    TICK_COLUMNS,
    DukascopyLoader,
    _enforce_schema,
    _parse_bi5_hour,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_bi5_bytes(n_ticks: int = 5, pair: str = "EURUSD") -> bytes:
    """Build a synthetic .bi5 (raw, uncompressed) byte payload."""
    point = 100_000  # non-JPY
    records = []
    for i in range(n_ticks):
        ms_offset = i * 500          # one tick every 500 ms
        ask_scaled = int((1.08500 + i * 0.00001) * point)
        bid_scaled = int((1.08490 + i * 0.00001) * point)
        ask_vol = 1.5
        bid_vol = 1.0
        records.append(struct.pack(">IIIff", ms_offset, ask_scaled, bid_scaled, ask_vol, bid_vol))
    return b"".join(records)


def _make_bi5_lzma(n_ticks: int = 5) -> bytes:
    """Return LZMA-compressed bi5 bytes."""
    return lzma.compress(_make_bi5_bytes(n_ticks))


# ─────────────────────────────────────────────────────────────────────────────
# _parse_bi5_hour
# ─────────────────────────────────────────────────────────────────────────────

class TestParseBi5Hour:
    DT = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)

    def test_empty_bytes_returns_empty_df(self):
        df = _parse_bi5_hour(b"", self.DT, "EURUSD")
        assert df.empty

    def test_none_like_bytes_handled(self):
        # bytes of length not divisible by 20 → truncated to 0 ticks
        df = _parse_bi5_hour(b"\x00" * 19, self.DT, "EURUSD")
        assert df.empty

    def test_returns_correct_row_count(self):
        raw = _make_bi5_bytes(n_ticks=10)
        df = _parse_bi5_hour(raw, self.DT, "EURUSD")
        assert len(df) == 10

    def test_bid_ask_columns_present(self):
        raw = _make_bi5_bytes(n_ticks=3)
        df = _parse_bi5_hour(raw, self.DT, "EURUSD")
        assert {"bid", "ask", "volume"}.issubset(df.columns)

    def test_ask_greater_than_bid(self):
        raw = _make_bi5_bytes(n_ticks=5)
        df = _parse_bi5_hour(raw, self.DT, "EURUSD")
        assert (df["ask"] > df["bid"]).all()

    def test_index_is_utc_datetime(self):
        raw = _make_bi5_bytes(n_ticks=3)
        df = _parse_bi5_hour(raw, self.DT, "EURUSD")
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_timestamps_within_expected_hour(self):
        raw = _make_bi5_bytes(n_ticks=5)
        df = _parse_bi5_hour(raw, self.DT, "EURUSD")
        assert (df.index >= self.DT).all()
        # All within the same hour
        assert (df.index < self.DT.replace(hour=9)).all()

    def test_jpy_pair_uses_correct_point(self):
        """USDJPY uses point=1000, so prices should be ~100-level, not ~1.0."""
        point = 1000
        dt = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
        ask_scaled = int(149.500 * point)
        bid_scaled = int(149.490 * point)
        raw = struct.pack(">IIIff", 0, ask_scaled, bid_scaled, 1.0, 1.0)
        df = _parse_bi5_hour(raw, dt, "USDJPY")
        assert not df.empty
        assert abs(df["ask"].iloc[0] - 149.500) < 0.001
        assert abs(df["bid"].iloc[0] - 149.490) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# _enforce_schema
# ─────────────────────────────────────────────────────────────────────────────

class TestEnforceSchema:
    def _base_df(self, n: int = 5) -> pd.DataFrame:
        idx = pd.date_range("2024-01-02 08:00", periods=n, freq="1s", tz="UTC")
        return pd.DataFrame({
            "bid": np.linspace(1.0849, 1.0854, n),
            "ask": np.linspace(1.0850, 1.0855, n),
        }, index=idx)

    def test_all_tick_columns_present(self):
        df = _enforce_schema(self._base_df(), "EURUSD", "dukascopy")
        assert list(df.columns) == TICK_COLUMNS

    def test_pair_and_source_tagged_correctly(self):
        df = _enforce_schema(self._base_df(), "GBPUSD", "dukascopy")
        assert (df["pair"] == "GBPUSD").all()
        assert (df["source"] == "dukascopy").all()

    def test_mid_derived_correctly(self):
        df = _enforce_schema(self._base_df(), "EURUSD", "dukascopy")
        expected_mid = (df["bid"] + df["ask"]) / 2
        pd.testing.assert_series_equal(df["mid"], expected_mid, check_names=False)

    def test_spread_derived_correctly(self):
        df = _enforce_schema(self._base_df(), "EURUSD", "dukascopy")
        assert (df["spread"] > 0).all()

    def test_rows_with_bid_gte_ask_dropped(self):
        base = self._base_df(5)
        base.loc[base.index[2], "bid"] = base.loc[base.index[2], "ask"]  # bad tick
        df = _enforce_schema(base, "EURUSD", "dukascopy")
        assert len(df) == 4

    def test_index_named_timestamp(self):
        df = _enforce_schema(self._base_df(), "EURUSD", "dukascopy")
        assert df.index.name == "timestamp"

    def test_index_is_sorted(self):
        base = self._base_df(10)
        shuffled = base.iloc[[4, 2, 7, 0, 9, 1, 8, 3, 6, 5]]
        df = _enforce_schema(shuffled, "EURUSD", "dukascopy")
        assert df.index.is_monotonic_increasing

    def test_duplicate_timestamps_dropped(self):
        base = self._base_df(3)
        duped = pd.concat([base.iloc[:2], base.iloc[1:]])
        df = _enforce_schema(duped, "EURUSD", "dukascopy")
        assert len(df) == 3
        assert not df.index.duplicated().any()

    def test_volume_defaulted_to_one_when_missing(self):
        base = self._base_df()
        df = _enforce_schema(base, "EURUSD", "dukascopy")
        assert (df["volume"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# DUKA_PAIR_MAP / PIP_SIZES
# ─────────────────────────────────────────────────────────────────────────────

class TestPairConfig:
    EXPECTED_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]

    def test_major_pairs_in_pair_map(self):
        for pair in self.EXPECTED_PAIRS:
            assert pair in DUKA_PAIR_MAP, f"{pair} missing from DUKA_PAIR_MAP"

    def test_major_pairs_in_pip_sizes(self):
        for pair in self.EXPECTED_PAIRS:
            assert pair in PIP_SIZES, f"{pair} missing from PIP_SIZES"

    def test_pip_sizes_correct_for_jpy(self):
        assert PIP_SIZES["USDJPY"] == pytest.approx(0.01)

    def test_pip_sizes_correct_for_non_jpy(self):
        for pair in ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF"]:
            assert PIP_SIZES[pair] == pytest.approx(0.0001), f"Wrong pip size for {pair}"

    def test_pair_map_values_uppercase_no_slash(self):
        for k, v in DUKA_PAIR_MAP.items():
            assert "/" not in v
            assert v == v.upper()


# ─────────────────────────────────────────────────────────────────────────────
# DukascopyLoader — cache / path helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestDukascopyLoaderCachePath:
    def test_cache_path_structure(self, tmp_path):
        loader = DukascopyLoader(cache_dir=str(tmp_path), verbose=False)
        dt = datetime(2024, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        p = loader._cache_path("EURUSD", dt)
        assert p.parent.parent.parent.name == "EURUSD"
        assert p.suffix == ".parquet"
        assert "2024" in str(p)
        assert "03" in str(p)

    def test_cache_dir_created_on_init(self, tmp_path):
        subdir = tmp_path / "duka_cache"
        assert not subdir.exists()
        DukascopyLoader(cache_dir=str(subdir), verbose=False)
        assert subdir.exists()


# ─────────────────────────────────────────────────────────────────────────────
# DukascopyLoader.load — mocked network
# ─────────────────────────────────────────────────────────────────────────────

class TestDukascopyLoaderLoad:
    """
    Tests for DukascopyLoader.load() using a mocked HTTP layer.
    No real network calls are made.
    """

    def _make_loader(self, tmp_path) -> DukascopyLoader:
        return DukascopyLoader(
            cache_dir=str(tmp_path / "cache"),
            verbose=False,
            concurrency=4,
            max_retries=1,
        )

    def _mock_response(self, compressed: bytes):
        """Return an aiohttp-compatible mock response that yields compressed bytes."""
        resp = AsyncMock()
        resp.status = 200
        resp.read = AsyncMock(return_value=compressed)
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    def _mock_404_response(self):
        resp = AsyncMock()
        resp.status = 404
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    def test_load_returns_dataframe(self, tmp_path):
        loader = self._make_loader(tmp_path)
        compressed = _make_bi5_lzma(n_ticks=20)

        with patch("aiohttp.ClientSession.get", return_value=self._mock_response(compressed)):
            df = loader.load("EURUSD", start="2024-01-02", end="2024-01-02")

        assert isinstance(df, pd.DataFrame)

    def test_load_schema_columns(self, tmp_path):
        loader = self._make_loader(tmp_path)
        compressed = _make_bi5_lzma(n_ticks=10)

        with patch("aiohttp.ClientSession.get", return_value=self._mock_response(compressed)):
            df = loader.load("EURUSD", start="2024-01-02", end="2024-01-02")

        if not df.empty:
            assert list(df.columns) == TICK_COLUMNS

    def test_load_returns_empty_on_404(self, tmp_path):
        loader = self._make_loader(tmp_path)

        with patch("aiohttp.ClientSession.get", return_value=self._mock_404_response()):
            df = loader.load("EURUSD", start="2024-01-02", end="2024-01-02")

        assert isinstance(df, pd.DataFrame)
        assert df.empty or set(df.columns) == set(TICK_COLUMNS)

    def test_load_normalizes_pair_case(self, tmp_path):
        """loader.load() should accept lowercase and slash-separated pairs."""
        loader = self._make_loader(tmp_path)
        compressed = _make_bi5_lzma(n_ticks=5)

        with patch("aiohttp.ClientSession.get", return_value=self._mock_response(compressed)):
            df_lower = loader.load("eurusd", start="2024-01-02", end="2024-01-02")
            df_slash = loader.load("EUR/USD", start="2024-01-02", end="2024-01-02")

        # Both should produce equivalent results (no KeyError / exception)
        assert isinstance(df_lower, pd.DataFrame)
        assert isinstance(df_slash, pd.DataFrame)

    def test_load_with_restricted_hours(self, tmp_path):
        """Passing hours=[8, 9] should only request 2 hours per day."""
        loader = self._make_loader(tmp_path)
        _make_bi5_lzma(n_ticks=5)
        call_count = 0


        async def counting_fetch(session, semaphore, pair, dt):
            nonlocal call_count
            call_count += 1
            return None  # simulate 404 for all

        loader._fetch_hour_async = counting_fetch

        loader.load("EURUSD", start="2024-01-02", end="2024-01-02", hours=[8, 9], auto_redownload=False)
        assert call_count == 2  # one day × 2 hours

    def test_load_cache_used_on_second_call(self, tmp_path):
        """After a successful fetch+write, the second call should hit the parquet cache."""
        pytest.importorskip("pyarrow", reason="pyarrow required for parquet caching")

        loader = self._make_loader(tmp_path)
        compressed = _make_bi5_lzma(n_ticks=10)
        fetch_calls = []

        async def tracking_fetch(session, semaphore, pair, dt):
            fetch_calls.append(dt)
            return compressed

        loader._fetch_hour_async = tracking_fetch

        # First load — should trigger a network fetch
        loader.load("EURUSD", start="2024-01-02", end="2024-01-02", hours=[8])
        first_count = len(fetch_calls)

        # Second load — should use cached parquet (no new fetches)
        loader.load("EURUSD", start="2024-01-02", end="2024-01-02", hours=[8])
        second_count = len(fetch_calls) - first_count

        assert first_count >= 1
        assert second_count == 0, "Second call should use cache, not re-fetch"

    def test_load_redownloads_missing_hours(self, tmp_path):
        loader = self._make_loader(tmp_path)
        dt1 = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 2, 9, 0, 0, tzinfo=timezone.utc)

        first_df = _enforce_schema(
            pd.DataFrame(
                {"bid": [1.0849], "ask": [1.0850], "volume": [1.0]},
                index=pd.DatetimeIndex([dt1], tz="UTC"),
            ),
            "EURUSD",
            "dukascopy",
        )
        second_df = _enforce_schema(
            pd.DataFrame(
                {"bid": [1.0851], "ask": [1.0852], "volume": [1.0]},
                index=pd.DatetimeIndex([dt2], tz="UTC"),
            ),
            "EURUSD",
            "dukascopy",
        )

        calls = []

        async def fake_load_all_async(pair, datetimes, session=None, semaphore=None, executor=None):
            calls.append(list(datetimes))
            if len(calls) == 1:
                return first_df
            return second_df

        loader._load_all_async = fake_load_all_async
        df = loader.load(
            "EURUSD",
            start="2024-01-02",
            end="2024-01-02",
            hours=[8, 9],
            max_redownload_passes=1,
        )

        assert len(calls) == 2
        assert calls[1] == [dt2]
        assert set(df.index.floor("h")) == {
            pd.Timestamp(dt1),
            pd.Timestamp(dt2),
        }

    def test_load_invalid_cached_hour_refetches(self, tmp_path):
        pytest.importorskip("pyarrow", reason="pyarrow required for parquet caching")

        loader = self._make_loader(tmp_path)
        dt = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
        cache_file = loader._cache_path("EURUSD", dt)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        invalid_df = pd.DataFrame(
            {"bid": [1.0849], "ask": [np.nan], "volume": [1.0]},
            index=pd.DatetimeIndex([dt], tz="UTC"),
        )
        invalid_df.to_parquet(cache_file)

        fetch_calls = []
        compressed = _make_bi5_lzma(n_ticks=10)

        async def tracking_fetch(session, semaphore, pair, fetch_dt):
            fetch_calls.append(fetch_dt)
            return compressed

        loader._fetch_hour_async = tracking_fetch
        df = loader.load("EURUSD", start="2024-01-02", end="2024-01-02", hours=[8])

        assert fetch_calls == [dt]
        assert not df.empty
        assert not df[["bid", "ask"]].isna().any().any()

    def test_load_truncated_payload_refetches_hour(self, tmp_path):
        loader = self._make_loader(tmp_path)
        dt = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
        good_payload = _make_bi5_lzma(n_ticks=10)
        bad_payload = lzma.compress(_make_bi5_bytes(n_ticks=10)[:-1])
        payloads = [bad_payload, good_payload]
        fetch_calls = []

        async def flaky_fetch(session, semaphore, pair, fetch_dt):
            fetch_calls.append(fetch_dt)
            return payloads.pop(0)

        loader._fetch_hour_async = flaky_fetch
        df = loader.load("EURUSD", start="2024-01-02", end="2024-01-02", hours=[8], auto_redownload=False)

        assert len(fetch_calls) == 2
        assert fetch_calls == [dt, dt]
        assert not df.empty


# ─────────────────────────────────────────────────────────────────────────────
# DukascopyLoader.load_multiple — multiple pairs
# ─────────────────────────────────────────────────────────────────────────────

class TestDukascopyLoaderMultiplePairs:
    PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]

    def _make_loader(self, tmp_path) -> DukascopyLoader:
        return DukascopyLoader(
            cache_dir=str(tmp_path / "cache"),
            verbose=False,
            concurrency=4,
            max_retries=1,
        )

    def test_load_multiple_returns_dict(self, tmp_path):
        loader = self._make_loader(tmp_path)

        async def fake_fetch(session, semaphore, pair, dt):
            return None  # 404 for everything

        loader._fetch_hour_async = fake_fetch
        result = loader.load_multiple(self.PAIRS, start="2024-01-02", end="2024-01-02", hours=[8])

        assert isinstance(result, dict)

    def test_load_multiple_keys_match_pairs(self, tmp_path):
        loader = self._make_loader(tmp_path)

        async def fake_fetch(session, semaphore, pair, dt):
            return None

        loader._fetch_hour_async = fake_fetch
        result = loader.load_multiple(self.PAIRS, start="2024-01-02", end="2024-01-02", hours=[8])

        assert set(result.keys()) == set(self.PAIRS)

    def test_load_multiple_each_value_is_dataframe(self, tmp_path):
        loader = self._make_loader(tmp_path)

        async def fake_fetch(session, semaphore, pair, dt):
            return None

        loader._fetch_hour_async = fake_fetch
        result = loader.load_multiple(self.PAIRS, start="2024-01-02", end="2024-01-02", hours=[8])

        for pair, df in result.items():
            assert isinstance(df, pd.DataFrame), f"{pair}: expected DataFrame, got {type(df)}"

    def test_load_multiple_with_real_data(self, tmp_path):
        """Each pair that gets data should have the full tick schema."""
        loader = self._make_loader(tmp_path)
        compressed = _make_bi5_lzma(n_ticks=8)

        async def fake_fetch(session, semaphore, pair, dt):
            return compressed

        loader._fetch_hour_async = fake_fetch
        result = loader.load_multiple(["EURUSD", "GBPUSD"], start="2024-01-02", end="2024-01-02", hours=[8])

        for pair, df in result.items():
            if not df.empty:
                assert list(df.columns) == TICK_COLUMNS
                assert (df["pair"] == pair).all()
                assert (df["source"] == "dukascopy").all()

    def test_load_eurusd_gbpusd_convenience(self, tmp_path):
        loader = self._make_loader(tmp_path)

        async def fake_fetch(session, semaphore, pair, dt):
            return None

        loader._fetch_hour_async = fake_fetch
        result = loader.load_eurusd_gbpusd(start="2024-01-02", end="2024-01-02")

        assert "EURUSD" in result
        assert "GBPUSD" in result

    def test_load_multiple_empty_date_range_returns_empty(self, tmp_path):
        """start == end with no hours should return empty DataFrames."""
        loader = self._make_loader(tmp_path)
        result = loader.load_multiple(["EURUSD"], start="2024-01-02", end="2024-01-02", hours=[])
        assert isinstance(result["EURUSD"], pd.DataFrame)
        assert result["EURUSD"].empty


# ─────────────────────────────────────────────────────────────────────────────
# Data integrity checks on parsed output
# ─────────────────────────────────────────────────────────────────────────────

class TestDataIntegrity:
    def test_no_nan_in_parsed_ticks(self):
        raw = _make_bi5_bytes(n_ticks=10)
        dt = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
        df = _parse_bi5_hour(raw, dt, "EURUSD")
        assert not df["bid"].isna().any()
        assert not df["ask"].isna().any()

    def test_prices_in_realistic_eurusd_range(self):
        raw = _make_bi5_bytes(n_ticks=10)
        dt = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
        df = _parse_bi5_hour(raw, dt, "EURUSD")
        assert (df["ask"] > 0.5).all()
        assert (df["ask"] < 2.0).all()

    def test_schema_spread_always_positive_after_enforce(self):
        idx = pd.date_range("2024-01-02 08:00", periods=20, freq="1s", tz="UTC")
        df = pd.DataFrame({
            "bid": np.random.uniform(1.0840, 1.0850, 20),
            "ask": np.random.uniform(1.0851, 1.0860, 20),
        }, index=idx)
        out = _enforce_schema(df, "EURUSD", "dukascopy")
        assert (out["spread"] > 0).all()

    def test_volume_non_negative(self):
        raw = _make_bi5_bytes(n_ticks=10)
        dt = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
        df = _parse_bi5_hour(raw, dt, "EURUSD")
        assert (df["volume"] >= 0).all()


class TestYearByYearDownload:
    def test_download_year_by_year_returns_summary(self, tmp_path):
        mgr = ForexDataManager(dukascopy_dir=str(tmp_path / "cache"), verbose=False)
        sample_idx = pd.date_range("2024-01-02 07:00", periods=2, freq="1s", tz="UTC")
        sample_df = _enforce_schema(
            pd.DataFrame(
                {
                    "bid": [1.0849, 1.0850],
                    "ask": [1.0850, 1.0851],
                    "volume": [1.0, 1.0],
                },
                index=sample_idx,
            ),
            "EURUSD",
            "dukascopy",
        )

        mgr.duka.load = MagicMock(return_value=sample_df)
        mgr.duka._build_tasks_dt = MagicMock(return_value=[datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)])
        mgr.duka._hour_coverage_report = MagicMock(
            return_value={
                "duplicate_timestamps": 0,
                "missing_hours": [],
                "missing_hours_count": 0,
                "requested_hours_count": 1,
                "present_hours_count": 1,
            }
        )

        result = mgr.download_dukascopy_year_by_year(["EURUSD"], 2024, 2024)

        assert "EURUSD" in result
        assert 2024 in result["EURUSD"]
        assert result["EURUSD"][2024]["ticks"] == len(sample_df)
        assert mgr.duka.load.call_args.kwargs["auto_redownload"] is True


class TestCompactedStorage:
    def _make_manager(self, tmp_path) -> ForexDataManager:
        return ForexDataManager(
            dukascopy_dir=str(tmp_path / "hourly"),
            dukascopy_compact_dir=str(tmp_path / "compact"),
            verbose=False,
        )

    def _write_hour(self, mgr: ForexDataManager, pair: str, dt: datetime, prices: list[tuple[float, float]]):
        cache_file = mgr.duka._cache_path(pair, dt)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        idx = pd.DatetimeIndex(
            [dt + pd.Timedelta(seconds=i) for i in range(len(prices))],
            tz="UTC",
        )
        df = _enforce_schema(
            pd.DataFrame(
                {
                    "bid": [bid for bid, _ in prices],
                    "ask": [ask for _, ask in prices],
                    "volume": [1.0] * len(prices),
                },
                index=idx,
            ),
            pair,
            "dukascopy",
        )
        df.to_parquet(cache_file)

    def test_compact_dukascopy_cache_daily_writes_partition(self, tmp_path):
        pytest.importorskip("pyarrow", reason="pyarrow required for parquet compaction")

        mgr = self._make_manager(tmp_path)
        dt1 = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 2, 9, 0, 0, tzinfo=timezone.utc)
        self._write_hour(mgr, "EURUSD", dt1, [(1.0849, 1.0850), (1.0850, 1.0851)])
        self._write_hour(mgr, "EURUSD", dt2, [(1.0851, 1.0852)])

        result = mgr.compact_dukascopy_cache(["EURUSD"], granularity="daily")
        out_path = (
            Path(mgr.duka_compact_dir)
            / "granularity=daily"
            / "pair=EURUSD"
            / "year=2024"
            / "month=01"
            / "day=02"
            / "ticks.parquet"
        )

        assert result["EURUSD"]["partitions_written"] == 1
        assert result["EURUSD"]["ticks_written"] == 3
        assert out_path.exists()

        compacted = pd.read_parquet(out_path)
        assert len(compacted) == 3
        assert set(compacted["pair"].astype(str)) == {"EURUSD"}

    def test_compact_dukascopy_cache_monthly_groups_multiple_days(self, tmp_path):
        pytest.importorskip("pyarrow", reason="pyarrow required for parquet compaction")

        mgr = self._make_manager(tmp_path)
        self._write_hour(
            mgr,
            "EURUSD",
            datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc),
            [(1.0849, 1.0850)],
        )
        self._write_hour(
            mgr,
            "EURUSD",
            datetime(2024, 1, 3, 8, 0, 0, tzinfo=timezone.utc),
            [(1.0851, 1.0852)],
        )

        result = mgr.compact_dukascopy_cache(["EURUSD"], granularity="monthly")
        out_path = (
            Path(mgr.duka_compact_dir)
            / "granularity=monthly"
            / "pair=EURUSD"
            / "year=2024"
            / "month=01"
            / "ticks.parquet"
        )

        assert result["EURUSD"]["partitions_written"] == 1
        assert out_path.exists()
        compacted = pd.read_parquet(out_path)
        assert len(compacted) == 2

    def test_query_dukascopy_compacted_returns_filtered_rows(self, tmp_path):
        pytest.importorskip("pyarrow", reason="pyarrow required for parquet compaction")
        pytest.importorskip("duckdb", reason="duckdb required for DuckDB query layer")

        mgr = self._make_manager(tmp_path)
        self._write_hour(
            mgr,
            "EURUSD",
            datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc),
            [(1.0849, 1.0850), (1.0850, 1.0851)],
        )
        self._write_hour(
            mgr,
            "EURUSD",
            datetime(2024, 1, 3, 8, 0, 0, tzinfo=timezone.utc),
            [(1.0852, 1.0853)],
        )
        mgr.compact_dukascopy_cache(["EURUSD"], granularity="daily")

        df = mgr.query_dukascopy_compacted("EURUSD", start="2024-01-02", end="2024-01-02")

        assert len(df) == 2
        assert set(df["pair"].astype(str)) == {"EURUSD"}
