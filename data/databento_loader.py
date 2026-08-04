from datetime import UTC
from pathlib import Path

import polars as pl


class DatabentoLoader:
    """
    Loader for Databento CME Level-2 (MBP-10) data.
    """
    def __init__(self, data_dir: str = "data/raw/databento", verbose: bool = True):
        self.data_dir = Path(data_dir)
        self.verbose = verbose

        # These CME futures trade as 1 USD = X Foreign Currency on spot,
        # but CME quotes them as 1 Foreign Currency = X USD.
        # So we must invert the price: 1 / price
        self.INVERT_PAIRS = {"USDJPY", "USDCAD", "USDCHF"}

    def load(self, pair: str, start: str = None, end: str = None) -> pl.DataFrame:
        if not self.data_dir.exists():
            if self.verbose:
                print(f"[Databento] Directory {self.data_dir} does not exist.")
            return pl.DataFrame()

        # Use LazyFrames for massive memory savings and speed
        glob_pattern = str(self.data_dir / f"{pair}_mbp-10_*.parquet")

        try:
            lazy_df = pl.scan_parquet(glob_pattern)
        except Exception as e:
            if self.verbose:
                print(f"[Databento] Failed to scan parquet files: {e}")
            return pl.DataFrame()

        # Filter by date if provided
        schema = lazy_df.collect_schema()
        ts_col = "ts_event" if "ts_event" in schema.names() else "ts_recv"

        # DS-003: enforce UTC timezone on timestamp column before filtering
        # to prevent silent misalignment between naive and aware datetimes
        lazy_df = lazy_df.with_columns(
            pl.col(ts_col).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")
        )

        if start:
            from datetime import datetime
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
            lazy_df = lazy_df.filter(pl.col(ts_col) >= start_dt)
        if end:
            from datetime import datetime
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=UTC)
            lazy_df = lazy_df.filter(pl.col(ts_col) <= end_dt)

        try:
            # Handle scaling if Databento didn't pre-scale it (if prices > 1e6, it's fixed precision)
            # We check the median of a small slice
            med_df = lazy_df.select(pl.col("bid_px_00").first()).collect()
            first_px = med_df["bid_px_00"][0] if not (len(med_df) == 0) else 0

            scale_factor = 1e9 if first_px is not None and first_px > 10000 else 1.0

            out_lazy = lazy_df.select([
                pl.col(ts_col).alias("timestamp_utc"),
                (pl.col("bid_px_00") / scale_factor).cast(pl.Float32).alias("bid"),
                (pl.col("ask_px_00") / scale_factor).cast(pl.Float32).alias("ask"),
                ((pl.col("bid_sz_00") + pl.col("ask_sz_00")) / 2.0).cast(pl.Float32).alias("volume")
            ])

            if pair in self.INVERT_PAIRS:
                # Spot = 1 / CME. Bid/ask are inverted.
                out_lazy = out_lazy.filter((pl.col("bid") > 1e-12) & (pl.col("ask") > 1e-12))
                out_lazy = out_lazy.with_columns([
                    (1.0 / pl.col("ask")).alias("real_bid"),
                    (1.0 / pl.col("bid")).alias("real_ask")
                ]).drop(["bid", "ask"]).rename({"real_bid": "bid", "real_ask": "ask"})

            # Add mid and spread
            out_lazy = out_lazy.with_columns([
                ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
                (pl.col("ask") - pl.col("bid")).alias("spread")
            ])

            # Add static columns
            out_lazy = out_lazy.with_columns([
                pl.lit(pair).alias("pair"),
                pl.lit("databento").alias("source")
            ])

            out_lazy = out_lazy.sort("timestamp_utc")
            return out_lazy.collect()

        except Exception as e:
            print(f"[Databento] Failed to collect data: {e}")
            return pl.DataFrame()
