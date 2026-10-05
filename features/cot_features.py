import polars as pl

# Weekly report; 21 days tolerates holiday-delayed releases. Older values are
# treated as missing: a discontinued series (e.g. a CFTC contract rename) must
# not be forward-filled for years.
COT_MAX_AGE = pl.duration(days=21)


def add_cot_features(df: pl.DataFrame, cot_df: pl.DataFrame, pair: str) -> pl.DataFrame:
    """
    Joins the Commitment of Traders (COT) weekly data onto the main tick/bar DataFrame
    using a backward-filling asof join to prevent forward-looking bias.
    Computes momentum and extreme positioning features.

    Args:
        df: The main tick or bar dataframe (must have 'timestamp_utc' and 'close')
        cot_df: The COT dataframe loaded from cot_financials_cleaned.parquet
        pair: The FX pair to filter COT data for (e.g., 'EURUSD')
    """
    # Filter COT for this specific pair
    pair_cot = cot_df.filter(pl.col("pair") == pair)

    if len(pair_cot) == 0:
        # If no COT data available for this pair, return default 0s
        return df.with_columns(
            [
                pl.lit(0.0).alias("cot_net_hf"),
                pl.lit(0.0).alias("cot_net_comm"),
                pl.lit(0.0).alias("cot_hf_mom_4w"),
                pl.lit(0.0).alias("cot_extreme"),
            ]
        )

    # Sort both sides for asof join; Polars requires sorted inputs and chunked
    # cache reads can arrive out of timestamp order.
    df = df.sort("timestamp_utc")
    pair_cot = pair_cot.sort("timestamp_utc")

    # Add rolling z-scores / momentum on the COT data BEFORE joining
    # COT is weekly, so a 4-period lookback = 4 weeks, 52-period = 1 year
    pair_cot = pair_cot.with_columns(
        [
            # 4-week momentum: How fast Hedge Funds are adding/removing longs
            (pl.col("net_hedge_fund") - pl.col("net_hedge_fund").shift(4)).alias("cot_hf_mom_4w"),
            # 52-week Z-score: Is the current positioning extremely long or short relative to the past year?
            (
                (pl.col("net_hedge_fund") - pl.col("net_hedge_fund").rolling_mean(52))
                / (pl.col("net_hedge_fund").rolling_std(52) + 1e-9)
            ).alias("cot_hf_zscore_52w"),
        ]
    )

    # Select columns to join
    cols_to_join = ["timestamp_utc", "net_hedge_fund", "net_commercial", "cot_hf_mom_4w", "cot_hf_zscore_52w"]
    pair_cot_clean = (
        pair_cot.select(cols_to_join)
        .with_columns(pl.col("timestamp_utc").cast(df.schema["timestamp_utc"]))
        .with_columns(pl.col("timestamp_utc").alias("_cot_release"))
    )

    # Perform backward-looking asof join
    df = df.join_asof(pair_cot_clean, on="timestamp_utc", strategy="backward")

    # Before the first report or after the series goes stale -> neutral 0.
    fresh = pl.col("_cot_release").is_not_null() & ((pl.col("timestamp_utc") - pl.col("_cot_release")) <= COT_MAX_AGE)
    df = df.with_columns(
        [
            pl.when(fresh).then(pl.col("net_hedge_fund")).otherwise(None).cast(pl.Float64).fill_null(0.0).alias("cot_net_hf"),
            pl.when(fresh).then(pl.col("net_commercial")).otherwise(None).cast(pl.Float64).fill_null(0.0).alias("cot_net_comm"),
            pl.when(fresh).then(pl.col("cot_hf_mom_4w")).otherwise(None).cast(pl.Float64).fill_null(0.0).alias("cot_hf_mom_4w"),
            pl.when(fresh).then(pl.col("cot_hf_zscore_52w")).otherwise(None).cast(pl.Float64).fill_null(0.0).alias("cot_hf_zscore_52w"),
        ]
    )

    # Create extreme positioning flag (e.g. Z-score > 2 or < -2)
    df = df.with_columns(
        [
            pl.when(pl.col("cot_hf_zscore_52w") > 2.0)
            .then(1.0)
            .when(pl.col("cot_hf_zscore_52w") < -2.0)
            .then(-1.0)
            .otherwise(0.0)
            .alias("cot_extreme")
        ]
    )

    # Drop intermediate columns
    df = df.drop(["net_hedge_fund", "net_commercial", "cot_hf_zscore_52w", "_cot_release"])

    return df
