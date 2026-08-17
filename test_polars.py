import datetime

import polars as pl


def compute_vpin(bars: pl.DataFrame, bucket_size: int = 50, n_buckets: int = 50) -> pl.Series:
    buy_vol = pl.when(pl.col("close") > pl.col("open")).then(pl.col("volume")).otherwise(0.0)
    sell_vol = pl.when(pl.col("close") <= pl.col("open")).then(pl.col("volume")).otherwise(0.0)

    df = bars.with_columns([buy_vol.alias("buy_vol"), sell_vol.alias("sell_vol")])

    df = df.with_columns(
        [
            pl.col("buy_vol").rolling_sum(window_size=bucket_size).alias("buy_bucket"),
            pl.col("sell_vol").rolling_sum(window_size=bucket_size).alias("sell_bucket"),
            pl.col("volume").rolling_sum(window_size=bucket_size).alias("total_bucket"),
        ]
    )

    vpin = ((df["buy_bucket"] - df["sell_bucket"]).abs() / (df["total_bucket"] + 1e-9)).rolling_mean(
        window_size=n_buckets
    )

    return vpin.fill_null(0.0).fill_nan(0.0).alias("vpin")


def compute_realized_moments(close: pl.Series, window: int = 20) -> pl.DataFrame:
    ret = (close / close.shift(1)).log().alias("ret")

    up_ret = pl.when(ret > 0).then(ret).otherwise(0.0)
    dn_ret = pl.when(ret < 0).then(ret).otherwise(0.0)

    df = pl.DataFrame({"ret": ret, "up_ret": up_ret, "dn_ret": dn_ret})

    df = df.with_columns(
        [
            pl.col("ret").rolling_skew(window_size=window).alias(f"rolling_skew_{window}"),
            pl.col("ret").rolling_mean(window_size=window).alias("mu"),
            (pl.col("ret") ** 2).rolling_mean(window_size=window).alias("mu2"),
            (pl.col("ret") ** 3).rolling_mean(window_size=window).alias("mu3"),
            (pl.col("ret") ** 4).rolling_mean(window_size=window).alias("mu4"),
            pl.col("up_ret").rolling_std(window_size=window).alias("up_std"),
            pl.col("dn_ret").rolling_std(window_size=window).alias("dn_std"),
        ]
    )

    df = df.with_columns([(pl.col("mu2") - pl.col("mu") ** 2).alias("var")])

    df = df.with_columns(
        [
            (
                (
                    pl.col("mu4")
                    - 4 * pl.col("mu3") * pl.col("mu")
                    + 6 * pl.col("mu2") * (pl.col("mu") ** 2)
                    - 3 * (pl.col("mu") ** 4)
                )
                / (pl.col("var") ** 2 + 1e-9)
            ).alias(f"rolling_kurt_{window}"),
            (pl.col("up_std") / (pl.col("dn_std") + 1e-9)).alias("rvol_ratio"),
        ]
    )

    res = df.select(
        [
            pl.col(f"rolling_skew_{window}").fill_nan(0.0).fill_null(0.0),
            pl.col(f"rolling_kurt_{window}").fill_nan(0.0).fill_null(0.0),
            pl.col("rvol_ratio").fill_nan(0.0).fill_null(0.0),
        ]
    )
    return res


def compute_asia_london_gap(bars: pl.DataFrame, atr: pl.Series = None) -> pl.Series:
    time_col = "timestamp" if "timestamp" in bars.columns else "datetime"

    df = bars.select([pl.col(time_col), pl.col("close")])
    df = df.with_columns([pl.col(time_col).dt.date().alias("date"), pl.col(time_col).dt.time().alias("time")])

    df = df.with_columns(
        [(pl.col("time") < pl.time(7, 0)).alias("is_asia"), (pl.col("time") >= pl.time(7, 0)).alias("is_london")]
    )

    london_open_times = (
        df.filter(pl.col("is_london")).group_by("date").agg(pl.col(time_col).first().alias("london_open_time"))
    )

    asia_close_vals = df.filter(pl.col("is_asia")).group_by("date").agg(pl.col("close").last().alias("asia_close"))

    daily_gaps = london_open_times.join(asia_close_vals, on="date", how="inner")

    df = df.join(daily_gaps, left_on=time_col, right_on="london_open_time", how="left")

    df = df.with_columns([(pl.col("close") - pl.col("asia_close")).alias("gap")])

    gap_series = df["gap"].forward_fill()

    if atr is not None:
        gap_series = gap_series / (atr + 1e-9)

    return gap_series.fill_null(0.0).fill_nan(0.0).alias("asia_london_gap")


bars = pl.DataFrame(
    {
        "timestamp": [
            datetime.datetime(2021, 1, 1, 6, 50),
            datetime.datetime(2021, 1, 1, 7, 0),
            datetime.datetime(2021, 1, 1, 7, 10),
        ],
        "open": [1.0, 1.1, 1.2],
        "close": [1.1, 1.2, 1.1],
        "volume": [100, 200, 150],
    }
)

print("VPIN:")
print(compute_vpin(bars, 2, 2))

print("Realized moments:")
print(compute_realized_moments(bars["close"], 2))

print("Gap:")
print(compute_asia_london_gap(bars))
