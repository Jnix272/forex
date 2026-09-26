"""
features/engineering/cross_asset.py
CrossAssetFeatures class: cross-asset correlations, beta, PCA/ICA factors.
"""

import polars as pl


class CrossAssetFeatures:
    def __init__(self, corr_window=60, regime_window=240, lags=(1, 5, 15)):
        self.cw = corr_window
        self.rw = regime_window
        self.lags = tuple(lags)

    def build(self, bars: pl.DataFrame, data: dict | None = None) -> pl.DataFrame:
        len(bars)
        synthetic = self._synthetic(bars)

        merged = dict(synthetic)
        if data is not None:
            import pandas as pd
            for k, v in data.items():
                if isinstance(v, pd.Series):
                    df_v = v.to_frame(name=k).reset_index()
                    df_v.columns = ["timestamp_utc", k]
                    v = pl.from_pandas(df_v)
                elif isinstance(v, pd.DataFrame):
                    df_v = v.copy()
                    if "timestamp_utc" not in df_v.columns:
                        if df_v.index.name in ("timestamp_utc", "date", "Date", "time", "timestamp") or isinstance(df_v.index, pd.DatetimeIndex):
                            df_v = df_v.reset_index()
                    v = pl.from_pandas(df_v)
                elif not isinstance(v, pl.DataFrame):
                    continue

                for date_col in ["Date", "date", "time", "timestamp", "datetime", "Datetime"]:
                    if date_col in v.columns and "timestamp_utc" not in v.columns:
                        v = v.rename({date_col: "timestamp_utc"})

                if "timestamp_utc" in v.columns:
                    try:
                        v = v.with_columns(pl.col("timestamp_utc").cast(bars["timestamp_utc"].dtype))
                    except Exception:
                        v = v.with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
                merged[k] = v

        # Start with bars
        F = bars

        # Calculate forex return
        F = F.with_columns([(pl.col("close").log() - pl.col("close").shift(1).log()).alias("forex_ret")])

        # Join assets
        for asset, s_df in merged.items():
            if not isinstance(s_df, pl.DataFrame):
                continue

            for date_col in ["Date", "date", "time", "timestamp", "datetime", "Datetime"]:
                if date_col in s_df.columns and "timestamp_utc" not in s_df.columns:
                    s_df = s_df.rename({date_col: "timestamp_utc"})

            if "timestamp_utc" not in s_df.columns:
                continue

            if "value" in s_df.columns:
                s_df = s_df.rename({"value": asset})
            elif asset not in s_df.columns:
                val_cols = [c for c in s_df.columns if c != "timestamp_utc"]
                if val_cols:
                    s_df = s_df.rename({val_cols[-1]: asset})

            if asset not in s_df.columns:
                continue

            try:
                s_df = s_df.with_columns(pl.col("timestamp_utc").cast(bars["timestamp_utc"].dtype))
            except Exception:
                pass
            s_df = s_df.select(["timestamp_utc", asset]).sort("timestamp_utc")
            try:
                F = F.join_asof(s_df, on="timestamp_utc", strategy="backward")
                F = F.with_columns([pl.col(asset).fill_null(strategy="forward").fill_null(0.0)])
            except Exception:
                continue

            # Returns
            lr = pl.col(asset).log() - pl.col(asset).shift(1).log()
            exprs = [lr.alias(f"{asset}_ret")]
            for lag in self.lags:
                exprs.append(lr.shift(lag).alias(f"{asset}_ret_l{lag}"))

            # rolling corr and beta
            cov = pl.rolling_cov(lr, pl.col("forex_ret"), window_size=self.cw)
            var_x = pl.col("forex_ret").rolling_var(window_size=self.cw)
            var_y = lr.rolling_var(window_size=self.cw)

            corr = cov / (var_x.sqrt() * var_y.sqrt() + 1e-9)
            beta = cov / (var_x + 1e-9)

            exprs.append(corr.alias(f"{asset}_corr"))
            exprs.append(beta.alias(f"{asset}_beta"))
            F = F.with_columns(exprs)

        cols = F.columns
        exprs = []
        if "US10Y" in cols and "DE10Y" in cols:
            exprs.extend(
                [
                    (pl.col("US10Y") - pl.col("DE10Y")).alias("yield_spread_us_de_10y"),
                    (pl.col("US10Y") - pl.col("DE10Y")).diff().alias("yield_spread_us_de_10y_chg"),
                ]
            )
        if "US10Y" in cols and "US2Y" in cols:
            exprs.extend(
                [
                    (pl.col("US10Y") - pl.col("US2Y")).alias("us_2s10s_spread"),
                    (pl.col("US10Y") - pl.col("US2Y")).diff().alias("us_2s10s_spread_chg"),
                ]
            )
        if "SPX" in cols and "VIX" in cols:
            spx_ret = pl.col("SPX").log().diff()
            vix_ret = pl.col("VIX").log().diff()
            exprs.append(
                (((spx_ret < 0).cast(pl.Float64) + (vix_ret > 0).cast(pl.Float64)) / 2.0).alias("risk_off_signal")
            )
        if "GOLD_ret" in cols and "DXY_ret" in cols:
            gd_cov = pl.rolling_cov(pl.col("GOLD_ret"), pl.col("DXY_ret"), window_size=self.cw)
            gd_vx = pl.col("GOLD_ret").rolling_var(window_size=self.cw)
            gd_vy = pl.col("DXY_ret").rolling_var(window_size=self.cw)
            gd_corr = gd_cov / (gd_vx.sqrt() * gd_vy.sqrt() + 1e-9)
            gd_corr_base = gd_corr.rolling_mean(self.rw)
            exprs.extend([gd_corr.alias("gold_dxy_corr"), (gd_corr - gd_corr_base).abs().alias("gold_dxy_corr_break")])
        if "WTI_ret" in cols and "COPPER_ret" in cols:
            exprs.append(
                (0.6 * pl.col("COPPER_ret").shift(1) + 0.4 * pl.col("WTI_ret").shift(1)).alias("commodity_fx_lead")
            )

        if exprs:
            F = F.with_columns(exprs)

        # PCA/ICA common-factor model + Granger causality + lead-lag network
        try:
            F = self._cross_asset_factors(F)
        except Exception as e:
            print(f"[CrossAssetFeatures] WARNING: factor model failed: {e}")

        return F.fill_null(strategy="forward").fill_null(0.0)

    def _cross_asset_factors(self, F: pl.DataFrame) -> pl.DataFrame:
        """Append cross-asset PCA/ICA factors, Granger and lead-lag features.

        Uses the ``{asset}_ret`` columns computed above plus the primary pair's
        own returns (``forex_ret``) as a ``PRIMARY`` series. Requires at least
        two assets; otherwise returns F unchanged.
        """
        import numpy as np
        import pandas as pd

        from features.cross_asset_factors import build_cross_asset_factors

        ret_cols = [c for c in F.columns if c.endswith("_ret") and c not in ("forex_ret",)]
        if "forex_ret" in F.columns:
            ret_cols = [*ret_cols, "forex_ret"]
        if len(ret_cols) < 2:
            return F

        panel = pd.DataFrame(
            {c[:-4]: F[c].to_numpy() for c in ret_cols},
            index=range(len(F)),
        )
        # Daily assets forward-filled onto intraday bars are mostly exact zeros /
        # NaN; with those, the rolling least squares hit "SVD did not converge" in
        # every window and no factor features were produced. Clean the panel and
        # drop columns that barely move.
        panel = panel.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        _moving = (panel != 0.0).mean(axis=0) >= 0.05
        panel = panel.loc[:, _moving & (panel.std(axis=0) > 1e-12)]
        if panel.shape[1] < 2:
            return F
        fact = build_cross_asset_factors(
            panel,
            n_factors=3,
            method="pca",
            factor_window=120,
            factor_step=20,
            maxlag=1,
            granger_window=120,
            granger_step=20,
            max_lag=5,
            leadlag_window=120,
            leadlag_step=20,
            min_abs_corr=0.05,
        )
        new_cols = [c for c in fact.columns if c not in F.columns]
        if new_cols:
            F = pl.concat([F, fact.select(new_cols)], how="horizontal_extend")
        return F

    def _synthetic(self, bars: pl.DataFrame) -> dict:
        # Deprecated: synthetic data generation replaced by real external assets.
        # Return an empty dict so that only provided real data is merged.
        return {}
