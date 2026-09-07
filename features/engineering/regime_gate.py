"""
features/engineering/regime_gate.py
RegimeGateClassifier: expanding-window logistic-regression regime break probability.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression


class RegimeGateClassifier:
    def __init__(self, min_samples: int = 80, random_state: int = 42):
        self.min_samples = int(min_samples)
        self.random_state = int(random_state)
        self.model = LogisticRegression(max_iter=500, random_state=self.random_state)

    def _zscore(self, col: str, lb: int = 60) -> pl.Expr:
        mu = pl.col(col).rolling_mean(window_size=lb)
        sd = pl.col(col).rolling_std(window_size=lb) + 1e-9
        return ((pl.col(col) - mu) / sd).clip(-6, 6).alias(f"{col}_z")

    def fit_predict(self, F: pl.DataFrame) -> pl.Series:
        req = ["gold_dxy_corr_break", "us_2s10s_spread_chg", "yield_spread_us_de_10y_chg", "risk_off_signal"]
        if not all(c in F.columns for c in req):
            return pl.Series("regime_break_prob", np.zeros(len(F)))

        X_df = F.select(
            [
                self._zscore("gold_dxy_corr_break").alias("gold_break_z"),
                self._zscore("us_2s10s_spread_chg").alias("curve_chg_z"),
                self._zscore("yield_spread_us_de_10y_chg").alias("yield_chg_z"),
                self._zscore("risk_off_signal", lb=30).alias("risk_off_z"),
            ]
        )

        y_expr = (
            (pl.col("gold_break_z") > 1.0) & ((pl.col("risk_off_z") > 0.5) | (pl.col("curve_chg_z").abs() > 1.0))
        ).cast(pl.Int32)

        y = X_df.select(y_expr.alias("y"))["y"].to_numpy()
        X_pd = X_df.to_pandas()
        ok = X_pd.notna().all(axis=1)

        if ok.sum() < self.min_samples or len(np.unique(y[ok])) < 2:
            score = (
                0.8 * X_pd["gold_break_z"].fillna(0)
                + 0.5 * X_pd["risk_off_z"].fillna(0)
                + 0.3 * X_pd["curve_chg_z"].abs().fillna(0)
            )
            prob = 1.0 / (1.0 + np.exp(-score.clip(-8, 8)))
            return pl.Series("regime_break_prob", prob)

        ok_idx = np.where(ok)[0]
        prob = np.zeros(len(F), dtype=np.float64)

        # Baseline heuristic score for the entire series (fully causal)
        score = (
            0.8 * X_pd["gold_break_z"].fillna(0)
            + 0.5 * X_pd["risk_off_z"].fillna(0)
            + 0.3 * X_pd["curve_chg_z"].abs().fillna(0)
        )
        ok_mask = ok.to_numpy(dtype=bool, copy=False)
        prob[ok_mask] = 1.0 / (1.0 + np.exp(-score.clip(-8, 8).to_numpy()[ok_mask]))

        # Expanding window fit: fit up to i, predict [i, i+step)
        step = max(self.min_samples, len(ok_idx) // 10)

        if step < len(ok_idx):
            for i in range(step, len(ok_idx), step):
                train_idx = ok_idx[:i]
                test_idx = ok_idx[i : i + step] if i + step < len(ok_idx) else ok_idx[i:]

                if len(np.unique(y[train_idx])) >= 2:
                    try:
                        self.model.fit(X_pd.iloc[train_idx].values, y[train_idx])
                        prob[test_idx] = self.model.predict_proba(X_pd.iloc[test_idx].values)[:, 1]
                    except Exception:
                        pass

        return pl.Series("regime_break_prob", prob)
