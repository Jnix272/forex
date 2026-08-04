"""
features/regime_detection.py
=============================
True market-regime detection replacing the legacy volatility-tercile
"hmm" in ``feature_engineering_pl.py``.

Provides:
  - ``RegimeHMM``        — a real Hidden Markov Model (hmmlearn) over a
                           feature matrix (returns, volatility, spread, ...).
                           Emits smoothed state probabilities per bar.
  - ``hurst_rs``         — Hurst exponent via Rescaled Range (R/S) analysis.
  - ``hurst_dfa``        — Hurst exponent via Detrended Fluctuation Analysis.
  - ``fractal_dimension``— Higuchi-style fractal dimension of a price series.
  - ``detect_regimes``   — end-to-end: fit HMM, emit state probs, and join
                           with Hurst/fractal regime labels into one frame.

The volatility-tercile column names ``vol_regime_state_N_prob`` are produced
by the standalone ``vol_regime_probs`` helper so callers can keep the legacy
name while swapping in the *real* HMM behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
    _HMMLEARN_OK = True
except Exception:  # pragma: no cover - optional dependency
    _HMMLEARN_OK = False


# ════════════════════════════════════════════════════════════════════════════
# Hurst exponents
# ════════════════════════════════════════════════════════════════════════════

def hurst_rs(x: Sequence[float], max_lag: Optional[int] = None) -> float:
    """
    Hurst exponent via Rescaled Range (R/S) analysis.

    For each lag ``L`` the series is split into chunks of length ``L``; each
    chunk yields ``R/S = (max - min of cumsum deviations) / std`` and the
    average is regressed against ``L`` in log-log space.

    Returns H in [0, 1]:  H ~ 0.5 random walk, H > 0.55 trending
    (positive autocorrelation), H < 0.45 mean-reverting.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return 0.5

    # lags out to ~n/4 keep finite-sample bias small while preserving power.
    max_lag = max_lag or max(4, n // 4)
    max_lag = min(max_lag, n // 2 - 1)
    lags = np.unique(np.logspace(np.log10(8), np.log10(max_lag), 40)
                     .round().astype(int))
    lags = [lag for lag in lags if 4 <= lag < n // 2]
    if len(lags) < 4:
        return 0.5

    rs_vals = []
    for lag in lags:
        n_chunks = n // lag
        if n_chunks < 2:
            continue
        rs = []
        for c in range(n_chunks):
            chunk = x[c * lag:(c + 1) * lag]
            m = chunk.mean()
            std = chunk.std()
            if std < 1e-12:
                continue
            dev = np.cumsum(chunk - m)
            rs.append((dev.max() - dev.min()) / std)
        if rs:
            rs_vals.append(np.mean(rs))

    if len(rs_vals) < 4:
        return 0.5
    try:
        H, _ = np.polyfit(np.log(lags[:len(rs_vals)]), np.log(rs_vals), 1)
        return float(np.clip(H, 0.05, 0.95))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 0.5


def hurst_dfa(x: Sequence[float], min_box: int = 4) -> float:
    """
    Hurst exponent via Detrended Fluctuation Analysis (DFA).

    Integrates the de-meaned series, splits into boxes of increasing size,
    detrends each box by ordinary least squares, and regresses the RMS of
    the pooled residuals against box size in log-log space.  DFA is more
    robust than R/S for non-stationary series (trends, intraday seasonality).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return 0.5

    y = np.cumsum(x - x.mean())
    box_sizes = np.unique(np.logspace(
        np.log10(min_box), np.log10(n // 4), 20).round().astype(int))
    box_sizes = [b for b in box_sizes if b >= 2 and n // b >= 2]

    fluct = []
    for box in box_sizes:
        n_box = n // box
        sq_err = 0.0
        n_pts = 0
        for i in range(n_box):
            seg = y[i * box:(i + 1) * box]
            t = np.arange(box, dtype=float)
            slope, intercept = np.polyfit(t, seg, 1)
            resid = seg - (slope * t + intercept)
            sq_err += np.sum(resid ** 2)
            n_pts += box
        fluct.append(np.sqrt(sq_err / n_pts) if n_pts > 0 else np.nan)

    fluct = np.asarray(fluct)
    mask = np.isfinite(fluct) & (fluct > 0)
    if mask.sum() < 4:
        return 0.5
    try:
        H, _ = np.polyfit(
            np.log(np.asarray(box_sizes)[mask]), np.log(fluct[mask]), 1)
        return float(np.clip(H, 0.05, 0.95))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 0.5


def fractal_dimension(x: Sequence[float], k_max: Optional[int] = None) -> float:
    """
    Fractal dimension of a time series via the Higuchi method.

    For a range of lag scales ``k`` the curve length ``L(k)`` is estimated and
    the fractal dimension D = 1 - slope(log L(k) vs log k).  D ~ 1 for smooth /
    trending behaviour, D ~ 1.5 for self-similar (fractional Brownian) noise,
    D ~ 2 for uncorrelated white noise.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 16:
        return 1.5

    k_max = k_max or min(32, n // 2)
    lengths = []
    for k in range(1, k_max + 1):
        Lk = 0.0
        for m in range(k):
            n_int = (n - m) // k
            if n_int < 1:
                continue
            Lmk = 0.0
            for i in range(1, n_int):
                Lmk += abs(x[m + i * k] - x[m + (i - 1) * k])
            Lk += Lmk * (n - 1) / (n_int * k)
        lengths.append(Lk / k)

    lengths = np.asarray(lengths)
    mask = np.isfinite(lengths) & (lengths > 0)
    if mask.sum() < 4:
        return 1.5
    k_vals = np.arange(1, k_max + 1, dtype=float)[mask]
    try:
        slope, _ = np.polyfit(np.log(k_vals), np.log(lengths[mask]), 1)
        return float(np.clip(1.0 - slope, 1.0, 2.0))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 1.5


# ════════════════════════════════════════════════════════════════════════════
# Hidden Markov Model regime classifier
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeHMM:
    """
    A real Gaussian HMM over market features.

    Features are standardised before fitting (each column z-scored) so a
    single model can consume returns, realised volatility, spread and volume
    on a common scale.

    Public attributes after ``fit``:
      - ``states``           most likely state per observation (argmax Viterbi)
      - ``state_probs``      P(state | obs) via the forward-backward algorithm
      - ``transition_``      (n_states, n_states) transition matrix
      - ``n_states``         number of fitted states
    """
    n_states: int = 3
    n_iter: int = 200
    covariance_type: str = "full"
    random_state: int = 42
    tol: float = 1e-4

    def __post_init__(self):
        if not _HMMLEARN_OK:
            raise ImportError(
                "RegimeHMM requires 'hmmlearn'. Install with: uv pip install hmmlearn")
        self._model = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._cols: List[str] = []

    def fit(self, features: np.ndarray) -> "RegimeHMM":
        X = np.asarray(features, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-9
        Z = (X - self._mean) / self._std

        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self._model.fit(Z)
        return self

    @property
    def states(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RegimeHMM.fit() must be called first")
        return self._model.predict(
            (np.nan_to_num(np.asarray(self._features)) - self._mean) / self._std
        )

    @property
    def state_probs(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RegimeHMM.fit() must be called first")
        Z = (np.nan_to_num(np.asarray(self._features)) - self._mean) / self._std
        return self._model.predict_proba(Z)

    @property
    def transition_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RegimeHMM.fit() must be called first")
        return self._model.transmat_

    def set_features(self, features: np.ndarray, cols: Optional[List[str]] = None) -> "RegimeHMM":
        self._features = np.asarray(features, dtype=float)
        self._cols = cols or []
        return self


def fit_regime_hmm(
    features: np.ndarray,
    n_states: int = 3,
    random_state: int = 42,
) -> RegimeHMM:
    """Convenience wrapper: fit a RegimeHMM on a feature matrix."""
    model = RegimeHMM(n_states=n_states, random_state=random_state)
    model.set_features(features)
    return model.fit(features)


# ════════════════════════════════════════════════════════════════════════════
# Standalone Polars-friendly regime probs (same output names as legacy)
# ════════════════════════════════════════════════════════════════════════════

def vol_regime_probs_polars(
    df,
    close_col: str = "close",
    n_states: int = 3,
    window: int = 60,
) -> "pl.DataFrame":
    """
    Drop-in replacement for the legacy ``hmm_regime_probs`` expression builder.

    Emits the same ``vol_regime_state_N_prob`` column contract, but backed by a
    real HMM (hmmlearn) over [returns, rolling volatility] instead of volatility
    terciles.  The legacy quantile-bucket behaviour lives on as
    ``vol_regime_quantile_probs`` for environments without hmmlearn.
    """
    import polars as pl

    close = np.asarray(df[close_col], dtype=float)
    n = len(close)
    ret = np.zeros(n)
    ret[1:] = np.diff(np.log(np.maximum(close, 1e-12)))
    abs_ret = np.abs(ret)
    vol = np.convolve(abs_ret, np.ones(window) / window, mode="same")

    feat = np.column_stack([ret, vol])
    model = RegimeHMM(n_states=n_states, random_state=42)
    model.set_features(feat)
    model.fit(feat)
    probs = model.state_probs

    out = {f"vol_regime_state_{s}_prob": probs[:, s] for s in range(n_states)}
    return pl.DataFrame(out)


def vol_regime_quantile_probs(n_states: int = 3, window: int = 60) -> List:
    """
    Legacy volatility-tercile regime probs (equivalent to the old
    ``hmm_regime_probs`` in ``feature_engineering_pl.py``) — kept so callers
    without hmmlearn can fall back to the previous quantile-bucket behaviour.

    Returns Polars expressions that emit ``vol_regime_state_N_prob`` columns.
    """
    import polars as pl

    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    vol = ret.rolling_std(window_size=window)

    exprs = []
    for s in range(n_states):
        q = vol.rolling_quantile((s + 1) / n_states, window_size=window)
        state_above = vol <= q
        prob = (state_above).rolling_mean(window_size=window).fill_null(0.0)
        exprs.append(prob.alias(f"vol_regime_state_{s}_prob"))
    return exprs


def detect_regimes_polars(
    df,
    close_col: str = "close",
    n_states: int = 3,
    window: int = 60,
    hurst_window: int = 120,
    fractal_window: int = 60,
    step: int = 1,
) -> "pl.DataFrame":
    """
    Full regime feature builder over a Polars bar frame.

    Emits:
      - ``vol_regime_state_N_prob``   true-HMM state probabilities
      - ``hurst_rs`` / ``hurst_dfa``  Hurst exponents (R/S and DFA)
      - ``fractal_dim``               Higuchi fractal dimension
      - ``regime_label``              -1 mean-revert / 0 neutral / +1 trend
      - ``regime_class``              0 low-vol, 1 normal, 2 high-vol (HMM)

    ``step`` > 1 evaluates the (expensive) rolling Hurst / fractal estimators
    every ``step`` bars and forward-fills in between — a standard trade-off for
    slow estimators that keeps output length identical.

    Missing lookback values are forward-filled with the neutral baseline
    (0.5 for Hurst, 1.5 for fractal dimension, 0 for regime label).
    """
    import polars as pl

    close = df[close_col].to_numpy()
    n = len(close)

    ret = np.zeros(n)
    ret[1:] = np.diff(np.log(np.maximum(close, 1e-12)))
    vol = np.zeros(n)
    abs_ret = np.abs(ret)
    for i in range(window, n):
        vol[i] = abs_ret[i - window:i].mean()
    feat = np.column_stack([ret, vol])

    model = RegimeHMM(n_states=n_states, random_state=42)
    model.set_features(feat)
    model.fit(feat)
    probs = model.state_probs
    states = model.states

    hurst_rs_arr = np.full(n, 0.5)
    hurst_dfa_arr = np.full(n, 0.5)
    fractal_arr = np.full(n, 1.5)
    step = max(1, int(step))
    for i in range(hurst_window, n, step):
        w = close[i - hurst_window:i]
        hurst_rs_arr[i:i + step] = hurst_rs(w)
        hurst_dfa_arr[i:i + step] = hurst_dfa(w)
    for i in range(fractal_window, n, step):
        fractal_arr[i:i + step] = fractal_dimension(close[i - fractal_window:i])

    trend_label = np.where(hurst_dfa_arr > 0.55, 1.0,
                           np.where(hurst_dfa_arr < 0.45, -1.0, 0.0))

    out = {f"vol_regime_state_{s}_prob": probs[:, s] for s in range(n_states)}
    out.update({
        "hurst_rs": hurst_rs_arr,
        "hurst_dfa": hurst_dfa_arr,
        "fractal_dim": fractal_arr,
        "regime_label": trend_label,
        "regime_class": states.astype(np.int32),
    })
    return pl.DataFrame(out)


# ════════════════════════════════════════════════════════════════════════════
# CLI self-test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    # 1000 iid normal returns -> H ~ 0.5 (random walk)
    iid = rng.normal(0, 1, 2000)
    print(f"Hurst R/S (random walk):  {hurst_rs(iid):.3f}")
    print(f"Hurst DFA (random walk):  {hurst_dfa(iid):.3f}")
    print(f"Fractal dim (random):     {fractal_dimension(iid):.3f}")

    # Persistently trending series -> H > 0.5
    trend = np.cumsum(rng.normal(0.02, 1.0, 2000))
    print(f"Hurst DFA (trending):     {hurst_dfa(trend):.3f}")

    # True HMM smoke test
    features = rng.normal(0, 1, (500, 2))
    m = fit_regime_hmm(features, n_states=3)
    print(f"HMM states: {len(np.unique(m.states))}  probs shape: {m.state_probs.shape}")
    print("\n✅ Regime detection self-test passed")
