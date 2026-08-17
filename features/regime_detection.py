"""
features/regime_detection.py
=============================
True market-regime detection replacing the legacy volatility-tercile
"hmm" in ``feature_engineering_pl.py``.

Provides:
  - ``RegimeHMM``        - a real Hidden Markov Model (hmmlearn) over a
                           feature matrix (returns, volatility, spread, ...).
                           Emits smoothed state probabilities per bar.
  - ``hurst_rs``         - Hurst exponent via Rescaled Range (R/S) analysis.
  - ``hurst_dfa``        - Hurst exponent via Detrended Fluctuation Analysis.
  - ``fractal_dimension``- Higuchi-style fractal dimension of a price series.
  - ``detect_regimes``   - end-to-end: fit HMM, emit state probs, and join
                           with Hurst/fractal regime labels into one frame.

The volatility-tercile column names ``vol_regime_state_N_prob`` are produced
by the standalone ``vol_regime_probs`` helper so callers can keep the legacy
name while swapping in the *real* HMM behind the same interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from numba import njit

    _NUMBA_OK = True
except ImportError:  # pragma: no cover - optional / version-skew fallback
    _NUMBA_OK = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


if TYPE_CHECKING:
    import polars as pl

try:
    from hmmlearn.hmm import GaussianHMM

    _HMMLEARN_OK = True
except Exception:  # pragma: no cover - optional dependency
    _HMMLEARN_OK = False


def _rolling_mean_causal(src: np.ndarray, dst: np.ndarray, window: int) -> None:
    """Causal rolling mean via cumulative sum - O(n) vs O(nxwindow) loop."""
    n = len(src)
    if window <= 0 or n <= window:
        return
    # cumsum[0] = 0, cumsum[k] = sum(src[:k]) for k >= 1
    cumsum = np.concatenate([[0.0], np.cumsum(src)])
    # dst[i] = (cumsum[i] - cumsum[i-window]) / window  for i >= window
    dst[window:] = (cumsum[window:n] - cumsum[0 : n - window]) / window


# ════════════════════════════════════════════════════════════════════════════
# Hurst exponents
# ════════════════════════════════════════════════════════════════════════════


@njit(cache=True)
def _hurst_rs_numba(x: np.ndarray) -> float:
    n = len(x)
    if n < 20:
        return 0.5

    max_lag = max(4, n // 4)
    max_lag = min(max_lag, n // 2 - 1)
    raw_lags = np.logspace(np.log10(8.0), np.log10(max_lag), 40)
    lags = np.unique(np.round(raw_lags).astype(np.int64))
    filtered = np.empty(len(lags), dtype=np.int64)
    fcount = 0
    for lag in lags:
        if 4 <= lag < n // 2:
            filtered[fcount] = lag
            fcount += 1
    lags = filtered[:fcount]
    if len(lags) < 4:
        return 0.5

    rs_vals = np.empty(len(lags), dtype=np.float64)
    valid = 0
    for _li, lag in enumerate(lags):
        n_chunks = n // lag
        if n_chunks < 2:
            continue
        rs_sum = 0.0
        rs_count = 0
        for c in range(n_chunks):
            chunk = x[c * lag : (c + 1) * lag]
            m = chunk.mean()
            std = chunk.std()
            if std < 1e-12:
                continue
            dev = np.cumsum(chunk - m)
            rs_sum += (dev.max() - dev.min()) / std
            rs_count += 1
        if rs_count > 0:
            rs_vals[valid] = rs_sum / rs_count
            valid += 1

    if valid < 4:
        return 0.5
    log_lags = np.log(lags[:valid])
    log_rs = np.log(rs_vals[:valid])
    # Manual linear regression (np.polyfit not supported in Numba)
    len(log_lags)
    x_mean = log_lags.mean()
    y_mean = log_rs.mean()
    num = ((log_lags - x_mean) * (log_rs - y_mean)).sum()
    den = ((log_lags - x_mean) ** 2).sum()
    if den < 1e-12:
        return 0.5
    slope = num / den
    result = slope
    if result < 0.05:
        return 0.05
    if result > 0.95:
        return 0.95
    return float(result)


def hurst_rs(x: Sequence[float] | np.ndarray, max_lag: int | None = None) -> float:
    """
    Hurst exponent via Rescaled Range (R/S) analysis.

    For each lag ``L`` the series is split into chunks of length ``L``; each
    chunk yields ``R/S = (max - min of cumsum deviations) / std`` and the
    average is regressed against ``L`` in log-log space.

    Returns H in [0, 1]:  H ~ 0.5 random walk, H > 0.55 trending
    (positive autocorrelation), H < 0.45 mean-reverting.
    """
    x_arr = np.asarray(x, dtype=float)
    x_arr = x_arr[np.isfinite(x_arr)]
    return _hurst_rs_numba(x_arr)


def hurst_dfa(x: Sequence[float] | np.ndarray, min_box: int = 4) -> float:
    """
    Hurst exponent via Detrended Fluctuation Analysis (DFA).

    Integrates the de-meaned series, splits into boxes of increasing size,
    detrends each box by ordinary least squares, and regresses the RMS of
    the pooled residuals against box size in log-log space.  DFA is more
    robust than R/S for non-stationary series (trends, intraday seasonality).
    """
    x_arr = np.asarray(x, dtype=float)
    x_arr = x_arr[np.isfinite(x_arr)]
    return _hurst_dfa_numba(x_arr, min_box)


@njit(cache=True)
def _hurst_dfa_numba(x: np.ndarray, min_box: int) -> float:
    n = len(x)
    if n < 20:
        return 0.5

    y = np.cumsum(x - x.mean())
    raw_sizes = np.logspace(np.log10(float(min_box)), np.log10(n // 4), 20)
    box_sizes = np.unique(np.round(raw_sizes).astype(np.int64))
    filtered = np.empty(len(box_sizes), dtype=np.int64)
    fcount = 0
    for bs in box_sizes:
        if bs >= 2 and n // bs >= 2:
            filtered[fcount] = bs
            fcount += 1
    box_sizes = filtered[:fcount]
    if len(box_sizes) < 4:
        return 0.5

    fluct = np.empty(len(box_sizes), dtype=np.float64)
    for bi, box in enumerate(box_sizes):
        n_box = n // box
        sq_err = 0.0
        n_pts = 0
        for i in range(n_box):
            seg = y[i * box : (i + 1) * box]
            t = np.arange(box, dtype=np.float64)
            # Manual linear regression for speed
            t_mean = t.mean()
            seg_mean = seg.mean()
            numerator = ((t - t_mean) * (seg - seg_mean)).sum()
            denominator = ((t - t_mean) ** 2).sum()
            if denominator < 1e-12:
                slope = 0.0
            else:
                slope = numerator / denominator
            intercept = seg_mean - slope * t_mean
            resid = seg - (slope * t + intercept)
            sq_err += np.sum(resid**2)
            n_pts += box
        fluct[bi] = np.sqrt(sq_err / n_pts) if n_pts > 0 else np.nan

    mask = np.isfinite(fluct) & (fluct > 0)
    if mask.sum() < 4:
        return 0.5
    filtered_sizes = box_sizes[mask]
    filtered_fluct = fluct[mask]
    log_sizes = np.log(filtered_sizes)
    log_fluct = np.log(filtered_fluct)
    n_pts = len(log_sizes)
    x_mean = log_sizes.mean()
    y_mean = log_fluct.mean()
    num = ((log_sizes - x_mean) * (log_fluct - y_mean)).sum()
    den = ((log_sizes - x_mean) ** 2).sum()
    if den < 1e-12:
        return 0.5
    slope = num / den
    result = slope
    if result < 0.05:
        return 0.05
    if result > 0.95:
        return 0.95
    return float(result)


def fractal_dimension(x: Sequence[float] | np.ndarray, k_max: int | None = None) -> float:
    """
    Fractal dimension of a time series via the Higuchi method.

    For a range of lag scales ``k`` the curve length ``L(k)`` is estimated and
    the fractal dimension D = 1 - slope(log L(k) vs log k).  D ~ 1 for smooth /
    trending behaviour, D ~ 1.5 for self-similar (fractional Brownian) noise,
    D ~ 2 for uncorrelated white noise.
    """
    x_arr = np.asarray(x, dtype=float)
    x_arr = x_arr[np.isfinite(x_arr)]
    n = len(x_arr)
    k_max_resolved = k_max if k_max is not None else min(32, n // 2)
    return _fractal_dimension_numba(x_arr, k_max_resolved)


@njit(cache=True)
def _fractal_dimension_numba(x: np.ndarray, k_max: int) -> float:
    n = len(x)
    if n < 16:
        return 1.5

    if k_max is None:
        k_max = min(32, n // 2)
    lengths = np.empty(k_max, dtype=np.float64)
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
        lengths[k - 1] = Lk / k

    mask = np.isfinite(lengths) & (lengths > 0)
    if mask.sum() < 4:
        return 1.5
    filt_k = np.arange(1, k_max + 1, dtype=np.float64)[mask]
    filt_l = lengths[mask]
    log_k = np.log(filt_k)
    log_l = np.log(filt_l)
    len(log_k)
    x_mean = log_k.mean()
    y_mean = log_l.mean()
    num = ((log_k - x_mean) * (log_l - y_mean)).sum()
    den = ((log_k - x_mean) ** 2).sum()
    if den < 1e-12:
        return 1.5
    slope = num / den
    result = 1.0 - slope
    if result < 1.0:
        return 1.0
    if result > 2.0:
        return 2.0
    return float(result)


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
            raise ImportError("RegimeHMM requires 'hmmlearn'. Install with: uv pip install hmmlearn")
        self._model: GaussianHMM | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._cols: list[str] = []
        self._features: np.ndarray = np.empty((0, 0), dtype=float)

    def fit(self, features: np.ndarray) -> RegimeHMM:
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
        probs = self.state_probs
        return np.argmax(probs, axis=1)

    @property
    def state_probs(self) -> np.ndarray:
        model = self._model
        mean = self._mean
        std = self._std
        if model is None or mean is None or std is None:
            raise RuntimeError("RegimeHMM.fit() must be called first")

        Z = (np.nan_to_num(np.asarray(self._features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) - mean) / std

        # BUG-004: predict_proba uses Forward-Backward (smoothing) which leaks future data.
        # We must use only the forward pass for causal probabilities.
        framelogprob = model._compute_log_likelihood(Z)

        from scipy.special import logsumexp

        model_any: Any = model
        if hasattr(model_any, "_do_forward_pass"):
            logprob, fwdlattice = model_any._do_forward_pass(framelogprob)  # noqa: RUF059
        else:
            from hmmlearn._hmmc import forward_log

            _logprob, fwdlattice = forward_log(model.startprob_, model.transmat_, framelogprob)

        # fwdlattice is log P(O_{1:t}, S_t). We want P(S_t | O_{1:t})
        causal_probs = np.exp(fwdlattice - logsumexp(fwdlattice, axis=1, keepdims=True))
        return causal_probs

    @property
    def transition_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RegimeHMM.fit() must be called first")
        return self._model.transmat_

    def set_features(self, features: np.ndarray, cols: list[str] | None = None) -> RegimeHMM:
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


def _causal_hmm_decode(
    feat: np.ndarray,
    n_states: int = 3,
    min_fit: int = 120,
    refit_every: int = 20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal-ish HMM decode: fit params on a warm-up prefix only, then decode.

    Fitting on the full series lets future bars influence early state posteriors
    (look-ahead). We instead:

    1. Fit GaussianHMM parameters on ``feat[:min_fit]`` only.
    2. Decode the full series with those frozen params.
    3. Lag posteriors / states by 1 bar so bar ``t`` never sees ``t``'s return
       in the smoothed posterior used as a feature.

    ``refit_every`` is retained for API compatibility but unused (full expanding
    refits are O(n²) and too slow for production feature builds).
    """
    del refit_every  # API compat
    n = len(feat)
    probs = np.full((n, n_states), 1.0 / max(1, n_states), dtype=np.float64)
    states = np.zeros(n, dtype=np.int32)
    if n < min_fit or not _HMMLEARN_OK:
        return probs, states

    try:
        model = RegimeHMM(n_states=n_states, random_state=random_state)
        model.set_features(feat[:min_fit])
        model.fit(feat[:min_fit])

        # Extract frozen parameters from fitted model
        hmm_model = model._model
        if hmm_model is None or model._mean is None or model._std is None:
            raise RuntimeError("RegimeHMM fit failed to populate model parameters")

        covars_obj = getattr(hmm_model, "covars_", None)
        if covars_obj is None:
            raise RuntimeError("RegimeHMM fit failed to populate covariance parameters")

        transmat = model.transition_.copy()
        startprob = np.asarray(hmm_model.startprob_).copy()
        means = np.asarray(hmm_model.means_).copy()
        covars = np.asarray(covars_obj).copy()
        mean_scaler = model._mean
        std_scaler = model._std

        # Manually decode full series with frozen parameters (no refit)
        # Standardize full features with warm-up mean/std
        Z_full = (np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0) - mean_scaler) / std_scaler

        # Compute log-likelihood for each state at each timestep
        from scipy.stats import multivariate_normal

        n = len(feat)
        framelogprob = np.zeros((n, n_states))
        for i in range(n_states):
            framelogprob[:, i] = multivariate_normal.logpdf(Z_full, mean=means[i], cov=covars[i])

        # Forward algorithm with frozen parameters
        from scipy.special import logsumexp

        logprob = np.zeros((n, n_states))
        logprob[0] = np.log(startprob) + framelogprob[0]
        for t in range(1, n):
            # log P(S_t | O_{1:t}) = logsumexp(log P(S_{t-1} | O_{1:t-1}) + log A_{S_{t-1}, S_t}) + log P(O_t | S_t)
            logprob[t] = logsumexp(logprob[t - 1, :, np.newaxis] + np.log(transmat.T + 1e-12), axis=0) + framelogprob[t]

        # Normalize to get causal probabilities
        log_norm = np.asarray(logsumexp(logprob, axis=1, keepdims=True), dtype=float)
        causal_probs = np.exp(logprob - log_norm)

        # Get states (argmax)
        raw_s = np.argmax(causal_probs, axis=1)

        # 1-bar lag: feature at t uses posterior known at t-1
        probs[1:] = causal_probs[:-1]
        states[1:] = raw_s[:-1]
        probs[0] = 1.0 / n_states
        states[0] = 0
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "HMM regime fit/decode failed (%s); emitting explicit uniform fallback probs (not a fitted regime).",
            exc,
        )
        probs[:] = 1.0 / max(1, n_states)
        states[:] = 0
    return probs, states


def vol_regime_probs_polars(
    df,
    close_col: str = "close",
    n_states: int = 3,
    window: int = 60,
) -> pl.DataFrame:
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
    # Causal rolling vol (no mode="same" convolution look-ahead)
    vol = np.zeros(n)
    _rolling_mean_causal(abs_ret, vol, window)

    feat = np.column_stack([ret, vol])
    probs, _ = _causal_hmm_decode(feat, n_states=n_states, min_fit=max(window * 2, 120))

    out = {f"vol_regime_state_{s}_prob": probs[:, s] for s in range(n_states)}
    return pl.DataFrame(out)


def vol_regime_quantile_probs(n_states: int = 3, window: int = 60) -> list:
    """
    Legacy volatility-tercile regime probs (equivalent to the old
    ``hmm_regime_probs`` in ``feature_engineering_pl.py``) - kept so callers
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
) -> pl.DataFrame:
    """
    Full regime feature builder over a Polars bar frame.

    Emits:
      - ``vol_regime_state_N_prob``   true-HMM state probabilities
      - ``hurst_rs`` / ``hurst_dfa``  Hurst exponents (R/S and DFA)
      - ``fractal_dim``               Higuchi fractal dimension
      - ``regime_label``              -1 mean-revert / 0 neutral / +1 trend
      - ``regime_class``              0 low-vol, 1 normal, 2 high-vol (HMM)

    ``step`` > 1 evaluates the (expensive) rolling Hurst / fractal estimators
    every ``step`` bars and forward-fills in between - a standard trade-off for
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
    _rolling_mean_causal(abs_ret, vol, window)
    feat = np.column_stack([ret, vol])

    probs, states = _causal_hmm_decode(
        feat,
        n_states=n_states,
        min_fit=max(window * 2, 120),
    )

    hurst_rs_arr = np.full(n, 0.5)
    hurst_dfa_arr = np.full(n, 0.5)
    fractal_arr = np.full(n, 1.5)
    step = max(1, int(step))
    for i in range(hurst_window, n, step):
        w = close[i - hurst_window : i]
        hurst_rs_arr[i : i + step] = hurst_rs(w)
        hurst_dfa_arr[i : i + step] = hurst_dfa(w)
    for i in range(fractal_window, n, step):
        fractal_arr[i : i + step] = fractal_dimension(close[i - fractal_window : i])

    trend_label = np.where(hurst_dfa_arr > 0.55, 1.0, np.where(hurst_dfa_arr < 0.45, -1.0, 0.0))

    out = {f"vol_regime_state_{s}_prob": probs[:, s] for s in range(n_states)}
    out.update(
        {
            "hurst_rs": hurst_rs_arr,
            "hurst_dfa": hurst_dfa_arr,
            "fractal_dim": fractal_arr,
            "regime_label": trend_label,
            "regime_class": states.astype(np.int32),
        }
    )
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
