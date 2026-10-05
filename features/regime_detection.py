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
    from numba import njit, prange

    _NUMBA_OK = True
except ImportError:  # pragma: no cover - optional / version-skew fallback
    _NUMBA_OK = False
    prange = range

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


@njit(cache=True)
def _forward_filter_segment(framelogprob: np.ndarray, log_transmat: np.ndarray, log_alpha0: np.ndarray, start_fresh: bool):
    """Normalised forward filter: returns P(S_t | O_{1:t}) and the final log-alpha.

    ``log_alpha0`` is the normalised log posterior at the bar before the segment
    (or the log start distribution when ``start_fresh``).
    """
    n, k = framelogprob.shape
    out = np.empty((n, k), dtype=np.float64)
    prev = log_alpha0.copy()
    cur = np.empty(k, dtype=np.float64)
    for t in range(n):
        for j in range(k):
            if t == 0 and start_fresh:
                cur[j] = prev[j] + framelogprob[t, j]
            else:
                m = -np.inf
                for i in range(k):
                    v = prev[i] + log_transmat[i, j]
                    if v > m:
                        m = v
                s = 0.0
                for i in range(k):
                    s += np.exp(prev[i] + log_transmat[i, j] - m)
                cur[j] = m + np.log(s) + framelogprob[t, j]
        m = cur.max()
        s = 0.0
        for j in range(k):
            s += np.exp(cur[j] - m)
        norm = m + np.log(s)
        for j in range(k):
            prev[j] = cur[j] - norm
            out[t, j] = np.exp(prev[j])
    return out, prev


def _fit_hmm_window(Z: np.ndarray, n_states: int, random_state: int, prev: dict | None, vol_col: int) -> dict:
    """Fit a GaussianHMM on standardised ``Z`` and return params with states
    sorted by ascending mean of ``vol_col`` (0 = low vol ... n-1 = high vol).

    Warm-starts from ``prev`` so consecutive refits converge in a few EM steps
    and stay aligned with the previous state ordering.
    """
    import warnings

    kw = dict(
        n_components=n_states,
        covariance_type="full",
        n_iter=30 if prev is not None else 100,
        tol=1e-3,
        random_state=random_state,
        min_covar=1e-4,
    )
    def _fit(warm: dict | None):
        if warm is not None:
            # EM cannot revive zero transition/start mass, so a state that died in
            # one window would stay dead forever; mix in a uniform floor.
            m = GaussianHMM(init_params="", **kw)
            m.startprob_ = np.full(n_states, 1.0 / n_states)
            m.transmat_ = 0.9 * warm["transmat"] + 0.1 / n_states
            m.means_ = warm["means"]
            m.covars_ = warm["covars"]
        else:
            m = GaussianHMM(**{**kw, "n_iter": 100})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Z)
        return m

    def _params(m) -> dict:
        means = np.asarray(m.means_, dtype=float)
        covars = np.asarray(m.covars_, dtype=float)
        if covars.ndim == 2:
            covars = np.stack([np.diag(c) for c in covars])
        transmat = np.asarray(m.transmat_, dtype=float)
        startprob = np.asarray(m.startprob_, dtype=float)
        if not (np.isfinite(means).all() and np.isfinite(covars).all()):
            raise ValueError("non-finite HMM emission parameters")
        # States never left in the window get all-zero / NaN transmat rows.
        transmat = np.where(np.isfinite(transmat), transmat, 0.0)
        bad = transmat.sum(axis=1) <= 0
        transmat[bad] = 1.0 / n_states
        transmat = transmat / transmat.sum(axis=1, keepdims=True)
        startprob = np.where(np.isfinite(startprob), startprob, 0.0)
        startprob = startprob / startprob.sum() if startprob.sum() > 0 else np.full(n_states, 1.0 / n_states)
        eye = np.eye(covars.shape[-1])
        covars = np.stack([0.5 * (c + c.T) + 1e-4 * eye for c in covars])
        order = np.argsort(means[:, vol_col], kind="stable")
        return {
            "startprob": startprob[order],
            "transmat": transmat[np.ix_(order, order)],
            "means": means[order],
            "covars": covars[order],
        }

    if prev is not None:
        try:
            model = _fit(prev)
            occ = np.bincount(model.predict(Z), minlength=n_states) / max(1, len(Z))
            if occ.min() >= 0.02:
                return _params(model)
        except Exception:
            pass
    return _params(_fit(None))


def _causal_hmm_decode(
    feat: np.ndarray,
    n_states: int = 3,
    min_fit: int = 1000,
    refit_every: int = 500,
    random_state: int = 42,
    fit_window: int = 2000,
    vol_col: int = 1,
    skip_head: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling-refit HMM decode.

    Every ``refit_every`` bars (starting at ``min_fit``) a GaussianHMM is fitted
    on the trailing ``fit_window`` bars only, standardised with that window's
    mean/std, and used to filter the next segment. States are ordered by the
    mean of ``vol_col`` so ``regime_class`` means 0 = low vol, 1 = normal,
    2 = high vol consistently across refits, build windows and live (labeling
    relies on that mapping). The first ``skip_head`` rows (rolling-feature
    warm-up, e.g. zero volatility) are never fitted on: they would otherwise
    claim a degenerate state that no later bar matches. Bars before
    ``skip_head + min_fit`` carry uniform probs and state 1 (normal). Output is
    lagged one bar so bar ``t`` never sees its own return.
    """
    n = len(feat)
    probs = np.full((n, n_states), 1.0 / max(1, n_states), dtype=np.float64)
    states = np.full(n, min(1, n_states - 1), dtype=np.int32)
    min_fit = max(int(min_fit), 2 * n_states + 10)
    refit_every = max(1, int(refit_every))
    fit_window = max(int(fit_window), min_fit)
    skip_head = max(0, int(skip_head))
    if n <= skip_head + min_fit or not _HMMLEARN_OK:
        return probs, states

    from scipy.stats import multivariate_normal

    X = np.nan_to_num(np.asarray(feat, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    causal = np.full((n, n_states), 1.0 / n_states, dtype=np.float64)
    decoded = np.zeros(n, dtype=bool)
    params: dict | None = None
    log_alpha: np.ndarray | None = None
    _EPS = 1e-12
    try:
        for t0 in range(skip_head + min_fit, n, refit_every):
            t1 = min(n, t0 + refit_every)
            W = X[max(skip_head, t0 - fit_window) : t0]
            mu = W.mean(axis=0)
            sd = W.std(axis=0) + 1e-9
            try:
                params = _fit_hmm_window((W - mu) / sd, n_states, random_state, params, vol_col)
            except Exception:
                if params is None:
                    continue
            assert params is not None
            Z = (X[t0:t1] - mu) / sd
            flp = np.empty((t1 - t0, n_states))
            for i in range(n_states):
                flp[:, i] = multivariate_normal.logpdf(
                    Z, mean=params["means"][i], cov=params["covars"][i], allow_singular=True
                )
            flp = np.nan_to_num(flp, nan=-1e6, neginf=-1e6)
            log_A = np.log(np.clip(params["transmat"], _EPS, 1.0))
            fresh = log_alpha is None
            la0 = np.log(np.clip(params["startprob"], _EPS, 1.0)) if fresh else log_alpha
            seg, log_alpha = _forward_filter_segment(flp, log_A, la0, fresh)
            causal[t0:t1] = seg
            decoded[t0:t1] = True

        raw_s = np.where(decoded, np.argmax(causal, axis=1), states).astype(np.int32)
        probs[1:] = causal[:-1]
        states[1:] = raw_s[:-1]
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "HMM regime fit/decode failed (%s); emitting explicit uniform fallback probs (not a fitted regime).",
            exc,
        )
        probs[:] = 1.0 / max(1, n_states)
        states[:] = min(1, n_states - 1)
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
    probs, _ = _causal_hmm_decode(feat, n_states=n_states, skip_head=window)

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


@njit(parallel=True, cache=True)
def _scan_outcomes_numba_parallel(close, ret, hurst_rs_arr, hurst_dfa_arr, fractal_arr, hurst_window, fractal_window, step):
    """Hurst (R/S, DFA) on log returns; Higuchi fractal dimension on the price path.

    Both Hurst estimators integrate their input, so feeding price levels (already
    integrated) pins H at the 0.95 clip and makes ``regime_label`` constant.
    """
    n = len(close)

    h_steps = (n - hurst_window + step - 1) // step
    for idx in prange(h_steps):
        i = hurst_window + idx * step
        if i >= n: continue
        w = ret[i - hurst_window + 1 : i]
        w_clean = w[np.isfinite(w)]
        if len(w_clean) > 0:
            h_rs = _hurst_rs_numba(w_clean)
            h_dfa = _hurst_dfa_numba(w_clean, 4)
            for j in range(step):
                if i + j < n:
                    hurst_rs_arr[i + j] = h_rs
                    hurst_dfa_arr[i + j] = h_dfa

    f_steps = (n - fractal_window + step - 1) // step
    for idx in prange(f_steps):
        i = fractal_window + idx * step
        if i >= n: continue
        w = close[i - fractal_window : i]
        w_clean = w[np.isfinite(w)]
        if len(w_clean) > 0:
            k_max = min(32, len(w_clean) // 2)
            fd = _fractal_dimension_numba(w_clean, k_max)
            for j in range(step):
                if i + j < n:
                    fractal_arr[i + j] = fd

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

    probs, states = _causal_hmm_decode(feat, n_states=n_states, skip_head=window)

    hurst_rs_arr = np.full(n, 0.5)
    hurst_dfa_arr = np.full(n, 0.5)
    fractal_arr = np.full(n, 1.5)
    step = max(1, int(step))
    
    close = np.asarray(close, dtype=np.float64)
    if _NUMBA_OK:
        _scan_outcomes_numba_parallel(
            close, ret, hurst_rs_arr, hurst_dfa_arr, fractal_arr,
            hurst_window, fractal_window, step
        )
    else:
        for i in range(hurst_window, n, step):
            w = ret[i - hurst_window + 1 : i]
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
