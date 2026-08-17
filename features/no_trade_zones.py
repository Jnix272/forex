"""
No-Trade Zones: Learned Abstention + Conformal Prediction (Improvement #7)
==========================================================================
Extends the existing heuristic `no_trade_score` with:
  1. **Learned Abstention Model** - predicts when to abstain based on
     prediction uncertainty, market regime, feature quality, etc.
  2. **Conformal Prediction for Abstention** - uses calibration to produce
     prediction sets with guaranteed coverage; abstain when set contains
     both long and short (i.e., ambiguous).
  3. **Unified No-Trade Decision** - combines heuristic score, learned
     abstention probability, and conformal prediction set ambiguity.

All components are self-contained and integrate with existing TBM labels
and meta-labeler from `triple_barrier_meta.py`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════════════
# 1. Learned Abstention Model
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class AbstentionConfig:
    """Configuration for learned abstention model."""

    # Base model (sklearn-compatible, must have predict_proba)
    model: Any = None
    # Features to use for abstention prediction
    features: list[str] | None = None
    # Target: 1 = should trade (profitable), 0 = should abstain
    # Derived from TBM labels: profitable if hit TP before SL
    min_samples: int = 500
    train_frac: float = 0.7
    # Probability threshold for trading
    prob_threshold: float = 0.55
    # Random seed
    random_state: int = 42


class LearnedAbstentionModel:
    """
    Learns when to abstain from trading based on market conditions,
    prediction uncertainty, feature quality, etc.

    Training target: 1 if trade would be profitable (TBM: hit TP before SL),
    0 otherwise. Only defined for bars where a trade was signaled.

    Usage:
        abstention = LearnedAbstentionModel(config, model=XGBClassifier())
        abstention.fit(features, tbm_labels, primary_predictions)
        prob_trade = abstention.predict_proba(features, primary_predictions)
        trade_mask = prob_trade > config.prob_threshold
    """

    def __init__(self, config: AbstentionConfig):
        self.config = config
        self.model = config.model
        self._is_fitted = False
        self._feature_names: list[str] = []

    def _prepare_features(
        self,
        X: pd.DataFrame,
        primary_pred: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build feature matrix: base features + primary prediction + uncertainty proxies."""
        feat_list = []
        if self.config.features:
            feat_list.append(X[self.config.features].values.astype(float))
        else:
            feat_list.append(X.select_dtypes(include=[np.number]).values.astype(float))

        if primary_pred is not None:
            primary = np.asarray(primary_pred, dtype=float).reshape(-1, 1)
            # Also add uncertainty proxy: |primary| distance from 0
            uncertainty = np.abs(primary)
            feat_list.append(primary)
            feat_list.append(uncertainty)

        if not feat_list:
            raise ValueError("No features available for abstention model")

        return np.hstack(feat_list)

    def _make_target(
        self,
        tbm_labels: np.ndarray,
        primary_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create abstention target from TBM labels and primary predictions.

        Target = 1 (trade) if:
          - primary predicted long (1) AND TBM label = 1 (hit TP)
          - primary predicted short (-1) AND TBM label = -1 (hit TP)
        Target = 0 (abstain) if:
          - primary predicted direction but hit SL (label = -primary)
          - primary predicted hold (0)
        """
        primary = np.asarray(primary_pred, dtype=float).ravel()
        tbm = np.asarray(tbm_labels, dtype=int).ravel()

        # Only defined where primary made a non-zero prediction
        trade_mask = primary != 0
        target = np.zeros(len(primary), dtype=int)
        target[~trade_mask] = 0  # abstain on hold

        # For trades: profitable if TBM label matches primary direction
        target[trade_mask] = (
            (primary[trade_mask] > 0) & (tbm[trade_mask] == 1) | (primary[trade_mask] < 0) & (tbm[trade_mask] == -1)
        ).astype(int)

        return target, trade_mask

    def fit(
        self,
        X: pd.DataFrame,
        tbm_labels: np.ndarray,
        primary_pred: np.ndarray,
    ) -> LearnedAbstentionModel:
        """Train the abstention model."""
        primary = np.asarray(primary_pred, dtype=float).ravel()
        tbm = np.asarray(tbm_labels, dtype=int).ravel()

        target, trade_mask = self._make_target(tbm, primary)

        # Only train on bars where a trade was signaled
        if not trade_mask.any():
            warnings.warn("[LearnedAbstentionModel] No trades signaled by primary model.", stacklevel=2)
            return self

        y = target[trade_mask]
        X_trades = X.iloc[trade_mask] if hasattr(X, "iloc") else X[trade_mask]

        if len(y) < self.config.min_samples:
            warnings.warn(
                f"[LearnedAbstentionModel] Only {len(y)} trade samples, "
                f"minimum {self.config.min_samples}. Skipping training.", stacklevel=2
            )
            return self

        # Prepare features
        X_feat = self._prepare_features(X_trades, primary[trade_mask])
        self._feature_names = self._get_feature_names(X_trades)

        # Train/val split (temporal)
        n = len(X_feat)
        split = int(n * self.config.train_frac)
        X_train, X_val = X_feat[:split], X_feat[split:]
        y_train, y_val = y[:split], y[split:]

        # Default model
        if self.model is None:
            try:
                from sklearn.ensemble import RandomForestClassifier

                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=20,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
            except ImportError:
                raise ImportError("scikit-learn required for default abstention model")

        self.model.fit(X_train, y_train)

        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val) if len(X_val) > 0 else 0.0

        self._is_fitted = True
        print(
            f"[LearnedAbstentionModel] Trained: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
            f"n_train={len(X_train)}, n_val={len(X_val)}, pos_rate={y.mean():.3f}"
        )
        return self

    def _get_feature_names(self, X: pd.DataFrame) -> list[str]:
        names = []
        if self.config.features:
            names.extend(self.config.features)
        else:
            names.extend(X.select_dtypes(include=[np.number]).columns.tolist())
        names.append("primary_pred")
        names.append("pred_uncertainty")
        return names

    def predict_proba(
        self,
        X: pd.DataFrame,
        primary_pred: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict P(trade_profitable | features, primary_pred). Returns [0,1]."""
        if not self._is_fitted:
            # Return 0.5 (neutral) if not fitted
            n = len(X) if hasattr(X, "__len__") else len(primary_pred) if primary_pred is not None else 0
            return np.full(n, 0.5)

        X_feat = self._prepare_features(X, primary_pred)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_feat)[:, 1]

        model_obj: Any = self.model
        if hasattr(model_obj, "decision_function"):
            scores = model_obj.decision_function(X_feat)
            return 1 / (1 + np.exp(-scores))

        preds = model_obj.predict(X_feat)
        return preds.astype(float)

    def should_trade(
        self,
        X: pd.DataFrame,
        primary_pred: np.ndarray | None = None,
    ) -> np.ndarray:
        """Boolean mask: True where P(profitable) > threshold."""
        probs = self.predict_proba(X, primary_pred)
        return probs >= self.config.prob_threshold


# ════════════════════════════════════════════════════════════════════════════
# 2. Conformal Prediction for Abstention
# ════════════════════════════════════════════════════════════════════════════


def conformal_abstention_scores(
    logits: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.10,
) -> tuple[np.ndarray, float, dict]:
    """
    Compute conformal prediction sets and abstention scores.

    Returns:
        prediction_sets: bool array (n_samples, n_classes) where True = class in set
        threshold: conformal threshold (1-alpha quantile of nonconformity scores)
        info: dict with coverage, avg_set_size, etc.
    """
    # Softmax probabilities
    exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

    n = len(labels)
    # Nonconformity score: 1 - P(true_label)
    scores = 1.0 - probs[np.arange(n), labels]

    # Conformal threshold: (1-alpha) quantile
    threshold = float(np.quantile(scores, 1.0 - alpha))

    # Prediction sets: classes j where 1 - probs[j] <= threshold
    # i.e., probs[j] >= 1 - threshold
    sets = probs >= (1.0 - threshold)

    coverage = float(np.mean(sets[np.arange(n), labels]))
    avg_set_size = float(np.mean(np.sum(sets, axis=-1)))

    # Ambiguity: set contains both long (1) and short (-1) classes
    # Assuming labels: 0=hold, 1=long, -1=short (need mapping)
    # For 3-class: 0=short, 1=hold, 2=long
    ambiguity = np.logical_and(sets[:, 0], sets[:, 2]) if logits.shape[1] >= 3 else np.zeros(len(logits), dtype=bool)

    info = {
        "coverage": coverage,
        "threshold": threshold,
        "avg_set_size": avg_set_size,
        "ambiguity_rate": float(ambiguity.mean()),
    }

    return sets, threshold, info


def conformal_should_abstain(
    logits: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.10,
    abstain_on_ambiguous: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Determine abstention based on conformal prediction sets.

    Returns:
        abstain_mask: True where should abstain
        info: conformal info dict
    """
    sets, _threshold, info = conformal_abstention_scores(logits, labels, alpha)

    if abstain_on_ambiguous:
        # Abstain if prediction set contains both long and short
        # For 3-class: 0=short, 1=hold, 2=long
        n_classes = sets.shape[1]
        if n_classes == 3:
            # Classes: 0=short, 1=hold, 2=long
            ambiguous = np.logical_and(sets[:, 0], sets[:, 2])
            # Also abstain if set is empty (shouldn't happen with proper alpha)
            empty = sets.sum(axis=1) == 0
            abstain = np.logical_or(ambiguous, empty)
        else:
            # Multi-class: abstain if >1 class in set
            ambiguous = sets.sum(axis=1) > 1
            abstain = ambiguous
    else:
        # Only abstain if true label not in set (shouldn't happen with correct alpha)
        abstain = np.zeros(len(logits), dtype=bool)

    info["abstain_rate"] = float(abstain.mean())
    return abstain, info


# ════════════════════════════════════════════════════════════════════════════
# 3. Heuristic No-Trade Score (enhanced)
# ════════════════════════════════════════════════════════════════════════════


def compute_heuristic_no_trade_score(
    features: pd.DataFrame,
    atr_col: str = "atr_6",
    vol_quantile: float = 0.33,
    ofi_quantile: float = 0.30,
    trend_quantile: float = 0.05,
) -> np.ndarray:
    """
    Enhanced heuristic no-trade score combining:
    - Low volatility (ATR below quantile)
    - Neutral order flow imbalance
    - Choppy/flat trend

    Returns score in [0, 1] where 1 = strong no-trade signal.
    """
    atr = features.get(atr_col)
    if atr is None:
        return np.zeros(len(features))

    atr = np.asarray(atr, dtype=float)
    n = len(atr)

    # Low volatility: ATR below rolling quantile
    vol_q = np.full(n, np.nan)
    window = min(200, n // 2)
    for i in range(window, n):
        vol_q[i] = np.quantile(atr[i - window : i], vol_quantile)
    vol_q = pd.Series(vol_q).ffill().fillna(atr.max()).values
    low_vol = (atr < vol_q).astype(float)

    # Neutral OFI / order flow (use spread or volume proxy if OFI not available)
    ofi_proxy = features.get("spread_pips", np.zeros(n))
    if isinstance(ofi_proxy, pd.Series):
        ofi_proxy = ofi_proxy.values
    ofi_q = np.full(n, np.nan)
    for i in range(window, n):
        ofi_q[i] = np.quantile(np.abs(ofi_proxy[i - window : i]), ofi_quantile)
    ofi_q = pd.Series(ofi_q).ffill().fillna(np.abs(ofi_proxy).max()).values
    neutral_ofi = (np.abs(ofi_proxy) < ofi_q).astype(float)

    # Choppy trend: low ADX (using causal rolling 0.3 quantile) or flat RSI
    adx = features.get("adx_14", np.zeros(n))
    if isinstance(adx, pd.Series):
        adx = adx.to_numpy(dtype=float)
    adx_arr = np.asarray(adx, dtype=float)

    rsi = features.get("rsi_14", np.zeros(n))
    if isinstance(rsi, pd.Series):
        rsi = rsi.to_numpy(dtype=float)
    rsi_arr = np.asarray(rsi, dtype=float)

    # Causal rolling quantile (D4 fix)
    adx_s = pd.Series(adx_arr)
    adx_q = adx_s.replace(0, np.nan).rolling(window=200, min_periods=10).quantile(0.3).ffill().fillna(0).to_numpy(dtype=float)

    trend_unstable = ((adx_arr < adx_q) & (np.abs(rsi_arr - 50.0) < 10.0)).astype(float)

    score = (low_vol + neutral_ofi + trend_unstable) / 3.0
    return np.clip(score, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# 4. Unified No-Trade Decision
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class NoTradeConfig:
    """Unified configuration for no-trade decision."""

    # Heuristic score weight
    heuristic_weight: float = 0.3
    # Learned abstention weight
    learned_weight: float = 0.4
    # Conformal abstention weight
    conformal_weight: float = 0.3
    # Threshold for final no-trade decision (higher = more conservative)
    no_trade_threshold: float = 0.5
    # Conformal alpha
    conformal_alpha: float = 0.10
    # Whether to require ALL signals to agree (AND) or weighted average (AVG)
    mode: str = "avg"  # "avg" or "and"


class NoTradeZoneManager:
    """
    Unified no-trade zone manager combining:
    1. Heuristic no-trade score (volatility, OFI, trend)
    2. Learned abstention model (probabilistic)
    3. Conformal prediction abstention (coverage-guaranteed)

    Produces a single no-trade decision per bar.
    """

    def __init__(
        self,
        config: NoTradeConfig,
        abstention_model: Any | None = None,  # LearnedAbstentionModel instance
        conformal_calibrator: Callable | None = None,  # function(logits) -> (abstain, info)
    ):
        self.config = config
        self.abstention_model = abstention_model
        self.conformal_calibrator = conformal_calibrator
        self._is_fitted = False

    def fit_abstention(
        self,
        X: pd.DataFrame,
        tbm_labels: np.ndarray,
        primary_pred: np.ndarray,
    ) -> NoTradeZoneManager:
        """Fit the learned abstention model."""
        if self.abstention_model is None:
            abst_config = AbstentionConfig(
                prob_threshold=0.55,
                min_samples=500,
            )
            self.abstention_model = LearnedAbstentionModel(abst_config)
        self.abstention_model.fit(X, tbm_labels, primary_pred)
        return self

    def fit_conformal(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        alpha: float = 0.10,
    ) -> NoTradeZoneManager:
        """Calibrate conformal prediction on validation set."""
        # Store calibration data for later use
        self._conformal_logits = val_logits
        self._conformal_labels = val_labels
        self._conformal_alpha = alpha
        _, info = conformal_should_abstain(val_logits, val_labels, alpha=alpha, abstain_on_ambiguous=True)
        self._conformal_info = info
        print(
            f"[NoTradeZoneManager] Conformal calibrated: coverage={info['coverage']:.3f}, "
            f"ambiguity_rate={info['ambiguity_rate']:.3f}, abstain_rate={info['abstain_rate']:.3f}"
        )
        return self

    def compute_no_trade_mask(
        self,
        X: pd.DataFrame,
        primary_pred: np.ndarray,
        tbm_labels: np.ndarray | None = None,
        val_logits: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Compute unified no-trade mask.

        Returns:
            no_trade_mask: True where should abstain
            info: dict with individual scores and final decision
        """
        n = len(X)
        np.asarray(primary_pred, dtype=float).ravel()

        # 1. Heuristic score
        heuristic_score = compute_heuristic_no_trade_score(X)

        # 2. Learned abstention probability
        if self.abstention_model is not None and self.abstention_model._is_fitted:
            learned_prob = self.abstention_model.predict_proba(X, primary_pred)
            learned_abstain = 1.0 - learned_prob  # high prob_trade = low abstain
        else:
            learned_abstain = np.full(n, 0.5)  # neutral

        # 3. Conformal abstention
        if val_logits is not None and self._conformal_info:
            # Use the calibrated threshold from calibration set
            threshold = self._conformal_info["threshold"]
            # Apply threshold to val_logits (or main logits if provided)
            apply_logits = val_logits
            if len(apply_logits) != n:
                # If lengths don't match, use the average abstain rate
                conformal_abstain = np.full(n, float(self._conformal_info["abstain_rate"]))
            else:
                # Apply threshold to get prediction sets
                exp_l = np.exp(apply_logits - np.max(apply_logits, axis=-1, keepdims=True))
                probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
                sets = probs >= (1.0 - threshold)
                # Ambiguity: set contains both short (0) and long (2) classes
                n_classes = sets.shape[1]
                if n_classes >= 3:
                    ambiguous = np.logical_and(sets[:, 0], sets[:, 2])
                    empty = sets.sum(axis=1) == 0
                    conformal_abstain = np.logical_or(ambiguous, empty).astype(float)
                else:
                    conformal_abstain = (sets.sum(axis=1) > 1).astype(float)
        else:
            conformal_abstain = np.full(n, 0.5)  # neutral

        # Combine signals
        if self.config.mode == "and":
            # All signals must agree
            heuristic_flag = heuristic_score > self.config.no_trade_threshold
            learned_flag = learned_abstain > self.config.no_trade_threshold
            conformal_flag = conformal_abstain > self.config.no_trade_threshold
            no_trade_mask = heuristic_flag & learned_flag & conformal_flag
            combined_score = np.maximum.reduce([heuristic_score, learned_abstain, conformal_abstain])
        else:
            # Weighted average
            weights = np.array([self.config.heuristic_weight, self.config.learned_weight, self.config.conformal_weight])
            weights = weights / weights.sum()
            combined_score = (
                weights[0] * heuristic_score + weights[1] * learned_abstain + weights[2] * conformal_abstain
            )
            no_trade_mask = combined_score > self.config.no_trade_threshold

        info = {
            "heuristic_score": heuristic_score,
            "learned_abstain": learned_abstain,
            "conformal_abstain": conformal_abstain,
            "combined_score": combined_score,
            "no_trade_mask": no_trade_mask,
        }
        return no_trade_mask, info


# ════════════════════════════════════════════════════════════════════════════
# 5. Integration with existing pipeline
# ════════════════════════════════════════════════════════════════════════════


def apply_no_trade_zones(
    features: pd.DataFrame,
    primary_pred: np.ndarray,
    tbm_labels: np.ndarray | None = None,
    no_trade_config: NoTradeConfig | None = None,
    abstention_model: Any | None = None,
    val_logits: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    One-shot function to compute and apply no-trade zones to features.

    Returns:
        features: DataFrame with added no_trade columns
        no_trade_mask: boolean mask
        info: dict with scores and mask
    """
    if no_trade_config is None:
        no_trade_config = NoTradeConfig()

    n = len(features)
    np.asarray(primary_pred, dtype=float).ravel()

    # Heuristic score (always available)
    heuristic_score = compute_heuristic_no_trade_score(features)

    # Learned abstention (if model provided)
    if abstention_model is not None and hasattr(abstention_model, "_is_fitted") and abstention_model._is_fitted:
        learned_prob = abstention_model.predict_proba(features, primary_pred)
        learned_abstain = 1.0 - learned_prob
    else:
        learned_abstain = np.full(n, 0.5)

    # Conformal (if calibration provided)
    conformal_abstain = np.full(n, 0.5)
    conformal_info = {}
    if val_logits is not None and val_labels is not None:
        # Calibrate on val set, but we need to apply to main data
        # For now, use the calibration info to set a global abstain rate
        # In practice, you'd apply the calibrator to the main logits
        conformal_abstain_cal, conformal_info = conformal_should_abstain(
            val_logits, val_labels, alpha=0.10, abstain_on_ambiguous=True
        )
        # Use the average abstain rate for the main data (or apply calibrator to main logits if available)
        if len(conformal_abstain_cal) == n:
            conformal_abstain = conformal_abstain_cal.astype(float)
        else:
            # Use average abstain rate
            conformal_abstain = np.full(n, float(conformal_abstain_cal.mean()))

    # Combine
    np.array([0.3, 0.4, 0.3])
    combined = 0.3 * heuristic_score + 0.4 * learned_abstain + 0.3 * conformal_abstain
    no_trade_mask = combined > 0.5

    # Add to features
    out = features.copy()
    out["no_trade_heuristic"] = heuristic_score
    out["no_trade_learned"] = learned_abstain
    out["no_trade_conformal"] = conformal_abstain
    out["no_trade_combined"] = combined
    out["no_trade_mask"] = no_trade_mask.astype(int)

    info = {
        "heuristic_score": heuristic_score,
        "learned_abstain": learned_abstain,
        "conformal_abstain": conformal_abstain,
        "combined": combined,
        "mask": no_trade_mask,
        "conformal_info": conformal_info,
    }
    return out, no_trade_mask, info
