"""
Meta-Labeling & Bayesian Barrier Search (Improvement #6)
========================================================
Builds on the existing Triple Barrier Method (TBM) to provide:

1. **Meta-Labeling** (Lopez de Prado, 2018): A secondary classifier that predicts
   whether the primary model's directional prediction will be profitable.
   - Trains on: primary model's predicted direction + market features
   - Target: did the trade hit TP before SL? (1) or not? (0)
   - At inference: only take trades where meta-model predicts P(profitable) > threshold

2. **Bayesian Barrier Search**: Finds optimal triple-barrier parameters
   (profit_mult, stop_mult, vertical_bars) by Bayesian optimization
   using Optuna, maximizing a configurable objective (Sharpe, win rate,
   profit factor, etc.).

Both components are self-contained and work with the existing TBM infrastructure
in `triple_barrier_labeling.py`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# 1. Meta-Labeling
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MetaLabelConfig:
    """Configuration for meta-labeling."""
    # Model for meta-classifier (any sklearn-compatible estimator)
    meta_model: any = None  # e.g., XGBClassifier, RandomForestClassifier
    # Features to use for meta-model (in addition to primary prediction)
    meta_features: list[str] | None = None  # column names from features DataFrame
    # How to map primary predictions: -1/0/1 -> direction
    primary_threshold: float = 0.0  # primary prediction > thresh -> long
    # Target definition: trade is "good" if hits TP before SL
    # (only defined for bars where primary made a non-hold prediction)
    min_meta_samples: int = 500  # minimum samples for meta-model training
    meta_train_frac: float = 0.7  # fraction for training (rest validation)
    # Probability threshold for taking trade
    meta_prob_threshold: float = 0.55
    # Random seed
    random_state: int = 42


class MetaLabeler:
    """
    Meta-labeling wrapper: trains a secondary classifier to predict whether
    the primary model's directional prediction will be profitable.

    Usage:
        meta = MetaLabeler(config, meta_model=xgb_clf)
        meta.fit(primary_preds, features, labels)  # labels from TBM
        meta_preds = meta.predict_proba(primary_preds, features)
        # Take trade only if meta_prob > threshold
    """

    def __init__(self, config: MetaLabelConfig):
        self.config = config
        self.meta_model = config.meta_model
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._scaler = None

    def _prepare_meta_features(
        self,
        primary_pred: np.ndarray,
        features: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Construct meta-feature matrix: primary prediction + optional features."""
        primary = np.asarray(primary_pred, dtype=float).reshape(-1, 1)
        if features is not None and self.config.meta_features:
            extra = features[self.config.meta_features].values.astype(float)
            return np.hstack([primary, extra])
        return primary

    def fit(
        self,
        primary_pred: np.ndarray,
        labels: np.ndarray,
        features: pd.DataFrame | None = None,
    ) -> MetaLabeler:
        """
        Train the meta-labeler.

        Args:
            primary_pred: Primary model's predictions (-1/0/1 or probabilities).
            labels: Ground truth labels from TBM (-1/0/1). Only bars where
                    primary != 0 (i.e., a trade was signaled) are used.
            features: Optional feature DataFrame for meta-features.
        """
        primary = np.asarray(primary_pred, dtype=float).ravel()
        y = np.asarray(labels, dtype=int).ravel()

# Only use bars where primary made a non-hold prediction
        trade_mask = primary != 0
        if not trade_mask.any():
            warnings.warn("[MetaLabeler] No trades signaled by primary model.")
            return self

        primary_trades = primary[trade_mask]
        y_trades = y[trade_mask]

        # Target: 1 if trade was profitable (hit TP before SL), 0 otherwise
        # (only defined for bars where primary made a non-hold prediction)
        meta_y = np.where(
            (primary_trades > 0) & (y_trades == 1) |
            (primary_trades < 0) & (y_trades == -1),
            1, 0
        )

        if len(meta_y) < self.config.min_meta_samples:
            warnings.warn(
                f"[MetaLabeler] Only {len(meta_y)} trade samples, "
                f"minimum {self.config.min_meta_samples}. Skipping meta-training."
            )
            return self

        # Build meta features
        if features is not None and self.config.meta_features:
            feat_trades = features.iloc[trade_mask][self.config.meta_features]
            X = np.hstack([
                primary_trades.reshape(-1, 1),
                feat_trades.values.astype(float)
            ])
            self._feature_names = ["primary_pred"] + list(self.config.meta_features)
        else:
            X = primary_trades.reshape(-1, 1)
            self._feature_names = ["primary_pred"]

        # Train/validation split (temporal)
        n = len(X)
        split = int(n * self.config.meta_train_frac)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = meta_y[:split], meta_y[split:]

        # Train meta-model
        if self.meta_model is None:
            try:
                from sklearn.ensemble import RandomForestClassifier
                self.meta_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=20,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
            except ImportError:
                raise ImportError("scikit-learn required for default meta-model")

        self.meta_model.fit(X_train, y_train)

        # Validate
        train_score = self.meta_model.score(X_train, y_train)
        val_score = self.meta_model.score(X_val, y_val) if len(X_val) > 0 else 0.0

        self._is_fitted = True
        print(f"[MetaLabeler] Trained: train_acc={train_score:.3f}, val_acc={val_score:.3f}, "
              f"n_train={len(X_train)}, n_val={len(X_val)}")
        return self

    def predict_proba(self, primary_pred: np.ndarray, features: pd.DataFrame | None = None) -> np.ndarray:
        """Predict P(profitable | primary_pred, features). Returns probabilities in [0,1]."""
        if not self._is_fitted:
            # Return zeros if not fitted (e.g., no trades to train on)
            return np.zeros(len(primary_pred))
        X = self._prepare_meta_features(primary_pred, features)
        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(X)[:, 1]
        # Fallback: use decision_function or predict
        if hasattr(self.meta_model, "decision_function"):
            scores = self.meta_model.decision_function(X)
            return 1 / (1 + np.exp(-scores))
        preds = self.meta_model.predict(X)
        return preds.astype(float)

    def should_trade(
        self,
        primary_pred: np.ndarray,
        features: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Boolean mask: True where meta-model predicts P(profitable) > threshold."""
        probs = self.predict_proba(primary_pred, features)
        return probs >= self.config.meta_prob_threshold


# ════════════════════════════════════════════════════════════════════════════
# 2. Bayesian Barrier Search
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class BarrierSearchSpace:
    """Search space for barrier parameters."""
    profit_mult: tuple[float, float] = (0.5, 3.0)   # profit_target_atr
    stop_mult: tuple[float, float] = (0.3, 2.0)     # stop_loss_atr
    vertical_bars: tuple[int, int] = (5, 40)        # lookahead_bars
    # Optional: execution delay, pip size (if variable)
    execution_delay_bars: tuple[int, int] = (0, 3)
    # Pip size (for JPY vs non-JPY)
    pip_size: tuple[float, float] | None = None


@dataclass
class BarrierSearchConfig:
    """Configuration for Bayesian barrier search."""
    search_space: BarrierSearchSpace = field(default_factory=BarrierSearchSpace)
    n_trials: int = 50
    timeout: float | None = None  # seconds
    # Objective: "sharpe", "win_rate", "profit_factor", "expectancy"
    objective: str = "sharpe"
    # Minimum trades per trial
    min_trades_per_trial: int = 50
    # Study direction
    direction: str = "maximize"
    # Random seed
    seed: int = 42
    # Pruner
    pruner: str = "median"  # "median", "hyperband", "none"


class BayesianBarrierOptimizer:
    """
    Bayesian optimization of triple-barrier parameters using Optuna.

    The optimizer evaluates candidate (profit_mult, stop_mult, vertical_bars)
    by running the TBM on the provided data and computing the chosen objective.

    Usage:
        optimizer = BayesianBarrierOptimizer(config)
        best_params = optimizer.optimize(bars, features, primary_model_predict_fn)
    """

    def __init__(self, config: BarrierSearchConfig):
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna required for BayesianBarrierOptimizer. pip install optuna")
        self.config = config
        self.study = None
        self._best_params = None

    def _objective(
        self,
        trial: optuna.Trial,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        primary_pred_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    ) -> float:
        """Objective function for Optuna trial."""
        # Sample parameters
        profit_mult = trial.suggest_float(
            "profit_mult",
            self.config.search_space.profit_mult[0],
            self.config.search_space.profit_mult[1],
        )
        stop_mult = trial.suggest_float(
            "stop_mult",
            self.config.search_space.stop_mult[0],
            self.config.search_space.stop_mult[1],
        )
        vertical_bars = trial.suggest_int(
            "vertical_bars",
            self.config.search_space.vertical_bars[0],
            self.config.search_space.vertical_bars[1],
        )
        delay = trial.suggest_int(
            "execution_delay_bars",
            self.config.search_space.execution_delay_bars[0],
            self.config.search_space.execution_delay_bars[1],
        )

        # Get primary model predictions on this data
        primary_pred = primary_pred_fn(bars, features)

        # Run TBM with candidate parameters
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels

        tbm_result = compute_triple_barrier_labels(
            bars=bars,
            features=features,
            profit_atr_mult=profit_mult,
            stop_atr_mult=stop_mult,
            vertical_bars=vertical_bars,
            execution_delay_bars=delay,
        )

        if len(tbm_result) == 0:
            return -1e9  # penalty for no labels

        labels = tbm_result["label"].values
        n_trades = np.sum(labels != 0)

        if n_trades < self.config.min_trades_per_trial:
            return -1e9  # penalty for too few trades

        # Compute objective
        rewards_long = tbm_result["reward_long"].values
        rewards_short = tbm_result["reward_short"].values

        # Only consider trades that were actually taken (non-zero label)
        trade_mask = labels != 0
        if not trade_mask.any():
            return -1e9

        # For each trade, use the reward in the direction of the label
        trade_rewards = np.where(
            labels[trade_mask] > 0,
            rewards_long[trade_mask],
            rewards_short[trade_mask],
        )

        if self.config.objective == "sharpe":
            if len(trade_rewards) < 2 or trade_rewards.std() == 0:
                return -1e9
            return trade_rewards.mean() / trade_rewards.std() * np.sqrt(252)
        elif self.config.objective == "win_rate":
            return (trade_rewards > 0).mean()
        elif self.config.objective == "profit_factor":
            pos = trade_rewards[trade_rewards > 0].sum()
            neg = -trade_rewards[trade_rewards < 0].sum()
            if neg == 0:
                return 1e6
            return pos / neg
        elif self.config.objective == "expectancy":
            return trade_rewards.mean()
        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")

    def optimize(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        primary_pred_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    ) -> dict[str, any]:
        """
        Run Bayesian optimization to find best barrier parameters.

        Args:
            bars: OHLCV DataFrame with 'close' (and optionally bid/ask)
            features: Feature DataFrame aligned to bars
            primary_pred_fn: Function(bars, features) -> primary predictions (-1/0/1)

        Returns:
            Dict with best parameters and study object.
        """
        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=optuna.samplers.TPESampler(seed=self.config.seed),
            pruner=(
                optuna.pruners.MedianPruner()
                if self.config.pruner == "median"
                else optuna.pruners.HyperbandPruner()
                if self.config.pruner == "hyperband"
                else optuna.pruners.NopPruner()
            ),
        )

        def objective(trial):
            return self._objective(trial, bars, features, primary_pred_fn)

        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )

        self._best_params = self.study.best_params
        print(f"[BayesianBarrierOptimizer] Best params: {self._best_params}")
        print(f"[BayesianBarrierOptimizer] Best value: {self.study.best_value:.4f}")
        return {
            "best_params": self._best_params,
            "best_value": self.study.best_value,
            "study": self.study,
        }

    @property
    def best_params(self) -> dict | None:
        return self._best_params


# ════════════════════════════════════════════════════════════════════════════
# 3. Integrated Pipeline: TBM + Meta-Labeling + Bayesian Search
# ════════════════════════════════════════════════════════════════════════════

def run_meta_tbm_pipeline(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    primary_model,
    meta_features: list[str] | None = None,
    meta_model: any = None,
    tbm_params: dict | None = None,
    bayesian_search: bool = False,
    bayesian_config: BarrierSearchConfig | None = None,
) -> tuple[pd.DataFrame, MetaLabeler, BayesianBarrierOptimizer | None]:
    """
    Full meta-labeling + TBM pipeline.

    1. Optionally run Bayesian barrier search to find optimal TBM parameters
    2. Run TBM with (optimized or default) parameters to get ground-truth labels
    3. Get primary model predictions
    4. Train meta-labeler on (primary_pred, features, TBM_labels)
    5. Return filtered labels (only where meta P(profitable) > threshold)

    Returns:
        filtered_labels: DataFrame with 'label' column (only high-confidence trades)
        meta_labeler: Fitted MetaLabeler instance
        bayesian_optimizer: BayesianBarrierOptimizer if search was run, else None
    """
    # Step 1: Bayesian barrier search (optional)
    bayesian_opt = None
    if bayesian_search:
        if bayesian_config is None:
            bayesian_config = BarrierSearchConfig()
        optimizer = BayesianBarrierOptimizer(bayesian_config)
        optimizer.optimize(
            bars=bars,
            features=features,
            primary_pred_fn=lambda b, f: primary_model.predict(b, f) if hasattr(primary_model, "predict") else primary_model(b, f),
        )
        bayesian_opt = optimizer
        tbm_params = tbm_params or {}
        tbm_params.update(optimizer.best_params)

    # Step 2: Run TBM with best/fixed parameters
    tbm_defaults = {
        "profit_atr_mult": 1.8,
        "stop_atr_mult": 0.9,
        "vertical_bars": 20,
        "execution_delay_bars": 1,
    }
    tbm_defaults.update(tbm_params or {})

    from labeling.triple_barrier_labeling import compute_triple_barrier_labels
    tbm_result = compute_triple_barrier_labels(
        bars=bars,
        features=features,
        **tbm_defaults,
    )

    # Step 3: Get primary model predictions
    if hasattr(primary_model, "predict"):
        primary_pred = primary_model.predict(bars, features)
    elif hasattr(primary_model, "predict_proba"):
        # Use probability difference as continuous prediction
        probs = primary_model.predict_proba(features)
        primary_pred = probs[:, 1] - probs[:, 0]  # long - short
    else:
        primary_pred = primary_model(bars, features)

    # Step 5: Train meta-labeler
    meta_config = MetaLabelConfig(
        meta_model=meta_model,
        meta_features=meta_features,
    )
    meta = MetaLabeler(meta_config)
    meta.fit(primary_pred, tbm_result["label"].values, features)

    # Step 6: Filter labels by meta-model confidence
    trade_mask = meta.should_trade(primary_pred, features)
    filtered = tbm_result.copy()
    filtered.loc[~trade_mask, "label"] = 0  # suppress low-confidence trades

    return filtered, meta, bayesian_opt


# ════════════════════════════════════════════════════════════════════════════
# 4. Convenience: quick evaluation of barrier parameters
# ════════════════════════════════════════════════════════════════════════════

def evaluate_barrier_params(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    profit_mult: float,
    stop_mult: float,
    vertical_bars: int,
    primary_pred_fn: Callable,
    delay: int = 1,
) -> dict[str, float]:
    """Quick evaluation of a single parameter set. Returns metrics dict."""
    from labeling.triple_barrier_labeling import compute_triple_barrier_labels

    primary_pred = primary_pred_fn(bars, features)
    tbm = compute_triple_barrier_labels(
        bars=bars,
        features=features,
        profit_atr_mult=profit_mult,
        stop_atr_mult=stop_mult,
        vertical_bars=vertical_bars,
        execution_delay_bars=1,
    )

    if len(tbm) == 0:
        return {"error": "no_labels"}

    labels = tbm["label"].values
    trade_mask = labels != 0
    if not trade_mask.any():
        return {"n_trades": 0}

    rewards_long = tbm["reward_long"].values
    rewards_short = tbm["reward_short"].values
    trade_rewards = np.where(
        labels[trade_mask] > 0,
        rewards_long[trade_mask],
        rewards_short[trade_mask],
    )

    return {
        "n_trades": int(trade_mask.sum()),
        "win_rate": float((trade_rewards > 0).mean()),
        "avg_reward": float(trade_rewards.mean()),
        "sharpe": float(trade_rewards.mean() / trade_rewards.std() * np.sqrt(252)) if trade_rewards.std() > 0 else 0,
        "profit_factor": float(trade_rewards[trade_rewards > 0].sum() / -trade_rewards[trade_rewards < 0].sum()) if (trade_rewards < 0).any() else 1e6,
        "expectancy": float(trade_rewards.mean()),
        "max_dd": float((np.maximum.accumulate(trade_rewards.cumsum()) - trade_rewards.cumsum()).max()),
    }
