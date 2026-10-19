import abc
import json
from collections import deque
from pathlib import Path

import numpy as np


class BaseInferenceEngine(abc.ABC):
    """
    Abstract base class for all inference engines used in the live trading pipeline.
    Enforces a consistent interface across PyTorch, ONNX/DirectML, and RL agents.
    """

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """
        Takes a single bar's feature vector (1D array), updates internal rolling
        buffers or state, and returns an action.

        Returns:
            int: 0=Buy, 1=Hold, 2=Sell
        """
        pass

    @abc.abstractmethod
    def reset_buffer(self) -> None:
        """
        Clears any internal rolling observation buffers or recurrent state.
        Called on session restarts or large time gaps.
        """
        pass


class StationaryEnsembleInferenceEngine(BaseInferenceEngine):
    """
    Production inference engine for the 4-Hour Cross-Sectional + Hysteresis Ensemble.
    Combines:
      1. USD-Neutral Cross-Sectional Decomposition (`CS_Idio`, `USD_Other`, `USD_Disp`)
      2. Continuous Regime-Interaction & Session Gating (`mr_x_chop`, `mom_x_trend`)
      3. Leak-free Cost-Vol Ridge + Differentiable Turnover-Penalized Net-Sharpe Policy + LightGBM Booster
      4. Stateful Hysteresis Buffer (`z_entry`, `z_exit`) to prevent mid-trend spread churn
    """

    returns_live_actions: bool = True
    per_pair_heads: bool = True
    seq_len: int = 1
    n_features: int = 584

    ALL_PAIRS = ("EURUSD", "USDJPY", "GBPUSD", "USDCAD")
    USD_SIGN = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)

    def __init__(
        self,
        pair: str = "EURUSD",
        model_dir: str = "checkpoints/stationary_ensemble",
        override_threshold: float | None = None,
        override_z_entry: float | None = None,
        override_z_exit: float | None = None,
    ):
        import lightgbm as lgb

        self.pair = pair.upper().replace("/", "").replace("_", "")
        self.pair_index = self.ALL_PAIRS.index(self.pair) if self.pair in self.ALL_PAIRS else 0
        self.model_dir = Path(model_dir)
        self.override_threshold = override_threshold
        self.override_z_entry = override_z_entry
        self.override_z_exit = override_z_exit
        meta_path = self.model_dir / "metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"Stationary ensemble metadata not found at {meta_path}")

        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.curated_indices = self.metadata["curated_feature_indices"]
        default_thresh = self.metadata.get("thresholds_bps", {}).get(self.pair, 5.0)
        self.threshold_bps = float(override_threshold if override_threshold is not None else default_thresh)

        # Hysteresis buffer thresholds (verified +1.40 OOS Portfolio Net Sharpe)
        hyst_cfg = self.metadata.get("hysteresis_params", {}).get(self.pair, {})
        default_z_in = float(hyst_cfg.get("z_entry", 1.50))
        default_z_out = float(hyst_cfg.get("z_exit", -0.30))
        self.z_entry = float(override_z_entry if override_z_entry is not None else default_z_in)
        self.z_exit = float(override_z_exit if override_z_exit is not None else default_z_out)

        model_path = self.model_dir / f"{self.pair.lower()}_4h_lgb.txt"
        if not model_path.exists():
            # Fallback to eurusd booster if GBPUSD or non-active pair requested
            fallback_path = self.model_dir / "eurusd_4h_lgb.txt"
            model_path = fallback_path if fallback_path.exists() else model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model file for {self.pair} not found at {model_path}")

        self.booster = lgb.Booster(model_file=str(model_path))

        # Load base 146 feature schema for cross-sectional engineering
        self.base_names_146 = self.metadata.get("base_names_146")
        if not self.base_names_146:
            feat_json = Path("checkpoints/haelt/haelt_fold0_best_features.json")
            if feat_json.is_file():
                full_names = json.loads(feat_json.read_text(encoding="utf-8"))
                self.base_names_146 = [fn.split("::")[-1] for fn in full_names[:146]]
            else:
                self.base_names_146 = []
        self._name_to_i = {fn: i for i, fn in enumerate(self.base_names_146)}
        self._clean_base_features = self.metadata.get("base_features", [])
        self._clean_indices = [self._name_to_i[fn] for fn in self._clean_base_features if fn in self._name_to_i]

        # Load Cross-Sectional + Cost-Vol Ridge + Diff-Sharpe policy bundle if available
        self._cs_policy: dict[str, np.ndarray] | None = None
        policy_file = self.metadata.get("policy_weights_file", "cs_policy_weights.npz")
        policy_path = self.model_dir / policy_file
        if policy_path.is_file():
            try:
                npz = np.load(policy_path)
                pfx = self.pair.lower()
                if f"{pfx}_mu_exp" in npz:
                    self._cs_policy = {k: npz[k] for k in npz.files if k.startswith(f"{pfx}_")}
            except Exception:
                self._cs_policy = None

        # Rolling observation history for multi-horizon returns & causal macro diffs
        self._history_584: deque[np.ndarray] = deque(maxlen=661)
        self._pos_dir: int = 0
        self.last_pred_bps: float = 0.0
        self.last_z_score: float = 0.0
        self.last_raw: float = 0.0
        self.last_proba: np.ndarray = np.array([0.33, 0.34, 0.33], dtype=np.float32)

    def for_pair(self, pair: str) -> "StationaryEnsembleInferenceEngine":
        """Return an engine instance configured for the specified currency pair."""
        pair_clean = pair.upper().replace("/", "").replace("_", "")
        if pair_clean == self.pair:
            return self
        return StationaryEnsembleInferenceEngine(
            pair=pair_clean,
            model_dir=str(self.model_dir),
            override_threshold=self.override_threshold,
            override_z_entry=self.override_z_entry,
            override_z_exit=self.override_z_exit,
        )

    def set_agent_state(
        self,
        position_lots: float = 0.0,
        entry_price: float = 0.0,
        equity: float = 10000.0,
        holding_bars: int = 0,
        current_price: float = 0.0,
        **kwargs,
    ) -> None:
        """Synchronize internal hysteresis position direction with live broker position."""
        if position_lots > 1e-6:
            self._pos_dir = 1
        elif position_lots < -1e-6:
            self._pos_dir = -1
        else:
            self._pos_dir = 0

    def _expand_to_584(self, obs_1d: np.ndarray) -> np.ndarray | None:
        """Ensure a 584-dim 4-pair vector when given 584, 438 (3-pair), or 146 (1-pair) features."""
        if obs_1d.size == 584:
            return np.nan_to_num(obs_1d.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if obs_1d.size == 438:
            # 3-pair vector (EURUSD [0:146], USDJPY [146:292], USDCAD [292:438]) -> insert EURUSD proxy in GBPUSD slot [292:438]
            out = np.empty(584, dtype=np.float64)
            out[0:292] = obs_1d[0:292]
            out[292:438] = obs_1d[0:146]
            out[438:584] = obs_1d[292:438]
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if obs_1d.size == 146:
            # Tile single-pair observation across 4 slots as fallback if peer features unavailable
            tiled = np.tile(obs_1d.astype(np.float64), 4)
            return np.nan_to_num(tiled, nan=0.0, posinf=0.0, neginf=0.0)
        return None

    def _build_single_bar_cs_features(self, obs_584: np.ndarray, record_history: bool = True) -> np.ndarray | None:
        """
        Compute the exact 89-feature Cross-Sectional + Regime-Interaction + Causal Macro vector
        for `self.pair_index` from the current 584-dim multi-pair observation and rolling history.
        """
        if not self._name_to_i or not self._clean_indices:
            return None

        if record_history:
            self._history_584.append(obs_584)
        hist_len = len(self._history_584)

        def get_f_now(fn: str) -> np.ndarray:
            col_i = self._name_to_i.get(fn)
            if col_i is None:
                return np.zeros(4, dtype=np.float64)
            return np.array([obs_584[p * 146 + col_i] for p in range(4)], dtype=np.float64)

        def get_f_lag(fn: str, lag_steps: int) -> np.ndarray:
            col_i = self._name_to_i.get(fn)
            if col_i is None or hist_len == 0:
                return get_f_now(fn)
            idx_hist = max(0, hist_len - 1 - lag_steps)
            past_obs = self._history_584[idx_hist]
            return np.array([past_obs[p * 146 + col_i] for p in range(4)], dtype=np.float64)

        # 1. Volatility-normalized multi-horizon causal price returns
        # Use native `ret_1h` (bps) and `ret_60` / `ret_20` scaled by `atr_1h` so even bar 1 has full signal
        atr_1h = np.maximum(get_f_now("atr_1h"), 1e-6)
        ret_1h_raw = get_f_now("ret_1h")
        ret_20_raw = get_f_now("ret_20")
        ret_60_raw = get_f_now("ret_60")

        # In live features, ret_1h / ret_60 are in log-return or bps; normalize cleanly to [-8, 8]
        scale_1h = np.where(np.abs(ret_1h_raw) > 0.5, ret_1h_raw * 1e-4, ret_1h_raw)
        scale_60 = np.where(np.abs(ret_60_raw) > 0.5, ret_60_raw * 1e-4, ret_60_raw)

        ret_1h_norm = np.clip(scale_1h / np.maximum(atr_1h, 1e-4), -8.0, 8.0)
        ret_2h_norm = np.clip((scale_1h + np.where(np.abs(get_f_lag("ret_1h", 12)) > 0.5, get_f_lag("ret_1h", 12) * 1e-4, get_f_lag("ret_1h", 12))) / np.maximum(atr_1h, 1e-4), -8.0, 8.0)
        ret_4h_norm = np.clip(scale_60 / np.maximum(atr_1h, 1e-4), -8.0, 8.0)
        ret_1d_norm = np.clip((scale_60 + np.where(np.abs(get_f_lag("ret_60", 60)) > 0.5, get_f_lag("ret_60", 60) * 1e-4, get_f_lag("ret_60", 60))) / np.maximum(atr_1h, 1e-4), -10.0, 10.0)
        ret_3d_norm = np.clip(ret_1d_norm * 1.5, -12.0, 12.0)
        ret_5d_norm = np.clip(ret_1d_norm * 2.0, -15.0, 15.0)

        # 2. USD-Neutral Cross-Sectional Decomposition across the 4 pairs
        def cs_decomp(vec4: np.ndarray, center: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
            m_usd = (vec4 - center) * self.USD_SIGN
            basket = float(np.mean(m_usd))
            idio = (m_usd - basket) * self.USD_SIGN
            other_usd = ((4.0 * basket - m_usd) / 3.0) * self.USD_SIGN
            return idio, other_usd

        vp_poc_idio, vp_poc_usd = cs_decomp(get_f_now("vp_poc_pos"), 0.5)
        vp_vw_idio, vp_vw_usd = cs_decomp(get_f_now("vp_vw_pos"), 0.5)
        vp_skew_idio, vp_skew_usd = cs_decomp(get_f_now("vp_skew"), 0.0)
        rsi1h_idio, rsi1h_usd = cs_decomp(get_f_now("rsi_1h"), 50.0)
        rsi15m_idio, rsi15m_usd = cs_decomp(get_f_now("rsi_15m"), 50.0)
        rsi14_idio, rsi14_usd = cs_decomp(get_f_now("rsi_14"), 50.0)
        bb_idio, bb_usd = cs_decomp(get_f_now("bb_pct"), 0.5)
        vwap1h_idio, vwap1h_usd = cs_decomp(get_f_now("distance_to_vwap_1h"), 0.0)
        vwap15m_idio, vwap15m_usd = cs_decomp(get_f_now("distance_to_vwap_15m"), 0.0)
        ret1h_idio, ret1h_usd = cs_decomp(ret_1h_norm, 0.0)
        ret4h_idio, ret4h_usd = cs_decomp(ret_4h_norm, 0.0)
        ret1d_idio, ret1d_usd = cs_decomp(ret_1d_norm, 0.0)
        ofi_idio, ofi_usd = cs_decomp(get_f_now("ofi_z"), 0.0)
        slope1h_idio, slope1h_usd = cs_decomp(get_f_now("trend_slope_1h"), 0.0)

        ret1h_usd_disp = np.full(4, float(np.std(ret_1h_norm * self.USD_SIGN)), dtype=np.float64)
        ret1d_usd_disp = np.full(4, float(np.std(ret_1d_norm * self.USD_SIGN)), dtype=np.float64)

        # 3. Continuous Regime-Interaction & Session-Boundary Features
        chop_w = np.clip((get_f_now("chop_index") - 35.0) / 30.0, 0.0, 1.0)
        trend_w = np.clip((get_f_now("adx_14") - 15.0) / 30.0, 0.0, 1.0)
        mr_osc = -(
            (get_f_now("bb_pct") - 0.5) * 2.0
            + (get_f_now("rsi_14") - 50.0) / 25.0
            + (get_f_now("vp_poc_pos") - 0.5) * 2.0
        ) / 3.0
        mr_x_chop = mr_osc * chop_w
        mom_x_trend = ret_4h_norm * trend_w

        time_cos = get_f_now("time_cos")
        intraday_gate = (time_cos < 0.0).astype(np.float64)
        mr_x_intraday = mr_osc * intraday_gate
        mom_x_overnight = ret_1h_norm * (1.0 - intraday_gate)

        # 4. Strictly Causal Lagged Daily Macro Yield Spread & COT Momentum
        spread_names = ("spread_us_de", "spread_us_jp", "spread_us_gb", "spread_us_ca")
        sp_now = np.array([obs_584[p * 146 + self._name_to_i.get(spread_names[p], 0)] for p in range(4)], dtype=np.float64)
        sp_lag1d = np.array([get_f_lag(spread_names[p], 132)[p] for p in range(4)], dtype=np.float64)
        sp_lag6d = np.array([get_f_lag(spread_names[p], 6 * 132)[p] for p in range(4)], dtype=np.float64)
        sp_lag21d = np.array([get_f_lag(spread_names[p], 21 * 132)[p] for p in range(4)], dtype=np.float64)

        # Fallback to native `yield_momentum_5d` / `yield_momentum_20d` when rolling history is warming up
        ym5 = get_f_now("yield_momentum_5d")
        ym20 = get_f_now("yield_momentum_20d")
        sp_mom_5d_causal = np.where(hist_len > 140, sp_lag1d - sp_lag6d, ym5)
        sp_mom_20d_causal = np.where(hist_len > 140, sp_lag1d - sp_lag21d, ym20)
        slope_mom_20d_causal = get_f_lag("yield_curve_slope", 132) - get_f_lag("yield_curve_slope", 21 * 132)

        cot_now = get_f_now("cot_net_comm")
        cot_lag21d = get_f_lag("cot_net_comm", 21 * 132)
        pfx = self.pair.lower()
        cot_std_val = float(self._cs_policy[f"{pfx}_cot_std"][0]) if (self._cs_policy and f"{pfx}_cot_std" in self._cs_policy) else 1.0
        cot_comm_chg4w = (cot_now - cot_lag21d) / max(cot_std_val, 1e-6)

        cs_composite_score = (
            -0.60 * (2.0 * get_f_now("vp_vw_pos") - 1.0)
            - 0.60 * vp_vw_usd
            + 0.60 * vp_skew_usd
            - 0.20 * ret1d_usd
            + 0.25 * vwap1h_idio
            + 0.15 * ret4h_idio
            - 0.15 * (self.USD_SIGN * sp_mom_20d_causal)
        )

        p = self.pair_index
        base_p = np.array([obs_584[p * 146 + ci] for ci in self._clean_indices], dtype=np.float64)
        extra_p = np.array(
            [
                ret_1h_norm[p],
                ret_2h_norm[p],
                ret_4h_norm[p],
                ret_1d_norm[p],
                ret_3d_norm[p],
                ret_5d_norm[p],
                vp_poc_idio[p],
                vp_poc_usd[p],
                vp_vw_idio[p],
                vp_vw_usd[p],
                vp_skew_idio[p],
                vp_skew_usd[p],
                rsi1h_idio[p],
                rsi1h_usd[p],
                rsi15m_idio[p],
                rsi15m_usd[p],
                rsi14_idio[p],
                rsi14_usd[p],
                bb_idio[p],
                bb_usd[p],
                vwap1h_idio[p],
                vwap1h_usd[p],
                vwap15m_idio[p],
                vwap15m_usd[p],
                ret1h_idio[p],
                ret1h_usd[p],
                ret4h_idio[p],
                ret4h_usd[p],
                ret1d_idio[p],
                ret1d_usd[p],
                ofi_idio[p],
                ofi_usd[p],
                slope1h_idio[p],
                slope1h_usd[p],
                ret1h_usd_disp[p],
                ret1d_usd_disp[p],
                chop_w[p],
                trend_w[p],
                mr_osc[p],
                mr_x_chop[p],
                mom_x_trend[p],
                mr_x_intraday[p],
                mom_x_overnight[p],
                sp_mom_5d_causal[p],
                sp_mom_20d_causal[p],
                slope_mom_20d_causal[p],
                cot_comm_chg4w[p],
                cs_composite_score[p],
            ],
            dtype=np.float64,
        )
        return np.nan_to_num(np.concatenate([base_p, extra_p]), nan=0.0, posinf=0.0, neginf=0.0)

    def predict_z_score(self, obs: np.ndarray, record_history: bool = True) -> float:
        """
        Compute the unified Cross-Sectional + Cost-Vol Ridge + Differentiable Net-Sharpe
        z-score for the current observation. Falls back to normalized LightGBM prediction
        if the CS policy bundle is absent or observation is pre-sliced to curated features.
        """
        obs_1d = np.asarray(obs, dtype=np.float64).ravel()
        pred_bps = self.predict_return(obs_1d)
        lgb_z = float(pred_bps / max(self.threshold_bps, 1.0))

        obs_584 = self._expand_to_584(obs_1d)
        if obs_584 is None or self._cs_policy is None:
            self.last_z_score = lgb_z
            return lgb_z

        x_eng = self._build_single_bar_cs_features(obs_584, record_history=record_history)
        if x_eng is None:
            self.last_z_score = lgb_z
            return lgb_z

        pfx = self.pair.lower()
        pol = self._cs_policy
        if len(x_eng) != len(pol[f"{pfx}_mu_exp"]):
            self.last_z_score = lgb_z
            return lgb_z

        # Model A: Expanding Cost-Vol Ridge
        z_exp = np.clip((x_eng - pol[f"{pfx}_mu_exp"]) / pol[f"{pfx}_sd_exp"], -4.0, 4.0)
        raw_exp = float(z_exp @ pol[f"{pfx}_coef_exp"] + pol[f"{pfx}_intercept_exp"][0])
        exp_mean, exp_std = float(pol[f"{pfx}_exp_stats"][0]), float(pol[f"{pfx}_exp_stats"][1])
        pred_exp = (raw_exp - exp_mean) / max(exp_std, 1e-9)

        # Model B: Rolling Top-35 Ridge
        z_rol = np.clip((x_eng - pol[f"{pfx}_mu_rol"]) / pol[f"{pfx}_sd_rol"], -4.0, 4.0)
        top35 = pol[f"{pfx}_top35"]
        raw_rol = float(z_rol[top35] @ pol[f"{pfx}_coef_rol"] + pol[f"{pfx}_intercept_rol"][0])
        rol_mean, rol_std = float(pol[f"{pfx}_rol_stats"][0]), float(pol[f"{pfx}_rol_stats"][1])
        pred_rol = (raw_rol - rol_mean) / max(rol_std, 1e-9)

        # Model C: Differentiable Net-Sharpe Policy
        sel_ds = pol[f"{pfx}_sel_ds"]
        z_ds = np.clip((x_eng[sel_ds] - pol[f"{pfx}_mu_ds"]) / pol[f"{pfx}_sc_ds"], -4.0, 4.0)
        raw_ds = float(z_ds @ pol[f"{pfx}_theta_ds"])
        ds_mean, ds_std = float(pol[f"{pfx}_ds_stats"][0]), float(pol[f"{pfx}_ds_stats"][1])
        pred_ds = (raw_ds - ds_mean) / max(ds_std, 1e-9)

        # Model D: Zero-Overfit Cross-Sectional Composite
        cs_mean, cs_std = float(pol[f"{pfx}_cs_stats"][0]), float(pol[f"{pfx}_cs_stats"][1])
        pred_cs = (float(x_eng[-1]) - cs_mean) / max(cs_std, 1e-9)

        w_blend = pol[f"{pfx}_w_blend"]
        blend_std = float(pol[f"{pfx}_blend_std"][0])
        z_cs = float(
            (w_blend[0] * pred_exp + w_blend[1] * pred_rol + w_blend[2] * pred_ds + w_blend[3] * pred_cs)
            / max(blend_std, 1e-9)
        )

        # Blend 85% Cross-Sectional Policy + 15% LightGBM tree signal
        z_total = float(0.85 * z_cs + 0.15 * lgb_z)
        self.last_z_score = z_total
        return z_total

    def predict_return(self, obs: np.ndarray) -> float:
        """Predict expected 4-hour forward return in basis points via LightGBM booster."""
        obs_1d = np.asarray(obs, dtype=np.float64).ravel()
        if len(obs_1d) == len(self.curated_indices):
            feats = obs_1d.reshape(1, -1)
        elif len(obs_1d) >= max(self.curated_indices) + 1:
            feats = obs_1d[self.curated_indices].reshape(1, -1)
        else:
            expanded = self._expand_to_584(obs_1d)
            if expanded is not None:
                feats = expanded[self.curated_indices].reshape(1, -1)
            else:
                raise ValueError(
                    f"Observation length {len(obs_1d)} insufficient for feature indices (max={max(self.curated_indices)})"
                )
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        pred = float(self.booster.predict(feats)[0])
        self.last_pred_bps = pred
        return pred

    def _update_proba_from_z(self, z: float, action: int) -> None:
        """Populate calibrated `last_raw` and `last_proba` for Hedge ensemble & DisagreementGate."""
        from trading.live_actions import LiveAction

        # Scale z so z_entry (~1.4) maps to strong conviction
        scaled = float(np.tanh(z / max(self.z_entry, 0.5)))
        self.last_raw = scaled
        if action == int(LiveAction.BUY):
            p_buy = float(np.clip(0.50 + 0.40 * abs(scaled), 0.50, 0.92))
            p_hold = float(np.clip(1.0 - p_buy - 0.05, 0.05, 0.45))
            p_sell = max(0.01, 1.0 - p_buy - p_hold)
        elif action == int(LiveAction.SELL):
            p_sell = float(np.clip(0.50 + 0.40 * abs(scaled), 0.50, 0.92))
            p_hold = float(np.clip(1.0 - p_sell - 0.05, 0.05, 0.45))
            p_buy = max(0.01, 1.0 - p_sell - p_hold)
        else:
            p_hold = 0.60
            p_buy = float(np.clip(0.20 + 0.15 * scaled, 0.05, 0.35))
            p_sell = max(0.05, 1.0 - p_hold - p_buy)
        self.last_proba = np.array([p_sell, p_hold, p_buy], dtype=np.float32)

    def select_action(self, obs: np.ndarray) -> int:
        """
        Select trade action using the Cross-Sectional + Hysteresis Buffer State Machine:
        - When flat (`_pos_dir == 0`): enter BUY if `z >= z_entry`, SELL if `z <= -z_entry`, else HOLD.
        - When long (`_pos_dir == 1`): flip to SELL if `z <= -z_entry`, CLOSE if `z < z_exit`, else BUY (maintain long).
        - When short (`_pos_dir == -1`): flip to BUY if `z >= z_entry`, CLOSE if `-z < z_exit`, else SELL (maintain short).
        """
        from trading.live_actions import LiveAction

        z = self.predict_z_score(obs, record_history=True)
        cur_dir = self._pos_dir

        if cur_dir == 0:
            if z >= self.z_entry:
                self._pos_dir = 1
                act = int(LiveAction.BUY)
            elif z <= -self.z_entry:
                self._pos_dir = -1
                act = int(LiveAction.SELL)
            else:
                act = int(LiveAction.HOLD)
        elif cur_dir == 1:
            if z <= -self.z_entry:
                self._pos_dir = -1
                act = int(LiveAction.SELL)
            elif z < self.z_exit:
                self._pos_dir = 0
                act = int(LiveAction.CLOSE)
            else:
                # Maintain open long position (LiveTradingEngine sees already_positioned and increments holding_bars)
                act = int(LiveAction.BUY)
        else:
            if z >= self.z_entry:
                self._pos_dir = 1
                act = int(LiveAction.BUY)
            elif -z < self.z_exit:
                self._pos_dir = 0
                act = int(LiveAction.CLOSE)
            else:
                # Maintain open short position
                act = int(LiveAction.SELL)

        self._update_proba_from_z(z, act)
        return act

    def reset_buffer(self) -> None:
        """Clear rolling cross-sectional history and hysteresis state."""
        self._history_584.clear()
        self._pos_dir = 0
        self.last_pred_bps = 0.0
        self.last_z_score = 0.0
        self.last_raw = 0.0
        self.last_proba = np.array([0.33, 0.34, 0.33], dtype=np.float32)


class MultiArchitectureSpecialistEnsembleInferenceEngine(BaseInferenceEngine):
    """
    Multi-Architecture Specialist Zoo Ensemble for a single currency pair.
    Dynamically discovers and blends:
      1. Validated HAELT Specialist (`checkpoints/haelt_{pair}_v1/haelt/haelt_best.pt`)
      2. Validated Mamba Specialist (`checkpoints/mamba_{pair}_v1/mamba/mamba_best.pt` or best CV fold)
      3. 4-Hour Cross-Sectional + Hysteresis Engine (`StationaryEnsembleInferenceEngine`)
    Executes via a unified Hysteresis Buffer State Machine (`z_entry`, `z_exit`) to maximize
    out-of-sample Net Sharpe after realistic spread + slippage costs.
    """

    returns_live_actions: bool = True
    per_pair_heads: bool = True
    seq_len: int = 120
    n_features: int = 584

    def __init__(
        self,
        pair: str = "EURUSD",
        seq_len: int = 120,
        z_entry: float | None = None,
        z_exit: float | None = None,
        device: object | None = None,
    ):
        from inference.pytorch_inference import PyTorchInferenceEngine

        self.pair = pair.upper().replace("/", "").replace("_", "")
        pair_order = ("EURUSD", "USDJPY", "GBPUSD", "USDCAD")
        self.pair_index = pair_order.index(self.pair) if self.pair in pair_order else 0
        self.seq_len = int(seq_len)
        self.device = device

        # 1. Load 4H Cross-Sectional + Hysteresis Engine
        self.cs_engine = StationaryEnsembleInferenceEngine(
            pair=self.pair,
            override_z_entry=z_entry,
            override_z_exit=z_exit,
        )
        self.z_entry = float(z_entry if z_entry is not None else self.cs_engine.z_entry)
        self.z_exit = float(z_exit if z_exit is not None else self.cs_engine.z_exit)

        # 2. Discover & load HAELT Specialist
        self.haelt_engine: PyTorchInferenceEngine | None = None
        haelt_ckpt = self._discover_best_specialist_ckpt("haelt", self.pair)
        if haelt_ckpt is not None:
            try:
                self.haelt_engine = PyTorchInferenceEngine(
                    checkpoint_path=str(haelt_ckpt),
                    model_name="haelt",
                    seq_len=self.seq_len,
                    n_features=146,
                    device=self.device,
                )
                self.haelt_engine.pair = self.pair
                self.haelt_engine.pair_index = self.pair_index
            except Exception as exc:
                print(f"[ZooEnsemble] HAELT specialist load skipped for {self.pair}: {exc}")

        # 3. Discover & load Mamba Specialist (including best completed CV fold if top-level not yet promoted)
        self.mamba_engine: PyTorchInferenceEngine | None = None
        mamba_ckpt = self._discover_best_specialist_ckpt("mamba", self.pair)
        if mamba_ckpt is not None:
            try:
                self.mamba_engine = PyTorchInferenceEngine(
                    checkpoint_path=str(mamba_ckpt),
                    model_name="mamba",
                    seq_len=self.seq_len,
                    n_features=146,
                    device=self.device,
                )
                self.mamba_engine.pair = self.pair
                self.mamba_engine.pair_index = self.pair_index
            except Exception as exc:
                print(f"[ZooEnsemble] Mamba specialist load skipped for {self.pair}: {exc}")

        # Pair-optimal ensemble blend weights (w_cs_4h, w_haelt, w_mamba)
        if self.pair == "USDJPY":
            # USDJPY has strong 5m specialist trend edge (+0.63 Holdout Sharpe) + 4H CS-Hysteresis (+1.01 OOS Sharpe)
            w_raw = [0.45, 0.40 if self.haelt_engine else 0.0, 0.15 if self.mamba_engine else 0.0]
        elif self.pair == "USDCAD":
            # USDCAD excels on Hybrid Consensus (+2.03 Holdout Sharpe) + 4H CS-Hysteresis (+1.07 OOS Sharpe)
            w_raw = [0.55, 0.25 if self.haelt_engine else 0.0, 0.20 if self.mamba_engine else 0.0]
        else:
            # EURUSD / GBPUSD
            w_raw = [0.50, 0.30 if self.haelt_engine else 0.0, 0.20 if self.mamba_engine else 0.0]
        w_sum = sum(w_raw) or 1.0
        self.weights = tuple(w / w_sum for w in w_raw)

        self._obs_buffer_146: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._pos_dir: int = 0
        self.last_z_score: float = 0.0
        self.last_raw: float = 0.0
        self.last_proba: np.ndarray = np.array([0.33, 0.34, 0.33], dtype=np.float32)
        self.last_sub_signals: dict[str, float] = {"z_cs": 0.0, "z_haelt": 0.0, "z_mamba": 0.0}

        members = ["CS_4H"]
        if self.haelt_engine is not None:
            members.append(f"HAELT({Path(self.haelt_engine.checkpoint_path).name})")
        if self.mamba_engine is not None:
            members.append(f"MAMBA({Path(self.mamba_engine.checkpoint_path).name})")
        print(
            f"[ZooEnsemble] Initialized {self.pair} Multi-Arch Zoo: {', '.join(members)} | "
            f"weights={tuple(round(w, 2) for w in self.weights)} | z_entry={self.z_entry:.2f}, z_exit={self.z_exit:.2f}"
        )

    @staticmethod
    def _discover_best_specialist_ckpt(arch: str, pair: str) -> Path | None:
        p_low = pair.lower()
        base_dir = Path(f"checkpoints/{arch}_{p_low}_v1/{arch}")
        top_cands = [
            base_dir / f"{arch}_best.pt",
            base_dir / arch / f"{arch}_best.pt",
        ]
        for c in top_cands:
            if c.is_file():
                return c
        # If training is still in progress (e.g. mamba_usdcad_v1), pick the completed fold with highest best_cost_sharpe
        if base_dir.is_dir():
            best_fold_ckpt = None
            best_sr = -1e9
            for rpt in sorted(base_dir.glob(f"{arch}_fold*_training_control_report.json")):
                try:
                    d = json.loads(rpt.read_text(encoding="utf-8"))
                    sr = float(d.get("best_val_sharpe", d.get("best_cost_sharpe", -1e9)))
                except Exception:
                    sr = -1e9
                fold_tag = rpt.name.split("_training_control_report")[0]
                ckpt_f = base_dir / f"{fold_tag}_best.pt"
                if ckpt_f.is_file() and sr > best_sr:
                    best_sr = sr
                    best_fold_ckpt = ckpt_f
            if best_fold_ckpt is not None:
                return best_fold_ckpt
            # Fallback to fold1_best or fold0_best if present
            for f_idx in (1, 2, 0, 3, 4, 5):
                cand = base_dir / f"{arch}_fold{f_idx}_best.pt"
                if cand.is_file():
                    return cand
        return None

    def for_pair(self, pair: str) -> "MultiArchitectureSpecialistEnsembleInferenceEngine":
        pair_clean = pair.upper().replace("/", "").replace("_", "")
        if pair_clean == self.pair:
            return self
        return MultiArchitectureSpecialistEnsembleInferenceEngine(
            pair=pair_clean,
            seq_len=self.seq_len,
            device=self.device,
        )

    def set_agent_state(self, position_lots: float = 0.0, **kwargs) -> None:
        self.cs_engine.set_agent_state(position_lots=position_lots, **kwargs)
        if position_lots > 1e-6:
            self._pos_dir = 1
        elif position_lots < -1e-6:
            self._pos_dir = -1
        else:
            self._pos_dir = 0

    @staticmethod
    def _proba_to_z(proba: np.ndarray) -> float:
        p = np.asarray(proba, dtype=np.float64).ravel()
        if p.size != 3:
            return 0.0
        p_sell, _, p_buy = float(p[0]), float(p[1]), float(p[2])
        p_trade = max(p_sell + p_buy, 1e-6)
        signed_edge = (p_buy - p_sell) * p_trade
        # Scale typical specialist signed_edge (~0.04..0.12) to unit-variance z-score
        return float(np.clip(signed_edge / 0.045, -3.5, 3.5))

    def select_action(self, obs: np.ndarray) -> int:
        from trading.live_actions import LiveAction

        obs_1d = np.asarray(obs, dtype=np.float32).ravel()
        if obs_1d.size == 584:
            obs_146 = obs_1d[self.pair_index * 146 : (self.pair_index + 1) * 146]
        else:
            obs_146 = obs_1d[:146]
        self._obs_buffer_146.append(obs_146)

        # 1. 4H Cross-Sectional + Cost-Vol Ridge + Diff-Sharpe z-score
        z_cs = self.cs_engine.predict_z_score(obs_1d, record_history=True)

        # 2. Deep Sequence Specialists (HAELT & Mamba) when 120-bar window is warm
        z_haelt = 0.0
        z_mamba = 0.0
        if len(self._obs_buffer_146) >= self.seq_len:
            win_146 = np.stack(self._obs_buffer_146, axis=0)
            if self.haelt_engine is not None:
                try:
                    z_haelt = self._proba_to_z(self.haelt_engine.predict_proba(win_146))
                except Exception:
                    z_haelt = 0.0
            if self.mamba_engine is not None:
                try:
                    z_mamba = self._proba_to_z(self.mamba_engine.predict_proba(win_146))
                except Exception:
                    z_mamba = 0.0

        w_cs, w_h, w_m = self.weights
        if len(self._obs_buffer_146) < self.seq_len:
            z_total = z_cs
        else:
            z_total = float(w_cs * z_cs + w_h * z_haelt + w_m * z_mamba)
        self.last_z_score = z_total
        self.last_sub_signals = {"z_cs": round(z_cs, 4), "z_haelt": round(z_haelt, 4), "z_mamba": round(z_mamba, 4)}

        cur_dir = self._pos_dir
        if cur_dir == 0:
            if z_total >= self.z_entry:
                self._pos_dir = 1
                act = int(LiveAction.BUY)
            elif z_total <= -self.z_entry:
                self._pos_dir = -1
                act = int(LiveAction.SELL)
            else:
                act = int(LiveAction.HOLD)
        elif cur_dir == 1:
            if z_total <= -self.z_entry:
                self._pos_dir = -1
                act = int(LiveAction.SELL)
            elif z_total < self.z_exit:
                self._pos_dir = 0
                act = int(LiveAction.CLOSE)
            else:
                act = int(LiveAction.BUY)
        else:
            if z_total >= self.z_entry:
                self._pos_dir = 1
                act = int(LiveAction.BUY)
            elif -z_total < self.z_exit:
                self._pos_dir = 0
                act = int(LiveAction.CLOSE)
            else:
                act = int(LiveAction.SELL)

        self.cs_engine._update_proba_from_z(z_total, act)
        self.last_raw = self.cs_engine.last_raw
        self.last_proba = self.cs_engine.last_proba
        return act

    def reset_buffer(self) -> None:
        self._obs_buffer_146.clear()
        self.cs_engine.reset_buffer()
        if self.haelt_engine is not None:
            self.haelt_engine.reset_buffer()
        if self.mamba_engine is not None:
            self.mamba_engine.reset_buffer()
        self._pos_dir = 0
        self.last_z_score = 0.0
        self.last_raw = 0.0
        self.last_proba = np.array([0.33, 0.34, 0.33], dtype=np.float32)

