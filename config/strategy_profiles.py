"""Strategy profiles for scalping and normal trading modes.

These values intentionally override the LABELING defaults in config/settings.py
for inference-time behavior (tighter stops, faster exits for scalping; wider
parameters for normal/swing).
"""

STRATEGY_PROFILES = {
    "scalping": {
        "bar_freq": "5min",
        "seq_len": 120,  # match config/run.yaml training.seq_len
        "lookahead_bars": 30,  # match strategy / LABELING defaults
        "profit_target_atr": 1.2,  # match strategy / LABELING defaults
        "stop_loss_atr": 0.8,
        "execution_delay_bars": 1,
        "max_spread_pips": 2.5,
        "guard_min_confidence": 0.45,
        "checkpoint_dir": "checkpoints",
    },
    "normal": {
        "bar_freq": "1h",
        "seq_len": 60,
        "lookahead_bars": 24,
        "profit_target_atr": 2.5,
        "stop_loss_atr": 1.5,
        "execution_delay_bars": 1,
        "max_spread_pips": 4.0,
        "guard_min_confidence": 0.55,
        "checkpoint_dir": "checkpoints/normal",
    },
}


def strategy_profile(name: str | None = None) -> dict:
    """Return a copy of a named strategy profile, defaulting to scalping."""
    key = str(name or "scalping").strip().lower()
    if key not in STRATEGY_PROFILES:
        valid = ", ".join(sorted(STRATEGY_PROFILES))
        raise ValueError(f"Unknown strategy profile '{name}'. Valid profiles: {valid}")
    return dict(STRATEGY_PROFILES[key])
