"""
features package
================
Canonical feature engineering modules.
"""

from importlib import import_module

# Explicit exports for clarity
__all__ = [
    # feature_engineering_pl
    "sanitize_frame",
    "order_flow_imbalance",
    "order_book_imbalance_proxy",
    "trade_arrival_rate",
    "average_true_range",
    "rolling_volatility",
    "bollinger_bands",
    "rsi",
    "macd",
    "lag_returns",
    "stochastic_oscillator",
    "williams_r",
    "cci",
    "vwap_bands",
    "volume_weighted_momentum",
    "regime_gated_features",
    "interaction_features",
    "FeatureEngineer",
    "CrossAssetFeatures",
    "RegimeGateClassifier",
    # quality
    "FeatureQuality",
    "FeatureQualityReport",
    "compute_quality_report",
    "print_quality_report",
    # multipair
    "compute_vpin",
    "compute_realized_moments",
    "compute_asia_london_gap",
    "compute_multipair_features",
    "build_multipair_features",
    # alternative_data
    "fetch_binance_funding_rates",
    "fetch_binance_current_funding",
    "fetch_whale_alerts_recent",
    "fetch_cryptopanic_news",
    "fetch_fear_greed_index",
    "fetch_crypto_alt_data_for_pair",
    "build_alt_features_for_fx",
    "CRYPTO_FX_MAP",
]


_EXPORT_MODULES = {
    # feature_engineering_pl
    "sanitize_frame": "features.feature_engineering_pl",
    "order_flow_imbalance": "features.feature_engineering_pl",
    "order_book_imbalance_proxy": "features.feature_engineering_pl",
    "trade_arrival_rate": "features.feature_engineering_pl",
    "average_true_range": "features.feature_engineering_pl",
    "rolling_volatility": "features.feature_engineering_pl",
    "bollinger_bands": "features.feature_engineering_pl",
    "rsi": "features.feature_engineering_pl",
    "macd": "features.feature_engineering_pl",
    "lag_returns": "features.feature_engineering_pl",
    "stochastic_oscillator": "features.feature_engineering_pl",
    "williams_r": "features.feature_engineering_pl",
    "cci": "features.feature_engineering_pl",
    "vwap_bands": "features.feature_engineering_pl",
    "volume_weighted_momentum": "features.feature_engineering_pl",
    "regime_gated_features": "features.feature_engineering_pl",
    "interaction_features": "features.feature_engineering_pl",
    "FeatureEngineer": "features.feature_engineering_pl",
    "CrossAssetFeatures": "features.feature_engineering_pl",
    "RegimeGateClassifier": "features.feature_engineering_pl",
    # quality
    "FeatureQuality": "features.quality",
    "FeatureQualityReport": "features.quality",
    "compute_quality_report": "features.quality",
    "print_quality_report": "features.quality",
    # multipair
    "compute_vpin": "features.multipair",
    "compute_realized_moments": "features.multipair",
    "compute_asia_london_gap": "features.multipair",
    "compute_multipair_features": "features.multipair",
    "build_multipair_features": "features.multipair",
    # alternative_data
    "fetch_binance_funding_rates": "features.alternative_data",
    "fetch_binance_current_funding": "features.alternative_data",
    "fetch_whale_alerts_recent": "features.alternative_data",
    "fetch_cryptopanic_news": "features.alternative_data",
    "fetch_fear_greed_index": "features.alternative_data",
    "fetch_crypto_alt_data_for_pair": "features.alternative_data",
    "build_alt_features_for_fx": "features.alternative_data",
    "CRYPTO_FX_MAP": "features.alternative_data",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'features' has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
