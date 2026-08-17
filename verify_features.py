import os

checks = {
    "1. OnlineHardExampleMiner": "_online_miner.get_oversampled_indices"
    in __import__("pathlib").Path("training/supervised_loop.py").read_text(encoding="utf-8"),
    "2. Label Smoothing Config": "label_smoothing: 0.05" in __import__("pathlib").Path("config/run.yaml").read_text(encoding="utf-8"),
    "3. MixUp Augmentation": "MixupBatch" in __import__("pathlib").Path("training/train_gpu.py").read_text(encoding="utf-8"),
    "4. VolatilityStratifiedSampler": "VolatilityStratifiedSampler"
    in __import__("pathlib").Path("training/train_gpu.py").read_text(encoding="utf-8"),
    "5. Correlated Pair Dropout": "corr_dropout_p" in __import__("pathlib").Path("models/architectures.py").read_text(encoding="utf-8"),
    "6. Regime Early Stopping": "RegimeTierTracker" in __import__("pathlib").Path("training/train_gpu.py").read_text(encoding="utf-8"),
    "7. VPIN": "def compute_vpin" in __import__("pathlib").Path("features/advanced_features.py").read_text(encoding="utf-8"),
    "8. Skewness/Kurtosis": "def compute_realized_moments"
    in __import__("pathlib").Path("features/advanced_features.py").read_text(encoding="utf-8"),
    "9. Asia-London Gap": "def compute_asia_london_gap"
    in __import__("pathlib").Path("features/advanced_features.py").read_text(encoding="utf-8"),
    "10. Attention Hooks": os.path.exists("monitoring/attention_logger.py"),
    "11. Conformal Prediction": "compute_conformal_coverage"
    in __import__("pathlib").Path("validation/promotion_gate.py").read_text(encoding="utf-8"),
}

for k, v in checks.items():
    print(f"{k}: {'YES' if v else 'NO'}")
