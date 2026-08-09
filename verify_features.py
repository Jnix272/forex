import os

checks = {
    '1. OnlineHardExampleMiner': '_online_miner.get_oversampled_indices' in open('training/supervised_loop.py', encoding='utf-8').read(),
    '2. Label Smoothing Config': 'label_smoothing: 0.05' in open('config/run.yaml', encoding='utf-8').read(),
    '3. MixUp Augmentation': 'MixupBatch' in open('training/train_gpu.py', encoding='utf-8').read(),
    '4. VolatilityStratifiedSampler': 'VolatilityStratifiedSampler' in open('training/train_gpu.py', encoding='utf-8').read(),
    '5. Correlated Pair Dropout': 'corr_dropout_p' in open('models/architectures.py', encoding='utf-8').read(),
    '6. Regime Early Stopping': 'RegimeTierTracker' in open('training/train_gpu.py', encoding='utf-8').read(),
    '7. VPIN': 'def compute_vpin' in open('features/advanced_features.py', encoding='utf-8').read(),
    '8. Skewness/Kurtosis': 'def compute_realized_moments' in open('features/advanced_features.py', encoding='utf-8').read(),
    '9. Asia-London Gap': 'def compute_asia_london_gap' in open('features/advanced_features.py', encoding='utf-8').read(),
    '10. Attention Hooks': os.path.exists('monitoring/attention_logger.py'),
    '11. Conformal Prediction': 'compute_conformal_coverage' in open('validation/promotion_gate.py', encoding='utf-8').read()
}

for k, v in checks.items():
    print(f'{k}: {"YES" if v else "NO"}')
