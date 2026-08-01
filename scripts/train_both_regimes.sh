#!/bin/bash
# scripts/train_both_regimes.sh
# -----------------------------------------------------------------------------
# Automates the 2-regime training process for the best 3 models 
# (HAELT, Mamba, CatBoost) sequentially without requiring manual intervention.
# -----------------------------------------------------------------------------

set -e # Exit immediately if any command fails

echo "====================================================================="
echo " Starting Regime 1 (2010-2018) Training"
echo " Configuration: 5 Walk-Forward Folds, 60 Epochs, Top 3 Models"
echo "====================================================================="

python training/train_gpu.py \
    --data-source dukascopy \
    --data-start 2010-01-01 \
    --data-end 2018-12-31 \
    --walk-forward-folds 5 \
    --epochs 60 \
    --config config/run_ubuntu.yaml \
    --all-models

echo "====================================================================="
echo " Regime 1 Complete! Starting Regime 2 (2019-2026) Training"
echo " Configuration: 3 Walk-Forward Folds, 60 Epochs, Top 3 Models"
echo "====================================================================="

python training/train_gpu.py \
    --data-source dukascopy \
    --data-start 2019-01-01 \
    --data-end 2026-12-31 \
    --walk-forward-folds 3 \
    --epochs 60 \
    --config config/run_ubuntu.yaml \
    --all-models

echo "====================================================================="
echo " ALL REGIMES AND FOLDS HAVE COMPLETED TRAINING!"
echo " The models are now ready for the TemporalFoldEnsemble."
echo "====================================================================="
