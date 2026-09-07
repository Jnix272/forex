import torch
from training.post_train import run_ensemble_meta
from training.config_loader import load_run_config
from training.gpu_device import setup_device
import argparse

args = load_run_config('config/run.yaml')
args.train_ensemble = True
device = setup_device(args)
# Need cache_path and n_features.
cache_path = 'd:/forex-main/data/zarr_cache/dataset_4pair_2015_2025.zarr'
n_features = 584 # Based on haelt summary
run_ensemble_meta(cache_path, n_features, args, device)
