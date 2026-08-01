"""
scripts/train_rl.py
===================
Trains a Reinforcement Learning execution policy on top of a frozen
supervised model's signal. Includes support for Ensemble risk filtering.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import PATHS
from models.rl_agents import ForexTradingEnv, PPOAgent, DQNAgent, train_agent
from training.train_gpu import ZarrStreamDataset
from scripts.train_ensemble_meta import load_base_model, resolve_checkpoint
from models.ensemble import EnsembleMetaLearner
from models.rl_advanced import CurriculumScheduler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL Execution Layer")
    parser.add_argument("--model-name", type=str, default="haelt", help="Supervised model name (e.g. haelt, mamba, ensemble)")
    parser.add_argument("--ensemble-bases", nargs="+", default=["haelt", "mamba", "gnn"], help="Bases if model is ensemble")
    parser.add_argument("--cache", type=str, required=True, help="Zarr cache path")
    parser.add_argument("--episodes", type=int, default=100, help="Number of RL episodes")
    parser.add_argument("--agent", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--checkpoint-dir", type=str, default=str(PATHS["checkpoints"]))
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for signal inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def extract_signals(model, dataloader, device, is_ensemble=False):
    model.eval()
    signals = []
    disagreements = []
    logging.info("Extracting supervised signals...")
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device, non_blocking=True)
            if is_ensemble:
                out, dis = model.predict_with_disagreement(x)
                signals.append(out.cpu().numpy())
                disagreements.append(dis.cpu().numpy())
            else:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                signals.append(out.cpu().numpy())
                # For non-ensemble, disagreement is 0
                disagreements.append(np.zeros(out.shape[0]))
                
    signals_np = np.concatenate(signals, axis=0)
    disagreements_np = np.concatenate(disagreements, axis=0)
    
    # Return features block: signals as feature 0, disagreement as feature 1 (if ensemble)
    if is_ensemble:
        # Reshape to (N, 2)
        return np.stack([signals_np, disagreements_np], axis=-1)
    else:
        # Reshape to (N, 1)
        return signals_np.reshape(-1, 1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    cache_path = Path(args.cache)
    if not cache_path.exists():
        logging.error(f"Zarr cache not found: {cache_path}")
        sys.exit(1)
        
    z = zarr.open(str(cache_path), mode='r')
    expected_keys = ['X', 'close', 'atr', 'spread']
    for k in expected_keys:
        if k not in z:
            logging.error(f"Zarr cache missing required array: {k}")
            sys.exit(1)
            
    num_samples = z['X'].shape[0]
    seq_len = z['X'].shape[1]
    num_features = z['X'].shape[2]
    
    logging.info(f"Loaded Zarr: {num_samples} samples, seq_len={seq_len}, features={num_features}")
    
    ckpt_dir = Path(args.checkpoint_dir)
    is_ensemble = args.model_name.lower() == "ensemble"
    
    if is_ensemble:
        logging.info("Loading Ensemble Meta Learner...")
        bases = []
        names = []
        for name in args.ensemble_bases:
            ckpt = resolve_checkpoint(name, ckpt_dir)
            if ckpt:
                b_model, _ = load_base_model(name, ckpt, num_features, seq_len, device)
                bases.append(b_model)
                names.append(name)
        
        model = EnsembleMetaLearner(bases, context_dim=32, hidden=64, base_names=names).to(device)
        ens_ckpt = resolve_checkpoint("ensemble_meta", ckpt_dir)
        if ens_ckpt:
            model.load_state_dict(torch.load(ens_ckpt, map_location=device), strict=False)
            logging.info(f"Loaded ensemble weights from {ens_ckpt}")
        else:
            logging.warning("No trained ensemble meta weights found. Using default initialization.")
    else:
        ckpt_path = resolve_checkpoint(args.model_name, ckpt_dir)
        if not ckpt_path:
            raise FileNotFoundError(f"Could not find checkpoint for {args.model_name}")
        logging.info(f"Loading base model {args.model_name} from {ckpt_path}")
        model, _ = load_base_model(args.model_name, ckpt_path, num_features, seq_len, device)
        
    model.eval()
    
    dataset = ZarrStreamDataset(str(cache_path), np.arange(num_samples), shuffle_chunks=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    
    signals = extract_signals(model, dataloader, device, is_ensemble=is_ensemble)
    logging.info(f"Extracted signals shape: {signals.shape}")
    
    # Extract market arrays
    prices = np.array(z['close'])
    atr = np.array(z['atr'])
    spreads = np.array(z['spread'])
    
    # Extract raw X for state enrichment
    raw_x = np.array(z['X'])
    
    min_len = min(len(signals), len(prices), len(raw_x))
    signals = signals[:min_len]
    prices = prices[:min_len]
    atr = atr[:min_len]
    spreads = spreads[:min_len]
    raw_x = raw_x[:min_len, -1, :]  # Take the last timestep's features
    
    # Enrich the RL state: supervised signal + raw market features
    enriched_features = np.concatenate([signals, raw_x], axis=1)
    
    logging.info(f"Initializing ForexTradingEnv with {min_len} bars...")
    env = ForexTradingEnv(
        features=enriched_features,
        prices=prices,
        atr=atr,
        spreads=spreads,
        initial_equity=10_000.0,
        lot_size=10_000.0,
        max_lots=3.0,
        commission_per_lot=3.5,
        slippage_pips=0.5,
        random_reset=True,
        episode_len=min(5000, min_len - 1)
    )
    
    if args.agent == "ppo":
        agent = PPOAgent(obs_size=env.obs_size, n_actions=env.n_actions, device=device)
    else:
        agent = DQNAgent(obs_size=env.obs_size, n_actions=env.n_actions, device=device)
        
    curriculum = CurriculumScheduler(total_episodes=args.episodes)
        
    logging.info(f"Training {args.agent.upper()} agent for {args.episodes} episodes...")
    returns = train_agent(agent, env, n_episodes=args.episodes, agent_type=args.agent, curriculum=curriculum)
    
    final_summary = env.summary()
    report = {
        "supervised_model": args.model_name,
        "is_ensemble": is_ensemble,
        "agent": args.agent,
        "episodes": args.episodes,
        "train_return_pct": float(np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)),
        "max_drawdown_pct": float(final_summary["max_dd_pct"]),
        "n_trades": int(final_summary["n_trades"]),
        "sharpe": float(final_summary["sharpe"])
    }
    
    out_dir = ckpt_dir / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "rl_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    rl_ckpt_path = out_dir / "rl_best.pt"
    if args.agent == "ppo":
        torch.save(agent.net.state_dict(), rl_ckpt_path)
    else:
        torch.save(agent.policy_net.state_dict(), rl_ckpt_path)
        
    logging.info(f"Saved RL report to {report_path}")
    logging.info(f"Saved RL checkpoint to {rl_ckpt_path}")


if __name__ == "__main__":
    main()
