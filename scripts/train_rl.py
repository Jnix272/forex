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
from models.ensemble import EnsembleMetaLearner
from models.rl_advanced import CurriculumScheduler, RLEnsemble
from models.rl_agents import DQNAgent, ForexTradingEnv, PPOAgent, evaluate_agent, train_agent
from scripts.train_ensemble_meta import load_base_model, resolve_checkpoint
from training.gpu_datasets import ZarrStreamDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL Execution Layer")
    parser.add_argument(
        "--model-name", type=str, default="haelt", help="Supervised model name (e.g. haelt, mamba, ensemble, or none)"
    )
    parser.add_argument(
        "--ensemble-bases", nargs="+", default=["haelt", "mamba", "gnn"], help="Bases if model is ensemble"
    )
    parser.add_argument("--cache", type=str, required=True, help="Zarr cache path")
    parser.add_argument("--episodes", type=int, default=100, help="Number of RL episodes")
    parser.add_argument("--agent", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--checkpoint-dir", type=str, default=str(PATHS["checkpoints"]))
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for signal inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Multi-Agent Flags
    parser.add_argument("--multi-agent", action="store_true", help="Train an ensemble of multiple RL agents")
    parser.add_argument("--num-agents", type=int, default=3, help="Number of RL agents in ensemble")
    parser.add_argument(
        "--ensemble-agents",
        type=str,
        default=None,
        help="Comma-separated agent types (e.g. 'ppo,ppo,ppo' or 'ppo,dqn')",
    )
    parser.add_argument(
        "--consensus-mode",
        type=str,
        choices=["soft_vote", "majority", "conservative"],
        default="soft_vote",
        help="Consensus mode for RLEnsemble ('soft_vote', 'majority', 'conservative')",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed for RL training")
    # Recurrent / Memory Flags
    parser.add_argument("--use-lstm", action="store_true", help="Use LSTM recurrent memory in PPO ActorCritic")
    parser.add_argument("--hist-len", type=int, default=32, help="Observation history length for LSTM")
    parser.add_argument("--lstm-hidden", type=int, default=128, help="Hidden size for LSTM layer")
    return parser.parse_args()


def extract_last_features_streaming(z_group, min_len: int) -> np.ndarray:
    """Extract (N, F) features from z['X'] taking the last timestep if 3D.
    Uses chunked streaming to prevent allocating 48+ GiB of 3D array in RAM.
    """
    arr_x = z_group["X"]
    is_3d = (arr_x.ndim == 3)
    num_samples = min(min_len, arr_x.shape[0])
    num_features = arr_x.shape[-1]

    logging.info(f"Extracting last-timestep features ({num_samples}, {num_features}) from Zarr...")
    out = np.empty((num_samples, num_features), dtype=np.float32)
    chunk_size = 10000
    for i in range(0, num_samples, chunk_size):
        end = min(i + chunk_size, num_samples)
        if is_3d:
            out[i:end] = arr_x[i:end, -1, :]
        else:
            out[i:end] = arr_x[i:end]
    return out


def extract_signals_and_features_from_zarr(
    model, z_x, device, is_ensemble=False, batch_size=232, max_samples=None
):
    """
    Extract supervised signals and last-timestep features directly from Zarr X array.
    Reads chunk-by-chunk in strict chronological order with zero DataLoader tuple-overhead.
    """
    if model is not None:
        model.eval()

    num_samples = z_x.shape[0] if max_samples is None else min(z_x.shape[0], max_samples)
    num_features = z_x.shape[2] if z_x.ndim == 3 else z_x.shape[1]

    # Pre-allocate output arrays in contiguous RAM (~434 MB total)
    n_sig_cols = 2 if is_ensemble else 1
    signals_out = np.zeros((num_samples, n_sig_cols), dtype=np.float32)
    features_out = np.zeros((num_samples, num_features), dtype=np.float32)

    # Align batch step with on-disk chunk boundary to eliminate zarr cache thrashing
    chunk_rows = getattr(z_x, "chunks", None)
    step = int(chunk_rows[0]) if chunk_rows is not None and len(chunk_rows) > 0 else batch_size
    step = max(32, min(step, 512))

    logging.info(
        f"Extracting supervised signals and features chunk-by-chunk ({num_samples} samples, step={step})..."
    )

    with torch.no_grad():
        for start in range(0, num_samples, step):
            end = min(start + step, num_samples)
            # 1. Read contiguous slice directly from Zarr
            chunk = np.array(z_x[start:end], dtype=np.float32)
            np.nan_to_num(chunk, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

            # 2. Store last-timestep features
            features_out[start:end] = chunk[:, -1, :] if chunk.ndim == 3 else chunk

            # 3. Model inference on GPU/CPU
            if model is not None:
                x_dev = torch.from_numpy(chunk).to(device)
                if is_ensemble:
                    out, dis = model.predict_with_disagreement(x_dev)
                    signals_out[start:end, 0] = out.cpu().numpy().flatten()
                    signals_out[start:end, 1] = dis.cpu().numpy().flatten()
                    del dis
                else:
                    out = model(x_dev)
                    if isinstance(out, tuple):
                        out = out[1] if len(out) >= 2 else out[0]
                    signals_out[start:end, 0] = out.cpu().numpy().flatten()
                del x_dev, out

            del chunk
            chunk_idx = start // step
            if chunk_idx % 50 == 0 or end == num_samples:
                if getattr(device, "type", str(device)) == "cuda":
                    torch.cuda.empty_cache()
                logging.info(f"  Signal extraction progress: {end:,}/{num_samples:,} ({end*100/num_samples:.1f}%)")

    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.empty_cache()

    logging.info(f"Completed extraction: signals shape {signals_out.shape}, features shape {features_out.shape}")
    return signals_out, features_out


def extract_signals_and_features(model, dataloader_or_arr, device, is_ensemble=False):
    """Unified extractor accepting either a Zarr/Numpy array or a PyTorch DataLoader."""
    if hasattr(dataloader_or_arr, "shape") and not hasattr(dataloader_or_arr, "dataset"):
        return extract_signals_and_features_from_zarr(
            model, dataloader_or_arr, device, is_ensemble=is_ensemble
        )

    dataloader = dataloader_or_arr
    if model is not None:
        model.eval()
    signals = []
    disagreements = []
    last_features = []
    logging.info("Extracting supervised signals and features via DataLoader...")
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            # Extract last-timestep feature on CPU: shape (B, F)
            last_feat = x[:, -1, :].cpu().numpy() if x.ndim == 3 else x.cpu().numpy()
            last_features.append(last_feat)

            if model is None:
                continue

            x_dev = x.to(device, non_blocking=True)
            if is_ensemble:
                out, dis = model.predict_with_disagreement(x_dev)
                signals.append(out.cpu().numpy())
                disagreements.append(dis.cpu().numpy())
                del dis
            else:
                out = model(x_dev)
                if isinstance(out, tuple):
                    out = out[1] if len(out) >= 2 else out[0]
                signals.append(out.cpu().numpy())
                disagreements.append(np.zeros(out.shape[0]))

            del x_dev, out
            if getattr(device, "type", str(device)) == "cuda" and (i + 1) % 25 == 0:
                torch.cuda.empty_cache()

    raw_x_np = np.concatenate(last_features, axis=0)
    if model is None:
        signals_out = np.zeros((len(raw_x_np), 1), dtype=np.float32)
        return signals_out, raw_x_np

    signals_np = np.concatenate(signals, axis=0)
    disagreements_np = np.concatenate(disagreements, axis=0)

    if is_ensemble:
        signals_out = np.stack([signals_np, disagreements_np], axis=-1)
    else:
        signals_out = signals_np.reshape(-1, 1)

    return signals_out, raw_x_np


def extract_signals(model, dataloader, device, is_ensemble=False):
    """Backward-compatible helper returning only signals."""
    sig, _ = extract_signals_and_features(model, dataloader, device, is_ensemble=is_ensemble)
    return sig


def main():
    args = parse_args()
    device = torch.device(args.device)

    cache_path = Path(args.cache)
    if not cache_path.exists():
        logging.error(f"Zarr cache not found: {cache_path}")
        sys.exit(1)

    z = zarr.open(str(cache_path), mode="r")
    expected_keys = ["X", "close", "atr", "spread"]
    for k in expected_keys:
        if k not in z:
            logging.error(f"Zarr cache missing required array: {k}")
            sys.exit(1)

    num_samples = z["X"].shape[0]
    seq_len = z["X"].shape[1]
    num_features = z["X"].shape[2]

    logging.info(f"Loaded Zarr: {num_samples} samples, seq_len={seq_len}, features={num_features}")

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = ckpt_dir / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    is_dummy = args.model_name.lower() in ("none", "dummy", "identity")
    is_ensemble = args.model_name.lower() == "ensemble"

    sig_cache_file = out_dir / "signals_cache.npy"
    feat_cache_file = ckpt_dir / "ensemble" / "features_last_step.npy"

    signals = None
    raw_x = None

    # Cached signals/features are only valid for the exact supervised
    # checkpoints that produced them.  Invalidate them when any relevant
    # checkpoint is newer; shape-only validation was allowing stale signals
    # after retraining.
    relevant_ckpts = []
    if is_ensemble:
        relevant_ckpts.append(ckpt_dir / "ensemble" / "ensemble_meta_best.pt")
        relevant_ckpts.extend(resolve_checkpoint(name, ckpt_dir) for name in args.ensemble_bases)
    elif not is_dummy:
        resolved = resolve_checkpoint(args.model_name, ckpt_dir)
        if resolved:
            relevant_ckpts.append(resolved)
    relevant_ckpts = [Path(p) for p in relevant_ckpts if p and Path(p).exists()]
    cache_is_stale = bool(relevant_ckpts) and any(
        p.stat().st_mtime > cache.stat().st_mtime
        for p in relevant_ckpts
        for cache in (sig_cache_file, feat_cache_file)
        if cache.exists()
    )
    if cache_is_stale:
        logging.info("Invalidating supervised signal/feature caches: checkpoint is newer")

    if sig_cache_file.exists() and not cache_is_stale:
        try:
            cached_sig = np.load(sig_cache_file)
            if len(cached_sig) == num_samples:
                signals = cached_sig
                logging.info(f"Loaded cached supervised signals from {sig_cache_file} {signals.shape}")
        except Exception as e:
            logging.warning(f"Could not load cached signals: {e}")

    if feat_cache_file.exists() and not cache_is_stale:
        try:
            cached_feat = np.load(feat_cache_file)
            if len(cached_feat) == num_samples:
                raw_x = cached_feat
                logging.info(f"Loaded cached last-step features from {feat_cache_file} {raw_x.shape}")
        except Exception as e:
            logging.warning(f"Could not load cached features: {e}")

    if signals is None or raw_x is None:
        if is_dummy:
            logging.info("Model name is 'none/dummy' - proceeding with zero supervised signal.")
            signals = np.zeros((num_samples, 1), dtype=np.float32)
            if raw_x is None:
                raw_x = extract_last_features_streaming(z, num_samples)
                if not is_dummy or str(ckpt_dir) != ".":
                    try:
                        feat_cache_file.parent.mkdir(parents=True, exist_ok=True)
                        np.save(feat_cache_file, raw_x)
                    except Exception:
                        pass
        elif is_ensemble:
            logging.info("Loading Ensemble Meta Learner...")
            bases = []
            names = []
            base_seq_lens = []
            for name in args.ensemble_bases:
                ckpt = resolve_checkpoint(name, ckpt_dir)
                if ckpt:
                    b_model, cfg = load_base_model(name, ckpt, num_features, seq_len, device)
                    bases.append(b_model)
                    names.append(name)
                    base_seq_lens.append(int(cfg.get("seq_len", seq_len)))

            model = EnsembleMetaLearner(
                bases, context_dim=32, hidden=64, base_names=names, base_seq_lens=base_seq_lens
            ).to(device)
            ens_ckpt = resolve_checkpoint("ensemble_meta", ckpt_dir) or (
                ckpt_dir / "ensemble" / "ensemble_meta_best.pt"
                if (ckpt_dir / "ensemble" / "ensemble_meta_best.pt").exists()
                else None
            )
            if ens_ckpt and Path(ens_ckpt).exists():
                model.load_state_dict(torch.load(ens_ckpt, map_location=device, weights_only=True), strict=False)
                logging.info(f"Loaded ensemble weights from {ens_ckpt}")
            else:
                logging.warning("No trained ensemble meta weights found. Using default initialization.")
            model.eval()
            extracted_signals, extracted_features = extract_signals_and_features_from_zarr(
                model, z["X"], device, is_ensemble=is_ensemble, batch_size=args.batch_size
            )
            if signals is None:
                signals = extracted_signals
                try:
                    np.save(sig_cache_file, signals)
                    logging.info(f"Cached supervised signals to {sig_cache_file}")
                except Exception as e:
                    logging.warning(f"Failed to cache signals: {e}")
            if raw_x is None:
                raw_x = extracted_features
                try:
                    feat_cache_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(feat_cache_file, raw_x)
                    logging.info(f"Cached last-step features to {feat_cache_file}")
                except Exception as e:
                    logging.warning(f"Failed to cache features: {e}")
        else:
            ckpt_path = resolve_checkpoint(args.model_name, ckpt_dir)
            if not ckpt_path:
                raise FileNotFoundError(f"Could not find checkpoint for {args.model_name}")
            logging.info(f"Loading base model {args.model_name} from {ckpt_path}")
            model, _ = load_base_model(args.model_name, ckpt_path, num_features, seq_len, device)
            model.eval()
            extracted_signals, extracted_features = extract_signals_and_features_from_zarr(
                model, z["X"], device, is_ensemble=is_ensemble, batch_size=args.batch_size
            )
            if signals is None:
                signals = extracted_signals
                try:
                    np.save(sig_cache_file, signals)
                    logging.info(f"Cached supervised signals to {sig_cache_file}")
                except Exception as e:
                    logging.warning(f"Failed to cache signals: {e}")
            if raw_x is None:
                raw_x = extracted_features
                try:
                    feat_cache_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(feat_cache_file, raw_x)
                    logging.info(f"Cached last-step features to {feat_cache_file}")
                except Exception as e:
                    logging.warning(f"Failed to cache features: {e}")

    logging.info(f"Supervised signals shape: {signals.shape}")
    logging.info(f"Enriched features shape: {raw_x.shape}")

    # Extract market arrays (1D arrays, negligible RAM)
    prices = np.array(z["close"])
    atr = np.array(z["atr"])
    spreads = np.array(z["spread"])

    min_len = min(len(signals), len(prices), len(raw_x))
    signals = signals[:min_len]
    prices = prices[:min_len]
    atr = atr[:min_len]
    spreads = spreads[:min_len]
    raw_x = raw_x[:min_len]

    # Enrich the RL state: supervised signal + raw market features
    enriched_features = np.concatenate([signals, raw_x], axis=1)

    # Keep RL training and evaluation strictly chronological.  The previous
    # implementation trained and evaluated random episodes from the same full
    # history, which made the reported RL metrics in-sample.
    train_end = max(2, int(min_len * 0.70))
    eval_start = min_len - max(2, int(min_len * 0.20))
    embargo = max(1, int(getattr(args, "hist_len", 32)))
    eval_start = max(train_end + embargo, eval_start)
    if eval_start >= min_len - 1:
        raise ValueError("RL dataset is too short for chronological train/evaluation split")
    logging.info(
        f"Initializing chronological RL environments: train=[0,{train_end}), "
        f"eval=[{eval_start},{min_len})"
    )
    env = ForexTradingEnv(
        features=enriched_features[:train_end],
        prices=prices[:train_end],
        atr=atr[:train_end],
        spreads=spreads[:train_end],
        initial_equity=10_000.0,
        lot_size=10_000.0,
        max_lots=3.0,
        commission_per_lot=3.5,
        slippage_pips=0.5,
        random_reset=True,
        episode_len=min(5000, train_end - 1),
    )
    eval_env = ForexTradingEnv(
        features=enriched_features[eval_start:],
        prices=prices[eval_start:],
        atr=atr[eval_start:],
        spreads=spreads[eval_start:],
        initial_equity=10_000.0,
        lot_size=10_000.0,
        max_lots=3.0,
        commission_per_lot=3.5,
        slippage_pips=0.5,
        random_reset=False,
        episode_len=eval_start and min(5000, min_len - eval_start - 1),
    )

    if not args.multi_agent:
        # ── Single-Agent RL Training (Standard) ──────────────────────────
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.agent == "ppo":
            agent = PPOAgent(
                obs_size=env.obs_size,
                n_actions=env.n_actions,
                device=device,
                use_lstm=args.use_lstm,
                hist_len=args.hist_len,
                lstm_hidden=args.lstm_hidden,
            )
        else:
            agent = DQNAgent(obs_size=env.obs_size, n_actions=env.n_actions, device=device, use_lstm=args.use_lstm)

        curriculum = CurriculumScheduler(total_episodes=args.episodes)

        logging.info(f"Training single {args.agent.upper()} agent for {args.episodes} episodes...")
        returns = train_agent(agent, env, n_episodes=args.episodes, agent_type=args.agent, curriculum=curriculum)

        eval_episodes = max(10, min(20, args.episodes))
        logging.info(f"Evaluating single {args.agent.upper()} agent over {eval_episodes} evaluation episodes...")
        eval_returns, final_summary = evaluate_agent(agent, eval_env, n_episodes=eval_episodes, agent_type=args.agent, greedy=True)
        eval_ret_avg = float(np.mean(eval_returns))

        report = {
            "supervised_model": args.model_name,
            "is_ensemble": is_ensemble,
            "multi_agent": False,
            "agent": args.agent,
            "episodes": args.episodes,
            "eval_episodes": eval_episodes,
            "train_return_pct": float(np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)),
            "eval_return_pct": eval_ret_avg,
            "max_drawdown_pct": float(final_summary["max_dd_pct"]),
            "n_trades": int(final_summary["n_trades"]),
            "sharpe": float(final_summary["sharpe"]),
        }

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

    else:
        # ── Multi-Agent RL Training & Consensus Ensembling ───────────────
        if args.ensemble_agents:
            agent_types = [s.strip().lower() for s in args.ensemble_agents.split(",") if s.strip()]
        else:
            agent_types = [args.agent] * args.num_agents

        logging.info(
            f"Initiating Multi-RL training with {len(agent_types)} agents: {agent_types} | "
            f"Consensus mode: '{args.consensus_mode}'"
        )

        trained_agents = []
        agent_metrics = []

        for i, atype in enumerate(agent_types):
            sub_seed = args.seed + i * 100
            torch.manual_seed(sub_seed)
            np.random.seed(sub_seed)

            # Diversity in hyperparameters across ensemble members
            if atype == "ppo":
                lr = 3e-4 * (0.85 + 0.15 * i)
                entropy_coef = 0.01 * (0.8 + 0.4 * i)
                sub_agent = PPOAgent(
                    obs_size=env.obs_size,
                    n_actions=env.n_actions,
                    lr=lr,
                    entropy_coef=entropy_coef,
                    device=device,
                    use_lstm=args.use_lstm,
                    hist_len=args.hist_len,
                    lstm_hidden=args.lstm_hidden,
                )
            elif atype == "dqn":
                lr = 1e-4 * (0.85 + 0.15 * i)
                eps_decay = 0.995 - 0.002 * i
                sub_agent = DQNAgent(
                    obs_size=env.obs_size,
                    n_actions=env.n_actions,
                    lr=lr,
                    eps_decay=eps_decay,
                    device=device,
                    use_lstm=args.use_lstm,
                )
            else:
                raise ValueError(f"Unknown agent type '{atype}'. Must be 'ppo' or 'dqn'.")

            curriculum = CurriculumScheduler(total_episodes=args.episodes)
            logging.info(
                f"[Multi-RL Agent {i+1}/{len(agent_types)}] Training {atype.upper()} "
                f"(seed={sub_seed}, episodes={args.episodes})..."
            )
            sub_returns = train_agent(
                sub_agent, env, n_episodes=args.episodes, agent_type=atype, curriculum=curriculum
            )
            eval_episodes = max(10, min(20, args.episodes))
            sub_eval_returns, sub_summary = evaluate_agent(
                sub_agent, eval_env, n_episodes=eval_episodes, agent_type=atype, greedy=True
            )
            ret_avg = float(np.mean(sub_eval_returns))
            train_ret_avg = float(np.mean(sub_returns[-10:]) if len(sub_returns) >= 10 else np.mean(sub_returns))
            logging.info(
                f"[Multi-RL Agent {i+1}/{len(agent_types)}] Finished | Eval Return: {ret_avg:+.2f}% | "
                f"Sharpe: {sub_summary['sharpe']:.2f} | Trades: {sub_summary['n_trades']}"
            )

            trained_agents.append(sub_agent)
            agent_metrics.append({
                "agent_id": i,
                "agent_type": atype,
                "seed": sub_seed,
                "train_return_pct": train_ret_avg,
                "eval_return_pct": ret_avg,
                "sharpe": float(sub_summary["sharpe"]),
                "max_drawdown_pct": float(sub_summary["max_dd_pct"]),
                "n_trades": int(sub_summary["n_trades"]),
                "total_costs": float(sub_summary["total_costs"]),
            })

        # Wrap into RLEnsemble
        logging.info(f"Assembling {len(trained_agents)} policies into RLEnsemble (mode='{args.consensus_mode}')...")
        ensemble = RLEnsemble(
            agents=trained_agents,
            agent_types=agent_types,
            consensus_mode=args.consensus_mode,
            obs_size=env.obs_size,
            n_actions=env.n_actions,
            device=str(device),
        )

        # Consensus Evaluation
        eval_episodes = max(10, min(20, args.episodes))
        logging.info(f"Evaluating RLEnsemble consensus policy over {eval_episodes} evaluation episodes...")
        ens_returns, ens_summary, ens_diag = ensemble.evaluate(eval_env, n_episodes=eval_episodes, greedy=True)
        ens_ret_avg = float(np.mean(ens_returns))
        logging.info(
            f"RLEnsemble Evaluation complete | Return: {ens_ret_avg:+.2f}% | "
            f"Sharpe: {ens_summary['sharpe']:.2f} | "
            f"Agreement: {ens_diag['mean_agreement_score']:.1%} | "
            f"Policy Uncertainty: {ens_diag['mean_policy_uncertainty']:.3f} | "
            f"Conflict Rate: {ens_diag['conflict_rate']:.1%}"
        )
        # Prune bankrupt/high-conflict ensemble: if 2/3 agents bankrupt or conflict >50%, rebuild with eligible only
        if float(ens_diag["conflict_rate"]) > 0.50 or float(ens_diag["mean_agreement_score"]) < 0.20:
            eligible_idx = [
                i for i, m in enumerate(agent_metrics)
                if float(m.get("eval_return_pct", 0.0)) > -10.0 and float(m.get("max_drawdown_pct", 100.0)) < 50.0
            ]
            if 1 <= len(eligible_idx) < len(trained_agents):
                logging.warning(f"Pruning {len(trained_agents)-len(eligible_idx)} bankrupt agent(s) due to conflict {ens_diag['conflict_rate']:.1%}")
                from models.rl_advanced import RLEnsemble as _RE
                pruned_agents = [trained_agents[i] for i in eligible_idx]
                pruned_types = [agent_types[i] for i in eligible_idx]
                ensemble = _RE(pruned_agents, pruned_types, consensus_mode=args.consensus_mode)
                ens_returns, ens_summary, ens_diag = ensemble.evaluate(eval_env, n_episodes=eval_episodes, greedy=True)
                ens_ret_avg = float(np.mean(ens_returns))
                logging.info(f"Pruned ensemble re-eval | Return {ens_ret_avg:+.2f}% Sharpe {ens_summary['sharpe']:.2f} Conflict {ens_diag['conflict_rate']:.1%}")

        # Save Artifact 1: RLEnsemble checkpoint
        ens_ckpt_path = out_dir / "rl_ensemble_best.pt"
        ensemble.save_checkpoint(
            ens_ckpt_path,
            meta={
                "supervised_model": args.model_name,
                "is_ensemble": is_ensemble,
                "consensus_mode": args.consensus_mode,
                "episodes_per_agent": args.episodes,
                "agent_types": agent_types,
                "agent_metrics": agent_metrics,
                "ensemble_summary": ens_summary,
                "ensemble_diagnostics": ens_diag,
            },
        )
        logging.info(f"Saved RLEnsemble checkpoint to {ens_ckpt_path}")

        # Save Artifact 2: Top individual agent as rl_best.pt (for backward compatibility)
        # Sharpe alone can select a catastrophic policy when it is computed
        # from a noisy/annualised P&L stream. Prefer positive-return agents
        # with controlled drawdown; use Sharpe only as the tie-breaker.
        eligible = [
            i for i, m in enumerate(agent_metrics)
            if float(m.get("eval_return_pct", m.get("total_return_pct", 0.0))) > 0.0
            and float(m.get("max_drawdown_pct", m.get("max_dd_pct", float("inf")))) <= 20.0
        ]
        ranked = eligible or list(range(len(agent_metrics)))
        best_idx = max(
            ranked,
            key=lambda i: (
                float(agent_metrics[i].get("eval_return_pct", agent_metrics[i].get("total_return_pct", 0.0))),
                -float(agent_metrics[i].get("max_drawdown_pct", agent_metrics[i].get("max_dd_pct", float("inf")))),
                float(agent_metrics[i].get("sharpe", 0.0)),
            ),
        )
        best_agent = trained_agents[best_idx]
        best_atype = agent_types[best_idx]
        rl_ckpt_path = out_dir / "rl_best.pt"
        if best_atype == "ppo":
            torch.save(best_agent.net.state_dict(), rl_ckpt_path)
        else:
            torch.save(best_agent.policy_net.state_dict(), rl_ckpt_path)
        logging.info(f"Saved top individual RL agent ({best_atype} #{best_idx}) to {rl_ckpt_path}")

        # Save Artifact 3: Extended rl_report.json
        report = {
            "supervised_model": args.model_name,
            "is_ensemble": is_ensemble,
            "multi_agent": True,
            "num_agents": len(trained_agents),
            "ensemble_agents": agent_types,
            "consensus_mode": args.consensus_mode,
            "episodes_per_agent": args.episodes,
            "individual_agents": agent_metrics,
            "ensemble_consensus": {
                "consensus_mode": args.consensus_mode,
                "eval_return_pct": ens_ret_avg,
                "sharpe": float(ens_summary["sharpe"]),
                "max_drawdown_pct": float(ens_summary["max_dd_pct"]),
                "n_trades": int(ens_summary["n_trades"]),
                "total_costs": float(ens_summary["total_costs"]),
                "mean_agreement_score": float(ens_diag["mean_agreement_score"]),
                "mean_disagreement_score": float(ens_diag["mean_disagreement_score"]),
                "mean_policy_uncertainty": float(ens_diag["mean_policy_uncertainty"]),
                "conflict_rate": float(ens_diag["conflict_rate"]),
                "evaluation_episodes": eval_episodes,
            },
        }
        report_path = out_dir / "rl_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logging.info(f"Saved extended Multi-RL report to {report_path}")


if __name__ == "__main__":
    main()
