"""
scripts/slice_single_pair_caches.py
===================================
Creates dedicated single-pair specialist YAML configurations:
  - config/run_eurusd.yaml  -> checkpoints/haelt_eurusd_v1
  - config/run_usdcad.yaml  -> checkpoints/haelt_usdcad_v1
  - config/run_usdjpy.yaml  -> checkpoints/haelt_usdjpy_v1

And slices the existing 3-pair processed Zarr dataset
(EURUSD, USDJPY, USDCAD; 191,271 x 120 x 438) into three isolated
single-pair Zarr datasets (191,271 x 120 x 146 each) in a single
chunk-aligned pass so single-pair specialist training starts immediately
without a 9+ hour tick-to-feature rebuild.

Usage:
  python scripts/slice_single_pair_caches.py --configs-only
  python scripts/slice_single_pair_caches.py
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPECIALIST_PAIRS = ("EURUSD", "USDCAD", "USDJPY")


def generate_single_pair_configs(base_config_path: Path | None = None) -> dict[str, Path]:
    """Generate config/run_<pair>.yaml for each specialist pair from config/run.yaml."""
    base_path = base_config_path or (ROOT / "config" / "run.yaml")
    with open(base_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    written: dict[str, Path] = {}
    for pair in SPECIALIST_PAIRS:
        cfg = copy.deepcopy(base_cfg)
        pair_lower = pair.lower()

        cfg.setdefault("data", {})
        cfg["data"]["pair"] = pair
        cfg["data"]["pairs"] = [pair]

        cfg.setdefault("paths", {})
        cfg["paths"]["checkpoint_dir"] = f"checkpoints/haelt_{pair_lower}_v1"

        cfg.setdefault("tracking", {})
        cfg["tracking"]["run_name"] = f"haelt-{pair_lower}-specialist"

        cfg.setdefault("training", {})
        # Keep per_pair_heads=True so MultiPairMultiTaskWrapper(pairs=[pair]) and
        # honest_eval pooled_pair_metrics (trade filters, pip_size, gap windows,
        # FOREX_TRADE_FRACTION quantile gating) run identically to multi-pair.
        cfg["training"]["per_pair_heads"] = True
        cfg["training"]["batch_size"] = 300
        # Dedicated Single-Pair Specialist Sharpe Reliability & Quantile Settings:
        cfg["training"]["min_reliable_val_trades"] = 30
        cfg["training"]["trade_fraction"] = 0.05
        # Enable Continual Learning & Weight-Space Regularizers for Specialist Models:
        # 1. SI (Synaptic Intelligence) + Dynamic SI (relaxes during feature drift shocks,
        #    anchors weights when stable so peak-epoch representations aren't forgotten).
        # 2. EWC (Elastic Weight Consolidation: locks Fisher diagonal on --warm-start).
        # 3. SWA (Stochastic Weight Averaging over final 25% of epochs).
        cfg["training"]["enable_si"] = True
        cfg["training"]["si_dynamic"] = True
        cfg["training"]["si_lambda"] = 0.5
        cfg["training"]["si_lambda_min"] = 0.05
        cfg["training"]["si_lambda_max"] = 2.0
        cfg["training"]["enable_ewc"] = True
        cfg["training"]["ewc_lambda"] = 400.0
        cfg["training"]["swa_enabled"] = True

        # Enable Self-Paced Learning (SPL, Kumar et al. 2010) + Miner Feedback
        # via Unified CurriculumManager in "self_paced" mode so per-sample loss
        # weights v_i = exp(-L_i / (lambda * tau)) smoothly pace easy -> hard
        # bars across epochs without hard-cropping the training set.
        cfg.setdefault("curriculum", {})
        cfg["curriculum"]["manager"] = {
            "enabled": True,
            "mode": "self_paced",
        }
        cfg["curriculum"].setdefault("self_paced", {})
        cfg["curriculum"]["self_paced"]["enabled"] = True
        cfg["curriculum"]["self_paced"]["pace"] = "linear"
        cfg["curriculum"]["self_paced"]["lambda_pace"] = 1.0
        cfg["curriculum"]["self_paced"]["models"] = "haelt,tft,transformer"
        cfg["curriculum"].setdefault("miner_feedback", {})
        cfg["curriculum"]["miner_feedback"]["enabled"] = True
        cfg["curriculum"]["miner_feedback"]["models"] = "haelt,tft,transformer"
        cfg["curriculum"]["miner_feedback"]["forgetting_threshold"] = 0.15
        cfg["curriculum"]["miner_feedback"]["easy_threshold"] = 0.60
        cfg["curriculum"]["miner_feedback"]["freeze_patience"] = 1

        # Unlock all stationary advanced feature groups (Volume Profile, Volatility Clock,
        # Hurst, HMM regime, COT positioning, Yield momentum, Eco Surprise, OBI/Liquidity Vacuum)
        # while keeping raw price-level columns neutralized via neutralize_price_level_columns
        # and dropping non-stationary raw price levels and noisy raw FinBERT projection dims (fb_0..fb_7).
        cfg.setdefault("feature_ablation", {})
        cfg["feature_ablation"]["enabled"] = True
        cfg["feature_ablation"]["name"] = "specialist_advanced_stationary"
        cfg["feature_ablation"]["keep_groups"] = [
            "core",
            "cross_asset",
            "session",
            "momentum",
            "higher_timeframe",
            "execution_cost",
            "microstructure",
            "volatility",
            "market_regime",
            "macro",
            "news",
        ]
        cfg["feature_ablation"]["drop_ungrouped"] = False
        raw_price_cols = ["open", "high", "low", "close", "bid_close", "ask_close"]
        cfg["feature_ablation"]["drop_features"] = [f"fb_{i}" for i in range(8)] + raw_price_cols

        out_path = ROOT / "config" / f"run_{pair_lower}.yaml"
        header = (
            f"# Dedicated Single-Pair Specialist Config for {pair}\n"
            f"# Auto-generated from config/run.yaml by scripts/slice_single_pair_caches.py\n"
            f"# Train with:\n"
            f"#   $env:FOREX_PURE_BCE=\"1\"; $env:FOREX_TRADE_FRACTION=\"0.05\"; "
            f"python scripts/train.py --config config/run_{pair_lower}.yaml --full\n\n"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        written[pair] = out_path
        print(f"[Config] Wrote {out_path.relative_to(ROOT)} -> {cfg['paths']['checkpoint_dir']}")

    return written


def _find_source_multipair_zarr(processed_dir: Path) -> Path:
    candidates = sorted(
        processed_dir.glob(
            "dataset_scalping_5m_EURUSD-USDCAD-USDJPY_*_triple_barrier_reward_lh30_*.zarr"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No 3-pair EURUSD-USDCAD-USDJPY Zarr cache found in {processed_dir}"
        )
    return candidates[0]


def slice_multipair_to_single_pair_zarrs(
    source_zarr: Path | None = None,
    pairs_to_slice: tuple[str, ...] = SPECIALIST_PAIRS,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Slice 3-pair Zarr into 1-pair Zarr stores in a single sequential chunk pass."""
    import zarr
    from common.cache_io import _zarr_open_group, _zarr_create, make_training_zarr_compressor, _scaler_npz_path
    from sklearn.preprocessing import RobustScaler
    from training.cache_integrity import _validate_cache_integrity
    from training.dataset_builder import _save_scaler_npz

    processed_dir = ROOT / "data" / "processed"
    src_path = source_zarr or _find_source_multipair_zarr(processed_dir)
    print(f"[Slicer] Source 3-pair Zarr: {src_path.name}", flush=True)

    src_manifest_path = Path(str(src_path) + "_manifest.json")
    src_schema_path = Path(str(src_path) + "_feature_schema.json")
    src_audit_path = Path(str(src_path) + "_feature_schema_audit.json")

    src_manifest = json.loads(src_manifest_path.read_text(encoding="utf-8"))
    src_schema: list[str] = json.loads(src_schema_path.read_text(encoding="utf-8"))
    src_pairs: list[str] = list(src_manifest["pairs"])
    n_samples = int(src_manifest["total_samples"])
    seq_len = int(src_manifest["seq_len"])
    total_feat = int(src_manifest["n_features"])
    f_per_pair = total_feat // len(src_pairs)
    assert f_per_pair * len(src_pairs) == total_feat, (
        f"Expected divisible features, got {total_feat} across {src_pairs}"
    )

    src_root = zarr.open(str(src_path), mode="r")
    src_X = src_root["X"]
    chunk_rows = int(src_X.chunks[0])
    compressor_x = make_training_zarr_compressor()
    compressor_1d = make_training_zarr_compressor()

    # Prepare target paths and Zarr groups for each requested pair
    targets: dict[str, dict] = {}
    for pair in pairs_to_slice:
        if pair not in src_pairs:
            raise ValueError(f"Pair {pair} not in source Zarr pairs {src_pairs}")
        slot_idx = src_pairs.index(pair)
        f_start = slot_idx * f_per_pair
        f_end = f_start + f_per_pair

        dst_name = src_path.name.replace("EURUSD-USDCAD-USDJPY", pair)
        dst_path = processed_dir / dst_name
        manifest_path = Path(str(dst_path) + "_manifest.json")
        schema_path = Path(str(dst_path) + "_feature_schema.json")
        scaler_path = _scaler_npz_path(dst_path)

        is_complete = False
        if dst_path.exists() and manifest_path.exists() and schema_path.exists() and scaler_path.exists():
            try:
                dst_check = zarr.open(str(dst_path), mode="r")
                if "X" in dst_check and dst_check["X"].shape == (n_samples, seq_len, f_per_pair):
                    is_complete = True
            except Exception:
                is_complete = False

        if is_complete and not overwrite:
            print(
                f"[Slicer] {dst_path.name} already exists with full {n_samples:,} rows "
                f"and valid manifest. Reusing.",
                flush=True,
            )
            continue

        print(
            f"[Slicer] Preparing {pair} single-pair slice into {dst_path.name} "
            f"(full {n_samples:,} rows)...",
            flush=True,
        )
        if dst_path.exists():
            shutil.rmtree(dst_path)

        dst_root = _zarr_open_group(str(dst_path), mode="w")
        dst_X = _zarr_create(
            dst_root,
            "X",
            shape=(n_samples, seq_len, f_per_pair),
            chunks=(chunk_rows, seq_len, f_per_pair),
            dtype=np.float32,
            compressor=compressor_x,
            overwrite=True,
        )

        # Slice 1D / (N, 1) target and market arrays immediately (fast, tiny in RAM)
        y_1d = np.asarray(src_root["y_pairs"][:, slot_idx], dtype=np.float32)
        ycls_1d = np.asarray(src_root["ycls_pairs"][:, slot_idx], dtype=np.float32)
        close_1d = np.asarray(src_root["close_pairs"][:, slot_idx], dtype=np.float32)
        atr_1d = np.asarray(src_root["atr_pairs"][:, slot_idx], dtype=np.float32)
        spread_1d = np.asarray(src_root["spread_pairs"][:, slot_idx], dtype=np.float32)
        t_ns_1d = np.asarray(src_root["t_ns"][:], dtype=np.int64)

        for name, arr in (
            ("y", y_1d),
            ("y_cls", ycls_1d),
            ("close", close_1d),
            ("atr", atr_1d),
            ("spread", spread_1d),
        ):
            ds = _zarr_create(
                dst_root,
                name,
                shape=arr.shape,
                chunks=(chunk_rows,),
                dtype=np.float32,
                compressor=compressor_1d,
                overwrite=True,
            )
            ds[:] = arr

        for name, arr2d in (
            ("y_pairs", y_1d[:, None]),
            ("ycls_pairs", ycls_1d[:, None]),
            ("close_pairs", close_1d[:, None]),
            ("atr_pairs", atr_1d[:, None]),
            ("spread_pairs", spread_1d[:, None]),
        ):
            ds = _zarr_create(
                dst_root,
                name,
                shape=arr2d.shape,
                chunks=(chunk_rows, 1),
                dtype=np.float32,
                compressor=compressor_1d,
                overwrite=True,
            )
            ds[:] = arr2d

        ds_t = _zarr_create(
            dst_root,
            "t_ns",
            shape=t_ns_1d.shape,
            chunks=(chunk_rows,),
            dtype=np.int64,
            compressor=compressor_1d,
            overwrite=True,
        )
        ds_t[:] = t_ns_1d

        pair_schema = src_schema[f_start:f_end]
        targets[pair] = {
            "slot_idx": slot_idx,
            "f_start": f_start,
            "f_end": f_end,
            "dst_path": dst_path,
            "dst_root": dst_root,
            "dst_X": dst_X,
            "pair_schema": pair_schema,
            "sample_last_bars": [],
        }

    if not targets:
        return {
            p: processed_dir / src_path.name.replace("EURUSD-USDCAD-USDJPY", p)
            for p in pairs_to_slice
        }

    # Single sequential pass over src_X chunks
    n_chunks = (n_samples + chunk_rows - 1) // chunk_rows
    t0 = time.time()
    print(
        f"[Slicer] Streaming {n_samples:,} rows ({n_chunks} chunks of {chunk_rows}) "
        f"into {len(targets)} single-pair Zarr stores...",
        flush=True,
    )
    for c_idx, start in enumerate(range(0, n_samples, chunk_rows)):
        end = min(start + chunk_rows, n_samples)
        blk = np.asarray(src_X[start:end], dtype=np.float32)
        for pair, info in targets.items():
            sub = np.ascontiguousarray(blk[:, :, info["f_start"] : info["f_end"]])
            info["dst_X"][start:end] = sub
            # Collect last-bar samples from first ~130 chunks (~50k rows) to fit scaler
            if c_idx < 130:
                info["sample_last_bars"].append(sub[:, -1, :].copy())

        if (c_idx + 1) % 25 == 0 or (c_idx + 1) == n_chunks:
            elapsed = time.time() - t0
            rate = (c_idx + 1) / max(elapsed, 1e-6)
            eta = (n_chunks - (c_idx + 1)) / max(rate, 1e-6)
            print(
                f"  [{c_idx + 1:3d}/{n_chunks}] rows={end:,}/{n_samples:,} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    # Finalize attrs, manifests, schemas, and scalers
    out_paths: dict[str, Path] = {}
    for pair, info in targets.items():
        dst_path: Path = info["dst_path"]
        dst_root = info["dst_root"]
        pair_schema: list[str] = info["pair_schema"]

        # .zattrs
        zattrs = dict(src_root.attrs)
        zattrs["n_features"] = f_per_pair
        zattrs["n_pairs"] = 1
        zattrs["pairs"] = pair
        dst_root.attrs.update(zattrs)

        # _manifest.json
        dst_manifest = copy.deepcopy(src_manifest)
        dst_manifest["n_features"] = f_per_pair
        dst_manifest["n_pairs"] = 1
        dst_manifest["pairs"] = [pair]
        Path(str(dst_path) + "_manifest.json").write_text(
            json.dumps(dst_manifest), encoding="utf-8"
        )

        # _feature_schema.json
        Path(str(dst_path) + "_feature_schema.json").write_text(
            json.dumps(pair_schema), encoding="utf-8"
        )
        if src_audit_path.exists():
            shutil.copyfile(
                src_audit_path, Path(str(dst_path) + "_feature_schema_audit.json")
            )

        # Slice 146-feature RobustScaler directly from source 438-feature scaler
        from training.dataset_builder import _load_scaler_npz

        src_scaler = _load_scaler_npz(src_path)
        scaler = RobustScaler(quantile_range=(5.0, 95.0))
        if src_scaler is not None and hasattr(src_scaler, "center_") and len(src_scaler.center_) == total_feat:
            scaler.center_ = np.asarray(src_scaler.center_[info["f_start"] : info["f_end"]], dtype=np.float64)
            scaler.scale_ = np.asarray(src_scaler.scale_[info["f_start"] : info["f_end"]], dtype=np.float64)
            scaler.n_features_in_ = f_per_pair
            scaler.n_samples_seen_ = int(getattr(src_scaler, "n_samples_seen_", n_samples) or n_samples)
        else:
            sample_mat = np.concatenate(info["sample_last_bars"], axis=0)
            finite_mask = np.isfinite(sample_mat).all(axis=1)
            scaler.fit(sample_mat[finite_mask])
        scaler.feature_names_in_ = np.asarray(pair_schema, dtype=object)
        _save_scaler_npz(dst_path, scaler)

        ok, reason = _validate_cache_integrity(str(dst_path), None)
        print(
            f"[Slicer] Finalized {pair}: {dst_path.name} | "
            f"shape=({n_samples}, {seq_len}, {f_per_pair}) | integrity={ok} ({reason})"
        )
        out_paths[pair] = dst_path

    return out_paths


def main() -> int:
    p = argparse.ArgumentParser(description="Generate single-pair configs and slice 3-pair Zarr cache.")
    p.add_argument("--configs-only", action="store_true", help="Only write config/run_<pair>.yaml files")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing single-pair Zarr caches")
    p.add_argument(
        "--pairs",
        type=str,
        default=",".join(SPECIALIST_PAIRS),
        help="Comma-separated specialist pairs to slice (default: EURUSD,USDCAD,USDJPY)",
    )
    args = p.parse_args()

    generate_single_pair_configs()
    if args.configs_only:
        return 0

    pairs = tuple(x.strip().upper() for x in args.pairs.split(",") if x.strip())
    slice_multipair_to_single_pair_zarrs(pairs_to_slice=pairs, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
