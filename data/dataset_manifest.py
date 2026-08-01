"""
data/dataset_manifest.py
=========================
Generates and strictly validates dataset manifests to ensure the model
never trains on mismatched or unexpectedly altered caches.

Also provides:
  - Structured JSON build log (build_log.jsonl) written during dataset construction
  - Future-leak correlation check (flags features correlated with forward returns)
  - Lockbox reservation (holds out the latest N days from training)
  - Label contamination check (verifies feature timestamps < label timestamps)
  - Curriculum difficulty sidecar audit log
"""

import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()[:12]
    except Exception:
        return ""


class DatasetManifest:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.logger = logging.getLogger(__name__)

    # ── Manifest writing ──────────────────────────────────────────────

    def write_manifest(
        self,
        source: str,
        pairs: list,
        start: str,
        end: str,
        freq: str,
        news_mode: bool,
        feature_count: int,
        label_method: str,
        seq_len: int,
        schema_hash: str,
        feature_list: list | None = None,
        n_rows_per_pair: dict | None = None,
        n_rows_total: int | None = None,
        build_duration_seconds: float | None = None,
        lookahead_bars: int | None = None,
        embargo_bars: int | None = None,
        purge_bars: int | None = None,
        lockbox_start: str | None = None,
        lockbox_end: str | None = None,
    ) -> dict:
        """Writes a strict manifest file next to the processed cache."""
        manifest = {
            "source": source,
            "pairs": pairs,
            "start": start,
            "end": end,
            "frequency": freq,
            "news_mode": news_mode,
            "feature_count": feature_count,
            "feature_list": feature_list or [],
            "label_method": label_method,
            "sequence_length": seq_len,
            "schema_hash": schema_hash,
            "cache_creation_time": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "n_rows_per_pair": n_rows_per_pair or {},
            "n_rows_total": n_rows_total or 0,
            "build_duration_seconds": round(build_duration_seconds or 0, 2),
            "leakage_prevention": {
                "lookahead_bars": lookahead_bars,
                "embargo_bars": embargo_bars,
                "purge_bars": purge_bars,
            },
            "lockbox": {
                "start": lockbox_start,
                "end": lockbox_end,
                "reserved": bool(lockbox_start and lockbox_end),
            },
        }

        manifest_path = self.cache_dir / "dataset_manifest.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Wrote dataset manifest to {manifest_path}")
        return manifest

    # ── Manifest validation ───────────────────────────────────────────

    def validate_cache(self, expected_seq_len: int, expected_schema_hash: str) -> bool:
        """Fails fast if the cache manifest doesn't match training expectations."""
        manifest_path = self.cache_dir / "dataset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Missing dataset_manifest.json. Refusing to train.")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if manifest["sequence_length"] != expected_seq_len:
            raise ValueError(
                f"Cache sequence length ({manifest['sequence_length']}) "
                f"does not match expected ({expected_seq_len})."
            )

        if expected_schema_hash and manifest["schema_hash"] != expected_schema_hash:
            raise ValueError(
                "Schema hash mismatch! The features in this cache have been altered."
            )

        self.logger.info("Dataset manifest validated successfully.")
        return True

    # ── Structured build log ──────────────────────────────────────────

    def log_build_event(
        self,
        event: str,
        pair: str | None = None,
        chunk_idx: int | None = None,
        n_rows: int | None = None,
        n_features: int | None = None,
        duration_s: float | None = None,
        extra: dict | None = None,
    ) -> None:
        """Append a structured JSON line to build_log.jsonl."""
        log_path = self.cache_dir / "build_log.jsonl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "pair": pair,
            "chunk_idx": chunk_idx,
            "n_rows": n_rows,
            "n_features": n_features,
            "duration_s": round(duration_s, 3) if duration_s is not None else None,
            "extra": extra or {},
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Data fingerprinting (INF-011) ────────────────────────────────

    @staticmethod
    def fingerprint_data(data_path: str, n_sample_rows: int = 10000) -> dict:
        """INF-011: Fingerprint raw data payload to detect silent modification.

        Hashes the timestamp index + shape + first/last row content.
        Stores in manifest so training provenance can be verified.
        """

        p = Path(data_path)
        if not p.exists():
            return {"error": f"path not found: {data_path}"}

        try:
            if p.suffix == ".zarr" or p.is_dir():
                import zarr
                z = zarr.open(str(p), mode="r")
                if hasattr(z, "shape"):
                    shape = z.shape
                    content = f"{shape}"
                    if len(shape) >= 1 and shape[0] > 0:
                        first_row = str(z[0][:min(10, shape[1] if len(shape) > 1 else 1)])
                        last_row = str(z[-1][:min(10, shape[1] if len(shape) > 1 else 1)])
                        content += first_row + last_row
                else:
                    arrays = list(z.arrays()) if hasattr(z, "arrays") else []
                    shape = (len(arrays),)
                    content = str([(name, arr.shape) for name, arr in arrays[:5]])
            elif p.suffix == ".parquet":
                import pyarrow.parquet as pq
                meta = pq.read_metadata(str(p))
                shape = (meta.num_rows, meta.num_columns)
                content = f"{shape}_{meta.serialized_size}"
            elif p.suffix == ".csv":
                shape = (0, 0)
                with open(p, "r") as f:
                    header = f.readline()
                    content = header
                    line_count = sum(1 for _ in f) + 1
                    shape = (line_count, header.count(",") + 1)
                    content += f"{shape}"
            else:
                stat = p.stat()
                shape = (stat.st_size,)
                content = f"{stat.st_size}_{stat.st_mtime}"

            fingerprint = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
            return {
                "fingerprint": fingerprint,
                "shape": list(shape),
                "path": str(p),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"error": str(e), "path": str(p)}

    def write_fingerprint(self, data_path: str) -> dict:
        """Generate and store a data fingerprint in the cache manifest directory."""
        fp = self.fingerprint_data(data_path)
        fp_path = self.cache_dir / "data_fingerprint.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(fp_path, "w", encoding="utf-8") as f:
            json.dump(fp, f, indent=2)
        self.logger.info(f"Wrote data fingerprint to {fp_path}")
        return fp

    # ── Future-leak check ─────────────────────────────────────────────

    @staticmethod
    def check_future_leak(
        feature_df,
        forward_returns: list,
        max_abs_corr: float = 0.3,
    ) -> list[dict]:
        """
        Flag features that are correlated with forward returns — a sign
        they may contain future information (data leakage).

        Returns list of {feature, corr, flagged} dicts.
        """
        import numpy as np

        if feature_df is None or feature_df.empty or not forward_returns:
            return []

        if isinstance(forward_returns, list):
            fwd = np.array(forward_returns, dtype=np.float32)
        else:
            fwd = np.asarray(forward_returns, dtype=np.float32)

        min_len = min(len(feature_df), len(fwd))
        feature_df = feature_df.iloc[:min_len]
        fwd = fwd[:min_len]

        flagged = []
        for col in feature_df.columns:
            try:
                vals = np.asarray(feature_df[col], dtype=np.float32)
                if np.std(vals) < 1e-12:
                    continue
                corr = np.corrcoef(vals, fwd)[0, 1]
                if abs(corr) > max_abs_corr:
                    flagged.append({"feature": col, "corr": round(float(corr), 4)})
            except Exception:
                continue

        flagged.sort(key=lambda x: abs(x["corr"]), reverse=True)
        return flagged

    # ── Label contamination check ─────────────────────────────────────

    @staticmethod
    def check_label_contamination(
        feature_timestamps,
        label_timestamps,
        max_tolerance_seconds: float = 0.0,
    ) -> dict:
        """
        Verify that every feature row's timestamp is strictly before its
        corresponding label row's timestamp. Returns {ok: bool, violations: int}.
        """
        import numpy as np

        feat = np.asarray(feature_timestamps, dtype=np.int64)
        labels = np.asarray(label_timestamps, dtype=np.int64)
        min_len = min(len(feat), len(labels))
        feat = feat[:min_len]
        labels = labels[:min_len]

        violations = int(np.sum(feat >= labels - max_tolerance_seconds))
        return {
            "ok": violations == 0,
            "violations": violations,
            "total_checked": min_len,
        }

    # ── Lockbox reservation ───────────────────────────────────────────

    @staticmethod
    def reserve_lockbox(
        cache_dir: str,
        lockbox_start: str,
        lockbox_end: str,
    ) -> dict:
        """
        Reserve the latest N days as a locked test set that is never
        touched during feature computation or training. Writes lockbox.json
        next to the cache for audit trail.
        """
        lockbox_path = Path(cache_dir) / "lockbox.json"
        info = {
            "lockbox_start": lockbox_start,
            "lockbox_end": lockbox_end,
            "reserved_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
        }
        with open(lockbox_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        logger.info(f"Wrote lockbox reservation to {lockbox_path}")
        return info

    # ── Curriculum difficulty audit ────────────────────────────────────

    def log_curriculum_stage(
        self,
        epoch: int,
        stage: str,
        seq_len: int,
        unfrozen_groups: list[str],
        difficulty: int,
    ) -> None:
        """Log a curriculum stage transition for audit trail."""
        self.log_build_event(
            "curriculum_stage",
            chunk_idx=epoch,
            extra={
                "stage": stage,
                "seq_len": seq_len,
                "unfrozen_groups": unfrozen_groups,
                "difficulty": difficulty,
            },
        )