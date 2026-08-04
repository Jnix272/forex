"""
audit — full lineage + reproducibility manifests (Improvement #6)

Two modules:
  * audit.lineage   — data lineage (dataset → preprocessing → features →
                      labels → training run), model registry records, and an
                      audit trail of promotion / rollback decisions.
  * audit.manifest  — generate / verify / regenerate reproducibility manifests
                      (JSON snapshots of every artifact + inputs + environment
                      for a training run) stored alongside checkpoints.
"""

from .lineage import DataLineage, LineageStep, ModelRegistryRecord, decision_trail, DecisionRecord
from .manifest import (
    compute_file_hash,
    compute_dir_hash,
    hash_inputs,
    capture_env,
    git_info,
    generate_manifest,
    write_manifest,
    verify_manifest,
    regenerate_manifest,
    verify_manifests_in_tree,
)

__all__ = [
    "DataLineage",
    "LineageStep",
    "ModelRegistryRecord",
    "decision_trail",
    "DecisionRecord",
    "compute_file_hash",
    "compute_dir_hash",
    "hash_inputs",
    "capture_env",
    "git_info",
    "generate_manifest",
    "write_manifest",
    "verify_manifest",
    "regenerate_manifest",
    "verify_manifests_in_tree",
]
