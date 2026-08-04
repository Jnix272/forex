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

from .lineage import DataLineage, DecisionRecord, LineageStep, ModelRegistryRecord, decision_trail
from .manifest import (
    capture_env,
    compute_dir_hash,
    compute_file_hash,
    generate_manifest,
    git_info,
    hash_inputs,
    regenerate_manifest,
    verify_manifest,
    verify_manifests_in_tree,
    write_manifest,
)

__all__ = [
    "DataLineage",
    "DecisionRecord",
    "LineageStep",
    "ModelRegistryRecord",
    "capture_env",
    "compute_dir_hash",
    "compute_file_hash",
    "decision_trail",
    "generate_manifest",
    "git_info",
    "hash_inputs",
    "regenerate_manifest",
    "verify_manifest",
    "verify_manifests_in_tree",
    "write_manifest",
]
