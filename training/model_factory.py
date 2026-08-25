"""Model construction helpers for GPU training.

Re-export shim: the implementations now live in ``models/factory.py`` so that
inference code can import model construction without pulling in ``training``
(breaks the training <-> inference cycle). See docs/CONTINUE.md.
"""

from __future__ import annotations

from config.model_training_profile import (
    MODEL_PROFILES,
    ModelTrainingProfile,
    get_training_profile,
)
from models.factory import (  # noqa: F401  (re-export)
    _core_model,
    _multitask_head_in,
    _strict_load_report,
    build_model,
)


def get_model_training_profile(model_name: str) -> ModelTrainingProfile:
    """Convenience function to get training profile from model factory."""
    return get_training_profile(model_name)


__all__ = [
    "MODEL_PROFILES",
    "ModelTrainingProfile",
    "_core_model",
    "_multitask_head_in",
    "_strict_load_report",
    "build_model",
    "get_model_training_profile",
]
