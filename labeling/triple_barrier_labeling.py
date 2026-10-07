"""Backward-compat shim — the module was renamed to labeling/cpar_labeling.py."""
from labeling.cpar_labeling import *  # noqa: F401, F403
from labeling.cpar_labeling import (  # noqa: F401
    _NUMBA_IMPORT_OK,
    _scan_outcomes_cpar_numba,
    _scan_outcomes_cpar_sequential,
)

