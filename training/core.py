"""Core globals and shims to break circular imports across the training module."""

from typing import Any

# -----------------------------------------------------------------------------
# Configuration Globals
# -----------------------------------------------------------------------------
try:
    from config.settings import GPU as _GPU_CFG
except ImportError:
    _GPU_CFG = {}

try:
    import wandb  # noqa: F401

    WANDB = True
except ImportError:
    WANDB = False

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA = True
except ImportError:
    OPTUNA = False

_FIRST_CHUNK_COLS: list | None = None  # feature-column order lock for chunk builds

# -----------------------------------------------------------------------------
# Logger & Shims
# -----------------------------------------------------------------------------
try:
    from monitoring.train_logger import TrainingLogger as _TrainingLogger

    _TRAIN_LOGGER_AVAILABLE = True
except ImportError:
    _TrainingLogger = None  # type: ignore[misc, assignment]
    _TRAIN_LOGGER_AVAILABLE = False

_TRAIN_LOGGER: Any | None = None  # TrainingLogger instance, set in supervised_train


def _log_error(msg: str, exc: Exception | None = None) -> None:
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.error(msg, exc)


def _log_warn(msg: str) -> None:
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.warning(msg)


def _log_info(msg: str) -> None:
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.info(msg)


def _log_oom(batch_idx: int, epoch: int, oom_count: int) -> None:
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.on_batch_oom(batch_idx, epoch, oom_count)


def _log_nan(batch_idx: int, epoch: int, nan_count: int) -> None:
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.on_batch_nan(batch_idx, epoch, nan_count)


class _DummyCtx:
    """No-op context manager - used when rich display is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# W&B Safety Shims
_WANDB_BROKEN = False


def _safe_wandb_log(run, payload: dict, *, step=None) -> None:
    global _WANDB_BROKEN
    if not (WANDB and run is not None) or _WANDB_BROKEN:
        return
    try:
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=step)
    except Exception as exc:
        _WANDB_BROKEN = True
        print(f"[W&B] Logging disabled after failure: {exc}")


def _safe_wandb_summary_update(run, payload: dict) -> None:
    global _WANDB_BROKEN
    if not (WANDB and run is not None) or _WANDB_BROKEN:
        return
    try:
        run.summary.update(payload)
    except Exception as exc:
        _WANDB_BROKEN = True
        print(f"[W&B] Summary updates disabled after failure: {exc}")


def _crop_to_seq_len(t, seq_len):
    """Crop the last dimension of a tensor/array to seq_len if longer.

    Works with both torch.Tensor (dim 1) and numpy arrays (axis 1).
    """
    if seq_len is None:
        return t
    target = int(seq_len)
    if target <= 0:
        return t
    # torch.Tensor: dim 1, numpy: axis 1
    try:
        # torch.Tensor
        if hasattr(t, "size") and t.size(1) > target:
            return t[:, -target:]
    except Exception:
        pass
    try:
        # numpy array
        if hasattr(t, "shape") and t.shape[1] > target:
            return t[:, -target:]
    except Exception:
        pass
    return t
