"""
Loss Plateau Checks.

Detects training plateaus and loss divergence.
"""

from collections import deque

from monitoring.checks import CheckContext, CheckResult, CheckStatus, register_check

# Global history for epoch-level tracking
_loss_history: dict[str, deque] = {}


def check_loss_plateau(context: CheckContext) -> CheckResult:
    """Check for loss plateau at epoch level."""
    config = context.config
    patience = config.get("loss_plateau_patience", 6)
    min_delta = config.get("loss_plateau_min_delta", 1e-4)
    window = config.get("loss_plateau_window", 10)

    # Get current loss
    if context.metrics:
        loss = context.metrics.get("train_loss") or context.metrics.get("loss")
    else:
        loss = None

    if loss is None and context.loss is not None:
        if hasattr(context.loss, "item"):
            loss = context.loss.item()
        else:
            loss = float(context.loss)

    if loss is None:
        return CheckResult(
            name="loss_plateau",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No loss available",
        )

    # Update history
    key = f"{context.run_id}:{context.model_name}"
    if key not in _loss_history:
        _loss_history[key] = deque(maxlen=window)
    _loss_history[key].append(loss)

    history = _loss_history[key]
    if len(history) < 3:
        return CheckResult(
            name="loss_plateau",
            status=CheckStatus.PASSED,
            passed=True,
            message=f"Building history ({len(history)}/{window})",
            value=loss,
            details={"history_len": len(history), "window": window},
        )

    # Check for plateau
    list(history)[-patience:] if len(history) >= patience else list(history)
    best = min(history)
    current = history[-1]
    improvement = best - current

    plateau = improvement < min_delta
    diverging = current > best * 2  # Loss doubled from best

    if diverging:
        return CheckResult(
            name="loss_plateau",
            status=CheckStatus.FAILED,
            passed=False,
            value=loss,
            threshold=best * 2,
            message=f"LOSS DIVERGING: current {loss:.6f} > 2x best {best:.6f}",
            details={
                "current": current,
                "best": best,
                "diverging": True,
                "history": list(history),
            },
        )

    if plateau and len(history) >= patience:
        return CheckResult(
            name="loss_plateau",
            status=CheckStatus.FAILED,
            passed=False,
            value=loss,
            threshold=min_delta,
            message=f"PLATEAU: No improvement > {min_delta} for {patience} epochs (best={best:.6f}, current={current:.6f})",
            details={
                "current": current,
                "best": best,
                "improvement": improvement,
                "patience": patience,
                "history": list(history),
            },
        )

    return CheckResult(
        name="loss_plateau",
        status=CheckStatus.PASSED,
        passed=True,
        value=loss,
        message=f"Loss OK: {loss:.6f} (best={best:.6f}, imp={best - loss:.6f})",
        details={
            "current": current,
            "best": best,
            "improvement": improvement,
            "history_len": len(history),
        },
    )


def check_val_loss_plateau(context: CheckContext) -> CheckResult:
    """Check validation loss plateau."""
    config = context.config
    patience = config.get("val_loss_plateau_patience", 10)
    min_delta = config.get("val_loss_plateau_min_delta", 1e-4)

    val_loss = context.metrics.get("val_loss") if context.metrics else None
    if val_loss is None:
        return CheckResult(
            name="val_loss_plateau",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No val_loss available",
        )

    key = f"{context.run_id}:{context.model_name}:val"
    if key not in _loss_history:
        _loss_history[key] = deque(maxlen=patience + 5)
    _loss_history[key].append(val_loss)

    history = _loss_history[key]
    if len(history) < patience:
        return CheckResult(
            name="val_loss_plateau",
            status=CheckStatus.PASSED,
            passed=True,
            message=f"Building val history ({len(history)}/{patience})",
            value=val_loss,
        )

    best = min(history)
    current = history[-1]
    improvement = best - current
    plateau = improvement < min_delta

    if plateau:
        return CheckResult(
            name="val_loss_plateau",
            status=CheckStatus.FAILED,
            passed=False,
            value=val_loss,
            threshold=min_delta,
            message=f"VAL PLATEAU: No improvement > {min_delta} for {patience} epochs (best={best:.6f}, current={current:.6f})",
            details={
                "current": current,
                "best": best,
                "improvement": improvement,
                "patience": patience,
                "history": list(history),
            },
        )

    return CheckResult(
        name="val_loss_plateau",
        status=CheckStatus.PASSED,
        passed=True,
        value=val_loss,
        message=f"Val loss OK: {val_loss:.6f} (best={best:.6f}, imp={best - val_loss:.6f})",
    )


def check_loss_divergence(context: CheckContext) -> CheckResult:
    """Check for loss divergence (sudden spikes)."""
    config = context.config
    spike_factor = config.get("loss_spike_factor", 5.0)
    window = config.get("loss_spike_window", 5)

    if context.metrics:
        loss = context.metrics.get("train_loss") or context.metrics.get("loss")
    else:
        loss = None

    if loss is None and context.loss is not None:
        loss = context.loss.item() if hasattr(context.loss, "item") else float(context.loss)

    if loss is None:
        return CheckResult(
            name="loss_divergence",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No loss available",
        )

    key = f"{context.run_id}:{context.model_name}:recent"
    if key not in _loss_history:
        _loss_history[key] = deque(maxlen=window)
    _loss_history[key].append(loss)

    history = _loss_history[key]
    if len(history) < 3:
        return CheckResult(
            name="loss_divergence",
            status=CheckStatus.PASSED,
            passed=True,
            message="Building recent history",
            value=loss,
        )

    # Check if current loss is spike_factor times the median of recent
    recent = list(history)[-window:]
    median = sorted(recent)[len(recent) // 2]

    if median > 0 and loss > median * spike_factor:
        return CheckResult(
            name="loss_divergence",
            status=CheckStatus.FAILED,
            passed=False,
            value=loss,
            threshold=median * spike_factor,
            message=f"LOSS SPIKE: {loss:.6f} > {spike_factor}x median ({median:.6f})",
            details={
                "current": loss,
                "median": median,
                "spike_factor": spike_factor,
                "recent": recent,
            },
        )

    return CheckResult(
        name="loss_divergence",
        status=CheckStatus.PASSED,
        passed=True,
        value=loss,
        message=f"Loss stable: {loss:.6f} (median={median:.6f})",
    )


# Register loss checks

register_check(
    name="loss_plateau",
    phase="epoch",
    func=check_loss_plateau,
    description="Detect training loss plateau",
    severity="warning",
    tags={"loss", "plateau", "convergence"},
    threshold={
        "patience": 6,
        "min_delta": 1e-4,
        "window": 10,
    },
    dependencies=["loss_nan"],
)

register_check(
    name="val_loss_plateau",
    phase="validation",
    func=check_val_loss_plateau,
    description="Detect validation loss plateau",
    severity="warning",
    tags={"loss", "validation", "plateau"},
    threshold={
        "patience": 10,
        "min_delta": 1e-4,
    },
)

register_check(
    name="loss_divergence",
    phase="batch",
    func=check_loss_divergence,
    description="Detect sudden loss spikes",
    severity="error",
    tags={"loss", "spike", "divergence"},
    threshold={
        "spike_factor": 5.0,
        "window": 5,
    },
)
