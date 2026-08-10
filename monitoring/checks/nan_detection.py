"""
NaN/Inf Detection Checks.

Detects NaN and Inf values in tensors during training.
"""

import torch
import numpy as np
from typing import Any

from monitoring.checks import CheckContext, CheckResult, CheckStatus, register_check


def check_batch_nan(context: CheckContext) -> CheckResult:
    """Check for NaN/Inf in batch inputs and targets."""
    issues = []
    
    if context.batch_data is not None:
        data = context.batch_data
        if isinstance(data, torch.Tensor):
            if torch.isnan(data).any():
                issues.append("NaN in batch_data")
            if torch.isinf(data).any():
                issues.append("Inf in batch_data")
        elif isinstance(data, np.ndarray):
            if np.isnan(data).any():
                issues.append("NaN in batch_data")
            if np.isinf(data).any():
                issues.append("Inf in batch_data")
    
    if context.batch_targets is not None:
        targets = context.batch_targets
        if isinstance(targets, torch.Tensor):
            if torch.isnan(targets).any():
                issues.append("NaN in batch_targets")
            if torch.isinf(targets).any():
                issues.append("Inf in batch_targets")
        elif isinstance(targets, np.ndarray):
            if np.isnan(targets).any():
                issues.append("NaN in batch_targets")
            if np.isinf(targets).any():
                issues.append("Inf in batch_targets")
    
    passed = len(issues) == 0
    return CheckResult(
        name="batch_nan",
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        passed=passed,
        message="; ".join(issues) if issues else "No NaN/Inf in batch",
        details={"issues": issues},
    )


def check_model_output_nan(context: CheckContext) -> CheckResult:
    """Check for NaN/Inf in model outputs."""
    if context.outputs is None:
        return CheckResult(
            name="output_nan",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No outputs to check",
        )
    
    outputs = context.outputs
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    
    issues = []
    if isinstance(outputs, torch.Tensor):
        if torch.isnan(outputs).any():
            issues.append("NaN in model outputs")
        if torch.isinf(outputs).any():
            issues.append("Inf in model outputs")
    
    passed = len(issues) == 0
    return CheckResult(
        name="output_nan",
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        passed=passed,
        message="; ".join(issues) if issues else "No NaN/Inf in outputs",
        details={"issues": issues},
    )


def check_loss_nan(context: CheckContext) -> CheckResult:
    """Check for NaN/Inf in loss value."""
    if context.loss is None:
        return CheckResult(
            name="loss_nan",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No loss to check",
        )
    
    loss = context.loss
    if isinstance(loss, torch.Tensor):
        loss_val = loss.item()
    else:
        loss_val = float(loss)
    
    issues = []
    if np.isnan(loss_val):
        issues.append(f"Loss is NaN: {loss_val}")
    elif np.isinf(loss_val):
        issues.append(f"Loss is Inf: {loss_val}")
    
    passed = len(issues) == 0
    return CheckResult(
        name="loss_nan",
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        passed=passed,
        value=loss_val,
        message="; ".join(issues) if issues else f"Loss OK: {loss_val:.6f}",
        details={"loss": loss_val, "issues": issues},
    )


def check_gradient_nan(context: CheckContext) -> CheckResult:
    """Check for NaN/Inf in gradients."""
    if context.model is None:
        return CheckResult(
            name="grad_nan",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No model to check gradients",
        )
    
    issues = []
    nan_params = 0
    inf_params = 0
    
    for name, param in context.model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append(f"NaN gradient in {name}")
                nan_params += 1
            if torch.isinf(param.grad).any():
                issues.append(f"Inf gradient in {name}")
                inf_params += 1
    
    passed = len(issues) == 0
    return CheckResult(
        name="grad_nan",
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        passed=passed,
        message=f"NaN params: {nan_params}, Inf params: {inf_params}" if issues else "No NaN/Inf gradients",
        details={"issues": issues, "nan_params": nan_params, "inf_params": inf_params},
    )


# Register all NaN checks
register_check(
    name="batch_nan",
    phase="batch",
    func=check_batch_nan,
    description="Check for NaN/Inf in batch inputs and targets",
    severity="error",
    tags={"nan", "batch", "data"},
)

register_check(
    name="output_nan",
    phase="batch",
    func=check_model_output_nan,
    description="Check for NaN/Inf in model outputs",
    severity="error",
    tags={"nan", "output", "model"},
)

register_check(
    name="loss_nan",
    phase="batch",
    func=check_loss_nan,
    description="Check for NaN/Inf in loss value",
    severity="error",
    tags={"nan", "loss"},
)

register_check(
    name="grad_nan",
    phase="batch",
    func=check_gradient_nan,
    description="Check for NaN/Inf in gradients",
    severity="error",
    tags={"nan", "gradient", "backward"},
)