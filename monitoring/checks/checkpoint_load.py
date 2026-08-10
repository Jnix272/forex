"""
Checkpoint Validation Checks.

Validates model checkpoint loading and compatibility.
"""

import torch
import torch.nn as nn
from typing import Any

from monitoring.checks import CheckContext, CheckResult, CheckStatus, register_check


def check_checkpoint_load(context: CheckContext) -> CheckResult:
    """Validate checkpoint loading - checks param loading fraction."""
    config = context.config
    min_frac_loaded = config.get("min_frac_loaded", 0.6)
    
    # This check is typically run after loading a checkpoint
    # We need the load report from the loading process
    load_report = context.metrics.get("load_report") if context.metrics else None
    
    if load_report is None:
        # Try to get from extra
        load_report = context.extra.get("load_report")
    
    if load_report is None:
        return CheckResult(
            name="checkpoint_load",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No load report available",
        )
    
    frac_loaded = load_report.get("frac_loaded", 0.0)
    n_loaded = load_report.get("n_loaded", 0)
    n_target = load_report.get("n_target", 0)
    missing = load_report.get("missing", [])
    unexpected = load_report.get("unexpected", [])
    shape_mismatch = load_report.get("shape_mismatch", [])
    
    passed = frac_loaded >= min_frac_loaded
    
    issues = []
    if not passed:
        issues.append(f"Only {frac_loaded:.1%} of params loaded (min: {min_frac_loaded:.0%})")
    if missing:
        issues.append(f"Missing keys: {len(missing)}")
    if unexpected:
        issues.append(f"Unexpected keys: {len(unexpected)}")
    if shape_mismatch:
        issues.append(f"Shape mismatches: {len(shape_mismatch)}")
    
    return CheckResult(
        name="checkpoint_load",
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        passed=passed,
        value=frac_loaded,
        threshold=min_frac_loaded,
        message=f"Loaded {frac_loaded:.1%} params ({n_loaded}/{n_target})" + (f" - {'; '.join(issues)}" if issues else " OK"),
        details={
            "frac_loaded": frac_loaded,
            "n_loaded": n_loaded,
            "n_target": n_target,
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "shape_mismatch_count": len(shape_mismatch),
            "missing_sample": missing[:5] if missing else [],
            "unexpected_sample": unexpected[:5] if unexpected else [],
            "shape_mismatch_sample": shape_mismatch[:5] if shape_mismatch else [],
        },
    )


def check_model_output_shape(context: CheckContext) -> CheckResult:
    """Validate model output shape matches expectations."""
    if context.model is None or context.batch_data is None:
        return CheckResult(
            name="output_shape",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No model or batch data",
        )
    
    config = context.config
    expected_shape = config.get("expected_output_shape")
    expected_classes = config.get("num_classes")
    
    try:
        with torch.no_grad():
            # Forward pass on a small batch
            sample = context.batch_data[:2] if len(context.batch_data) >= 2 else context.batch_data
            output = context.model(sample)
            
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            actual_shape = list(output.shape)
            
            # Check basic shape
            if expected_shape:
                expected = list(expected_shape)
                # Replace batch dim with -1
                expected[0] = -1
                actual_shape[0] = -1
                
                if actual_shape != expected:
                    return CheckResult(
                        name="output_shape",
                        status=CheckStatus.FAILED,
                        passed=False,
                        message=f"Output shape mismatch: got {actual_shape}, expected {expected}",
                        details={"actual": list(output.shape), "expected": expected_shape},
                    )
            
            # Check num_classes if specified
            if expected_classes and output.dim() >= 2:
                actual_classes = output.shape[-1] if output.shape[-1] > 1 else 1
                if actual_classes != expected_classes:
                    return CheckResult(
                        name="output_shape",
                        status=CheckStatus.FAILED,
                        passed=False,
                        message=f"Num classes mismatch: got {actual_classes}, expected {expected_classes}",
                        details={"actual_classes": actual_classes, "expected_classes": expected_classes},
                    )
            
            return CheckResult(
                name="output_shape",
                status=CheckStatus.PASSED,
                passed=True,
                message=f"Output shape OK: {list(output.shape)}",
                details={"output_shape": list(output.shape)},
            )
    
    except Exception as e:
        return CheckResult(
            name="output_shape",
            status=CheckStatus.ERROR,
            passed=False,
            message=f"Forward pass failed: {e}",
            details={"error": str(e)},
        )


def check_optimizer_state(context: CheckContext) -> CheckResult:
    """Validate optimizer state after checkpoint load."""
    if context.optimizer is None:
        return CheckResult(
            name="optimizer_state",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No optimizer to check",
        )
    
    issues = []
    
    # Determine expected state keys based on optimizer type
    optimizer_class = context.optimizer.__class__.__name__.lower()
    if 'adam' in optimizer_class:
        expected_keys = ['exp_avg', 'exp_avg_sq']
    elif 'sgd' in optimizer_class:
        expected_keys = ['momentum_buffer']
    elif 'rmsprop' in optimizer_class:
        expected_keys = ['square_avg', 'momentum_buffer']  # RMSprop can have both
    else:
        # Unknown optimizer - accept any state keys
        expected_keys = []
    
    # Check if optimizer has state
    if hasattr(context.optimizer, 'state'):
        state = context.optimizer.state
        if len(state) == 0:
            issues.append("Optimizer state is empty")
        else:
            # Check a few parameters have expected state keys
            sample_params = list(state.keys())[:5]
            for p in sample_params:
                param_state = state[p]
                if expected_keys:
                    # At least one expected key should be present
                    if not any(key in param_state for key in expected_keys):
                        issues.append(f"Missing expected optimizer state keys ({expected_keys}) for param {p}")
    
    # Check LR matches
    if context.config.get("expected_lr"):
        expected_lr = context.config["expected_lr"]
        for i, group in enumerate(context.optimizer.param_groups):
            actual_lr = group.get('lr', 0)
            if abs(actual_lr - expected_lr) > 1e-8:
                issues.append(f"LR mismatch group {i}: {actual_lr} vs expected {expected_lr}")
    
    passed = len(issues) == 0
    
    return CheckResult(
        name="optimizer_state",
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        passed=passed,
        message="Optimizer state OK" if passed else "; ".join(issues),
        details={"issues": issues, "optimizer_type": optimizer_class},
    )


# Register checkpoint checks
from monitoring.checks import register_check

register_check(
    name="checkpoint_load",
    phase="checkpoint",
    func=check_checkpoint_load,
    description="Validate checkpoint loading fraction and integrity",
    severity="error",
    tags={"checkpoint", "load", "integrity"},
    threshold={
        "min_frac_loaded": 0.6,
    },
)

register_check(
    name="output_shape",
    phase="checkpoint",
    func=check_model_output_shape,
    description="Validate model output shape matches expectations",
    severity="error",
    tags={"checkpoint", "shape", "forward"},
    dependencies=["checkpoint_load"],
)

register_check(
    name="optimizer_state",
    phase="checkpoint",
    func=check_optimizer_state,
    description="Validate optimizer state after checkpoint load",
    severity="warning",
    tags={"checkpoint", "optimizer", "state"},
    dependencies=["checkpoint_load"],
)