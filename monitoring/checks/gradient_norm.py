"""
Gradient Norm Checks.

Monitors gradient norms for explosion/vanishing detection.
"""

from monitoring.checks import CheckContext, CheckResult, CheckStatus, register_check


def check_gradient_norm(context: CheckContext) -> CheckResult:
    """Check gradient norms against thresholds."""
    if context.model is None:
        return CheckResult(
            name="grad_norm",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No model to check gradients",
        )

    # Get thresholds from context config or use defaults
    config = context.config
    warn_threshold = config.get("grad_norm_warn", 10.0)
    crit_threshold = config.get("grad_norm_crit", 50.0)
    vanish_threshold = config.get("grad_norm_vanish", 1e-6)

    total_norm = 0.0
    max_norm = 0.0
    min_norm = float("inf")
    param_count = 0

    for _name, param in context.model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
            max_norm = max(max_norm, param_norm)
            min_norm = min(min_norm, param_norm)
            param_count += 1

    if param_count == 0:
        return CheckResult(
            name="grad_norm",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No gradients to check",
        )

    total_norm = total_norm**0.5

    # Compute meaningful average: mean of individual parameter gradient norms
    # (not total L2 norm / count, which is not a standard metric)
    avg_norm = 0.0
    for _name, param in context.model.named_parameters():
        if param.grad is not None:
            avg_norm += param.grad.data.norm(2).item()
    avg_norm = avg_norm / param_count if param_count > 0 else 0.0

    issues = []
    if total_norm > crit_threshold:
        issues.append(f"CRITICAL: Total grad norm {total_norm:.4f} > {crit_threshold}")
    elif total_norm > warn_threshold:
        issues.append(f"WARNING: Total grad norm {total_norm:.4f} > {warn_threshold}")

    if min_norm < vanish_threshold:
        issues.append(f"Vanishing gradient: min norm {min_norm:.2e} < {vanish_threshold}")

    passed = len([i for i in issues if i.startswith("CRITICAL")]) == 0

    return CheckResult(
        name="grad_norm",
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        passed=passed,
        value=total_norm,
        threshold=crit_threshold,
        message="; ".join(issues)
        if issues
        else f"Grad norm OK: total={total_norm:.4f}, max={max_norm:.4f}, min={min_norm:.2e}",
        details={
            "total_norm": total_norm,
            "max_norm": max_norm,
            "min_norm": min_norm,
            "avg_norm": avg_norm,
            "param_count": param_count,
            "warn_threshold": warn_threshold,
            "crit_threshold": crit_threshold,
            "vanish_threshold": vanish_threshold,
        },
    )


def check_gradient_norm_per_layer(context: CheckContext) -> CheckResult:
    """Check gradient norms per layer for detailed analysis."""
    if context.model is None:
        return CheckResult(
            name="grad_norm_per_layer",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No model to check gradients",
        )

    config = context.config
    warn_threshold = config.get("grad_norm_per_layer_warn", 5.0)

    layer_norms = {}
    issues = []

    for name, param in context.model.named_parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2).item()
            layer_norms[name] = norm
            if norm > warn_threshold:
                issues.append(f"Layer {name}: grad norm {norm:.4f} > {warn_threshold}")

    passed = len(issues) == 0

    # Get top 5 layers by norm
    top_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)[:5]

    return CheckResult(
        name="grad_norm_per_layer",
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        passed=passed,
        message="; ".join(issues)
        if issues
        else f"All layer grad norms OK (max: {max(layer_norms.values()) if layer_norms else 0:.4f})",
        details={
            "layer_norms": layer_norms,
            "top_layers": top_layers,
            "issues": issues,
        },
    )


# Register gradient norm checks

register_check(
    name="grad_norm",
    phase="batch",
    func=check_gradient_norm,
    description="Check total gradient norm against explosion/vanishing thresholds",
    severity="warning",
    tags={"gradient", "norm", "explosion", "vanishing"},
    threshold={
        "grad_norm_warn": 10.0,
        "grad_norm_crit": 50.0,
        "grad_norm_vanish": 1e-6,
    },
)

register_check(
    name="grad_norm_per_layer",
    phase="batch",
    func=check_gradient_norm_per_layer,
    description="Check per-layer gradient norms",
    severity="info",
    tags={"gradient", "norm", "layer"},
    threshold={
        "warn": 5.0,
    },
)
