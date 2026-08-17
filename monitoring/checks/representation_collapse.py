"""
Representation Collapse Checks.

Detects embedding collapse in contrastive/pretraining models.
"""

import torch

from monitoring.checks import CheckContext, CheckResult, CheckStatus, register_check


def check_embedding_std(context: CheckContext) -> CheckResult:
    """Check embedding standard deviation to detect collapse."""
    config = context.config
    min_std = config.get("embedding_min_std", 0.01)
    max_std = config.get("embedding_max_std", 10.0)

    # Try to get embeddings from model
    embeddings = None

    # Try common attribute names
    if context.model is not None:
        if hasattr(context.model, "get_embeddings"):
            embeddings = context.model.get_embeddings()
        elif hasattr(context.model, "embeddings"):
            embeddings = context.model.embeddings
        elif hasattr(context.model, "encoder"):
            # Try to get encoder output
            pass

    # Fallback: check metrics for embedding stats
    if embeddings is None and context.metrics:
        embed_std = context.metrics.get("embedding_std") or context.metrics.get("embed_std")
        if embed_std is not None:
            return CheckResult(
                name="embedding_std",
                status=CheckStatus.PASSED if min_std <= embed_std <= max_std else CheckStatus.FAILED,
                passed=min_std <= embed_std <= max_std,
                value=embed_std,
                threshold=min_std,
                message=f"Embedding std: {embed_std:.6f} (range: [{min_std}, {max_std}])",
                details={"embed_std": embed_std, "min_std": min_std, "max_std": max_std},
            )

    if embeddings is not None:  # noqa: SIM102
        if isinstance(embeddings, torch.Tensor):
            # Flatten to 2D: (batch * seq, dim)
            if embeddings.dim() > 2:
                embeddings = embeddings.view(-1, embeddings.size(-1))

            std_per_dim = embeddings.std(dim=0)
            mean_std = std_per_dim.mean().item()
            min_dim_std = std_per_dim.min().item()
            max_dim_std = std_per_dim.max().item()

            collapsed_dims = (std_per_dim < min_std).sum().item()
            total_dims = std_per_dim.numel()

            passed = mean_std >= min_std and mean_std <= max_std and collapsed_dims == 0

            return CheckResult(
                name="embedding_std",
                status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
                passed=passed,
                value=mean_std,
                threshold=min_std,
                message=f"Embedding std: mean={mean_std:.6f}, min_dim={min_dim_std:.6f}, max_dim={max_dim_std:.6f}, collapsed_dims={collapsed_dims}/{total_dims}",
                details={
                    "mean_std": mean_std,
                    "min_dim_std": min_dim_std,
                    "max_dim_std": max_dim_std,
                    "collapsed_dims": collapsed_dims,
                    "total_dims": total_dims,
                    "min_std": min_std,
                    "max_std": max_std,
                },
            )

    return CheckResult(
        name="embedding_std",
        status=CheckStatus.SKIPPED,
        passed=True,
        message="No embeddings available to check",
    )


def check_embedding_uniformity(context: CheckContext) -> CheckResult:
    """Check embedding uniformity (for contrastive learning)."""
    config = context.config
    max_uniformity = config.get("max_uniformity", -0.1)  # More negative = more uniform

    embeddings = None
    if context.model is not None and hasattr(context.model, "get_embeddings"):
        embeddings = context.model.get_embeddings()

    if embeddings is None and context.metrics:
        uniformity = context.metrics.get("uniformity") or context.metrics.get("lunif")
        if uniformity is not None:
            passed = uniformity <= max_uniformity  # More negative is better
            return CheckResult(
                name="embedding_uniformity",
                status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
                passed=passed,
                value=uniformity,
                threshold=max_uniformity,
                message=f"Uniformity: {uniformity:.4f} (target: < {max_uniformity})",
                details={"uniformity": uniformity, "max_uniformity": max_uniformity},
            )

    if embeddings is not None and isinstance(embeddings, torch.Tensor):
        if embeddings.dim() > 2:
            embeddings = embeddings.view(-1, embeddings.size(-1))

        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Compute pairwise distances (sample for efficiency)
        n = embeddings.size(0)
        if n > 1000:
            idx = torch.randperm(n)[:1000]
            embeddings = embeddings[idx]
            n = 1000

        # Compute uniformity: log(mean(exp(-||x-y||^2)))
        # Sample pairs using efficient vectorized approach
        pairs = min(50000, n * (n - 1) // 2)

        # Use vectorized sampling instead of O(n²) loop
        num_samples = min(10000, pairs)
        if num_samples > 0:
            # Generate random pairs efficiently using randint
            i = torch.randint(0, n, (num_samples,), device=embeddings.device)
            j = torch.randint(0, n, (num_samples,), device=embeddings.device)
            # Ensure i != j
            mask = i != j
            i = i[mask]
            j = j[mask]
            if len(i) > 0:
                # Take up to num_samples pairs
                take = min(num_samples, len(i))
                i = i[:take]
                j = j[:take]
                # Vectorized distance computation
                diff = embeddings[i] - embeddings[j]
                dist = (diff * diff).sum(dim=1)
                total_exp = torch.exp(-dist).sum().item()
                mean_exp = total_exp / take
                uniformity = torch.log(torch.tensor(mean_exp)).item()

                passed = uniformity <= max_uniformity
                return CheckResult(
                    name="embedding_uniformity",
                    status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
                    passed=passed,
                    value=uniformity,
                    threshold=max_uniformity,
                    message=f"Uniformity: {uniformity:.4f} (target: < {max_uniformity})",
                    details={"uniformity": uniformity, "max_uniformity": max_uniformity, "samples": take},
                )

    return CheckResult(
        name="embedding_uniformity",
        status=CheckStatus.SKIPPED,
        passed=True,
        message="No embeddings available for uniformity check",
    )


def check_embedding_alignment(context: CheckContext) -> CheckResult:
    """Check embedding alignment (for BYOL-style pretraining)."""
    config = context.config
    min_alignment = config.get("min_alignment", 0.0)

    # Check metrics first
    if context.metrics:
        alignment = context.metrics.get("alignment") or context.metrics.get("lalign")
        if alignment is not None:
            passed = alignment >= min_alignment
            return CheckResult(
                name="embedding_alignment",
                status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
                passed=passed,
                value=alignment,
                threshold=min_alignment,
                message=f"Alignment: {alignment:.4f} (target: > {min_alignment})",
                details={"alignment": alignment, "min_alignment": min_alignment},
            )

    return CheckResult(
        name="embedding_alignment",
        status=CheckStatus.SKIPPED,
        passed=True,
        message="No alignment metric available",
    )


def check_representation_collapse(context: CheckContext) -> CheckResult:
    """Comprehensive representation collapse check."""

    # Run sub-checks
    std_result = check_embedding_std(context)
    uniform_result = check_embedding_uniformity(context)
    align_result = check_embedding_alignment(context)

    all_passed = std_result.passed and uniform_result.passed and align_result.passed

    details = {
        "embedding_std": {
            "passed": std_result.passed,
            "value": std_result.value,
            "threshold": std_result.threshold,
        },
        "embedding_uniformity": {
            "passed": uniform_result.passed,
            "value": uniform_result.value,
            "threshold": uniform_result.threshold,
        },
        "embedding_alignment": {
            "passed": align_result.passed,
            "value": align_result.value,
            "threshold": align_result.threshold,
        },
    }

    return CheckResult(
        name="representation_collapse",
        status=CheckStatus.PASSED if all_passed else CheckStatus.FAILED,
        passed=all_passed,
        message=f"Representation: std={std_result.value if std_result.value is not None else 'N/A'}, uniform={uniform_result.value if uniform_result.value is not None else 'N/A'}, align={align_result.value if align_result.value is not None else 'N/A'}",
        details=details,
    )


# Register representation checks

register_check(
    name="embedding_std",
    phase="pretrain",
    func=check_embedding_std,
    description="Check embedding standard deviation to detect collapse",
    severity="error",
    tags={"embedding", "collapse", "std", "pretrain"},
    threshold={
        "min_std": 0.01,
        "max_std": 10.0,
    },
)

register_check(
    name="embedding_uniformity",
    phase="pretrain",
    func=check_embedding_uniformity,
    description="Check embedding uniformity (contrastive)",
    severity="error",
    tags={"embedding", "uniformity", "contrastive", "pretrain"},
    threshold={
        "max_uniformity": -0.1,
    },
)

register_check(
    name="embedding_alignment",
    phase="pretrain",
    func=check_embedding_alignment,
    description="Check embedding alignment (BYOL)",
    severity="warning",
    tags={"embedding", "alignment", "byol", "pretrain"},
    threshold={
        "min_alignment": 0.0,
    },
)

register_check(
    name="representation_collapse",
    phase="pretrain",
    func=check_representation_collapse,
    description="Comprehensive representation collapse detection",
    severity="error",
    tags={"embedding", "collapse", "comprehensive", "pretrain"},
    dependencies=["embedding_std", "embedding_uniformity", "embedding_alignment"],
)
