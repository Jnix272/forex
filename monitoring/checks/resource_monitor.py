"""
Resource Monitoring Checks.

Monitors GPU/CPU memory, temperature, disk I/O, and other resources.
"""

import psutil
import torch

from monitoring.checks import CheckContext, CheckResult, CheckStatus, register_check


def check_gpu_memory(context: CheckContext) -> CheckResult:
    """Check GPU memory usage."""
    if not torch.cuda.is_available():
        return CheckResult(
            name="gpu_memory",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="CUDA not available",
        )

    config = context.config
    warn_frac = config.get("gpu_mem_warn_frac", 0.85)
    crit_frac = config.get("gpu_mem_crit_frac", 0.95)

    issues = []
    max_frac = 0.0

    for i in range(torch.cuda.device_count()):
        try:
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = torch.cuda.get_device_properties(i).total_memory

            alloc_frac = allocated / total
            reserv_frac = reserved / total
            max_frac = max(max_frac, alloc_frac, reserv_frac)

            if alloc_frac > crit_frac:
                issues.append(
                    f"GPU {i} CRITICAL: {alloc_frac:.1%} allocated ({allocated / 1e9:.1f}GB/{total / 1e9:.1f}GB)"
                )
            elif alloc_frac > warn_frac:
                issues.append(
                    f"GPU {i} WARNING: {alloc_frac:.1%} allocated ({allocated / 1e9:.1f}GB/{total / 1e9:.1f}GB)"
                )

            if reserved > total * 0.98:
                issues.append(f"GPU {i} reserved near limit: {reserved / 1e9:.1f}GB/{total / 1e9:.1f}GB")
        except Exception as e:
            issues.append(f"GPU {i} check error: {e}")

    if not issues:
        return CheckResult(
            name="gpu_memory",
            status=CheckStatus.PASSED,
            passed=True,
            value=max_frac,
            message=f"GPU memory OK: max {max_frac:.1%} allocated",
            details={"max_alloc_frac": max_frac},
        )

    has_critical = any("CRITICAL" in i for i in issues)
    return CheckResult(
        name="gpu_memory",
        status=CheckStatus.FAILED if has_critical else CheckStatus.FAILED,  # noqa: RUF034
        passed=False,
        value=max_frac,
        threshold=crit_frac,
        message="; ".join(issues),
        details={"issues": issues, "max_frac": max_frac},
    )


def check_gpu_temperature(context: CheckContext) -> CheckResult:
    """Check GPU temperature."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return CheckResult(
            name="gpu_temperature",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="pynvml not available",
        )

    config = context.config
    warn_temp = config.get("gpu_temp_warn", 80)
    crit_temp = config.get("gpu_temp_crit", 85)

    issues = []
    max_temp = 0

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            max_temp = max(max_temp, temp)

            if temp >= crit_temp:
                issues.append(f"GPU {i} CRITICAL: {temp}°C")
            elif temp >= warn_temp:
                issues.append(f"GPU {i} WARNING: {temp}°C")
    except Exception as e:
        issues.append(f"Temperature check error: {e}")

    if not issues:
        return CheckResult(
            name="gpu_temperature",
            status=CheckStatus.PASSED,
            passed=True,
            message=f"GPU temps OK: max {max_temp}°C",
            value=max_temp,
        )

    any("CRITICAL" in i for i in issues)
    return CheckResult(
        name="gpu_temperature",
        status=CheckStatus.FAILED,
        passed=False,
        value=max_temp,
        threshold=crit_temp,
        message="; ".join(issues),
        details={"max_temp": max_temp, "issues": issues},
    )


def check_cpu_memory(context: CheckContext) -> CheckResult:
    """Check CPU memory usage."""
    config = context.config
    warn_frac = config.get("cpu_mem_warn_frac", 0.85)
    crit_frac = config.get("cpu_mem_crit_frac", 0.95)

    mem = psutil.virtual_memory()
    frac = mem.percent / 100.0
    used_gb = mem.used / 1e9
    total_gb = mem.total / 1e9

    if frac >= crit_frac:
        return CheckResult(
            name="cpu_memory",
            status=CheckStatus.FAILED,
            passed=False,
            value=frac,
            threshold=crit_frac,
            message=f"CRITICAL CPU memory: {frac:.1%} ({used_gb:.1f}GB/{total_gb:.1f}GB)",
            details={"used_gb": used_gb, "total_gb": total_gb, "frac": frac},
        )
    elif frac >= warn_frac:
        return CheckResult(
            name="cpu_memory",
            status=CheckStatus.FAILED,
            passed=False,
            value=frac,
            threshold=warn_frac,
            message=f"WARNING CPU memory: {frac:.1%} ({used_gb:.1f}GB/{total_gb:.1f}GB)",
            details={"used_gb": used_gb, "total_gb": total_gb, "frac": frac},
        )

    return CheckResult(
        name="cpu_memory",
        status=CheckStatus.PASSED,
        passed=True,
        value=frac,
        message=f"CPU memory OK: {frac:.1%} ({used_gb:.1f}GB/{total_gb:.1f}GB)",
    )


def check_disk_space(context: CheckContext) -> CheckResult:
    """Check disk space for logs/checkpoints."""
    config = context.config
    warn_frac = config.get("disk_warn_frac", 0.85)
    crit_frac = config.get("disk_crit_frac", 0.95)
    paths = config.get("disk_paths", ["logs", "checkpoints", "data"])

    issues = []
    worst_frac = 0.0

    for path in paths:
        try:
            usage = psutil.disk_usage(path)
            frac = usage.used / usage.total
            worst_frac = max(worst_frac, usage.used / usage.total)

            if frac >= crit_frac:
                issues.append(f"CRITICAL disk {path}: {usage.used / 1e9:.1f}GB/{usage.total / 1e9:.1f}GB ({frac:.1%})")
            elif frac >= warn_frac:
                issues.append(f"WARNING disk {path}: {usage.used / 1e9:.1f}GB/{usage.total / 1e9:.1f}GB ({frac:.1%})")
        except Exception:
            pass  # Path might not exist

    if not issues:
        return CheckResult(
            name="disk_space",
            status=CheckStatus.PASSED,
            passed=True,
            message="Disk space OK",
        )

    any("CRITICAL" in i for i in issues)
    return CheckResult(
        name="disk_space",
        status=CheckStatus.FAILED,
        passed=False,
        message="; ".join(issues),
        details={"issues": issues},
    )


def check_dataloader_latency(context: CheckContext) -> CheckResult:
    """Check DataLoader latency (time between batches)."""
    config = context.config
    warn_ms = config.get("dataloader_warn_ms", 1000)
    crit_ms = config.get("dataloader_crit_ms", 5000)

    # Get timing from context extra
    elapsed_ms = context.extra.get("dataloader_elapsed_ms")
    if elapsed_ms is None:
        return CheckResult(
            name="dataloader_latency",
            status=CheckStatus.SKIPPED,
            passed=True,
            message="No timing data",
        )

    if elapsed_ms >= crit_ms:
        return CheckResult(
            name="dataloader_latency",
            status=CheckStatus.FAILED,
            passed=False,
            value=elapsed_ms,
            threshold=crit_ms,
            message=f"CRITICAL DataLoader latency: {elapsed_ms:.0f}ms",
            details={"elapsed_ms": elapsed_ms},
        )
    elif elapsed_ms >= warn_ms:
        return CheckResult(
            name="dataloader_latency",
            status=CheckStatus.FAILED,
            passed=False,
            value=elapsed_ms,
            threshold=warn_ms,
            message=f"WARNING DataLoader latency: {elapsed_ms:.0f}ms",
            details={"elapsed_ms": elapsed_ms},
        )

    return CheckResult(
        name="dataloader_latency",
        status=CheckStatus.PASSED,
        passed=True,
        value=elapsed_ms,
        message=f"DataLoader OK: {elapsed_ms:.0f}ms",
    )


# Register resource checks

register_check(
    name="gpu_memory",
    phase="batch",
    func=check_gpu_memory,
    description="Monitor GPU memory allocation",
    severity="warning",
    tags={"resource", "gpu", "memory"},
    threshold={
        "warn_frac": 0.85,
        "crit_frac": 0.95,
    },
)

register_check(
    name="gpu_temperature",
    phase="batch",
    func=check_gpu_temperature,
    description="Monitor GPU temperature",
    severity="warning",
    tags={"resource", "gpu", "temperature"},
    threshold={
        "warn_temp": 80,
        "crit_temp": 85,
    },
)

register_check(
    name="cpu_memory",
    phase="epoch",
    func=check_cpu_memory,
    description="Monitor CPU memory usage",
    severity="warning",
    tags={"resource", "cpu", "memory"},
    threshold={
        "warn_frac": 0.85,
        "crit_frac": 0.95,
    },
)

register_check(
    name="disk_space",
    phase="epoch",
    func=check_disk_space,
    description="Monitor disk space for logs/checkpoints",
    severity="warning",
    tags={"resource", "disk", "storage"},
    threshold={
        "warn_frac": 0.85,
        "crit_frac": 0.95,
        "disk_paths": ["logs", "checkpoints", "data"],
    },
)

register_check(
    name="dataloader_latency",
    phase="batch",
    func=check_dataloader_latency,
    description="Monitor DataLoader latency between batches",
    severity="warning",
    tags={"resource", "dataloader", "latency", "performance"},
    threshold={
        "warn_ms": 1000,
        "crit_ms": 5000,
    },
)
