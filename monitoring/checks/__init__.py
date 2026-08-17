"""
Check Registry - Central registry for all validation checks.

Enables dynamic enable/disable, configuration, and execution of checks.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone  # noqa: F401
from enum import Enum, StrEnum  # noqa: F401
from typing import Any, Optional  # noqa: F401

from monitoring.events import CheckPhase, EventType, Severity, TrainingEvent  # noqa: F401


class CheckStatus(StrEnum):
    """Check execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a single check execution."""

    name: str
    status: CheckStatus
    passed: bool
    message: str = ""
    details: dict = field(default_factory=dict)
    threshold: float | None = None
    value: float | None = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON output."""
        return {
            "name": self.name,
            "status": self.status.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "threshold": self.threshold,
            "value": self.value,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class CheckMetadata:
    """Metadata for a registered check."""

    name: str
    phase: CheckPhase
    func: Callable
    description: str = ""
    severity: str = "warning"
    tags: set[str] = field(default_factory=set)
    threshold: dict | None = None  # Configurable thresholds
    enabled: bool = True
    async_fn: bool = False
    dependencies: list[str] = field(default_factory=list)  # Other checks that must run first


class CheckContext:
    """Context passed to check functions."""

    def __init__(
        self,
        run_id: str = "",
        session_id: str = "",
        epoch: int | None = None,
        batch: int | None = None,
        model_name: str = "",
        model: Any = None,
        optimizer: Any = None,
        loss: Any = None,
        batch_data: Any = None,
        batch_targets: Any = None,
        outputs: Any = None,
        gradients: Any = None,
        metrics: dict | None = None,
        history: dict | None = None,
        config: dict | None = None,
        **kwargs,
    ):
        self.run_id = run_id
        self.session_id = session_id
        self.epoch = epoch
        self.batch = batch
        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.batch_data = batch_data
        self.batch_targets = batch_targets
        self.outputs = outputs
        self.gradients = gradients
        self.metrics = metrics or {}
        self.history = history or {}
        self.config = config or {}
        self.extra = kwargs

        # Results from previous checks in this phase
        self.check_results: dict[str, CheckResult] = {}


class CheckRegistry:
    """Central registry for all validation checks."""

    def __init__(self):
        self._checks: dict[str, CheckMetadata] = {}
        self._phase_order = [
            CheckPhase.PREFLIGHT,
            CheckPhase.BATCH,
            CheckPhase.EPOCH,
            CheckPhase.VALIDATION,
            CheckPhase.CHECKPOINT,
            CheckPhase.PRETRAIN,
            CheckPhase.RL_EPISODE,
            CheckPhase.POST_TRAIN,
        ]

    def register(
        self,
        name: str,
        phase: CheckPhase | str,
        func: Callable,
        description: str = "",
        severity: str = "warning",
        tags: set[str] | None = None,
        threshold: dict | None = None,
        enabled: bool = True,
        dependencies: list[str] | None = None,
    ) -> CheckMetadata:
        """Register a check function."""
        # Convert string phase to enum
        if isinstance(phase, str):
            try:
                phase = CheckPhase(phase)
            except ValueError:
                raise ValueError(f"Invalid phase: {phase}. Valid: {[p.value for p in CheckPhase]}")
        if name in self._checks:
            raise ValueError(f"Check '{name}' already registered")

        metadata = CheckMetadata(
            name=name,
            phase=phase,
            func=func,
            description=description,
            severity=severity,
            tags=tags or set(),
            threshold=threshold or {},
            enabled=enabled,
            async_fn=asyncio.iscoroutinefunction(func),
            dependencies=dependencies or [],
        )

        self._checks[name] = metadata
        return metadata

    def unregister(self, name: str):
        """Unregister a check."""
        if name in self._checks:
            del self._checks[name]

    def get(self, name: str) -> CheckMetadata | None:
        """Get check metadata by name."""
        return self._checks.get(name)

    def get_phase_checks(self, phase: CheckPhase) -> list[CheckMetadata]:
        """Get all checks for a phase, sorted by dependencies."""
        checks = [c for c in self._checks.values() if c.phase == phase and c.enabled]
        # Simple topological sort by dependencies
        return self._topological_sort(checks)

    def _topological_sort(self, checks: list[CheckMetadata]) -> list[CheckMetadata]:
        """Sort checks by dependencies (simple Kahn's algorithm)."""
        # Build adjacency
        graph = {c.name: set(c.dependencies) for c in checks}
        name_to_check = {c.name: c for c in checks}

        # Find nodes with no dependencies
        in_degree = {name: len(deps) for name, deps in graph.items()}
        queue = [name for name, deg in in_degree.items() if deg == 0]

        sorted_checks = []
        while queue:
            name = queue.pop(0)
            sorted_checks.append(name_to_check[name])
            # Remove edges
            for other_name, deps in graph.items():
                if name in deps:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0:
                        queue.append(other_name)

        # If cycle detected, fall back to original order
        if len(sorted_checks) != len(checks):
            return checks

        return sorted_checks

    def list_all(self) -> list[CheckMetadata]:
        """List all registered checks."""
        return list(self._checks.values())

    def enable(self, name: str):
        """Enable a check."""
        if name in self._checks:
            self._checks[name].enabled = True

    def disable(self, name: str):
        """Disable a check."""
        if name in self._checks:
            self._checks[name].enabled = False

    def set_threshold(self, name: str, threshold: dict):
        """Update check threshold."""
        if name in self._checks:
            self._checks[name].threshold = threshold


class CheckEngine:
    """Orchestrates check execution for each phase."""

    def __init__(self, registry: CheckRegistry, logger=None):
        self.registry = registry
        self.logger = logger
        self._results: dict[str, list[CheckResult]] = defaultdict(list)
        # Store all results across phases for cross-phase dependency resolution
        self._all_results: dict[str, CheckResult] = {}
        self._running = False

    async def run_phase(
        self,
        phase: CheckPhase | str,
        context: CheckContext,
    ) -> dict[str, CheckResult]:
        """Run all checks for a phase."""
        # Convert string phase to enum
        if isinstance(phase, str):
            try:
                phase = CheckPhase(phase)
            except ValueError:
                raise ValueError(f"Invalid phase: {phase}. Valid: {[p.value for p in CheckPhase]}")

        # Initialize context.check_results with all previous phase results for cross-phase deps
        context.check_results = dict(self._all_results)

        checks = self.registry.get_phase_checks(phase)
        results = {}

        for check_meta in checks:
            if not check_meta.enabled:
                continue

            # Check dependencies (can now reference checks from previous phases)
            deps_ok = all(
                context.check_results.get(dep, CheckResult("", CheckStatus.PENDING, False)).passed
                for dep in check_meta.dependencies
            )
            if not deps_ok:
                result = CheckResult(
                    name=check_meta.name,
                    status=CheckStatus.SKIPPED,
                    passed=False,
                    message=f"Dependencies not met: {check_meta.dependencies}",
                )
                results[check_meta.name] = result
                context.check_results[check_meta.name] = result
                continue

            # Propagate threshold config from CheckMetadata to context.config
            # This allows checks to read thresholds via context.config.get("key")
            if check_meta.threshold:
                # Merge check-specific thresholds into context.config
                context.config.update(check_meta.threshold)

            # Run check
            try:
                if check_meta.async_fn:
                    result = await check_meta.func(context)
                else:
                    result = check_meta.func(context)

                if not isinstance(result, CheckResult):
                    # Convert bool/None to CheckResult
                    passed = bool(result) if result is not None else False
                    result = CheckResult(
                        name=check_meta.name,
                        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
                        passed=passed,
                    )

                # Ensure required fields
                result.name = check_meta.name
                if result.status == CheckStatus.PENDING:
                    result.status = CheckStatus.PASSED if result.passed else CheckStatus.FAILED

            except Exception as e:
                result = CheckResult(
                    name=check_meta.name,
                    status=CheckStatus.ERROR,
                    passed=False,
                    message=f"Check error: {e}",
                )

            results[check_meta.name] = result
            context.check_results[check_meta.name] = result
            # Also store in global all_results for cross-phase access
            self._all_results[check_meta.name] = result

            # Log result
            if self.logger:
                self.logger.check(
                    name=check_meta.name,
                    passed=result.passed,
                    value=result.value,
                    threshold=result.threshold,
                    details=result.details,
                    source=f"checks.{check_meta.name}",
                    epoch=context.epoch,
                    batch=context.batch,
                    model_name=context.model_name,
                )

        self._results[phase.value].extend(results.values())
        return results

    def get_results(self, phase: CheckPhase | None = None) -> dict[str, list[CheckResult]]:
        """Get check results."""
        if phase:
            return {phase.value: self._results.get(phase.value, [])}
        return dict(self._results)

    def get_summary(self) -> dict:
        """Get summary of all check results."""
        summary = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        for results in self._results.values():
            for r in results:
                summary["total"] += 1
                if r.status == CheckStatus.PASSED:
                    summary["passed"] += 1
                elif r.status == CheckStatus.FAILED:
                    summary["failed"] += 1
                elif r.status == CheckStatus.ERROR:
                    summary["errors"] += 1
                elif r.status == CheckStatus.SKIPPED:
                    summary["skipped"] += 1
        return summary


# Global registry
_global_registry: CheckRegistry | None = None
_global_engine: CheckEngine | None = None


def get_registry() -> CheckRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = CheckRegistry()
    return _global_registry


def get_engine(logger=None) -> CheckEngine:
    global _global_engine
    if _global_engine is None:
        _global_engine = CheckEngine(get_registry(), logger)
    return _global_engine


def register_check(*args, **kwargs) -> CheckMetadata:
    """Convenience function to register a check."""
    return get_registry().register(*args, **kwargs)
