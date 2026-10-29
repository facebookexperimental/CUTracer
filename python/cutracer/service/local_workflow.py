"""Local diagnosis orchestration, independent of command-line and cloud adapters."""

from __future__ import annotations

import dataclasses
import hashlib
import os
from collections.abc import Callable

from cutracer.service.contracts import (
    ChildDiagnosisReport,
    DEFAULT_MAX_DISCOVERED_KERNELS,
    DiagnosisReport,
    ExperimentKind,
    SanitizerSweepResult,
    SessionReport,
    WorkUnit,
)
from cutracer.service.kernel_discovery import discover_work_units, KernelDiscovery
from cutracer.service.policy import InitialCampaignPolicy
from cutracer.service.runner import (
    ExperimentExecutor,
    FileSessionStore,
    LocalExperimentExecutor,
    LocalRuntimeConfig,
    run_local_session,
)
from cutracer.service.sanitizer_results import is_target_scoped_sanitizer
from cutracer.service.triage import TriagePolicy


@dataclasses.dataclass(frozen=True)
class KernelDiagnosisFailure:
    unit: WorkUnit
    reason: str

    def __str__(self) -> str:
        kernel = self.unit.kernel
        if kernel is None:
            return f"{self.unit.unit_id}: {self.reason}"
        return f"{kernel.name} ({self.unit.unit_id}): {self.reason}"


class PartialDiagnosisError(Exception):
    """Discovered kernels failed; ``report`` aggregates the kernels that finished."""

    def __init__(
        self,
        report: DiagnosisReport,
        failures: tuple[KernelDiagnosisFailure, ...],
    ) -> None:
        total = len(report.children) + len(failures)
        super().__init__(
            f"{len(failures)} of {total} discovered kernels failed: "
            + "; ".join(str(failure) for failure in failures)
        )
        self.report = report
        self.failures = failures


def _target_sanitizers(
    unit: WorkUnit,
    policy: InitialCampaignPolicy,
    executor: ExperimentExecutor,
) -> tuple[SanitizerSweepResult, ...]:
    shared = []
    for spec in policy.plan(f"session-{unit.unit_id}-target-scope", unit).experiments:
        if (
            spec.kind == ExperimentKind.COMPUTE_SANITIZER
            and spec.sanitizer_tool is not None
            and is_target_scoped_sanitizer(spec.sanitizer_tool)
        ):
            result = executor.execute(spec)
            if not isinstance(result, SanitizerSweepResult):
                raise TypeError(
                    "target-scoped sanitizer returned the wrong result type"
                )
            shared.append(result)
    return tuple(shared)


def _compatibility_child(
    child: WorkUnit,
    *,
    runtime: LocalRuntimeConfig,
    state: str | None,
    policy: InitialCampaignPolicy,
    triage: TriagePolicy,
    run_local_fn: Callable[..., SessionReport],
    executor_factory: Callable[[LocalRuntimeConfig], ExperimentExecutor],
    shared: tuple[SanitizerSweepResult, ...],
) -> ChildDiagnosisReport:
    key = hashlib.sha256(child.unit_id.encode()).hexdigest()[:16]
    report = run_local_fn(
        child,
        executor=executor_factory(
            dataclasses.replace(
                runtime, out_dir=os.path.join(runtime.out_dir, f"kernel-{key}")
            )
        ),
        session_id=f"session-{child.unit_id}",
        campaign_policy=policy,
        shared_sanitizer_results=shared,
        triage_policy=triage,
        sink=None,
        store=None if state is None else FileSessionStore(f"{state}.kernel-{key}.json"),
    )
    return ChildDiagnosisReport(child, report)


def run_local_diagnosis(
    unit: WorkUnit,
    *,
    runtime: LocalRuntimeConfig,
    state: str | None = None,
    campaign_policy: InitialCampaignPolicy,
    triage_policy: TriagePolicy,
    max_kernels: int = DEFAULT_MAX_DISCOVERED_KERNELS,
    run_local_fn: Callable[..., SessionReport] = run_local_session,
    executor_factory: Callable[
        [LocalRuntimeConfig], ExperimentExecutor
    ] = LocalExperimentExecutor,
    discovery_fn: Callable[..., KernelDiscovery] = discover_work_units,
) -> SessionReport | DiagnosisReport:
    """Compatibility workflow returning the existing per-target report schema."""
    if unit.kernel is not None:
        return run_local_fn(
            unit,
            executor=executor_factory(runtime),
            campaign_policy=campaign_policy,
            triage_policy=triage_policy,
            sink=None,
            store=None if state is None else FileSessionStore(state),
        )
    if runtime.cutracer_so is None:
        raise ValueError("kernel discovery requires a resolved cutracer.so")
    discovery = discovery_fn(
        unit,
        out_dir=os.path.join(runtime.out_dir, "kernel-discovery"),
        cutracer_so=runtime.cutracer_so,
        timeout=runtime.timeout,
        max_kernels=max_kernels,
        cutracer_version=runtime.cutracer_version,
        toolchain_version=runtime.toolchain_version,
    )
    shared = _target_sanitizers(unit, campaign_policy, executor_factory(runtime))
    completed = []
    failures = []
    for child in discovery.work_units:
        try:
            completed.append(
                _compatibility_child(
                    child,
                    runtime=runtime,
                    state=state,
                    policy=campaign_policy,
                    triage=triage_policy,
                    run_local_fn=run_local_fn,
                    executor_factory=executor_factory,
                    shared=shared,
                )
            )
        except Exception as error:
            failures.append(KernelDiagnosisFailure(child, str(error)))
    children = tuple(completed)
    if failures:
        raise PartialDiagnosisError(
            DiagnosisReport(
                unit, discovery.result, discovery.manifest_artifact, children
            ),
            tuple(failures),
        )
    return DiagnosisReport(
        unit, discovery.result, discovery.manifest_artifact, children
    )
