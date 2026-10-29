"""Local diagnosis orchestration, independent of command-line and cloud adapters."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cutracer.runner import resolve_cutracer_so
from cutracer.service.artifacts import local_artifact
from cutracer.service.checkpoint import atomic_write_json, read_json
from cutracer.service.contracts import (
    AnalysisSession,
    ChildDiagnosisReport,
    DEFAULT_MAX_DISCOVERED_KERNELS,
    DiagnosisReport,
    ExperimentKind,
    ReportPublicationStatus,
    SanitizerSweepResult,
    SessionReport,
    SessionStatus,
    WorkUnit,
)
from cutracer.service.file_run_store import FileRunControlStore
from cutracer.service.file_store import FileSessionStore as DurableFileSessionStore
from cutracer.service.kernel_discovery import discover_work_units, KernelDiscovery
from cutracer.service.policy import InitialCampaignPolicy
from cutracer.service.run_contracts import (
    canonical_input_digest,
    DeploymentProfile,
    RunControl,
    RunHandle,
    RunInput,
    RunInputConflict,
    RunOutputMode,
    RunPublicationStatus,
    RunResult,
    RunSummary,
)
from cutracer.service.run_summary import build_run_summary, expected_run_units
from cutracer.service.runner import (
    ExperimentExecutor,
    FileSessionStore,
    LocalExperimentExecutor,
    LocalRuntimeConfig,
    ReasonerBackend,
    resume_local_session,
    run_local_session,
    SessionFailedError,
)
from cutracer.service.sanitizer_results import is_target_scoped_sanitizer
from cutracer.service.session import (
    build_session_report,
    create_discovery_session,
    create_session,
    record_discovery_children_started,
    record_report_published,
)
from cutracer.service.storage import SessionNotFound, StoreConflict
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
        except SessionFailedError as error:
            failures.append(KernelDiagnosisFailure(child, error.termination_reason))
        except (OSError, ValueError) as error:
            failures.append(KernelDiagnosisFailure(child, str(error)))
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


class _SessionCheckpoint:
    def __init__(self, store: DurableFileSessionStore) -> None:
        self._store = store

    def save(self, session: AnalysisSession) -> None:
        self._store.compare_and_swap(
            session.session_id, session.state_revision - 1, session
        )


class LocalBackend:
    """Synchronous Run driver with resumable local state and explicit publication."""

    def __init__(
        self,
        runtime: LocalRuntimeConfig | None,
        *,
        state_dir: str,
        reasoner: ReasonerBackend | None = None,
        triage_policy: TriagePolicy | None = None,
        executor_factory: Callable[
            [LocalRuntimeConfig], ExperimentExecutor
        ] = LocalExperimentExecutor,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._runtime = None if runtime is None else dataclasses.replace(runtime)
        self._controls = FileRunControlStore(os.path.join(state_dir, "runs"))
        self._sessions = DurableFileSessionStore(os.path.join(state_dir, "sessions"))
        self._checkpoint = _SessionCheckpoint(self._sessions)
        self._reasoner = reasoner
        self._triage = triage_policy
        self._executor_factory = executor_factory
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def run(self, request: RunInput) -> RunResult:
        if request.profile != DeploymentProfile.LOCAL:
            raise ValueError("LocalBackend requires the local deployment profile")
        if request.toolchain_version != self._execution_runtime().toolchain_version:
            raise RunInputConflict("local runtime toolchain differs from Run input")
        with self._controls.execution(request.run_id):
            try:
                self._controls.create(RunControl(0, request))
            except StoreConflict:
                existing = self._controls.load(request.run_id)
                if existing.input.input_digest != request.input_digest:
                    raise RunInputConflict(
                        "Run ID is already bound to different input"
                    ) from None
            return self._resume(request.run_id)

    def resume(self, run_id: str) -> RunResult:
        with self._controls.execution(run_id):
            return self._resume(run_id)

    def advance(self, run_id: str) -> RunSummary:
        return self.resume(run_id).summary

    def submit(self, request: RunInput) -> RunHandle:
        raise ValueError(
            "local runs execute in the foreground; hosted submission requires the hosted profile"
        )

    def _execution_runtime(self) -> LocalRuntimeConfig:
        if self._runtime is None:
            raise ValueError("executing a local Run requires a runtime configuration")
        return self._runtime

    def _bind_backend(self, control: RunControl) -> RunControl:
        runtime = self._execution_runtime()
        if control.input.toolchain_version != runtime.toolchain_version:
            raise RunInputConflict("local runtime toolchain differs from Run input")
        native_sha = ""
        if runtime.cutracer_so is not None:
            artifact = local_artifact(runtime.cutracer_so, include_sha256=True)
            if artifact.sha256 is None:
                raise ValueError("could not identify the local CUTracer runtime")
            native_sha = artifact.sha256
        identity = canonical_input_digest(
            {"runtime": dataclasses.asdict(runtime), "native_sha256": native_sha}
        )
        declared = control.input.deployment_config_digest
        if declared and declared != identity:
            raise RunInputConflict(
                "local backend differs from the declared deployment identity"
            )
        if control.backend_identity:
            if control.backend_identity != identity:
                raise RunInputConflict("local backend differs from the original Run")
            return control
        updated = dataclasses.replace(
            control, revision=control.revision + 1, backend_identity=identity
        )
        self._controls.compare_and_swap(control.run_id, control.revision, updated)
        return updated

    def _ensure_session(self, session: AnalysisSession, digest: str) -> AnalysisSession:
        session.admission_input_digest = digest
        try:
            self._sessions.create(session)
            return session
        except StoreConflict:
            existing = self._sessions.load(session.session_id)
            if existing.admission_input_digest != digest:
                raise RunInputConflict(
                    "session ID is already bound to different or legacy input"
                ) from None
            return existing

    def _root(self, request: RunInput) -> AnalysisSession:
        try:
            existing = self._sessions.load(request.run_id)
        except SessionNotFound:
            if request.unit.kernel is None:
                transition = create_discovery_session(
                    request.run_id,
                    request.unit,
                    budget=request.budget,
                    campaign_policy=request.campaign_policy,
                    max_kernels=request.max_kernels,
                )
            else:
                transition = create_session(
                    request.run_id,
                    request.unit,
                    budget=request.budget,
                    campaign_policy=request.campaign_policy,
                )
            session = transition.session
            return self._ensure_session(session, request.input_digest)
        if existing.admission_input_digest != request.input_digest:
            raise RunInputConflict("root session is bound to different or legacy input")
        return existing

    def _drive(self, request: RunInput, session: AnalysisSession) -> AnalysisSession:
        key = hashlib.sha256(session.session_id.encode()).hexdigest()[:20]
        configured = self._execution_runtime()
        runtime = dataclasses.replace(
            configured, out_dir=os.path.join(configured.out_dir, key)
        )
        return resume_local_session(
            session,
            executor=self._executor_factory(runtime),
            reasoner=self._reasoner,
            triage_policy=self._triage,
            store=self._checkpoint,
            deadline_ms=request.deadline_ms,
            clock_ms=self._clock_ms,
        )

    def _create_children(
        self, request: RunInput, root: AnalysisSession
    ) -> AnalysisSession:
        if self._clock_ms() >= request.deadline_ms:
            return root
        shared = tuple(
            result
            for result in root.evidence.sanitizer
            if is_target_scoped_sanitizer(result.tool)
        )
        child_ids = []
        for session_id, unit in expected_run_units(request, root).items():
            if session_id == root.session_id:
                continue
            child = create_session(
                session_id,
                unit,
                budget=request.budget,
                campaign_policy=request.campaign_policy,
                shared_sanitizer_results=shared,
            ).session
            digest = canonical_input_digest(
                {"run_input_digest": request.input_digest, "unit": unit.to_dict()}
            )
            self._ensure_session(child, digest)
            child_ids.append(session_id)
        updated = record_discovery_children_started(root, child_ids).session
        updated.state_revision = root.state_revision + 1
        self._checkpoint.save(updated)
        return updated

    def _resume(self, run_id: str) -> RunResult:
        control = self._bind_backend(self._controls.load(run_id))
        if control.summary is not None and control.summary.is_terminal:
            return self.result(run_id)
        request = control.input
        root = self._drive(request, self._root(request))
        if root.status == SessionStatus.WAITING_CHILDREN:
            root = self._create_children(request, root)
        for session_id in expected_run_units(request, root):
            if session_id == root.session_id:
                continue
            try:
                child = self._sessions.load(session_id)
            except SessionNotFound:
                continue
            self._drive(request, child)
        result = self.result(run_id)
        diagnoses_finished = result.summary.is_terminal or all(
            item.status
            in (
                SessionStatus.COMPLETED,
                SessionStatus.INCONCLUSIVE,
                SessionStatus.FAILED,
            )
            for item in result.summary.sessions
        )
        if diagnoses_finished and request.output.mode == RunOutputMode.FILE:
            result = self._publish(request, result)
        updated = dataclasses.replace(
            control, revision=control.revision + 1, summary=result.summary
        )
        self._controls.compare_and_swap(run_id, control.revision, updated)
        return result

    def _publication_sessions(
        self, result: RunResult
    ) -> tuple[list[AnalysisSession], list[AnalysisSession]]:
        sessions = []
        updates = []
        for item in result.summary.sessions:
            try:
                session = self._sessions.load(item.session_id)
            except SessionNotFound:
                continue
            if (
                item.has_report
                and session.report_publication == ReportPublicationStatus.PENDING
            ):
                updated = record_report_published(session).session
                updated.state_revision = session.state_revision + 1
                updates.append(updated)
                session = updated
            sessions.append(session)
        return sessions, updates

    def _publish(self, request: RunInput, result: RunResult) -> RunResult:
        sessions, updates = self._publication_sessions(result)
        root = self._sessions.load(request.run_id)
        published = build_run_summary(
            request,
            sessions,
            expected_run_units(request, root),
            now_ms=self._clock_ms(),
            publication=RunPublicationStatus.PUBLISHED,
        )
        published = dataclasses.replace(
            published, output_uri=Path(request.output.path).as_uri()
        )
        delivered = dataclasses.replace(result, summary=published)
        try:
            atomic_write_json(os.path.abspath(request.output.path), delivered.to_dict())
        except OSError as error:
            actual_sessions = [
                self._sessions.load(session.session_id) for session in sessions
            ]
            failed = build_run_summary(
                request,
                actual_sessions,
                expected_run_units(request, root),
                now_ms=self._clock_ms(),
                publication=RunPublicationStatus.FAILED,
                error=str(error),
            )
            return dataclasses.replace(
                result,
                summary=failed,
            )
        for updated in updates:
            self._checkpoint.save(updated)
        return delivered

    def result(self, run_id: str) -> RunResult:
        control = self._controls.load(run_id)
        request = control.input
        try:
            root = self._sessions.load(run_id)
        except SessionNotFound:
            expected = {run_id: request.unit}
        else:
            expected = expected_run_units(request, root)
        sessions = []
        for session_id in expected:
            try:
                sessions.append(self._sessions.load(session_id))
            except SessionNotFound:
                continue
        publication = None if control.summary is None else control.summary.publication
        summary = build_run_summary(
            request,
            sessions,
            expected,
            now_ms=self._clock_ms(),
            publication=publication,
        )
        if control.summary is not None and control.summary.is_terminal:
            summary = control.summary
        return RunResult(
            summary,
            tuple(
                build_session_report(session)
                for session in sessions
                if session.report is not None
            ),
        )

    def status(self, run_id: str) -> RunSummary:
        return self.result(run_id).summary


def _local_runtime_options(config: object) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("local runtime config must be a JSON object")
    fields = dataclasses.fields(LocalRuntimeConfig)
    configurable = {field.name for field in fields} - {"cutracer_version"}
    unsupported = sorted(str(key) for key in config if key not in configurable)
    if unsupported:
        raise ValueError(
            f"unsupported local runtime config keys: {', '.join(unsupported)}"
        )
    for name in ("timeout", "reduce_timeout"):
        value = config.get(name, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"local runtime {name} must be a positive integer")
    for name in ("out_dir", "toolchain_version"):
        if not isinstance(config.get(name, ""), str):
            raise ValueError(f"local runtime {name} must be a string")
    for name in ("cutracer_so", "compute_sanitizer"):
        if not isinstance(config.get(name), (str, type(None))):
            raise ValueError(f"local runtime {name} must be a string or null")
    return dict(config)


def create_local_backend(
    *,
    state_dir: str,
    request: RunInput | None = None,
    config_path: str | None = None,
    execute: bool = True,
) -> LocalBackend:
    if not execute:
        return LocalBackend(None, state_dir=state_dir)
    if request is None:
        raise ValueError("executing a local Run requires its immutable request")
    config: dict[str, Any] = (
        {} if config_path is None else _local_runtime_options(read_json(config_path))
    )
    config.setdefault("out_dir", os.path.join(os.path.abspath(state_dir), "artifacts"))
    config["out_dir"] = os.path.abspath(config["out_dir"])
    config.setdefault("toolchain_version", request.toolchain_version)
    config["cutracer_so"] = resolve_cutracer_so(config.get("cutracer_so"))
    artifact = local_artifact(config["cutracer_so"], include_sha256=True)
    if artifact.sha256 is None:
        raise ValueError("could not identify the local CUTracer runtime")
    config["cutracer_version"] = f"sha256:{artifact.sha256}"
    return LocalBackend(LocalRuntimeConfig(**config), state_dir=state_dir)
