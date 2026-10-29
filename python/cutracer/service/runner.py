"""Backend drivers for the shared analysis-session state machine."""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import time
import uuid
from typing import Any, Callable, Optional, Protocol, Sequence

from cutracer.service.contracts import (
    AnalysisDecision,
    AnalysisSession,
    ExperimentKind,
    ExperimentResult,
    ExperimentSpec,
    SanitizerSweepResult,
    SessionBudget,
    SessionReport,
    SessionStatus,
    WorkUnit,
)
from cutracer.service.experiments.reduce import run_reduce_experiment
from cutracer.service.experiments.sanitizer import run_sanitizer_experiment
from cutracer.service.experiments.stress import run_stress_experiment
from cutracer.service.experiments.trace import run_trace_experiment
from cutracer.service.kernel_discovery import discover_work_units, KernelDiscoveryError
from cutracer.service.policy import (
    DEFAULT_LOCAL_SANITIZER_TOOLS,
    ExperimentGuardPolicy,
    InitialCampaignPolicy,
)
from cutracer.service.reasoner import ClaudeReasoner
from cutracer.service.session import (
    build_session_report,
    create_session,
    EffectKind,
    ExperimentResultValidationError,
    record_analysis_decision,
    record_experiment_result,
    record_infrastructure_failure,
    record_report_publication_failed,
    record_report_published,
    record_triage_decision,
    resume_session,
    SessionEffect,
)
from cutracer.service.triage import TriagePolicy

logger: logging.Logger = logging.getLogger(__name__)


class SessionFailedError(RuntimeError):
    """A local session ended FAILED without producing a diagnosis report."""

    def __init__(self, session_id: str, termination_reason: str) -> None:
        super().__init__(f"session {session_id} failed: {termination_reason}")
        self.session_id = session_id
        self.termination_reason = termination_reason


class ExperimentExecutor(Protocol):
    def execute(self, spec: ExperimentSpec) -> ExperimentResult: ...


class ReasonerBackend(Protocol):
    def analyze(self, session: AnalysisSession) -> AnalysisDecision: ...


class SessionStore(Protocol):
    def save(self, session: AnalysisSession) -> None: ...


class ReportSink(Protocol):
    def publish(self, report: SessionReport) -> None: ...


@dataclasses.dataclass
class LocalRuntimeConfig:
    out_dir: str
    cutracer_so: Optional[str] = None
    compute_sanitizer: Optional[str] = None
    timeout: int = 1800
    reduce_timeout: int = 7200
    cutracer_version: str = ""
    toolchain_version: str = ""


class LocalExperimentExecutor:
    """Run experiment specs synchronously on the current GPU host."""

    def __init__(
        self,
        runtime: LocalRuntimeConfig,
        *,
        sanitizer_runner: Optional[Any] = None,
        stress_runner: Optional[Any] = None,
        trace_runner: Optional[Any] = None,
        trace_analyzer: Optional[Any] = None,
        reduce_runner: Optional[Any] = None,
        discovery_runner: Optional[Any] = None,
    ) -> None:
        self._runtime = runtime
        self._sanitizer_runner = sanitizer_runner
        self._stress_runner = stress_runner
        self._trace_runner = trace_runner
        self._trace_analyzer = trace_analyzer
        self._reduce_runner = reduce_runner
        self._discovery_runner = discovery_runner

    def execute(self, spec: ExperimentSpec) -> ExperimentResult:
        work_dir = os.path.join(self._runtime.out_dir, spec.experiment_id)
        if spec.kind == ExperimentKind.KERNEL_DISCOVERY:
            if self._runtime.cutracer_so is None or spec.discovery is None:
                raise ValueError(
                    "kernel discovery requires a runtime and discovery options"
                )
            return discover_work_units(
                spec.unit,
                out_dir=work_dir,
                cutracer_so=self._runtime.cutracer_so,
                timeout=self._runtime.timeout,
                max_kernels=spec.discovery.max_kernels,
                runner=self._discovery_runner,
                experiment_id=spec.experiment_id,
                cutracer_version=self._runtime.cutracer_version,
                toolchain_version=self._runtime.toolchain_version,
            ).result
        if spec.kind == ExperimentKind.COMPUTE_SANITIZER:
            return run_sanitizer_experiment(
                spec,
                out_dir=work_dir,
                runner=self._sanitizer_runner,
                compute_sanitizer=self._runtime.compute_sanitizer,
                timeout=self._runtime.timeout,
            )
        if spec.kind == ExperimentKind.RANDOM_DELAY_STRESS:
            return run_stress_experiment(
                spec,
                out_dir=work_dir,
                runner=self._stress_runner,
                cutracer_so=self._runtime.cutracer_so,
                timeout=self._runtime.timeout,
                cutracer_version=self._runtime.cutracer_version,
                toolchain_version=self._runtime.toolchain_version,
            )
        if spec.kind in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
            return run_trace_experiment(
                spec,
                out_dir=work_dir,
                runner=self._trace_runner,
                trace_analyzer=self._trace_analyzer,
                cutracer_so=self._runtime.cutracer_so,
                timeout=self._runtime.timeout,
                cutracer_version=self._runtime.cutracer_version,
                toolchain_version=self._runtime.toolchain_version,
            )
        if spec.kind == ExperimentKind.REDUCE_DELAY_CONFIG:
            return run_reduce_experiment(
                spec,
                out_dir=work_dir,
                runner=self._reduce_runner,
                cutracer_so=self._runtime.cutracer_so,
                timeout=self._runtime.reduce_timeout,
                cutracer_version=self._runtime.cutracer_version,
                toolchain_version=self._runtime.toolchain_version,
            )
        raise ValueError(f"unsupported experiment kind: {spec.kind}")


def _atomic_write(path: str, text: str) -> None:
    # Write via a temp file + os.replace so an interrupted write never leaves a
    # half-written checkpoint/report behind.
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class FileSessionStore:
    def __init__(self, path: str) -> None:
        self._path = path

    def save(self, session: AnalysisSession) -> None:
        _atomic_write(self._path, session.to_json(indent=2))


class FileReportSink:
    def __init__(self, path: str) -> None:
        self._path = path

    def publish(self, report: SessionReport) -> None:
        _atomic_write(self._path, report.to_json(indent=2))


def _run_experiment_batch(
    session: AnalysisSession,
    experiments: Sequence[ExperimentSpec],
    *,
    executor: ExperimentExecutor,
    store: Optional[SessionStore],
    deadline_ms: int | None,
    clock_ms: Callable[[], int],
) -> tuple[AnalysisSession, list[SessionEffect]]:
    current = session
    effects = []
    # Acceptance is durable between peers; triage still waits for fan-in in
    # record_experiment_result, so a clean peer cannot cancel another.
    for spec in experiments:
        if deadline_ms is not None and clock_ms() >= deadline_ms:
            return current, effects
        try:
            result = executor.execute(spec)
        except (OSError, subprocess.SubprocessError, KernelDiscoveryError) as error:
            transition = record_infrastructure_failure(current, str(error))
            return _checkpoint(current, transition.session, store), []
        try:
            transition = record_experiment_result(current, result)
        except ExperimentResultValidationError as error:
            transition = record_infrastructure_failure(current, str(error))
            return _checkpoint(current, transition.session, store), []
        current = _checkpoint(current, transition.session, store)
        effects.extend(transition.effects)
    return current, effects


def _run_effect(
    session: AnalysisSession,
    effect: SessionEffect,
    *,
    executor: ExperimentExecutor,
    reasoner: ReasonerBackend,
    guard_policy: ExperimentGuardPolicy,
    triage_policy: TriagePolicy,
    sink: Optional[ReportSink],
    store: Optional[SessionStore],
    deadline_ms: int | None,
    clock_ms: Callable[[], int],
) -> tuple[AnalysisSession, list[SessionEffect]]:
    effects: list[SessionEffect] = []
    current = session
    if effect.kind == EffectKind.RUN_TRIAGE:
        transition = record_triage_decision(current, triage_policy.decide(current))
        current = transition.session
        effects.extend(transition.effects)
    elif effect.kind == EffectKind.SUBMIT_EXPERIMENTS:
        return _run_experiment_batch(
            current,
            effect.experiments,
            executor=executor,
            store=store,
            deadline_ms=deadline_ms,
            clock_ms=clock_ms,
        )
    elif effect.kind == EffectKind.RUN_REASONER:
        try:
            decision = reasoner.analyze(current)
        except (OSError, subprocess.SubprocessError) as error:
            transition = record_infrastructure_failure(current, str(error))
            return _checkpoint(current, transition.session, store), []
        transition = record_analysis_decision(
            current, decision, guard_policy=guard_policy
        )
        current = transition.session
        effects.extend(transition.effects)
    elif effect.kind == EffectKind.PUBLISH_REPORT:
        report = build_session_report(current)
        if sink is not None:
            try:
                sink.publish(report)
            except OSError as error:
                logger.warning(
                    "report publication failed for session %s: %s",
                    current.session_id,
                    error,
                )
                current = record_report_publication_failed(current, str(error)).session
            else:
                current = record_report_published(current).session
                triage_policy.record_published_baselines(current)
        else:
            # With no sink, run_local_session returns the report to its caller:
            # that return is the delivery, so acknowledge it instead of leaving
            # the session pending reconciliation forever.
            current = record_report_published(current).session
    else:
        raise ValueError(f"unknown effect kind: {effect.kind}")
    return _checkpoint(session, current, store), effects


def _checkpoint(
    previous: AnalysisSession,
    current: AnalysisSession,
    store: Optional[SessionStore],
) -> AnalysisSession:
    if current is not previous:
        current.state_revision = previous.state_revision + 1
        if store is not None:
            store.save(current)
    return current


def resume_local_session(
    session: AnalysisSession,
    *,
    executor: ExperimentExecutor,
    reasoner: Optional[ReasonerBackend] = None,
    guard_policy: Optional[ExperimentGuardPolicy] = None,
    triage_policy: Optional[TriagePolicy] = None,
    store: Optional[SessionStore] = None,
    sink: Optional[ReportSink] = None,
    deadline_ms: int | None = None,
    clock_ms: Callable[[], int] | None = None,
) -> AnalysisSession:
    """Resume accepted progress, stopping at discovery fanout for the Run driver."""
    current = session
    effects = list(resume_session(current).effects)
    active_reasoner = reasoner or ClaudeReasoner()
    active_guard = guard_policy or ExperimentGuardPolicy()
    active_triage = triage_policy or TriagePolicy()
    clock = clock_ms or (lambda: time.time_ns() // 1_000_000)
    steps = 0
    while effects:
        steps += 1
        if steps > 100:
            raise RuntimeError("analysis session exceeded its transition safety limit")
        effect = effects.pop(0)
        if effect.kind == EffectKind.CREATE_CHILD_SESSIONS:
            return current
        if deadline_ms is not None and clock() >= deadline_ms:
            return current
        current, generated = _run_effect(
            current,
            effect,
            executor=executor,
            reasoner=active_reasoner,
            guard_policy=active_guard,
            triage_policy=active_triage,
            sink=sink,
            store=store,
            deadline_ms=deadline_ms,
            clock_ms=clock,
        )
        effects.extend(generated)
    return current


def run_local_session(
    unit: WorkUnit,
    *,
    executor: ExperimentExecutor,
    reasoner: Optional[ReasonerBackend] = None,
    session_id: Optional[str] = None,
    budget: Optional[SessionBudget] = None,
    campaign_policy: Optional[InitialCampaignPolicy] = None,
    shared_sanitizer_results: Sequence[SanitizerSweepResult] = (),
    guard_policy: Optional[ExperimentGuardPolicy] = None,
    triage_policy: Optional[TriagePolicy] = None,
    store: Optional[SessionStore] = None,
    sink: Optional[ReportSink] = None,
) -> SessionReport:
    active_campaign = campaign_policy or InitialCampaignPolicy(
        sanitizer_tools=DEFAULT_LOCAL_SANITIZER_TOOLS
    )
    transition = create_session(
        session_id or f"session-{uuid.uuid4().hex}",
        unit,
        budget=budget,
        campaign_policy=active_campaign,
        shared_sanitizer_results=shared_sanitizer_results,
    )
    if store is not None:
        store.save(transition.session)
    current = resume_local_session(
        transition.session,
        executor=executor,
        reasoner=reasoner,
        guard_policy=guard_policy,
        triage_policy=triage_policy,
        store=store,
        sink=sink,
    )
    if current.status == SessionStatus.FAILED:
        raise SessionFailedError(current.session_id, current.termination_reason)
    return build_session_report(current)
