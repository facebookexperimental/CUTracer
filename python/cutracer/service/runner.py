"""Backend drivers for the shared analysis-session state machine."""

from __future__ import annotations

import dataclasses
import os
import uuid
from typing import Any, Optional, Protocol

from cutracer.service.contracts import (
    AnalysisDecision,
    AnalysisSession,
    ExperimentKind,
    ExperimentResult,
    ExperimentSpec,
    SessionBudget,
    SessionReport,
    WorkUnit,
)
from cutracer.service.experiments.reduce import run_reduce_experiment
from cutracer.service.experiments.sanitizer import run_sanitizer_experiment
from cutracer.service.experiments.stress import run_stress_experiment
from cutracer.service.experiments.trace import run_trace_experiment
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
    record_analysis_decision,
    record_experiment_result,
    record_triage_decision,
    SessionEffect,
)
from cutracer.service.triage import TriagePolicy


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
    ) -> None:
        self._runtime = runtime
        self._sanitizer_runner = sanitizer_runner
        self._stress_runner = stress_runner
        self._trace_runner = trace_runner
        self._trace_analyzer = trace_analyzer
        self._reduce_runner = reduce_runner

    def execute(self, spec: ExperimentSpec) -> ExperimentResult:
        work_dir = os.path.join(self._runtime.out_dir, spec.experiment_id)
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


def _run_effect(
    session: AnalysisSession,
    effect: SessionEffect,
    *,
    executor: ExperimentExecutor,
    reasoner: ReasonerBackend,
    guard_policy: ExperimentGuardPolicy,
    triage_policy: TriagePolicy,
    sink: Optional[ReportSink],
) -> tuple[AnalysisSession, list[SessionEffect]]:
    effects: list[SessionEffect] = []
    current = session
    if effect.kind == EffectKind.RUN_TRIAGE:
        transition = record_triage_decision(current, triage_policy.decide(current))
        current = transition.session
        effects.extend(transition.effects)
    elif effect.kind == EffectKind.SUBMIT_EXPERIMENTS:
        # Local may serialize these on one GPU, but they remain peer experiments:
        # no result is consulted before all specs in this effect have been run.
        results = [executor.execute(spec) for spec in effect.experiments]
        for result in results:
            transition = record_experiment_result(current, result)
            current = transition.session
            effects.extend(transition.effects)
    elif effect.kind == EffectKind.RUN_REASONER:
        decision = reasoner.analyze(current)
        transition = record_analysis_decision(
            current, decision, guard_policy=guard_policy
        )
        current = transition.session
        effects.extend(transition.effects)
    elif effect.kind == EffectKind.PUBLISH_REPORT:
        report = build_session_report(current)
        if sink is not None:
            sink.publish(report)
    else:
        raise ValueError(f"unknown effect kind: {effect.kind}")
    return current, effects


def run_local_session(
    unit: WorkUnit,
    *,
    executor: ExperimentExecutor,
    reasoner: Optional[ReasonerBackend] = None,
    session_id: Optional[str] = None,
    budget: Optional[SessionBudget] = None,
    campaign_policy: Optional[InitialCampaignPolicy] = None,
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
    )
    current = transition.session
    effects = list(transition.effects)
    active_reasoner = reasoner or ClaudeReasoner()
    active_guard = guard_policy or ExperimentGuardPolicy()
    active_triage = triage_policy or TriagePolicy()
    steps = 0
    while effects:
        steps += 1
        if steps > 100:
            raise RuntimeError("analysis session exceeded its transition safety limit")
        effect = effects.pop(0)
        current, generated = _run_effect(
            current,
            effect,
            executor=executor,
            reasoner=active_reasoner,
            guard_policy=active_guard,
            triage_policy=active_triage,
            sink=sink,
        )
        effects.extend(generated)
        if store is not None:
            store.save(current)
    return build_session_report(current)
