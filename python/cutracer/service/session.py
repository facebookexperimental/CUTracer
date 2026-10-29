"""Pure, backend-independent analysis-session state machine."""

from __future__ import annotations

import dataclasses
from collections import Counter
from enum import Enum
from typing import List, NoReturn, Optional, Sequence

from cutracer.service.artifacts import same_artifact_content
from cutracer.service.citations import evidence_artifact_refs
from cutracer.service.contracts import (
    AnalysisDecision,
    AnalysisSession,
    AnalysisTurn,
    Confidence,
    CrossValidation,
    DecisionKind,
    EvidenceBundle,
    EvidenceSufficiency,
    ExecutionStatus,
    ExperimentKind,
    ExperimentPhase,
    ExperimentRecord,
    ExperimentResult,
    ExperimentSpec,
    ExplainReport,
    KernelId,
    RaceClass,
    ReduceExperimentResult,
    ReduceOutcome,
    ReportPublicationStatus,
    ResultCompleteness,
    SanitizerOutcome,
    SanitizerSweepResult,
    SessionBudget,
    SessionReport,
    SessionStatus,
    StressOutcome,
    StressTestResult,
    TraceAnalysisSeverity,
    TraceExperimentResult,
    TriageDecision,
    TriageOutcome,
    TriggeringDelayConfig,
    WorkUnit,
)
from cutracer.service.policy import (
    CampaignPlan,
    experiment_spec_fingerprint,
    ExperimentGuardPolicy,
    InitialCampaignPolicy,
)
from cutracer.service.sanitizer_results import (
    ERROR_TYPE_INVALID_MEMORY_ACCESS,
    ERROR_TYPE_MEMORY_LEAK,
    ERROR_TYPE_MISALIGNED_ACCESS,
    ERROR_TYPE_RACE_CONDITION,
    ERROR_TYPE_UNINITIALIZED_READ,
    is_target_scoped_sanitizer,
    project_target_scoped_sanitizer_result,
)
from cutracer.service.triage import initial_evidence_requires_inconclusive


class EffectKind(str, Enum):
    SUBMIT_EXPERIMENTS = "submit_experiments"
    RUN_TRIAGE = "run_triage"
    RUN_REASONER = "run_reasoner"
    PUBLISH_REPORT = "publish_report"


@dataclasses.dataclass(frozen=True)
class SessionEffect:
    kind: EffectKind
    experiments: tuple[ExperimentSpec, ...] = ()


@dataclasses.dataclass(frozen=True)
class Transition:
    session: AnalysisSession
    effects: tuple[SessionEffect, ...]


class ExperimentResultValidationError(ValueError):
    """Raised when a result cannot be safely bound to the current session."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


_MAX_TERMINATION_REASON_LENGTH = 2048
_TERMINAL_STATUSES = frozenset(
    (
        SessionStatus.COMPLETED,
        SessionStatus.INCONCLUSIVE,
        SessionStatus.FAILED,
    )
)


def _clone(session: AnalysisSession) -> AnalysisSession:
    return AnalysisSession.from_dict(session.to_dict())


def record_infrastructure_failure(
    session: AnalysisSession,
    reason: str,
) -> Transition:
    """Terminate one active revision without fabricating a diagnosis report."""
    normalized = reason.strip()
    if not normalized:
        raise ValueError("infrastructure failure requires a reason")
    if session.status in _TERMINAL_STATUSES:
        return Transition(session=session, effects=())
    updated = _clone(session)
    updated.status = SessionStatus.FAILED
    updated.pending_experiments = []
    updated.report = None
    updated.termination_reason = normalized[:_MAX_TERMINATION_REASON_LENGTH]
    return Transition(session=updated, effects=())


def _artifact_refs(evidence: EvidenceBundle) -> List[str]:
    return list(dict.fromkeys(ref.uri for ref in evidence_artifact_refs(evidence)))


def _evidence_results(evidence: EvidenceBundle) -> List[ExperimentResult]:
    results: List[ExperimentResult] = []
    results.extend(evidence.sanitizer)
    results.extend(evidence.stress)
    results.extend(evidence.traces)
    results.extend(evidence.reductions)
    return results


def _reject_result(code: str, message: str) -> NoReturn:
    raise ExperimentResultValidationError(code, message)


def _validate_session_result_indexes(session: AnalysisSession) -> None:
    pending_ids = session.pending_experiment_ids
    completed_ids = session.completed_experiment_ids
    if len(pending_ids) != len(set(pending_ids)):
        _reject_result("session_state_corrupt", "pending experiment IDs are not unique")
    if len(completed_ids) != len(set(completed_ids)):
        _reject_result(
            "session_state_corrupt", "completed experiment IDs are not unique"
        )
    overlap = set(pending_ids).intersection(completed_ids)
    if overlap:
        _reject_result(
            "session_state_corrupt",
            f"experiments are both pending and completed: {sorted(overlap)}",
        )

    evidence = _evidence_results(session.evidence)
    evidence_counts = Counter(item.experiment_id for item in evidence)
    for experiment_id in completed_ids:
        count = evidence_counts[experiment_id]
        if count != 1:
            _reject_result(
                "session_state_corrupt",
                f"completed experiment {experiment_id} has {count} evidence records",
            )


def _validate_spec_context(
    session: AnalysisSession,
    spec: ExperimentSpec,
) -> None:
    if spec.session_id != session.session_id:
        _reject_result(
            "session_mismatch",
            f"pending spec belongs to session {spec.session_id}",
        )
    if spec.unit != session.unit:
        _reject_result(
            "session_mismatch", "pending spec work unit differs from the session"
        )

    if spec.phase == ExperimentPhase.INITIAL:
        if session.status != SessionStatus.WAITING_INITIAL:
            _reject_result(
                "phase_mismatch",
                f"initial result received in state {session.status.value}",
            )
        if spec.round_index != 0 or session.round_index != 0:
            _reject_result(
                "round_mismatch", "initial experiments must belong to round zero"
            )
    elif spec.phase == ExperimentPhase.FOLLOWUP:
        if session.status != SessionStatus.WAITING_FOLLOWUP:
            _reject_result(
                "phase_mismatch",
                f"follow-up result received in state {session.status.value}",
            )
        if spec.round_index < 1 or spec.round_index != session.round_index:
            _reject_result(
                "round_mismatch",
                f"follow-up round {spec.round_index} does not match session round "
                f"{session.round_index}",
            )
    else:
        _reject_result("phase_mismatch", f"unsupported experiment phase: {spec.phase}")


def _validate_provenance(spec: ExperimentSpec, result: ExperimentResult) -> None:
    provenance = result.provenance
    expected = spec.unit
    if provenance.source_revision != expected.source_revision:
        _reject_result(
            "source_revision_mismatch",
            f"expected {expected.source_revision!r}, got {provenance.source_revision!r}",
        )
    if provenance.arch != expected.arch:
        _reject_result(
            "arch_mismatch", f"expected {expected.arch!r}, got {provenance.arch!r}"
        )
    if (provenance.kernel is None) != (expected.kernel is None):
        _reject_result(
            "kernel_identity_mismatch", "kernel identity presence does not match"
        )
    if provenance.kernel is None or expected.kernel is None:
        return
    if provenance.kernel.name != expected.kernel.name:
        _reject_result(
            "kernel_name_mismatch",
            f"expected {expected.kernel.name!r}, got {provenance.kernel.name!r}",
        )
    if provenance.kernel.cubin_hash != expected.kernel.cubin_hash:
        _reject_result(
            "cubin_hash_mismatch",
            f"expected {expected.kernel.cubin_hash!r}, "
            f"got {provenance.kernel.cubin_hash!r}",
        )


def _validate_triggering_config(
    spec: ExperimentSpec,
    result: StressTestResult,
    config: TriggeringDelayConfig,
) -> None:
    unit = spec.unit
    if (
        config.work_unit_id != unit.unit_id
        or config.target_argv != unit.argv
        or config.oracle != unit.oracle
        or config.source_revision != unit.source_revision
        or config.arch != unit.arch
        or config.kernel != unit.kernel
        or config.cutracer_version != result.provenance.tool_version
        or config.toolchain_version != result.provenance.toolchain_version
    ):
        _reject_result(
            "trigger_config_mismatch",
            "triggering config replay context differs from the pending spec",
        )
    assert spec.stress is not None
    options = spec.stress
    if config.delay_ns not in options.delay_ladder_ns:
        _reject_result(
            "trigger_config_mismatch", "trigger delay is outside the planned ladder"
        )
    if config.enable_prob != options.enable_prob:
        _reject_result(
            "trigger_config_mismatch", "trigger probability differs from the plan"
        )
    if options.warpgroup_ids:
        if config.warpgroup_id not in options.warpgroup_ids:
            _reject_result(
                "trigger_config_mismatch",
                "trigger warpgroup is outside the planned targets",
            )
    elif config.warpgroup_id is not None:
        _reject_result(
            "trigger_config_mismatch", "an untargeted plan returned a warpgroup target"
        )
    if not 0 <= config.attempt_index < options.attempts_per_delay:
        _reject_result(
            "trigger_config_mismatch", "trigger attempt is outside the planned range"
        )
    if (
        config.completed_trials != result.completed_trials
        or config.reproductions != result.reproductions
    ):
        _reject_result(
            "trigger_config_mismatch",
            "trigger campaign counts differ from the stress result",
        )


def _validate_sanitizer_result(
    spec: ExperimentSpec,
    result: SanitizerSweepResult,
) -> None:
    if result.tool != spec.sanitizer_tool:
        _reject_result(
            "sanitizer_tool_mismatch",
            f"expected {spec.sanitizer_tool!r}, got {result.tool!r}",
        )
    has_reported_evidence = bool(result.findings) or (
        result.summary is not None and result.summary.is_positive
    )
    if result.outcome == SanitizerOutcome.FINDING and not has_reported_evidence:
        _reject_result(
            "sanitizer_result_inconsistent",
            "finding outcome has no finding block or positive summary",
        )
    if result.outcome == SanitizerOutcome.CLEAN and not result.has_trustworthy_clean:
        _reject_result(
            "sanitizer_result_inconsistent",
            "clean outcome requires a complete zero racecheck summary",
        )
    if result.completeness == ResultCompleteness.COMPLETE and result.summary_issues:
        _reject_result(
            "sanitizer_result_inconsistent",
            "complete sanitizer result cannot retain summary diagnostics",
        )
    if result.completeness == ResultCompleteness.COMPLETE_EARLY:
        _reject_result(
            "sanitizer_result_inconsistent",
            "sanitizer checks do not support complete-early results",
        )


def _validate_stress_counts(
    spec: ExperimentSpec,
    result: StressTestResult,
) -> None:
    assert spec.stress is not None
    if result.planned_trials != spec.stress.planned_trials:
        _reject_result(
            "stress_plan_mismatch",
            f"expected {spec.stress.planned_trials} planned trials, "
            f"got {result.planned_trials}",
        )
    # ``infra_errors`` may include a post-campaign config persistence failure,
    # which is outside the planned trial count. Only completed and timed-out
    # trials are guaranteed to consume a planned slot.
    if (
        result.completed_trials < 0
        or result.reproductions < 0
        or result.unattributed_reproductions < 0
        or result.infra_errors < 0
        or result.timed_out_trials < 0
        or result.completed_trials > result.planned_trials
        or (
            result.reproductions + result.unattributed_reproductions
            > result.completed_trials
        )
        or result.completed_trials + result.timed_out_trials > result.planned_trials
    ):
        _reject_result(
            "stress_plan_mismatch", "stress campaign counts are outside the plan"
        )


def _validate_stress_triggering_config(
    spec: ExperimentSpec,
    result: StressTestResult,
) -> None:
    if result.triggering_config is not None:
        if result.reproductions == 0:
            _reject_result(
                "trigger_config_mismatch",
                "triggering config has no reproduced trial",
            )
        _validate_triggering_config(spec, result, result.triggering_config)


def _validate_skipped_stress_outcome(result: StressTestResult) -> None:
    if result.outcome != StressOutcome.SKIPPED:
        return
    skipped_scope_is_invalid = (
        result.execution_status != ExecutionStatus.SUCCEEDED
        or result.planned_trials != 0
        or result.completed_trials != 0
        or result.reproductions != 0
        or result.unattributed_reproductions != 0
        or result.infra_errors != 0
        or result.timed_out_trials != 0
        or result.triggering_config is not None
        or result.completeness != ResultCompleteness.SKIPPED
    )
    if skipped_scope_is_invalid:
        _reject_result(
            "stress_result_inconsistent",
            "skipped stress result must have an empty skipped scope",
        )


def _validate_stress_outcome(
    spec: ExperimentSpec,
    result: StressTestResult,
) -> None:
    _validate_stress_triggering_config(spec, result)
    if result.outcome == StressOutcome.REPRODUCED and (
        result.reproductions == 0 or result.triggering_config is None
    ):
        _reject_result(
            "stress_result_inconsistent",
            "reproduced outcome requires a persisted triggering config",
        )
    if result.outcome == StressOutcome.UNATTRIBUTED_REPRODUCTION and (
        result.unattributed_reproductions == 0
        or result.reproductions != 0
        or result.triggering_config is not None
    ):
        _reject_result(
            "stress_result_inconsistent",
            "unattributed outcome requires only unattributed reproductions",
        )
    if (
        result.outcome == StressOutcome.NOT_REPRODUCED
        and not result.has_trustworthy_non_reproduction
    ):
        _reject_result(
            "stress_result_inconsistent",
            "non-reproduced outcome requires complete error-free trial coverage",
        )
    _validate_skipped_stress_outcome(result)


def _validate_stress_completeness(
    spec: ExperimentSpec,
    result: StressTestResult,
) -> None:
    assert spec.stress is not None
    if result.outcome != StressOutcome.SKIPPED:
        if result.completeness == ResultCompleteness.SKIPPED:
            _reject_result(
                "stress_result_inconsistent",
                "only a skipped outcome can claim skipped completeness",
            )
    if result.outcome == StressOutcome.INCOMPLETE and result.completeness in (
        ResultCompleteness.COMPLETE,
        ResultCompleteness.COMPLETE_EARLY,
    ):
        _reject_result(
            "stress_result_inconsistent",
            "incomplete outcome cannot claim complete coverage",
        )
    if result.completeness == ResultCompleteness.COMPLETE and (
        result.completed_trials != result.planned_trials
        or result.infra_errors != 0
        or result.timed_out_trials != 0
    ):
        _reject_result(
            "stress_result_inconsistent",
            "complete stress result must account for every planned trial",
        )
    if result.completeness == ResultCompleteness.COMPLETE_EARLY and (
        result.outcome != StressOutcome.REPRODUCED
        or not spec.stress.stop_on_first_reproduction
        or result.completed_trials >= result.planned_trials
        or result.infra_errors != 0
        or result.timed_out_trials != 0
    ):
        _reject_result(
            "stress_result_inconsistent",
            "complete-early stress result requires an error-free early reproduction",
        )


def _validate_stress_result(
    spec: ExperimentSpec,
    result: StressTestResult,
) -> None:
    _validate_stress_counts(spec, result)
    _validate_stress_outcome(spec, result)
    _validate_stress_completeness(spec, result)


def _validate_trace_success_shape(result: TraceExperimentResult) -> None:
    if not result.traces or result.error is not None:
        _reject_result(
            "trace_result_inconsistent",
            "successful trace result requires one or more traces and no error",
        )
    if result.completeness not in (
        ResultCompleteness.COMPLETE,
        ResultCompleteness.UNKNOWN,
    ):
        _reject_result(
            "trace_result_inconsistent",
            "successful trace result must be complete",
        )
    if result.completeness == ResultCompleteness.UNKNOWN and (
        len(result.traces) != 1 or result.supporting_artifacts
    ):
        _reject_result(
            "trace_result_inconsistent",
            "legacy unknown completeness supports one primary trace only",
        )
    if result.completeness != ResultCompleteness.COMPLETE:
        return
    if not any(
        artifact.relative_path.endswith(".cubin")
        for artifact in result.supporting_artifacts
    ):
        _reject_result(
            "trace_result_inconsistent",
            "complete trace result requires a supporting cubin",
        )
    if result.log is None:
        _reject_result(
            "trace_result_inconsistent",
            "complete trace result requires a capture log",
        )


def _validate_trace_failure_shape(result: TraceExperimentResult) -> None:
    if result.error is None or not result.error.strip():
        _reject_result(
            "trace_result_inconsistent",
            "unsuccessful trace result requires an error",
        )
    has_partial_artifacts = bool(result.traces or result.supporting_artifacts)
    if has_partial_artifacts and result.completeness != ResultCompleteness.PARTIAL:
        _reject_result(
            "trace_result_inconsistent",
            "unsuccessful trace artifacts require partial completeness",
        )
    if result.completeness in (
        ResultCompleteness.COMPLETE,
        ResultCompleteness.COMPLETE_EARLY,
        ResultCompleteness.SKIPPED,
    ):
        _reject_result(
            "trace_result_inconsistent",
            "unsuccessful trace result cannot be complete",
        )


def _validate_trace_artifact_identity(result: TraceExperimentResult) -> List[str]:
    if result.completeness in (
        ResultCompleteness.COMPLETE,
        ResultCompleteness.PARTIAL,
    ):
        artifacts = [*result.traces, *result.supporting_artifacts]
        if result.log is not None:
            artifacts.append(result.log)
        if result.analysis is not None and result.analysis.artifact is not None:
            artifacts.append(result.analysis.artifact)
        if any(
            artifact.sha256 is None
            or len(artifact.sha256) != 64
            or artifact.size_bytes is None
            or artifact.size_bytes < 0
            for artifact in artifacts
        ):
            _reject_result(
                "trace_result_inconsistent",
                "N15 trace artifacts require immutable digest and size identity",
            )
    trace_names = [artifact.relative_path for artifact in result.traces]
    if len(set(trace_names)) != len(trace_names):
        _reject_result(
            "trace_result_inconsistent",
            "trace result contains duplicate trace artifact names",
        )
    return trace_names


def _validate_trace_analysis_references(
    result: TraceExperimentResult,
    trace_names: List[str],
) -> None:
    assert result.analysis is not None
    require_trace_file = len(trace_names) > 1
    evidence_entries = (
        *result.analysis.detector_summaries,
        *result.analysis.findings,
    )
    for evidence in evidence_entries:
        trace_file = evidence.trace_file
        if trace_file is None and require_trace_file:
            _reject_result(
                "trace_result_inconsistent",
                "multi-file trace analysis requires trace_file identity",
            )
        if trace_file is not None and trace_file not in trace_names:
            _reject_result(
                "trace_result_inconsistent",
                "trace analysis references an unknown trace_file",
            )


def _validate_trace_analysis_status(result: TraceExperimentResult) -> None:
    assert result.analysis is not None
    if result.execution_status == ExecutionStatus.SUCCEEDED:
        # UNKNOWN is the pre-N15 singleton contract. It historically allowed
        # best-effort failed analysis alongside a successful capture, so keep
        # that wire behavior while requiring N15 COMPLETE results to succeed.
        if (
            result.completeness == ResultCompleteness.COMPLETE
            and result.analysis.execution_status != ExecutionStatus.SUCCEEDED
        ):
            _reject_result(
                "trace_result_inconsistent",
                "successful trace result requires successful deterministic analysis",
            )
    elif (
        result.execution_status != ExecutionStatus.INFRA_ERROR
        or result.completeness != ResultCompleteness.PARTIAL
        or not result.traces
    ):
        _reject_result(
            "trace_result_inconsistent",
            "only partial trace infrastructure failure may publish analysis",
        )
    if result.analysis.execution_status == ExecutionStatus.SUCCEEDED:
        if result.analysis.artifact is None or result.analysis.error is not None:
            _reject_result(
                "trace_result_inconsistent",
                "successful deterministic analysis requires an artifact and no error",
            )
    elif result.analysis.error is None or not result.analysis.error.strip():
        _reject_result(
            "trace_result_inconsistent",
            "unsuccessful deterministic analysis requires an error",
        )


def _validate_trace_result(result: TraceExperimentResult) -> None:
    if result.execution_status == ExecutionStatus.SUCCEEDED:
        _validate_trace_success_shape(result)
    else:
        _validate_trace_failure_shape(result)
    trace_names = _validate_trace_artifact_identity(result)
    if result.analysis is None:
        if (
            result.execution_status == ExecutionStatus.SUCCEEDED
            and result.completeness == ResultCompleteness.COMPLETE
        ):
            _reject_result(
                "trace_result_inconsistent",
                "complete trace result requires deterministic analysis",
            )
        return
    _validate_trace_analysis_references(result, trace_names)
    _validate_trace_analysis_status(result)


def _validate_reduce_result(
    spec: ExperimentSpec,
    result: ReduceExperimentResult,
) -> None:
    assert spec.reduction is not None
    expected_input = spec.reduction.triggering_config
    returned_input = result.input_config
    input_matches = returned_input.to_dict() == expected_input.to_dict()
    if (
        not input_matches
        and result.execution_status == ExecutionStatus.SUCCEEDED
        and result.outcome == ReduceOutcome.REDUCED
        and same_artifact_content(returned_input.artifact, expected_input.artifact)
    ):
        input_matches = (
            dataclasses.replace(
                returned_input, artifact=expected_input.artifact
            ).to_dict()
            == expected_input.to_dict()
        )
    if not input_matches:
        _reject_result(
            "reduce_input_config_mismatch",
            "reduction input differs from the pending triggering config",
        )
    if result.outcome == ReduceOutcome.REDUCED:
        if (
            result.execution_status != ExecutionStatus.SUCCEEDED
            or result.report is None
            or result.error is not None
        ):
            _reject_result(
                "reduce_result_inconsistent",
                "reduced outcome requires a successful report and no error",
            )
    else:
        if (
            result.minimized_config is not None
            or result.report is not None
            or result.essential_points
            or result.essential_points_artifact is not None
        ):
            _reject_result(
                "reduce_result_inconsistent",
                "unsuccessful reduction cannot publish reduced artifacts",
            )
        expected_statuses = {
            ReduceOutcome.NON_REPRODUCIBLE: (ExecutionStatus.FAILED,),
            ReduceOutcome.STALE_INPUT: (ExecutionStatus.FAILED,),
            ReduceOutcome.INFRA_ERROR: (
                ExecutionStatus.INFRA_ERROR,
                ExecutionStatus.TIMED_OUT,
            ),
        }
        expected = expected_statuses.get(result.outcome)
        if expected is None:
            _reject_result(
                "reduce_result_inconsistent",
                f"unsupported reduction outcome: {result.outcome.value}",
            )
        if result.execution_status not in expected:
            _reject_result(
                "reduce_result_inconsistent",
                "reduction outcome does not match its execution status",
            )


def _validate_result_against_spec(
    spec: ExperimentSpec,
    result: ExperimentResult,
) -> None:
    if result.kind != spec.kind:
        _reject_result(
            "kind_mismatch",
            f"expected {spec.kind.value}, got {result.kind.value}",
        )
    expected_type_matches = (
        (
            spec.kind == ExperimentKind.COMPUTE_SANITIZER
            and isinstance(result, SanitizerSweepResult)
        )
        or (
            spec.kind == ExperimentKind.RANDOM_DELAY_STRESS
            and isinstance(result, StressTestResult)
        )
        or (
            spec.kind in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE)
            and isinstance(result, TraceExperimentResult)
        )
        or (
            spec.kind == ExperimentKind.REDUCE_DELAY_CONFIG
            and isinstance(result, ReduceExperimentResult)
        )
    )
    if not expected_type_matches:
        _reject_result(
            "kind_mismatch", "result payload type does not match the pending spec"
        )

    _validate_provenance(spec, result)
    if isinstance(result, SanitizerSweepResult):
        _validate_sanitizer_result(spec, result)
    elif isinstance(result, StressTestResult):
        _validate_stress_result(spec, result)
    elif isinstance(result, TraceExperimentResult):
        _validate_trace_result(result)
    elif isinstance(result, ReduceExperimentResult):
        _validate_reduce_result(spec, result)


def _inconclusive_report(session: AnalysisSession, reason: str) -> ExplainReport:
    reproduced = any(result.has_positive_signal for result in session.evidence.stress)
    return ExplainReport(
        root_cause="",
        race_class=RaceClass.UNKNOWN,
        confidence=Confidence.L,
        is_reproduced=reproduced,
        cross_validation=CrossValidation(agrees=False, notes=reason),
        evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
        evidence_refs=_artifact_refs(session.evidence),
        recommended_action="Review the preserved evidence and run a targeted experiment.",
        raw_ai_output="",
    )


def _experiment_record(
    spec: ExperimentSpec,
    result: ExperimentResult,
) -> ExperimentRecord:
    outcome = ""
    completeness = ""
    if isinstance(result, SanitizerSweepResult):
        outcome = result.outcome.value
        completeness = result.completeness.value
    elif isinstance(result, StressTestResult):
        outcome = result.outcome.value
        completeness = result.completeness.value
    elif isinstance(result, TraceExperimentResult):
        completeness = result.completeness.value
    elif isinstance(result, ReduceExperimentResult):
        outcome = result.outcome.value
    # Unknown/future result types record with empty outcome/completeness:
    # history recording must not abort a transition that validation (which
    # rejects kind/type mismatches above) has already accepted.
    return ExperimentRecord(
        experiment_id=result.experiment_id,
        kind=result.kind,
        phase=spec.phase,
        round_index=spec.round_index,
        execution_status=result.execution_status,
        outcome=outcome,
        completeness=completeness,
    )


def _triage_report(session: AnalysisSession, decision: TriageDecision) -> ExplainReport:
    """Terminal report for a session that finished without an AI turn.

    Finishing at triage means the findings were already known, suppressed, or
    absent -- not that the evidence says nothing happened. The report therefore
    reports the evidence as it stands and makes no root-cause claim, rather
    than asserting a clean run the evidence may contradict.
    """
    reproduced = any(result.has_positive_signal for result in session.evidence.stress)
    return ExplainReport(
        root_cause="",
        race_class=RaceClass.UNKNOWN,
        confidence=Confidence.L,
        is_reproduced=reproduced,
        cross_validation=CrossValidation(agrees=True, notes=decision.reason),
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        evidence_refs=_artifact_refs(session.evidence),
        recommended_action=f"No new action required: {decision.reason}",
        raw_ai_output="",
    )


def _initial_campaign_state(
    plan: CampaignPlan,
    shared_results: Sequence[SanitizerSweepResult],
    kernel: KernelId | None,
) -> tuple[
    EvidenceBundle,
    list[ExperimentSpec],
    list[str],
    list[ExperimentRecord],
]:
    shared_by_tool: dict[str, SanitizerSweepResult] = {}
    for result in shared_results:
        if not is_target_scoped_sanitizer(result.tool):
            raise ValueError(f"sanitizer tool is not target-scoped: {result.tool}")
        if result.tool in shared_by_tool:
            raise ValueError(f"duplicate shared sanitizer result: {result.tool}")
        shared_by_tool[result.tool] = result

    evidence = EvidenceBundle()
    for skipped in plan.skipped_results:
        evidence.add(skipped)
    pending: list[ExperimentSpec] = []
    completed_ids: list[str] = []
    history: list[ExperimentRecord] = []
    for spec in plan.experiments:
        tool = spec.sanitizer_tool
        shared = None if tool is None else shared_by_tool.pop(tool, None)
        if shared is None:
            pending.append(spec)
            continue
        if kernel is None:
            raise ValueError("shared sanitizer evidence requires a pinned kernel")
        projected = project_target_scoped_sanitizer_result(
            shared,
            experiment_id=spec.experiment_id,
            kernel=kernel,
        )
        _validate_result_against_spec(spec, projected)
        evidence.add(projected)
        completed_ids.append(projected.experiment_id)
        history.append(_experiment_record(spec, projected))
    if shared_by_tool:
        raise ValueError(
            "shared sanitizer results are not present in the child policy: "
            f"{', '.join(sorted(shared_by_tool))}"
        )
    return evidence, pending, completed_ids, history


def create_session(
    session_id: str,
    unit: WorkUnit,
    *,
    budget: Optional[SessionBudget] = None,
    campaign_policy: Optional[InitialCampaignPolicy] = None,
    shared_sanitizer_results: Sequence[SanitizerSweepResult] = (),
) -> Transition:
    kernel = unit.kernel
    policy = campaign_policy or InitialCampaignPolicy()
    plan = policy.plan(session_id, unit)
    active_budget = budget or SessionBudget()
    evidence, pending, completed_ids, history = _initial_campaign_state(
        plan,
        shared_sanitizer_results,
        kernel,
    )
    if len(pending) > active_budget.max_experiments:
        reason = (
            "initial experiment budget exhausted: "
            f"required={len(pending)}, "
            f"limit={active_budget.max_experiments}"
        )
        session = AnalysisSession(
            session_id=session_id,
            unit=unit,
            # Transient: _terminate_inconclusive replaces it immediately. The
            # pre-reasoner state is the honest placeholder now that entering
            # the reasoner is a decision triage has to make.
            status=SessionStatus.WAITING_TRIAGE,
            evidence=evidence,
            budget=active_budget,
            completed_experiment_ids=completed_ids,
            experiment_history=history,
        )
        return _terminate_inconclusive(session, reason)

    session = AnalysisSession(
        session_id=session_id,
        unit=unit,
        status=(
            SessionStatus.WAITING_INITIAL if pending else SessionStatus.WAITING_TRIAGE
        ),
        evidence=evidence,
        budget=active_budget,
        pending_experiments=pending,
        experiment_fingerprints=[experiment_spec_fingerprint(spec) for spec in pending],
        submitted_experiments=len(pending),
        completed_experiment_ids=completed_ids,
        experiment_history=history,
    )
    if pending:
        effects = (
            SessionEffect(
                kind=EffectKind.SUBMIT_EXPERIMENTS,
                experiments=tuple(pending),
            ),
        )
    else:
        effects = (SessionEffect(kind=EffectKind.RUN_TRIAGE),)
    return Transition(session=session, effects=effects)


def record_experiment_result(
    session: AnalysisSession,
    result: ExperimentResult,
) -> Transition:
    _validate_session_result_indexes(session)
    if result.experiment_id in session.completed_experiment_ids:
        accepted = next(
            item
            for item in _evidence_results(session.evidence)
            if item.experiment_id == result.experiment_id
        )
        if accepted.to_dict() == result.to_dict():
            return Transition(session=session, effects=())
        _reject_result(
            "duplicate_conflict",
            f"experiment {result.experiment_id} already completed with a different payload",
        )

    matches = [
        spec
        for spec in session.pending_experiments
        if spec.experiment_id == result.experiment_id
    ]
    if not matches:
        _reject_result(
            "experiment_not_pending",
            f"unexpected experiment result: {result.experiment_id}",
        )
    if len(matches) != 1:
        _reject_result(
            "session_state_corrupt",
            f"experiment {result.experiment_id} resolves to {len(matches)} pending specs",
        )
    spec = matches[0]
    _validate_spec_context(session, spec)
    _validate_result_against_spec(spec, result)

    # Clone only after all validation succeeds, so a rejected callback cannot
    # partially consume pending state or append untrusted evidence.
    updated = _clone(session)
    updated.evidence.add(result)
    updated.pending_experiments = [
        spec
        for spec in updated.pending_experiments
        if spec.experiment_id != result.experiment_id
    ]
    updated.completed_experiment_ids.append(result.experiment_id)
    # Record while the spec is still known: phase and round cannot be recovered
    # afterwards, and inferring them from the result kind would mislabel a
    # follow-up sanitizer run as part of the initial campaign.
    updated.experiment_history.append(_experiment_record(spec, result))
    if updated.pending_experiments:
        return Transition(session=updated, effects=())
    if spec.phase == ExperimentPhase.INITIAL:
        # Completeness, correlation, suppression, and baseline all run before
        # the reasoner is ever entered; the reasoner is for evidence that
        # survived triage, not for deciding whether there is any.
        updated.status = SessionStatus.WAITING_TRIAGE
        return Transition(
            session=updated,
            effects=(SessionEffect(kind=EffectKind.RUN_TRIAGE),),
        )
    updated.status = SessionStatus.WAITING_REASONER
    return Transition(
        session=updated,
        effects=(SessionEffect(kind=EffectKind.RUN_REASONER),),
    )


def record_triage_decision(
    session: AnalysisSession,
    decision: TriageDecision,
) -> Transition:
    """Apply the deterministic pre-reasoning verdict.

    Only ``ANALYZE`` costs an AI turn. A campaign that is clean, unchanged, or
    entirely suppressed finishes here, and an incomplete one becomes
    inconclusive rather than being handed to the reasoner as if it were clean.
    """
    if session.status != SessionStatus.WAITING_TRIAGE:
        raise ValueError(f"cannot triage session in state: {session.status}")

    if decision.outcome == TriageOutcome.INCONCLUSIVE:
        return _terminate_inconclusive(
            session,
            decision.reason,
            triage=decision,
        )

    updated = _clone(session)
    updated.triage = decision
    if decision.outcome == TriageOutcome.FINISH:
        updated.status = SessionStatus.COMPLETED
        updated.termination_reason = decision.reason
        updated.report = _triage_report(updated, decision)
        updated.report_publication = ReportPublicationStatus.PENDING
        return Transition(
            session=updated,
            effects=(SessionEffect(kind=EffectKind.PUBLISH_REPORT),),
        )

    if decision.outcome != TriageOutcome.ANALYZE:
        # Falling through to the reasoner would turn an unhandled outcome into
        # a silent AI turn, which is the cost triage exists to control.
        raise ValueError(f"unsupported triage outcome: {decision.outcome}")

    updated.status = SessionStatus.WAITING_REASONER
    return Transition(
        session=updated,
        effects=(SessionEffect(kind=EffectKind.RUN_REASONER),),
    )


def _terminate_inconclusive(
    session: AnalysisSession,
    reason: str,
    *,
    triage: Optional[TriageDecision] = None,
) -> Transition:
    updated = _clone(session)
    if triage is not None:
        updated.triage = triage
    updated.status = SessionStatus.INCONCLUSIVE
    updated.termination_reason = reason
    updated.report = _inconclusive_report(updated, reason)
    updated.report_publication = ReportPublicationStatus.PENDING
    return Transition(
        session=updated,
        effects=(SessionEffect(kind=EffectKind.PUBLISH_REPORT),),
    )


def _cited_complete_sanitizer_results(
    evidence: EvidenceBundle, report: ExplainReport
) -> List[SanitizerSweepResult]:
    return [
        result
        for result in evidence.sanitizer
        if result.execution_status == ExecutionStatus.SUCCEEDED
        and result.outcome == SanitizerOutcome.FINDING
        and result.completeness == ResultCompleteness.COMPLETE
        and not result.output_truncated
        and not result.summary_issues
        and result.log is not None
        and result.log.uri in report.evidence_refs
    ]


def _has_cited_sanitizer_diagnosis(
    evidence: EvidenceBundle, report: ExplainReport
) -> bool:
    """Support a diagnosis without a race class, not a claim of race freedom."""
    return any(
        finding.error_type
        in (
            ERROR_TYPE_UNINITIALIZED_READ,
            ERROR_TYPE_INVALID_MEMORY_ACCESS,
            ERROR_TYPE_MISALIGNED_ACCESS,
            ERROR_TYPE_MEMORY_LEAK,
        )
        for result in _cited_complete_sanitizer_results(evidence, report)
        for finding in result.findings
    )


def _raw_trace_claim_is_unsupported(
    evidence: EvidenceBundle, report: ExplainReport
) -> bool:
    if any(
        result.tool == "racecheck"
        and any(
            finding.error_type == ERROR_TYPE_RACE_CONDITION
            and finding.count > 0
            and (finding.error_detail.strip() or finding.raw.strip())
            for finding in result.findings
        )
        for result in _cited_complete_sanitizer_results(evidence, report)
    ):
        return False
    limited_raw_evidence = False
    cited = set(report.evidence_refs)
    for result in evidence.traces:
        analysis = result.analysis
        trace_cited = (
            analysis is not None
            and analysis.artifact is not None
            and analysis.artifact.uri in cited
        ) or any(trace.uri in cited for trace in result.traces)
        if analysis is None:
            limited_raw_evidence |= trace_cited
            continue
        summaries = [
            summary
            for summary in analysis.detector_summaries
            if summary.source_tool == "cutracer"
            and summary.sub_component in ("data_race_raw", "global_mem_raw")
        ]
        complete = (
            result.execution_status == ExecutionStatus.SUCCEEDED
            and result.completeness == ResultCompleteness.COMPLETE
            and analysis.execution_status == ExecutionStatus.SUCCEEDED
        )
        limited_raw_evidence |= (trace_cited and not complete) or any(
            not complete
            or summary.verdict == "indeterminate"
            or summary.severity == TraceAnalysisSeverity.WARNING
            for summary in summaries
        )
        if (
            complete
            and trace_cited
            and any(
                summary.verdict == "violations_found"
                and summary.severity == TraceAnalysisSeverity.ERROR
                and summary.finding_count > 0
                for summary in summaries
            )
        ):
            return False
    return limited_raw_evidence


def _analysis_report_rejection(
    session: AnalysisSession,
    decision: AnalysisDecision,
) -> Optional[str]:
    report = decision.report
    if report is None:
        return None
    available = set(_artifact_refs(session.evidence))
    foreign = sorted(set(report.evidence_refs) - available)
    if foreign:
        return "reasoner cited artifacts outside this session: " + ", ".join(foreign)
    if decision.kind != DecisionKind.FINAL:
        return None
    if report.evidence_sufficiency != EvidenceSufficiency.SUFFICIENT:
        return "reasoner final claim is not supported by sufficient evidence"
    if report.race_class == RaceClass.UNKNOWN:
        return "reasoner final claim has an unknown race class"
    if report.race_class == RaceClass.RAW and _raw_trace_claim_is_unsupported(
        session.evidence, report
    ):
        return (
            "reasoner RAW diagnosis relies on incomplete ordering evidence; "
            "no cited complete RAW finding supports the mechanism"
        )
    if (
        report.race_class == RaceClass.NONE
        and report.root_cause.strip()
        and not _has_cited_sanitizer_diagnosis(session.evidence, report)
    ):
        return (
            "reasoner reported a root cause without a race class "
            "or a cited complete sanitizer finding"
        )
    if report.race_class != RaceClass.NONE and not report.root_cause.strip():
        return "reasoner classified a race without a root-cause explanation"
    return None


def _record_reasoner_turn(
    session: AnalysisSession,
    decision: AnalysisDecision,
) -> tuple[AnalysisSession, Optional[Transition]]:
    updated = _clone(session)
    updated.turns.append(AnalysisTurn(turn_index=len(updated.turns), decision=decision))
    if session.round_index == 0 and initial_evidence_requires_inconclusive(
        session.evidence
    ):
        reason = "initial evidence is incomplete"
        updated.turns[-1].guard_notes.append(reason)
        return updated, _terminate_inconclusive(updated, reason)

    report_rejection = _analysis_report_rejection(session, decision)
    if report_rejection is None:
        return updated, None
    updated.turns[-1].guard_notes.append(report_rejection)
    return updated, _terminate_inconclusive(updated, report_rejection)


def record_analysis_decision(
    session: AnalysisSession,
    decision: AnalysisDecision,
    *,
    guard_policy: Optional[ExperimentGuardPolicy] = None,
) -> Transition:
    if session.status != SessionStatus.WAITING_REASONER:
        raise ValueError(f"reasoner result in invalid state: {session.status}")
    updated, rejected = _record_reasoner_turn(session, decision)
    if rejected is not None:
        return rejected

    if decision.kind == DecisionKind.FINAL:
        if decision.report is None:
            raise ValueError("FINAL decision must include a report")
        updated.status = SessionStatus.COMPLETED
        updated.report = decision.report
        updated.termination_reason = "reasoner finalized the report"
        updated.report_publication = ReportPublicationStatus.PENDING
        return Transition(
            session=updated,
            effects=(SessionEffect(kind=EffectKind.PUBLISH_REPORT),),
        )
    if decision.kind == DecisionKind.INCONCLUSIVE:
        if decision.report is None:
            raise ValueError("INCONCLUSIVE decision must include a report")
        updated.status = SessionStatus.INCONCLUSIVE
        updated.report = decision.report
        updated.termination_reason = decision.rationale
        updated.report_publication = ReportPublicationStatus.PENDING
        return Transition(
            session=updated,
            effects=(SessionEffect(kind=EffectKind.PUBLISH_REPORT),),
        )
    if decision.kind != DecisionKind.FOLLOWUP_REQUIRED:
        raise ValueError(f"unsupported decision kind: {decision.kind}")
    if updated.round_index >= updated.budget.max_rounds:
        return _terminate_inconclusive(updated, "follow-up round budget exhausted")

    guard = guard_policy or ExperimentGuardPolicy()
    guarded = guard.expand(updated, decision.requests)
    updated.turns[-1].guard_notes.extend((*guarded.notes, *guarded.rejections))
    if not guarded.experiments:
        reason = "no new valid follow-up experiment"
        if guarded.rejections:
            reason += ": " + "; ".join(guarded.rejections)
        return _terminate_inconclusive(updated, reason)
    if (
        updated.submitted_experiments + len(guarded.experiments)
        > updated.budget.max_experiments
    ):
        return _terminate_inconclusive(updated, "experiment budget exhausted")

    updated.round_index += 1
    updated.status = SessionStatus.WAITING_FOLLOWUP
    updated.pending_experiments = list(guarded.experiments)
    updated.experiment_fingerprints.extend(guarded.fingerprints)
    updated.submitted_experiments += len(guarded.experiments)
    return Transition(
        session=updated,
        effects=(
            SessionEffect(
                kind=EffectKind.SUBMIT_EXPERIMENTS,
                experiments=guarded.experiments,
            ),
        ),
    )


def record_report_published(session: AnalysisSession) -> Transition:
    """Acknowledge successful delivery without changing diagnosis semantics."""
    if session.status not in (SessionStatus.COMPLETED, SessionStatus.INCONCLUSIVE):
        raise ValueError(f"cannot publish report in state: {session.status}")
    if session.report is None:
        raise ValueError("cannot publish a missing report")
    if session.report_publication == ReportPublicationStatus.PUBLISHED:
        return Transition(session=session, effects=())
    if session.report_publication != ReportPublicationStatus.PENDING:
        raise ValueError(
            f"report is not pending publication: {session.report_publication}"
        )
    updated = _clone(session)
    updated.report_publication = ReportPublicationStatus.PUBLISHED
    return Transition(session=updated, effects=())


def build_session_report(session: AnalysisSession) -> SessionReport:
    if session.report is None:
        raise ValueError("session has no terminal report")
    if session.status not in (SessionStatus.COMPLETED, SessionStatus.INCONCLUSIVE):
        raise ValueError(f"session is not terminal: {session.status}")
    return SessionReport(
        session_id=session.session_id,
        status=session.status,
        explanation=session.report,
        evidence=session.evidence,
        turns=session.turns,
        termination_reason=session.termination_reason,
        unit=session.unit,
        triage=session.triage,
        experiments=list(session.experiment_history),
        artifact_citations=_artifact_refs(session.evidence),
        triggering_configs=list(
            dict.fromkeys(
                result.triggering_config.artifact.uri
                for result in session.evidence.stress
                if result.triggering_config is not None
            )
        ),
        minimized_configs=list(
            dict.fromkeys(
                result.minimized_config.uri
                for result in session.evidence.reductions
                if result.minimized_config is not None
            )
        ),
        rounds_used=(
            max(record.round_index for record in session.experiment_history) + 1
            if session.experiment_history
            else 0
        ),
    )
