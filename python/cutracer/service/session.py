"""Pure, backend-independent analysis-session state machine."""

from __future__ import annotations

import dataclasses
from collections import Counter
from enum import Enum
from typing import List, NoReturn, Optional

from cutracer.service.contracts import (
    AnalysisDecision,
    AnalysisSession,
    AnalysisTurn,
    Confidence,
    CrossValidation,
    DecisionKind,
    EvidenceBundle,
    EvidenceSufficiency,
    ExperimentKind,
    ExperimentPhase,
    ExperimentResult,
    ExperimentSpec,
    ExplainReport,
    ReduceExperimentResult,
    SanitizerOutcome,
    SanitizerSweepResult,
    SessionBudget,
    SessionReport,
    SessionStatus,
    StressOutcome,
    StressTestResult,
    TraceExperimentResult,
    TriggeringDelayConfig,
    WorkUnit,
)
from cutracer.service.policy import (
    experiment_spec_fingerprint,
    ExperimentGuardPolicy,
    InitialCampaignPolicy,
)


class EffectKind(str, Enum):
    SUBMIT_EXPERIMENTS = "submit_experiments"
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


def _clone(session: AnalysisSession) -> AnalysisSession:
    return AnalysisSession.from_dict(session.to_dict())


def _artifact_refs(evidence: EvidenceBundle) -> List[str]:
    refs: List[str] = []
    for result in evidence.sanitizer:
        if result.log is not None:
            refs.append(result.log.uri)
    for result in evidence.stress:
        if result.triggering_config is not None:
            refs.append(result.triggering_config.artifact.uri)
        if result.log is not None:
            refs.append(result.log.uri)
    for result in evidence.traces:
        if result.trace is not None:
            refs.append(result.trace.uri)
    for result in evidence.reductions:
        if result.minimized_config is not None:
            refs.append(result.minimized_config.uri)
        if result.report is not None:
            refs.append(result.report.uri)
    return refs


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


def _validate_stress_result(
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
    if result.triggering_config is not None:
        if result.reproductions == 0:
            _reject_result(
                "trigger_config_mismatch",
                "triggering config has no reproduced trial",
            )
        _validate_triggering_config(spec, result, result.triggering_config)
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


def _validate_reduce_result(
    spec: ExperimentSpec,
    result: ReduceExperimentResult,
) -> None:
    assert spec.reduction is not None
    if result.input_config != spec.reduction.triggering_config:
        _reject_result(
            "reduce_input_config_mismatch",
            "reduction input differs from the pending triggering config",
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
    elif isinstance(result, ReduceExperimentResult):
        _validate_reduce_result(spec, result)


def _inconclusive_report(session: AnalysisSession, reason: str) -> ExplainReport:
    reproduced = any(result.has_positive_signal for result in session.evidence.stress)
    return ExplainReport(
        root_cause="",
        race_class="unknown",
        confidence=Confidence.L,
        is_reproduced=reproduced,
        cross_validation=CrossValidation(agrees=False, notes=reason),
        evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
        evidence_refs=_artifact_refs(session.evidence),
        recommended_action="Review the preserved evidence and run a targeted experiment.",
        raw_ai_output="",
    )


def create_session(
    session_id: str,
    unit: WorkUnit,
    *,
    budget: Optional[SessionBudget] = None,
    campaign_policy: Optional[InitialCampaignPolicy] = None,
) -> Transition:
    policy = campaign_policy or InitialCampaignPolicy()
    plan = policy.plan(session_id, unit)
    active_budget = budget or SessionBudget()
    evidence = EvidenceBundle()
    for skipped in plan.skipped_results:
        evidence.add(skipped)
    if len(plan.experiments) > active_budget.max_experiments:
        reason = (
            "initial experiment budget exhausted: "
            f"required={len(plan.experiments)}, "
            f"limit={active_budget.max_experiments}"
        )
        session = AnalysisSession(
            session_id=session_id,
            unit=unit,
            status=SessionStatus.WAITING_REASONER,
            evidence=evidence,
            budget=active_budget,
        )
        return _terminate_inconclusive(session, reason)

    pending = list(plan.experiments)
    session = AnalysisSession(
        session_id=session_id,
        unit=unit,
        status=(
            SessionStatus.WAITING_INITIAL if pending else SessionStatus.WAITING_REASONER
        ),
        evidence=evidence,
        budget=active_budget,
        pending_experiments=pending,
        experiment_fingerprints=[
            experiment_spec_fingerprint(spec) for spec in plan.experiments
        ],
        submitted_experiments=len(plan.experiments),
    )
    if plan.experiments:
        effects = (
            SessionEffect(
                kind=EffectKind.SUBMIT_EXPERIMENTS,
                experiments=plan.experiments,
            ),
        )
    else:
        effects = (SessionEffect(kind=EffectKind.RUN_REASONER),)
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
    if updated.pending_experiments:
        return Transition(session=updated, effects=())
    updated.status = SessionStatus.WAITING_REASONER
    return Transition(
        session=updated,
        effects=(SessionEffect(kind=EffectKind.RUN_REASONER),),
    )


def _terminate_inconclusive(session: AnalysisSession, reason: str) -> Transition:
    updated = _clone(session)
    updated.status = SessionStatus.INCONCLUSIVE
    updated.termination_reason = reason
    updated.report = _inconclusive_report(updated, reason)
    return Transition(
        session=updated,
        effects=(SessionEffect(kind=EffectKind.PUBLISH_REPORT),),
    )


def record_analysis_decision(
    session: AnalysisSession,
    decision: AnalysisDecision,
    *,
    guard_policy: Optional[ExperimentGuardPolicy] = None,
) -> Transition:
    updated = _clone(session)
    if updated.status != SessionStatus.WAITING_REASONER:
        raise ValueError(f"reasoner result in invalid state: {updated.status}")
    updated.turns.append(AnalysisTurn(turn_index=len(updated.turns), decision=decision))

    if decision.kind == DecisionKind.FINAL:
        if decision.report is None:
            raise ValueError("FINAL decision must include a report")
        updated.status = SessionStatus.COMPLETED
        updated.report = decision.report
        updated.termination_reason = "reasoner finalized the report"
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
    )
