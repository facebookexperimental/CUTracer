# pyre-strict
"""Pure, backend-independent analysis-session state machine."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import List, Optional

from cutracer.service.contracts import (
    AnalysisDecision,
    AnalysisSession,
    AnalysisTurn,
    Confidence,
    CrossValidation,
    DecisionKind,
    EvidenceBundle,
    EvidenceSufficiency,
    ExperimentResult,
    ExperimentSpec,
    ExplainReport,
    SessionBudget,
    SessionReport,
    SessionStatus,
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


def _inconclusive_report(session: AnalysisSession, reason: str) -> ExplainReport:
    reproduced = any(result.reproductions > 0 for result in session.evidence.stress)
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
    evidence = EvidenceBundle()
    for skipped in plan.skipped_results:
        evidence.add(skipped)
    pending = list(plan.experiments)
    session = AnalysisSession(
        session_id=session_id,
        unit=unit,
        status=(
            SessionStatus.WAITING_INITIAL if pending else SessionStatus.WAITING_REASONER
        ),
        evidence=evidence,
        budget=budget or SessionBudget(),
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
    updated = _clone(session)
    if result.experiment_id in updated.completed_experiment_ids:
        return Transition(session=updated, effects=())
    pending_ids = updated.pending_experiment_ids
    if result.experiment_id not in pending_ids:
        raise ValueError(f"unexpected experiment result: {result.experiment_id}")
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
