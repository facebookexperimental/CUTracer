"""Event-driven ports for the MAST/Sandcastle deployment driver."""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Protocol

from cutracer.service.contracts import (
    AnalysisDecision,
    AnalysisSession,
    ExperimentEnvelope,
    ExperimentResult,
    ExperimentSpec,
    SessionBudget,
    SessionReport,
    SessionStatus,
    WorkUnit,
)
from cutracer.service.policy import ExperimentGuardPolicy, InitialCampaignPolicy
from cutracer.service.session import (
    build_session_report,
    create_session,
    EffectKind,
    ExperimentResultValidationError,
    record_analysis_decision,
    record_experiment_result,
    record_triage_decision,
    SessionEffect,
    Transition,
)
from cutracer.service.storage import DurableSessionStore, StoreConflict
from cutracer.service.triage import TriagePolicy

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ExperimentCallbackResult:
    session: AnalysisSession
    accepted: bool


@dataclasses.dataclass(frozen=True)
class ReasoningCallbackResult:
    session: AnalysisSession
    accepted: bool


class MastDispatcher(Protocol):
    """Submit by revision-bound dispatch key; implementations are idempotent."""

    def submit(
        self,
        session_id: str,
        state_revision: int,
        experiments: tuple[ExperimentSpec, ...],
    ) -> None: ...


class SandcastleDispatcher(Protocol):
    """Submit by (session_id, state_revision); implementations must be idempotent."""

    def submit_reasoning_turn(self, session_id: str, state_revision: int) -> None: ...


class DistributedReportSink(Protocol):
    """Publish by session_id; implementations must be idempotent."""

    def publish(self, report: SessionReport) -> None: ...


class DistributedCoordinator:
    """CAS session state before dispatching replayable, idempotent effects."""

    _MAX_CAS_ATTEMPTS = 8

    def __init__(
        self,
        *,
        store: DurableSessionStore,
        mast: MastDispatcher,
        sandcastle: SandcastleDispatcher,
        reports: DistributedReportSink,
        guard_policy: ExperimentGuardPolicy | None = None,
        triage_policy: TriagePolicy | None = None,
    ) -> None:
        self._store = store
        self._mast = mast
        self._sandcastle = sandcastle
        self._reports = reports
        self._guard = guard_policy or ExperimentGuardPolicy()
        self._triage = triage_policy or TriagePolicy()

    def _dispatch(self, transition: Transition) -> AnalysisSession:
        session = transition.session
        # Validated before anything is dispatched: a violation discovered
        # mid-loop would leave a submitted MAST job behind an error.
        pending_triage = any(
            effect.kind == EffectKind.RUN_TRIAGE for effect in transition.effects
        )
        if pending_triage and len(transition.effects) != 1:
            raise ValueError("RUN_TRIAGE must be the only effect in a transition")
        for effect in transition.effects:
            if effect.kind == EffectKind.SUBMIT_EXPERIMENTS:
                self._mast.submit(
                    session.session_id,
                    session.state_revision,
                    effect.experiments,
                )
            elif effect.kind == EffectKind.RUN_TRIAGE:
                # Pure computation with no remote effect: resolved after the
                # loop and committed through its own CAS round.
                continue
            elif effect.kind == EffectKind.RUN_REASONER:
                self._sandcastle.submit_reasoning_turn(
                    session.session_id, session.state_revision
                )
            elif effect.kind == EffectKind.PUBLISH_REPORT:
                self._reports.publish(build_session_report(session))
            else:
                raise ValueError(f"unknown effect kind: {effect.kind}")
        if pending_triage:
            return self._apply(
                session.session_id,
                self._record_triage_if_pending,
            )
        return session

    def _record_triage_if_pending(self, current: AnalysisSession) -> Transition:
        if current.status != SessionStatus.WAITING_TRIAGE:
            return Transition(session=current, effects=())
        return record_triage_decision(current, self._triage.decide(current))

    def _apply(
        self,
        session_id: str,
        reducer: Callable[[AnalysisSession], Transition],
    ) -> AnalysisSession:
        for _attempt in range(self._MAX_CAS_ATTEMPTS):
            current = self._store.load(session_id)
            transition = reducer(current)
            if (
                not transition.effects
                and transition.session.to_dict() == current.to_dict()
            ):
                return current
            transition.session.state_revision = current.state_revision + 1
            try:
                self._store.compare_and_swap(
                    session_id,
                    current.state_revision,
                    transition.session,
                )
                return self._dispatch(transition)
            except StoreConflict:
                continue
        raise RuntimeError(
            f"session update contention exceeded retry limit: {session_id}"
        )

    def start(
        self,
        session_id: str,
        unit: WorkUnit,
        *,
        budget: SessionBudget | None = None,
        campaign_policy: InitialCampaignPolicy | None = None,
    ) -> AnalysisSession:
        transition = create_session(
            session_id,
            unit,
            budget=budget,
            campaign_policy=campaign_policy,
        )
        transition.session.state_revision = 1
        try:
            self._store.create(transition.session)
        except StoreConflict:
            return self.resume(session_id)
        return self._dispatch(transition)

    def resume(self, session_id: str) -> AnalysisSession:
        session = self._store.load(session_id)
        if session.status in (
            SessionStatus.WAITING_INITIAL,
            SessionStatus.WAITING_FOLLOWUP,
        ):
            effect = SessionEffect(
                EffectKind.SUBMIT_EXPERIMENTS,
                tuple(session.pending_experiments),
            )
        elif session.status == SessionStatus.WAITING_TRIAGE:
            effect = SessionEffect(EffectKind.RUN_TRIAGE)
        elif session.status == SessionStatus.WAITING_REASONER:
            effect = SessionEffect(EffectKind.RUN_REASONER)
        elif session.status in (SessionStatus.COMPLETED, SessionStatus.INCONCLUSIVE):
            effect = SessionEffect(EffectKind.PUBLISH_REPORT)
        else:
            raise ValueError(f"cannot resume session in state: {session.status}")
        return self._dispatch(Transition(session, (effect,)))

    def on_experiment_result(
        self, session_id: str, result: ExperimentResult
    ) -> AnalysisSession:
        return self._apply(
            session_id,
            lambda session: record_experiment_result(session, result),
        )

    def on_experiment_envelope(
        self,
        envelope: ExperimentEnvelope,
    ) -> ExperimentCallbackResult:
        accepted = False

        def apply_envelope(session: AnalysisSession) -> Transition:
            nonlocal accepted
            if envelope.analyzed_state_revision > session.state_revision:
                raise ExperimentResultValidationError(
                    "future_dispatch_revision",
                    "experiment callback was dispatched from a future session revision",
                )
            accepted = envelope.experiment_id not in session.completed_experiment_ids
            return record_experiment_result(session, envelope.result)

        session = self._apply(envelope.session_id, apply_envelope)
        return ExperimentCallbackResult(session=session, accepted=accepted)

    def on_reasoning_decision(
        self,
        session_id: str,
        state_revision: int,
        decision: AnalysisDecision,
    ) -> ReasoningCallbackResult:
        accepted = False

        def apply_decision(session: AnalysisSession) -> Transition:
            nonlocal accepted
            if session.state_revision != state_revision:
                accepted = False
                logger.info(
                    "Discarding stale reasoning decision: session=%s "
                    "expected_revision=%d current_revision=%d",
                    session.session_id,
                    state_revision,
                    session.state_revision,
                )
                return Transition(session, ())
            accepted = True
            return record_analysis_decision(session, decision, guard_policy=self._guard)

        session = self._apply(
            session_id,
            apply_decision,
        )
        return ReasoningCallbackResult(session=session, accepted=accepted)
