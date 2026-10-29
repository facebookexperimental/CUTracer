# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Pure aggregation of one request's sessions and requested delivery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cutracer.service.contracts import (
    AnalysisSession,
    ReportPublicationStatus,
    SessionStatus,
    WorkUnit,
)
from cutracer.service.discovery_plan import (
    discovery_child_session_id,
    pin_discovered_work_units,
)
from cutracer.service.run_contracts import (
    RunInput,
    RunInputConflict,
    RunOutputMode,
    RunPublicationStatus,
    RunSessionSummary,
    RunStatus,
    RunSummary,
)


_TERMINAL = frozenset(
    (SessionStatus.COMPLETED, SessionStatus.INCONCLUSIVE, SessionStatus.FAILED)
)


def expected_run_units(
    request: RunInput, root: AnalysisSession | None
) -> dict[str, WorkUnit]:
    expected = {request.root_session_id: request.unit}
    if root is None:
        return expected
    if (
        root.session_id != request.root_session_id
        or root.unit.to_dict() != request.unit.to_dict()
    ):
        raise RunInputConflict("root session does not match this Run")
    if root.discovery_plan is not None and root.evidence.discovery:
        manifest = root.evidence.discovery[0].discovery
        if manifest is not None:
            expected.update(
                (discovery_child_session_id(root.session_id, unit), unit)
                for unit in pin_discovered_work_units(root.unit, manifest.kernels)
            )
    if set(root.child_session_ids) - expected.keys():
        raise RunInputConflict("stored children do not match the discovery manifest")
    return expected


def summarize_session(session: AnalysisSession) -> RunSessionSummary:
    report = session.report
    snapshot = None if session.triage is None else session.triage.snapshot
    return RunSessionSummary(
        session_id=session.session_id,
        unit_id=session.unit.unit_id,
        status=session.status,
        kernel=session.unit.kernel,
        termination_reason=session.termination_reason,
        report_publication=session.report_publication,
        report_publication_error=session.report_publication_error,
        has_report=report is not None,
        report_uri=""
        if session.published_report is None
        else session.published_report.uri,
        report_sha256=""
        if session.published_report is None
        else (session.published_report.sha256 or ""),
        race_class=None if report is None else report.race_class,
        evidence_sufficiency=None if report is None else report.evidence_sufficiency,
        initial_evidence_complete=(
            snapshot is not None and snapshot.initial_evidence_complete
        ),
        sanitizer_positive_signal=any(
            result.has_usable_positive_signal for result in session.evidence.sanitizer
        ),
        stress_positive_signal=any(
            result.has_positive_signal for result in session.evidence.stress
        ),
        pending_experiment_ids=tuple(session.pending_experiment_ids),
    )


def build_run_summary(
    request: RunInput,
    sessions: Sequence[AnalysisSession],
    expected_units: Mapping[str, WorkUnit],
    *,
    now_ms: int,
    publication: RunPublicationStatus | None = None,
    error: str = "",
) -> RunSummary:
    by_id = {session.session_id: session for session in sessions}
    if len(by_id) != len(sessions) or request.root_session_id not in expected_units:
        raise ValueError("run summary requires unique sessions and the requested root")
    if set(by_id) - expected_units.keys():
        raise ValueError("run summary contains a session outside this request")
    if any(
        session.unit.to_dict() != expected_units[session.session_id].to_dict()
        for session in sessions
    ):
        raise RunInputConflict("session unit differs from the planned Run child")
    rows = tuple(
        summarize_session(by_id[session_id])
        if session_id in by_id
        else RunSessionSummary(
            session_id=session_id,
            unit_id=unit.unit_id,
            kernel=unit.kernel,
            status=None,
            termination_reason="not started",
        )
        for session_id, unit in expected_units.items()
    )
    root = by_id.get(request.root_session_id)
    root_failed = root is not None and root.status == SessionStatus.FAILED
    missing = len(by_id) != len(expected_units)
    active = any(session.status not in _TERMINAL for session in sessions)
    pending = active or (missing and not root_failed)
    failed = any(session.status == SessionStatus.FAILED for session in sessions)

    if request.output.mode == RunOutputMode.RETURN_ONLY:
        delivery = RunPublicationStatus.NOT_REQUESTED
    elif request.output.mode == RunOutputMode.FILE:
        delivery = publication or RunPublicationStatus.PENDING
    else:
        reports = [session for session in sessions if session.report is not None]
        if any(
            session.report_publication == ReportPublicationStatus.FAILED
            for session in reports
        ):
            delivery = RunPublicationStatus.FAILED
        elif pending or any(
            session.report_publication != ReportPublicationStatus.PUBLISHED
            for session in reports
        ):
            delivery = RunPublicationStatus.PENDING
        elif reports:
            delivery = RunPublicationStatus.PUBLISHED
        else:
            delivery = RunPublicationStatus.NOT_REQUESTED

    pending |= delivery == RunPublicationStatus.PENDING
    failed |= delivery == RunPublicationStatus.FAILED or (root_failed and missing)
    if pending and now_ms >= request.deadline_ms:
        status = RunStatus.EXPIRED
        error = error or "run deadline reached; outstanding jobs were not cancelled"
    elif pending:
        status = RunStatus.RUNNING
    elif failed:
        status = RunStatus.FAILED
    else:
        status = RunStatus.COMPLETED
    return RunSummary(
        run_id=request.run_id,
        root_session_id=request.root_session_id,
        status=status,
        sessions=rows,
        publication=delivery,
        error=error,
    )
