"""Bounded operational signals derived from a published diagnosis report."""

from __future__ import annotations

import dataclasses
from enum import Enum

from cutracer.service.blind_spots import BlindSpotCatalog
from cutracer.service.contracts import (
    BaselineVerdict,
    EvidenceSufficiency,
    ExecutionStatus,
    ExperimentResult,
    ResultCompleteness,
    SessionReport,
    SessionStatus,
    TraceExperimentResult,
)


class DiagnosisDisposition(str, Enum):
    ACTIONABLE = "actionable"
    NO_CHANGE = "no_change"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"


class ServiceAlert(str, Enum):
    NONE = "none"
    DIAGNOSIS_FAILED = "diagnosis_failed"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"


@dataclasses.dataclass(frozen=True)
class OperationalSignals:
    disposition: DiagnosisDisposition
    service_alert: ServiceAlert
    new_finding_count: int
    unbaselined_finding_count: int
    suppressed_finding_count: int
    incomplete_scope_count: int
    infra_error_experiment_count: int
    timed_out_experiment_count: int
    failed_experiment_count: int
    stress_infra_error_trial_count: int
    stress_timed_out_trial_count: int
    evidence_duration_ms: int
    initial_evidence_complete: bool
    sanitizer_positive_signal: bool
    stress_positive_signal: bool
    quality_slo_evaluable: bool
    quality_slo_met: bool
    blind_spot_ids: tuple[str, ...]
    blind_spot_catalog_revision: str

    @classmethod
    def from_report(
        cls,
        report: SessionReport,
        blind_spots: BlindSpotCatalog | None = None,
    ) -> OperationalSignals:
        snapshot = None if report.triage is None else report.triage.snapshot
        new_keys: set[tuple[str, str]] = set()
        unbaselined_keys: set[tuple[str, str]] = set()
        incomplete_scopes = 0
        suppressed = 0
        if snapshot is not None:
            for scope in snapshot.scopes:
                if scope.baseline_verdict in (
                    BaselineVerdict.NO_BASELINE,
                    BaselineVerdict.NOT_EVALUATED,
                ):
                    unbaselined_keys.update(
                        (scope.scope.key, key) for key in scope.observed_keys
                    )
                else:
                    new_keys.update((scope.scope.key, key) for key in scope.new_keys)
                incomplete_scopes += int(
                    scope.completeness != ResultCompleteness.COMPLETE
                )
            suppressed = len(snapshot.suppression_audit)
        sanitizer_positive_signal = any(
            result.has_usable_positive_signal for result in report.evidence.sanitizer
        )
        stress_positive_signal = any(
            result.has_positive_signal for result in report.evidence.stress
        )
        initial_complete = snapshot is not None and snapshot.initial_evidence_complete
        counts = _execution_counts(report)
        actionable = bool(new_keys or unbaselined_keys or stress_positive_signal)
        alert = _service_alert(report, initial_complete, counts)
        execution_failed = report.status == SessionStatus.FAILED or any(
            (counts.infra_errors, counts.timed_out, counts.failed)
        )
        disposition = _disposition(
            report,
            actionable,
            alert,
            execution_failed=execution_failed,
        )
        matching = ()
        catalog_revision = ""
        if blind_spots is not None:
            catalog_revision = blind_spots.catalog_revision
            if report.unit is not None:
                matching = blind_spots.matching_entries(report.unit)
        evaluable = snapshot is not None
        return cls(
            disposition=disposition,
            service_alert=alert,
            new_finding_count=len(new_keys),
            unbaselined_finding_count=len(unbaselined_keys),
            suppressed_finding_count=suppressed,
            incomplete_scope_count=incomplete_scopes,
            infra_error_experiment_count=counts.infra_errors,
            timed_out_experiment_count=counts.timed_out,
            failed_experiment_count=counts.failed,
            stress_infra_error_trial_count=counts.stress_infra_errors,
            stress_timed_out_trial_count=counts.stress_timed_out,
            evidence_duration_ms=_evidence_duration_ms(report),
            initial_evidence_complete=initial_complete,
            sanitizer_positive_signal=sanitizer_positive_signal,
            stress_positive_signal=stress_positive_signal,
            quality_slo_evaluable=evaluable,
            quality_slo_met=(
                evaluable
                and alert == ServiceAlert.NONE
                and disposition
                in (DiagnosisDisposition.ACTIONABLE, DiagnosisDisposition.NO_CHANGE)
            ),
            blind_spot_ids=tuple(entry.blind_spot_id for entry in matching),
            blind_spot_catalog_revision=catalog_revision,
        )


@dataclasses.dataclass(frozen=True)
class _ExecutionCounts:
    infra_errors: int = 0
    timed_out: int = 0
    failed: int = 0
    stress_infra_errors: int = 0
    stress_timed_out: int = 0


def _execution_counts(report: SessionReport) -> _ExecutionCounts:
    statuses = [record.execution_status for record in report.experiments]
    return _ExecutionCounts(
        infra_errors=statuses.count(ExecutionStatus.INFRA_ERROR),
        timed_out=statuses.count(ExecutionStatus.TIMED_OUT),
        failed=statuses.count(ExecutionStatus.FAILED),
        stress_infra_errors=sum(
            result.infra_errors for result in report.evidence.stress
        ),
        stress_timed_out=sum(
            result.timed_out_trials for result in report.evidence.stress
        ),
    )


def _service_alert(
    report: SessionReport,
    initial_complete: bool,
    counts: _ExecutionCounts,
) -> ServiceAlert:
    # A post-campaign persistence error also loses the durable triggering
    # evidence needed for action, so every recorded stress infra error alerts.
    if report.status == SessionStatus.FAILED or any(
        (
            counts.infra_errors,
            counts.timed_out,
            counts.failed,
            counts.stress_infra_errors,
            counts.stress_timed_out,
        )
    ):
        return ServiceAlert.DIAGNOSIS_FAILED
    if (
        not initial_complete
        or report.explanation.evidence_sufficiency != EvidenceSufficiency.SUFFICIENT
    ):
        return ServiceAlert.EVIDENCE_INCOMPLETE
    return ServiceAlert.NONE


def _disposition(
    report: SessionReport,
    actionable: bool,
    alert: ServiceAlert,
    *,
    execution_failed: bool,
) -> DiagnosisDisposition:
    if execution_failed:
        return DiagnosisDisposition.FAILED
    if (
        report.status == SessionStatus.INCONCLUSIVE
        or alert == ServiceAlert.EVIDENCE_INCOMPLETE
    ):
        return DiagnosisDisposition.INCONCLUSIVE
    if actionable:
        return DiagnosisDisposition.ACTIONABLE
    if alert == ServiceAlert.DIAGNOSIS_FAILED:
        return DiagnosisDisposition.FAILED
    return DiagnosisDisposition.NO_CHANGE


def _evidence_results(report: SessionReport) -> list[ExperimentResult]:
    evidence = report.evidence
    return [
        *evidence.discovery,
        *evidence.sanitizer,
        *evidence.stress,
        *evidence.traces,
        *evidence.reductions,
    ]


def _evidence_duration_ms(report: SessionReport) -> int:
    seconds = 0.0
    for result in _evidence_results(report):
        seconds += max(0.0, result.duration_s)
        if isinstance(result, TraceExperimentResult):
            seconds += max(0.0, result.analysis_duration_s)
    return round(seconds * 1000)
