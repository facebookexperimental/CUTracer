# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Pure conversion from canonical sanitizer output to service evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from cutracer.compute_sanitizer.parser import (
    parse_racecheck_log as parse_canonical_racecheck_log,
)
from cutracer.service.contracts import (
    ArtifactRef,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentSpec,
    FindingRecord,
    ResultCompleteness,
    SanitizerOutcome,
    SanitizerSummary,
    SanitizerSweepResult,
    SourceLoc,
)


@dataclass(frozen=True)
class RacecheckEvidence:
    findings: List[FindingRecord]
    summary: Optional[SanitizerSummary]
    completeness: ResultCompleteness
    issues: List[str]


def parse_racecheck_evidence(text: str) -> RacecheckEvidence:
    """Adapt the canonical parser without weakening its completeness facts."""
    parsed = parse_canonical_racecheck_log(text)
    records: List[FindingRecord] = []
    for finding in parsed.findings:
        primary = finding.accesses[0]
        count = finding.max_hazards
        records.append(
            FindingRecord(
                source_tool="compute_sanitizer/racecheck",
                error_type="race-condition",
                kernel_name=primary.function,
                source=SourceLoc(file=primary.file, line=primary.line),
                count=count if count is not None else 1,
                raw=finding.raw_block,
            )
        )
    summary = (
        None
        if parsed.summary is None
        else SanitizerSummary(
            total_hazards=parsed.summary.total_hazards,
            errors=parsed.summary.errors,
            warnings=parsed.summary.warnings,
            raw=parsed.summary.raw,
        )
    )
    return RacecheckEvidence(
        findings=records,
        summary=summary,
        completeness=(
            ResultCompleteness.COMPLETE
            if parsed.is_complete
            else ResultCompleteness.PARTIAL
        ),
        issues=[diagnostic.value for diagnostic in parsed.diagnostics],
    )


def parse_racecheck_log(text: str) -> List[FindingRecord]:
    """Compatibility projection returning only converted finding blocks."""
    return parse_racecheck_evidence(text).findings


def build_sanitizer_result(
    spec: ExperimentSpec,
    *,
    execution_status: ExecutionStatus,
    log_text: str,
    log: Optional[ArtifactRef],
    duration_s: float = 0.0,
    error: Optional[str] = None,
    provenance: Optional[ExecutionProvenance] = None,
) -> SanitizerSweepResult:
    """Build the canonical result used by both Local and MAST producers."""
    tool = spec.sanitizer_tool
    if tool is None:
        raise ValueError("sanitizer experiment requires a sanitizer_tool")

    racecheck = parse_racecheck_evidence(log_text) if tool == "racecheck" else None
    findings = [] if racecheck is None else racecheck.findings
    summary = None if racecheck is None else racecheck.summary
    completeness = (
        ResultCompleteness.UNKNOWN if racecheck is None else racecheck.completeness
    )
    summary_issues = [] if racecheck is None else racecheck.issues

    if racecheck is None or racecheck.completeness != ResultCompleteness.COMPLETE:
        outcome = SanitizerOutcome.UNKNOWN
    elif summary is not None and summary.is_positive:
        outcome = SanitizerOutcome.FINDING
    elif execution_status == ExecutionStatus.SUCCEEDED and log is not None:
        outcome = SanitizerOutcome.CLEAN
    else:
        outcome = SanitizerOutcome.UNKNOWN

    return SanitizerSweepResult(
        experiment_id=spec.experiment_id,
        execution_status=execution_status,
        outcome=outcome,
        tool=tool,
        findings=findings,
        provenance=provenance
        or ExecutionProvenance(
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
        ),
        log=log,
        duration_s=duration_s,
        error=error,
        summary=summary,
        completeness=completeness,
        summary_issues=summary_issues,
    )
