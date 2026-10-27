# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Pure conversion from canonical sanitizer output to service evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace

from cutracer.compute_sanitizer.parser import (
    CheckerIssue,
    CheckerTool,
    parse_checker_log,
    RacecheckParseDiagnostic,
)
from cutracer.service.contracts import (
    ArtifactRef,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentSpec,
    FindingRecord,
    KernelId,
    ResultCompleteness,
    SanitizerOutcome,
    SanitizerSummary,
    SanitizerSweepResult,
    SourceLoc,
)

TARGET_SCOPED_SANITIZER_TOOLS = ("initcheck",)

# Stable, vendor-neutral finding classes. The tool's own wording is preserved
# separately in ``FindingRecord.error_detail``; it encodes specifics such as
# access width that would otherwise split one defect into several groups.
ERROR_TYPE_UNINITIALIZED_READ = "uninitialized-read"
ERROR_TYPE_RACE_CONDITION = "race-condition"
ERROR_TYPE_INVALID_SHARED_ACCESS = "invalid-shared-access"
ERROR_TYPE_BARRIER_ERROR = "barrier-error"
ERROR_TYPE_INVALID_MEMORY_ACCESS = "invalid-memory-access"
ERROR_TYPE_MISALIGNED_ACCESS = "misaligned-access"
ERROR_TYPE_HARDWARE_EXCEPTION = "hardware-exception"
ERROR_TYPE_MEMORY_LEAK = "memory-leak"
ERROR_TYPE_UNCLASSIFIED = "unclassified"
_INTERNAL_TOOL_ERROR = "Compute Sanitizer reported an internal tool error"

# Keep the vendor spellings together with their stable service classifications.
_ERROR_TYPE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("Potential invalid __shared__", ERROR_TYPE_INVALID_SHARED_ACCESS),
    ("Invalid __shared__", ERROR_TYPE_INVALID_SHARED_ACCESS),
    ("Uninitialized ", ERROR_TYPE_UNINITIALIZED_READ),
    ("Race reported", ERROR_TYPE_RACE_CONDITION),
    ("Barrier error detected", ERROR_TYPE_BARRIER_ERROR),
    ("Invalid ", ERROR_TYPE_INVALID_MEMORY_ACCESS),
    ("Out of bounds ", ERROR_TYPE_INVALID_MEMORY_ACCESS),
    ("Misaligned ", ERROR_TYPE_MISALIGNED_ACCESS),
    ("Program hit ", ERROR_TYPE_HARDWARE_EXCEPTION),
    ("Leaked ", ERROR_TYPE_MEMORY_LEAK),
)


def is_target_scoped_sanitizer(tool: str) -> bool:
    """Whether the checker must observe every kernel launched by the target."""
    return tool in TARGET_SCOPED_SANITIZER_TOOLS


def classify_error_type(message: str) -> str:
    """Map a sanitizer message to its stable class.

    An unrecognized message is classified rather than dropped: losing a finding
    is worse than grouping it under a catch-all that shows up in review.
    """
    for prefix, error_type in _ERROR_TYPE_PREFIXES:
        if message.startswith(prefix):
            return error_type
    return ERROR_TYPE_UNCLASSIFIED


@dataclass(frozen=True)
class CheckerEvidence:
    """Service-shaped evidence from one sanitizer log, for any tool."""

    findings: list[FindingRecord]
    summary: SanitizerSummary | None
    completeness: ResultCompleteness
    issues: list[str]
    output_truncated: bool


def _to_finding(tool: CheckerTool, issue: CheckerIssue) -> FindingRecord:
    site = issue.primary_site
    source: SourceLoc | None = None
    if site is not None and site.file is not None and site.line is not None:
        source = SourceLoc(file=site.file, line=site.line)
    return FindingRecord(
        source_tool=f"compute_sanitizer/{tool.value}",
        error_type=classify_error_type(issue.message),
        error_detail=issue.message,
        kernel_name=None if site is None else site.symbol,
        source=source,
        count=issue.hazards if issue.hazards is not None else 1,
        raw=issue.raw_block,
    )


def parse_checker_evidence(tool: CheckerTool, text: str) -> CheckerEvidence:
    """Adapt the canonical parser without weakening its completeness facts."""
    parsed = parse_checker_log(tool, text)
    summary = (
        None
        if parsed.summary is None
        else SanitizerSummary(
            total_hazards=parsed.summary.total_hazards,
            errors=parsed.summary.errors,
            warnings=parsed.summary.warnings,
            raw=parsed.summary.raw,
            not_printed=parsed.summary.not_printed,
        )
    )
    return CheckerEvidence(
        findings=[_to_finding(tool, issue) for issue in parsed.issues],
        summary=summary,
        completeness=(
            ResultCompleteness.COMPLETE
            if parsed.is_complete
            else ResultCompleteness.PARTIAL
        ),
        issues=[diagnostic.value for diagnostic in parsed.diagnostics],
        output_truncated=parsed.output_truncated,
    )


def parse_racecheck_log(text: str) -> list[FindingRecord]:
    """Return the legacy findings-only projection with current classifications."""
    return parse_checker_evidence(CheckerTool.RACECHECK, text).findings


def _checker_evidence(tool: str, log_text: str) -> CheckerEvidence:
    try:
        checker_tool = CheckerTool(tool)
    except ValueError:
        # An unrecognized tool cannot prove anything about the run.
        return CheckerEvidence(
            findings=[],
            summary=None,
            completeness=ResultCompleteness.UNKNOWN,
            issues=["unsupported_sanitizer_tool"],
            output_truncated=False,
        )
    return parse_checker_evidence(checker_tool, log_text)


def _sanitizer_outcome(
    evidence: CheckerEvidence,
    execution_status: ExecutionStatus,
    log: ArtifactRef | None,
) -> SanitizerOutcome:
    if evidence.completeness != ResultCompleteness.COMPLETE:
        return SanitizerOutcome.UNKNOWN
    if evidence.summary is not None and evidence.summary.is_positive:
        return SanitizerOutcome.FINDING
    if execution_status == ExecutionStatus.SUCCEEDED and log is not None:
        return SanitizerOutcome.CLEAN
    return SanitizerOutcome.UNKNOWN


def _matches_kernel(finding: FindingRecord, kernel: KernelId) -> bool:
    if finding.kernel_name is None:
        return False
    names = {kernel.name}
    mangled_name = kernel.mangled_name
    if isinstance(mangled_name, str):
        names.add(mangled_name)
    return finding.kernel_name in names


def project_target_scoped_sanitizer_result(
    result: SanitizerSweepResult,
    *,
    experiment_id: str,
    kernel: KernelId,
) -> SanitizerSweepResult:
    """Conservatively project one whole-target checker run onto a kernel."""
    if not is_target_scoped_sanitizer(result.tool):
        raise ValueError(f"sanitizer tool is not target-scoped: {result.tool}")

    provenance = replace(result.provenance, kernel=kernel)
    if not result.has_positive_signal:
        return replace(result, experiment_id=experiment_id, provenance=provenance)

    matching = [
        replace(finding, kernel_name=kernel.name)
        for finding in result.findings
        if _matches_kernel(finding, kernel)
    ]
    observed_count = sum(finding.count for finding in result.findings)
    counts_match = (
        result.summary is not None and result.summary.errors == observed_count
    )
    fully_attributed = (
        bool(matching)
        and len(matching) == len(result.findings)
        and counts_match
        and not result.output_truncated
    )
    if fully_attributed:
        return replace(
            result,
            experiment_id=experiment_id,
            findings=matching,
            provenance=provenance,
        )

    issues = list(result.summary_issues)
    if result.output_truncated:
        issues.append("target_scope_output_truncated")
    if not counts_match:
        issues.append("target_scope_finding_count_mismatch")
    if any(finding.kernel_name is None for finding in result.findings):
        issues.append("target_scope_contains_unattributed_findings")
    if any(
        finding.kernel_name is not None and not _matches_kernel(finding, kernel)
        for finding in result.findings
    ):
        issues.append("target_scope_contains_other_kernel_findings")
    if not matching:
        issues.append("target_scope_has_no_matching_finding")
    return replace(
        result,
        experiment_id=experiment_id,
        outcome=SanitizerOutcome.UNKNOWN,
        findings=matching,
        provenance=provenance,
        summary=None,
        completeness=ResultCompleteness.PARTIAL,
        summary_issues=list(dict.fromkeys(issues)),
    )


def build_sanitizer_result(
    spec: ExperimentSpec,
    *,
    execution_status: ExecutionStatus,
    log_text: str,
    log: ArtifactRef | None,
    duration_s: float = 0.0,
    error: str | None = None,
    provenance: ExecutionProvenance | None = None,
) -> SanitizerSweepResult:
    """Build the canonical result used by both Local and MAST producers."""
    tool = spec.sanitizer_tool
    if tool is None:
        raise ValueError("sanitizer experiment requires a sanitizer_tool")
    evidence = _checker_evidence(tool, log_text)
    if RacecheckParseDiagnostic.INTERNAL_TOOL_ERROR.value in evidence.issues:
        execution_status = ExecutionStatus.INFRA_ERROR
        error = (
            _INTERNAL_TOOL_ERROR
            if error is None
            else f"{error}; {_INTERNAL_TOOL_ERROR}"
        )

    result = SanitizerSweepResult(
        experiment_id=spec.experiment_id,
        execution_status=execution_status,
        outcome=_sanitizer_outcome(evidence, execution_status, log),
        tool=tool,
        findings=evidence.findings,
        provenance=provenance
        or ExecutionProvenance(
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
        ),
        log=log,
        duration_s=duration_s,
        error=error,
        summary=evidence.summary,
        completeness=evidence.completeness,
        summary_issues=evidence.issues,
        output_truncated=evidence.output_truncated,
    )
    if spec.unit.kernel is None or not is_target_scoped_sanitizer(tool):
        return result
    return project_target_scoped_sanitizer_result(
        result,
        experiment_id=spec.experiment_id,
        kernel=spec.unit.kernel,
    )
