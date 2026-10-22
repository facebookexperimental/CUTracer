"""Compute Sanitizer experiment adapter."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from click import ClickException
from cutracer.compute_sanitizer.parser import (
    parse_racecheck_log as parse_canonical_racecheck_log,
)
from cutracer.compute_sanitizer.run import (
    ComputeSanitizerConfig,
    run_compute_sanitizer,
    RunTarget,
    SupportsTensorOps,
)
from cutracer.service.artifacts import publish_local_artifact
from cutracer.service.contracts import (
    ArtifactRef,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentKind,
    ExperimentSpec,
    FindingRecord,
    ResultCompleteness,
    SanitizerOutcome,
    SanitizerSummary,
    SanitizerSweepResult,
    SourceLoc,
)

Reader = Callable[[str], str]


@dataclass(frozen=True)
class _RacecheckEvidence:
    findings: List[FindingRecord]
    summary: Optional[SanitizerSummary]
    completeness: ResultCompleteness
    issues: List[str]


def _default_reader(path: str) -> str:
    # Sanitizer logs may contain non-UTF-8 bytes. Preserve the rest of a useful
    # report instead of turning one bad byte into an infrastructure failure.
    with open(path, errors="replace") as fh:
        return fh.read()


def _append_error(error: Optional[str], detail: str) -> str:
    return detail if error is None else f"{error}; {detail}"


def _record_artifact_failure(status: ExecutionStatus) -> ExecutionStatus:
    if status == ExecutionStatus.SUCCEEDED:
        return ExecutionStatus.INFRA_ERROR
    return status


def _parse_racecheck_evidence(text: str) -> _RacecheckEvidence:
    """Convert one canonical parser result without discarding run-level facts."""
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
    return _RacecheckEvidence(
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
    """Compatibility wrapper returning only converted finding blocks."""
    return _parse_racecheck_evidence(text).findings


def _invoke_sanitizer(
    spec: ExperimentSpec,
    *,
    tool: str,
    attempt_dir: str,
    runner: Optional[RunTarget],
    compute_sanitizer: Optional[str],
    supports_tensor_ops: Optional[SupportsTensorOps],
    timeout: int,
) -> tuple[ExecutionStatus, str | None, str]:
    log_path = os.path.join(attempt_dir, f"{tool}.log")
    try:
        captured = run_compute_sanitizer(
            ComputeSanitizerConfig(
                argv=spec.unit.argv,
                tool=tool,
                output_dir=attempt_dir,
                compute_sanitizer=compute_sanitizer,
                cwd=spec.unit.cwd or None,
                timeout=timeout,
                base_env=os.environ,
                env=spec.unit.env,
            ),
            runner=runner,
            supports_tensor_ops=supports_tensor_ops,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecutionStatus.TIMED_OUT, str(exc), log_path
    except (ClickException, OSError, ValueError) as exc:
        return ExecutionStatus.INFRA_ERROR, str(exc), log_path

    status = (
        ExecutionStatus.SUCCEEDED
        if captured.process.returncode == 0
        else ExecutionStatus.FAILED
    )
    error = (
        None
        if captured.process.returncode == 0
        else captured.process.stderr or "process failed"
    )
    return status, error, captured.log_path or log_path


def _read_and_publish_log(
    log_path: str,
    *,
    out_dir: str,
    tool: str,
    read: Reader,
    status: ExecutionStatus,
    error: str | None,
) -> tuple[str, ArtifactRef | None, ExecutionStatus, str | None]:
    try:
        text = read(log_path)
    except FileNotFoundError:
        return (
            "",
            None,
            _record_artifact_failure(status),
            _append_error(error, "sanitizer completed without producing its log"),
        )
    except OSError as exc:
        return (
            "",
            None,
            _record_artifact_failure(status),
            _append_error(error, f"could not read sanitizer log {log_path}: {exc}"),
        )

    try:
        log = publish_local_artifact(log_path, os.path.join(out_dir, f"{tool}.log"))
    except OSError as exc:
        return (
            text,
            None,
            _record_artifact_failure(status),
            _append_error(error, f"could not publish sanitizer log: {exc}"),
        )
    return text, log, status, error


def _sanitizer_outcome(
    racecheck: _RacecheckEvidence | None,
    *,
    status: ExecutionStatus,
    log: ArtifactRef | None,
) -> SanitizerOutcome:
    if racecheck is None or racecheck.completeness != ResultCompleteness.COMPLETE:
        return SanitizerOutcome.UNKNOWN
    if racecheck.summary is not None and racecheck.summary.is_positive:
        return SanitizerOutcome.FINDING
    if status == ExecutionStatus.SUCCEEDED and log is not None:
        return SanitizerOutcome.CLEAN
    return SanitizerOutcome.UNKNOWN


def _run_sanitizer_attempt(
    spec: ExperimentSpec,
    *,
    out_dir: str,
    attempt_dir: str,
    runner: Optional[RunTarget] = None,
    reader: Optional[Reader] = None,
    compute_sanitizer: Optional[str] = None,
    supports_tensor_ops: Optional[SupportsTensorOps] = None,
    timeout: int = 1800,
) -> SanitizerSweepResult:
    if spec.sanitizer_tool is None:
        raise ValueError("sanitizer experiment requires a sanitizer_tool")
    read = reader or _default_reader
    tool = spec.sanitizer_tool
    started = time.monotonic()
    status, error, log_path = _invoke_sanitizer(
        spec,
        tool=tool,
        attempt_dir=attempt_dir,
        runner=runner,
        compute_sanitizer=compute_sanitizer,
        supports_tensor_ops=supports_tensor_ops,
        timeout=timeout,
    )
    text, log, status, error = _read_and_publish_log(
        log_path,
        out_dir=out_dir,
        tool=tool,
        read=read,
        status=status,
        error=error,
    )

    racecheck = _parse_racecheck_evidence(text) if tool == "racecheck" else None
    findings = [] if racecheck is None else racecheck.findings
    summary = None if racecheck is None else racecheck.summary
    completeness = (
        ResultCompleteness.UNKNOWN if racecheck is None else racecheck.completeness
    )
    summary_issues = [] if racecheck is None else racecheck.issues

    return SanitizerSweepResult(
        experiment_id=spec.experiment_id,
        execution_status=status,
        outcome=_sanitizer_outcome(racecheck, status=status, log=log),
        tool=tool,
        findings=findings,
        provenance=ExecutionProvenance(
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
        ),
        log=log,
        duration_s=time.monotonic() - started,
        error=error,
        summary=summary,
        completeness=completeness,
        summary_issues=summary_issues,
    )


def run_sanitizer_experiment(
    spec: ExperimentSpec,
    *,
    out_dir: str,
    runner: Optional[RunTarget] = None,
    reader: Optional[Reader] = None,
    compute_sanitizer: Optional[str] = None,
    supports_tensor_ops: Optional[SupportsTensorOps] = None,
    timeout: int = 1800,
) -> SanitizerSweepResult:
    """Run an independent sanitizer branch without invoking the CUTracer CLI."""
    if spec.kind != ExperimentKind.COMPUTE_SANITIZER:
        raise ValueError(f"not a sanitizer experiment: {spec.kind}")
    if spec.sanitizer_tool is None:
        raise ValueError("sanitizer experiment requires a sanitizer_tool")

    os.makedirs(out_dir, exist_ok=True)
    attempt_dir = tempfile.mkdtemp(prefix=".sanitizer_attempt_", dir=out_dir)
    result: Optional[SanitizerSweepResult] = None
    try:
        result = _run_sanitizer_attempt(
            spec,
            out_dir=out_dir,
            attempt_dir=attempt_dir,
            runner=runner,
            reader=reader,
            compute_sanitizer=compute_sanitizer,
            supports_tensor_ops=supports_tensor_ops,
            timeout=timeout,
        )
        return result
    finally:
        shutil.rmtree(attempt_dir, ignore_errors=True)
