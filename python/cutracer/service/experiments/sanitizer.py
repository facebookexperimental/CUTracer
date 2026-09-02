"""Compute Sanitizer experiment adapter."""

from __future__ import annotations

import os
import subprocess
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
from cutracer.service.artifacts import local_artifact
from cutracer.service.contracts import (
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
    read = reader or _default_reader
    tool = spec.sanitizer_tool
    started = time.monotonic()
    log_path = os.path.join(out_dir, f"{tool}.log")
    log = None
    text = ""

    try:
        captured = run_compute_sanitizer(
            ComputeSanitizerConfig(
                argv=spec.unit.argv,
                tool=tool,
                output_dir=out_dir,
                compute_sanitizer=compute_sanitizer,
                cwd=spec.unit.cwd or None,
                timeout=timeout,
                base_env=os.environ,
                env=spec.unit.env,
            ),
            runner=runner,
            supports_tensor_ops=supports_tensor_ops,
        )
        proc = captured.process
        log_path = captured.log_path or log_path
        status = (
            ExecutionStatus.SUCCEEDED
            if proc.returncode == 0
            else ExecutionStatus.FAILED
        )
        error = None if proc.returncode == 0 else (proc.stderr or "process failed")
    except subprocess.TimeoutExpired as exc:
        status = ExecutionStatus.TIMED_OUT
        error = str(exc)
    except (ClickException, OSError, ValueError) as exc:
        status = ExecutionStatus.INFRA_ERROR
        error = str(exc)

    try:
        text = read(log_path)
        log = local_artifact(log_path)
    except FileNotFoundError:
        if status == ExecutionStatus.SUCCEEDED:
            status = ExecutionStatus.INFRA_ERROR
            error = "sanitizer completed without producing its log"
    except OSError as exc:
        if status == ExecutionStatus.SUCCEEDED:
            status = ExecutionStatus.INFRA_ERROR
            error = f"could not read sanitizer log {log_path}: {exc}"

    racecheck = _parse_racecheck_evidence(text) if tool == "racecheck" else None
    findings = [] if racecheck is None else racecheck.findings
    summary = None if racecheck is None else racecheck.summary
    completeness = (
        ResultCompleteness.UNKNOWN if racecheck is None else racecheck.completeness
    )
    summary_issues = [] if racecheck is None else racecheck.issues

    if racecheck is None or racecheck.completeness != ResultCompleteness.COMPLETE:
        outcome = SanitizerOutcome.UNKNOWN
    elif summary is not None and summary.is_positive:
        # A complete positive summary is useful evidence even if the target
        # process itself returned non-zero. Execution and semantic outcome are
        # intentionally kept separate.
        outcome = SanitizerOutcome.FINDING
    elif status == ExecutionStatus.SUCCEEDED and log is not None:
        outcome = SanitizerOutcome.CLEAN
    else:
        outcome = SanitizerOutcome.UNKNOWN

    return SanitizerSweepResult(
        experiment_id=spec.experiment_id,
        execution_status=status,
        outcome=outcome,
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
