# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Deterministic analyzer adapter for service trace experiments."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from cutracer.analyze.fb.common.findings import AnalysisResult, Finding
from cutracer.service.artifacts import local_artifact
from cutracer.service.contracts import (
    ExecutionStatus,
    SourceLoc,
    TraceAnalysisEvidence,
    TraceAnalysisFinding,
    TraceAnalysisSeverity,
    TraceDetectorSummary,
)

MAX_COMPACT_FINDINGS = 32

logger: logging.Logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DeterministicAnalysisOutput:
    analyzer: str
    detector_results: Tuple[AnalysisResult, ...]
    warnings: Tuple[str, ...]
    payload: Dict[str, Any]


TraceAnalyzer = Callable[[str, Optional[str]], DeterministicAnalysisOutput]


def _run_data_race_analyzer(
    trace_path: str,
    arch: Optional[str],
) -> DeterministicAnalysisOutput:
    # Keep the heavyweight analyzer imports off service control-plane imports.
    from cutracer.analyze.fb.data_race.cli import analyze_data_race, violations_to_dict

    path = Path(trace_path)
    (
        trace_records,
        kernel_config,
        pc_mappings,
        detector_results,
        warnings,
        instructions,
    ) = analyze_data_race(path, arch=arch, want_instructions=True)
    payload = violations_to_dict(
        detector_results,
        trace_records,
        path,
        kernel_config,
        pc_mappings,
        warnings,
        instructions,
    )
    return DeterministicAnalysisOutput(
        analyzer="data_race",
        detector_results=tuple(detector_results),
        warnings=tuple(warnings),
        payload=payload,
    )


def _summary(result: AnalysisResult) -> TraceDetectorSummary:
    summary = result.summary
    return TraceDetectorSummary(
        source_tool=summary.source.tool,
        sub_component=summary.source.sub_component,
        verdict=summary.verdict,
        severity=TraceAnalysisSeverity(summary.severity.value),
        finding_count=len(result.findings),
        description=summary.description,
    )


def _finding(result: AnalysisResult, finding: Finding) -> TraceAnalysisFinding:
    source = result.summary.source
    return TraceAnalysisFinding(
        source_tool=source.tool,
        sub_component=source.sub_component,
        severity=TraceAnalysisSeverity(finding.severity.value),
        category=finding.category.value,
        description=finding.description,
        ctas=[list(cta) for cta in finding.ctas],
        warps=list(finding.warps),
        pcs=[hex(pc) for pc in finding.pcs],
        trace_indices=list(finding.trace_indices),
        source_locations=[
            None
            if location is None
            else SourceLoc(file=location.file, line=location.line)
            for location in finding.source_locations
        ],
    )


def _analyzer_name(analyzer: TraceAnalyzer) -> str:
    if analyzer is _run_data_race_analyzer:
        return "data_race"
    name = getattr(analyzer, "__name__", "")
    normalized = name.strip() if isinstance(name, str) else ""
    if normalized:
        return normalized
    return type(analyzer).__name__.strip() or "trace_analyzer"


def _sample_findings(
    detector_results: Tuple[AnalysisResult, ...],
    max_findings: int,
) -> tuple[list[TraceAnalysisFinding], list[str]]:
    iterators = [
        (index, result, iter(result.findings))
        for index, result in enumerate(detector_results)
    ]
    sampled: list[TraceAnalysisFinding] = []
    errors: list[str] = []
    while len(sampled) < max_findings:
        made_progress = False
        for detector_index, result, findings in iterators:
            if len(sampled) == max_findings:
                break
            try:
                finding = next(findings)
            except StopIteration:
                continue
            made_progress = True
            try:
                sampled.append(_finding(result, finding))
            except Exception as exc:
                logger.exception(
                    "could not project deterministic detector %d finding",
                    detector_index,
                )
                if len(errors) < MAX_COMPACT_FINDINGS:
                    errors.append(f"detector {detector_index} finding: {exc}")
        if not made_progress:
            break
    return sampled, errors


def _project_detector_results(
    detector_results: Tuple[AnalysisResult, ...],
    max_findings: int,
) -> tuple[
    list[TraceDetectorSummary],
    list[TraceAnalysisFinding],
    int,
    list[str],
]:
    summaries: list[TraceDetectorSummary] = []
    valid_results: list[AnalysisResult] = []
    total_findings = 0
    errors: list[str] = []
    for detector_index, result in enumerate(detector_results):
        try:
            summaries.append(_summary(result))
        except Exception as exc:
            logger.exception(
                "could not project deterministic detector %d summary",
                detector_index,
            )
            errors.append(f"detector {detector_index} summary: {exc}")
            continue
        valid_results.append(result)
        total_findings += len(result.findings)
    compact_findings, finding_errors = _sample_findings(
        tuple(valid_results), max_findings
    )
    errors.extend(finding_errors)
    return summaries, compact_findings, total_findings, errors


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".trace_analysis_", dir=directory)
    try:
        fh = os.fdopen(fd, "w")
        fd = -1
        with fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
        temp_path = ""
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def run_trace_analysis(
    trace_path: str,
    *,
    arch: str,
    out_dir: str,
    analyzer: Optional[TraceAnalyzer] = None,
    max_findings: int = MAX_COMPACT_FINDINGS,
) -> TraceAnalysisEvidence:
    """Analyze one captured trace and persist a lossless structured report."""
    if max_findings < 0:
        raise ValueError("max_findings must be non-negative")
    analyze = analyzer or _run_data_race_analyzer
    analyzer_name = _analyzer_name(analyze)
    output: Optional[DeterministicAnalysisOutput] = None
    try:
        output = analyze(trace_path, arch or None)
        if not output.analyzer.strip():
            raise ValueError("deterministic analyzer returned an empty identity")
        analyzer_name = output.analyzer.strip()
    except Exception as exc:
        # Analyzer implementations are an isolation boundary: a bad trace or
        # detector must not discard the successfully captured trace artifact.
        logger.exception("deterministic trace analysis failed")
        artifact = None
        persistence_error = ""
        if output is not None:
            report_path = os.path.join(out_dir, "trace_analysis.json")
            try:
                _atomic_write_json(report_path, output.payload)
                artifact = local_artifact(
                    report_path,
                    media_type="application/json",
                    include_sha256=True,
                )
            except (OSError, TypeError, ValueError) as persist_exc:
                logger.exception("could not persist failed analyzer payload")
                persistence_error = f"; could not persist payload: {persist_exc}"
        return TraceAnalysisEvidence(
            analyzer=analyzer_name,
            execution_status=ExecutionStatus.INFRA_ERROR,
            detector_summaries=[],
            findings=[],
            total_findings=0,
            warnings=[] if output is None else list(output.warnings),
            artifact=artifact,
            error=f"deterministic trace analysis failed: {exc}{persistence_error}",
        )
    summaries, compact_findings, total_findings, projection_errors = (
        _project_detector_results(output.detector_results, max_findings)
    )
    status = (
        ExecutionStatus.INFRA_ERROR if projection_errors else ExecutionStatus.SUCCEEDED
    )
    error = (
        None
        if not projection_errors
        else "deterministic trace analysis projection failed: "
        + "; ".join(projection_errors)
    )

    report_path = os.path.join(out_dir, "trace_analysis.json")
    try:
        _atomic_write_json(report_path, output.payload)
        artifact = local_artifact(
            report_path,
            media_type="application/json",
            include_sha256=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        logger.exception("could not persist deterministic trace analysis")
        artifact = None
        status = ExecutionStatus.INFRA_ERROR
        persistence_error = f"could not persist deterministic trace analysis: {exc}"
        error = persistence_error if error is None else f"{error}; {persistence_error}"

    return TraceAnalysisEvidence(
        analyzer=analyzer_name,
        execution_status=status,
        detector_summaries=summaries,
        findings=compact_findings,
        total_findings=total_findings,
        warnings=[*output.warnings, *projection_errors],
        artifact=artifact,
        error=error,
    )
