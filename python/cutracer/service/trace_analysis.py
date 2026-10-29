# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Deterministic analyzer adapter for service trace experiments."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from itertools import islice, zip_longest
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

from cutracer.service.artifacts import local_artifact
from cutracer.service.contracts import (
    ExecutionStatus,
    SourceLoc,
    TraceAnalysisEvidence,
    TraceAnalysisFinding,
    TraceAnalysisSeverity,
    TraceDetectorSummary,
)

if TYPE_CHECKING:
    # The OSS export has no fb/ packages, and the CLI imports this module there.
    from cutracer.analyze.fb.common.findings import AnalysisResult, Finding

MAX_COMPACT_FINDINGS = 32
MAX_CAPTURE_INSTRUCTIONS = 32
MAX_CAPTURE_SUMMARIES = 4

logger: logging.Logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DeterministicAnalysisOutput:
    analyzer: str
    detector_results: Tuple[AnalysisResult, ...]
    warnings: Tuple[str, ...]
    payload: Dict[str, Any]
    static_instructions: Dict[int, Tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )
    source_mappings: Dict[int, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    static_context_status: str = "unavailable"


@dataclasses.dataclass
class _TraceProjection:
    analyzer: str
    analyzer_confirmed: bool
    summaries: list[TraceDetectorSummary]
    finding_groups: list[Tuple[AnalysisResult, str]]
    total_findings: int
    warnings: list[str]
    payload: Optional[Dict[str, Any]]
    error: Optional[str]


@dataclasses.dataclass
class _TraceBatchProjection:
    analyzer: Optional[str] = None
    summaries: List[TraceDetectorSummary] = dataclasses.field(default_factory=list)
    finding_groups: List[Tuple[AnalysisResult, str]] = dataclasses.field(
        default_factory=list
    )
    total_findings: int = 0
    warnings: List[str] = dataclasses.field(default_factory=list)
    failures: List[str] = dataclasses.field(default_factory=list)
    trace_results: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    singleton_payload: Optional[Dict[str, Any]] = None


TraceAnalyzer = Callable[[str, Optional[str]], DeterministicAnalysisOutput]


def load_trace_static_context(
    cubin_path: Path, arch: str, kernel_name: str, tool_env: Mapping[str, str] | None
) -> tuple[dict[int, tuple[str, str]], str]:
    """Load planning-only SASS when the existing CFG proves one kernel scope."""
    from cutracer.analyze.fb.arch_decoder import get_decoder
    from cutracer.analyze.fb.data_race.event_kind import classify
    from cutracer.analyze.fb.data_race.tmem_lifetime.cfg_loader import (
        load_hyperblock_map_with_status,
    )

    try:
        loaded = load_hyperblock_map_with_status(cubin_path, tool_env=tool_env)
        if loaded.cfg is None:
            return {}, loaded.status
        entries = loaded.cfg.entry_blocks
        if len(entries) != 1 or next(iter(entries)).rsplit("@", 1)[0] != kernel_name:
            return {}, "kernel_scope_unverified"
        pending = list(entries)
        reachable = set()
        while pending:
            block = pending.pop()
            if block not in reachable:
                reachable.add(block)
                pending.extend(loaded.cfg.successors[block] - reachable)
        if reachable != set(loaded.cfg.blocks):
            return {}, "kernel_scope_unverified"
        decoder = get_decoder(arch)
        instructions = {}
        for block in loaded.cfg.blocks.values():
            for pc, sass in block.instructions:
                if pc in instructions:
                    return {}, "kernel_scope_unverified"
                instructions[pc] = (sass, classify(sass, decoder).name)
        return instructions, loaded.status
    except Exception:
        logger.warning(
            "static capture context unavailable for %s", cubin_path, exc_info=True
        )
        return {}, "unavailable"


def _sample_capture_pcs(
    candidates: Mapping[int, Tuple[str, str]],
    observed: Mapping[int, Dict[str, Any]],
    requested: Sequence[int],
) -> List[int]:
    requested_sample = list(requested[:MAX_CAPTURE_INSTRUCTIONS])
    groups: dict[str, list[int]] = {}
    for pc, (_, kind) in sorted(candidates.items()):
        if pc not in requested_sample and (kind != "NONE" or pc in observed):
            groups.setdefault(kind, []).append(pc)
    return requested_sample + list(
        islice(
            (
                pc
                for row in zip_longest(*groups.values())
                for pc in row
                if pc is not None
            ),
            MAX_CAPTURE_INSTRUCTIONS - len(requested_sample),
        )
    )


def _capture_instruction(
    output: DeterministicAnalysisOutput,
    pc: int,
    instruction: Tuple[str, str],
    observed: Mapping[int, Dict[str, Any]],
) -> Dict[str, Any]:
    sass, kind = instruction
    source = (
        output.source_mappings.get(pc, {})
        if output.static_context_status == "ok"
        else {}
    )
    source_files = (
        output.payload.get("source_files", {})
        if output.static_context_status == "ok"
        else {}
    )
    dynamic = observed.get(pc, {})
    source_id = dynamic.get("source_file")
    source_file = (
        source.get("file")
        or source_files.get(source_id, source_files.get(str(source_id), ""))
        or ""
    )
    return {
        "pc": hex(pc),
        "sass": sass[:512],
        "event_kind": kind,
        "source_file": source_file[:256],
        "source_line": source.get(
            "line",
            dynamic.get("source_line")
            if output.static_context_status == "ok"
            else None,
        ),
        "static": pc in output.static_instructions,
        "observed": pc in observed,
        "text_clipped": len(sass) > 512 or len(source_file) > 256,
    }


def _capture_summary(output: DeterministicAnalysisOutput) -> Dict[str, Any]:
    metadata = output.payload.get("kernel_metadata")
    instructions = output.payload.get("instructions")
    if not isinstance(metadata, dict) or not isinstance(instructions, dict):
        return {}
    observed = {int(pc, 0): info for pc, info in instructions.items()}
    requested = metadata.get("trace_pc_offsets")
    requested_pcs = sorted(set(requested or []))
    missing = sorted(set(requested_pcs) - observed.keys())
    candidates = dict(output.static_instructions)
    for pc, info in observed.items():
        candidates.setdefault(pc, (info.get("sass", ""), "UNCLASSIFIED"))
    for pc in requested_pcs:
        candidates.setdefault(pc, ("", "UNAVAILABLE"))
    entries = [
        _capture_instruction(output, pc, candidates[pc], observed)
        for pc in _sample_capture_pcs(candidates, observed, requested_pcs)
    ]
    return {
        "instrument_modes": metadata.get("instrument_modes", []),
        "grid_launch_id": metadata.get("grid_launch_id"),
        "capture_selectors": {
            key: metadata[key]
            for key in (
                "instr_categories",
                "trace_instr_categories",
                "trace_launch_id",
                "trace_warp_mask",
                "trace_pc_filter_valid",
            )
            if key in metadata
        },
        "requested_pc_offsets": None
        if requested is None
        else [hex(pc) for pc in requested_pcs[:MAX_CAPTURE_INSTRUCTIONS]],
        "requested_pc_count": len(requested_pcs),
        "omitted_requested_pcs": max(0, len(requested_pcs) - MAX_CAPTURE_INSTRUCTIONS),
        "observed_pc_offsets": [
            hex(pc) for pc in sorted(observed)[:MAX_CAPTURE_INSTRUCTIONS]
        ],
        "observed_pc_count": len(observed),
        "omitted_observed_pcs": max(0, len(observed) - MAX_CAPTURE_INSTRUCTIONS),
        "missing_requested_pc_offsets": [
            hex(pc) for pc in missing[:MAX_CAPTURE_INSTRUCTIONS]
        ],
        "missing_requested_pc_count": len(missing),
        "omitted_missing_requested_pcs": max(
            0, len(missing) - MAX_CAPTURE_INSTRUCTIONS
        ),
        "static_context_status": output.static_context_status,
        "static_pc_count": len(output.static_instructions),
        "instructions": entries,
        "omitted_instruction_count": len(candidates) - len(entries),
        "notes": [
            "Missing requested PCs were not observed; this does not prove they were not executed.",
            "Static instructions are capture-planning context, not evidence that an instruction executed.",
            "Capture selectors intersect. Omitted selector metadata means unknown, not unfiltered.",
            "A completed capture can leave individual detectors indeterminate. Missing EXIT records do not alone prove truncation.",
        ],
    }


def _run_data_race_analyzer(
    trace_path: str,
    arch: Optional[str],
    *,
    tool_env: Mapping[str, str] | None = None,
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
    ) = analyze_data_race(
        path,
        arch=arch,
        want_instructions=True,
        tool_env=tool_env,
    )
    payload = violations_to_dict(
        detector_results,
        trace_records,
        path,
        kernel_config,
        pc_mappings,
        warnings,
        instructions,
    )
    static_instructions = {}
    static_context_status = "cubin_unavailable"
    try:
        if kernel_config is not None and kernel_config.cubin_path:
            from cutracer.analyze.fb.common.arch_resolve import resolve_arch
            from cutracer.validation.compression import open_trace_file

            with open_trace_file(path) as trace_file:
                metadata = json.loads(next(line for line in trace_file if line.strip()))
            if metadata.get("type") != "kernel_metadata":
                raise ValueError("static context requires kernel_metadata header")
            kernel_symbol = metadata.get("mangled_name") or kernel_config.kernel_name

            static_instructions, static_context_status = load_trace_static_context(
                path.parent / kernel_config.cubin_path,
                resolve_arch(kernel_config, arch),
                kernel_symbol,
                tool_env,
            )
    except Exception:
        logger.warning(
            "static capture context unavailable for %s", trace_path, exc_info=True
        )
        static_context_status = "unavailable"
    return DeterministicAnalysisOutput(
        analyzer="data_race",
        detector_results=tuple(detector_results),
        warnings=tuple(warnings),
        payload=payload,
        static_instructions=static_instructions,
        source_mappings=pc_mappings,
        static_context_status=static_context_status,
    )


def _summary(result: AnalysisResult, trace_file: str) -> TraceDetectorSummary:
    summary = result.summary
    return TraceDetectorSummary(
        source_tool=summary.source.tool,
        sub_component=summary.source.sub_component,
        verdict=summary.verdict,
        severity=TraceAnalysisSeverity(summary.severity.value),
        finding_count=len(result.findings),
        description=summary.description,
        trace_file=trace_file,
    )


def _finding(
    result: AnalysisResult,
    finding: Finding,
    trace_file: str,
) -> TraceAnalysisFinding:
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
        trace_file=trace_file,
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
    finding_groups: Sequence[Tuple[AnalysisResult, str]],
    max_findings: int,
) -> tuple[list[TraceAnalysisFinding], list[str]]:
    iterators = [
        (index, result, trace_file, iter(result.findings))
        for index, (result, trace_file) in enumerate(finding_groups)
    ]
    sampled: list[TraceAnalysisFinding] = []
    errors: list[str] = []
    while len(sampled) < max_findings:
        made_progress = False
        for detector_index, result, trace_file, findings in iterators:
            if len(sampled) == max_findings:
                break
            try:
                finding = next(findings)
            except StopIteration:
                continue
            made_progress = True
            try:
                sampled.append(_finding(result, finding, trace_file))
            except Exception as exc:
                logger.exception(
                    "could not project %s detector %d finding",
                    trace_file,
                    detector_index,
                )
                if len(errors) < MAX_COMPACT_FINDINGS:
                    errors.append(
                        f"{trace_file}: detector {detector_index} finding: {exc}"
                    )
        if not made_progress:
            break
    return sampled, errors


def _project_trace_detectors(
    detector_results: Tuple[AnalysisResult, ...],
    trace_file: str,
) -> tuple[
    list[TraceDetectorSummary],
    list[Tuple[AnalysisResult, str]],
    int,
    list[str],
]:
    summaries: list[TraceDetectorSummary] = []
    finding_groups: list[Tuple[AnalysisResult, str]] = []
    total_findings = 0
    errors: list[str] = []
    for detector_index, result in enumerate(detector_results):
        try:
            summaries.append(_summary(result, trace_file))
        except Exception as exc:
            logger.exception(
                "could not project %s detector %d summary",
                trace_file,
                detector_index,
            )
            errors.append(f"detector {detector_index} summary: {exc}")
            continue
        finding_groups.append((result, trace_file))
        total_findings += len(result.findings)
    return summaries, finding_groups, total_findings, errors


def _analyze_trace_file(
    analyze: TraceAnalyzer,
    path: str,
    *,
    trace_file: str,
    arch: str,
    expected_analyzer: Optional[str],
    fallback_analyzer: str,
    tool_env: Mapping[str, str] | None,
) -> _TraceProjection:
    output: Optional[DeterministicAnalysisOutput] = None
    analyzer: Optional[str] = None
    analyzer_confirmed = False
    try:
        if analyze is _run_data_race_analyzer:
            output = _run_data_race_analyzer(
                path,
                arch or None,
                tool_env=tool_env,
            )
        else:
            output = analyze(path, arch or None)
        analyzer = output.analyzer.strip()
        if not analyzer:
            raise ValueError("deterministic analyzer returned an empty identity")
        if expected_analyzer is not None and analyzer != expected_analyzer:
            raise ValueError(
                f"analyzer identity changed from {expected_analyzer!r} to {analyzer!r}"
            )
        analyzer_confirmed = True
        if not isinstance(output.payload, dict):
            raise ValueError("deterministic analyzer returned no report payload")
        payload = dict(output.payload)
        try:
            capture_summary = _capture_summary(output)
            if capture_summary:
                payload["capture_summary"] = capture_summary
        except Exception:
            logger.warning(
                "could not summarize capture context for %s", trace_file, exc_info=True
            )
            payload["capture_summary"] = {
                "static_context_status": "summary_unavailable",
                "notes": [
                    "Capture summary unavailable; consult the lossless analysis artifact."
                ],
            }
        summaries, finding_groups, total_findings, errors = _project_trace_detectors(
            output.detector_results, trace_file
        )
        error = None
        if errors:
            error = "detector projection failed: " + "; ".join(errors)
        return _TraceProjection(
            analyzer=analyzer,
            analyzer_confirmed=analyzer_confirmed,
            summaries=summaries,
            finding_groups=finding_groups,
            total_findings=total_findings,
            warnings=[*output.warnings, *errors],
            payload=payload,
            error=error,
        )
    except Exception as exc:
        logger.exception("deterministic trace analysis failed for %s", trace_file)
        return _TraceProjection(
            analyzer=analyzer or expected_analyzer or fallback_analyzer,
            analyzer_confirmed=analyzer_confirmed,
            summaries=[],
            finding_groups=[],
            total_findings=0,
            warnings=[] if output is None else list(output.warnings),
            payload=(
                output.payload
                if output is not None and isinstance(output.payload, dict)
                else None
            ),
            error=str(exc),
        )


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


def _finish_trace_analysis(
    *,
    trace_paths: Sequence[str],
    analyzer_name: Optional[str],
    fallback_analyzer_name: str,
    summaries: List[TraceDetectorSummary],
    finding_groups: List[Tuple[AnalysisResult, str]],
    total_findings: int,
    warnings: List[str],
    failures: List[str],
    trace_results: List[Dict[str, Any]],
    singleton_payload: Optional[Dict[str, Any]],
    max_findings: int,
    out_dir: str,
) -> TraceAnalysisEvidence:
    compact_findings, finding_errors = _sample_findings(finding_groups, max_findings)
    if finding_errors:
        failures.append("compact finding projection: " + "; ".join(finding_errors))
        warnings.extend(finding_errors)
    if len(trace_paths) == 1:
        if singleton_payload is None:
            failure = failures[0] if failures else "analyzer returned no report payload"
            return TraceAnalysisEvidence(
                analyzer=analyzer_name or fallback_analyzer_name,
                execution_status=ExecutionStatus.INFRA_ERROR,
                detector_summaries=summaries,
                findings=compact_findings,
                total_findings=total_findings,
                warnings=warnings,
                error=f"deterministic trace analysis failed: {failure}",
            )
        # Preserve the N08 single-file artifact contract byte-for-structure.
        report_payload = singleton_payload
    else:
        failed_trace_count = sum(
            result["execution_status"] == ExecutionStatus.INFRA_ERROR.value
            for result in trace_results
        )
        report_payload = {
            "schema_version": 1,
            "analyzer": analyzer_name or fallback_analyzer_name,
            "summary": {
                "total_findings": total_findings,
                "trace_count": len(trace_paths),
                "analyzed_trace_count": len(trace_paths) - failed_trace_count,
                "failed_trace_count": failed_trace_count,
            },
            "trace_results": trace_results,
        }
    error = (
        None
        if not failures
        else "deterministic trace analysis failed for "
        f"{len(failures)} trace(s): {'; '.join(failures)}"
    )
    report_path = os.path.join(out_dir, "trace_analysis.json")
    try:
        _atomic_write_json(report_path, report_payload)
        artifact = local_artifact(
            report_path,
            media_type="application/json",
            include_sha256=True,
        )
        status = (
            ExecutionStatus.SUCCEEDED if not failures else ExecutionStatus.INFRA_ERROR
        )
    except (OSError, TypeError, ValueError) as exc:
        logger.exception("could not persist deterministic trace analysis")
        artifact = None
        status = ExecutionStatus.INFRA_ERROR
        persistence_error = f"could not persist deterministic trace analysis: {exc}"
        error = persistence_error if error is None else f"{error}; {persistence_error}"

    return TraceAnalysisEvidence(
        analyzer=analyzer_name or fallback_analyzer_name,
        execution_status=status,
        detector_summaries=summaries,
        findings=compact_findings,
        total_findings=total_findings,
        warnings=warnings,
        artifact=artifact,
        error=error,
        capture_summaries=[
            {"trace_file": item["trace_file"], **item["payload"]["capture_summary"]}
            for item in trace_results
            if item["payload"] is not None and item["payload"].get("capture_summary")
        ][:MAX_CAPTURE_SUMMARIES],
        omitted_capture_summaries=max(
            0,
            sum(
                bool(
                    item["payload"] is not None
                    and item["payload"].get("capture_summary")
                )
                for item in trace_results
            )
            - MAX_CAPTURE_SUMMARIES,
        ),
    )


def _trace_file_labels(trace_paths: Sequence[str]) -> List[str]:
    basenames = [os.path.basename(path) for path in trace_paths]
    if len(set(basenames)) == len(basenames):
        return basenames
    common_root = os.path.commonpath(
        [os.path.dirname(path) or os.curdir for path in trace_paths]
    )
    return [os.path.relpath(path, common_root) for path in trace_paths]


def _trace_result(
    trace_file: str,
    projection: _TraceProjection,
) -> Dict[str, Any]:
    return {
        "trace_file": trace_file,
        "execution_status": (
            ExecutionStatus.INFRA_ERROR.value
            if projection.error is not None
            else ExecutionStatus.SUCCEEDED.value
        ),
        "analyzer": projection.analyzer,
        "detector_summaries": [summary.to_dict() for summary in projection.summaries],
        "total_findings": projection.total_findings,
        "warnings": projection.warnings,
        "payload": projection.payload,
        "error": projection.error,
    }


def _analyze_trace_files(
    analyze: TraceAnalyzer,
    trace_paths: Sequence[str],
    trace_files: Sequence[str],
    *,
    arch: str,
    fallback_analyzer: str,
    tool_env: Mapping[str, str] | None,
) -> _TraceBatchProjection:
    batch = _TraceBatchProjection()
    for path, trace_file in zip(trace_paths, trace_files):
        projection = _analyze_trace_file(
            analyze,
            path,
            trace_file=trace_file,
            arch=arch,
            expected_analyzer=batch.analyzer,
            fallback_analyzer=fallback_analyzer,
            tool_env=tool_env,
        )
        if projection.analyzer_confirmed and batch.analyzer is None:
            batch.analyzer = projection.analyzer
        if len(trace_paths) == 1:
            batch.singleton_payload = projection.payload
        batch.summaries.extend(projection.summaries)
        batch.finding_groups.extend(projection.finding_groups)
        batch.total_findings += projection.total_findings
        batch.warnings.extend(
            f"{trace_file}: {warning}" for warning in projection.warnings
        )
        if projection.error is not None:
            batch.failures.append(f"{trace_file}: {projection.error}")
        batch.trace_results.append(_trace_result(trace_file, projection))
    return batch


def run_trace_analysis(
    trace_path: Union[str, Sequence[str]],
    *,
    arch: str,
    out_dir: str,
    analyzer: Optional[TraceAnalyzer] = None,
    max_findings: int = MAX_COMPACT_FINDINGS,
    tool_env: Mapping[str, str] | None = None,
) -> TraceAnalysisEvidence:
    """Analyze captured traces in stable order and persist one lossless report.

    A CUTracer invocation can emit one trace per kernel launch.  Each trace is
    analyzed independently because the deterministic analyzer's instruction
    metadata and trace indices are file-local.  The compact service evidence
    is aggregated only after every file has been attempted; the persisted
    report retains each analyzer payload and each per-file failure.
    ``tool_env`` pins CUDA host-tool resolution to the capture runtime.
    """
    if max_findings < 0:
        raise ValueError("max_findings must be non-negative")
    raw_paths = [trace_path] if isinstance(trace_path, str) else list(trace_path)
    trace_paths = sorted(
        {os.path.abspath(path) for path in raw_paths},
        key=lambda path: (os.path.basename(path), path),
    )
    analyze = analyzer or _run_data_race_analyzer
    fallback_analyzer_name = _analyzer_name(analyze)
    if not trace_paths:
        return TraceAnalysisEvidence(
            analyzer=fallback_analyzer_name,
            execution_status=ExecutionStatus.INFRA_ERROR,
            detector_summaries=[],
            findings=[],
            total_findings=0,
            error="deterministic trace analysis requires at least one trace",
        )
    trace_files = _trace_file_labels(trace_paths)
    batch = _analyze_trace_files(
        analyze,
        trace_paths,
        trace_files,
        arch=arch,
        fallback_analyzer=fallback_analyzer_name,
        tool_env=tool_env,
    )

    return _finish_trace_analysis(
        trace_paths=trace_paths,
        analyzer_name=batch.analyzer,
        fallback_analyzer_name=fallback_analyzer_name,
        summaries=batch.summaries,
        finding_groups=batch.finding_groups,
        total_findings=batch.total_findings,
        warnings=batch.warnings,
        failures=batch.failures,
        trace_results=batch.trace_results,
        singleton_payload=batch.singleton_payload,
        max_findings=max_findings,
        out_dir=out_dir,
    )
