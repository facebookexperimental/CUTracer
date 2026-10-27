"""Register and memory-value trace experiment adapter."""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import time
from typing import Callable, Optional

from click import ClickException
from cutracer.runner import InstrumentationConfig, run_instrumented_target, RunTarget
from cutracer.service.artifacts import local_artifact, local_path
from cutracer.service.contracts import (
    ArtifactRef,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentKind,
    ExperimentSpec,
    KernelId,
    TraceAnalysisEvidence,
    TraceExperimentResult,
)
from cutracer.service.trace_analysis import run_trace_analysis

TraceLocator = Callable[[str], Optional[str]]
TraceAnalysisRunner = Callable[..., TraceAnalysisEvidence]
logger: logging.Logger = logging.getLogger(__name__)


def _default_trace_locator(out_dir: str) -> Optional[str]:
    matches = glob.glob(os.path.join(out_dir, "*.ndjson"))
    matches += glob.glob(os.path.join(out_dir, "*.ndjson.zst"))
    return max(matches, key=os.path.getmtime) if matches else None


def _capture_to_log(
    log_path: str,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> Optional[ArtifactRef]:
    """Persist captured stdout/stderr to a log artifact (None if unwritable)."""

    def _text(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return value or ""

    try:
        with open(log_path, "w") as fh:
            fh.write(_text(stdout) + "\n" + _text(stderr))
        return local_artifact(log_path)
    except OSError:
        return None


def _run_trace_analyzer(
    trace_path: str,
    *,
    arch: str,
    out_dir: str,
    trace_analyzer: Optional[TraceAnalysisRunner],
) -> TraceAnalysisEvidence:
    analysis_runner = trace_analyzer or run_trace_analysis
    try:
        return analysis_runner(trace_path, arch=arch, out_dir=out_dir)
    except Exception as exc:
        logger.exception("deterministic trace analysis runner failed")
        candidate = getattr(analysis_runner, "__name__", "")
        name = candidate.strip() if isinstance(candidate, str) else ""
        if not name:
            name = type(analysis_runner).__name__.strip() or "trace_analyzer"
        return TraceAnalysisEvidence(
            analyzer=name,
            execution_status=ExecutionStatus.INFRA_ERROR,
            detector_summaries=[],
            findings=[],
            total_findings=0,
            error=f"deterministic trace analysis failed: {exc}",
        )


def _selected_kernel(spec: ExperimentSpec) -> Optional[KernelId]:
    assert spec.trace is not None
    requested = spec.trace.selector.kernel
    if requested is not None and requested != spec.unit.kernel:
        raise ValueError("trace selector kernel does not match the pinned WorkUnit")
    return spec.unit.kernel


def _parse_config_pc(value: object) -> int:
    text = str(value)
    try:
        return int(text, 0)
    except ValueError:
        if text.isdecimal():
            return int(text, 10)
        raise


def _enabled_config_pc_offsets(points: object) -> list[str]:
    if not isinstance(points, dict):
        return []
    offsets: list[str] = []
    for key, point in points.items():
        if not isinstance(point, dict) or not point.get("on", True):
            continue
        offsets.append(hex(_parse_config_pc(point.get("pc", key))))
    return offsets


def _config_pc_offsets(spec: ExperimentSpec) -> list[str]:
    assert spec.trace is not None
    artifact = spec.trace.config_artifact
    if artifact is None:
        if spec.trace.selector.config_source is not None:
            raise ValueError(
                "config-targeted trace requires a resolved config artifact"
            )
        return []
    if spec.trace.selector.config_source is None:
        raise ValueError("trace config artifact requires a config_source selector")
    with open(local_path(artifact)) as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or not isinstance(payload.get("kernels"), dict):
        raise ValueError("selected trace config has no kernels mapping")

    pinned = _selected_kernel(spec)
    if pinned is None:
        raise ValueError("config-targeted trace requires a pinned WorkUnit kernel")
    matched_kernel = False
    offsets: list[str] = []
    for raw_kernel in payload["kernels"].values():
        if not isinstance(raw_kernel, dict):
            continue
        if raw_kernel.get("kernel_checksum") != pinned.cubin_hash:
            continue
        # Delay configs record the mangled CUDA name, while the WorkUnit may
        # carry the demangled launch name.  The cubin checksum is the stable
        # identity shared by both representations.
        matched_kernel = True
        offsets.extend(
            _enabled_config_pc_offsets(raw_kernel.get("instrumentation_points"))
        )
    if not matched_kernel:
        raise ValueError(
            "selected trace config has no entry for the pinned kernel checksum "
            f"{pinned.cubin_hash}; config names may be mangled while the WorkUnit "
            f"name is {pinned.name}"
        )
    if not offsets:
        raise ValueError(
            "selected trace config has no enabled PCs for the pinned kernel"
        )
    return offsets


def _trace_pc_offsets(spec: ExperimentSpec) -> list[str]:
    assert spec.trace is not None
    offsets = [*spec.trace.selector.pc_offsets, *_config_pc_offsets(spec)]
    normalized = sorted(
        {hex(int(offset, 0)) for offset in offsets}, key=lambda x: int(x, 0)
    )
    if len(normalized) > 32:
        raise ValueError("targeted trace accepts at most 32 resolved PC offsets")
    return normalized


def _trace_warp_mask(spec: ExperimentSpec) -> Optional[str]:
    assert spec.trace is not None
    selector = spec.trace.selector
    if selector.warp_id is not None and selector.warpgroup_id is not None:
        raise ValueError("trace selector cannot set both warp_id and warpgroup_id")
    if selector.warp_id is not None:
        if not 0 <= selector.warp_id < 32:
            raise ValueError("trace warp_id must be in [0, 31]")
        return hex(1 << selector.warp_id)
    if selector.warpgroup_id is not None:
        if not 0 <= selector.warpgroup_id < 8:
            raise ValueError("trace warpgroup_id must be in [0, 7]")
        return hex(0xF << (selector.warpgroup_id * 4))
    return None


def run_trace_experiment(
    spec: ExperimentSpec,
    *,
    out_dir: str,
    runner: Optional[RunTarget] = None,
    trace_locator: Optional[TraceLocator] = None,
    trace_analyzer: Optional[TraceAnalysisRunner] = None,
    cutracer_so: Optional[str] = None,
    timeout: int = 1800,
    cutracer_version: str = "",
    toolchain_version: str = "",
) -> TraceExperimentResult:
    """Capture a targeted trace with this package's CUTracer runtime."""
    if spec.kind not in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
        raise ValueError(f"not a trace experiment: {spec.kind}")
    if spec.trace is None:
        raise ValueError("trace options are required")

    os.makedirs(out_dir, exist_ok=True)
    locate = trace_locator or _default_trace_locator
    base_env = os.environ.copy()
    base_env.update(spec.unit.env)
    started = time.monotonic()
    capture_finished: Optional[float] = None
    analysis_duration_s = 0.0
    log_path = os.path.join(out_dir, "trace.log")
    try:
        selected_kernel = _selected_kernel(spec)
        kernel_filter = (
            selected_kernel.name
            if selected_kernel is not None and selected_kernel.name
            else None
        )
        pc_offsets = _trace_pc_offsets(spec)
        categories = ",".join(
            category.value for category in spec.trace.selector.instruction_categories
        )
        proc = run_instrumented_target(
            spec.unit.argv,
            InstrumentationConfig(
                cutracer_so=cutracer_so,
                instrument=spec.kind.value,
                kernel_filters=kernel_filter,
                trace_instr_categories=categories or None,
                trace_launch_id=spec.trace.selector.launch_id,
                trace_warp_mask=_trace_warp_mask(spec),
                trace_pc_offsets=",".join(pc_offsets) or None,
                output_dir=out_dir,
                trace_size_limit_mb=spec.trace.trace_size_limit_mb,
                cwd=spec.unit.cwd or None,
                timeout=timeout,
                base_env=base_env,
            ),
            runner=runner,
        )
        log = _capture_to_log(log_path, proc.stdout, proc.stderr)

        trace_path = locate(out_dir)
        if proc.returncode == 0 and trace_path is not None:
            status = ExecutionStatus.SUCCEEDED
            error = None
            media_type = (
                "application/zstd"
                if trace_path.endswith(".zst")
                else "application/x-ndjson"
            )
            trace = local_artifact(trace_path, media_type=media_type)
            capture_finished = time.monotonic()
            analysis_started = capture_finished
            analysis = _run_trace_analyzer(
                trace_path,
                arch=spec.unit.arch,
                out_dir=out_dir,
                trace_analyzer=trace_analyzer,
            )
            analysis_duration_s = time.monotonic() - analysis_started
        elif proc.returncode == 0:
            status = ExecutionStatus.INFRA_ERROR
            error = "CUTracer completed without producing a trace"
            trace = None
            analysis = None
        else:
            status = ExecutionStatus.FAILED
            error = proc.stderr or "instrumented target failed"
            trace = None
            analysis = None
    except subprocess.TimeoutExpired as exc:
        status = ExecutionStatus.TIMED_OUT
        error = str(exc)
        trace = None
        analysis = None
        # The child is killed at the deadline, but subprocess still captured
        # whatever it emitted beforehand; persist it instead of dropping it.
        log = _capture_to_log(log_path, exc.stdout, exc.stderr)
    except (ClickException, OSError, ValueError) as exc:
        status = ExecutionStatus.INFRA_ERROR
        error = str(exc)
        trace = None
        analysis = None
        log = None

    return TraceExperimentResult(
        experiment_id=spec.experiment_id,
        execution_status=status,
        mode=spec.kind,
        provenance=ExecutionProvenance(
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
            tool_version=cutracer_version,
            toolchain_version=toolchain_version,
        ),
        trace=trace,
        log=log,
        analysis=analysis,
        duration_s=(
            (capture_finished if capture_finished is not None else time.monotonic())
            - started
        ),
        analysis_duration_s=analysis_duration_s,
        error=error,
    )
