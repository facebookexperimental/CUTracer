# pyre-strict
"""Register and memory-value trace experiment adapter."""

from __future__ import annotations

import glob
import os
import subprocess
import time
from typing import Callable, Optional

from click import ClickException
from cutracer.runner import InstrumentationConfig, run_instrumented_target, RunTarget
from cutracer.service.artifacts import local_artifact
from cutracer.service.contracts import (
    ArtifactRef,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentKind,
    ExperimentSpec,
    TraceExperimentResult,
)

TraceLocator = Callable[[str], Optional[str]]


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


def run_trace_experiment(
    spec: ExperimentSpec,
    *,
    out_dir: str,
    runner: Optional[RunTarget] = None,
    trace_locator: Optional[TraceLocator] = None,
    cutracer_so: Optional[str] = None,
    timeout: int = 1800,
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
    kernel_filter = (
        spec.unit.kernel.name
        if spec.unit.kernel is not None and spec.unit.kernel.name
        else None
    )
    started = time.monotonic()
    log_path = os.path.join(out_dir, "trace.log")
    try:
        proc = run_instrumented_target(
            spec.unit.argv,
            InstrumentationConfig(
                cutracer_so=cutracer_so,
                instrument=spec.kind.value,
                kernel_filters=kernel_filter,
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
        elif proc.returncode == 0:
            status = ExecutionStatus.INFRA_ERROR
            error = "CUTracer completed without producing a trace"
            trace = None
        else:
            status = ExecutionStatus.FAILED
            error = proc.stderr or "instrumented target failed"
            trace = None
    except subprocess.TimeoutExpired as exc:
        status = ExecutionStatus.TIMED_OUT
        error = str(exc)
        trace = None
        # The child is killed at the deadline, but subprocess still captured
        # whatever it emitted beforehand; persist it instead of dropping it.
        log = _capture_to_log(log_path, exc.stdout, exc.stderr)
    except (ClickException, OSError, ValueError) as exc:
        status = ExecutionStatus.INFRA_ERROR
        error = str(exc)
        trace = None
        log = None

    return TraceExperimentResult(
        experiment_id=spec.experiment_id,
        execution_status=status,
        mode=spec.kind,
        provenance=ExecutionProvenance(
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
        ),
        trace=trace,
        log=log,
        duration_s=time.monotonic() - started,
        error=error,
    )
