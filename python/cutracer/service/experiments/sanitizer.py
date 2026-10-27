"""Compute Sanitizer experiment adapter."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Callable, Optional

from click import ClickException
from cutracer.compute_sanitizer.run import (
    ComputeSanitizerConfig,
    run_compute_sanitizer,
    RunTarget,
    SupportsTensorOps,
)
from cutracer.service.artifacts import publish_local_artifact
from cutracer.service.contracts import (
    ArtifactRef,
    ExecutionStatus,
    ExperimentKind,
    ExperimentSpec,
    SanitizerSweepResult,
)
from cutracer.service.sanitizer_results import build_sanitizer_result

Reader = Callable[[str], str]


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

    return build_sanitizer_result(
        spec,
        execution_status=status,
        log_text=text,
        log=log,
        duration_s=time.monotonic() - started,
        error=error,
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
