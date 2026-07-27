# pyre-strict
"""Compute Sanitizer experiment adapter."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Callable, List, Optional

from click import ClickException
from cutracer.analyze.fb.compute_sanitizer.parser import (
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
    SanitizerOutcome,
    SanitizerSweepResult,
    SourceLoc,
)

Reader = Callable[[str], str]


def _default_reader(path: str) -> str:
    # Sanitizer logs may contain non-UTF-8 bytes. Preserve the rest of a useful
    # report instead of turning one bad byte into an infrastructure failure.
    with open(path, errors="replace") as fh:
        return fh.read()


def parse_racecheck_log(text: str) -> List[FindingRecord]:
    """Convert the canonical CUTracer parser output to service wire records."""
    records: List[FindingRecord] = []
    for finding in parse_canonical_racecheck_log(text).findings:
        payload = finding.payload
        accesses = payload.get("accesses", [])
        primary = accesses[0] if isinstance(accesses, list) and accesses else {}
        if not isinstance(primary, dict):
            primary = {}

        file = primary.get("file")
        line = primary.get("line")
        source = None
        if isinstance(file, str) and isinstance(line, int):
            source = SourceLoc(file=file, line=line)

        kernel = primary.get("func")
        count = payload.get("max_hazards", 1)
        records.append(
            FindingRecord(
                source_tool="compute_sanitizer/racecheck",
                error_type="race-condition",
                kernel_name=kernel if isinstance(kernel, str) else None,
                source=source,
                count=count if isinstance(count, int) else 1,
                raw=str(payload.get("raw_block", "")),
            )
        )
    return records


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

    findings = parse_racecheck_log(text) if tool == "racecheck" else []
    if findings:
        outcome = SanitizerOutcome.FINDING
    elif status != ExecutionStatus.SUCCEEDED or log is None or tool != "racecheck":
        outcome = SanitizerOutcome.UNKNOWN
    else:
        outcome = SanitizerOutcome.CLEAN

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
    )
