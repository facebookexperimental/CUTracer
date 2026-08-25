"""Delay-config replay and reduction experiment adapter."""

from __future__ import annotations

import os
import shlex
import time
from typing import Optional

from cutracer.reduce.reduce import (
    ConfigDoesNotTriggerError,
    reduce_bisect,
    reduce_delay_points,
    ReduceConfig,
    ReplayConfig,
    ReplayExecutionError,
    ReplayOutcome,
)
from cutracer.reduce.report import generate_report, save_report
from cutracer.runner import RunTarget
from cutracer.service.artifacts import local_artifact, local_path
from cutracer.service.contracts import (
    EssentialDelayPoint,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentKind,
    ExperimentSpec,
    ReduceExperimentResult,
    ReduceOutcome,
)


def run_reduce_experiment(
    spec: ExperimentSpec,
    *,
    out_dir: str,
    runner: Optional[RunTarget] = None,
    cutracer_so: Optional[str] = None,
    timeout: int = 1800,
) -> ReduceExperimentResult:
    """Reduce a required triggering config; never discover one implicitly."""
    if spec.kind != ExperimentKind.REDUCE_DELAY_CONFIG:
        raise ValueError(f"not a reduce experiment: {spec.kind}")
    if spec.reduction is None:
        raise ValueError("reduce requires an existing triggering delay config")
    if spec.unit.oracle is None:
        raise ValueError("reduce requires the same pre-approved correctness oracle")

    os.makedirs(out_dir, exist_ok=True)
    input_path = local_path(spec.reduction.triggering_config.artifact)
    report_path = os.path.join(out_dir, "reduce_report.json")
    minimal_path = os.path.join(out_dir, "minimal_config.json")
    base_env = os.environ.copy()
    base_env.update(spec.unit.env)
    kernel_filter = (
        spec.unit.kernel.name
        if spec.unit.kernel is not None and spec.unit.kernel.name
        else None
    )
    started = time.monotonic()
    report = None
    minimized = None
    points = []
    note = ""

    reduce_config = ReduceConfig(
        config_path=input_path,
        output_path=minimal_path,
        replay=ReplayConfig(
            oracle_argv=list(spec.unit.oracle.argv),
            cutracer_so=cutracer_so,
            kernel_filters=kernel_filter,
            delay_warpgroup_id=spec.reduction.triggering_config.warpgroup_id,
            output_dir=out_dir,
            timeout=timeout,
            cwd=spec.unit.cwd or None,
            base_env=base_env,
            not_interesting_exit_codes=tuple(
                spec.unit.oracle.not_interesting_exit_codes
            ),
        ),
        replay_runner=runner,
    )

    try:
        if spec.reduction.strategy == "linear":
            result = reduce_delay_points(
                reduce_config,
                confidence_runs=spec.reduction.confidence_runs,
            )
        elif spec.reduction.strategy == "bisect":
            result = reduce_bisect(
                reduce_config,
                confidence_runs=spec.reduction.confidence_runs,
            )
        else:
            raise ValueError(f"unknown reduction strategy: {spec.reduction.strategy}")

        report_data = generate_report(
            result=result,
            config_path=input_path,
            test_script=shlex.join(spec.unit.oracle.argv),
        )
        save_report(report_data, report_path)
        report = local_artifact(report_path, media_type="application/json")
        if result.minimal_config_path is not None:
            minimized = local_artifact(
                result.minimal_config_path,
                media_type="application/json",
            )
        points = [
            EssentialDelayPoint(
                kernel_name=point.kernel_name,
                pc_offset=point.pc_offset,
                sass=point.sass,
                delay_ns=point.delay_ns,
            )
            for point in result.essential_points
        ]
        outcome = ReduceOutcome.REDUCED
        status = ExecutionStatus.SUCCEEDED
        error = None
        if not points:
            # A legitimate deterministic result: the config reproduces even with
            # every delay point disabled, so there is nothing to minimize. Keep
            # REDUCED but flag it so downstream does not read it as a multi-point
            # reduction that happened to find nothing.
            note = (
                "0 essential points -- reproduces without any injected delay "
                "(deterministic, not a timing-dependent race)"
            )
    except ReplayExecutionError as exc:
        outcome = ReduceOutcome.FAILED
        status = (
            ExecutionStatus.TIMED_OUT
            if exc.outcome == ReplayOutcome.TIMED_OUT
            else ExecutionStatus.INFRA_ERROR
        )
        error = str(exc)
    except ConfigDoesNotTriggerError:
        # Robust typed match (subclass of ValueError) for the stale-config case,
        # replacing a fragile substring match on the human-readable message.
        outcome = ReduceOutcome.REPLAY_FAILED
        status = ExecutionStatus.FAILED
        error = "triggering config no longer reproduces under the pinned oracle"
    except ValueError as exc:
        outcome = ReduceOutcome.FAILED
        status = ExecutionStatus.INFRA_ERROR
        error = str(exc)
    except OSError as exc:
        outcome = ReduceOutcome.FAILED
        status = ExecutionStatus.INFRA_ERROR
        error = str(exc)

    return ReduceExperimentResult(
        experiment_id=spec.experiment_id,
        execution_status=status,
        outcome=outcome,
        provenance=ExecutionProvenance(
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
        ),
        input_config=spec.reduction.triggering_config,
        minimized_config=minimized,
        report=report,
        essential_points=points,
        duration_s=time.monotonic() - started,
        error=error,
        note=note,
    )
