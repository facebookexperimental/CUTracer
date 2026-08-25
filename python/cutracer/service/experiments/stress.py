"""Random-delay stress experiment adapter."""

from __future__ import annotations

import os
import shutil
import time
from typing import Optional

from click import ClickException
from cutracer.runner import RunTarget
from cutracer.service.artifacts import local_artifact
from cutracer.service.contracts import (
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentKind,
    ExperimentSpec,
    StressOutcome,
    StressTestResult,
    TriggeringDelayConfig,
)
from cutracer.stress.stress import run_stress, StressConfig


def run_stress_experiment(
    spec: ExperimentSpec,
    *,
    out_dir: str,
    runner: Optional[RunTarget] = None,
    cutracer_so: Optional[str] = None,
    timeout: int = 1800,
    cutracer_version: str = "",
    toolchain_version: str = "",
) -> StressTestResult:
    """Run CUTracer's in-process discovery loop and map its typed result."""
    if spec.kind != ExperimentKind.RANDOM_DELAY_STRESS or spec.stress is None:
        raise ValueError(f"not a stress experiment: {spec.kind}")
    if spec.unit.oracle is None:
        raise ValueError("random-delay stress requires a correctness oracle")

    os.makedirs(out_dir, exist_ok=True)
    started = time.monotonic()

    def _provenance() -> ExecutionProvenance:
        return ExecutionProvenance(
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
            tool_version=cutracer_version,
            toolchain_version=toolchain_version,
        )

    def _terminal(status: ExecutionStatus, error: str) -> StressTestResult:
        return StressTestResult(
            experiment_id=spec.experiment_id,
            execution_status=status,
            outcome=StressOutcome.INCOMPLETE,
            completed_trials=0,
            reproductions=0,
            infra_errors=1,
            provenance=_provenance(),
            duration_s=time.monotonic() - started,
            error=error,
        )

    base_env = os.environ.copy()
    base_env.update(spec.unit.env)
    kernel_filter = (
        spec.unit.kernel.name
        if spec.unit.kernel is not None and spec.unit.kernel.name
        else None
    )
    try:
        result = run_stress(
            StressConfig(
                oracle_argv=list(spec.unit.oracle.argv),
                delay_ladder_ns=list(spec.stress.delay_ladder_ns),
                enable_prob=spec.stress.enable_prob,
                warpgroup_ids=list(spec.stress.warpgroup_ids),
                attempts_per_delay=spec.stress.attempts_per_delay,
                stop_on_first=spec.stress.stop_on_first_reproduction,
                not_interesting_exit_codes=list(
                    spec.unit.oracle.not_interesting_exit_codes
                ),
                kernel_filters=kernel_filter,
                output_dir=out_dir,
                timeout=timeout,
                cwd=spec.unit.cwd or None,
                base_env=base_env,
            ),
            cutracer_so=cutracer_so,
            runner=runner,
        )
    except (ClickException, OSError, ValueError) as exc:
        # ``run_stress`` catches per-attempt ``TimeoutExpired`` internally and
        # reports it via ``StressResult.timed_out``; a timeout never propagates
        # here, so it is classified below rather than in this except clause.
        return _terminal(ExecutionStatus.INFRA_ERROR, str(exc))

    log = (
        local_artifact(result.log_path)
        if result.log_path is not None and os.path.isfile(result.log_path)
        else None
    )
    triggering_config = None
    if result.triggering is not None:
        stable_config_path = os.path.join(out_dir, "triggering_config.json")
        try:
            shutil.copyfile(result.triggering.config_path, stable_config_path)
        except OSError as exc:
            # The campaign already ran to completion (a reproduction was even
            # found); only persisting its config failed. Preserve the real
            # accumulated counts instead of discarding them as zero.
            terminal = _terminal(ExecutionStatus.INFRA_ERROR, str(exc))
            terminal.completed_trials = result.completed_trials
            terminal.reproductions = result.reproductions
            terminal.infra_errors = result.infra_errors + 1
            terminal.log = log
            return terminal
        triggering_config = TriggeringDelayConfig(
            artifact=local_artifact(
                stable_config_path,
                media_type="application/json",
                include_sha256=True,
            ),
            work_unit_id=spec.unit.unit_id,
            target_argv=list(spec.unit.argv),
            oracle=spec.unit.oracle,
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
            delay_ns=result.triggering.delay_ns,
            enable_prob=result.triggering.enable_prob,
            warpgroup_id=result.triggering.warpgroup_id,
            attempt_index=result.triggering.attempt_index,
            completed_trials=result.completed_trials,
            reproductions=result.reproductions,
            reproduction_rate=result.reproduction_rate,
            cutracer_version=cutracer_version,
            toolchain_version=toolchain_version,
        )

    if triggering_config is not None:
        outcome = StressOutcome.REPRODUCED
        status = ExecutionStatus.SUCCEEDED
        error = None
    elif result.completed_trials > 0:
        # At least one oracle trial ran to a clean verdict and none reproduced.
        # Infra errors / timeouts on other attempts are surfaced as metadata
        # (infra_errors) rather than downgrading a genuine non-reproduction.
        outcome = StressOutcome.NOT_REPRODUCED
        status = ExecutionStatus.SUCCEEDED
        error = None
    elif result.timed_out > 0 and result.timed_out >= result.infra_errors:
        # No attempt produced a verdict and the dominant failure was the repro
        # command hanging: report a timeout, distinct from a generic infra error.
        outcome = StressOutcome.INCOMPLETE
        status = ExecutionStatus.TIMED_OUT
        error = "stress campaign timed out before completing any valid oracle trial"
    else:
        outcome = StressOutcome.INCOMPLETE
        status = ExecutionStatus.INFRA_ERROR
        error = "stress campaign did not complete enough valid oracle trials"

    return StressTestResult(
        experiment_id=spec.experiment_id,
        execution_status=status,
        outcome=outcome,
        completed_trials=result.completed_trials,
        reproductions=result.reproductions,
        infra_errors=result.infra_errors,
        provenance=_provenance(),
        triggering_config=triggering_config,
        log=log,
        duration_s=time.monotonic() - started,
        error=error,
    )
