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
    ResultCompleteness,
    StressOutcome,
    StressTestResult,
    TriggeringDelayConfig,
)
from cutracer.stress.stress import run_stress, StressConfig, StressResult


def _campaign_completed(result: StressResult, planned_trials: int) -> bool:
    return (
        result.completed_trials == planned_trials
        and result.infra_errors == 0
        and result.timed_out == 0
    )


def _classify_result(
    result: StressResult,
    *,
    planned_trials: int,
    stop_on_first: bool,
    triggering_config: Optional[TriggeringDelayConfig],
) -> tuple[
    StressOutcome,
    ExecutionStatus,
    Optional[str],
    ResultCompleteness,
]:
    if triggering_config is not None:
        attempted_trials = (
            result.completed_trials + result.infra_errors + result.timed_out
        )
        if _campaign_completed(result, planned_trials):
            completeness = ResultCompleteness.COMPLETE
        elif (
            stop_on_first
            and attempted_trials < planned_trials
            and result.infra_errors == 0
            and result.timed_out == 0
        ):
            completeness = ResultCompleteness.COMPLETE_EARLY
        else:
            completeness = ResultCompleteness.PARTIAL
        return (
            StressOutcome.REPRODUCED,
            ExecutionStatus.SUCCEEDED,
            None,
            completeness,
        )
    if result.unattributed_reproductions > 0:
        return (
            StressOutcome.UNATTRIBUTED_REPRODUCTION,
            ExecutionStatus.SUCCEEDED,
            None,
            (
                ResultCompleteness.COMPLETE
                if _campaign_completed(result, planned_trials)
                else ResultCompleteness.PARTIAL
            ),
        )
    if _campaign_completed(result, planned_trials) and result.reproductions == 0:
        return (
            StressOutcome.NOT_REPRODUCED,
            ExecutionStatus.SUCCEEDED,
            None,
            ResultCompleteness.COMPLETE,
        )
    if result.infra_errors > 0:
        return (
            StressOutcome.INCOMPLETE,
            ExecutionStatus.INFRA_ERROR,
            "stress campaign had one or more infrastructure failures",
            ResultCompleteness.PARTIAL,
        )
    if result.timed_out > 0:
        return (
            StressOutcome.INCOMPLETE,
            ExecutionStatus.TIMED_OUT,
            "stress campaign timed out before completing its planned trials",
            ResultCompleteness.PARTIAL,
        )
    return (
        StressOutcome.INCOMPLETE,
        ExecutionStatus.INFRA_ERROR,
        "stress campaign did not complete enough valid oracle trials",
        ResultCompleteness.PARTIAL,
    )


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
    planned_trials = spec.stress.planned_trials

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
            planned_trials=planned_trials,
            completeness=ResultCompleteness.PARTIAL,
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
            artifact = local_artifact(
                stable_config_path,
                media_type="application/json",
                include_sha256=True,
            )
            if artifact.sha256 is None or artifact.size_bytes is None:
                raise OSError("triggering config disappeared before it was hashed")
            triggering_config = TriggeringDelayConfig(
                artifact=artifact,
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
        except OSError as exc:
            # The campaign already ran to completion (a reproduction was even
            # found); only persisting its config failed. Preserve the real
            # accumulated counts instead of discarding them as zero.
            terminal = _terminal(ExecutionStatus.INFRA_ERROR, str(exc))
            terminal.completed_trials = result.completed_trials
            terminal.reproductions = result.reproductions
            terminal.infra_errors = result.infra_errors + 1
            terminal.timed_out_trials = result.timed_out
            terminal.unattributed_reproductions = result.unattributed_reproductions
            terminal.log = log
            return terminal

    outcome, status, error, completeness = _classify_result(
        result,
        planned_trials=planned_trials,
        stop_on_first=spec.stress.stop_on_first_reproduction,
        triggering_config=triggering_config,
    )

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
        planned_trials=planned_trials,
        timed_out_trials=result.timed_out,
        completeness=completeness,
        unattributed_reproductions=result.unattributed_reproductions,
    )
