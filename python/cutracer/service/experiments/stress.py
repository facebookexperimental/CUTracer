"""Random-delay stress experiment adapter."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence

from click import ClickException
from cutracer.runner import RunTarget
from cutracer.service.artifacts import publish_local_artifact
from cutracer.service.contracts import (
    ArtifactRef,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentKind,
    ExperimentSpec,
    ResultCompleteness,
    StressOutcome,
    StressTestResult,
    TriggeringDelayConfig,
)
from cutracer.stress.retention import (
    MAX_REJECTED_CONFIGS as _MAX_REJECTED_CONFIGS,
    retain_oldest_and_newest,
)
from cutracer.stress.stress import (
    run_stress,
    StressConfig,
    StressResult,
    TriggeringConfig,
)


logger: logging.Logger = logging.getLogger(__name__)
StressProgressCallback = Callable[[StressTestResult], None]
MAX_REJECTED_CONFIGS = _MAX_REJECTED_CONFIGS
_REJECTED_CONFIG_NAME_RE = re.compile(r"^rejected_delay_config_[0-9a-f]{64}\.json$")


def _promote_rejected_configs(
    result: StressResult,
    out_dir: str,
) -> List[ArtifactRef]:
    """Move rejected delay configs out of the attempt dir before it is removed.

    Diagnostic artifacts only. A rejected config is by definition not a
    triggering config, so it is published under its own name and never as
    ``triggering_config``. Promotion failures are swallowed: losing a
    diagnostic must not turn a classified result into an error.
    """
    unique_sources = list(dict.fromkeys(result.rejected_config_paths))
    preferred = retain_oldest_and_newest(unique_sources)
    preferred_set = set(preferred)
    sources = [
        *preferred,
        *(source for source in unique_sources if source not in preferred_set),
    ]
    promoted: dict[str, ArtifactRef] = {}
    attempted = 0
    for source in sources:
        if len(promoted) >= MAX_REJECTED_CONFIGS:
            break
        attempted += 1
        destination = os.path.join(out_dir, "rejected_delay_config.json")
        try:
            artifact = publish_local_artifact(
                source,
                destination,
                media_type="application/json",
                content_addressed=True,
            )
            promoted[artifact.relative_path] = artifact
        except Exception:
            logger.exception("failed to promote rejected delay config %s", source)
            continue
    if attempted < len(unique_sources):
        logger.warning(
            "omitting %d rejected delay config source paths above the diagnostic cap",
            len(unique_sources) - attempted,
        )
    return list(promoted.values())


def _merge_rejected_configs(
    previous: Sequence[ArtifactRef],
    current: Sequence[ArtifactRef],
) -> List[ArtifactRef]:
    by_path = {artifact.relative_path: artifact for artifact in previous}
    for artifact in current:
        by_path[artifact.relative_path] = artifact
    merged = list(by_path.values())
    return retain_oldest_and_newest(merged)


def _cleanup_unreferenced_rejected_configs(
    out_dir: str,
    referenced: Sequence[ArtifactRef],
) -> None:
    referenced_names = {artifact.relative_path for artifact in referenced}
    for name in os.listdir(out_dir):
        if name in referenced_names or _REJECTED_CONFIG_NAME_RE.fullmatch(name) is None:
            continue
        try:
            os.unlink(os.path.join(out_dir, name))
        except OSError as error:
            logger.warning(
                "could not remove unreferenced rejected config %s: %s",
                name,
                error,
            )


def _stage_rejected_configs_for_resume(
    resume_from: StressTestResult,
    out_dir: str,
    attempt_dir: str,
) -> List[str]:
    staged: List[str] = []
    for index, artifact in enumerate(resume_from.rejected_configs):
        source = os.path.join(out_dir, artifact.relative_path)
        destination = os.path.join(
            attempt_dir,
            f"resumed_rejected_config_{index}.json",
        )
        try:
            shutil.copyfile(source, destination)
        except OSError as error:
            logger.warning(
                "could not stage rejected config %s for resume: %s",
                source,
                error,
            )
            continue
        staged.append(destination)
    return staged


def _campaign_completed(result: StressResult, planned_trials: int) -> bool:
    return (
        result.completed_trials == planned_trials
        and result.infra_errors == 0
        and result.timed_out == 0
    )


def _classify_stress_result(
    result: StressResult,
    *,
    planned_trials: int,
    stop_on_first: bool,
    reproduced: bool,
) -> tuple[StressOutcome, ExecutionStatus, Optional[str], ResultCompleteness]:
    if reproduced:
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
        return StressOutcome.REPRODUCED, ExecutionStatus.SUCCEEDED, None, completeness
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


def _partial_stress_outcome(
    triggering_config: Optional[TriggeringDelayConfig],
    *,
    reproductions: int,
    unattributed_reproductions: int,
) -> StressOutcome:
    if triggering_config is not None:
        return StressOutcome.REPRODUCED
    if reproductions == 0 and unattributed_reproductions > 0:
        return StressOutcome.UNATTRIBUTED_REPRODUCTION
    return StressOutcome.INCOMPLETE


def _stress_provenance(
    spec: ExperimentSpec,
    cutracer_version: str,
    toolchain_version: str,
) -> ExecutionProvenance:
    return ExecutionProvenance(
        source_revision=spec.unit.source_revision,
        arch=spec.unit.arch,
        kernel=spec.unit.kernel,
        tool_version=cutracer_version,
        toolchain_version=toolchain_version,
    )


def _resume_core_result(
    spec: ExperimentSpec,
    out_dir: str,
    attempt_dir: str,
    planned_trials: int,
    provenance: ExecutionProvenance,
    resume_from: Optional[StressTestResult],
) -> Optional[StressResult]:
    if resume_from is None:
        return None
    if (
        resume_from.experiment_id != spec.experiment_id
        or resume_from.planned_trials != planned_trials
        or resume_from.provenance != provenance
        or resume_from.completeness != ResultCompleteness.PARTIAL
    ):
        raise ValueError("resumed stress progress does not match the experiment")
    resume_log_path = (
        None
        if resume_from.log is None
        else os.path.join(out_dir, resume_from.log.relative_path)
    )
    resume_triggering: Optional[TriggeringConfig] = None
    if resume_from.triggering_config is not None:
        config = resume_from.triggering_config
        resume_triggering = TriggeringConfig(
            config_path=os.path.join(out_dir, config.artifact.relative_path),
            delay_ns=config.delay_ns,
            warpgroup_id=config.warpgroup_id,
            enable_prob=config.enable_prob,
            attempt_index=config.attempt_index,
            enabled_point_count=config.enabled_point_count,
        )
    return StressResult(
        reproduced=resume_triggering is not None,
        completed_trials=resume_from.completed_trials,
        reproductions=resume_from.reproductions,
        infra_errors=resume_from.infra_errors,
        reproduction_rate=resume_from.reproduction_rate,
        triggering=resume_triggering,
        log_path=resume_log_path,
        timed_out=resume_from.timed_out_trials,
        unattributed_reproductions=resume_from.unattributed_reproductions,
        rejected_config_paths=_stage_rejected_configs_for_resume(
            resume_from,
            out_dir,
            attempt_dir,
        ),
        log_size_bytes=(
            None
            if resume_from.log is None or resume_from.log.size_bytes is None
            else resume_from.log.size_bytes
        ),
    )


def _triggering_delay_config(
    spec: ExperimentSpec,
    core_result: StressResult,
    artifact: ArtifactRef,
    *,
    cutracer_version: str,
    toolchain_version: str,
) -> TriggeringDelayConfig:
    if core_result.triggering is None:
        raise ValueError("stress result requires a triggering config")
    oracle = spec.unit.oracle
    if oracle is None:
        raise ValueError("stress result requires a pinned oracle")
    return TriggeringDelayConfig(
        artifact=artifact,
        work_unit_id=spec.unit.unit_id,
        target_argv=list(spec.unit.argv),
        oracle=oracle,
        source_revision=spec.unit.source_revision,
        arch=spec.unit.arch,
        kernel=spec.unit.kernel,
        delay_ns=core_result.triggering.delay_ns,
        enable_prob=core_result.triggering.enable_prob,
        warpgroup_id=core_result.triggering.warpgroup_id,
        attempt_index=core_result.triggering.attempt_index,
        completed_trials=core_result.completed_trials,
        reproductions=core_result.reproductions,
        reproduction_rate=(
            core_result.reproductions / core_result.completed_trials
            if core_result.completed_trials
            else 0.0
        ),
        enabled_point_count=core_result.triggering.enabled_point_count,
        cutracer_version=cutracer_version,
        toolchain_version=toolchain_version,
    )


def _republish_stress_log(
    out_dir: str,
    log: Optional[ArtifactRef],
) -> Optional[ArtifactRef]:
    if log is None or log.relative_path == "stress.log":
        return log
    source = os.path.join(out_dir, log.relative_path)
    try:
        return publish_local_artifact(
            source,
            os.path.join(out_dir, "stress.log"),
            media_type="text/plain",
        )
    except OSError as error:
        logger.warning(
            "could not republish stress progress log %s as stress.log: %s",
            log.relative_path,
            error,
        )
    return log if os.path.isfile(source) else None


def _recover_stress_progress(
    out_dir: str,
    latest_core: Optional[StressResult],
    latest: Optional[StressTestResult],
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    Optional[ArtifactRef],
    Optional[TriggeringDelayConfig],
]:
    counts: tuple[int, int, int, int, int]
    if latest_core is not None:
        counts = (
            latest_core.completed_trials,
            latest_core.reproductions,
            latest_core.infra_errors + 1,
            latest_core.timed_out,
            latest_core.unattributed_reproductions,
        )
    elif latest is not None:
        counts = (
            latest.completed_trials,
            latest.reproductions,
            latest.infra_errors + 1,
            latest.timed_out_trials,
            latest.unattributed_reproductions,
        )
    else:
        counts = (0, 0, 1, 0, 0)
    terminal_log = _republish_stress_log(
        out_dir,
        None if latest is None else latest.log,
    )
    triggering = None if latest is None else latest.triggering_config
    return (
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4],
        terminal_log,
        triggering,
    )


def _publish_stress_artifacts(
    spec: ExperimentSpec,
    result: StressResult,
    out_dir: str,
    *,
    cutracer_version: str,
    toolchain_version: str,
) -> tuple[Optional[ArtifactRef], Optional[TriggeringDelayConfig], list[str]]:
    log: Optional[ArtifactRef] = None
    triggering: Optional[TriggeringDelayConfig] = None
    errors: list[str] = []
    if result.log_path is None:
        errors.append("stress campaign did not persist stress.log")
    else:
        try:
            log = publish_local_artifact(
                result.log_path,
                os.path.join(out_dir, "stress.log"),
                media_type="text/plain",
            )
        except OSError as exc:
            errors.append(f"could not publish stress log: {exc}")
    if result.triggering is not None:
        try:
            artifact = publish_local_artifact(
                result.triggering.config_path,
                os.path.join(out_dir, "triggering_config.json"),
                media_type="application/json",
                content_addressed=True,
            )
            triggering = _triggering_delay_config(
                spec,
                result,
                artifact,
                cutracer_version=cutracer_version,
                toolchain_version=toolchain_version,
            )
        except OSError as exc:
            errors.append(f"could not publish triggering config: {exc}")
    return log, triggering, errors


def _fallback_stress_artifacts(
    out_dir: str,
    log: Optional[ArtifactRef],
    triggering: Optional[TriggeringDelayConfig],
    latest: Optional[StressTestResult],
) -> tuple[Optional[ArtifactRef], Optional[TriggeringDelayConfig]]:
    if latest is not None:
        log = log or latest.log
        triggering = triggering or latest.triggering_config
    return _republish_stress_log(out_dir, log), triggering


def _terminal_stress_result(
    spec: ExperimentSpec,
    provenance: ExecutionProvenance,
    planned_trials: int,
    started: float,
    status: ExecutionStatus,
    error: str,
    *,
    outcome: StressOutcome = StressOutcome.INCOMPLETE,
    completed_trials: int = 0,
    reproductions: int = 0,
    infra_errors: int = 1,
    timed_out_trials: int = 0,
    unattributed_reproductions: int = 0,
    log: Optional[ArtifactRef] = None,
    triggering_config: Optional[TriggeringDelayConfig] = None,
    rejected_configs: Optional[Sequence[ArtifactRef]] = None,
) -> StressTestResult:
    return StressTestResult(
        experiment_id=spec.experiment_id,
        execution_status=status,
        outcome=outcome,
        completed_trials=completed_trials,
        reproductions=reproductions,
        infra_errors=infra_errors,
        provenance=provenance,
        triggering_config=triggering_config,
        log=log,
        duration_s=time.monotonic() - started,
        error=error,
        planned_trials=planned_trials,
        timed_out_trials=timed_out_trials,
        completeness=ResultCompleteness.PARTIAL,
        unattributed_reproductions=unattributed_reproductions,
        rejected_configs=list(rejected_configs or []),
    )


def _publish_progress_artifacts(
    spec: ExperimentSpec,
    core_result: StressResult,
    out_dir: str,
    *,
    cutracer_version: str,
    toolchain_version: str,
) -> tuple[Optional[ArtifactRef], Optional[TriggeringDelayConfig], Optional[OSError]]:
    error: Optional[OSError] = None
    log: Optional[ArtifactRef] = None
    if core_result.log_path is None:
        error = OSError("stress campaign did not persist stress.log")
    else:
        try:
            log = publish_local_artifact(
                core_result.log_path,
                os.path.join(out_dir, "stress.log"),
                media_type="text/plain",
            )
        except OSError as exc:
            error = exc
    triggering: Optional[TriggeringDelayConfig] = None
    if core_result.triggering is not None:
        try:
            triggering = _triggering_delay_config(
                spec,
                core_result,
                publish_local_artifact(
                    core_result.triggering.config_path,
                    os.path.join(out_dir, "triggering_config.json"),
                    media_type="application/json",
                    content_addressed=True,
                ),
                cutracer_version=cutracer_version,
                toolchain_version=toolchain_version,
            )
        except OSError as exc:
            if error is None:
                error = exc
    return log, triggering, error


def _adapt_core_progress(
    spec: ExperimentSpec,
    core_result: StressResult,
    previous: Optional[StressTestResult],
    out_dir: str,
    planned_trials: int,
    provenance: ExecutionProvenance,
    started: float,
    *,
    cutracer_version: str,
    toolchain_version: str,
) -> tuple[StressTestResult, Optional[OSError]]:
    log, triggering, error = _publish_progress_artifacts(
        spec,
        core_result,
        out_dir,
        cutracer_version=cutracer_version,
        toolchain_version=toolchain_version,
    )
    durable_log = log
    if durable_log is None and previous is not None:
        durable_log = previous.log
    durable_trigger = (
        triggering
        if triggering is not None
        else None
        if previous is None
        else previous.triggering_config
    )
    durable_rejected = _merge_rejected_configs(
        [] if previous is None else previous.rejected_configs,
        _promote_rejected_configs(core_result, out_dir),
    )
    progress = StressTestResult(
        experiment_id=spec.experiment_id,
        execution_status=ExecutionStatus.SUCCEEDED,
        outcome=_partial_stress_outcome(
            durable_trigger,
            reproductions=core_result.reproductions,
            unattributed_reproductions=core_result.unattributed_reproductions,
        ),
        completed_trials=core_result.completed_trials,
        reproductions=core_result.reproductions,
        infra_errors=core_result.infra_errors,
        provenance=provenance,
        triggering_config=durable_trigger,
        log=durable_log,
        duration_s=time.monotonic() - started,
        planned_trials=planned_trials,
        timed_out_trials=core_result.timed_out,
        completeness=ResultCompleteness.PARTIAL,
        unattributed_reproductions=core_result.unattributed_reproductions,
        rejected_configs=durable_rejected,
    )
    return progress, error


@dataclass
class _StressProgressAdapter:
    spec: ExperimentSpec
    out_dir: str
    planned_trials: int
    provenance: ExecutionProvenance
    started: float
    cutracer_version: str
    toolchain_version: str
    callback: Optional[StressProgressCallback]
    latest: Optional[StressTestResult]
    latest_core: Optional[StressResult] = None

    def __call__(self, core_result: StressResult) -> None:
        previous = self.latest
        self.latest_core = core_result
        self.latest, error = _adapt_core_progress(
            self.spec,
            core_result,
            previous,
            self.out_dir,
            self.planned_trials,
            self.provenance,
            self.started,
            cutracer_version=self.cutracer_version,
            toolchain_version=self.toolchain_version,
        )
        if self.callback is None:
            _cleanup_unreferenced_rejected_configs(
                self.out_dir,
                self.latest.rejected_configs,
            )
        if error is not None:
            raise error
        if self.callback is not None:
            self.callback(self.latest)


def _run_stress_attempt(
    spec: ExperimentSpec,
    *,
    out_dir: str,
    attempt_dir: str,
    runner: Optional[RunTarget] = None,
    cutracer_so: Optional[str] = None,
    timeout: int = 1800,
    cutracer_version: str = "",
    toolchain_version: str = "",
    resolved_oracle_argv: Optional[Sequence[str]] = None,
    resolved_cwd: Optional[str] = None,
    resolved_base_env: Optional[Mapping[str, str]] = None,
    resume_from: Optional[StressTestResult] = None,
    progress_callback: Optional[StressProgressCallback] = None,
    use_bundled_cuda_tools: bool = True,
) -> StressTestResult:
    assert spec.stress is not None
    assert spec.unit.oracle is not None
    started = time.monotonic()
    planned_trials = spec.stress.planned_trials
    provenance = _stress_provenance(spec, cutracer_version, toolchain_version)

    if resolved_base_env is None:
        base_env = os.environ.copy()
        base_env.update(spec.unit.env)
    else:
        base_env = dict(resolved_base_env)
    oracle_argv = list(
        spec.unit.oracle.argv if resolved_oracle_argv is None else resolved_oracle_argv
    )
    cwd = resolved_cwd if resolved_cwd is not None else (spec.unit.cwd or None)
    kernel_filter = (
        spec.unit.kernel.name
        if spec.unit.kernel is not None and spec.unit.kernel.name
        else None
    )

    core_resume = _resume_core_result(
        spec,
        out_dir,
        attempt_dir,
        planned_trials,
        provenance,
        resume_from,
    )
    progress_adapter = _StressProgressAdapter(
        spec=spec,
        out_dir=out_dir,
        planned_trials=planned_trials,
        provenance=provenance,
        started=started,
        cutracer_version=cutracer_version,
        toolchain_version=toolchain_version,
        callback=progress_callback,
        latest=resume_from,
    )
    try:
        result = run_stress(
            StressConfig(
                oracle_argv=oracle_argv,
                delay_ladder_ns=list(spec.stress.delay_ladder_ns),
                enable_prob=spec.stress.enable_prob,
                warpgroup_ids=list(spec.stress.warpgroup_ids),
                attempts_per_delay=spec.stress.attempts_per_delay,
                stop_on_first=spec.stress.stop_on_first_reproduction,
                not_interesting_exit_codes=list(
                    spec.unit.oracle.not_interesting_exit_codes
                ),
                kernel_filters=kernel_filter,
                output_dir=attempt_dir,
                timeout=timeout,
                cwd=cwd,
                base_env=base_env,
                resume_result=core_resume,
                progress_callback=progress_adapter,
                use_bundled_cuda_tools=use_bundled_cuda_tools,
                expected_kernel_name=(
                    None if spec.unit.kernel is None else spec.unit.kernel.name
                ),
                expected_kernel_checksum=(
                    None if spec.unit.kernel is None else spec.unit.kernel.cubin_hash
                ),
            ),
            cutracer_so=cutracer_so,
            runner=runner,
        )
    except (ClickException, OSError, ValueError) as exc:
        # ``run_stress`` catches per-attempt ``TimeoutExpired`` internally and
        # reports it via ``StressResult.timed_out``; a timeout never propagates
        # here, so it is classified below rather than in this except clause.
        (
            completed_trials,
            reproductions,
            infra_errors,
            timed_out_trials,
            unattributed_reproductions,
            terminal_log,
            terminal_trigger,
        ) = _recover_stress_progress(
            out_dir,
            progress_adapter.latest_core,
            progress_adapter.latest,
        )
        observed = progress_adapter.latest_core
        rejected_configs = _merge_rejected_configs(
            (
                []
                if progress_adapter.latest is None
                else progress_adapter.latest.rejected_configs
            ),
            [] if observed is None else _promote_rejected_configs(observed, out_dir),
        )
        terminal = _terminal_stress_result(
            spec,
            provenance,
            planned_trials,
            started,
            ExecutionStatus.INFRA_ERROR,
            str(exc),
            outcome=_partial_stress_outcome(
                terminal_trigger,
                reproductions=reproductions,
                unattributed_reproductions=unattributed_reproductions,
            ),
            completed_trials=completed_trials,
            reproductions=reproductions,
            infra_errors=infra_errors,
            timed_out_trials=timed_out_trials,
            unattributed_reproductions=unattributed_reproductions,
            log=terminal_log,
            triggering_config=terminal_trigger,
            rejected_configs=rejected_configs,
        )
        if progress_callback is None:
            _cleanup_unreferenced_rejected_configs(out_dir, rejected_configs)
        return terminal

    log, triggering_config, artifact_errors = _publish_stress_artifacts(
        spec,
        result,
        out_dir,
        cutracer_version=cutracer_version,
        toolchain_version=toolchain_version,
    )
    rejected_configs = _merge_rejected_configs(
        (
            []
            if progress_adapter.latest is None
            else progress_adapter.latest.rejected_configs
        ),
        _promote_rejected_configs(result, out_dir),
    )
    if artifact_errors:
        # Preserve independently published evidence from this attempt, then
        # fall back to the last durable checkpoint for anything still missing.
        fallback_log, fallback_trigger = _fallback_stress_artifacts(
            out_dir,
            log,
            triggering_config,
            progress_adapter.latest,
        )
        preserved_outcome = _partial_stress_outcome(
            fallback_trigger,
            reproductions=result.reproductions,
            unattributed_reproductions=result.unattributed_reproductions,
        )
        terminal = _terminal_stress_result(
            spec,
            provenance,
            planned_trials,
            started,
            ExecutionStatus.INFRA_ERROR,
            "; ".join(artifact_errors),
            outcome=preserved_outcome,
            completed_trials=result.completed_trials,
            reproductions=result.reproductions,
            infra_errors=result.infra_errors + 1,
            timed_out_trials=result.timed_out,
            unattributed_reproductions=result.unattributed_reproductions,
            log=fallback_log,
            triggering_config=fallback_trigger,
            rejected_configs=rejected_configs,
        )
        if progress_callback is None:
            _cleanup_unreferenced_rejected_configs(out_dir, rejected_configs)
        return terminal

    outcome, status, error, completeness = _classify_stress_result(
        result,
        planned_trials=planned_trials,
        stop_on_first=spec.stress.stop_on_first_reproduction,
        reproduced=triggering_config is not None,
    )

    terminal_result = StressTestResult(
        experiment_id=spec.experiment_id,
        execution_status=status,
        outcome=outcome,
        completed_trials=result.completed_trials,
        reproductions=result.reproductions,
        infra_errors=result.infra_errors,
        provenance=provenance,
        triggering_config=triggering_config,
        log=log,
        duration_s=time.monotonic() - started,
        error=error,
        planned_trials=planned_trials,
        timed_out_trials=result.timed_out,
        completeness=completeness,
        unattributed_reproductions=result.unattributed_reproductions,
        rejected_configs=rejected_configs,
    )
    if progress_callback is None:
        _cleanup_unreferenced_rejected_configs(out_dir, rejected_configs)
    return terminal_result


def run_stress_experiment(
    spec: ExperimentSpec,
    *,
    out_dir: str,
    runner: Optional[RunTarget] = None,
    cutracer_so: Optional[str] = None,
    timeout: int = 1800,
    cutracer_version: str = "",
    toolchain_version: str = "",
    resolved_oracle_argv: Optional[Sequence[str]] = None,
    resolved_cwd: Optional[str] = None,
    resolved_base_env: Optional[Mapping[str, str]] = None,
    resume_from: Optional[StressTestResult] = None,
    progress_callback: Optional[StressProgressCallback] = None,
    use_bundled_cuda_tools: bool = True,
) -> StressTestResult:
    """Run CUTracer's in-process discovery loop and map its typed result.

    Resolved execution inputs may point into a materialized sandbox. They are
    deliberately not copied into the result: replay identity remains the
    portable argv/oracle/revision recorded in ``spec``.
    """
    if spec.kind != ExperimentKind.RANDOM_DELAY_STRESS or spec.stress is None:
        raise ValueError(f"not a stress experiment: {spec.kind}")
    if spec.unit.oracle is None:
        raise ValueError("random-delay stress requires a correctness oracle")

    os.makedirs(out_dir, exist_ok=True)
    attempt_dir = tempfile.mkdtemp(prefix=".stress_attempt_", dir=out_dir)
    terminal_result: Optional[StressTestResult] = None
    try:
        terminal_result = _run_stress_attempt(
            spec,
            out_dir=out_dir,
            attempt_dir=attempt_dir,
            runner=runner,
            cutracer_so=cutracer_so,
            timeout=timeout,
            cutracer_version=cutracer_version,
            toolchain_version=toolchain_version,
            resolved_oracle_argv=resolved_oracle_argv,
            resolved_cwd=resolved_cwd,
            resolved_base_env=resolved_base_env,
            resume_from=resume_from,
            progress_callback=progress_callback,
            use_bundled_cuda_tools=use_bundled_cuda_tools,
        )
        return terminal_result
    finally:
        shutil.rmtree(attempt_dir, ignore_errors=True)
