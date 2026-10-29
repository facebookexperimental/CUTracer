"""Delay-config replay and reduction experiment adapter."""

from __future__ import annotations

import json
import os
import shlex
import time
from typing import Mapping, Optional, Sequence

from cutracer.reduce.config_mutator import DelayConfigMutator
from cutracer.reduce.reduce import (
    ConfigDoesNotTriggerError,
    reduce_bisect,
    reduce_delay_points,
    ReduceConfig,
    ReduceResult,
    ReplayConfig,
    ReplayEvaluator,
    ReplayExecutionError,
    ReplayOutcome,
)
from cutracer.reduce.report import generate_report, save_report
from cutracer.runner import RunTarget
from cutracer.service.artifacts import (
    ArtifactIdentityError,
    ArtifactIntegrityError,
    ArtifactMaterializationError,
    ArtifactVerificationError,
    local_artifact,
    runtime_identity_is_missing,
    stage_verified_local_artifact,
)
from cutracer.service.contracts import (
    ArtifactRef,
    EssentialDelayPoint,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentKind,
    ExperimentSpec,
    OracleSpec,
    ReduceExperimentResult,
    ReduceOptions,
    ReduceOutcome,
)


class StaleReplayInputError(ValueError):
    """The replay input identity does not match the active runtime."""


class MissingActiveRuntimeIdentityError(ValueError):
    """The replay caller did not identify its active runtime."""


def _require_runtime_identity(
    identities: dict[str, str],
    *,
    error_type: type[ValueError],
    message: str,
) -> None:
    missing = [
        name for name, value in identities.items() if runtime_identity_is_missing(value)
    ]
    if missing:
        raise error_type(f"{message}: {', '.join(missing)}")


def _require_active_runtime_identity(
    cutracer_version: str,
    toolchain_version: str,
) -> None:
    _require_runtime_identity(
        {
            "active CUTracer": cutracer_version,
            "active toolchain": toolchain_version,
        },
        error_type=MissingActiveRuntimeIdentityError,
        message="reduction caller requires immutable runtime identity",
    )


def _require_trigger_runtime_identity(reduction: ReduceOptions) -> None:
    config = reduction.triggering_config
    _require_runtime_identity(
        {
            "trigger CUTracer": config.cutracer_version,
            "trigger toolchain": config.toolchain_version,
        },
        error_type=StaleReplayInputError,
        message="reduction requires immutable runtime identity",
    )


def _resolve_expected_kernel_identity(
    spec: ExperimentSpec,
    reduction: ReduceOptions,
) -> tuple[Optional[str], Optional[str]]:
    if not reduction.require_kernel_identity:
        return None, None
    kernel = spec.unit.kernel
    config_kernel = reduction.triggering_config.kernel
    if (
        kernel is None
        or not kernel.name.strip()
        or not kernel.cubin_hash.strip()
        or config_kernel is None
        or not config_kernel.name.strip()
        or not config_kernel.cubin_hash.strip()
        or config_kernel != kernel
    ):
        raise StaleReplayInputError(
            "reduction requires matching kernel name and cubin identity"
        )
    return kernel.name, kernel.cubin_hash


def _validate_runtime_identity(
    spec: ExperimentSpec,
    reduction: ReduceOptions,
    *,
    cutracer_version: str,
    toolchain_version: str,
) -> tuple[Optional[str], Optional[str]]:
    config = reduction.triggering_config
    _require_active_runtime_identity(cutracer_version, toolchain_version)
    _require_trigger_runtime_identity(reduction)
    if config.cutracer_version != cutracer_version:
        raise StaleReplayInputError(
            "triggering config CUTracer version does not match the active runtime"
        )
    if config.toolchain_version != toolchain_version:
        raise StaleReplayInputError(
            "triggering config toolchain version does not match the active runtime"
        )
    return _resolve_expected_kernel_identity(spec, reduction)


def _validate_artifact_kernel_identity(
    spec: ExperimentSpec,
    reduction: ReduceOptions,
    path: str,
) -> None:
    if not reduction.require_kernel_identity:
        return
    kernel = spec.unit.kernel
    if kernel is None:
        raise StaleReplayInputError("reduction requires pinned kernel identity")
    try:
        with open(path) as fh:
            document = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise StaleReplayInputError("verified delay config is not valid JSON") from exc
    if not isinstance(document, dict) or not isinstance(document.get("kernels"), dict):
        raise StaleReplayInputError("verified delay config has no kernel metadata")
    kernels = document["kernels"]
    if not any(
        isinstance(entry, dict) and entry.get("kernel_checksum") == kernel.cubin_hash
        for entry in kernels.values()
    ):
        raise StaleReplayInputError(
            "verified delay config does not contain the pinned kernel identity "
            "checksum "
            f"{kernel.cubin_hash!r}; config names may be mangled while the "
            f"WorkUnit name is {kernel.name!r}"
        )


def _build_reduce_config(
    spec: ExperimentSpec,
    reduction: ReduceOptions,
    oracle: OracleSpec,
    *,
    input_path: str,
    minimal_path: str,
    out_dir: str,
    runner: Optional[RunTarget],
    cutracer_so: Optional[str],
    timeout: int,
    expected_kernel_name: Optional[str],
    expected_kernel_checksum: Optional[str],
    resolved_oracle_argv: Optional[Sequence[str]],
    resolved_cwd: Optional[str],
    resolved_base_env: Optional[Mapping[str, str]],
    replay_evaluator: Optional[ReplayEvaluator],
) -> ReduceConfig:
    if resolved_base_env is None:
        base_env = os.environ.copy()
        base_env.update(spec.unit.env)
    else:
        base_env = dict(resolved_base_env)
    # Reduction must replay the one checksum-pinned kernel from the triggering
    # config; broad catalog filters are insufficient evidence for this stage.
    kernel = spec.unit.kernel
    return ReduceConfig(
        config_path=input_path,
        output_path=minimal_path,
        replay=ReplayConfig(
            oracle_argv=list(
                oracle.argv if resolved_oracle_argv is None else resolved_oracle_argv
            ),
            cutracer_so=cutracer_so,
            exact_kernel_name=(
                None if kernel is None else kernel.mangled_name or kernel.name
            ),
            exact_kernel_checksum=None if kernel is None else kernel.cubin_hash,
            delay_warpgroup_id=reduction.triggering_config.warpgroup_id,
            output_dir=out_dir,
            timeout=timeout,
            cwd=(resolved_cwd if resolved_cwd is not None else (spec.unit.cwd or None)),
            base_env=base_env,
            not_interesting_exit_codes=tuple(oracle.not_interesting_exit_codes),
            expected_kernel_name=expected_kernel_name,
            expected_kernel_checksum=expected_kernel_checksum,
        ),
        replay_runner=runner,
        replay_evaluator=replay_evaluator,
    )


def _validate_reduction_input(path: str) -> None:
    try:
        mutator = DelayConfigMutator(
            path,
            validate=True,
        )
    except ValueError as exc:
        raise StaleReplayInputError(str(exc)) from exc
    if not mutator.enabled_points:
        raise StaleReplayInputError("No enabled delay points in config")


def validate_reduce_input(
    spec: ExperimentSpec,
    path: str,
    *,
    cutracer_version: str,
    toolchain_version: str,
) -> tuple[Optional[str], Optional[str]]:
    """Validate staged replay bytes and their complete execution identity."""
    reduction = spec.reduction
    if spec.kind != ExperimentKind.REDUCE_DELAY_CONFIG or reduction is None:
        raise ValueError("reduce input validation requires a reduce experiment")
    expected = _validate_runtime_identity(
        spec,
        reduction,
        cutracer_version=cutracer_version,
        toolchain_version=toolchain_version,
    )
    _validate_artifact_kernel_identity(spec, reduction, path)
    _validate_reduction_input(path)
    return expected


def _run_reduction(reduction: ReduceOptions, config: ReduceConfig) -> ReduceResult:
    if reduction.strategy == "linear":
        return reduce_delay_points(
            config,
            confidence_runs=reduction.confidence_runs,
        )
    if reduction.strategy == "bisect":
        return reduce_bisect(
            config,
            confidence_runs=reduction.confidence_runs,
        )
    raise ValueError(f"unknown reduction strategy: {reduction.strategy}")


def _publish_reduction_result(
    result: ReduceResult,
    *,
    input_path: str,
    report_path: str,
    oracle_argv: Sequence[str],
) -> tuple[ArtifactRef, Optional[ArtifactRef], list[EssentialDelayPoint], str]:
    report_data = generate_report(
        result=result,
        config_path=input_path,
        test_script=shlex.join(oracle_argv),
    )
    save_report(report_data, report_path)
    report = local_artifact(
        report_path, media_type="application/json", include_sha256=True
    )
    minimized = None
    if result.minimal_config_path is not None:
        minimized = local_artifact(
            result.minimal_config_path,
            media_type="application/json",
            include_sha256=True,
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
    note = ""
    if not points:
        note = (
            "0 essential points -- reproduces without any injected delay "
            "(deterministic, not a timing-dependent race)"
        )
    return report, minimized, points, note


def _execute_reduction(
    spec: ExperimentSpec,
    reduction: ReduceOptions,
    oracle: OracleSpec,
    *,
    out_dir: str,
    runner: Optional[RunTarget],
    cutracer_so: Optional[str],
    timeout: int,
    cutracer_version: str,
    toolchain_version: str,
    resolved_oracle_argv: Optional[Sequence[str]],
    resolved_cwd: Optional[str],
    resolved_base_env: Optional[Mapping[str, str]],
    replay_evaluator: Optional[ReplayEvaluator],
) -> tuple[ArtifactRef, Optional[ArtifactRef], list[EssentialDelayPoint], str]:
    os.makedirs(out_dir, exist_ok=True)
    effective_oracle_argv = (
        oracle.argv if resolved_oracle_argv is None else resolved_oracle_argv
    )
    verified = stage_verified_local_artifact(
        reduction.triggering_config.artifact,
        destination=os.path.join(out_dir, "verified_triggering_config.json"),
    )
    expected_name, expected_checksum = validate_reduce_input(
        spec,
        verified.path,
        cutracer_version=cutracer_version,
        toolchain_version=toolchain_version,
    )
    config = _build_reduce_config(
        spec,
        reduction,
        oracle,
        input_path=verified.path,
        minimal_path=os.path.join(out_dir, "minimal_config.json"),
        out_dir=out_dir,
        runner=runner,
        cutracer_so=cutracer_so,
        timeout=timeout,
        expected_kernel_name=expected_name,
        expected_kernel_checksum=expected_checksum,
        resolved_oracle_argv=resolved_oracle_argv,
        resolved_cwd=resolved_cwd,
        resolved_base_env=resolved_base_env,
        replay_evaluator=replay_evaluator,
    )
    result = _run_reduction(reduction, config)
    return _publish_reduction_result(
        result,
        input_path=verified.path,
        report_path=os.path.join(out_dir, "reduce_report.json"),
        oracle_argv=effective_oracle_argv,
    )


def classify_reduction_failure(
    exc: Exception,
) -> tuple[ReduceOutcome, ExecutionStatus, str]:
    if isinstance(
        exc,
        (ArtifactIdentityError, ArtifactIntegrityError, StaleReplayInputError),
    ):
        return ReduceOutcome.STALE_INPUT, ExecutionStatus.FAILED, str(exc)
    if isinstance(exc, ArtifactMaterializationError):
        return ReduceOutcome.INFRA_ERROR, ExecutionStatus.INFRA_ERROR, str(exc)
    if isinstance(exc, MissingActiveRuntimeIdentityError):
        return ReduceOutcome.INFRA_ERROR, ExecutionStatus.INFRA_ERROR, str(exc)
    if isinstance(exc, ReplayExecutionError):
        if exc.outcome == ReplayOutcome.STALE_INPUT:
            return ReduceOutcome.STALE_INPUT, ExecutionStatus.FAILED, str(exc)
        status = (
            ExecutionStatus.TIMED_OUT
            if exc.outcome == ReplayOutcome.TIMED_OUT
            else ExecutionStatus.INFRA_ERROR
        )
        return ReduceOutcome.INFRA_ERROR, status, str(exc)
    if isinstance(exc, ConfigDoesNotTriggerError):
        return (
            ReduceOutcome.NON_REPRODUCIBLE,
            ExecutionStatus.FAILED,
            "triggering config no longer reproduces under the pinned oracle",
        )
    return ReduceOutcome.INFRA_ERROR, ExecutionStatus.INFRA_ERROR, str(exc)


def run_reduce_experiment(
    spec: ExperimentSpec,
    *,
    out_dir: str,
    cutracer_version: str,
    toolchain_version: str,
    runner: Optional[RunTarget] = None,
    cutracer_so: Optional[str] = None,
    timeout: int = 1800,
    resolved_oracle_argv: Optional[Sequence[str]] = None,
    resolved_cwd: Optional[str] = None,
    resolved_base_env: Optional[Mapping[str, str]] = None,
    replay_evaluator: Optional[ReplayEvaluator] = None,
) -> ReduceExperimentResult:
    """Reduce a required triggering config; never discover one implicitly."""
    if spec.kind != ExperimentKind.REDUCE_DELAY_CONFIG:
        raise ValueError(f"not a reduce experiment: {spec.kind}")
    if spec.reduction is None:
        raise ValueError("reduce requires an existing triggering delay config")
    if spec.unit.oracle is None:
        raise ValueError("reduce requires the same pre-approved correctness oracle")
    reduction = spec.reduction
    oracle = spec.unit.oracle

    started = time.monotonic()
    report: Optional[ArtifactRef] = None
    minimized: Optional[ArtifactRef] = None
    points: list[EssentialDelayPoint] = []
    note = ""

    try:
        report, minimized, points, note = _execute_reduction(
            spec,
            reduction,
            oracle,
            out_dir=out_dir,
            runner=runner,
            cutracer_so=cutracer_so,
            timeout=timeout,
            cutracer_version=cutracer_version,
            toolchain_version=toolchain_version,
            resolved_oracle_argv=resolved_oracle_argv,
            resolved_cwd=resolved_cwd,
            resolved_base_env=resolved_base_env,
            replay_evaluator=replay_evaluator,
        )
        outcome = ReduceOutcome.REDUCED
        status = ExecutionStatus.SUCCEEDED
        error = None
    except (
        ArtifactVerificationError,
        MissingActiveRuntimeIdentityError,
        StaleReplayInputError,
        ReplayExecutionError,
        ConfigDoesNotTriggerError,
        OSError,
        ValueError,
    ) as exc:
        outcome, status, error = classify_reduction_failure(exc)

    return ReduceExperimentResult(
        experiment_id=spec.experiment_id,
        execution_status=status,
        outcome=outcome,
        provenance=ExecutionProvenance(
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
            tool_version=cutracer_version,
            toolchain_version=toolchain_version,
        ),
        input_config=reduction.triggering_config,
        minimized_config=minimized,
        report=report,
        essential_points=points,
        duration_s=time.monotonic() - started,
        error=error,
        note=note,
    )
