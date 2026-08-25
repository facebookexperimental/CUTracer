"""Initial campaign planning and guarded AI follow-up expansion."""

from __future__ import annotations

import dataclasses
import hashlib
from typing import List, Sequence

from cutracer.service.contracts import (
    AnalysisSession,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentKind,
    ExperimentPhase,
    ExperimentRequest,
    ExperimentSpec,
    ReduceOptions,
    StressOptions,
    StressOutcome,
    StressTestResult,
    TraceOptions,
    WorkUnit,
)


def _experiment_id(
    session_id: str,
    phase: ExperimentPhase,
    round_index: int,
    kind: ExperimentKind,
    discriminator: str = "",
) -> str:
    raw = f"{session_id}:{phase.value}:{round_index}:{kind.value}:{discriminator}"
    return f"exp-{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def _fingerprint(kind: ExperimentKind, discriminator: str = "") -> str:
    return f"{kind.value}:{discriminator}"


def experiment_spec_fingerprint(spec: ExperimentSpec) -> str:
    """Fingerprint a planned spec with the identical key the guard dedups on.

    Mirrors the per-kind discriminator that ``ExperimentGuardPolicy.expand``
    derives from a request, so a later follow-up that repeats an already-planned
    experiment collides with this key and is rejected.
    """
    discriminator = ""
    if spec.kind == ExperimentKind.COMPUTE_SANITIZER:
        discriminator = spec.sanitizer_tool or ""
    elif spec.kind == ExperimentKind.REDUCE_DELAY_CONFIG and spec.reduction is not None:
        discriminator = spec.reduction.triggering_config.artifact.uri
    return _fingerprint(spec.kind, discriminator)


@dataclasses.dataclass(frozen=True)
class CampaignPlan:
    experiments: tuple[ExperimentSpec, ...]
    skipped_results: tuple[StressTestResult, ...] = ()


@dataclasses.dataclass(frozen=True)
class InitialCampaignPolicy:
    """Plan peer initial sources without consulting either source's result."""

    sanitizer_tools: tuple[str, ...] = ("racecheck",)
    stress_options: StressOptions = dataclasses.field(default_factory=StressOptions)

    def plan(self, session_id: str, unit: WorkUnit) -> CampaignPlan:
        specs: List[ExperimentSpec] = []
        for tool in self.sanitizer_tools:
            specs.append(
                ExperimentSpec(
                    experiment_id=_experiment_id(
                        session_id,
                        ExperimentPhase.INITIAL,
                        0,
                        ExperimentKind.COMPUTE_SANITIZER,
                        tool,
                    ),
                    session_id=session_id,
                    phase=ExperimentPhase.INITIAL,
                    kind=ExperimentKind.COMPUTE_SANITIZER,
                    unit=unit,
                    sanitizer_tool=tool,
                )
            )

        stress_id = _experiment_id(
            session_id,
            ExperimentPhase.INITIAL,
            0,
            ExperimentKind.RANDOM_DELAY_STRESS,
        )
        if unit.oracle is not None:
            specs.append(
                ExperimentSpec(
                    experiment_id=stress_id,
                    session_id=session_id,
                    phase=ExperimentPhase.INITIAL,
                    kind=ExperimentKind.RANDOM_DELAY_STRESS,
                    unit=unit,
                    stress=self.stress_options,
                )
            )
            skipped = ()
        else:
            skipped = (
                StressTestResult(
                    experiment_id=stress_id,
                    execution_status=ExecutionStatus.SUCCEEDED,
                    outcome=StressOutcome.SKIPPED,
                    completed_trials=0,
                    reproductions=0,
                    infra_errors=0,
                    provenance=ExecutionProvenance(
                        source_revision=unit.source_revision,
                        arch=unit.arch,
                        kernel=unit.kernel,
                    ),
                    error="work unit has no pre-approved correctness oracle",
                ),
            )
        return CampaignPlan(experiments=tuple(specs), skipped_results=skipped)


@dataclasses.dataclass(frozen=True)
class GuardedExperiments:
    experiments: tuple[ExperimentSpec, ...]
    fingerprints: tuple[str, ...]
    rejections: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ExperimentGuardPolicy:
    """Convert model requests into allowlisted, reproducible experiment specs."""

    default_trace_size_limit_mb: int = 1024
    auto_reduce_triggering_config: bool = True

    def expand(
        self,
        session: AnalysisSession,
        requests: Sequence[ExperimentRequest],
    ) -> GuardedExperiments:
        requested = list(requests)
        config = session.evidence.triggering_config()
        if (
            self.auto_reduce_triggering_config
            and config is not None
            and not any(
                request.kind == ExperimentKind.REDUCE_DELAY_CONFIG
                for request in requested
            )
        ):
            requested.append(
                ExperimentRequest(
                    kind=ExperimentKind.REDUCE_DELAY_CONFIG,
                    rationale="standard reduction of the reproduced delay config",
                )
            )

        specs: List[ExperimentSpec] = []
        fingerprints: List[str] = []
        rejections: List[str] = []
        for index, request in enumerate(requested):
            discriminator = ""
            if request.kind == ExperimentKind.COMPUTE_SANITIZER:
                discriminator = "racecheck"
            elif request.kind == ExperimentKind.REDUCE_DELAY_CONFIG:
                discriminator = "" if config is None else config.artifact.uri
            fingerprint = _fingerprint(request.kind, discriminator)
            if (
                fingerprint in session.experiment_fingerprints
                or fingerprint in fingerprints
            ):
                rejections.append(f"duplicate experiment: {fingerprint}")
                continue

            common = {
                "experiment_id": _experiment_id(
                    session.session_id,
                    ExperimentPhase.FOLLOWUP,
                    session.round_index + 1,
                    request.kind,
                    f"{index}:{discriminator}",
                ),
                "session_id": session.session_id,
                "phase": ExperimentPhase.FOLLOWUP,
                "kind": request.kind,
                "unit": session.unit,
                "round_index": session.round_index + 1,
            }
            if request.kind == ExperimentKind.COMPUTE_SANITIZER:
                spec = ExperimentSpec(**common, sanitizer_tool="racecheck")
            elif request.kind == ExperimentKind.RANDOM_DELAY_STRESS:
                if session.unit.oracle is None:
                    rejections.append("random-delay stress requires an approved oracle")
                    continue
                spec = ExperimentSpec(**common, stress=StressOptions())
            elif request.kind in (
                ExperimentKind.REG_TRACE,
                ExperimentKind.MEM_VALUE_TRACE,
            ):
                spec = ExperimentSpec(
                    **common,
                    trace=TraceOptions(
                        mode=request.kind,
                        trace_size_limit_mb=self.default_trace_size_limit_mb,
                    ),
                )
            elif request.kind == ExperimentKind.REDUCE_DELAY_CONFIG:
                if config is not None and session.unit.oracle is not None:
                    spec = ExperimentSpec(
                        **common,
                        reduction=ReduceOptions(triggering_config=config),
                    )
                else:
                    rejections.append(
                        "reduce requires a reproduced config and its approved oracle"
                    )
                    continue
            else:
                rejections.append(f"unsupported experiment kind: {request.kind.value}")
                continue
            specs.append(spec)
            fingerprints.append(fingerprint)

        return GuardedExperiments(
            experiments=tuple(specs),
            fingerprints=tuple(fingerprints),
            rejections=tuple(rejections),
        )
