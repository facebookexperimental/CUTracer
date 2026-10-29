"""Initial campaign planning and guarded AI follow-up expansion."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, List, Optional, Sequence

from cutracer.service.contracts import (
    AnalysisSession,
    ArtifactRef,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentKind,
    ExperimentPhase,
    ExperimentRequest,
    ExperimentSpec,
    ReduceOptions,
    ResultCompleteness,
    StressOptions,
    StressOutcome,
    StressTestResult,
    TraceConfigSource,
    TraceOptions,
    TraceSelector,
    TriggeringDelayConfig,
    WorkUnit,
)

_MAX_TRACE_PC_OFFSETS = 32

DEFAULT_LOCAL_SANITIZER_TOOLS = (
    "memcheck",
    "synccheck",
    "initcheck",
    "racecheck",
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


def _request_discriminator(
    request: ExperimentRequest,
    config: Optional[TriggeringDelayConfig],
) -> str:
    if request.kind == ExperimentKind.COMPUTE_SANITIZER:
        return "racecheck"
    if request.kind == ExperimentKind.REDUCE_DELAY_CONFIG and config is not None:
        return config.artifact.uri
    return ""


def experiment_spec_fingerprint(spec: ExperimentSpec) -> str:
    """Fingerprint a planned spec with the identical key the guard dedups on.

    Mirrors the per-kind discriminator that ``ExperimentGuardPolicy.expand``
    derives from a request, so a later follow-up that repeats an already-planned
    experiment collides with this key and is rejected.
    """
    discriminator = ""
    if spec.kind == ExperimentKind.COMPUTE_SANITIZER:
        discriminator = spec.sanitizer_tool or ""
    elif (
        spec.kind in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE)
        and spec.trace is not None
    ):
        discriminator = _trace_options_discriminator(spec.trace, spec.unit)
    elif spec.kind == ExperimentKind.REDUCE_DELAY_CONFIG and spec.reduction is not None:
        discriminator = spec.reduction.triggering_config.artifact.uri
    return _fingerprint(spec.kind, discriminator)


def _trace_options_discriminator(options: TraceOptions, unit: WorkUnit) -> str:
    selector = options.selector.to_dict()
    if options.selector.kernel == unit.kernel:
        selector["kernel"] = None
    artifact = options.config_artifact
    artifact_identity = None
    if artifact is not None:
        artifact_identity = (
            {"sha256": artifact.sha256, "size_bytes": artifact.size_bytes}
            if artifact.sha256 is not None and artifact.size_bytes is not None
            else {"uri": artifact.uri}
        )
    if (
        all(value is None or value == [] for value in selector.values())
        and artifact_identity is None
        and not options.instrument_modes
    ):
        return ""
    payload: dict[str, object] = {
        "selector": selector,
        "config_artifact": artifact_identity,
    }
    if options.instrument_modes:
        payload["instrument_modes"] = options.effective_instrument_modes
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selected_config_artifact(
    session: AnalysisSession,
    source: TraceConfigSource,
) -> ArtifactRef | None:
    if source == TraceConfigSource.TRIGGERING:
        config = session.evidence.triggering_config()
        return None if config is None else config.artifact
    for result in reversed(session.evidence.reductions):
        if (
            result.execution_status == ExecutionStatus.SUCCEEDED
            and result.minimized_config is not None
        ):
            return result.minimized_config
    return None


def _selected_config_point_count(
    session: AnalysisSession,
    source: TraceConfigSource,
) -> int | None:
    if source == TraceConfigSource.TRIGGERING:
        config = session.evidence.triggering_config()
        return None if config is None else config.enabled_point_count
    for result in reversed(session.evidence.reductions):
        if (
            result.execution_status == ExecutionStatus.SUCCEEDED
            and result.minimized_config is not None
        ):
            return len({point.pc_offset for point in result.essential_points})
    return None


def _resolve_trace_options(
    session: AnalysisSession,
    request: ExperimentRequest,
    trace_size_limit_mb: int,
) -> tuple[TraceOptions, str | None]:
    kernel = session.unit.kernel
    if kernel is None or not kernel.name.strip() or not kernel.cubin_hash.strip():
        raise ValueError(
            "targeted trace requires pinned kernel name and cubin identity"
        )
    selector = request.trace_selector or TraceSelector()
    if selector.kernel is not None and selector.kernel != session.unit.kernel:
        raise ValueError("trace selector kernel does not match the pinned WorkUnit")
    note = None
    if selector.pc_offsets and selector.config_source is not None:
        note = (
            "trace selector supplied both explicit PC offsets and a config source; "
            "explicit PC offsets take precedence"
        )
        selector = dataclasses.replace(selector, config_source=None)
    artifact = None
    if selector.config_source is not None:
        artifact = _selected_config_artifact(session, selector.config_source)
        if artifact is None:
            raise ValueError(
                f"trace selector requires available {selector.config_source.value} config"
            )
        point_count = _selected_config_point_count(session, selector.config_source)
        if point_count is None:
            raise ValueError(
                f"trace from {selector.config_source.value} config is deferred: "
                "enabled PC count is unknown"
            )
        if point_count == 0:
            raise ValueError(
                f"trace from {selector.config_source.value} config has no enabled PCs"
            )
        if point_count > _MAX_TRACE_PC_OFFSETS:
            raise ValueError(
                f"trace from {selector.config_source.value} config is deferred: "
                f"{point_count} enabled PCs exceed the {_MAX_TRACE_PC_OFFSETS}-PC limit"
            )
    return (
        TraceOptions(
            mode=request.kind,
            trace_size_limit_mb=trace_size_limit_mb,
            selector=selector,
            config_artifact=artifact,
            instrument_modes=request.instrument_modes,
        ),
        note,
    )


@dataclasses.dataclass(frozen=True)
class CampaignPlan:
    experiments: tuple[ExperimentSpec, ...]
    skipped_results: tuple[StressTestResult, ...] = ()


@dataclasses.dataclass(frozen=True)
class InitialCampaignPolicy:
    """Plan peer initial sources without consulting either source's result."""

    sanitizer_tools: tuple[str, ...] = ("racecheck",)
    stress_options: StressOptions = dataclasses.field(default_factory=StressOptions)
    enable_stress: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enable_stress, bool):
            raise ValueError("enable_stress must be boolean")
        if not isinstance(self.sanitizer_tools, tuple) or any(
            not isinstance(tool, str) for tool in self.sanitizer_tools
        ):
            raise ValueError("sanitizer_tools must be a tuple of strings")
        unknown = sorted(set(self.sanitizer_tools) - set(DEFAULT_LOCAL_SANITIZER_TOOLS))
        if unknown:
            raise ValueError(f"unsupported sanitizer tools: {', '.join(unknown)}")
        if len(self.sanitizer_tools) != len(set(self.sanitizer_tools)):
            raise ValueError("sanitizer tools must not contain duplicates")
        if not self.sanitizer_tools and not self.enable_stress:
            raise ValueError(
                "initial campaign must enable at least one evidence source"
            )

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
        if self.enable_stress and unit.oracle is not None:
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
        elif self.enable_stress:
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
                    planned_trials=0,
                    completeness=ResultCompleteness.SKIPPED,
                ),
            )
        else:
            skipped = ()
        return CampaignPlan(experiments=tuple(specs), skipped_results=skipped)


@dataclasses.dataclass(frozen=True)
class GuardedExperiments:
    experiments: tuple[ExperimentSpec, ...]
    fingerprints: tuple[str, ...]
    rejections: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ExperimentGuardPolicy:
    """Convert model requests into allowlisted, reproducible experiment specs."""

    default_trace_size_limit_mb: int = 1024
    auto_reduce_triggering_config: bool = True
    require_kernel_identity_for_reduce: bool = True

    def _reduce_identity_rejection(
        self,
        unit: WorkUnit,
        config: TriggeringDelayConfig,
    ) -> Optional[str]:
        if not self.require_kernel_identity_for_reduce:
            return None
        kernel = unit.kernel
        config_kernel = config.kernel
        if (
            kernel is None
            or not kernel.name.strip()
            or not kernel.cubin_hash.strip()
            or config_kernel is None
            or not config_kernel.name.strip()
            or not config_kernel.cubin_hash.strip()
        ):
            return "reduce requires kernel name and cubin identity"
        if config_kernel != kernel:
            return "reduce triggering config kernel identity differs from work unit"
        return None

    def _build_reduce_spec(
        self,
        session: AnalysisSession,
        config: Optional[TriggeringDelayConfig],
        common: dict[str, Any],
    ) -> tuple[Optional[ExperimentSpec], Optional[str]]:
        if config is None or session.unit.oracle is None:
            return None, "reduce requires a reproduced config and its approved oracle"
        rejection = self._reduce_identity_rejection(session.unit, config)
        if rejection is not None:
            return None, rejection
        return (
            ExperimentSpec(
                **common,
                reduction=ReduceOptions(
                    triggering_config=config,
                    require_kernel_identity=self.require_kernel_identity_for_reduce,
                ),
            ),
            None,
        )

    def _build_spec(
        self,
        session: AnalysisSession,
        request: ExperimentRequest,
        config: Optional[TriggeringDelayConfig],
        *,
        index: int,
        discriminator: str,
        trace_options: Optional[TraceOptions] = None,
    ) -> tuple[Optional[ExperimentSpec], Optional[str]]:
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
            return ExperimentSpec(**common, sanitizer_tool="racecheck"), None
        if request.kind == ExperimentKind.RANDOM_DELAY_STRESS:
            if session.unit.oracle is None:
                return None, "random-delay stress requires an approved oracle"
            return ExperimentSpec(**common, stress=StressOptions()), None
        if request.kind in (
            ExperimentKind.REG_TRACE,
            ExperimentKind.MEM_VALUE_TRACE,
        ):
            if trace_options is None:
                return None, "trace options could not be resolved"
            return (
                ExperimentSpec(
                    **common,
                    trace=trace_options,
                ),
                None,
            )
        if request.kind == ExperimentKind.REDUCE_DELAY_CONFIG:
            return self._build_reduce_spec(session, config, common)
        return None, f"unsupported experiment kind: {request.kind.value}"

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
        notes: List[str] = []
        for index, request in enumerate(requested):
            trace_options = None
            if request.kind in (
                ExperimentKind.REG_TRACE,
                ExperimentKind.MEM_VALUE_TRACE,
            ):
                try:
                    trace_options, note = _resolve_trace_options(
                        session,
                        request,
                        self.default_trace_size_limit_mb,
                    )
                    if note is not None:
                        notes.append(note)
                except ValueError as exc:
                    rejections.append(str(exc))
                    continue
                discriminator = _trace_options_discriminator(
                    trace_options,
                    session.unit,
                )
            else:
                discriminator = _request_discriminator(request, config)
            fingerprint = _fingerprint(request.kind, discriminator)
            if (
                fingerprint in session.experiment_fingerprints
                or fingerprint in fingerprints
            ):
                rejections.append(f"duplicate experiment: {fingerprint}")
                continue

            spec, rejection = self._build_spec(
                session,
                request,
                config,
                index=index,
                discriminator=discriminator,
                trace_options=trace_options,
            )
            if spec is None:
                if rejection is not None:
                    rejections.append(rejection)
                continue
            specs.append(spec)
            fingerprints.append(fingerprint)

        return GuardedExperiments(
            experiments=tuple(specs),
            fingerprints=tuple(fingerprints),
            rejections=tuple(rejections),
            notes=tuple(notes),
        )
