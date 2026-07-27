# pyre-strict
"""Wire contracts for the CUTracer automated diagnosis service.

This module is the service-layer handoff schema. It is pure stdlib and has zero
dependency on CUTracer. It contains the session-oriented experiment/evidence
contracts plus the original single-trace explain contracts retained for
compatibility.

Design notes:
  * Enums subclass ``(str, Enum)`` so JSON is stable human-readable text.
  * ``to_dict`` recurses into nested dataclasses, lowers enums to ``.value``,
    and preserves ``None`` optionals as ``null`` (keys are kept).
  * ``from_dict`` is FORWARD-COMPATIBLE: it reads only known fields and silently
    drops unknown keys, so a newer producer never breaks an older consumer.
  * ``race_class`` is a free-form ``str`` (not an enum) because the taxonomy is
    not yet frozen; keeping it a plain string avoids churn.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any, Dict, List, Optional, TextIO, Type, TypeVar, Union

# ---------------------------------------------------------------------------
# Enums (str-valued for stable JSON text)
# ---------------------------------------------------------------------------


class Confidence(str, Enum):
    """AI confidence in the explanation."""

    H = "H"
    M = "M"
    L = "L"


class EvidenceSufficiency(str, Enum):
    """Whether the evidence was sufficient for the AI to reason.

    ``DEGRADED`` is defined now (used from D2 onward) for the claude-absent
    Phase-1 degradation path, so D2 needs no contract change.
    """

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    DEGRADED = "degraded"


class ConfirmStatus(str, Enum):
    """Verdict status from the best-effort confirm stage.

    ``SKIPPED`` is the D3 stub value; ``CONFIRMED`` / ``UNCONFIRMED`` are the
    real D4 outcomes (an unconfirmed finding still proceeds to explain).
    """

    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    SKIPPED = "skipped"


class ExecutionStatus(str, Enum):
    """Whether the wrapped check completed well enough to interpret its log."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INFRA_ERROR = "infra_error"


class SanitizerOutcome(str, Enum):
    """Sanitizer semantic outcome, kept separate from process execution status."""

    CLEAN = "clean"
    FINDING = "finding"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _lower(value: Any) -> Any:
    """Recursively lower a value into JSON-native form."""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Every dataclass in this module implements to_dict().
        return value.to_dict()  # type: ignore[attr-defined]
    if isinstance(value, list):
        return [_lower(v) for v in value]
    if isinstance(value, tuple):
        return [_lower(v) for v in value]
    if isinstance(value, dict):
        return {k: _lower(v) for k, v in value.items()}
    return value


def _known(cls: Type[Any], d: Dict[str, Any]) -> Dict[str, Any]:
    """Filter ``d`` to the dataclass field names of ``cls`` (drops unknown keys)."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def _load_source(source: Union[str, bytes, TextIO]) -> str:
    """Accept a JSON string/bytes or an open file object and return the text."""
    if hasattr(source, "read"):
        source = source.read()  # type: ignore[union-attr]
    if isinstance(source, bytes):
        return source.decode("utf-8")
    return source  # type: ignore[return-value]


def _as_bool(value: Any) -> bool:
    """Parse a bool that may arrive as a JSON bool or a string.

    ``bool("false")`` is ``True`` in Python, so a string-valued producer would
    otherwise silently deserialize to the wrong state. Accepts real bools and the
    common string spellings; everything else falls back to ``bool()``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


# ---------------------------------------------------------------------------
# Leaf dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KernelId:
    name: str
    cubin_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "cubin_hash": self.cubin_hash}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelId":
        d = _known(cls, d)
        return cls(name=d["name"], cubin_hash=d["cubin_hash"])


@dataclasses.dataclass
class SourceLoc:
    file: str
    line: int

    def to_dict(self) -> Dict[str, Any]:
        return {"file": self.file, "line": self.line}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceLoc":
        d = _known(cls, d)
        return cls(file=d["file"], line=int(d["line"]))


@dataclasses.dataclass
class ConfirmVerdict:
    status: ConfirmStatus
    rate: float
    delay_ns: Optional[int]
    warpgroup: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "rate": self.rate,
            "delay_ns": self.delay_ns,
            "warpgroup": self.warpgroup,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfirmVerdict":
        d = _known(cls, d)
        delay_ns = d.get("delay_ns")
        warpgroup = d.get("warpgroup")
        return cls(
            status=ConfirmStatus(d["status"]),
            rate=float(d["rate"]),
            delay_ns=None if delay_ns is None else int(delay_ns),
            warpgroup=None if warpgroup is None else int(warpgroup),
        )


@dataclasses.dataclass
class CrossValidation:
    agrees: bool
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {"agrees": self.agrees, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CrossValidation":
        d = _known(cls, d)
        return cls(agrees=_as_bool(d["agrees"]), notes=d["notes"])


# ---------------------------------------------------------------------------
# Artifact references
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ArtifactRef:
    """A portable reference to an artifact produced by MAST or a local run."""

    uri: str
    relative_path: str = ""
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    media_type: str = "text/plain"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactRef":
        d = _known(cls, d)
        size = d.get("size_bytes")
        return cls(
            uri=d["uri"],
            relative_path=d.get("relative_path", ""),
            sha256=d.get("sha256"),
            size_bytes=None if size is None else int(size),
            media_type=d.get("media_type", "text/plain"),
        )


# ---------------------------------------------------------------------------
# Top-level contracts
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ExplainInput:
    """Compatibility input for the single-trace analysis adapter in explain.py."""

    unit_id: str
    kernel: KernelId
    arch: str
    tool: str
    error_type: str
    source: SourceLoc
    sani_log_path: Optional[str]
    repro_cmd: str
    trace_path: Optional[str] = None
    confirm_verdict: Optional[ConfirmVerdict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kernel": self.kernel.to_dict(),
            "arch": self.arch,
            "tool": self.tool,
            "error_type": self.error_type,
            "source": self.source.to_dict(),
            "sani_log_path": self.sani_log_path,
            "repro_cmd": self.repro_cmd,
            "trace_path": self.trace_path,
            "confirm_verdict": (
                None if self.confirm_verdict is None else self.confirm_verdict.to_dict()
            ),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExplainInput":
        d = _known(cls, d)
        cv = d.get("confirm_verdict")
        return cls(
            unit_id=d["unit_id"],
            kernel=KernelId.from_dict(d["kernel"]),
            arch=d["arch"],
            tool=d["tool"],
            error_type=d["error_type"],
            source=SourceLoc.from_dict(d["source"]),
            sani_log_path=d.get("sani_log_path"),
            repro_cmd=d["repro_cmd"],
            trace_path=d.get("trace_path"),
            confirm_verdict=None if cv is None else ConfirmVerdict.from_dict(cv),
        )

    @classmethod
    def from_json(cls, source: Union[str, bytes, TextIO]) -> "ExplainInput":
        return cls.from_dict(json.loads(_load_source(source)))


@dataclasses.dataclass
class ExplainReport:
    """explain output; the Local and Sandcastle runners produce the same shape."""

    root_cause: str
    race_class: str
    confidence: Confidence
    is_reproduced: bool
    cross_validation: CrossValidation
    evidence_sufficiency: EvidenceSufficiency
    evidence_refs: List[str]
    recommended_action: str
    raw_ai_output: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "race_class": self.race_class,
            "confidence": self.confidence.value,
            "is_reproduced": self.is_reproduced,
            "cross_validation": self.cross_validation.to_dict(),
            "evidence_sufficiency": self.evidence_sufficiency.value,
            "evidence_refs": list(self.evidence_refs),
            "recommended_action": self.recommended_action,
            "raw_ai_output": self.raw_ai_output,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExplainReport":
        d = _known(cls, d)
        return cls(
            root_cause=d["root_cause"],
            race_class=d["race_class"],
            confidence=Confidence(d["confidence"]),
            is_reproduced=_as_bool(d["is_reproduced"]),
            cross_validation=CrossValidation.from_dict(d["cross_validation"]),
            evidence_sufficiency=EvidenceSufficiency(d["evidence_sufficiency"]),
            evidence_refs=list(d.get("evidence_refs", [])),
            recommended_action=d["recommended_action"],
            raw_ai_output=d["raw_ai_output"],
        )

    @classmethod
    def from_json(cls, source: Union[str, bytes, TextIO]) -> "ExplainReport":
        return cls.from_dict(json.loads(_load_source(source)))


# ---------------------------------------------------------------------------
# Session-oriented v2 contracts
# ---------------------------------------------------------------------------


class ExperimentKind(str, Enum):
    COMPUTE_SANITIZER = "compute_sanitizer"
    RANDOM_DELAY_STRESS = "random_delay_stress"
    REG_TRACE = "reg_trace"
    MEM_VALUE_TRACE = "mem_value_trace"
    REDUCE_DELAY_CONFIG = "reduce_delay_config"


class ExperimentPhase(str, Enum):
    INITIAL = "initial"
    FOLLOWUP = "followup"


class StressOutcome(str, Enum):
    REPRODUCED = "reproduced"
    NOT_REPRODUCED = "not_reproduced"
    INCOMPLETE = "incomplete"
    SKIPPED = "skipped"


class ReduceOutcome(str, Enum):
    REDUCED = "reduced"
    REPLAY_FAILED = "replay_failed"
    FAILED = "failed"


class DecisionKind(str, Enum):
    FINAL = "final"
    FOLLOWUP_REQUIRED = "followup_required"
    INCONCLUSIVE = "inconclusive"


class SessionStatus(str, Enum):
    WAITING_INITIAL = "waiting_initial"
    WAITING_REASONER = "waiting_reasoner"
    WAITING_FOLLOWUP = "waiting_followup"
    COMPLETED = "completed"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"


@dataclasses.dataclass
class OracleSpec:
    """Pre-approved correctness oracle; exit 0 means the issue manifested."""

    oracle_id: str
    argv: List[str]
    not_interesting_exit_codes: List[int] = dataclasses.field(
        default_factory=lambda: [1]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oracle_id": self.oracle_id,
            "argv": list(self.argv),
            "not_interesting_exit_codes": list(self.not_interesting_exit_codes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OracleSpec":
        return cls(
            oracle_id=d["oracle_id"],
            argv=list(d["argv"]),
            not_interesting_exit_codes=[
                int(x) for x in d.get("not_interesting_exit_codes", [1])
            ],
        )


@dataclasses.dataclass
class WorkUnit:
    """Portable target identity shared by Local and Distributed backends."""

    unit_id: str
    argv: List[str]
    workload: str = "triton_pytest"
    test_id: str = ""
    cwd: str = ""
    arch: str = ""
    source_revision: str = ""
    kernel: Optional[KernelId] = None
    oracle: Optional[OracleSpec] = None
    env: Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "argv": list(self.argv),
            "workload": self.workload,
            "test_id": self.test_id,
            "cwd": self.cwd,
            "arch": self.arch,
            "source_revision": self.source_revision,
            "kernel": None if self.kernel is None else self.kernel.to_dict(),
            "oracle": None if self.oracle is None else self.oracle.to_dict(),
            "env": dict(self.env),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkUnit":
        kernel = d.get("kernel")
        oracle = d.get("oracle")
        return cls(
            unit_id=d["unit_id"],
            argv=list(d["argv"]),
            workload=d.get("workload", "triton_pytest"),
            test_id=d.get("test_id", ""),
            cwd=d.get("cwd", ""),
            arch=d.get("arch", ""),
            source_revision=d.get("source_revision", ""),
            kernel=None if kernel is None else KernelId.from_dict(kernel),
            oracle=None if oracle is None else OracleSpec.from_dict(oracle),
            env=dict(d.get("env", {})),
        )

    @classmethod
    def from_json(cls, source: Union[str, bytes, TextIO]) -> "WorkUnit":
        return cls.from_dict(json.loads(_load_source(source)))


@dataclasses.dataclass
class TriggeringDelayConfig:
    """Immutable replay context for a random-delay configuration artifact."""

    artifact: ArtifactRef
    work_unit_id: str
    target_argv: List[str]
    oracle: OracleSpec
    source_revision: str
    arch: str
    kernel: Optional[KernelId]
    delay_ns: int
    enable_prob: float
    warpgroup_id: Optional[int]
    attempt_index: int
    completed_trials: int
    reproductions: int
    reproduction_rate: float
    cutracer_version: str = ""
    toolchain_version: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.reproduction_rate <= 1.0:
            raise ValueError("reproduction_rate must be between 0 and 1")
        if self.reproductions > self.completed_trials:
            raise ValueError("reproductions cannot exceed completed_trials")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "work_unit_id": self.work_unit_id,
            "target_argv": list(self.target_argv),
            "oracle": self.oracle.to_dict(),
            "source_revision": self.source_revision,
            "arch": self.arch,
            "kernel": None if self.kernel is None else self.kernel.to_dict(),
            "delay_ns": self.delay_ns,
            "enable_prob": self.enable_prob,
            "warpgroup_id": self.warpgroup_id,
            "attempt_index": self.attempt_index,
            "completed_trials": self.completed_trials,
            "reproductions": self.reproductions,
            "reproduction_rate": self.reproduction_rate,
            "cutracer_version": self.cutracer_version,
            "toolchain_version": self.toolchain_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TriggeringDelayConfig":
        kernel = d.get("kernel")
        warpgroup_id = d.get("warpgroup_id")
        return cls(
            artifact=ArtifactRef.from_dict(d["artifact"]),
            work_unit_id=d["work_unit_id"],
            target_argv=list(d["target_argv"]),
            oracle=OracleSpec.from_dict(d["oracle"]),
            source_revision=d.get("source_revision", ""),
            arch=d.get("arch", ""),
            kernel=None if kernel is None else KernelId.from_dict(kernel),
            delay_ns=int(d["delay_ns"]),
            enable_prob=float(d.get("enable_prob", 1.0)),
            warpgroup_id=(None if warpgroup_id is None else int(warpgroup_id)),
            attempt_index=int(d.get("attempt_index", 0)),
            completed_trials=int(d.get("completed_trials", 0)),
            reproductions=int(d.get("reproductions", 0)),
            reproduction_rate=float(d.get("reproduction_rate", 0.0)),
            cutracer_version=d.get("cutracer_version", ""),
            toolchain_version=d.get("toolchain_version", ""),
        )


@dataclasses.dataclass
class ExecutionProvenance:
    source_revision: str
    arch: str
    kernel: Optional[KernelId] = None
    tool_version: str = ""
    toolchain_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_revision": self.source_revision,
            "arch": self.arch,
            "kernel": None if self.kernel is None else self.kernel.to_dict(),
            "tool_version": self.tool_version,
            "toolchain_version": self.toolchain_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionProvenance":
        kernel = d.get("kernel")
        return cls(
            source_revision=d.get("source_revision", ""),
            arch=d.get("arch", ""),
            kernel=None if kernel is None else KernelId.from_dict(kernel),
            tool_version=d.get("tool_version", ""),
            toolchain_version=d.get("toolchain_version", ""),
        )


@dataclasses.dataclass
class FindingRecord:
    source_tool: str
    error_type: str
    kernel_name: Optional[str] = None
    source: Optional[SourceLoc] = None
    count: int = 1
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_tool": self.source_tool,
            "error_type": self.error_type,
            "kernel_name": self.kernel_name,
            "source": None if self.source is None else self.source.to_dict(),
            "count": self.count,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FindingRecord":
        source = d.get("source")
        return cls(
            source_tool=d["source_tool"],
            error_type=d["error_type"],
            kernel_name=d.get("kernel_name"),
            source=None if source is None else SourceLoc.from_dict(source),
            count=int(d.get("count", 1)),
            raw=d.get("raw", ""),
        )


@dataclasses.dataclass
class StressOptions:
    delay_ladder_ns: List[int] = dataclasses.field(
        default_factory=lambda: [1000, 5000, 10000, 50000, 100000]
    )
    attempts_per_delay: int = 3
    enable_prob: float = 1.0
    warpgroup_ids: List[int] = dataclasses.field(default_factory=list)
    stop_on_first_reproduction: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delay_ladder_ns": list(self.delay_ladder_ns),
            "attempts_per_delay": self.attempts_per_delay,
            "enable_prob": self.enable_prob,
            "warpgroup_ids": list(self.warpgroup_ids),
            "stop_on_first_reproduction": self.stop_on_first_reproduction,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StressOptions":
        return cls(
            delay_ladder_ns=[
                int(x)
                for x in d.get("delay_ladder_ns", [1000, 5000, 10000, 50000, 100000])
            ],
            attempts_per_delay=int(d.get("attempts_per_delay", 3)),
            enable_prob=float(d.get("enable_prob", 1.0)),
            warpgroup_ids=[int(x) for x in d.get("warpgroup_ids", [])],
            stop_on_first_reproduction=_as_bool(
                d.get("stop_on_first_reproduction", True)
            ),
        )


@dataclasses.dataclass
class TraceOptions:
    mode: ExperimentKind
    trace_size_limit_mb: int = 1024

    def __post_init__(self) -> None:
        if self.mode not in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
            raise ValueError(f"invalid trace mode: {self.mode}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "trace_size_limit_mb": self.trace_size_limit_mb,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceOptions":
        return cls(
            mode=ExperimentKind(d["mode"]),
            trace_size_limit_mb=int(d.get("trace_size_limit_mb", 1024)),
        )


@dataclasses.dataclass
class ReduceOptions:
    triggering_config: TriggeringDelayConfig
    strategy: str = "bisect"
    confidence_runs: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggering_config": self.triggering_config.to_dict(),
            "strategy": self.strategy,
            "confidence_runs": self.confidence_runs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReduceOptions":
        return cls(
            triggering_config=TriggeringDelayConfig.from_dict(d["triggering_config"]),
            strategy=d.get("strategy", "bisect"),
            confidence_runs=int(d.get("confidence_runs", 3)),
        )


@dataclasses.dataclass
class ExperimentSpec:
    experiment_id: str
    session_id: str
    phase: ExperimentPhase
    kind: ExperimentKind
    unit: WorkUnit
    round_index: int = 0
    sanitizer_tool: Optional[str] = None
    stress: Optional[StressOptions] = None
    trace: Optional[TraceOptions] = None
    reduction: Optional[ReduceOptions] = None

    def __post_init__(self) -> None:
        if self.kind == ExperimentKind.COMPUTE_SANITIZER and not self.sanitizer_tool:
            raise ValueError("compute-sanitizer experiment requires sanitizer_tool")
        if self.kind == ExperimentKind.RANDOM_DELAY_STRESS and self.stress is None:
            raise ValueError("random-delay experiment requires stress options")
        if self.kind in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
            if self.trace is None or self.trace.mode != self.kind:
                raise ValueError("trace experiment requires matching trace options")
        if self.kind == ExperimentKind.REDUCE_DELAY_CONFIG and self.reduction is None:
            raise ValueError("reduce experiment requires reduction options")
        if self.kind == ExperimentKind.REDUCE_DELAY_CONFIG:
            assert self.reduction is not None
            config = self.reduction.triggering_config
            if self.unit.oracle is None:
                raise ValueError("reduce experiment requires an approved oracle")
            if (
                config.work_unit_id != self.unit.unit_id
                or config.target_argv != self.unit.argv
                or config.oracle != self.unit.oracle
                or config.source_revision != self.unit.source_revision
                or config.arch != self.unit.arch
                or config.kernel != self.unit.kernel
            ):
                raise ValueError("reduce experiment does not match replay context")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "phase": self.phase.value,
            "kind": self.kind.value,
            "unit": self.unit.to_dict(),
            "round_index": self.round_index,
            "sanitizer_tool": self.sanitizer_tool,
            "stress": None if self.stress is None else self.stress.to_dict(),
            "trace": None if self.trace is None else self.trace.to_dict(),
            "reduction": (None if self.reduction is None else self.reduction.to_dict()),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentSpec":
        stress = d.get("stress")
        trace = d.get("trace")
        reduction = d.get("reduction")
        return cls(
            experiment_id=d["experiment_id"],
            session_id=d["session_id"],
            phase=ExperimentPhase(d["phase"]),
            kind=ExperimentKind(d["kind"]),
            unit=WorkUnit.from_dict(d["unit"]),
            round_index=int(d.get("round_index", 0)),
            sanitizer_tool=d.get("sanitizer_tool"),
            stress=None if stress is None else StressOptions.from_dict(stress),
            trace=None if trace is None else TraceOptions.from_dict(trace),
            reduction=(
                None if reduction is None else ReduceOptions.from_dict(reduction)
            ),
        )


@dataclasses.dataclass
class SanitizerSweepResult:
    experiment_id: str
    execution_status: ExecutionStatus
    outcome: SanitizerOutcome
    tool: str
    findings: List[FindingRecord]
    provenance: ExecutionProvenance
    log: Optional[ArtifactRef] = None
    duration_s: float = 0.0
    error: Optional[str] = None

    kind: ExperimentKind = dataclasses.field(
        default=ExperimentKind.COMPUTE_SANITIZER, init=False
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "experiment_id": self.experiment_id,
            "execution_status": self.execution_status.value,
            "outcome": self.outcome.value,
            "tool": self.tool,
            "findings": [x.to_dict() for x in self.findings],
            "provenance": self.provenance.to_dict(),
            "log": None if self.log is None else self.log.to_dict(),
            "duration_s": self.duration_s,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SanitizerSweepResult":
        log = d.get("log")
        return cls(
            experiment_id=d["experiment_id"],
            execution_status=ExecutionStatus(d["execution_status"]),
            outcome=SanitizerOutcome(d["outcome"]),
            tool=d["tool"],
            findings=[FindingRecord.from_dict(x) for x in d.get("findings", [])],
            provenance=ExecutionProvenance.from_dict(d["provenance"]),
            log=None if log is None else ArtifactRef.from_dict(log),
            duration_s=float(d.get("duration_s", 0.0)),
            error=d.get("error"),
        )


@dataclasses.dataclass
class StressTestResult:
    experiment_id: str
    execution_status: ExecutionStatus
    outcome: StressOutcome
    completed_trials: int
    reproductions: int
    infra_errors: int
    provenance: ExecutionProvenance
    triggering_config: Optional[TriggeringDelayConfig] = None
    log: Optional[ArtifactRef] = None
    duration_s: float = 0.0
    error: Optional[str] = None

    kind: ExperimentKind = dataclasses.field(
        default=ExperimentKind.RANDOM_DELAY_STRESS, init=False
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "experiment_id": self.experiment_id,
            "execution_status": self.execution_status.value,
            "outcome": self.outcome.value,
            "completed_trials": self.completed_trials,
            "reproductions": self.reproductions,
            "infra_errors": self.infra_errors,
            "provenance": self.provenance.to_dict(),
            "triggering_config": (
                None
                if self.triggering_config is None
                else self.triggering_config.to_dict()
            ),
            "log": None if self.log is None else self.log.to_dict(),
            "duration_s": self.duration_s,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StressTestResult":
        config = d.get("triggering_config")
        log = d.get("log")
        return cls(
            experiment_id=d["experiment_id"],
            execution_status=ExecutionStatus(d["execution_status"]),
            outcome=StressOutcome(d["outcome"]),
            completed_trials=int(d.get("completed_trials", 0)),
            reproductions=int(d.get("reproductions", 0)),
            infra_errors=int(d.get("infra_errors", 0)),
            provenance=ExecutionProvenance.from_dict(d["provenance"]),
            triggering_config=(
                None if config is None else TriggeringDelayConfig.from_dict(config)
            ),
            log=None if log is None else ArtifactRef.from_dict(log),
            duration_s=float(d.get("duration_s", 0.0)),
            error=d.get("error"),
        )


@dataclasses.dataclass
class TraceExperimentResult:
    experiment_id: str
    execution_status: ExecutionStatus
    mode: ExperimentKind
    provenance: ExecutionProvenance
    trace: Optional[ArtifactRef] = None
    log: Optional[ArtifactRef] = None
    duration_s: float = 0.0
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
            raise ValueError(f"invalid trace result mode: {self.mode}")

    @property
    def kind(self) -> ExperimentKind:
        return self.mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "experiment_id": self.experiment_id,
            "execution_status": self.execution_status.value,
            "mode": self.mode.value,
            "provenance": self.provenance.to_dict(),
            "trace": None if self.trace is None else self.trace.to_dict(),
            "log": None if self.log is None else self.log.to_dict(),
            "duration_s": self.duration_s,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceExperimentResult":
        trace = d.get("trace")
        log = d.get("log")
        return cls(
            experiment_id=d["experiment_id"],
            execution_status=ExecutionStatus(d["execution_status"]),
            mode=ExperimentKind(d.get("mode", d["kind"])),
            provenance=ExecutionProvenance.from_dict(d["provenance"]),
            trace=None if trace is None else ArtifactRef.from_dict(trace),
            log=None if log is None else ArtifactRef.from_dict(log),
            duration_s=float(d.get("duration_s", 0.0)),
            error=d.get("error"),
        )


@dataclasses.dataclass
class EssentialDelayPoint:
    kernel_name: str
    pc_offset: str
    sass: str
    delay_ns: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EssentialDelayPoint":
        return cls(
            kernel_name=d.get("kernel_name", ""),
            pc_offset=str(d.get("pc_offset", "")),
            sass=d.get("sass", ""),
            delay_ns=int(d.get("delay_ns", 0)),
        )


@dataclasses.dataclass
class ReduceExperimentResult:
    experiment_id: str
    execution_status: ExecutionStatus
    outcome: ReduceOutcome
    provenance: ExecutionProvenance
    input_config: TriggeringDelayConfig
    minimized_config: Optional[ArtifactRef] = None
    report: Optional[ArtifactRef] = None
    essential_points: List[EssentialDelayPoint] = dataclasses.field(
        default_factory=list
    )
    duration_s: float = 0.0
    error: Optional[str] = None

    kind: ExperimentKind = dataclasses.field(
        default=ExperimentKind.REDUCE_DELAY_CONFIG, init=False
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "experiment_id": self.experiment_id,
            "execution_status": self.execution_status.value,
            "outcome": self.outcome.value,
            "provenance": self.provenance.to_dict(),
            "input_config": self.input_config.to_dict(),
            "minimized_config": (
                None
                if self.minimized_config is None
                else self.minimized_config.to_dict()
            ),
            "report": None if self.report is None else self.report.to_dict(),
            "essential_points": [x.to_dict() for x in self.essential_points],
            "duration_s": self.duration_s,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReduceExperimentResult":
        minimized = d.get("minimized_config")
        report = d.get("report")
        return cls(
            experiment_id=d["experiment_id"],
            execution_status=ExecutionStatus(d["execution_status"]),
            outcome=ReduceOutcome(d["outcome"]),
            provenance=ExecutionProvenance.from_dict(d["provenance"]),
            input_config=TriggeringDelayConfig.from_dict(d["input_config"]),
            minimized_config=(
                None if minimized is None else ArtifactRef.from_dict(minimized)
            ),
            report=None if report is None else ArtifactRef.from_dict(report),
            essential_points=[
                EssentialDelayPoint.from_dict(x) for x in d.get("essential_points", [])
            ],
            duration_s=float(d.get("duration_s", 0.0)),
            error=d.get("error"),
        )


ExperimentResult = Union[
    SanitizerSweepResult,
    StressTestResult,
    TraceExperimentResult,
    ReduceExperimentResult,
]


def experiment_result_from_dict(d: Dict[str, Any]) -> ExperimentResult:
    kind = ExperimentKind(d["kind"])
    if kind == ExperimentKind.COMPUTE_SANITIZER:
        return SanitizerSweepResult.from_dict(d)
    if kind == ExperimentKind.RANDOM_DELAY_STRESS:
        return StressTestResult.from_dict(d)
    if kind in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
        return TraceExperimentResult.from_dict(d)
    if kind == ExperimentKind.REDUCE_DELAY_CONFIG:
        return ReduceExperimentResult.from_dict(d)
    raise ValueError(f"unknown experiment kind: {kind}")


@dataclasses.dataclass
class EvidenceBundle:
    sanitizer: List[SanitizerSweepResult] = dataclasses.field(default_factory=list)
    stress: List[StressTestResult] = dataclasses.field(default_factory=list)
    traces: List[TraceExperimentResult] = dataclasses.field(default_factory=list)
    reductions: List[ReduceExperimentResult] = dataclasses.field(default_factory=list)

    def add(self, result: ExperimentResult) -> None:
        if isinstance(result, SanitizerSweepResult):
            self.sanitizer.append(result)
        elif isinstance(result, StressTestResult):
            self.stress.append(result)
        elif isinstance(result, TraceExperimentResult):
            self.traces.append(result)
        else:
            self.reductions.append(result)

    def triggering_config(self) -> Optional[TriggeringDelayConfig]:
        for result in self.stress:
            if result.triggering_config is not None:
                return result.triggering_config
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sanitizer": [x.to_dict() for x in self.sanitizer],
            "stress": [x.to_dict() for x in self.stress],
            "traces": [x.to_dict() for x in self.traces],
            "reductions": [x.to_dict() for x in self.reductions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceBundle":
        return cls(
            sanitizer=[
                SanitizerSweepResult.from_dict(x) for x in d.get("sanitizer", [])
            ],
            stress=[StressTestResult.from_dict(x) for x in d.get("stress", [])],
            traces=[TraceExperimentResult.from_dict(x) for x in d.get("traces", [])],
            reductions=[
                ReduceExperimentResult.from_dict(x) for x in d.get("reductions", [])
            ],
        )


@dataclasses.dataclass
class ExperimentRequest:
    kind: ExperimentKind
    rationale: str

    def __post_init__(self) -> None:
        if self.kind == ExperimentKind.COMPUTE_SANITIZER:
            return
        if self.kind not in (
            ExperimentKind.RANDOM_DELAY_STRESS,
            ExperimentKind.REG_TRACE,
            ExperimentKind.MEM_VALUE_TRACE,
            ExperimentKind.REDUCE_DELAY_CONFIG,
        ):
            raise ValueError(f"unsupported experiment request: {self.kind}")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "rationale": self.rationale}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRequest":
        return cls(kind=ExperimentKind(d["kind"]), rationale=d.get("rationale", ""))


@dataclasses.dataclass
class AnalysisDecision:
    kind: DecisionKind
    rationale: str
    report: Optional[ExplainReport] = None
    requests: List[ExperimentRequest] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind == DecisionKind.FOLLOWUP_REQUIRED and not self.requests:
            raise ValueError("follow-up decision requires at least one request")
        if self.kind in (DecisionKind.FINAL, DecisionKind.INCONCLUSIVE):
            if self.report is None:
                raise ValueError("terminal decision requires an ExplainReport")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "rationale": self.rationale,
            "report": None if self.report is None else self.report.to_dict(),
            "requests": [x.to_dict() for x in self.requests],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisDecision":
        report = d.get("report")
        return cls(
            kind=DecisionKind(d["kind"]),
            rationale=d.get("rationale", ""),
            report=None if report is None else ExplainReport.from_dict(report),
            requests=[ExperimentRequest.from_dict(x) for x in d.get("requests", [])],
        )


@dataclasses.dataclass
class SessionBudget:
    max_rounds: int = 2
    max_experiments: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionBudget":
        return cls(
            max_rounds=int(d.get("max_rounds", 2)),
            max_experiments=int(d.get("max_experiments", 8)),
        )


@dataclasses.dataclass
class AnalysisTurn:
    turn_index: int
    decision: AnalysisDecision

    def to_dict(self) -> Dict[str, Any]:
        return {"turn_index": self.turn_index, "decision": self.decision.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisTurn":
        return cls(
            turn_index=int(d["turn_index"]),
            decision=AnalysisDecision.from_dict(d["decision"]),
        )


@dataclasses.dataclass
class AnalysisSession:
    session_id: str
    unit: WorkUnit
    status: SessionStatus
    evidence: EvidenceBundle
    budget: SessionBudget
    pending_experiments: List[ExperimentSpec] = dataclasses.field(default_factory=list)
    completed_experiment_ids: List[str] = dataclasses.field(default_factory=list)
    experiment_fingerprints: List[str] = dataclasses.field(default_factory=list)
    round_index: int = 0
    submitted_experiments: int = 0
    turns: List[AnalysisTurn] = dataclasses.field(default_factory=list)
    report: Optional[ExplainReport] = None
    termination_reason: str = ""
    state_revision: int = 0
    schema_version: int = 3

    @property
    def pending_experiment_ids(self) -> List[str]:
        return [spec.experiment_id for spec in self.pending_experiments]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "unit": self.unit.to_dict(),
            "status": self.status.value,
            "evidence": self.evidence.to_dict(),
            "budget": self.budget.to_dict(),
            "pending_experiments": [x.to_dict() for x in self.pending_experiments],
            "completed_experiment_ids": list(self.completed_experiment_ids),
            "experiment_fingerprints": list(self.experiment_fingerprints),
            "round_index": self.round_index,
            "submitted_experiments": self.submitted_experiments,
            "turns": [x.to_dict() for x in self.turns],
            "report": None if self.report is None else self.report.to_dict(),
            "termination_reason": self.termination_reason,
            "state_revision": self.state_revision,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisSession":
        version = int(d.get("schema_version", 3))
        if version < 3:
            raise ValueError(f"unsupported AnalysisSession schema_version: {version}")
        report = d.get("report")
        return cls(
            session_id=d["session_id"],
            unit=WorkUnit.from_dict(d["unit"]),
            status=SessionStatus(d["status"]),
            evidence=EvidenceBundle.from_dict(d.get("evidence", {})),
            budget=SessionBudget.from_dict(d.get("budget", {})),
            pending_experiments=[
                ExperimentSpec.from_dict(x) for x in d.get("pending_experiments", [])
            ],
            completed_experiment_ids=list(d.get("completed_experiment_ids", [])),
            experiment_fingerprints=list(d.get("experiment_fingerprints", [])),
            round_index=int(d.get("round_index", 0)),
            submitted_experiments=int(d.get("submitted_experiments", 0)),
            turns=[AnalysisTurn.from_dict(x) for x in d.get("turns", [])],
            report=None if report is None else ExplainReport.from_dict(report),
            termination_reason=d.get("termination_reason", ""),
            state_revision=int(d.get("state_revision", 0)),
            schema_version=version,
        )

    @classmethod
    def from_json(cls, source: Union[str, bytes, TextIO]) -> "AnalysisSession":
        return cls.from_dict(json.loads(_load_source(source)))


@dataclasses.dataclass
class SessionReport:
    session_id: str
    status: SessionStatus
    explanation: ExplainReport
    evidence: EvidenceBundle
    turns: List[AnalysisTurn]
    termination_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "explanation": self.explanation.to_dict(),
            "evidence": self.evidence.to_dict(),
            "turns": [x.to_dict() for x in self.turns],
            "termination_reason": self.termination_reason,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionReport":
        d = _known(cls, d)
        return cls(
            session_id=d["session_id"],
            status=SessionStatus(d["status"]),
            explanation=ExplainReport.from_dict(d["explanation"]),
            evidence=EvidenceBundle.from_dict(d.get("evidence", {})),
            turns=[AnalysisTurn.from_dict(x) for x in d.get("turns", [])],
            termination_reason=d.get("termination_reason", ""),
        )

    @classmethod
    def from_json(cls, source: Union[str, bytes, TextIO]) -> "SessionReport":
        return cls.from_dict(json.loads(_load_source(source)))
