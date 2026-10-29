"""Wire contracts for the CUTracer automated diagnosis service.

This module is the service-layer handoff schema. It and its shared retention
policy dependency are pure stdlib and do not load CUTracer runtime or analysis
code. It contains the session-oriented experiment/evidence contracts plus the
original single-trace explain contracts retained for compatibility.

Design notes:
  * Enums subclass ``(str, Enum)`` so JSON is stable human-readable text.
  * ``to_dict`` recurses into nested dataclasses, lowers enums to ``.value``,
    and normally preserves ``None`` optionals as ``null`` (keys are kept).
    Additive compatibility metadata may be omitted while unknown so legacy
    payload digests remain stable.
  * ``from_dict`` is FORWARD-COMPATIBLE: it reads only known fields and silently
    drops unknown keys, so a newer producer never breaks an older consumer.
  * ``race_class`` uses a stable enum; legacy spellings are accepted only at
    deserialization boundaries and are emitted in canonical form.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import re
from enum import Enum
from typing import Any, Dict, List, Optional, TextIO, Tuple, Type, TypeVar, Union

from cutracer.stress.retention import MAX_REJECTED_CONFIGS

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


class RaceClass(str, Enum):
    """Stable classification of the race mechanism diagnosed in a report.

    NONE makes no race diagnosis; it does not imply no GPU error or prove the
    kernel race-free. Sanitizer symptoms are classified by FindingRecord.error_type.
    """

    NONE = "none"
    UNKNOWN = "unknown"
    RAW = "raw"
    WAR = "war"
    WAW = "waw"
    ATOMICITY = "atomicity"
    SYNCHRONIZATION = "synchronization"
    LIFETIME = "lifetime"
    UNCLASSIFIED = "unclassified"

    @classmethod
    def from_wire(cls, value: str) -> "RaceClass":
        aliases = {
            "RAW": cls.RAW,
            "WAR": cls.WAR,
            "WAW": cls.WAW,
            "intra-wg TMEM WAR": cls.WAR,
        }
        alias = aliases.get(value)
        return alias if alias is not None else cls(value)


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


_SUPPORTED_SANITIZER_TOOLS = frozenset(
    {"initcheck", "memcheck", "racecheck", "synccheck"}
)


class ResultCompleteness(str, Enum):
    """How completely an experiment covered its declared verdict scope."""

    COMPLETE = "complete"
    COMPLETE_EARLY = "complete_early"
    PARTIAL = "partial"
    SKIPPED = "skipped"
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
    # Vendor-facing execution metadata; stable identity remains name + checksum.
    mangled_name: Optional[str] = dataclasses.field(default=None, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "cubin_hash": self.cubin_hash}
        if self.mangled_name is not None:
            result["mangled_name"] = self.mangled_name
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelId":
        d = _known(cls, d)
        return cls(
            name=d["name"],
            cubin_hash=d["cubin_hash"],
            mangled_name=d.get("mangled_name"),
        )


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


def _is_visible_artifact_basename(artifact: ArtifactRef) -> bool:
    path = artifact.relative_path
    return (
        bool(path) and not path.startswith(".") and "/" not in path and "\\" not in path
    )


def _deduplicate_artifact_refs(refs: List[ArtifactRef]) -> List[ArtifactRef]:
    """Return exact artifact references once, preserving producer order."""
    unique: List[ArtifactRef] = []
    for ref in refs:
        if ref not in unique:
            unique.append(ref)
    return unique


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
    race_class: RaceClass
    confidence: Confidence
    is_reproduced: bool
    cross_validation: CrossValidation
    evidence_sufficiency: EvidenceSufficiency
    evidence_refs: List[str]
    recommended_action: str
    raw_ai_output: str

    def __post_init__(self) -> None:
        if not isinstance(self.race_class, RaceClass):
            self.race_class = RaceClass.from_wire(self.race_class)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "race_class": self.race_class.value,
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
            race_class=RaceClass.from_wire(d["race_class"]),
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
    KERNEL_DISCOVERY = "kernel_discovery"
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
    UNATTRIBUTED_REPRODUCTION = "unattributed_reproduction"
    NOT_REPRODUCED = "not_reproduced"
    INCOMPLETE = "incomplete"
    SKIPPED = "skipped"


class KernelDiscoveryOutcome(str, Enum):
    DISCOVERED = "discovered"
    NO_KERNELS = "no_kernels"
    NO_MATCHING_KERNELS = "no_matching_kernels"
    TOO_MANY_KERNELS = "too_many_kernels"
    INFRA_ERROR = "infra_error"


class ReduceOutcome(str, Enum):
    REDUCED = "reduced"
    NON_REPRODUCIBLE = "non_reproducible"
    INFRA_ERROR = "infra_error"
    STALE_INPUT = "stale_input"

    # Legacy values remain readable while new producers emit unambiguous
    # semantic outcomes.
    REPLAY_FAILED = "replay_failed"
    FAILED = "failed"

    @classmethod
    def from_wire(cls, value: str) -> "ReduceOutcome":
        if value == cls.REPLAY_FAILED.value:
            return cls.NON_REPRODUCIBLE
        if value == cls.FAILED.value:
            return cls.INFRA_ERROR
        return cls(value)


class DecisionKind(str, Enum):
    FINAL = "final"
    FOLLOWUP_REQUIRED = "followup_required"
    INCONCLUSIVE = "inconclusive"


class TriageOutcome(str, Enum):
    """What triage decided to do with a completed initial campaign."""

    ANALYZE = "analyze"
    FINISH = "finish"
    INCONCLUSIVE = "inconclusive"


class BaselineVerdict(str, Enum):
    """How a scope's findings compare with the last complete campaign."""

    NOT_EVALUATED = "not_evaluated"
    NEW = "new"
    UNCHANGED = "unchanged"
    RESOLVED = "resolved"
    NO_BASELINE = "no_baseline"


class SessionStatus(str, Enum):
    WAITING_CHILDREN = "waiting_children"
    WAITING_INITIAL = "waiting_initial"
    WAITING_TRIAGE = "waiting_triage"
    WAITING_REASONER = "waiting_reasoner"
    WAITING_FOLLOWUP = "waiting_followup"
    COMPLETED = "completed"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"


class ReportPublicationStatus(str, Enum):
    """Durable delivery state for a terminal diagnosis report."""

    NOT_READY = "not_ready"
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"


_MAX_RANDOM_DELAY_NS = (1 << 32) - 2


def _derived_reproduction_rate(completed_trials: int, reproductions: int) -> float:
    return reproductions / completed_trials if completed_trials else 0.0


def _reproduction_rates_match(actual: float, expected: float) -> bool:
    # Persisted payloads may round this derived display value to three decimal
    # places. Counts remain authoritative; reject material disagreement only.
    return math.isclose(actual, expected, rel_tol=1e-3, abs_tol=5e-4)


@dataclasses.dataclass
class OracleSpec:
    """Pre-approved correctness oracle; exit 0 means the issue manifested."""

    oracle_id: str
    argv: List[str]
    not_interesting_exit_codes: List[int] = dataclasses.field(
        default_factory=lambda: [1]
    )

    def __post_init__(self) -> None:
        if not self.oracle_id.strip():
            raise ValueError("oracle_id must not be empty")
        if not self.argv or not self.argv[0].strip():
            raise ValueError("oracle argv must not be empty")
        if not self.not_interesting_exit_codes:
            raise ValueError("not_interesting_exit_codes must not be empty")
        if any(code <= 0 or code > 255 for code in self.not_interesting_exit_codes):
            raise ValueError("not_interesting_exit_codes must be between 1 and 255")
        if len(set(self.not_interesting_exit_codes)) != len(
            self.not_interesting_exit_codes
        ):
            raise ValueError("not_interesting_exit_codes must not contain duplicates")

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
    kernel_filters: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_filters, list) or any(
            not isinstance(value, str) for value in self.kernel_filters
        ):
            raise ValueError("WorkUnit kernel filters must be a string array")
        if len(self.kernel_filters) != len(set(self.kernel_filters)) or any(
            not value.strip() or "," in value for value in self.kernel_filters
        ):
            raise ValueError("WorkUnit kernel filters must be unique and comma-free")

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
            "kernel_filters": list(self.kernel_filters),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkUnit":
        kernel = d.get("kernel")
        oracle = d.get("oracle")
        kernel_filters = d.get("kernel_filters", [])
        if not isinstance(kernel_filters, list):
            raise ValueError("WorkUnit kernel filters must be a string array")
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
            kernel_filters=list(kernel_filters),
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
    enabled_point_count: Optional[int] = None
    cutracer_version: str = ""
    toolchain_version: str = ""

    def __post_init__(self) -> None:
        if self.completed_trials < 0 or self.reproductions < 0:
            raise ValueError("trial counts must not be negative")
        if not 0.0 <= self.reproduction_rate <= 1.0:
            raise ValueError("reproduction_rate must be between 0 and 1")
        if self.reproductions > self.completed_trials:
            raise ValueError("reproductions cannot exceed completed_trials")
        expected_rate = _derived_reproduction_rate(
            self.completed_trials, self.reproductions
        )
        if not _reproduction_rates_match(self.reproduction_rate, expected_rate):
            raise ValueError("reproduction_rate does not match trial counts")
        if self.enabled_point_count is not None and self.enabled_point_count <= 0:
            raise ValueError("enabled_point_count must be positive when present")

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
            "enabled_point_count": self.enabled_point_count,
            "cutracer_version": self.cutracer_version,
            "toolchain_version": self.toolchain_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TriggeringDelayConfig":
        kernel = d.get("kernel")
        warpgroup_id = d.get("warpgroup_id")
        completed_trials = int(d.get("completed_trials", 0))
        reproductions = int(d.get("reproductions", 0))
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
            completed_trials=completed_trials,
            reproductions=reproductions,
            reproduction_rate=float(
                d.get(
                    "reproduction_rate",
                    _derived_reproduction_rate(completed_trials, reproductions),
                )
            ),
            enabled_point_count=(
                None
                if d.get("enabled_point_count") is None
                else int(d["enabled_point_count"])
            ),
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
    """One finding, normalized across sanitizer tools.

    ``error_type`` is the stable vendor-neutral class used for correlation and
    suppression; ``error_detail`` keeps the tool's own wording, which encodes
    per-finding specifics such as access width that must not fragment grouping.
    ``kernel_name`` is the legacy wire name for the sanitizer's primary
    faulting symbol; it may identify an inlined or device function rather than
    the kernel entry point. The pinned entry-kernel identity lives in the
    enclosing result's ``provenance.kernel``.
    """

    source_tool: str
    error_type: str
    kernel_name: Optional[str] = None
    source: Optional[SourceLoc] = None
    count: int = 1
    raw: str = ""
    error_detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_tool": self.source_tool,
            "error_type": self.error_type,
            "kernel_name": self.kernel_name,
            "source": None if self.source is None else self.source.to_dict(),
            "count": self.count,
            "raw": self.raw,
            "error_detail": self.error_detail,
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
            error_detail=d.get("error_detail", ""),
        )


_YMD_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_ymd_date(value: str) -> Optional[datetime.date]:
    """Return the calendar date in ``YYYY-MM-DD`` ``value``, else ``None``.

    Both halves of the check are needed: the shape test alone would accept
    ``2026-13-45``, while ``date.fromisoformat`` alone would accept other ISO
    spellings such as the compact ``20260815`` and the week form
    ``2026-W33-1``. Callers store the string and rely on it denoting one exact
    day, so anything but the plain form is rejected.
    """
    if not _YMD_RE.fullmatch(value):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


@dataclasses.dataclass(frozen=True)
class SuppressionRule:
    """A versioned reason to stop treating a class of findings as actionable.

    Every field that narrows the rule is optional and empty means "any", so a
    rule is only as broad as it is written. ``version`` is bumped whenever the
    matcher changes so that an audit entry always names the exact rule text
    that fired.
    """

    rule_id: str
    version: int
    reason: str
    owner: str
    kernel_pattern: Optional[str] = None
    tools: Tuple[str, ...] = ()
    error_types: Tuple[str, ...] = ()
    expires_on: Optional[str] = None
    tracking_ticket: str = ""
    reviewed_by: str = ""
    reviewed_on: Optional[str] = None
    review_due_on: Optional[str] = None
    permanent: bool = False

    def __post_init__(self) -> None:
        # A malformed expiry would compare wrongly as a plain string and could
        # keep hiding findings past its intended life, so reject it at the
        # source rather than let the policy run on it.
        if self.expires_on is not None and parse_ymd_date(self.expires_on) is None:
            raise ValueError(
                f"suppression rule {self.rule_id}: expires_on must be YYYY-MM-DD, "
                f"got {self.expires_on!r}"
            )
        for field_name in ("reviewed_on", "review_due_on"):
            value = getattr(self, field_name)
            if value is not None and parse_ymd_date(value) is None:
                raise ValueError(
                    f"suppression rule {self.rule_id}: {field_name} must be "
                    f"YYYY-MM-DD, got {value!r}"
                )
        if self.permanent and self.expires_on is not None:
            raise ValueError("permanent suppression rule cannot have an expiry")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "version": self.version,
            "reason": self.reason,
            "owner": self.owner,
            "kernel_pattern": self.kernel_pattern,
            "tools": list(self.tools),
            "error_types": list(self.error_types),
            "expires_on": self.expires_on,
            "tracking_ticket": self.tracking_ticket,
            "reviewed_by": self.reviewed_by,
            "reviewed_on": self.reviewed_on,
            "review_due_on": self.review_due_on,
            "permanent": self.permanent,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SuppressionRule":
        return cls(
            rule_id=d["rule_id"],
            version=int(d["version"]),
            reason=d["reason"],
            owner=d["owner"],
            kernel_pattern=d.get("kernel_pattern"),
            tools=tuple(d.get("tools", ())),
            error_types=tuple(d.get("error_types", ())),
            expires_on=d.get("expires_on"),
            tracking_ticket=d.get("tracking_ticket", ""),
            reviewed_by=d.get("reviewed_by", ""),
            reviewed_on=d.get("reviewed_on"),
            review_due_on=d.get("review_due_on"),
            permanent=_as_bool(d.get("permanent", False)),
        )


@dataclasses.dataclass(frozen=True)
class ScopeId:
    """The unit a baseline is kept for.

    Deliberately per (test family, tool) rather than per campaign: the sweep
    corpus times out on 40% of its runs, so a campaign-wide scope would never
    be complete and no clean baseline could ever be established. Per-scope, the
    60% that do complete keep working while the rest stay inconclusive.
    """

    test_family: str
    tool: str

    @property
    def key(self) -> str:
        return f"{self.test_family}::{self.tool}"

    def to_dict(self) -> Dict[str, Any]:
        return {"test_family": self.test_family, "tool": self.tool}

    @classmethod
    def from_dict(cls, d: object) -> "ScopeId":
        if not isinstance(d, dict):
            raise ValueError("scope must be a JSON object")
        test_family = d.get("test_family")
        if not isinstance(test_family, str):
            raise ValueError("scope test_family must be a string")
        tool = d.get("tool")
        if not isinstance(tool, str):
            raise ValueError("scope tool must be a string")
        return cls(test_family=test_family, tool=tool)


@dataclasses.dataclass
class TriageDecision:
    """The verdict of the deterministic step that runs before any reasoning.

    Triage is what keeps an expensive AI turn from being spent on a campaign
    that is incomplete, entirely suppressed, or unchanged since the last run.
    """

    outcome: TriageOutcome
    reason: str
    snapshot: Optional["TriageSnapshot"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "reason": self.reason,
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TriageDecision":
        snapshot = d.get("snapshot")
        return cls(
            outcome=TriageOutcome(d["outcome"]),
            reason=d["reason"],
            snapshot=(None if snapshot is None else TriageSnapshot.from_dict(snapshot)),
        )


@dataclasses.dataclass(frozen=True)
class ScopeTriageSnapshot:
    """Auditable completeness and baseline result for one declared scope."""

    scope: ScopeId
    completeness: ResultCompleteness
    baseline_verdict: BaselineVerdict = BaselineVerdict.NOT_EVALUATED
    observed_keys: Tuple[str, ...] = ()
    new_keys: Tuple[str, ...] = ()
    resolved_keys: Tuple[str, ...] = ()
    unchanged_keys: Tuple[str, ...] = ()
    baseline_recorded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.to_dict(),
            "completeness": self.completeness.value,
            "baseline_verdict": self.baseline_verdict.value,
            "observed_keys": list(self.observed_keys),
            "new_keys": list(self.new_keys),
            "resolved_keys": list(self.resolved_keys),
            "unchanged_keys": list(self.unchanged_keys),
            "baseline_recorded": self.baseline_recorded,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScopeTriageSnapshot":
        return cls(
            scope=ScopeId.from_dict(d["scope"]),
            completeness=ResultCompleteness(d["completeness"]),
            baseline_verdict=BaselineVerdict(
                d.get("baseline_verdict", BaselineVerdict.NOT_EVALUATED.value)
            ),
            observed_keys=tuple(d.get("observed_keys", ())),
            new_keys=tuple(d.get("new_keys", ())),
            resolved_keys=tuple(d.get("resolved_keys", ())),
            unchanged_keys=tuple(d.get("unchanged_keys", ())),
            baseline_recorded=_as_bool(d.get("baseline_recorded", False)),
        )


@dataclasses.dataclass(frozen=True)
class TriageSnapshot:
    """Deterministic campaign facts retained independently of its decision."""

    scopes: Tuple[ScopeTriageSnapshot, ...] = ()
    stress_completeness: ResultCompleteness = ResultCompleteness.UNKNOWN
    suppression_audit: Tuple["SuppressionAuditEntry", ...] = ()

    def __post_init__(self) -> None:
        keys = [scope.scope.key for scope in self.scopes]
        if len(keys) != len(set(keys)):
            raise ValueError("triage snapshot scopes must be unique")

    @property
    def initial_evidence_complete(self) -> bool:
        scope_complete = bool(self.scopes) and all(
            scope.completeness == ResultCompleteness.COMPLETE for scope in self.scopes
        )
        stress_complete = self.stress_completeness in (
            ResultCompleteness.COMPLETE,
            ResultCompleteness.COMPLETE_EARLY,
        )
        return scope_complete and stress_complete

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scopes": [scope.to_dict() for scope in self.scopes],
            "stress_completeness": self.stress_completeness.value,
            "suppression_audit": [entry.to_dict() for entry in self.suppression_audit],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TriageSnapshot":
        return cls(
            scopes=tuple(
                ScopeTriageSnapshot.from_dict(scope) for scope in d.get("scopes", ())
            ),
            stress_completeness=ResultCompleteness(
                d.get("stress_completeness", ResultCompleteness.UNKNOWN.value)
            ),
            suppression_audit=tuple(
                SuppressionAuditEntry.from_dict(entry)
                for entry in d.get("suppression_audit", ())
            ),
        )


@dataclasses.dataclass(frozen=True)
class SuppressionDecision:
    """Audit record naming the exact rule version that hid a finding."""

    rule_id: str
    rule_version: int
    reason: str
    owner: str = ""
    expires_on: Optional[str] = None
    tracking_ticket: str = ""
    reviewed_by: str = ""
    reviewed_on: Optional[str] = None
    review_due_on: Optional[str] = None
    permanent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "reason": self.reason,
            "owner": self.owner,
            "expires_on": self.expires_on,
            "tracking_ticket": self.tracking_ticket,
            "reviewed_by": self.reviewed_by,
            "reviewed_on": self.reviewed_on,
            "review_due_on": self.review_due_on,
            "permanent": self.permanent,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SuppressionDecision":
        return cls(
            rule_id=d["rule_id"],
            rule_version=int(d["rule_version"]),
            reason=d["reason"],
            owner=d.get("owner", ""),
            expires_on=d.get("expires_on"),
            tracking_ticket=d.get("tracking_ticket", ""),
            reviewed_by=d.get("reviewed_by", ""),
            reviewed_on=d.get("reviewed_on"),
            review_due_on=d.get("review_due_on"),
            permanent=_as_bool(d.get("permanent", False)),
        )


@dataclasses.dataclass(frozen=True)
class SuppressionAuditEntry:
    """The exact finding and policy revision hidden during triage."""

    finding_key: str
    source_tool: str
    error_type: str
    kernel_name: Optional[str]
    source: Optional[SourceLoc]
    decision: SuppressionDecision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_key": self.finding_key,
            "source_tool": self.source_tool,
            "error_type": self.error_type,
            "kernel_name": self.kernel_name,
            "source": None if self.source is None else self.source.to_dict(),
            "decision": self.decision.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SuppressionAuditEntry":
        source = d.get("source")
        return cls(
            finding_key=d["finding_key"],
            source_tool=d["source_tool"],
            error_type=d["error_type"],
            kernel_name=d.get("kernel_name"),
            source=None if source is None else SourceLoc.from_dict(source),
            decision=SuppressionDecision.from_dict(d["decision"]),
        )


@dataclasses.dataclass
class SanitizerSummary:
    """Run-level aggregate copied from a canonical sanitizer summary."""

    total_hazards: int
    errors: int
    warnings: int
    raw: str = ""
    not_printed: Optional[int] = None

    @property
    def is_positive(self) -> bool:
        return self.total_hazards > 0 or self.errors > 0 or self.warnings > 0

    @property
    def is_zero(self) -> bool:
        return self.total_hazards == 0 and self.errors == 0 and self.warnings == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_hazards": self.total_hazards,
            "errors": self.errors,
            "warnings": self.warnings,
            "raw": self.raw,
            "not_printed": self.not_printed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SanitizerSummary":
        return cls(
            total_hazards=int(d["total_hazards"]),
            errors=int(d["errors"]),
            warnings=int(d["warnings"]),
            raw=d.get("raw", ""),
            not_printed=(
                None if d.get("not_printed") is None else int(d["not_printed"])
            ),
        )


DEFAULT_MAX_DISCOVERED_KERNELS = 32


@dataclasses.dataclass(frozen=True)
class KernelDiscoveryOptions:
    max_kernels: int = DEFAULT_MAX_DISCOVERED_KERNELS

    def __post_init__(self) -> None:
        if self.max_kernels <= 0:
            raise ValueError("max_kernels must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {"max_kernels": self.max_kernels}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelDiscoveryOptions":
        return cls(
            max_kernels=int(d.get("max_kernels", DEFAULT_MAX_DISCOVERED_KERNELS))
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

    def __post_init__(self) -> None:
        # Invalid schedules have no meaningful planned scope. Reject them on
        # construction, including deserialization, instead of fabricating a
        # completeness value from an unexecutable legacy payload.
        if not self.delay_ladder_ns:
            raise ValueError("stress delay ladder must not be empty")
        if any(
            delay_ns < 1 or delay_ns > _MAX_RANDOM_DELAY_NS
            for delay_ns in self.delay_ladder_ns
        ):
            raise ValueError("stress delays must be between 1 and UINT32_MAX - 1")
        if len(set(self.delay_ladder_ns)) != len(self.delay_ladder_ns):
            raise ValueError("stress delay ladder must not contain duplicates")
        if self.attempts_per_delay <= 0:
            raise ValueError("stress attempts_per_delay must be positive")
        if not 0.0 < self.enable_prob <= 1.0:
            raise ValueError("stress enable_prob must be greater than 0 and at most 1")
        if any(
            warpgroup_id < 0 or warpgroup_id > 7 for warpgroup_id in self.warpgroup_ids
        ):
            raise ValueError("stress warpgroup IDs must be between 0 and 7")
        if len(set(self.warpgroup_ids)) != len(self.warpgroup_ids):
            raise ValueError("stress warpgroup IDs must not contain duplicates")

    @property
    def planned_trials(self) -> int:
        return (
            len(self.delay_ladder_ns)
            * (len(self.warpgroup_ids) or 1)
            * self.attempts_per_delay
        )

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


class TraceInstructionCategory(str, Enum):
    MMA = "mma"
    TMA = "tma"
    SYNC = "sync"


class TraceConfigSource(str, Enum):
    TRIGGERING = "triggering"
    MINIMIZED = "minimized"


def _trace_instrument_modes(primary: ExperimentKind, modes: List[str]) -> List[str]:
    allowed = ("reg_trace", "mem_value_trace", "tma_trace")
    if not isinstance(modes, list) or any(
        not isinstance(mode, str) or mode not in allowed for mode in modes
    ):
        raise ValueError("trace instrument_modes must be an allowlisted list")
    if not modes:
        return []
    if primary.value not in modes:
        raise ValueError("trace instrument_modes must include the primary mode")
    normalized = [mode for mode in allowed if mode in modes]
    return [] if normalized == [primary.value] else normalized


def _pc_offset(value: Any) -> str:
    try:
        parsed = int(str(value), 0)
    except ValueError as exc:
        raise ValueError(f"invalid PC offset: {value!r}") from exc
    if not 0 <= parsed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("PC offset must fit an unsigned 64-bit integer")
    return hex(parsed)


@dataclasses.dataclass
class TraceSelector:
    """Allowlisted, data-only targeting requested for a follow-up trace."""

    kernel: Optional[KernelId] = None
    launch_id: Optional[int] = None
    warp_id: Optional[int] = None
    warpgroup_id: Optional[int] = None
    pc_offsets: List[str] = dataclasses.field(default_factory=list)
    instruction_categories: List[TraceInstructionCategory] = dataclasses.field(
        default_factory=list
    )
    config_source: Optional[TraceConfigSource] = None

    def __post_init__(self) -> None:
        if self.launch_id is not None and not 0 <= self.launch_id <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("trace launch_id must fit an unsigned 64-bit integer")
        if self.warp_id is not None and not 0 <= self.warp_id < 32:
            raise ValueError("trace warp_id must be a CTA-local warp in [0, 31]")
        if self.warpgroup_id is not None and not 0 <= self.warpgroup_id < 8:
            raise ValueError("trace warpgroup_id must be in [0, 7]")
        if self.warp_id is not None and self.warpgroup_id is not None:
            raise ValueError("trace selector cannot combine warp_id and warpgroup_id")
        self.pc_offsets = list(dict.fromkeys(_pc_offset(pc) for pc in self.pc_offsets))
        if len(self.pc_offsets) > 32:
            raise ValueError("trace selector accepts at most 32 PC offsets")
        self.instruction_categories = list(dict.fromkeys(self.instruction_categories))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel": None if self.kernel is None else self.kernel.to_dict(),
            "launch_id": self.launch_id,
            "warp_id": self.warp_id,
            "warpgroup_id": self.warpgroup_id,
            "pc_offsets": list(self.pc_offsets),
            "instruction_categories": [
                category.value for category in self.instruction_categories
            ],
            "config_source": (
                None if self.config_source is None else self.config_source.value
            ),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceSelector":
        kernel = d.get("kernel")
        config_source = d.get("config_source")
        launch_id = d.get("launch_id")
        warp_id = d.get("warp_id")
        warpgroup_id = d.get("warpgroup_id")
        return cls(
            kernel=None if kernel is None else KernelId.from_dict(kernel),
            launch_id=None if launch_id is None else int(launch_id),
            warp_id=None if warp_id is None else int(warp_id),
            warpgroup_id=None if warpgroup_id is None else int(warpgroup_id),
            pc_offsets=[str(pc) for pc in d.get("pc_offsets", [])],
            instruction_categories=[
                TraceInstructionCategory(category)
                for category in d.get("instruction_categories", [])
            ],
            config_source=(
                None if config_source is None else TraceConfigSource(config_source)
            ),
        )


@dataclasses.dataclass
class TraceOptions:
    mode: ExperimentKind
    trace_size_limit_mb: int = 1024
    selector: TraceSelector = dataclasses.field(default_factory=TraceSelector)
    config_artifact: Optional[ArtifactRef] = None
    instrument_modes: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
            raise ValueError(f"invalid trace mode: {self.mode}")
        self.instrument_modes = _trace_instrument_modes(
            self.mode, self.instrument_modes
        )
        if not 1 <= self.trace_size_limit_mb <= 0xFFFFFFFF:
            raise ValueError("trace_size_limit_mb must be in [1, UINT32_MAX]")
        if (self.selector.config_source is None) != (self.config_artifact is None):
            raise ValueError(
                "trace config_source and resolved config_artifact must be set together"
            )

    @property
    def effective_instrument_modes(self) -> List[str]:
        return _trace_instrument_modes(self.mode, self.instrument_modes) or [
            self.mode.value
        ]

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": self.mode.value,
            "trace_size_limit_mb": self.trace_size_limit_mb,
            "selector": self.selector.to_dict(),
            "config_artifact": (
                None if self.config_artifact is None else self.config_artifact.to_dict()
            ),
        }
        if self.instrument_modes:
            payload["instrument_modes"] = list(self.instrument_modes)
        return payload

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceOptions":
        selector = d.get("selector")
        config_artifact = d.get("config_artifact")
        return cls(
            mode=ExperimentKind(d["mode"]),
            trace_size_limit_mb=int(d.get("trace_size_limit_mb", 1024)),
            selector=(
                TraceSelector()
                if selector is None
                else TraceSelector.from_dict(selector)
            ),
            config_artifact=(
                None
                if config_artifact is None
                else ArtifactRef.from_dict(config_artifact)
            ),
            instrument_modes=d.get("instrument_modes", []),
        )


@dataclasses.dataclass
class ReduceOptions:
    triggering_config: TriggeringDelayConfig
    strategy: str = "bisect"
    confidence_runs: int = 3
    require_kernel_identity: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in ("linear", "bisect"):
            raise ValueError(f"unknown reduction strategy: {self.strategy}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggering_config": self.triggering_config.to_dict(),
            "strategy": self.strategy,
            "confidence_runs": self.confidence_runs,
            "require_kernel_identity": self.require_kernel_identity,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReduceOptions":
        return cls(
            triggering_config=TriggeringDelayConfig.from_dict(d["triggering_config"]),
            strategy=d.get("strategy", "bisect"),
            confidence_runs=int(d.get("confidence_runs", 3)),
            require_kernel_identity=_as_bool(d.get("require_kernel_identity", True)),
        )


@dataclasses.dataclass
class ExperimentSpec:
    experiment_id: str
    session_id: str
    phase: ExperimentPhase
    kind: ExperimentKind
    unit: WorkUnit
    round_index: int = 0
    discovery: Optional[KernelDiscoveryOptions] = None
    sanitizer_tool: Optional[str] = None
    stress: Optional[StressOptions] = None
    trace: Optional[TraceOptions] = None
    reduction: Optional[ReduceOptions] = None

    def __post_init__(self) -> None:
        if self.kind == ExperimentKind.KERNEL_DISCOVERY and self.discovery is None:
            raise ValueError("kernel discovery experiment requires discovery options")
        if (
            self.kind == ExperimentKind.KERNEL_DISCOVERY
            and self.unit.kernel is not None
        ):
            raise ValueError("kernel discovery requires an unpinned WorkUnit")
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
            "discovery": (None if self.discovery is None else self.discovery.to_dict()),
            "sanitizer_tool": self.sanitizer_tool,
            "stress": None if self.stress is None else self.stress.to_dict(),
            "trace": None if self.trace is None else self.trace.to_dict(),
            "reduction": (None if self.reduction is None else self.reduction.to_dict()),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentSpec":
        stress = d.get("stress")
        discovery = d.get("discovery")
        trace = d.get("trace")
        reduction = d.get("reduction")
        return cls(
            experiment_id=d["experiment_id"],
            session_id=d["session_id"],
            phase=ExperimentPhase(d["phase"]),
            kind=ExperimentKind(d["kind"]),
            unit=WorkUnit.from_dict(d["unit"]),
            round_index=int(d.get("round_index", 0)),
            discovery=(
                None
                if discovery is None
                else KernelDiscoveryOptions.from_dict(discovery)
            ),
            sanitizer_tool=d.get("sanitizer_tool"),
            stress=None if stress is None else StressOptions.from_dict(stress),
            trace=None if trace is None else TraceOptions.from_dict(trace),
            reduction=(
                None if reduction is None else ReduceOptions.from_dict(reduction)
            ),
        )


@dataclasses.dataclass(frozen=True)
class KernelDiscoveryManifest:
    target_unit_id: str
    source_revision: str
    arch: str
    kernels: Tuple[KernelId, ...]
    event_artifacts: Tuple[ArtifactRef, ...]
    target_returncode: int
    kernel_filters: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identities = [(kernel.name, kernel.cubin_hash) for kernel in self.kernels]
        if not self.target_unit_id.strip():
            raise ValueError("kernel discovery manifest requires target_unit_id")
        if not identities or identities != sorted(set(identities)):
            raise ValueError("discovered kernels must be nonempty, unique, and sorted")
        if any(
            not name.strip() or not checksum.strip() for name, checksum in identities
        ):
            raise ValueError("discovered kernel identities must not be empty")
        if not self.event_artifacts:
            raise ValueError("kernel discovery manifest requires event artifacts")
        if any(
            not _is_visible_artifact_basename(artifact)
            for artifact in self.event_artifacts
        ):
            raise ValueError(
                "kernel discovery event artifacts require visible basenames"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_unit_id": self.target_unit_id,
            "source_revision": self.source_revision,
            "arch": self.arch,
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "event_artifacts": [
                artifact.to_dict() for artifact in self.event_artifacts
            ],
            "target_returncode": self.target_returncode,
            "kernel_filters": list(self.kernel_filters),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelDiscoveryManifest":
        return cls(
            target_unit_id=d["target_unit_id"],
            source_revision=d.get("source_revision", ""),
            arch=d.get("arch", ""),
            kernels=tuple(KernelId.from_dict(item) for item in d.get("kernels", [])),
            event_artifacts=tuple(
                ArtifactRef.from_dict(item) for item in d.get("event_artifacts", [])
            ),
            target_returncode=int(d.get("target_returncode", 0)),
            kernel_filters=tuple(d.get("kernel_filters", [])),
        )


@dataclasses.dataclass
class KernelDiscoveryResult:
    experiment_id: str
    execution_status: ExecutionStatus
    outcome: KernelDiscoveryOutcome
    provenance: ExecutionProvenance
    discovery: Optional[KernelDiscoveryManifest] = None
    manifest_artifact: Optional[ArtifactRef] = None
    duration_s: float = 0.0
    error: Optional[str] = None

    kind: ExperimentKind = dataclasses.field(
        default=ExperimentKind.KERNEL_DISCOVERY, init=False
    )

    def __post_init__(self) -> None:
        if self.outcome == KernelDiscoveryOutcome.DISCOVERED:
            if (
                self.execution_status != ExecutionStatus.SUCCEEDED
                or self.discovery is None
                or self.manifest_artifact is None
                or self.error is not None
            ):
                raise ValueError(
                    "successful kernel discovery requires kernels and artifacts"
                )
            if not _is_visible_artifact_basename(self.manifest_artifact):
                raise ValueError(
                    "kernel discovery manifest artifact requires a visible basename"
                )
            return
        if self.execution_status == ExecutionStatus.SUCCEEDED:
            raise ValueError("unsuccessful kernel discovery cannot succeed")
        if self.discovery is not None or self.manifest_artifact is not None:
            raise ValueError("failed kernel discovery cannot publish partial evidence")
        if self.error is None or not self.error.strip():
            raise ValueError("failed kernel discovery requires an error")
        expected_statuses = {
            KernelDiscoveryOutcome.NO_KERNELS: (ExecutionStatus.FAILED,),
            KernelDiscoveryOutcome.NO_MATCHING_KERNELS: (ExecutionStatus.FAILED,),
            KernelDiscoveryOutcome.TOO_MANY_KERNELS: (ExecutionStatus.FAILED,),
            KernelDiscoveryOutcome.INFRA_ERROR: (
                ExecutionStatus.INFRA_ERROR,
                ExecutionStatus.TIMED_OUT,
            ),
        }
        if self.execution_status not in expected_statuses[self.outcome]:
            raise ValueError(
                "kernel discovery outcome does not match its execution status"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "experiment_id": self.experiment_id,
            "execution_status": self.execution_status.value,
            "outcome": self.outcome.value,
            "provenance": self.provenance.to_dict(),
            "discovery": None if self.discovery is None else self.discovery.to_dict(),
            "manifest_artifact": (
                None
                if self.manifest_artifact is None
                else self.manifest_artifact.to_dict()
            ),
            "duration_s": self.duration_s,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelDiscoveryResult":
        discovery = d.get("discovery")
        manifest_artifact = d.get("manifest_artifact")
        return cls(
            experiment_id=d["experiment_id"],
            execution_status=ExecutionStatus(d["execution_status"]),
            outcome=KernelDiscoveryOutcome(d["outcome"]),
            provenance=ExecutionProvenance.from_dict(d["provenance"]),
            discovery=(
                None
                if discovery is None
                else KernelDiscoveryManifest.from_dict(discovery)
            ),
            manifest_artifact=(
                None
                if manifest_artifact is None
                else ArtifactRef.from_dict(manifest_artifact)
            ),
            duration_s=float(d.get("duration_s", 0.0)),
            error=d.get("error"),
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
    summary: Optional[SanitizerSummary] = None
    completeness: ResultCompleteness = ResultCompleteness.UNKNOWN
    summary_issues: List[str] = dataclasses.field(default_factory=list)
    output_truncated: bool = False

    kind: ExperimentKind = dataclasses.field(
        default=ExperimentKind.COMPUTE_SANITIZER, init=False
    )

    @property
    def has_positive_signal(self) -> bool:
        return (
            self.outcome == SanitizerOutcome.FINDING
            or bool(self.findings)
            or (self.summary is not None and self.summary.is_positive)
        )

    @property
    def has_usable_positive_signal(self) -> bool:
        """Retain parsed findings, but do not promote a broken tool's footer."""
        return bool(self.findings) or (
            self.execution_status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED)
            and self.completeness == ResultCompleteness.COMPLETE
            and not self.summary_issues
            and self.has_positive_signal
        )

    @property
    def has_trustworthy_clean(self) -> bool:
        """Whether this run proves its tool found nothing.

        Every tool that reports a parseable summary can prove a clean run; the
        check is deliberately not restricted to one tool, because gating on
        racecheck alone would leave a complete clean initcheck, memcheck, or
        synccheck run indistinguishable from one that was never parsed.
        """
        return (
            self.execution_status == ExecutionStatus.SUCCEEDED
            and self.tool in _SUPPORTED_SANITIZER_TOOLS
            and self.outcome == SanitizerOutcome.CLEAN
            and self.completeness == ResultCompleteness.COMPLETE
            and self.summary is not None
            and self.summary.is_zero
            and not self.summary_issues
            and not self.findings
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
            "summary": None if self.summary is None else self.summary.to_dict(),
            "completeness": self.completeness.value,
            "summary_issues": list(self.summary_issues),
            "output_truncated": self.output_truncated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SanitizerSweepResult":
        log = d.get("log")
        summary = d.get("summary")
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
            summary=(None if summary is None else SanitizerSummary.from_dict(summary)),
            completeness=ResultCompleteness(
                d.get("completeness", ResultCompleteness.UNKNOWN.value)
            ),
            summary_issues=list(d.get("summary_issues", [])),
            output_truncated=bool(d.get("output_truncated", False)),
        )


MAX_REJECTED_STRESS_CONFIGS = MAX_REJECTED_CONFIGS


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
    planned_trials: int = 0
    timed_out_trials: int = 0
    completeness: ResultCompleteness = ResultCompleteness.UNKNOWN
    unattributed_reproductions: int = 0
    # Delay configs of attempts rejected by validation. Diagnostic only: these
    # are NOT triggering configs and must never be replayed as evidence. They
    # exist because a rejection is otherwise unexplainable from the published
    # artifacts of a remote worker.
    rejected_configs: List[ArtifactRef] = dataclasses.field(default_factory=list)

    kind: ExperimentKind = dataclasses.field(
        default=ExperimentKind.RANDOM_DELAY_STRESS, init=False
    )

    def __post_init__(self) -> None:
        counts = (
            self.completed_trials,
            self.reproductions,
            self.infra_errors,
            self.planned_trials,
            self.timed_out_trials,
            self.unattributed_reproductions,
        )
        if any(count < 0 for count in counts):
            raise ValueError("stress result counts must not be negative")
        # Attributed and unattributed reproductions are disjoint outcomes of a
        # completed oracle trial, so their union cannot exceed that trial set.
        if self.reproductions + self.unattributed_reproductions > self.completed_trials:
            raise ValueError(
                "attributed and unattributed reproductions cannot exceed "
                "completed_trials"
            )
        if len(self.rejected_configs) > MAX_REJECTED_STRESS_CONFIGS:
            raise ValueError("stress result contains too many rejected configs")

    @property
    def reproduction_rate(self) -> float:
        return _derived_reproduction_rate(self.completed_trials, self.reproductions)

    @property
    def has_positive_signal(self) -> bool:
        return (
            self.outcome
            in (
                StressOutcome.REPRODUCED,
                StressOutcome.UNATTRIBUTED_REPRODUCTION,
            )
            or self.reproductions > 0
            or self.unattributed_reproductions > 0
            or self.triggering_config is not None
        )

    @property
    def has_trustworthy_non_reproduction(self) -> bool:
        return (
            self.execution_status == ExecutionStatus.SUCCEEDED
            and self.outcome == StressOutcome.NOT_REPRODUCED
            and self.completeness == ResultCompleteness.COMPLETE
            and self.planned_trials > 0
            and self.completed_trials == self.planned_trials
            and self.reproductions == 0
            and self.infra_errors == 0
            and self.timed_out_trials == 0
            and self.unattributed_reproductions == 0
            and self.triggering_config is None
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
            "reproduction_rate": self.reproduction_rate,
            "provenance": self.provenance.to_dict(),
            "triggering_config": (
                None
                if self.triggering_config is None
                else self.triggering_config.to_dict()
            ),
            "log": None if self.log is None else self.log.to_dict(),
            "rejected_configs": [x.to_dict() for x in self.rejected_configs],
            "duration_s": self.duration_s,
            "error": self.error,
            "planned_trials": self.planned_trials,
            "timed_out_trials": self.timed_out_trials,
            "completeness": self.completeness.value,
            "unattributed_reproductions": self.unattributed_reproductions,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StressTestResult":
        config = d.get("triggering_config")
        log = d.get("log")
        result = cls(
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
            rejected_configs=[
                ArtifactRef.from_dict(x) for x in d.get("rejected_configs", [])
            ],
            duration_s=float(d.get("duration_s", 0.0)),
            error=d.get("error"),
            planned_trials=int(d.get("planned_trials", 0)),
            timed_out_trials=int(d.get("timed_out_trials", 0)),
            completeness=ResultCompleteness(
                d.get("completeness", ResultCompleteness.UNKNOWN.value)
            ),
            unattributed_reproductions=int(d.get("unattributed_reproductions", 0)),
        )
        if "reproduction_rate" in d and not _reproduction_rates_match(
            float(d["reproduction_rate"]), result.reproduction_rate
        ):
            raise ValueError("reproduction_rate does not match trial counts")
        return result


class TraceAnalysisSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclasses.dataclass(frozen=True)
class TraceDetectorSummary:
    source_tool: str
    sub_component: str
    verdict: str
    severity: TraceAnalysisSeverity
    finding_count: int
    description: str = ""
    trace_file: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.source_tool.strip() or not self.sub_component.strip():
            raise ValueError("detector summary requires structured source identity")
        if self.finding_count < 0:
            raise ValueError("detector finding_count must be non-negative")
        if self.trace_file is not None and not self.trace_file.strip():
            raise ValueError("detector trace_file cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_tool": self.source_tool,
            "sub_component": self.sub_component,
            "verdict": self.verdict,
            "severity": self.severity.value,
            "finding_count": self.finding_count,
            "description": self.description,
            "trace_file": self.trace_file,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceDetectorSummary":
        trace_file = d.get("trace_file")
        return cls(
            source_tool=d["source_tool"],
            sub_component=d["sub_component"],
            verdict=d["verdict"],
            severity=TraceAnalysisSeverity(d["severity"]),
            finding_count=int(d.get("finding_count", 0)),
            description=d.get("description", ""),
            trace_file=None if trace_file is None else str(trace_file),
        )


@dataclasses.dataclass
class TraceAnalysisFinding:
    """Compact detector finding.

    ``warps``, ``pcs``, and ``trace_indices`` are independent sets of relevant
    identifiers, matching the analyzer's generic Finding contract. Only source
    locations are positional: when present, they are parallel to ``pcs``.
    """

    source_tool: str
    sub_component: str
    severity: TraceAnalysisSeverity
    category: str
    description: str = ""
    ctas: List[List[int]] = dataclasses.field(default_factory=list)
    warps: List[int] = dataclasses.field(default_factory=list)
    pcs: List[str] = dataclasses.field(default_factory=list)
    trace_indices: List[int] = dataclasses.field(default_factory=list)
    source_locations: List[Optional[SourceLoc]] = dataclasses.field(
        default_factory=list
    )
    trace_file: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.source_tool.strip() or not self.sub_component.strip():
            raise ValueError("analysis finding requires structured source identity")
        if self.source_locations and len(self.source_locations) != len(self.pcs):
            raise ValueError("source_locations must be empty or parallel to pcs")
        if any(len(cta) != 3 for cta in self.ctas):
            raise ValueError("each CTA coordinate must have exactly three dimensions")
        if self.trace_file is not None and not self.trace_file.strip():
            raise ValueError("analysis finding trace_file cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_tool": self.source_tool,
            "sub_component": self.sub_component,
            "severity": self.severity.value,
            "category": self.category,
            "description": self.description,
            "ctas": [list(cta) for cta in self.ctas],
            "warps": list(self.warps),
            "pcs": list(self.pcs),
            "trace_indices": list(self.trace_indices),
            "source_locations": [
                None if location is None else location.to_dict()
                for location in self.source_locations
            ],
            "trace_file": self.trace_file,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceAnalysisFinding":
        trace_file = d.get("trace_file")
        return cls(
            source_tool=d["source_tool"],
            sub_component=d["sub_component"],
            severity=TraceAnalysisSeverity(d["severity"]),
            category=d["category"],
            description=d.get("description", ""),
            ctas=[[int(value) for value in cta] for cta in d.get("ctas", [])],
            warps=[int(value) for value in d.get("warps", [])],
            pcs=[str(value) for value in d.get("pcs", [])],
            trace_indices=[int(value) for value in d.get("trace_indices", [])],
            source_locations=[
                None if location is None else SourceLoc.from_dict(location)
                for location in d.get("source_locations", [])
            ],
            trace_file=None if trace_file is None else str(trace_file),
        )


TRACE_ANALYSIS_SCHEMA_VERSION = 1


@dataclasses.dataclass
class TraceAnalysisEvidence:
    analyzer: str
    execution_status: ExecutionStatus
    detector_summaries: List[TraceDetectorSummary]
    findings: List[TraceAnalysisFinding]
    total_findings: int
    warnings: List[str] = dataclasses.field(default_factory=list)
    artifact: Optional[ArtifactRef] = None
    error: Optional[str] = None
    schema_version: int = TRACE_ANALYSIS_SCHEMA_VERSION
    capture_summaries: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    omitted_capture_summaries: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != TRACE_ANALYSIS_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported TraceAnalysisEvidence schema_version: {self.schema_version}"
            )
        if not self.analyzer.strip():
            raise ValueError("trace analysis requires analyzer identity")
        if self.total_findings < 0:
            raise ValueError("total_findings must be non-negative")
        if self.total_findings < len(self.findings):
            raise ValueError("total_findings cannot be smaller than sampled findings")

    @property
    def omitted_findings(self) -> int:
        return self.total_findings - len(self.findings)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "analyzer": self.analyzer,
            "execution_status": self.execution_status.value,
            "detector_summaries": [
                summary.to_dict() for summary in self.detector_summaries
            ],
            "findings": [finding.to_dict() for finding in self.findings],
            "total_findings": self.total_findings,
            "warnings": list(self.warnings),
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "error": self.error,
        }
        if self.capture_summaries or self.omitted_capture_summaries:
            payload["capture_summaries"] = [
                dict(item) for item in self.capture_summaries
            ]
            payload["omitted_capture_summaries"] = self.omitted_capture_summaries
        return payload

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceAnalysisEvidence":
        if "schema_version" not in d:
            raise ValueError("TraceAnalysisEvidence requires schema_version")
        if "total_findings" not in d:
            raise ValueError("TraceAnalysisEvidence requires total_findings")
        artifact = d.get("artifact")
        return cls(
            analyzer=d["analyzer"],
            execution_status=ExecutionStatus(d["execution_status"]),
            detector_summaries=[
                TraceDetectorSummary.from_dict(summary)
                for summary in d.get("detector_summaries", [])
            ],
            findings=[
                TraceAnalysisFinding.from_dict(finding)
                for finding in d.get("findings", [])
            ],
            total_findings=int(d["total_findings"]),
            warnings=[str(warning) for warning in d.get("warnings", [])],
            artifact=None if artifact is None else ArtifactRef.from_dict(artifact),
            error=d.get("error"),
            schema_version=d["schema_version"],
            capture_summaries=[dict(item) for item in d.get("capture_summaries", [])],
            omitted_capture_summaries=int(d.get("omitted_capture_summaries", 0)),
        )


@dataclasses.dataclass
class TraceExperimentResult:
    """Trace capture result with a legacy primary-trace compatibility alias.

    ``traces`` is the authoritative ordered set of raw trace artifacts. ``trace``
    remains on the wire as an alias for its first entry so older consumers can
    keep reading a singleton result. Exact duplicates are removed without
    changing producer order. Supporting artifacts are preserved as metadata but
    are not raw trace inputs to the reasoner.
    """

    experiment_id: str
    execution_status: ExecutionStatus
    mode: ExperimentKind
    provenance: ExecutionProvenance
    trace: Optional[ArtifactRef] = None
    log: Optional[ArtifactRef] = None
    analysis: Optional[TraceAnalysisEvidence] = None
    duration_s: float = 0.0
    error: Optional[str] = None
    traces: List[ArtifactRef] = dataclasses.field(default_factory=list)
    supporting_artifacts: List[ArtifactRef] = dataclasses.field(default_factory=list)
    completeness: ResultCompleteness = ResultCompleteness.UNKNOWN
    analysis_duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
            raise ValueError(f"invalid trace result mode: {self.mode}")
        if self.trace is not None and self.traces and self.trace != self.traces[0]:
            raise ValueError("primary trace must equal traces[0]")

        ordered_traces = ([self.trace] if self.trace is not None else []) + self.traces
        self.traces = _deduplicate_artifact_refs(ordered_traces)
        # Trace identity in analysis is the relative_path, so distinct refs
        # sharing one name are ambiguous and must fail closed at construction
        # (matching session validation), not survive dedup to fail later.
        trace_names = [artifact.relative_path for artifact in self.traces]
        if len(set(trace_names)) != len(trace_names):
            raise ValueError("trace result contains duplicate trace artifact names")
        self.trace = self.traces[0] if self.traces else None

        supporting_artifacts = _deduplicate_artifact_refs(self.supporting_artifacts)
        if any(artifact in self.traces for artifact in supporting_artifacts):
            raise ValueError("raw traces and supporting artifacts cannot overlap")
        self.supporting_artifacts = supporting_artifacts

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
            "traces": [trace.to_dict() for trace in self.traces],
            "supporting_artifacts": [
                artifact.to_dict() for artifact in self.supporting_artifacts
            ],
            "log": None if self.log is None else self.log.to_dict(),
            "analysis": None if self.analysis is None else self.analysis.to_dict(),
            "duration_s": self.duration_s,
            "analysis_duration_s": self.analysis_duration_s,
            "error": self.error,
            "completeness": self.completeness.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceExperimentResult":
        trace = d.get("trace")
        traces = d.get("traces") or []
        log = d.get("log")
        analysis = d.get("analysis")
        return cls(
            experiment_id=d["experiment_id"],
            execution_status=ExecutionStatus(d["execution_status"]),
            mode=ExperimentKind(d.get("mode", d["kind"])),
            provenance=ExecutionProvenance.from_dict(d["provenance"]),
            trace=None if trace is None else ArtifactRef.from_dict(trace),
            log=None if log is None else ArtifactRef.from_dict(log),
            analysis=(
                None if analysis is None else TraceAnalysisEvidence.from_dict(analysis)
            ),
            duration_s=float(d.get("duration_s", 0.0)),
            analysis_duration_s=float(d.get("analysis_duration_s", 0.0)),
            error=d.get("error"),
            traces=[ArtifactRef.from_dict(item) for item in traces],
            supporting_artifacts=[
                ArtifactRef.from_dict(item)
                for item in (d.get("supporting_artifacts") or [])
            ],
            completeness=ResultCompleteness(
                d.get("completeness", ResultCompleteness.UNKNOWN.value)
            ),
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
    essential_points_artifact: Optional[ArtifactRef] = None
    essential_points: List[EssentialDelayPoint] = dataclasses.field(
        default_factory=list
    )
    duration_s: float = 0.0
    error: Optional[str] = None
    # Free-form advisory note on a successful reduction (e.g. a 0-essential-point
    # result that reproduces without any injected delay). Distinct from ``error``,
    # which is reserved for failures.
    note: str = ""

    kind: ExperimentKind = dataclasses.field(
        default=ExperimentKind.REDUCE_DELAY_CONFIG, init=False
    )

    def artifact_refs(self) -> List[ArtifactRef]:
        """Return the typed reduction artifacts, with report/config priority."""
        return [
            artifact
            for artifact in (
                self.report,
                self.minimized_config,
                self.essential_points_artifact,
                self.input_config.artifact,
            )
            if artifact is not None
        ]

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
            "essential_points_artifact": (
                None
                if self.essential_points_artifact is None
                else self.essential_points_artifact.to_dict()
            ),
            "essential_points": [x.to_dict() for x in self.essential_points],
            "duration_s": self.duration_s,
            "error": self.error,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReduceExperimentResult":
        minimized = d.get("minimized_config")
        report = d.get("report")
        essential_points_artifact = d.get("essential_points_artifact")
        return cls(
            experiment_id=d["experiment_id"],
            execution_status=ExecutionStatus(d["execution_status"]),
            outcome=ReduceOutcome.from_wire(d["outcome"]),
            provenance=ExecutionProvenance.from_dict(d["provenance"]),
            input_config=TriggeringDelayConfig.from_dict(d["input_config"]),
            minimized_config=(
                None if minimized is None else ArtifactRef.from_dict(minimized)
            ),
            report=None if report is None else ArtifactRef.from_dict(report),
            essential_points_artifact=(
                None
                if essential_points_artifact is None
                else ArtifactRef.from_dict(essential_points_artifact)
            ),
            essential_points=[
                EssentialDelayPoint.from_dict(x) for x in d.get("essential_points", [])
            ],
            duration_s=float(d.get("duration_s", 0.0)),
            error=d.get("error"),
            note=d.get("note", ""),
        )


ExperimentResult = Union[
    KernelDiscoveryResult,
    SanitizerSweepResult,
    StressTestResult,
    TraceExperimentResult,
    ReduceExperimentResult,
]


def experiment_result_from_dict(d: Dict[str, Any]) -> ExperimentResult:
    kind = ExperimentKind(d["kind"])
    if kind == ExperimentKind.KERNEL_DISCOVERY:
        return KernelDiscoveryResult.from_dict(d)
    if kind == ExperimentKind.COMPUTE_SANITIZER:
        return SanitizerSweepResult.from_dict(d)
    if kind == ExperimentKind.RANDOM_DELAY_STRESS:
        return StressTestResult.from_dict(d)
    if kind in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
        return TraceExperimentResult.from_dict(d)
    if kind == ExperimentKind.REDUCE_DELAY_CONFIG:
        return ReduceExperimentResult.from_dict(d)
    raise ValueError(f"unknown experiment kind: {kind}")


EXPERIMENT_ENVELOPE_SCHEMA_VERSION = 1


def experiment_dispatch_key(
    session_id: str,
    experiment_id: str,
    analyzed_state_revision: int,
) -> str:
    """Return the stable idempotency key for one analyzed-state dispatch."""
    identity = {
        "analyzed_state_revision": analyzed_state_revision,
        "experiment_id": experiment_id,
        "session_id": session_id,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return f"dispatch-{hashlib.sha256(encoded).hexdigest()}"


@dataclasses.dataclass(frozen=True)
class ExperimentDispatchBinding:
    """Immutable callback identity issued with an experiment dispatch."""

    session_id: str
    experiment_id: str
    analyzed_state_revision: int
    dispatch_key: str

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("experiment binding requires session_id")
        if not self.experiment_id.strip():
            raise ValueError("experiment binding requires experiment_id")
        if type(self.analyzed_state_revision) is not int:
            raise ValueError("analyzed_state_revision must be an integer")
        if self.analyzed_state_revision < 0:
            raise ValueError("analyzed_state_revision must be non-negative")
        expected = experiment_dispatch_key(
            self.session_id,
            self.experiment_id,
            self.analyzed_state_revision,
        )
        if self.dispatch_key != expected:
            raise ValueError("dispatch_key does not match experiment binding")

    @classmethod
    def create(
        cls,
        session_id: str,
        experiment_id: str,
        analyzed_state_revision: int,
    ) -> "ExperimentDispatchBinding":
        return cls(
            session_id=session_id,
            experiment_id=experiment_id,
            analyzed_state_revision=analyzed_state_revision,
            dispatch_key=experiment_dispatch_key(
                session_id,
                experiment_id,
                analyzed_state_revision,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "experiment_id": self.experiment_id,
            "analyzed_state_revision": self.analyzed_state_revision,
            "dispatch_key": self.dispatch_key,
        }

    def validate_experiment_identity(
        self,
        session_id: str,
        experiment_id: str,
    ) -> None:
        """Prove that a dispatched callback belongs to one exact spec."""
        if self.session_id != session_id:
            raise ValueError(
                "dispatch binding session_id does not match experiment spec"
            )
        if self.experiment_id != experiment_id:
            raise ValueError(
                "dispatch binding experiment_id does not match experiment spec"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentDispatchBinding":
        return cls(
            session_id=d["session_id"],
            experiment_id=d["experiment_id"],
            analyzed_state_revision=d["analyzed_state_revision"],
            dispatch_key=d["dispatch_key"],
        )


@dataclasses.dataclass(frozen=True)
class ExperimentEnvelope:
    """Versioned wire callback carrying exactly one typed experiment result."""

    session_id: str
    experiment_id: str
    analyzed_state_revision: int
    dispatch_key: str
    result: ExperimentResult
    schema_version: int = EXPERIMENT_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIMENT_ENVELOPE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ExperimentEnvelope schema_version: {self.schema_version}"
            )
        ExperimentDispatchBinding(
            session_id=self.session_id,
            experiment_id=self.experiment_id,
            analyzed_state_revision=self.analyzed_state_revision,
            dispatch_key=self.dispatch_key,
        )
        if self.result.experiment_id != self.experiment_id:
            raise ValueError("envelope result does not match experiment_id")

    @property
    def binding(self) -> ExperimentDispatchBinding:
        return ExperimentDispatchBinding(
            session_id=self.session_id,
            experiment_id=self.experiment_id,
            analyzed_state_revision=self.analyzed_state_revision,
            dispatch_key=self.dispatch_key,
        )

    @classmethod
    def bind(
        cls,
        binding: ExperimentDispatchBinding,
        result: ExperimentResult,
    ) -> "ExperimentEnvelope":
        return cls(
            session_id=binding.session_id,
            experiment_id=binding.experiment_id,
            analyzed_state_revision=binding.analyzed_state_revision,
            dispatch_key=binding.dispatch_key,
            result=result,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "experiment_id": self.experiment_id,
            "analyzed_state_revision": self.analyzed_state_revision,
            "dispatch_key": self.dispatch_key,
            "result": self.result.to_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentEnvelope":
        version = int(d.get("schema_version", 0))
        return cls(
            session_id=d["session_id"],
            experiment_id=d["experiment_id"],
            analyzed_state_revision=d["analyzed_state_revision"],
            dispatch_key=d["dispatch_key"],
            result=experiment_result_from_dict(d["result"]),
            schema_version=version,
        )

    @classmethod
    def from_json(cls, source: Union[str, bytes, TextIO]) -> "ExperimentEnvelope":
        return cls.from_dict(json.loads(_load_source(source)))


@dataclasses.dataclass
class EvidenceBundle:
    discovery: List[KernelDiscoveryResult] = dataclasses.field(default_factory=list)
    sanitizer: List[SanitizerSweepResult] = dataclasses.field(default_factory=list)
    stress: List[StressTestResult] = dataclasses.field(default_factory=list)
    traces: List[TraceExperimentResult] = dataclasses.field(default_factory=list)
    reductions: List[ReduceExperimentResult] = dataclasses.field(default_factory=list)

    def add(self, result: ExperimentResult) -> None:
        if isinstance(result, KernelDiscoveryResult):
            self.discovery.append(result)
        elif isinstance(result, SanitizerSweepResult):
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
            "discovery": [x.to_dict() for x in self.discovery],
            "sanitizer": [x.to_dict() for x in self.sanitizer],
            "stress": [x.to_dict() for x in self.stress],
            "traces": [x.to_dict() for x in self.traces],
            "reductions": [x.to_dict() for x in self.reductions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceBundle":
        return cls(
            discovery=[
                KernelDiscoveryResult.from_dict(x) for x in d.get("discovery", [])
            ],
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
    trace_selector: Optional[TraceSelector] = None
    instrument_modes: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in (
            ExperimentKind.COMPUTE_SANITIZER,
            ExperimentKind.RANDOM_DELAY_STRESS,
            ExperimentKind.REG_TRACE,
            ExperimentKind.MEM_VALUE_TRACE,
            ExperimentKind.REDUCE_DELAY_CONFIG,
        ):
            raise ValueError(f"unsupported experiment request: {self.kind}")
        if self.trace_selector is not None and self.kind not in (
            ExperimentKind.REG_TRACE,
            ExperimentKind.MEM_VALUE_TRACE,
        ):
            raise ValueError("trace_selector is only valid for trace requests")
        if self.instrument_modes and self.kind not in (
            ExperimentKind.REG_TRACE,
            ExperimentKind.MEM_VALUE_TRACE,
        ):
            raise ValueError("instrument_modes is only valid for trace requests")
        self.instrument_modes = _trace_instrument_modes(
            self.kind, self.instrument_modes
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "kind": self.kind.value,
            "rationale": self.rationale,
            "trace_selector": (
                None if self.trace_selector is None else self.trace_selector.to_dict()
            ),
        }
        if self.instrument_modes:
            payload["instrument_modes"] = list(self.instrument_modes)
        return payload

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRequest":
        selector = d.get("trace_selector")
        return cls(
            kind=ExperimentKind(d["kind"]),
            rationale=d.get("rationale", ""),
            trace_selector=(
                None if selector is None else TraceSelector.from_dict(selector)
            ),
            instrument_modes=d.get("instrument_modes", []),
        )


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

    def __post_init__(self) -> None:
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be non-negative")
        if self.max_experiments < 0:
            raise ValueError("max_experiments must be non-negative")

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
    guard_notes: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "decision": self.decision.to_dict(),
            "guard_notes": list(self.guard_notes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisTurn":
        return cls(
            turn_index=int(d["turn_index"]),
            decision=AnalysisDecision.from_dict(d["decision"]),
            guard_notes=list(d.get("guard_notes", [])),
        )


@dataclasses.dataclass
class ExperimentRecord:
    """One line of what the session actually ran, for the report history."""

    experiment_id: str
    kind: ExperimentKind
    phase: ExperimentPhase
    round_index: int
    execution_status: ExecutionStatus
    outcome: str = ""
    completeness: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "kind": self.kind.value,
            "phase": self.phase.value,
            "round_index": self.round_index,
            "execution_status": self.execution_status.value,
            "outcome": self.outcome,
            "completeness": self.completeness,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=d["experiment_id"],
            kind=ExperimentKind(d["kind"]),
            phase=ExperimentPhase(d["phase"]),
            round_index=int(d.get("round_index", 0)),
            execution_status=ExecutionStatus(d["execution_status"]),
            outcome=d.get("outcome", ""),
            completeness=d.get("completeness", ""),
        )


ANALYSIS_SESSION_SCHEMA_VERSION = 7
_MIN_ANALYSIS_SESSION_SCHEMA_VERSION = 3


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
    triage: Optional[TriageDecision] = None
    experiment_history: List["ExperimentRecord"] = dataclasses.field(
        default_factory=list
    )
    discovery_plan: Optional[KernelDiscoveryOptions] = None
    child_sanitizer_tools: List[str] = dataclasses.field(default_factory=list)
    child_enable_stress: bool = True
    child_stress_options: Optional[StressOptions] = None
    child_session_ids: List[str] = dataclasses.field(default_factory=list)
    termination_reason: str = ""
    report_publication: ReportPublicationStatus = ReportPublicationStatus.NOT_READY
    report_publication_error: str = ""
    state_revision: int = 0
    schema_version: int = ANALYSIS_SESSION_SCHEMA_VERSION
    admission_input_digest: Optional[str] = None
    published_report: Optional[ArtifactRef] = None

    @property
    def pending_experiment_ids(self) -> List[str]:
        return [spec.experiment_id for spec in self.pending_experiments]

    @property
    def requires_reconciliation(self) -> bool:
        active = self.status not in (
            SessionStatus.COMPLETED,
            SessionStatus.INCONCLUSIVE,
            SessionStatus.FAILED,
        )
        return active or self.report_publication == ReportPublicationStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        payload = {
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
            "triage": None if self.triage is None else self.triage.to_dict(),
            "experiment_history": [x.to_dict() for x in self.experiment_history],
            "discovery_plan": (
                None if self.discovery_plan is None else self.discovery_plan.to_dict()
            ),
            "child_sanitizer_tools": list(self.child_sanitizer_tools),
            "child_enable_stress": self.child_enable_stress,
            "child_stress_options": (
                None
                if self.child_stress_options is None
                else self.child_stress_options.to_dict()
            ),
            "child_session_ids": list(self.child_session_ids),
            "termination_reason": self.termination_reason,
            "report_publication": self.report_publication.value,
            "report_publication_error": self.report_publication_error,
            "state_revision": self.state_revision,
        }
        if self.admission_input_digest is not None:
            payload["admission_input_digest"] = self.admission_input_digest
        if self.published_report is not None:
            payload["published_report"] = self.published_report.to_dict()
        return payload

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisSession":
        version = int(d.get("schema_version", _MIN_ANALYSIS_SESSION_SCHEMA_VERSION))
        if version < _MIN_ANALYSIS_SESSION_SCHEMA_VERSION:
            raise ValueError(f"unsupported AnalysisSession schema_version: {version}")
        report = d.get("report")
        triage = d.get("triage")
        discovery_plan = d.get("discovery_plan")
        child_stress_options = d.get("child_stress_options")
        status = SessionStatus(d["status"])
        publication = d.get("report_publication")
        if publication is None:
            publication = (
                ReportPublicationStatus.PUBLISHED.value
                if status in (SessionStatus.COMPLETED, SessionStatus.INCONCLUSIVE)
                else ReportPublicationStatus.NOT_READY.value
            )
        return cls(
            session_id=d["session_id"],
            unit=WorkUnit.from_dict(d["unit"]),
            status=status,
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
            triage=None if triage is None else TriageDecision.from_dict(triage),
            experiment_history=[
                ExperimentRecord.from_dict(x) for x in d.get("experiment_history", [])
            ],
            discovery_plan=(
                None
                if discovery_plan is None
                else KernelDiscoveryOptions.from_dict(discovery_plan)
            ),
            child_sanitizer_tools=list(d.get("child_sanitizer_tools", [])),
            child_enable_stress=_as_bool(d.get("child_enable_stress", True)),
            child_stress_options=(
                None
                if child_stress_options is None
                else StressOptions.from_dict(child_stress_options)
            ),
            child_session_ids=list(d.get("child_session_ids", [])),
            termination_reason=d.get("termination_reason", ""),
            report_publication=ReportPublicationStatus(publication),
            report_publication_error=d.get("report_publication_error", ""),
            state_revision=int(d.get("state_revision", 0)),
            schema_version=max(version, ANALYSIS_SESSION_SCHEMA_VERSION),
            admission_input_digest=d.get("admission_input_digest"),
            published_report=(
                None
                if d.get("published_report") is None
                else ArtifactRef.from_dict(d["published_report"])
            ),
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
    unit: Optional[WorkUnit] = None
    triage: Optional[TriageDecision] = None
    experiments: List[ExperimentRecord] = dataclasses.field(default_factory=list)
    artifact_citations: List[str] = dataclasses.field(default_factory=list)
    triggering_configs: List[str] = dataclasses.field(default_factory=list)
    minimized_configs: List[str] = dataclasses.field(default_factory=list)
    rounds_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "explanation": self.explanation.to_dict(),
            "evidence": self.evidence.to_dict(),
            "turns": [x.to_dict() for x in self.turns],
            "termination_reason": self.termination_reason,
            "unit": None if self.unit is None else self.unit.to_dict(),
            "triage": None if self.triage is None else self.triage.to_dict(),
            "experiments": [x.to_dict() for x in self.experiments],
            "artifact_citations": list(self.artifact_citations),
            "triggering_configs": list(self.triggering_configs),
            "minimized_configs": list(self.minimized_configs),
            "rounds_used": self.rounds_used,
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
            unit=None if d.get("unit") is None else WorkUnit.from_dict(d["unit"]),
            triage=(
                None
                if d.get("triage") is None
                else TriageDecision.from_dict(d["triage"])
            ),
            experiments=[
                ExperimentRecord.from_dict(x) for x in d.get("experiments", [])
            ],
            artifact_citations=list(d.get("artifact_citations", [])),
            triggering_configs=list(d.get("triggering_configs", [])),
            minimized_configs=list(d.get("minimized_configs", [])),
            rounds_used=int(d.get("rounds_used", 0)),
        )

    @classmethod
    def from_json(cls, source: Union[str, bytes, TextIO]) -> "SessionReport":
        return cls.from_dict(json.loads(_load_source(source)))


@dataclasses.dataclass(frozen=True)
class ChildDiagnosisReport:
    unit: WorkUnit
    report: SessionReport

    def to_dict(self) -> Dict[str, Any]:
        return {"unit": self.unit.to_dict(), "report": self.report.to_dict()}


@dataclasses.dataclass(frozen=True)
class DiagnosisReport:
    target: WorkUnit
    discovery: KernelDiscoveryResult
    discovery_artifact: ArtifactRef
    children: Tuple[ChildDiagnosisReport, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "discovery": self.discovery.to_dict(),
            "discovery_artifact": self.discovery_artifact.to_dict(),
            "children": [child.to_dict() for child in self.children],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)
