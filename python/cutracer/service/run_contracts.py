"""Immutable requests and backend-independent control for a diagnosis run."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from enum import Enum
from typing import Any, Mapping, Protocol

from cutracer.service.contracts import (
    ArtifactRef,
    DEFAULT_MAX_DISCOVERED_KERNELS,
    EvidenceSufficiency,
    KernelId,
    RaceClass,
    ReportPublicationStatus,
    SessionBudget,
    SessionReport,
    SessionStatus,
    StressOptions,
    WorkUnit,
)
from cutracer.service.policy import DEFAULT_LOCAL_SANITIZER_TOOLS, InitialCampaignPolicy
from cutracer.service.storage import SessionStoreError, StoreConflict


RUN_SCHEMA_VERSION = 1


class DeploymentProfile(str, Enum):
    LOCAL = "local"
    DEVSERVER_REMOTE = "devserver_remote"
    HOSTED = "hosted"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class RunOutputMode(str, Enum):
    RETURN_ONLY = "return_only"
    FILE = "file"
    REMOTE = "remote"


class RunPublicationStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"


class RunSubmissionState(str, Enum):
    NOT_SUBMITTED = "not_submitted"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    FAILED = "failed"


class RunNotFound(SessionStoreError):
    """No control record exists for the requested run."""


class RunInputConflict(SessionStoreError):
    """An existing run is bound to a different immutable request."""


def canonical_input_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_input_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_input_json(payload).encode()).hexdigest()


def _check_version(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported diagnosis run schema version")


def resolve_campaign_policy(
    policy: InitialCampaignPolicy | None = None,
) -> InitialCampaignPolicy:
    selected = policy or InitialCampaignPolicy(
        sanitizer_tools=DEFAULT_LOCAL_SANITIZER_TOOLS
    )
    return InitialCampaignPolicy(
        sanitizer_tools=tuple(
            tool
            for tool in DEFAULT_LOCAL_SANITIZER_TOOLS
            if tool in selected.sanitizer_tools
        ),
        stress_options=StressOptions.from_dict(selected.stress_options.to_dict()),
        enable_stress=selected.enable_stress,
    )


def _policy_dict(policy: InitialCampaignPolicy) -> dict[str, Any]:
    return {
        "sanitizer_tools": list(policy.sanitizer_tools),
        "stress_options": policy.stress_options.to_dict(),
        "enable_stress": policy.enable_stress,
    }


def _policy_from_dict(payload: Mapping[str, Any]) -> InitialCampaignPolicy:
    return InitialCampaignPolicy(
        sanitizer_tools=tuple(payload["sanitizer_tools"]),
        stress_options=StressOptions.from_dict(payload["stress_options"]),
        enable_stress=payload["enable_stress"],
    )


@dataclasses.dataclass(frozen=True)
class RunOutput:
    mode: RunOutputMode = RunOutputMode.RETURN_ONLY
    path: str = ""

    def __post_init__(self) -> None:
        if (self.mode == RunOutputMode.FILE) != bool(self.path.strip()):
            raise ValueError("only file output requires a report path")
        if self.mode == RunOutputMode.FILE:
            object.__setattr__(self, "path", os.path.abspath(self.path))

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "path": self.path}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunOutput:
        return cls(RunOutputMode(payload["mode"]), payload.get("path", ""))


@dataclasses.dataclass(frozen=True)
class RunInput:
    run_id: str
    profile: DeploymentProfile
    toolchain_version: str
    deadline_ms: int
    _unit_json: str = dataclasses.field(repr=False)
    _policy_json: str = dataclasses.field(repr=False)
    _budget_json: str = dataclasses.field(repr=False)
    max_kernels: int = DEFAULT_MAX_DISCOVERED_KERNELS
    deployment_config_digest: str = ""
    output: RunOutput = dataclasses.field(default_factory=RunOutput)

    def __post_init__(self) -> None:
        if not self.run_id.strip() or len(self.run_id) > 255:
            raise ValueError("run_id must contain between 1 and 255 characters")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id) is None:
            raise ValueError(
                "run_id must be an ASCII identifier using letters, digits, '.', '_' or '-'"
            )
        if not self.toolchain_version.strip() or self.toolchain_version == "0+unknown":
            raise ValueError("run requires an immutable toolchain identity")
        if self.deadline_ms <= 0 or self.max_kernels <= 0:
            raise ValueError("run deadline and max_kernels must be positive")
        if self.profile == DeploymentProfile.LOCAL:
            if self.output.mode == RunOutputMode.REMOTE:
                raise ValueError("local runs cannot request remote publication")
        elif self.output.mode != RunOutputMode.REMOTE:
            raise ValueError("remote profiles require remote publication")
        if self.deployment_config_digest and (
            len(self.deployment_config_digest) != 64
            or any(
                char not in "0123456789abcdef" for char in self.deployment_config_digest
            )
        ):
            raise ValueError("deployment config digest must be lowercase SHA-256")

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        unit: WorkUnit,
        toolchain_version: str,
        deadline_ms: int,
        profile: DeploymentProfile = DeploymentProfile.LOCAL,
        campaign_policy: InitialCampaignPolicy | None = None,
        budget: SessionBudget | None = None,
        max_kernels: int = DEFAULT_MAX_DISCOVERED_KERNELS,
        deployment_config_digest: str = "",
        output: RunOutput | None = None,
    ) -> RunInput:
        selected_output = output or RunOutput(
            RunOutputMode.RETURN_ONLY
            if profile == DeploymentProfile.LOCAL
            else RunOutputMode.REMOTE
        )
        frozen_unit = WorkUnit.from_dict(unit.to_dict())
        if profile == DeploymentProfile.LOCAL:
            frozen_unit.cwd = os.path.abspath(frozen_unit.cwd or os.getcwd())
        return cls(
            run_id=run_id,
            profile=profile,
            toolchain_version=toolchain_version.strip(),
            deadline_ms=deadline_ms,
            _unit_json=canonical_input_json(frozen_unit.to_dict()),
            _policy_json=canonical_input_json(
                _policy_dict(resolve_campaign_policy(campaign_policy))
            ),
            _budget_json=canonical_input_json((budget or SessionBudget()).to_dict()),
            max_kernels=max_kernels,
            deployment_config_digest=deployment_config_digest,
            output=selected_output,
        )

    @property
    def unit(self) -> WorkUnit:
        return WorkUnit.from_dict(json.loads(self._unit_json))

    @property
    def campaign_policy(self) -> InitialCampaignPolicy:
        return _policy_from_dict(json.loads(self._policy_json))

    @property
    def budget(self) -> SessionBudget:
        return SessionBudget.from_dict(json.loads(self._budget_json))

    @property
    def root_session_id(self) -> str:
        return self.run_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "profile": self.profile.value,
            "toolchain_version": self.toolchain_version,
            "deadline_ms": self.deadline_ms,
            "unit": json.loads(self._unit_json),
            "campaign_policy": json.loads(self._policy_json),
            "budget": json.loads(self._budget_json),
            "max_kernels": self.max_kernels,
            "deployment_config_digest": self.deployment_config_digest,
            "output": self.output.to_dict(),
        }

    @property
    def input_digest(self) -> str:
        return canonical_input_digest(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "input_digest": self.input_digest}

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> RunInput:
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunInput:
        _check_version(payload)
        # Reconstruct from the stored payload without re-running create()'s
        # current normalizers: a later change to e.g. the default sanitizer
        # tool set must not make a previously persisted input unreadable.
        result = cls(
            run_id=payload["run_id"],
            profile=DeploymentProfile(payload["profile"]),
            toolchain_version=str(payload["toolchain_version"]).strip(),
            deadline_ms=payload["deadline_ms"],
            _unit_json=canonical_input_json(
                WorkUnit.from_dict(payload["unit"]).to_dict()
            ),
            _policy_json=canonical_input_json(
                _policy_dict(_policy_from_dict(payload["campaign_policy"]))
            ),
            _budget_json=canonical_input_json(
                SessionBudget.from_dict(payload["budget"]).to_dict()
            ),
            max_kernels=payload["max_kernels"],
            deployment_config_digest=payload.get("deployment_config_digest", ""),
            output=RunOutput.from_dict(payload["output"]),
        )
        if result.input_digest != payload.get("input_digest"):
            raise RunInputConflict("run input digest does not match its contents")
        return result


@dataclasses.dataclass(frozen=True)
class RunSessionSummary:
    session_id: str
    unit_id: str
    status: SessionStatus | None
    kernel: KernelId | None = None
    termination_reason: str = ""
    report_publication: ReportPublicationStatus = ReportPublicationStatus.NOT_READY
    report_publication_error: str = ""
    report_uri: str = ""
    report_sha256: str = ""
    has_report: bool = False
    race_class: RaceClass | None = None
    evidence_sufficiency: EvidenceSufficiency | None = None
    initial_evidence_complete: bool = False
    sanitizer_positive_signal: bool = False
    stress_positive_signal: bool = False
    pending_experiment_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["status"] = None if self.status is None else self.status.value
        result["kernel"] = None if self.kernel is None else self.kernel.to_dict()
        result["report_publication"] = self.report_publication.value
        result["race_class"] = (
            None if self.race_class is None else self.race_class.value
        )
        result["evidence_sufficiency"] = (
            None
            if self.evidence_sufficiency is None
            else self.evidence_sufficiency.value
        )
        result["pending_experiment_ids"] = list(self.pending_experiment_ids)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunSessionSummary:
        result = dict(payload)
        result["status"] = (
            None if payload["status"] is None else SessionStatus(payload["status"])
        )
        kernel = payload.get("kernel")
        result["kernel"] = None if kernel is None else KernelId.from_dict(kernel)
        result["report_publication"] = ReportPublicationStatus(
            payload["report_publication"]
        )
        race = payload.get("race_class")
        result["race_class"] = None if race is None else RaceClass.from_wire(race)
        sufficiency = payload.get("evidence_sufficiency")
        result["evidence_sufficiency"] = (
            None if sufficiency is None else EvidenceSufficiency(sufficiency)
        )
        result["pending_experiment_ids"] = tuple(
            payload.get("pending_experiment_ids", ())
        )
        return cls(**result)


@dataclasses.dataclass(frozen=True)
class RunSummary:
    run_id: str
    status: RunStatus
    root_session_id: str
    sessions: tuple[RunSessionSummary, ...]
    publication: RunPublicationStatus
    error: str = ""
    output_uri: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status != RunStatus.RUNNING

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self.status.value,
            "root_session_id": self.root_session_id,
            "sessions": [session.to_dict() for session in self.sessions],
            "publication": self.publication.value,
            "error": self.error,
            "output_uri": self.output_uri,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunSummary:
        _check_version(payload)
        return cls(
            run_id=payload["run_id"],
            status=RunStatus(payload["status"]),
            root_session_id=payload["root_session_id"],
            sessions=tuple(
                RunSessionSummary.from_dict(item) for item in payload["sessions"]
            ),
            publication=RunPublicationStatus(payload["publication"]),
            error=payload.get("error", ""),
            output_uri=payload.get("output_uri", ""),
        )


@dataclasses.dataclass(frozen=True)
class RunResult:
    summary: RunSummary
    reports: tuple[SessionReport, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "reports": [report.to_dict() for report in self.reports],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunResult:
        return cls(
            RunSummary.from_dict(payload["summary"]),
            tuple(SessionReport.from_dict(item) for item in payload["reports"]),
        )


@dataclasses.dataclass(frozen=True)
class RunHandle:
    run_id: str
    profile: DeploymentProfile
    controller_job_id: str = ""
    controller_link: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "profile": self.profile.value}

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclasses.dataclass(frozen=True)
class RunControl:
    revision: int
    input: RunInput
    owner_token: str = ""
    lease_expires_ms: int = 0
    next_poll_ms: int = 0
    controller_job_id: str = ""
    controller_link: str = ""
    controller_input: ArtifactRef | None = None
    controller_launch_digest: str = ""
    submission_state: RunSubmissionState = RunSubmissionState.NOT_SUBMITTED
    submission_token: str = ""
    summary: RunSummary | None = None
    error: str = ""
    backend_identity: str = ""

    def __post_init__(self) -> None:
        if self.revision < 0 or self.lease_expires_ms < 0 or self.next_poll_ms < 0:
            raise ValueError(
                "run control revisions and timestamps must be non-negative"
            )
        if self.summary is not None and self.summary.run_id != self.input.run_id:
            raise ValueError("run summary belongs to a different run")

    @property
    def run_id(self) -> str:
        return self.input.run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "revision": self.revision,
            "input": self.input.to_dict(),
            "owner_token": self.owner_token,
            "lease_expires_ms": self.lease_expires_ms,
            "next_poll_ms": self.next_poll_ms,
            "controller_job_id": self.controller_job_id,
            "controller_link": self.controller_link,
            "controller_input": None
            if self.controller_input is None
            else self.controller_input.to_dict(),
            "controller_launch_digest": self.controller_launch_digest,
            "submission_state": self.submission_state.value,
            "submission_token": self.submission_token,
            "summary": None if self.summary is None else self.summary.to_dict(),
            "error": self.error,
            "backend_identity": self.backend_identity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunControl:
        _check_version(payload)
        summary = payload.get("summary")
        controller_input = payload.get("controller_input")
        return cls(
            revision=payload["revision"],
            input=RunInput.from_dict(payload["input"]),
            owner_token=payload.get("owner_token", ""),
            lease_expires_ms=payload.get("lease_expires_ms", 0),
            next_poll_ms=payload.get("next_poll_ms", 0),
            controller_job_id=payload.get("controller_job_id", ""),
            controller_link=payload.get("controller_link", ""),
            controller_input=None
            if controller_input is None
            else ArtifactRef.from_dict(controller_input),
            controller_launch_digest=payload.get("controller_launch_digest", ""),
            submission_state=RunSubmissionState(
                payload.get("submission_state", "not_submitted")
            ),
            submission_token=payload.get("submission_token", ""),
            summary=None if summary is None else RunSummary.from_dict(summary),
            error=payload.get("error", ""),
            backend_identity=payload.get("backend_identity", ""),
        )


def validate_run_update(
    current: RunControl,
    expected_revision: int,
    updated: RunControl,
) -> None:
    if updated.run_id != current.run_id or updated.revision != expected_revision + 1:
        raise ValueError("replacement Run must keep its ID and advance one revision")
    if current.revision != expected_revision:
        raise StoreConflict(current.run_id)
    if current.input.input_digest != updated.input.input_digest:
        raise RunInputConflict("Run input cannot change during an update")
    for field in (
        "backend_identity",
        "controller_input",
        "controller_launch_digest",
        "submission_token",
    ):
        bound = getattr(current, field)
        if bound and bound != getattr(updated, field):
            raise RunInputConflict(f"Run {field} is already bound")


class RunControlStore(Protocol):
    def create(self, control: RunControl) -> None: ...

    def load(self, run_id: str) -> RunControl: ...

    def compare_and_swap(
        self, run_id: str, expected_revision: int, control: RunControl
    ) -> None: ...
