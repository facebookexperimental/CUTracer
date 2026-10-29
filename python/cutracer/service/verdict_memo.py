# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compatibility identity for reusing a previously complete verdict."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from cutracer.service.contracts import (
    AnalysisSession,
    parse_ymd_date,
    ReportPublicationStatus,
    SessionStatus,
    WorkUnit,
)
from cutracer.service.suppression import SuppressionPolicy


VERDICT_MEMO_KEY_SCHEMA_VERSION = 1
VERDICT_MEMO_RECORD_SCHEMA_VERSION = 1


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_runtime_versions(versions: tuple[tuple[str, str], ...]) -> None:
    names = tuple(name for name, _version in versions)
    if (
        not versions
        or versions != tuple(sorted(versions))
        or len(versions) != len(set(versions))
        or len(names) != len(set(names))
        or any(not name.strip() or not version.strip() for name, version in versions)
    ):
        raise ValueError("runtime versions must be non-empty, sorted, and unique")


def suppression_policy_digest(
    policy: SuppressionPolicy, *, today: str | None = None
) -> str:
    """Bind ordered rules and their activity using the same date as triage."""
    current_date = None if today is None else parse_ymd_date(today)
    if today is not None and current_date is None:
        raise ValueError(f"today must be YYYY-MM-DD, got {today!r}")
    rules = []
    for rule in policy.rules:
        if rule.expires_on is None:
            active = True
        elif current_date is None:
            active = False
        else:
            expires = parse_ymd_date(rule.expires_on)
            active = expires is not None and current_date < expires
        rules.append({"rule": rule.to_dict(), "active": active})
    return _canonical_digest({"schema_version": 2, "rules": rules})


def verdict_scope_digest(unit: WorkUnit) -> str:
    """Hash every target property that can change evidence interpretation."""
    payload = unit.to_dict()
    payload.pop("arch", None)
    kernel = payload.pop("kernel", None)
    if isinstance(kernel, dict):
        payload["kernel_name"] = kernel.get("name")
    return _canonical_digest(payload)


@dataclasses.dataclass(frozen=True)
class CompleteVerdictKey:
    cubin_hash: str
    arch: str
    evidence_tools: tuple[str, ...]
    runtime_versions: tuple[tuple[str, str], ...]
    toolchain_version: str
    scope_digest: str
    suppression_policy_digest: str
    schema_version: int = VERDICT_MEMO_KEY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != VERDICT_MEMO_KEY_SCHEMA_VERSION:
            raise ValueError("unsupported verdict memo key schema version")
        scalar_fields = (
            self.cubin_hash,
            self.arch,
            self.toolchain_version,
            self.scope_digest,
            self.suppression_policy_digest,
        )
        if any(not value.strip() for value in scalar_fields):
            raise ValueError("verdict memo key identity fields must not be empty")
        if not self.evidence_tools or self.evidence_tools != tuple(
            sorted(set(self.evidence_tools))
        ):
            raise ValueError("evidence tools must be non-empty, sorted, and unique")
        validate_runtime_versions(self.runtime_versions)

    @classmethod
    def for_unit(
        cls,
        unit: WorkUnit,
        *,
        evidence_tools: Sequence[str],
        runtime_versions: Mapping[str, str],
        toolchain_version: str,
        policy_digest: str,
    ) -> "CompleteVerdictKey":
        kernel = unit.kernel
        if kernel is None or not kernel.cubin_hash.strip():
            raise ValueError("verdict reuse requires a pinned cubin identity")
        return cls(
            cubin_hash=kernel.cubin_hash,
            arch=unit.arch,
            evidence_tools=tuple(sorted(set(evidence_tools))),
            runtime_versions=tuple(sorted(runtime_versions.items())),
            toolchain_version=toolchain_version,
            scope_digest=verdict_scope_digest(unit),
            suppression_policy_digest=policy_digest,
        )

    @property
    def digest(self) -> str:
        return _canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cubin_hash": self.cubin_hash,
            "arch": self.arch,
            "evidence_tools": list(self.evidence_tools),
            "runtime_versions": [list(item) for item in self.runtime_versions],
            "toolchain_version": self.toolchain_version,
            "scope_digest": self.scope_digest,
            "suppression_policy_digest": self.suppression_policy_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompleteVerdictKey":
        required = (
            "schema_version",
            "cubin_hash",
            "arch",
            "evidence_tools",
            "runtime_versions",
            "toolchain_version",
            "scope_digest",
            "suppression_policy_digest",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"verdict memo key is missing {', '.join(missing)}")
        evidence_tools = payload["evidence_tools"]
        if not isinstance(evidence_tools, list) or any(
            not isinstance(tool, str) for tool in evidence_tools
        ):
            raise ValueError("verdict memo evidence_tools must be a list of strings")
        runtime_versions = payload.get("runtime_versions")
        if not isinstance(runtime_versions, list) or any(
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(value, str) for value in item)
            for item in runtime_versions
        ):
            raise ValueError("verdict memo runtime_versions must contain string pairs")
        return cls(
            schema_version=int(payload["schema_version"]),
            cubin_hash=str(payload["cubin_hash"]),
            arch=str(payload["arch"]),
            evidence_tools=tuple(evidence_tools),
            runtime_versions=tuple((item[0], item[1]) for item in runtime_versions),
            toolchain_version=str(payload["toolchain_version"]),
            scope_digest=str(payload["scope_digest"]),
            suppression_policy_digest=str(payload["suppression_policy_digest"]),
        )


@dataclasses.dataclass(frozen=True)
class CompleteVerdictMemo:
    key: CompleteVerdictKey
    source_session_id: str
    source_state_revision: int
    session_digest: str
    report_digest: str
    recorded_at_ms: int
    expires_at_ms: int
    schema_version: int = VERDICT_MEMO_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != VERDICT_MEMO_RECORD_SCHEMA_VERSION:
            raise ValueError("unsupported verdict memo record schema version")
        if not self.source_session_id.strip() or self.source_state_revision <= 0:
            raise ValueError("verdict memo requires a persisted source session")
        for name, digest in (
            ("session", self.session_digest),
            ("report", self.report_digest),
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"verdict memo {name} digest is invalid")
        if self.recorded_at_ms < 0 or self.expires_at_ms <= self.recorded_at_ms:
            raise ValueError("verdict memo lifetime is invalid")

    @classmethod
    def from_complete_session(
        cls,
        key: CompleteVerdictKey,
        session: AnalysisSession,
        *,
        recorded_at_ms: int,
        ttl_ms: int,
    ) -> "CompleteVerdictMemo":
        triage = session.triage
        if (
            session.status != SessionStatus.COMPLETED
            or session.report is None
            or triage is None
            or triage.snapshot is None
            or not triage.snapshot.initial_evidence_complete
            or session.report_publication != ReportPublicationStatus.PUBLISHED
        ):
            raise ValueError("only a published complete verdict can be memoized")
        kernel = session.unit.kernel
        identity = (
            None if kernel is None else kernel.cubin_hash,
            session.unit.arch,
            verdict_scope_digest(session.unit),
        )
        if identity != (key.cubin_hash, key.arch, key.scope_digest):
            raise ValueError("verdict memo key does not match the source session")
        if ttl_ms <= 0:
            raise ValueError("verdict memo TTL must be positive")
        return cls(
            key=key,
            source_session_id=session.session_id,
            source_state_revision=session.state_revision,
            session_digest=_canonical_digest(session.to_dict()),
            report_digest=_canonical_digest(session.report.to_dict()),
            recorded_at_ms=recorded_at_ms,
            expires_at_ms=recorded_at_ms + ttl_ms,
        )

    def is_active_at(self, timestamp_ms: int) -> bool:
        return self.recorded_at_ms <= timestamp_ms < self.expires_at_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "key": self.key.to_dict(),
            "source_session_id": self.source_session_id,
            "source_state_revision": self.source_state_revision,
            "session_digest": self.session_digest,
            "report_digest": self.report_digest,
            "recorded_at_ms": self.recorded_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompleteVerdictMemo":
        try:
            key = payload["key"]
            if not isinstance(key, dict):
                raise ValueError("verdict memo key must be an object")
            return cls(
                schema_version=int(payload["schema_version"]),
                key=CompleteVerdictKey.from_dict(key),
                source_session_id=str(payload["source_session_id"]),
                source_state_revision=int(payload["source_state_revision"]),
                session_digest=str(payload["session_digest"]),
                report_digest=str(payload["report_digest"]),
                recorded_at_ms=int(payload["recorded_at_ms"]),
                expires_at_ms=int(payload["expires_at_ms"]),
            )
        except KeyError as error:
            raise ValueError(f"verdict memo is missing {error.args[0]}") from error
        except TypeError as error:
            raise ValueError(
                f"verdict memo field has the wrong type: {error}"
            ) from error
