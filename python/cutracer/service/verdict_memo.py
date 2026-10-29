# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compatibility identity for reusing a previously complete verdict."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from cutracer.service.contracts import parse_ymd_date, WorkUnit
from cutracer.service.suppression import SuppressionPolicy


VERDICT_MEMO_KEY_SCHEMA_VERSION = 1


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
        if not self.runtime_versions or self.runtime_versions != tuple(
            sorted(set(self.runtime_versions))
        ):
            raise ValueError("runtime versions must be non-empty, sorted, and unique")
        runtime_names = tuple(name for name, _version in self.runtime_versions)
        if len(runtime_names) != len(set(runtime_names)):
            raise ValueError("runtime version names must be unique")
        if any(
            not name.strip() or not version.strip()
            for name, version in self.runtime_versions
        ):
            raise ValueError("runtime version names and values must not be empty")

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
