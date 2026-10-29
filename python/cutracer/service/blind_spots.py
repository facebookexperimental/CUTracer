"""Versioned inventory of diagnosis blind spots and their mitigations."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import importlib.resources
import json
import re
from enum import Enum
from typing import Any, Mapping

from cutracer.service.contracts import parse_ymd_date, WorkUnit


BLIND_SPOT_CATALOG_SCHEMA_VERSION = 1
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class BlindSpotStatus(str, Enum):
    OPEN = "open"
    MITIGATED = "mitigated"
    ACCEPTED_RISK = "accepted_risk"


def _strings(
    payload: Mapping[str, Any],
    field: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if field not in payload:
        if required:
            raise ValueError(f"blind-spot {field} is required")
        return ()
    value = payload[field]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"blind-spot {field} must be a string array")
    if len(value) != len(set(value)) or any(not item.strip() for item in value):
        raise ValueError(f"blind-spot {field} must be nonempty and unique")
    if required and not value:
        raise ValueError(f"blind-spot {field} must not be empty")
    return tuple(value)


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"blind-spot {field} must be a nonempty string")
    return value


def _boolean(payload: Mapping[str, Any], field: str, default: bool) -> bool:
    value = payload.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"blind-spot {field} must be boolean")
    return value


@dataclasses.dataclass(frozen=True)
class BlindSpotMatcher:
    """AND matcher whose kernel constraint may opt into pre-discovery matching."""

    architectures: tuple[str, ...] = ()
    workloads: tuple[str, ...] = ()
    test_id_patterns: tuple[str, ...] = ()
    kernel_patterns: tuple[str, ...] = ()
    allow_unpinned_kernel: bool = False

    def __post_init__(self) -> None:
        selectors = (
            self.architectures,
            self.workloads,
            self.test_id_patterns,
            self.kernel_patterns,
        )
        if not any(selectors):
            raise ValueError("blind-spot matcher must narrow at least one dimension")
        if self.allow_unpinned_kernel and (
            not self.test_id_patterns or not self.kernel_patterns
        ):
            raise ValueError(
                "allow_unpinned_kernel requires test-id and kernel patterns"
            )
        for pattern in (*self.test_id_patterns, *self.kernel_patterns):
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"invalid blind-spot pattern {pattern!r}: {error}"
                ) from error

    def matches(self, unit: WorkUnit) -> bool:
        if self.architectures and unit.arch not in self.architectures:
            return False
        if self.workloads and unit.workload not in self.workloads:
            return False
        if self.test_id_patterns and not any(
            re.search(pattern, unit.test_id) for pattern in self.test_id_patterns
        ):
            return False
        if not self.kernel_patterns:
            return True
        kernel_names = [*unit.kernel_filters]
        if unit.kernel is not None:
            kernel_names.append(unit.kernel.name)
        if not kernel_names:
            return self.allow_unpinned_kernel
        return any(
            re.search(pattern, name)
            for pattern in self.kernel_patterns
            for name in kernel_names
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "architectures": list(self.architectures),
            "workloads": list(self.workloads),
            "test_id_patterns": list(self.test_id_patterns),
            "kernel_patterns": list(self.kernel_patterns),
            "allow_unpinned_kernel": self.allow_unpinned_kernel,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BlindSpotMatcher:
        return cls(
            architectures=_strings(payload, "architectures"),
            workloads=_strings(payload, "workloads"),
            test_id_patterns=_strings(payload, "test_id_patterns"),
            kernel_patterns=_strings(payload, "kernel_patterns"),
            allow_unpinned_kernel=_boolean(payload, "allow_unpinned_kernel", False),
        )


@dataclasses.dataclass(frozen=True)
class BlindSpotEntry:
    blind_spot_id: str
    version: int
    summary: str
    owner: str
    status: BlindSpotStatus
    matcher: BlindSpotMatcher
    missing_capability: str
    required_evidence: tuple[str, ...]
    oracle: str
    mitigation: str
    exit_criteria: str
    tracking_ticket: str
    last_reviewed_on: str
    review_due_on: str
    _review_due_date: datetime.date = dataclasses.field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not 3 <= len(self.blind_spot_id) <= 128
            or _ID.fullmatch(self.blind_spot_id) is None
        ):
            raise ValueError(f"blind-spot id {self.blind_spot_id!r} is invalid")
        if self.version <= 0:
            raise ValueError(
                f"blind-spot {self.blind_spot_id} version must be positive"
            )
        required = (
            self.summary,
            self.owner,
            self.missing_capability,
            self.oracle,
            self.mitigation,
            self.exit_criteria,
            self.tracking_ticket,
        )
        if any(not value.strip() for value in required) or not self.required_evidence:
            raise ValueError(
                f"blind-spot {self.blind_spot_id} is missing governance data"
            )
        reviewed = parse_ymd_date(self.last_reviewed_on)
        due = parse_ymd_date(self.review_due_on)
        if reviewed is None or due is None or due <= reviewed:
            raise ValueError(
                f"blind-spot {self.blind_spot_id} has an invalid review window"
            )
        object.__setattr__(self, "_review_due_date", due)

    def is_review_overdue(self, today: datetime.date) -> bool:
        return today > self._review_due_date

    def to_dict(self) -> dict[str, object]:
        return {
            "blind_spot_id": self.blind_spot_id,
            "version": self.version,
            "summary": self.summary,
            "owner": self.owner,
            "status": self.status.value,
            "matcher": self.matcher.to_dict(),
            "missing_capability": self.missing_capability,
            "required_evidence": list(self.required_evidence),
            "oracle": self.oracle,
            "mitigation": self.mitigation,
            "exit_criteria": self.exit_criteria,
            "tracking_ticket": self.tracking_ticket,
            "last_reviewed_on": self.last_reviewed_on,
            "review_due_on": self.review_due_on,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BlindSpotEntry:
        matcher = payload.get("matcher")
        if not isinstance(matcher, Mapping):
            raise ValueError("blind-spot matcher must be an object")
        version = payload.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("blind-spot version must be an integer")
        status_value = _required_text(payload, "status")
        try:
            status = BlindSpotStatus(status_value)
        except ValueError as error:
            raise ValueError(
                f"unsupported blind-spot status: {status_value!r}"
            ) from error
        return cls(
            blind_spot_id=_required_text(payload, "blind_spot_id"),
            version=version,
            summary=_required_text(payload, "summary"),
            owner=_required_text(payload, "owner"),
            status=status,
            matcher=BlindSpotMatcher.from_dict(matcher),
            missing_capability=_required_text(payload, "missing_capability"),
            required_evidence=_strings(payload, "required_evidence", required=True),
            oracle=_required_text(payload, "oracle"),
            mitigation=_required_text(payload, "mitigation"),
            exit_criteria=_required_text(payload, "exit_criteria"),
            tracking_ticket=_required_text(payload, "tracking_ticket"),
            last_reviewed_on=_required_text(payload, "last_reviewed_on"),
            review_due_on=_required_text(payload, "review_due_on"),
        )


@dataclasses.dataclass(frozen=True)
class BlindSpotCatalog:
    catalog_revision: str
    entries: tuple[BlindSpotEntry, ...]
    schema_version: int = BLIND_SPOT_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BLIND_SPOT_CATALOG_SCHEMA_VERSION:
            raise ValueError(f"unsupported blind-spot schema: {self.schema_version}")
        if not self.catalog_revision.strip():
            raise ValueError("blind-spot catalog revision must not be empty")
        identities = [entry.blind_spot_id for entry in self.entries]
        if len(identities) != len(set(identities)):
            raise ValueError("blind-spot catalog ids must be unique")

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def matching_entries(
        self,
        unit: WorkUnit,
        *,
        include_mitigated: bool = False,
    ) -> tuple[BlindSpotEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.matcher.matches(unit)
            and (include_mitigated or entry.status != BlindSpotStatus.MITIGATED)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "catalog_revision": self.catalog_revision,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BlindSpotCatalog:
        if "schema_version" not in payload:
            raise ValueError("blind-spot schema_version is required")
        schema_version = payload["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("blind-spot schema_version must be an integer")
        entries = payload.get("entries")
        if not isinstance(entries, list) or any(
            not isinstance(item, Mapping) for item in entries
        ):
            raise ValueError("blind-spot catalog entries must be objects")
        return cls(
            catalog_revision=_required_text(payload, "catalog_revision"),
            entries=tuple(BlindSpotEntry.from_dict(item) for item in entries),
            schema_version=schema_version,
        )

    @classmethod
    def from_json(cls, source: str | bytes) -> BlindSpotCatalog:
        payload = json.loads(source)
        if not isinstance(payload, Mapping):
            raise ValueError("blind-spot catalog must be a JSON object")
        return cls.from_dict(payload)


def load_blind_spot_catalog(path: str) -> BlindSpotCatalog:
    with open(path, encoding="utf-8") as source:
        return BlindSpotCatalog.from_json(source.read())


def default_blind_spot_catalog() -> BlindSpotCatalog:
    # The bundled entries name internal cases and tracking tickets, so the
    # data ships with the fb/ deployment package; elsewhere the default is
    # an empty catalog and callers may inject entries explicitly.
    try:
        resource = importlib.resources.files("cutracer.service.fb").joinpath(
            "blind_spots_v1.json"
        )
        if not resource.is_file():
            return BlindSpotCatalog(catalog_revision="none", entries=())
        with resource.open("r", encoding="utf-8") as source:
            return BlindSpotCatalog.from_json(source.read())
    except ModuleNotFoundError:
        return BlindSpotCatalog(catalog_revision="none", entries=())
