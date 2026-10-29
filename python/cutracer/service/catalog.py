"""Versioned workload-catalog contracts for scheduled diagnosis."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from enum import Enum
from typing import Any, Mapping

from cutracer.service.contracts import OracleSpec


WORKLOAD_CATALOG_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARCHITECTURE = re.compile(r"^sm_[0-9]+a?$")


def _strings(payload: Mapping[str, Any], field: str) -> tuple[str, ...]:
    values = payload.get(field)
    if not isinstance(values, list) or any(
        not isinstance(item, str) for item in values
    ):
        raise ValueError(f"catalog {field} must be a string array")
    return tuple(values)


def _positive_integer(payload: Mapping[str, Any], field: str, default: int) -> int:
    value = payload.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"catalog {field} must be a positive integer")
    return value


def _text(payload: Mapping[str, Any], field: str, *, default: str | None = None) -> str:
    value = payload.get(field, default)
    if not isinstance(value, str):
        raise ValueError(f"catalog {field} must be a string")
    return value


class CatalogAdapter(str, Enum):
    TRITON_PYTEST = "triton_pytest"
    TLX_PYTEST = "tlx_pytest"
    TRITONBENCH = "tritonbench"


@dataclasses.dataclass(frozen=True)
class CatalogSamplingPolicy:
    max_selected: int | None = None
    seed: str = "default"

    def __post_init__(self) -> None:
        if self.max_selected is not None and (
            isinstance(self.max_selected, bool)
            or not isinstance(self.max_selected, int)
            or self.max_selected <= 0
        ):
            raise ValueError("catalog sample limit must be positive")
        if not self.seed.strip():
            raise ValueError("catalog sampling seed must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogSamplingPolicy:
        limit = payload.get("max_selected")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int)
        ):
            raise ValueError("catalog max_selected must be an integer or null")
        return cls(max_selected=limit, seed=_text(payload, "seed", default="default"))


@dataclasses.dataclass(frozen=True)
class WorkloadCatalogEntry:
    entry_id: str
    adapter: CatalogAdapter
    selectors: tuple[str, ...]
    architectures: tuple[str, ...]
    source_revision: str
    scope_id: str
    cwd: str = ""
    extra_args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    oracle: OracleSpec | None = None
    kernel_filters: tuple[str, ...] = ()
    sampling: CatalogSamplingPolicy = CatalogSamplingPolicy()
    cadence_days: int = 1
    estimated_gpu_seconds: int = 900
    enabled: bool = True

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.entry_id) is None or not self.scope_id.strip():
            raise ValueError("catalog entry requires safe entry and scope identities")
        if not self.selectors or any(not value.strip() for value in self.selectors):
            raise ValueError("catalog entry requires nonempty selectors")
        if len(set(self.selectors)) != len(self.selectors):
            raise ValueError("catalog entry selectors must be unique")
        if not self.architectures or any(
            _ARCHITECTURE.fullmatch(value) is None for value in self.architectures
        ):
            raise ValueError("catalog entry requires valid architectures")
        if len(set(self.architectures)) != len(self.architectures):
            raise ValueError("catalog entry architectures must be unique")
        revision = self.source_revision.strip()
        if not revision or revision == "0+unknown":
            raise ValueError("catalog entry requires an immutable source revision")
        portable_cwd = self.cwd.replace("\\", "/")
        if self.cwd and (
            os.path.isabs(portable_cwd)
            or re.match(r"^[A-Za-z]:/", portable_cwd) is not None
            or ".." in portable_cwd.split("/")
        ):
            raise ValueError("catalog entry cwd must be repository-relative")
        if len(set(self.kernel_filters)) != len(self.kernel_filters) or any(
            not item.strip() or "," in item for item in self.kernel_filters
        ):
            raise ValueError(
                "catalog kernel filters must be nonempty, unique, and comma-free"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.cadence_days, self.estimated_gpu_seconds)
        ):
            raise ValueError(
                "catalog cadence and estimated GPU seconds must be positive"
            )
        environment_keys = [key for key, _value in self.env]
        if len(set(environment_keys)) != len(environment_keys) or any(
            not key or "=" in key for key in environment_keys
        ):
            raise ValueError("catalog environment keys must be unique and valid")
        object.__setattr__(self, "env", tuple(sorted(self.env)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "adapter": self.adapter.value,
            "selectors": list(self.selectors),
            "architectures": list(self.architectures),
            "source_revision": self.source_revision,
            "scope_id": self.scope_id,
            "cwd": self.cwd,
            "extra_args": list(self.extra_args),
            "env": dict(self.env),
            "oracle": None if self.oracle is None else self.oracle.to_dict(),
            "kernel_filters": list(self.kernel_filters),
            "sampling": self.sampling.to_dict(),
            "cadence_days": self.cadence_days,
            "estimated_gpu_seconds": self.estimated_gpu_seconds,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkloadCatalogEntry:
        oracle = payload.get("oracle")
        environment = payload.get("env", {})
        if not isinstance(environment, Mapping):
            raise ValueError("catalog entry env must be an object")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError("catalog entry env values must be strings")
        if oracle is not None and not isinstance(oracle, Mapping):
            raise ValueError("catalog entry oracle must be an object or null")
        sampling = payload.get("sampling", {})
        if not isinstance(sampling, Mapping):
            raise ValueError("catalog entry sampling must be an object")
        enabled = payload.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("catalog entry enabled must be boolean")
        return cls(
            entry_id=_text(payload, "entry_id"),
            adapter=CatalogAdapter(payload["adapter"]),
            selectors=_strings(payload, "selectors"),
            architectures=_strings(payload, "architectures"),
            source_revision=_text(payload, "source_revision"),
            scope_id=_text(payload, "scope_id"),
            cwd=_text(payload, "cwd", default=""),
            extra_args=_strings(
                {"extra_args": payload.get("extra_args", [])}, "extra_args"
            ),
            env=tuple(
                sorted((str(key), str(value)) for key, value in environment.items())
            ),
            oracle=None if oracle is None else OracleSpec.from_dict(dict(oracle)),
            kernel_filters=_strings(
                {"kernel_filters": payload.get("kernel_filters", [])},
                "kernel_filters",
            ),
            sampling=CatalogSamplingPolicy.from_dict(sampling),
            cadence_days=_positive_integer(payload, "cadence_days", 1),
            estimated_gpu_seconds=_positive_integer(
                payload, "estimated_gpu_seconds", 900
            ),
            enabled=enabled,
        )


@dataclasses.dataclass(frozen=True)
class WorkloadCatalog:
    catalog_revision: str
    entries: tuple[WorkloadCatalogEntry, ...]
    schema_version: int = WORKLOAD_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKLOAD_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported workload catalog schema_version: {self.schema_version}"
            )
        if not self.catalog_revision.strip():
            raise ValueError("workload catalog revision must not be empty")
        entry_ids = [entry.entry_id for entry in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("workload catalog entry ids must be unique")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalog_revision": self.catalog_revision,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkloadCatalog:
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("workload catalog requires an entries array")
        if any(not isinstance(item, Mapping) for item in entries):
            raise ValueError("workload catalog entries must be objects")
        schema_version = payload.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("workload catalog requires an integer schema_version")
        return cls(
            catalog_revision=_text(payload, "catalog_revision"),
            entries=tuple(WorkloadCatalogEntry.from_dict(item) for item in entries),
            schema_version=schema_version,
        )

    @classmethod
    def from_json(cls, source: str) -> WorkloadCatalog:
        payload = json.loads(source)
        if not isinstance(payload, dict):
            raise ValueError("workload catalog must be a JSON object")
        return cls.from_dict(payload)


def load_workload_catalog(path: str) -> WorkloadCatalog:
    try:
        with open(path, encoding="utf-8") as source:
            return WorkloadCatalog.from_json(source.read())
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"could not load workload catalog {path!r}: {error}"
        ) from error
