"""Versioned registry of approved diagnosis WorkUnits."""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections.abc import Mapping
from typing import Any

from cutracer.service.contracts import WorkUnit


PRESET_REGISTRY_SCHEMA_VERSION = 1
_PRESET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclasses.dataclass(frozen=True)
class DiagnosisPreset:
    name: str
    unit: WorkUnit
    description: str = ""

    def __post_init__(self) -> None:
        if _PRESET_NAME.fullmatch(self.name) is None:
            raise ValueError(f"invalid diagnosis preset name: {self.name!r}")
        if not self.unit.unit_id.strip() or not self.unit.argv or not self.unit.argv[0]:
            raise ValueError("diagnosis preset requires a runnable WorkUnit")
        revision = self.unit.source_revision.strip()
        if not self.unit.arch.strip() or not revision or revision == "0+unknown":
            raise ValueError(
                "diagnosis preset requires architecture and source identity"
            )
        if self.unit.cwd and (
            os.path.isabs(self.unit.cwd) or ".." in self.unit.cwd.split("/")
        ):
            raise ValueError("diagnosis preset cwd must be repository-relative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DiagnosisPreset:
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            unit=WorkUnit.from_dict(dict(payload["unit"])),
        )


@dataclasses.dataclass(frozen=True)
class DiagnosisPresetRegistry:
    presets: tuple[DiagnosisPreset, ...]
    schema_version: int = PRESET_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != PRESET_REGISTRY_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported preset registry schema_version: {self.schema_version}"
            )
        names = [preset.name for preset in self.presets]
        if len(names) != len(set(names)):
            raise ValueError("diagnosis preset names must be unique")

    def resolve(self, name: str) -> WorkUnit:
        for preset in self.presets:
            if preset.name == name:
                return WorkUnit.from_dict(preset.unit.to_dict())
        raise KeyError(f"unknown diagnosis preset: {name}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DiagnosisPresetRegistry:
        entries = payload.get("presets")
        if not isinstance(entries, list):
            raise ValueError("diagnosis preset registry requires a presets array")
        return cls(
            presets=tuple(DiagnosisPreset.from_dict(item) for item in entries),
            schema_version=payload.get("schema_version", 0),
        )


def load_preset_registry(path: str) -> DiagnosisPresetRegistry:
    try:
        with open(path, encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, dict):
            raise ValueError("preset registry must be a JSON object")
        return DiagnosisPresetRegistry.from_dict(payload)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"could not load preset registry {path!r}: {error}") from error
