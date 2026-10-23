# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Single-result checkpoint protocol shared by experiment producers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from typing import Any, cast, Generic, Mapping, TypeVar

from cutracer.service.contracts import (
    ExperimentDispatchBinding,
    ExperimentEnvelope,
    ExperimentResult,
)


CHECKPOINT_SCHEMA_VERSION = 1
ENVELOPE_NAME = "experiment_envelope.json"
SUCCESS_NAME = "_SUCCESS"

_ResultT = TypeVar("_ResultT", bound=ExperimentResult)


def _sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("checkpoint envelope_sha256 must be lowercase SHA-256")
    return value


@dataclasses.dataclass(frozen=True)
class TerminalCheckpointMarker:
    """Commit marker for exactly one immutable experiment envelope."""

    dispatch_key: str
    envelope_sha256: str
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema_version: {self.schema_version}"
            )
        if not self.dispatch_key.strip():
            raise ValueError("checkpoint marker requires dispatch_key")
        _sha256(self.envelope_sha256)

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TerminalCheckpointMarker:
        expected = {"schema_version", "dispatch_key", "envelope_sha256"}
        if set(payload) != expected:
            raise ValueError("checkpoint marker has unexpected fields")
        version = payload["schema_version"]
        dispatch_key = payload["dispatch_key"]
        envelope_sha256 = payload["envelope_sha256"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("checkpoint schema_version must be an integer")
        if not isinstance(dispatch_key, str):
            raise ValueError("checkpoint dispatch_key must be a string")
        if not isinstance(envelope_sha256, str):
            raise ValueError("checkpoint envelope_sha256 must be a string")
        return cls(dispatch_key, envelope_sha256, version)


@dataclasses.dataclass(frozen=True)
class TerminalCheckpoint(Generic[_ResultT]):
    result: _ResultT
    envelope: ExperimentEnvelope
    envelope_path: str
    success_path: str


def atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without exposing partial bytes."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = ""
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _parse_json_object(content: bytes | str, path: str) -> dict[str, Any]:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint file is not a JSON object: {path}")
    return payload


def read_json(path: str) -> dict[str, Any]:
    with open(path) as source:
        return _parse_json_object(source.read(), path)


def hash_file(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def load_terminal_checkpoint(
    output_dir: str,
    binding: ExperimentDispatchBinding,
    result_type: type[_ResultT],
) -> TerminalCheckpoint[_ResultT] | None:
    """Load one committed envelope and verify its marker and dispatch binding."""
    success_path = os.path.join(output_dir, SUCCESS_NAME)
    if not os.path.isfile(success_path):
        return None
    if os.path.islink(success_path):
        raise ValueError("checkpoint marker must not be a symbolic link")
    marker = TerminalCheckpointMarker.from_dict(read_json(success_path))
    if marker.dispatch_key != binding.dispatch_key:
        raise ValueError("checkpoint marker has a different dispatch key")

    envelope_path = os.path.join(output_dir, ENVELOPE_NAME)
    if os.path.islink(envelope_path) or not os.path.isfile(envelope_path):
        raise ValueError("checkpoint envelope is unavailable")
    with open(envelope_path, "rb") as source:
        content = source.read()
    if hashlib.sha256(content).hexdigest() != marker.envelope_sha256:
        raise ValueError("checkpoint envelope SHA-256 is inconsistent")
    envelope = ExperimentEnvelope.from_dict(_parse_json_object(content, envelope_path))
    if envelope.binding != binding:
        raise ValueError("checkpoint envelope has a different binding")
    if not isinstance(envelope.result, result_type):
        raise ValueError("checkpoint envelope contains an unexpected result type")
    return TerminalCheckpoint(
        cast(_ResultT, envelope.result),
        envelope,
        envelope_path,
        success_path,
    )


def seal_terminal_checkpoint(
    output_dir: str,
    binding: ExperimentDispatchBinding,
    result: _ResultT,
) -> TerminalCheckpoint[_ResultT]:
    """Persist an envelope and publish its content-bound marker last."""
    os.makedirs(output_dir, exist_ok=True)
    success_path = os.path.join(output_dir, SUCCESS_NAME)
    try:
        os.unlink(success_path)
    except FileNotFoundError:
        pass
    envelope = ExperimentEnvelope.bind(binding, result)
    envelope_path = os.path.join(output_dir, ENVELOPE_NAME)
    atomic_write_json(envelope_path, envelope.to_dict())
    digest, _ = hash_file(envelope_path)
    marker = TerminalCheckpointMarker(binding.dispatch_key, digest)
    atomic_write_json(success_path, marker.to_dict())
    return TerminalCheckpoint(result, envelope, envelope_path, success_path)


def initialize_terminal_checkpoint(
    output_dir: str,
    binding: ExperimentDispatchBinding,
    result_type: type[_ResultT],
) -> TerminalCheckpoint[_ResultT] | None:
    """Return an existing terminal or reset only its uncommitted metadata."""
    completed = load_terminal_checkpoint(output_dir, binding, result_type)
    if completed is not None:
        return completed
    reset_incomplete_terminal_checkpoint(output_dir)
    return None


def reset_incomplete_terminal_checkpoint(output_dir: str) -> None:
    """Remove uncommitted terminal metadata without touching evidence files."""
    os.makedirs(output_dir, exist_ok=True)
    for name in (ENVELOPE_NAME, SUCCESS_NAME):
        try:
            os.unlink(os.path.join(output_dir, name))
        except FileNotFoundError:
            pass
