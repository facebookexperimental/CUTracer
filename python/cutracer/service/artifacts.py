"""Artifact helpers shared by CUTracer service experiment adapters."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlparse

from cutracer.service.contracts import ArtifactRef


@dataclasses.dataclass(frozen=True)
class VerifiedLocalArtifact:
    path: str
    sha256: str
    size_bytes: int


class ArtifactVerificationError(RuntimeError):
    """Base class for typed replay-artifact verification failures."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class ArtifactIdentityError(ArtifactVerificationError):
    """The reference lacks a usable immutable content identity."""


class ArtifactIntegrityError(ArtifactVerificationError):
    """The materialized bytes do not match their declared identity."""


class ArtifactMaterializationError(ArtifactVerificationError):
    """The referenced artifact is not available as a readable local file."""


def local_file_uri(path: str) -> str:
    return Path(os.path.abspath(path)).as_uri()


def _local_uri_paths(uri: str) -> tuple[str, ...]:
    parsed = urlparse(uri)
    if not parsed.scheme:
        return (uri,)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return ()
    decoded_path = unquote(parsed.path)
    raw_path = ""
    if parsed.netloc == "":
        if uri.startswith("file://"):
            raw_path = uri[len("file://") :]
        elif uri.startswith("file:"):
            raw_path = uri[len("file:") :]
    if parsed.params or parsed.query or parsed.fragment:
        # Older producers did not percent-encode reserved path characters.  A
        # file URI is still local when that legacy path has since disappeared;
        # preserving the path lets callers distinguish "missing" from
        # "not local" without probing the filesystem during URI parsing.
        candidates = [decoded_path]
        if raw_path != decoded_path and os.path.isabs(raw_path):
            candidates.append(raw_path)
        return tuple(dict.fromkeys(candidates))
    candidates = [decoded_path]
    if raw_path != decoded_path and os.path.isabs(raw_path):
        candidates.append(raw_path)
    return tuple(dict.fromkeys(candidates))


def _matches_declared_identity(path: str, ref: ArtifactRef) -> bool:
    if ref.sha256 is None or ref.size_bytes is None:
        return False
    try:
        if os.path.getsize(path) != ref.size_bytes:
            return False
        hasher = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError:
        return False
    return hasher.hexdigest().lower() == ref.sha256.lower()


def _local_uri_path(ref: ArtifactRef) -> str | None:
    candidates = _local_uri_paths(ref.uri)
    if not candidates:
        return None
    if len(candidates) > 1:
        has_identity = ref.sha256 is not None and ref.size_bytes is not None
        if has_identity:
            for candidate in candidates:
                if _matches_declared_identity(candidate, ref):
                    return candidate
            if any(os.path.exists(candidate) for candidate in candidates):
                raise ArtifactIntegrityError(
                    "identity_mismatch",
                    f"no local path matches the declared artifact identity: {ref.uri}",
                )
        else:
            existing = [
                candidate for candidate in candidates if os.path.exists(candidate)
            ]
            if len(existing) == 1:
                return existing[0]
            if len(existing) > 1:
                raise ValueError(f"artifact URI has ambiguous local paths: {ref.uri}")
    return candidates[0]


def local_artifact(
    path: str,
    *,
    media_type: str = "text/plain",
    include_sha256: bool = False,
) -> ArtifactRef:
    absolute = os.path.abspath(path)
    size = None
    digest = None
    try:
        if include_sha256:
            hasher = hashlib.sha256()
            with open(absolute, "rb") as fh:
                size = 0
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    hasher.update(chunk)
                    size += len(chunk)
            digest = hasher.hexdigest()
        else:
            size = os.path.getsize(absolute)
    except FileNotFoundError:
        pass
    return ArtifactRef(
        uri=local_file_uri(absolute),
        relative_path=os.path.basename(absolute),
        sha256=digest,
        size_bytes=size,
        media_type=media_type,
    )


def local_path(ref: ArtifactRef) -> str:
    path = _local_uri_path(ref)
    if path is None:
        raise ValueError(f"artifact is not materialized locally: {ref.uri}")
    return path


def _verified_source_path(ref: ArtifactRef) -> str:
    path = _local_uri_path(ref)
    if path is None:
        raise ArtifactMaterializationError(
            "not_local", f"artifact is not materialized locally: {ref.uri}"
        )
    if not os.path.isabs(path):
        raise ArtifactMaterializationError(
            "not_absolute", f"verified artifact path must be absolute: {ref.uri}"
        )
    return path


def _expected_artifact_identity(ref: ArtifactRef) -> tuple[str, int]:
    if ref.sha256 is None or ref.size_bytes is None:
        raise ArtifactIdentityError(
            "identity_missing", "replay artifact requires SHA-256 and size"
        )
    if (
        len(ref.sha256) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in ref.sha256)
        or ref.size_bytes < 0
    ):
        raise ArtifactIdentityError(
            "identity_invalid", "replay artifact has an invalid SHA-256 or size"
        )
    return ref.sha256.lower(), ref.size_bytes


def same_artifact_content(left: ArtifactRef, right: ArtifactRef) -> bool:
    """Compare valid declared content identities independently of location."""
    try:
        return _expected_artifact_identity(left) == _expected_artifact_identity(right)
    except ArtifactIdentityError:
        return False


def _copy_and_hash(source: BinaryIO, destination: BinaryIO) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        hasher.update(chunk)
        size += len(chunk)
        destination.write(chunk)
    return hasher.hexdigest(), size


def _replace_and_sync(temporary: str, destination: str) -> None:
    os.replace(temporary, destination)
    parent_descriptor = os.open(os.path.dirname(destination), os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def publish_local_artifact(
    source: str,
    destination: str,
    *,
    media_type: str = "text/plain",
    content_addressed: bool = False,
) -> ArtifactRef:
    """Atomically publish one file at the stable experiment-output level."""
    destination = os.path.abspath(destination)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary: str | None = None
    descriptor = -1
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(destination)}_",
            dir=parent,
        )
        os.fchmod(descriptor, 0o644)
        with open(source, "rb") as source_file, os.fdopen(descriptor, "wb") as staged:
            descriptor = -1
            digest, size = _copy_and_hash(source_file, staged)
            staged.flush()
            os.fsync(staged.fileno())
        if content_addressed:
            stem, suffix = os.path.splitext(destination)
            destination = f"{stem}_{digest}{suffix}"
        _replace_and_sync(temporary, destination)
        temporary = None
        return ArtifactRef(
            uri=local_file_uri(destination),
            relative_path=os.path.basename(destination),
            sha256=digest,
            size_bytes=size,
            media_type=media_type,
        )
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _copy_to_temporary(source: str, parent: str) -> tuple[str, str, int]:
    temporary: str | None = None
    descriptor = -1
    try:
        source_file = open(source, "rb")
    except FileNotFoundError as exc:
        raise ArtifactMaterializationError(
            "not_found", f"artifact path does not exist: {source}"
        ) from exc
    try:
        with source_file:
            if not stat.S_ISREG(os.fstat(source_file.fileno()).st_mode):
                raise ArtifactMaterializationError(
                    "not_regular", f"artifact is not a regular file: {source}"
                )
            descriptor, temporary = tempfile.mkstemp(
                prefix=".cutracer_verify_", dir=parent
            )
            os.fchmod(descriptor, 0o644)
            with os.fdopen(descriptor, "wb") as staged_file:
                descriptor = -1
                digest, size = _copy_and_hash(source_file, staged_file)
                staged_file.flush()
                os.fsync(staged_file.fileno())
        if temporary is None:
            raise ArtifactMaterializationError(
                "stage_failed", "could not create a temporary artifact"
            )
        completed = temporary
        temporary = None
        return completed, digest, size
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def stage_verified_local_artifact(
    ref: ArtifactRef,
    *,
    destination: str,
) -> VerifiedLocalArtifact:
    """Copy verified bytes to an immutable replay snapshot.

    The digest and byte count are computed from the same stream written to the
    destination snapshot. Replay therefore cannot observe source-file changes
    made after verification.
    """
    expected_digest, expected_size = _expected_artifact_identity(ref)
    source = _verified_source_path(ref)
    destination = os.path.abspath(destination)
    parent = os.path.dirname(destination)
    temporary: str | None = None
    try:
        os.makedirs(parent, exist_ok=True)
        temporary, digest, size = _copy_to_temporary(source, parent)
        if size != expected_size:
            raise ArtifactIntegrityError(
                "size_mismatch",
                f"expected {expected_size} bytes, read {size}",
            )
        if digest != expected_digest:
            raise ArtifactIntegrityError(
                "sha256_mismatch",
                f"expected {expected_digest}, computed {digest}",
            )
        _replace_and_sync(temporary, destination)
        temporary = None
        return VerifiedLocalArtifact(destination, digest, size)
    except ArtifactVerificationError:
        raise
    except OSError as exc:
        raise ArtifactMaterializationError("stage_failed", str(exc)) from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
