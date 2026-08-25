"""Artifact helpers shared by CUTracer service experiment adapters."""

from __future__ import annotations

import hashlib
import os

from cutracer.service.contracts import ArtifactRef


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
        size = os.path.getsize(absolute)
        if include_sha256:
            hasher = hashlib.sha256()
            with open(absolute, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
    except FileNotFoundError:
        pass
    return ArtifactRef(
        uri=f"file://{absolute}",
        relative_path=os.path.basename(absolute),
        sha256=digest,
        size_bytes=size,
        media_type=media_type,
    )


def local_path(ref: ArtifactRef) -> str:
    if ref.uri.startswith("file://"):
        return ref.uri[len("file://") :]
    if "://" not in ref.uri:
        return ref.uri
    raise ValueError(f"artifact is not materialized locally: {ref.uri}")
