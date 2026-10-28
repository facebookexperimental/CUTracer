# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Typed evidence references and immutable, turn-scoped citation identifiers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable

from cutracer.service.contracts import ArtifactRef, EvidenceBundle


def evidence_artifact_refs(evidence: EvidenceBundle) -> list[ArtifactRef]:
    """Enumerate citable typed artifacts, not arbitrary strings in evidence."""
    refs: list[ArtifactRef] = []
    refs.extend(
        sanitizer.log for sanitizer in evidence.sanitizer if sanitizer.log is not None
    )
    for stress in evidence.stress:
        if stress.triggering_config is not None:
            refs.append(stress.triggering_config.artifact)
        if stress.log is not None:
            refs.append(stress.log)
    for trace in evidence.traces:
        if trace.analysis is not None and trace.analysis.artifact is not None:
            refs.append(trace.analysis.artifact)
        refs.extend(trace.traces)
        refs.extend(trace.supporting_artifacts)
        if trace.log is not None:
            refs.append(trace.log)
    refs.extend(
        artifact
        for reduction in evidence.reductions
        for artifact in reduction.artifact_refs()
    )
    return refs


@dataclasses.dataclass(frozen=True)
class CitationRegistry:
    entries: tuple[tuple[str, str], ...]

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        state_revision: int,
        uris: Iterable[str],
    ) -> CitationRegistry:
        ordered = sorted(set(uris))
        if any(not isinstance(uri, str) or not uri for uri in ordered):
            raise ValueError("citation references must be nonempty URI strings")
        identity = json.dumps(
            [session_id, state_revision, ordered],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        scope = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return cls(
            tuple(
                (f"c{scope}_{index}", uri) for index, uri in enumerate(ordered, start=1)
            )
        )

    def to_dict(self) -> dict[str, str]:
        return dict(self.entries)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, str],
        *,
        session_id: str,
        state_revision: int,
    ) -> CitationRegistry:
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in payload.items()
        ):
            raise ValueError("citation registry must map IDs to URI strings")
        registry = cls.create(
            session_id=session_id,
            state_revision=state_revision,
            uris=payload.values(),
        )
        if registry.to_dict() != payload:
            raise ValueError("citation registry does not match its frozen turn")
        return registry
