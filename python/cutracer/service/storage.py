"""Durable storage contracts for distributed diagnosis sessions."""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

from cutracer.service.contracts import (
    AnalysisSession,
    ArtifactRef,
    ReportPublicationStatus,
    SessionStatus,
)


class SessionStoreError(RuntimeError):
    """Base class for failures at the durable session-store boundary."""


class StoreConflict(SessionStoreError):
    """The requested create or compare-and-swap lost a concurrent race."""


class StoreUnavailable(SessionStoreError):
    """The backing store could not durably complete the requested operation."""


class SessionNotFound(SessionStoreError):
    """No durable session exists for the requested identifier."""


@dataclasses.dataclass(frozen=True)
class SessionRetentionPolicy:
    """Time-to-live policy for active and terminal session records."""

    active_ttl_s: int = 30 * 24 * 60 * 60
    terminal_ttl_s: int = 90 * 24 * 60 * 60
    prune_batch_size: int = 100

    def __post_init__(self) -> None:
        if self.active_ttl_s <= 0 or self.terminal_ttl_s <= 0:
            raise ValueError("session retention TTLs must be positive")
        if self.prune_batch_size <= 0:
            raise ValueError("session retention prune batch size must be positive")

    def expires_at_ms(self, status: SessionStatus, updated_at_ms: int) -> int:
        terminal = status in (
            SessionStatus.COMPLETED,
            SessionStatus.INCONCLUSIVE,
            SessionStatus.FAILED,
        )
        ttl_s = self.terminal_ttl_s if terminal else self.active_ttl_s
        return updated_at_ms + ttl_s * 1000


@dataclasses.dataclass(frozen=True)
class SessionMetadata:
    """Indexed session identity pointing to one content-addressed state blob."""

    session_id: str
    state_revision: int
    status: SessionStatus
    session_blob: ArtifactRef
    updated_at_ms: int
    expires_at_ms: int
    report_publication: ReportPublicationStatus = ReportPublicationStatus.NOT_READY

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session metadata requires session_id")
        if self.state_revision < 0:
            raise ValueError("session metadata revision must be non-negative")
        if self.updated_at_ms < 0 or self.expires_at_ms <= self.updated_at_ms:
            raise ValueError("session metadata has an invalid retention window")
        if (
            not self.session_blob.uri.strip()
            or self.session_blob.sha256 is None
            or len(self.session_blob.sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in self.session_blob.sha256
            )
            or self.session_blob.size_bytes is None
            or self.session_blob.size_bytes <= 0
            or self.session_blob.media_type != "application/json"
        ):
            raise ValueError("session metadata requires an identified session blob")

    @property
    def pending(self) -> bool:
        active = self.status not in (
            SessionStatus.COMPLETED,
            SessionStatus.INCONCLUSIVE,
            SessionStatus.FAILED,
        )
        return active or self.report_publication == ReportPublicationStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state_revision": self.state_revision,
            "status": self.status.value,
            "session_blob": self.session_blob.to_dict(),
            "updated_at_ms": self.updated_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "report_publication": self.report_publication.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionMetadata":
        status = SessionStatus(payload["status"])
        publication = payload.get("report_publication")
        if publication is None:
            publication = (
                ReportPublicationStatus.PUBLISHED.value
                if status in (SessionStatus.COMPLETED, SessionStatus.INCONCLUSIVE)
                else ReportPublicationStatus.NOT_READY.value
            )
        return cls(
            session_id=str(payload["session_id"]),
            state_revision=int(payload["state_revision"]),
            status=status,
            session_blob=ArtifactRef.from_dict(payload["session_blob"]),
            updated_at_ms=int(payload["updated_at_ms"]),
            expires_at_ms=int(payload["expires_at_ms"]),
            report_publication=ReportPublicationStatus(publication),
        )


class DurableArtifactStore(Protocol):
    """Content-addressed blobs; implementations must verify bytes on reads."""

    def publish(
        self,
        source_path: str,
        *,
        namespace: str,
        name: str,
        media_type: str,
    ) -> ArtifactRef: ...

    def materialize(self, artifact: ArtifactRef, destination: str) -> None: ...

    def delete(self, artifact: ArtifactRef) -> None: ...


class DurableSessionStore(Protocol):
    """Linearizable session operations used by `DistributedCoordinator`.

    `create` raises `StoreConflict` when the id exists. `compare_and_swap`
    raises it when `expected_revision` is stale. Backend/transport failures are
    `StoreUnavailable`; a missing id is `SessionNotFound`.
    """

    def create(self, session: AnalysisSession) -> None: ...

    def load(self, session_id: str) -> AnalysisSession: ...

    def compare_and_swap(
        self,
        session_id: str,
        expected_revision: int,
        session: AnalysisSession,
    ) -> None: ...


class ManagedSessionStore(DurableSessionStore, Protocol):
    """Operational indexes and bounded retention layered over CAS storage."""

    def list_pending(self, *, limit: int) -> tuple[str, ...]: ...

    def prune_expired(self, *, now_ms: int, limit: int | None = None) -> int: ...
