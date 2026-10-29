"""File-backed reference implementation of the durable session-store contract."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterator
from typing import Any, Callable

from cutracer.service.contracts import AnalysisSession, SessionStatus
from cutracer.service.storage import (
    ManagedSessionStore,
    SessionNotFound,
    SessionRetentionPolicy,
    StoreConflict,
    StoreUnavailable,
)


ClockMs = Callable[[], int]
_TERMINAL_STATUSES = (
    SessionStatus.COMPLETED,
    SessionStatus.INCONCLUSIVE,
    SessionStatus.FAILED,
)


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


class FileSessionStore(ManagedSessionStore):
    """Linearizable local store used for development and conformance tests."""

    def __init__(
        self,
        root: str,
        *,
        retention: SessionRetentionPolicy | None = None,
        clock_ms: ClockMs | None = None,
    ) -> None:
        self._root = os.path.abspath(root)
        self._retention = retention or SessionRetentionPolicy()
        self._clock_ms = clock_ms or _system_clock_ms
        try:
            os.makedirs(self._root, mode=0o700, exist_ok=True)
            if not os.path.isdir(self._root) or os.path.islink(self._root):
                raise OSError("store root must be a regular directory")
        except OSError as error:
            raise StoreUnavailable(
                f"could not initialize file store: {error}"
            ) from error

    def _key(self, session_id: str) -> str:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        return hashlib.sha256(session_id.encode()).hexdigest()

    def _record_path(self, session_id: str) -> str:
        return os.path.join(self._root, f"{self._key(session_id)}.json")

    @contextlib.contextmanager
    def _locked(self, session_id: str, *, exclusive: bool) -> Iterator[None]:
        lock_path = os.path.join(self._root, f"{self._key(session_id)}.lock")
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                lock_file = os.fdopen(descriptor, "a+")
            except OSError:
                os.close(descriptor)
                raise
            with lock_file as lock:
                fcntl.flock(
                    lock.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                )
                yield
        except OSError as error:
            raise StoreUnavailable(f"file-store lock failed: {error}") from error

    def _read_record(self, session_id: str) -> dict[str, Any]:
        path = self._record_path(session_id)
        try:
            if os.path.islink(path):
                raise OSError("session record must not be a symbolic link")
            with open(path) as source:
                payload = json.load(source)
        except FileNotFoundError as error:
            raise SessionNotFound(session_id) from error
        except (OSError, TypeError, ValueError) as error:
            raise StoreUnavailable(
                f"could not read session {session_id}: {error}"
            ) from error
        if not isinstance(payload, dict) or payload.get("session_id") != session_id:
            raise StoreUnavailable(f"invalid file-store record for {session_id}")
        return payload

    def _record(self, session: AnalysisSession, updated_at_ms: int) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "state_revision": session.state_revision,
            "status": session.status.value,
            "updated_at_ms": updated_at_ms,
            "expires_at_ms": self._retention.expires_at_ms(
                session.status,
                updated_at_ms,
            ),
            "pending": session.requires_reconciliation,
            "session": session.to_dict(),
        }

    def _write_record(self, session: AnalysisSession) -> None:
        path = self._record_path(session.session_id)
        descriptor = -1
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".session-", dir=self._root)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w") as output:
                descriptor = -1
                json.dump(
                    self._record(session, self._clock_ms()), output, sort_keys=True
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            temporary = None
            directory = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            raise StoreUnavailable(f"could not write session: {error}") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _session(record: dict[str, Any]) -> AnalysisSession:
        payload = record.get("session")
        if not isinstance(payload, dict):
            raise StoreUnavailable("file-store session payload is not a JSON object")
        try:
            session = AnalysisSession.from_dict(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise StoreUnavailable(
                f"invalid file-store session payload: {error}"
            ) from error
        if session.state_revision != int(record.get("state_revision", -1)):
            raise StoreUnavailable(
                "file-store session revision metadata is inconsistent"
            )
        if session.status.value != record.get("status"):
            raise StoreUnavailable("file-store session status metadata is inconsistent")
        if "pending" in record:
            expected_pending = session.requires_reconciliation
            if bool(record["pending"]) != expected_pending:
                raise StoreUnavailable(
                    "file-store session pending metadata is inconsistent"
                )
        return session

    def create(self, session: AnalysisSession) -> None:
        with self._locked(session.session_id, exclusive=True):
            if os.path.exists(self._record_path(session.session_id)):
                raise StoreConflict(session.session_id)
            self._write_record(session)

    def load(self, session_id: str) -> AnalysisSession:
        with self._locked(session_id, exclusive=False):
            return self._session(self._read_record(session_id))

    def compare_and_swap(
        self,
        session_id: str,
        expected_revision: int,
        session: AnalysisSession,
    ) -> None:
        if session.session_id != session_id:
            raise ValueError("replacement session_id differs from CAS key")
        if session.state_revision != expected_revision + 1:
            raise ValueError("replacement session revision must advance by exactly one")
        with self._locked(session_id, exclusive=True):
            current = self._session(self._read_record(session_id))
            if current.state_revision != expected_revision:
                raise StoreConflict(session_id)
            self._write_record(session)

    def _record_summaries(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            names = sorted(
                name for name in os.listdir(self._root) if name.endswith(".json")
            )
            for name in names:
                try:
                    path = os.path.join(self._root, name)
                    if os.path.islink(path):
                        raise OSError("session record must not be a symbolic link")
                    with open(path) as source:
                        payload = json.load(source)
                except FileNotFoundError:
                    continue
                if not isinstance(payload, dict):
                    raise ValueError(f"invalid record: {name}")
                records.append(payload)
        except (OSError, TypeError, ValueError) as error:
            raise StoreUnavailable(f"could not scan file store: {error}") from error
        return records

    @staticmethod
    def _is_pending_record(record: dict[str, Any]) -> bool:
        if "pending" in record:
            return bool(record["pending"])
        return SessionStatus(record["status"]) not in _TERMINAL_STATUSES

    def list_pending(self, *, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            raise ValueError("pending-session limit must be positive")
        try:
            pending = [
                (int(record["updated_at_ms"]), str(record["session_id"]))
                for record in self._record_summaries()
                if self._is_pending_record(record)
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise StoreUnavailable(
                f"invalid pending-session metadata: {error}"
            ) from error
        return tuple(session_id for _updated, session_id in sorted(pending)[:limit])

    @staticmethod
    def _retention_metadata(record: dict[str, Any]) -> tuple[int, str]:
        try:
            return int(record["expires_at_ms"]), str(record["session_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise StoreUnavailable(f"invalid retention metadata: {error}") from error

    def prune_expired(self, *, now_ms: int, limit: int | None = None) -> int:
        batch = self._retention.prune_batch_size if limit is None else limit
        if batch <= 0:
            raise ValueError("retention prune limit must be positive")
        candidates = (
            self._retention_metadata(record) for record in self._record_summaries()
        )
        expired = sorted(entry for entry in candidates if entry[0] <= now_ms)
        removed = 0
        for _expires_at, session_id in expired[:batch]:
            with self._locked(session_id, exclusive=True):
                try:
                    record = self._read_record(session_id)
                    expires_at_ms, _ = self._retention_metadata(record)
                except SessionNotFound:
                    continue
                if expires_at_ms > now_ms:
                    continue
                try:
                    os.unlink(self._record_path(session_id))
                except OSError as error:
                    raise StoreUnavailable(
                        f"could not prune session {session_id}: {error}"
                    ) from error
                removed += 1
        if removed:
            try:
                directory = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError as error:
                raise StoreUnavailable(
                    f"could not sync pruned sessions: {error}"
                ) from error
        return removed
