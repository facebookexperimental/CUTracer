"""Local Run control with atomic records and one active foreground driver."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
from collections.abc import Iterator

from cutracer.service.checkpoint import atomic_write_json, read_json
from cutracer.service.run_contracts import (
    RunControl,
    RunInputConflict,
    RunNotFound,
    validate_run_update,
)
from cutracer.service.storage import StoreConflict, StoreUnavailable


class FileRunControlStore:
    def __init__(self, root: str) -> None:
        self._root = os.path.abspath(root)
        try:
            os.makedirs(self._root, mode=0o700, exist_ok=True)
        except OSError as error:
            raise StoreUnavailable(
                f"could not initialize Run store: {error}"
            ) from error

    def _path(self, run_id: str, suffix: str) -> str:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        key = hashlib.sha256(run_id.encode()).hexdigest()
        return os.path.join(self._root, f"{key}.{suffix}")

    @contextlib.contextmanager
    def _lock(self, run_id: str, *, execution: bool = False) -> Iterator[None]:
        suffix = "execution.lock" if execution else "lock"
        try:
            descriptor = os.open(
                self._path(run_id, suffix), os.O_CREAT | os.O_RDWR, 0o600
            )
            lock = os.fdopen(descriptor, "a+")
        except OSError as error:
            raise StoreUnavailable(f"could not open Run lock: {error}") from error
        with lock:
            try:
                mode = fcntl.LOCK_EX | (fcntl.LOCK_NB if execution else 0)
                fcntl.flock(lock.fileno(), mode)
            except BlockingIOError as error:
                raise StoreConflict(
                    f"Run already has an active local driver: {run_id}"
                ) from error
            except OSError as error:
                raise StoreUnavailable(f"could not lock Run: {error}") from error
            yield

    @contextlib.contextmanager
    def execution(self, run_id: str) -> Iterator[None]:
        """OS releases ownership on process death; readers remain independent."""
        with self._lock(run_id, execution=True):
            yield

    def _read(self, run_id: str) -> RunControl:
        try:
            control = RunControl.from_dict(read_json(self._path(run_id, "json")))
        except FileNotFoundError as error:
            raise RunNotFound(run_id) from error
        except (OSError, KeyError, TypeError, ValueError, RunInputConflict) as error:
            raise StoreUnavailable(f"could not read Run {run_id}: {error}") from error
        if control.input.run_id != run_id:
            raise StoreUnavailable("Run record identity does not match its key")
        return control

    def _write(self, control: RunControl) -> None:
        try:
            atomic_write_json(
                self._path(control.input.run_id, "json"), control.to_dict()
            )
        except OSError as error:
            raise StoreUnavailable(f"could not write Run: {error}") from error

    def create(self, control: RunControl) -> None:
        run_id = control.input.run_id
        with self._lock(run_id):
            if os.path.exists(self._path(run_id, "json")):
                raise StoreConflict(run_id)
            self._write(control)

    def load(self, run_id: str) -> RunControl:
        return self._read(run_id)

    def compare_and_swap(
        self, run_id: str, expected_revision: int, control: RunControl
    ) -> None:
        if control.input.run_id != run_id or control.revision != expected_revision + 1:
            raise ValueError(
                "replacement Run must keep its ID and advance one revision"
            )
        with self._lock(run_id):
            current = self._read(run_id)
            validate_run_update(current, expected_revision, control)
            self._write(control)
