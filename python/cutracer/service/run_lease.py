# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Run ownership, independent of diagnostic session revisions."""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
import uuid
from collections.abc import Callable

from cutracer.service.run_contracts import RunControl, RunControlStore
from cutracer.service.storage import SessionStoreError, StoreConflict


logger: logging.Logger = logging.getLogger(__name__)


class RunLeaseLost(RuntimeError):
    """This process is no longer allowed to dispatch work for the Run."""


class RunDeadlineReached(RuntimeError):
    """The fixed Run deadline has passed; existing jobs are not cancelled."""


class RunLease:
    def __init__(
        self,
        store: RunControlStore,
        run_id: str,
        *,
        clock_ms: Callable[[], int] | None = None,
        lease_ms: int = 120_000,
    ) -> None:
        if lease_ms < 30:
            raise ValueError("Run lease must be at least 30 milliseconds")
        self._store = store
        self._run_id = run_id
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._lease_ms = lease_ms
        self._token = uuid.uuid4().hex
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._failure: str = ""
        self._thread: threading.Thread | None = None
        self._deadline_ms: int = 0

    def __enter__(self) -> RunLease:
        for _ in range(8):
            current = self._store.load(self._run_id)
            now = self._clock_ms()
            if now >= current.input.deadline_ms:
                raise RunDeadlineReached(self._run_id)
            if current.owner_token and current.lease_expires_ms > now:
                raise StoreConflict(f"Run has an active controller: {self._run_id}")
            updated = dataclasses.replace(
                current,
                revision=current.revision + 1,
                owner_token=self._token,
                lease_expires_ms=min(now + self._lease_ms, current.input.deadline_ms),
            )
            try:
                self._store.compare_and_swap(self._run_id, current.revision, updated)
                self._deadline_ms = current.input.deadline_ms
                break
            except StoreConflict:
                continue
        else:
            raise StoreConflict(f"Run ownership contention: {self._run_id}")
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def _check_unlocked(self) -> RunControl:
        if self._deadline_ms and self._clock_ms() >= self._deadline_ms:
            raise RunDeadlineReached(self._run_id)
        if self._failure:
            raise RunLeaseLost(self._failure)
        current = self._store.load(self._run_id)
        now = self._clock_ms()
        if now >= current.input.deadline_ms:
            raise RunDeadlineReached(self._run_id)
        if current.owner_token != self._token or current.lease_expires_ms <= now:
            raise RunLeaseLost(f"Run ownership expired or changed: {self._run_id}")
        return current

    def check(self) -> RunControl:
        with self._lock:
            return self._check_unlocked()

    def before_dispatch(self) -> None:
        self.check()

    def update(self, transform: Callable[[RunControl], RunControl]) -> RunControl:
        with self._lock:
            for _ in range(8):
                current = self._check_unlocked()
                updated = dataclasses.replace(
                    transform(current), revision=current.revision + 1
                )
                if updated.owner_token != self._token:
                    raise ValueError("an active Run update cannot change its owner")
                try:
                    self._store.compare_and_swap(
                        self._run_id, current.revision, updated
                    )
                    return updated
                except StoreConflict:
                    continue
            raise RunLeaseLost(f"Run update contention: {self._run_id}")

    def _renew(self) -> None:
        # Renew without holding _lock: serializing on the user update()
        # lock can starve renewal behind a slow compare-and-swap and let
        # the lease lapse under a live owner. The store CAS arbitrates.
        for _ in range(8):
            if self._stop.is_set():
                return
            current = self._check_unlocked()
            updated = dataclasses.replace(
                current,
                revision=current.revision + 1,
                lease_expires_ms=min(
                    self._clock_ms() + self._lease_ms,
                    current.input.deadline_ms,
                ),
            )
            try:
                self._store.compare_and_swap(self._run_id, current.revision, updated)
                return
            except StoreConflict:
                continue
        raise RunLeaseLost(f"Run update contention: {self._run_id}")

    def _heartbeat(self) -> None:
        # Floor the interval so a very short lease cannot hammer the store
        # at lease/3 rates (a 30 ms lease would otherwise renew ~100x/s).
        interval_s = max(self._lease_ms / 3000, 0.1)
        while not self._stop.wait(interval_s):
            try:
                self._renew()
            except (SessionStoreError, RunLeaseLost, RunDeadlineReached) as error:
                self._failure = str(error)
                return

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1)
            if thread.is_alive():
                # Do not abandon ownership to natural expiry: the
                # heartbeat checks _stop before renewing again, so
                # releasing below cannot be undone by a fresh renewal,
                # and the CAS retry absorbs an in-flight renewal.
                logger.warning(
                    "Run heartbeat is still returning; releasing ownership "
                    "without waiting for it to finish"
                )
        with self._lock:
            try:
                for _ in range(8):
                    current = self._store.load(self._run_id)
                    if current.owner_token != self._token:
                        break
                    try:
                        self._store.compare_and_swap(
                            self._run_id,
                            current.revision,
                            dataclasses.replace(
                                current,
                                revision=current.revision + 1,
                                owner_token="",
                                lease_expires_ms=0,
                            ),
                        )
                        break
                    except StoreConflict:
                        continue
            except SessionStoreError as error:
                logger.warning("Could not release Run ownership: %s", error)
