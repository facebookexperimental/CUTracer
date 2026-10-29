"""Public diagnosis operations with explicit, profile-specific backend assembly."""

from __future__ import annotations

from typing import Protocol

from cutracer.service.run_contracts import RunHandle, RunInput, RunResult, RunSummary


class DiagnosisBackend(Protocol):
    def run(self, request: RunInput) -> RunResult: ...

    def resume(self, run_id: str) -> RunResult: ...

    def advance(self, run_id: str) -> RunSummary: ...

    def status(self, run_id: str) -> RunSummary: ...

    def result(self, run_id: str) -> RunResult: ...

    def submit(self, request: RunInput) -> RunHandle: ...


def run(request: RunInput, *, backend: DiagnosisBackend) -> RunResult:
    """Drive a Run in the caller's control process, returning complete reports."""
    return backend.run(request)


def resume(run_id: str, *, backend: DiagnosisBackend) -> RunResult:
    """Continue the same immutable Run, without resetting its deadline."""
    return backend.resume(run_id)


def advance(run_id: str, *, backend: DiagnosisBackend) -> RunSummary:
    """Advance explicitly when the caller owns scheduling responsibility."""
    return backend.advance(run_id)


def status(run_id: str, *, backend: DiagnosisBackend) -> RunSummary:
    """Read status without submitting experiments or model turns."""
    return backend.status(run_id)


def result(run_id: str, *, backend: DiagnosisBackend) -> RunResult:
    """Read available reports; incomplete and failed sessions remain visible."""
    return backend.result(run_id)


def submit(request: RunInput, *, backend: DiagnosisBackend) -> RunHandle:
    """Submit hosted control and return its stable Run identity."""
    return backend.submit(request)
