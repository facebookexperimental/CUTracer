"""Deterministic retry and terminal-failure policy for reconciliation."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Protocol


class ReconcilePolicyAction(str, Enum):
    WAIT = "wait"
    RETRY = "retry"
    FAIL = "fail"


class RemoteEffectKind(str, Enum):
    EXPERIMENT_JOB = "mast"
    REASONING_JOB = "sandcastle"

    # Aliases under the deployment-specific names; the string values are
    # persisted in durable attempt records and must not change.
    MAST = "mast"
    SANDCASTLE = "sandcastle"


class _AttemptState(Protocol):
    @property
    def age_ms(self) -> int: ...

    @property
    def consecutive_failures(self) -> int: ...


@dataclasses.dataclass(frozen=True)
class ReconcilePolicyDecision:
    action: ReconcilePolicyAction
    reason: str = ""

    def __post_init__(self) -> None:
        if (self.action is ReconcilePolicyAction.FAIL) != bool(self.reason):
            raise ValueError("only failed reconciliation decisions require a reason")


@dataclasses.dataclass(frozen=True)
class ReconcilePolicy:
    revision_timeout_ms: int = 24 * 60 * 60 * 1000
    mast_timeout_ms: int = 7 * 60 * 60 * 1000
    sandcastle_timeout_ms: int = 45 * 60 * 1000
    report_timeout_ms: int = 6 * 60 * 60 * 1000
    max_consecutive_failures: int = 5
    max_remote_attempts: int = 3
    max_report_failures: int = 8

    def __post_init__(self) -> None:
        if (
            self.revision_timeout_ms <= 0
            or self.mast_timeout_ms <= 0
            or self.sandcastle_timeout_ms <= 0
            or self.report_timeout_ms <= 0
            or self.max_consecutive_failures <= 0
            or self.max_remote_attempts <= 0
            or self.max_report_failures <= 0
        ):
            raise ValueError("reconciliation limits must be positive")
        if max(
            self.mast_timeout_ms,
            self.sandcastle_timeout_ms,
            self.report_timeout_ms,
        ) > (self.revision_timeout_ms):
            raise ValueError("effect timeouts cannot exceed the revision deadline")

    def assess_revision(self, attempt: _AttemptState) -> ReconcilePolicyDecision:
        if attempt.age_ms >= self.revision_timeout_ms:
            return ReconcilePolicyDecision(
                ReconcilePolicyAction.FAIL,
                "distributed revision deadline exceeded",
            )
        if attempt.consecutive_failures >= self.max_consecutive_failures:
            return ReconcilePolicyDecision(
                ReconcilePolicyAction.FAIL,
                "distributed infrastructure failure retry limit exceeded",
            )
        return ReconcilePolicyDecision(ReconcilePolicyAction.WAIT)

    def assess_publication(
        self,
        attempt: _AttemptState,
    ) -> ReconcilePolicyDecision:
        if attempt.age_ms >= self.report_timeout_ms:
            return ReconcilePolicyDecision(
                ReconcilePolicyAction.FAIL,
                "report publication deadline exceeded",
            )
        if attempt.consecutive_failures >= self.max_report_failures:
            return ReconcilePolicyDecision(
                ReconcilePolicyAction.FAIL,
                "report publication retry limit exceeded",
            )
        action = (
            ReconcilePolicyAction.RETRY
            if attempt.consecutive_failures
            else ReconcilePolicyAction.WAIT
        )
        return ReconcilePolicyDecision(action)

    def assess_effect(
        self,
        *,
        kind: RemoteEffectKind,
        attempts: int,
        updated_at_ms: int,
        now_ms: int,
        terminal_failure: bool,
    ) -> ReconcilePolicyDecision:
        if attempts < 0 or updated_at_ms < 0 or now_ms < 0:
            raise ValueError("remote effect timing and attempts are invalid")
        timeout_ms = (
            self.mast_timeout_ms
            if kind is RemoteEffectKind.EXPERIMENT_JOB
            else self.sandcastle_timeout_ms
        )
        timed_out = now_ms >= updated_at_ms and now_ms - updated_at_ms >= timeout_ms
        if not terminal_failure and not timed_out:
            return ReconcilePolicyDecision(ReconcilePolicyAction.WAIT)
        if attempts >= self.max_remote_attempts:
            return ReconcilePolicyDecision(
                ReconcilePolicyAction.FAIL,
                "remote effect retry limit exceeded",
            )
        return ReconcilePolicyDecision(ReconcilePolicyAction.RETRY)
