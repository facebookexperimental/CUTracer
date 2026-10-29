"""Hard resource-reservation budgets for scheduled diagnosis campaigns."""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Mapping
from enum import Enum

from cutracer.service.campaign import AdmissionDisposition
from cutracer.service.catalog_adapters import CatalogWorkUnit


class ResourceConstraint(str, Enum):
    REASONING_TOKENS = "reasoning_tokens"
    ARTIFACT_BYTES = "artifact_bytes"
    SANDCASTLE_RATE = "sandcastle_rate"


@dataclasses.dataclass(frozen=True)
class CampaignResourceEstimate:
    reasoning_tokens: int
    artifact_bytes: int
    sandcastle_submissions: int

    def __post_init__(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"resource estimate {name} must be non-negative")


@dataclasses.dataclass(frozen=True)
class CampaignResourceUsage:
    reasoning_tokens: int = 0
    artifact_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"resource usage {name} must be non-negative")


@dataclasses.dataclass(frozen=True)
class SandcastleRateBudget:
    max_submissions: int
    window_seconds: int = 60 * 60

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_submissions, bool)
            or not isinstance(self.max_submissions, int)
            or self.max_submissions < 0
        ):
            raise ValueError("Remote max_submissions must be non-negative")
        if (
            isinstance(self.window_seconds, bool)
            or not isinstance(self.window_seconds, int)
            or self.window_seconds <= 0
        ):
            raise ValueError("Remote rate window must be positive")


@dataclasses.dataclass(frozen=True)
class CampaignResourceBudget:
    max_reasoning_tokens: int | None = None
    max_artifact_bytes: int | None = None
    sandcastle_rate: SandcastleRateBudget | None = None

    def __post_init__(self) -> None:
        values = {
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "max_artifact_bytes": self.max_artifact_bytes,
        }
        for name, value in values.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"campaign {name} must be non-negative")


@dataclasses.dataclass(frozen=True)
class ResourceAdmissionDecision:
    target: CatalogWorkUnit
    disposition: AdmissionDisposition
    constraints: tuple[ResourceConstraint, ...]
    cumulative_usage: CampaignResourceUsage
    cumulative_sandcastle_submissions: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.target.unit.unit_id,
            "disposition": self.disposition.value,
            "constraints": [constraint.value for constraint in self.constraints],
            "cumulative_usage": dataclasses.asdict(self.cumulative_usage),
            "cumulative_sandcastle_submissions": (
                self.cumulative_sandcastle_submissions
            ),
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class ResourceAdmissionPlan:
    decisions: tuple[ResourceAdmissionDecision, ...]
    recent_sandcastle_submissions: int

    @property
    def admitted(self) -> tuple[CatalogWorkUnit, ...]:
        return tuple(
            decision.target
            for decision in self.decisions
            if decision.disposition == AdmissionDisposition.ADMITTED
        )


def _recent_sandcastle_count(
    budget: SandcastleRateBudget | None,
    *,
    as_of: datetime.datetime,
    submitted_at: tuple[datetime.datetime, ...],
) -> int:
    if budget is None:
        return 0
    if as_of.tzinfo is None:
        raise ValueError("resource budget as_of must be timezone-aware")
    if any(value.tzinfo is None for value in submitted_at):
        raise ValueError("Remote submission times must be timezone-aware")
    if any(value > as_of for value in submitted_at):
        raise ValueError("Remote submission time must not be in the future")
    window_start = as_of - datetime.timedelta(seconds=budget.window_seconds)
    return sum(value > window_start for value in submitted_at)


def _constraints(
    budget: CampaignResourceBudget,
    estimate: CampaignResourceEstimate,
    usage: CampaignResourceUsage,
    sandcastle_submissions: int,
) -> tuple[ResourceConstraint, ...]:
    constraints: list[ResourceConstraint] = []
    if (
        budget.max_reasoning_tokens is not None
        and usage.reasoning_tokens + estimate.reasoning_tokens
        > budget.max_reasoning_tokens
    ):
        constraints.append(ResourceConstraint.REASONING_TOKENS)
    if (
        budget.max_artifact_bytes is not None
        and usage.artifact_bytes + estimate.artifact_bytes > budget.max_artifact_bytes
    ):
        constraints.append(ResourceConstraint.ARTIFACT_BYTES)
    if (
        budget.sandcastle_rate is not None
        and sandcastle_submissions + estimate.sandcastle_submissions
        > budget.sandcastle_rate.max_submissions
    ):
        constraints.append(ResourceConstraint.SANDCASTLE_RATE)
    return tuple(constraints)


def apply_resource_budget(
    targets: tuple[CatalogWorkUnit, ...],
    estimates: Mapping[str, CampaignResourceEstimate],
    budget: CampaignResourceBudget,
    *,
    as_of: datetime.datetime,
    sandcastle_submitted_at: tuple[datetime.datetime, ...] = (),
    initial_usage: CampaignResourceUsage | None = None,
) -> ResourceAdmissionPlan:
    unit_ids = [target.unit.unit_id for target in targets]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("resource budget targets must have unique unit ids")
    missing = sorted(set(unit_ids) - set(estimates))
    if missing:
        raise ValueError(f"resource estimates are missing for: {', '.join(missing)}")
    recent = _recent_sandcastle_count(
        budget.sandcastle_rate,
        as_of=as_of,
        submitted_at=sandcastle_submitted_at,
    )
    usage = CampaignResourceUsage() if initial_usage is None else initial_usage
    sandcastle_submissions = recent
    decisions: list[ResourceAdmissionDecision] = []
    for target in targets:
        estimate = estimates[target.unit.unit_id]
        constraints = _constraints(budget, estimate, usage, sandcastle_submissions)
        if not constraints:
            usage = CampaignResourceUsage(
                reasoning_tokens=usage.reasoning_tokens + estimate.reasoning_tokens,
                artifact_bytes=usage.artifact_bytes + estimate.artifact_bytes,
            )
            sandcastle_submissions += estimate.sandcastle_submissions
        decisions.append(
            ResourceAdmissionDecision(
                target=target,
                disposition=(
                    AdmissionDisposition.DEFERRED
                    if constraints
                    else AdmissionDisposition.ADMITTED
                ),
                constraints=constraints,
                cumulative_usage=usage,
                cumulative_sandcastle_submissions=sandcastle_submissions,
                reason=(
                    "within resource budget"
                    if not constraints
                    else "exceeded "
                    + ", ".join(constraint.value for constraint in constraints)
                ),
            )
        )
    return ResourceAdmissionPlan(tuple(decisions), recent)
