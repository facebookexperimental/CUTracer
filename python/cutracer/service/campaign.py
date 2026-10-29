"""Deterministic campaign planning primitives."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict
from enum import Enum

from cutracer.service.catalog_adapters import CatalogWorkUnit


@dataclasses.dataclass(frozen=True)
class ShardAssignment:
    target: CatalogWorkUnit
    shard_index: int


class SamplingDisposition(str, Enum):
    SELECTED = "selected"
    SAMPLED_OUT = "sampled_out"


class AdmissionDisposition(str, Enum):
    ADMITTED = "admitted"
    DEFERRED = "deferred"


class BudgetConstraint(str, Enum):
    RUN_CAP = "run_cap"
    GPU_SECONDS = "gpu_seconds"


@dataclasses.dataclass(frozen=True)
class SamplingDecision:
    target: CatalogWorkUnit
    disposition: SamplingDisposition
    rank: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target.to_dict(),
            "disposition": self.disposition.value,
            "rank": self.rank,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class SamplingPlan:
    catalog_revision: str
    decisions: tuple[SamplingDecision, ...]
    schema_version: int = 1

    @property
    def selected(self) -> tuple[CatalogWorkUnit, ...]:
        return tuple(
            decision.target
            for decision in self.decisions
            if decision.disposition == SamplingDisposition.SELECTED
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "catalog_revision": self.catalog_revision,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass(frozen=True)
class CampaignBudget:
    max_runs: int | None = None
    max_gpu_seconds: int | None = None

    def __post_init__(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"campaign {name} must be a non-negative integer")


@dataclasses.dataclass(frozen=True)
class AdmissionDecision:
    target: CatalogWorkUnit
    disposition: AdmissionDisposition
    constraints: tuple[BudgetConstraint, ...]
    cumulative_admitted_runs: int
    cumulative_admitted_gpu_seconds: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target.to_dict(),
            "disposition": self.disposition.value,
            "constraints": [constraint.value for constraint in self.constraints],
            "cumulative_admitted_runs": self.cumulative_admitted_runs,
            "cumulative_admitted_gpu_seconds": self.cumulative_admitted_gpu_seconds,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class AdmissionPlan:
    budget: CampaignBudget
    decisions: tuple[AdmissionDecision, ...]

    @property
    def admitted(self) -> tuple[CatalogWorkUnit, ...]:
        return tuple(
            decision.target
            for decision in self.decisions
            if decision.disposition == AdmissionDisposition.ADMITTED
        )

    @property
    def admitted_gpu_seconds(self) -> int:
        return sum(target.estimated_gpu_seconds for target in self.admitted)


def deterministic_shard(unit_id: str, shard_count: int, *, seed: str) -> int:
    if not unit_id.strip():
        raise ValueError("sharding requires a nonempty unit id")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not seed.strip():
        raise ValueError("shard seed must not be empty")
    digest = hashlib.sha256(f"{seed}\0{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % shard_count


def _sample_score(target: CatalogWorkUnit) -> bytes:
    identity = (
        f"{target.catalog_revision}\0{target.entry_id}\0"
        f"{target.sampling.seed}\0{target.selector_index}\0{target.unit.test_id}"
    )
    return hashlib.sha256(identity.encode()).digest()


def _selector_ranks(targets: list[CatalogWorkUnit]) -> dict[int, int]:
    representatives: dict[int, CatalogWorkUnit] = {}
    for target in targets:
        prior = representatives.setdefault(target.selector_index, target)
        if prior.unit.test_id != target.unit.test_id:
            raise ValueError("selector index maps to inconsistent test identities")
        if prior.sampling != target.sampling:
            raise ValueError("catalog entry has inconsistent sampling policies")
    ordered = sorted(
        representatives.values(),
        key=lambda target: (_sample_score(target), target.selector_index),
    )
    return {target.selector_index: rank for rank, target in enumerate(ordered)}


def plan_catalog_sampling(
    targets: tuple[CatalogWorkUnit, ...],
    *,
    catalog_revision: str,
) -> SamplingPlan:
    if not catalog_revision.strip():
        raise ValueError("sampling requires a catalog revision")
    if any(target.catalog_revision != catalog_revision for target in targets):
        raise ValueError("sampling target has the wrong catalog revision")
    unit_ids = [target.unit.unit_id for target in targets]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("sampling targets must have unique unit ids")
    grouped: dict[str, list[CatalogWorkUnit]] = defaultdict(list)
    for target in targets:
        grouped[target.entry_id].append(target)
    decisions: list[SamplingDecision] = []
    for entry_id in sorted(grouped):
        entry_targets = grouped[entry_id]
        if len({target.sampling for target in entry_targets}) != 1:
            raise ValueError("catalog entry has inconsistent sampling policies")
        ranks = _selector_ranks(entry_targets)
        limit = entry_targets[0].sampling.max_selected
        for target in sorted(entry_targets, key=lambda item: item.unit.unit_id):
            rank = ranks[target.selector_index]
            selected = limit is None or rank < limit
            disposition = (
                SamplingDisposition.SELECTED
                if selected
                else SamplingDisposition.SAMPLED_OUT
            )
            decisions.append(
                SamplingDecision(
                    target=target,
                    disposition=disposition,
                    rank=rank,
                    reason=f"selector rank {rank} with limit {limit}",
                )
            )
    return SamplingPlan(catalog_revision, tuple(decisions))


def _budget_constraints(
    target: CatalogWorkUnit,
    budget: CampaignBudget,
    admitted_runs: int,
    admitted_gpu_seconds: int,
) -> tuple[BudgetConstraint, ...]:
    constraints: list[BudgetConstraint] = []
    if budget.max_runs is not None and admitted_runs >= budget.max_runs:
        constraints.append(BudgetConstraint.RUN_CAP)
    projected_gpu_seconds = admitted_gpu_seconds + target.estimated_gpu_seconds
    if (
        budget.max_gpu_seconds is not None
        and projected_gpu_seconds > budget.max_gpu_seconds
    ):
        constraints.append(BudgetConstraint.GPU_SECONDS)
    return tuple(constraints)


def apply_campaign_budget(
    targets: tuple[CatalogWorkUnit, ...],
    budget: CampaignBudget,
) -> AdmissionPlan:
    unit_ids = [target.unit.unit_id for target in targets]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("campaign budget targets must have unique unit ids")
    decisions: list[AdmissionDecision] = []
    admitted_runs = 0
    admitted_gpu_seconds = 0
    for target in sorted(targets, key=lambda item: item.unit.unit_id):
        constraints = _budget_constraints(
            target,
            budget,
            admitted_runs,
            admitted_gpu_seconds,
        )
        disposition = (
            AdmissionDisposition.DEFERRED
            if constraints
            else AdmissionDisposition.ADMITTED
        )
        if disposition == AdmissionDisposition.ADMITTED:
            admitted_runs += 1
            admitted_gpu_seconds += target.estimated_gpu_seconds
        decisions.append(
            AdmissionDecision(
                target=target,
                disposition=disposition,
                constraints=constraints,
                cumulative_admitted_runs=admitted_runs,
                cumulative_admitted_gpu_seconds=admitted_gpu_seconds,
                reason=(
                    "within campaign budget"
                    if not constraints
                    else "exceeded "
                    + ", ".join(constraint.value for constraint in constraints)
                ),
            )
        )
    return AdmissionPlan(budget, tuple(decisions))


def assign_catalog_shards(
    targets: tuple[CatalogWorkUnit, ...],
    *,
    shard_count: int,
    seed: str,
) -> tuple[ShardAssignment, ...]:
    unit_ids = [target.unit.unit_id for target in targets]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("catalog targets must have unique unit ids")
    return tuple(
        ShardAssignment(
            target=target,
            shard_index=deterministic_shard(
                target.unit.unit_id,
                shard_count,
                seed=seed,
            ),
        )
        for target in sorted(targets, key=lambda item: item.unit.unit_id)
    )


def select_catalog_shard(
    targets: tuple[CatalogWorkUnit, ...],
    *,
    shard_index: int,
    shard_count: int,
    seed: str,
) -> tuple[CatalogWorkUnit, ...]:
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be inside the configured shard range")
    return tuple(
        assignment.target
        for assignment in assign_catalog_shards(
            targets,
            shard_count=shard_count,
            seed=seed,
        )
        if assignment.shard_index == shard_index
    )
