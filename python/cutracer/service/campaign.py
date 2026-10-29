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
