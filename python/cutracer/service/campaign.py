"""Deterministic campaign planning primitives."""

from __future__ import annotations

import dataclasses
import hashlib

from cutracer.service.catalog_adapters import CatalogWorkUnit


@dataclasses.dataclass(frozen=True)
class ShardAssignment:
    target: CatalogWorkUnit
    shard_index: int


def deterministic_shard(unit_id: str, shard_count: int, *, seed: str) -> int:
    if not unit_id.strip():
        raise ValueError("sharding requires a nonempty unit id")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not seed.strip():
        raise ValueError("shard seed must not be empty")
    digest = hashlib.sha256(f"{seed}\0{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % shard_count


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
