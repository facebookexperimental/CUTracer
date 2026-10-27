# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared bounded-retention policy for rejected stress configurations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar


MAX_REJECTED_CONFIGS = 16

_T = TypeVar("_T")


def retain_oldest_and_newest(items: Sequence[_T]) -> list[_T]:
    """Keep the oldest and newest halves of an oversized sequence."""
    if len(items) <= MAX_REJECTED_CONFIGS:
        return list(items)
    head_count = MAX_REJECTED_CONFIGS // 2
    tail_count = MAX_REJECTED_CONFIGS - head_count
    return [*items[:head_count], *items[-tail_count:]]
