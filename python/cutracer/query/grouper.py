# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Streaming grouper for trace record aggregation.

This module provides memory-efficient grouping of trace records,
using single-pass streaming with bounded memory per group.
"""

from collections import Counter, defaultdict, deque
from typing import Any, Iterator

from cutracer.types import TraceRecord


def _hashable(value: Any) -> Any:
    """Return a dict key for a field value; lists (e.g. cta) become tuples."""
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value


class StreamingGrouper:
    """
    Stream-based grouper for trace records.

    Processes records in a single pass, maintaining bounded memory
    per group using deque for tail operations.

    Design principles:
    - Single-pass: records iterator is consumed only once
    - Bounded memory: uses deque(maxlen=N) for tail operations
    - Memory complexity: O(groups × N) for head/tail, O(groups) for count

    Example:
        >>> records = iter([{"warp": 0, "pc": 16}, {"warp": 1, "pc": 32}, ...])
        >>> grouper = StreamingGrouper(records, "warp")
        >>> groups = grouper.tail_per_group(10)
        >>> for warp, records in groups.items():
        ...     print(f"Warp {warp}: {len(records)} records")
    """

    def __init__(self, records: Iterator[TraceRecord], group_field: str) -> None:
        """
        Initialize the streaming grouper.

        Args:
            records: Iterator of trace records (consumed once!)
            group_field: Field name to group by (e.g., "warp", "cta", "sass")
        """
        self._records = records
        self._group_field = group_field
        self._consumed = False

    def _ensure_not_consumed(self) -> None:
        """Raise error if records have already been consumed."""
        if self._consumed:
            raise RuntimeError(
                "StreamingGrouper records have already been consumed. "
                "Create a new grouper to process again."
            )
        self._consumed = True

    def head_per_group(self, n: int) -> dict[Any, list[TraceRecord]]:
        """
        Get first N records per group.

        Memory complexity: O(groups × N)

        Args:
            n: Maximum records per group

        Returns:
            Dict mapping group key to list of first N records
        """
        self._ensure_not_consumed()

        if n <= 0:
            return {}

        groups: dict[Any, list[TraceRecord]] = defaultdict(list)
        group_counts: Counter = Counter()

        for record in self._records:
            key = _hashable(record.get(self._group_field))
            if group_counts[key] < n:
                groups[key].append(record)
                group_counts[key] += 1

        return dict(groups)

    def tail_per_group(self, n: int) -> dict[Any, list[TraceRecord]]:
        """
        Get last N records per group.

        Uses deque(maxlen=N) for each group to bound memory usage.
        Memory complexity: O(groups × N)

        Args:
            n: Maximum records per group

        Returns:
            Dict mapping group key to list of last N records
        """
        self._ensure_not_consumed()

        if n <= 0:
            return {}

        groups: dict[Any, deque] = defaultdict(lambda: deque(maxlen=n))

        for record in self._records:
            key = _hashable(record.get(self._group_field))
            groups[key].append(record)

        return {k: list(v) for k, v in groups.items()}

    def count_per_group(self) -> dict[Any, int]:
        """
        Count records per group.

        Memory complexity: O(groups) - only stores counts, not records

        Returns:
            Dict mapping group key to record count
        """
        self._ensure_not_consumed()

        counts: Counter = Counter()

        for record in self._records:
            key = _hashable(record.get(self._group_field))
            counts[key] += 1

        return dict(counts)

    def all_per_group(self) -> dict[Any, list[TraceRecord]]:
        """
        Get all records per group.

        Warning: Memory usage is unbounded - O(total records).
        Use only when you need all records and memory is sufficient.

        Returns:
            Dict mapping group key to list of all records in that group
        """
        self._ensure_not_consumed()

        groups: dict[Any, list[TraceRecord]] = defaultdict(list)

        for record in self._records:
            key = _hashable(record.get(self._group_field))
            groups[key].append(record)

        return dict(groups)
