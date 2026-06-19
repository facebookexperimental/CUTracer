# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Warp execution status summary for GPU hang analysis.

This module provides utilities for analyzing warp execution status
from trace records grouped by warp ID. It identifies:
- Completed warps: executed EXIT instruction (normal termination)
- In-progress warps: did not execute EXIT (may be hung or interrupted)
- Missing warps: never appeared in trace (scheduling issues)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from cutracer.types import TraceRecord


@dataclass
class WarpSummary:
    """Summary statistics for warp grouping."""

    total_observed: int
    min_warp_id: int
    max_warp_id: int
    completed_warp_ids: list[int] = field(default_factory=list)
    inprogress_warp_ids: list[int] = field(default_factory=list)
    missing_warp_ids: list[int] = field(default_factory=list)


def is_exit_instruction(record: TraceRecord) -> bool:
    """
    Check if a record's SASS instruction is an EXIT instruction.

    EXIT instructions can be:
    - "EXIT;"
    - "@P0 EXIT;"  (predicated)
    - "EXIT.KEEPREFCOUNT;"  (with modifier)

    Args:
        record: A trace record dictionary

    Returns:
        True if the instruction is EXIT
    """
    sass = record.get("sass", "")
    if not sass:
        return False
    return "EXIT" in sass.upper() and sass.strip().endswith(";")


def merge_to_ranges(ids: list[int]) -> list[tuple[int, int]]:
    """
    Merge consecutive IDs into ranges.

    Args:
        ids: List of integer IDs (will be sorted)

    Returns:
        List of (start, end) tuples representing ranges

    Example:
        [0, 1, 2, 3, 6, 7, 8, 9] -> [(0, 3), (6, 9)]
    """
    if not ids:
        return []

    sorted_ids = sorted(ids)
    ranges = []
    start = end = sorted_ids[0]

    for i in sorted_ids[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append((start, end))
            start = end = i

    ranges.append((start, end))
    return ranges


def format_ranges(ranges: list[tuple[int, int]]) -> str:
    """
    Format ranges as a human-readable string.

    Args:
        ranges: List of (start, end) tuples

    Returns:
        Formatted string like "0-3, 6-9, 16-127"

    Example:
        [(0, 3), (6, 9)] -> "0-3, 6-9"
        [(5, 5)] -> "5"
    """
    if not ranges:
        return "(none)"

    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")

    return ", ".join(parts)


def compute_warp_summary(groups: dict[Any, list[TraceRecord]]) -> Optional[WarpSummary]:
    """
    Compute warp summary statistics from grouped records.

    Args:
        groups: Dict mapping warp ID to list of records

    Returns:
        WarpSummary object, or None if groups is empty or contains no
        integer-keyed warp groups
    """
    if not groups:
        return None

    # Tolerate non-integer group keys: a trace may include records without a
    # "warp" field (e.g. a kernel_metadata header), which get bucketed under a
    # None key by the grouper. Skip those groups rather than discarding the
    # whole summary; only bail out if there are no integer warp ids at all.
    warp_ids = []
    completed_ids = []
    inprogress_ids = []

    for warp_id, records in groups.items():
        try:
            warp_int = int(warp_id)
        except (ValueError, TypeError):
            continue
        warp_ids.append(warp_int)
        if records:
            last_record = records[-1]
            if is_exit_instruction(last_record):
                completed_ids.append(warp_int)
            else:
                inprogress_ids.append(warp_int)

    if not warp_ids:
        return None

    min_warp = min(warp_ids)
    max_warp = max(warp_ids)

    observed_set = set(warp_ids)
    all_expected = set(range(0, max_warp + 1))
    missing_ids = sorted(all_expected - observed_set)

    return WarpSummary(
        total_observed=len(warp_ids),
        min_warp_id=min_warp,
        max_warp_id=max_warp,
        completed_warp_ids=sorted(completed_ids),
        inprogress_warp_ids=sorted(inprogress_ids),
        missing_warp_ids=missing_ids,
    )


def format_warp_summary_text(summary: WarpSummary) -> str:
    """
    Format warp summary as human-readable text.

    Args:
        summary: WarpSummary object

    Returns:
        Formatted text string suitable for terminal output
    """
    total = summary.total_observed
    completed = len(summary.completed_warp_ids)
    inprogress = len(summary.inprogress_warp_ids)
    missing = len(summary.missing_warp_ids)

    completed_pct = (completed / total * 100) if total > 0 else 0
    inprogress_pct = (inprogress / total * 100) if total > 0 else 0
    expected_count = summary.max_warp_id + 1
    missing_pct = (missing / expected_count * 100) if expected_count > 0 else 0

    completed_ranges = merge_to_ranges(summary.completed_warp_ids)
    inprogress_ranges = merge_to_ranges(summary.inprogress_warp_ids)
    missing_ranges = merge_to_ranges(summary.missing_warp_ids)

    lines = [
        "",
        "─" * 50,
        "Warp Summary",
        "─" * 50,
        f"  Total warps observed:   {total}",
        f"  Warp ID range:          {summary.min_warp_id} - {summary.max_warp_id}",
        "",
        f"  Completed (EXIT):       {completed:>6}  ({completed_pct:.1f}%)",
        f"    IDs: {format_ranges(completed_ranges)}",
        "",
        f"  In-progress:            {inprogress:>6}  ({inprogress_pct:.1f}%)",
        f"    IDs: {format_ranges(inprogress_ranges)}",
        "",
        f"  Missing (never seen):   {missing:>6}  ({missing_pct:.1f}%)",
        f"    IDs: {format_ranges(missing_ranges)}",
    ]
    return "\n".join(lines)


def warp_summary_to_dict(summary: WarpSummary) -> dict:
    """
    Convert WarpSummary to a dictionary for JSON output.

    Args:
        summary: WarpSummary object

    Returns:
        Dictionary representation suitable for JSON serialization
    """
    total = summary.total_observed
    completed = len(summary.completed_warp_ids)
    inprogress = len(summary.inprogress_warp_ids)
    missing = len(summary.missing_warp_ids)
    expected_count = summary.max_warp_id + 1

    return {
        "total_observed": total,
        "warp_id_range": [summary.min_warp_id, summary.max_warp_id],
        "completed": {
            "count": completed,
            "percentage": round(completed / total * 100, 1) if total > 0 else 0,
            "ranges": merge_to_ranges(summary.completed_warp_ids),
        },
        "in_progress": {
            "count": inprogress,
            "percentage": round(inprogress / total * 100, 1) if total > 0 else 0,
            "ranges": merge_to_ranges(summary.inprogress_warp_ids),
        },
        "missing": {
            "count": missing,
            "percentage": (
                round(missing / expected_count * 100, 1) if expected_count > 0 else 0
            ),
            "ranges": merge_to_ranges(summary.missing_warp_ids),
        },
    }
