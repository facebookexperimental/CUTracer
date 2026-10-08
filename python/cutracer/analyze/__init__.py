# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer analyze module.

Provides analysis algorithms that produce derived insights:
- warp-summary: Warp execution status analysis (completed, in-progress, missing)
- tma: TMA descriptor analysis for data flow modeling
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cutracer.query.warp_summary import (
        compute_warp_summary,
        format_ranges,
        format_warp_summary_text,
        is_exit_instruction,
        is_exit_sass,
        merge_to_ranges,
        warp_completed,
        warp_summary_to_dict,
        WarpSummary,
    )

__all__ = [
    # Warp summary
    "WarpSummary",
    "is_exit_sass",
    "is_exit_instruction",
    "warp_completed",
    "merge_to_ranges",
    "format_ranges",
    "compute_warp_summary",
    "format_warp_summary_text",
    "warp_summary_to_dict",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from cutracer.query import warp_summary

        value = getattr(warp_summary, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
