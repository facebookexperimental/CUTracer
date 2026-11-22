# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Which executed instructions and warps a capture leaves out of its trace.

Read from a trace's ``kernel_metadata`` header. With register tracing,
``capture_config.reg_trace_instruction_count`` below
``root_instruction_count`` is what decides that instructions are missing; the
window, PC and category fields only name the cause. A warp mask drops whole
warps on the device and leaves no count behind.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

ScopeReason = Literal[
    "no_instruction_trace",
    "plan_capture",
    "instruction_window",
    "pc_filter",
    "instruction_categories",
    "partial_instrumentation",
    "warp_mask",
]

SCOPE_TEXT: dict[ScopeReason, str] = {
    "no_instruction_trace": "the capture recorded no instruction trace (reg_trace or opcode_only)",
    "plan_capture": "a capture plan chose the traced instructions",
    "instruction_window": "the capture traced an instruction window only",
    "pc_filter": "the capture traced selected PCs only",
    "instruction_categories": "the capture traced selected instruction categories only",
    "partial_instrumentation": "not every instruction of the kernel was instrumented",
    "warp_mask": "the capture traced a subset of each CTA's warps",
}


def warp_mask_covers_cta(mask: object, warps_per_cta: int | None) -> bool:
    """Whether ``trace_warp_mask`` keeps every warp of a CTA."""
    if mask in (None, 0):
        return True
    if not isinstance(mask, int) or not warps_per_cta or warps_per_cta <= 0:
        return False
    full = (1 << min(warps_per_cta, 32)) - 1
    return mask & full == full


def _named_instruction_filters(
    metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> list[ScopeReason]:
    reasons: list[ScopeReason] = []
    begin, end = config.get("instr_begin"), config.get("instr_end")
    root = config.get("root_instruction_count")
    if isinstance(begin, int) and isinstance(end, int) and isinstance(root, int):
        if begin > 0 or end < root:
            reasons.append("instruction_window")
    if metadata.get("trace_pc_offsets") is not None:
        reasons.append("pc_filter")
    if (
        metadata.get("instr_categories") is not None
        or metadata.get("trace_instr_categories") is not None
    ):
        reasons.append("instruction_categories")
    return reasons


def _instruction_scope(metadata: Mapping[str, Any]) -> list[ScopeReason]:
    config = metadata.get("capture_config")
    config = config if isinstance(config, Mapping) else {}
    modes = metadata.get("instrument_modes")
    modes = set(modes) if isinstance(modes, list) else None
    if config.get("protocol") is not None:
        return ["plan_capture"]
    if modes is not None and not modes & {"reg_trace", "opcode_only"}:
        return ["no_instruction_trace"]
    named = _named_instruction_filters(metadata, config)
    traced = config.get("reg_trace_instruction_count")
    root = config.get("root_instruction_count")
    counted = (
        modes is not None
        and "opcode_only" not in modes
        and isinstance(traced, int)
        and isinstance(root, int)
    )
    if not counted:
        return named
    if traced >= root:
        return []
    return named or ["partial_instrumentation"]


def capture_scope_reasons(
    metadata: Mapping[str, Any] | None, warps_per_cta: int | None
) -> list[ScopeReason]:
    """Capture options that leave executed instructions or warps untraced."""
    if not metadata:
        return []
    reasons = _instruction_scope(metadata)
    if not warp_mask_covers_cta(metadata.get("trace_warp_mask"), warps_per_cta):
        reasons.append("warp_mask")
    return reasons
