# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the capture filters read from a trace's kernel_metadata header."""

import unittest

from cutracer.analyze.capture_scope import capture_scope_reasons


def _header(
    config: dict[str, object] | None = None, **fields: object
) -> dict[str, object]:
    capture: dict[str, object] = {
        "instr_begin": 0,
        "instr_end": 0xFFFFFFFF,
        "root_instruction_count": 100,
        "reg_trace_instruction_count": 100,
        "instrumentation_errors": 0,
    }
    capture.update(config or {})
    return {"instrument_modes": ["reg_trace"], "capture_config": capture, **fields}


# (case, header, warps per CTA, reasons)
_CASES: list[tuple[str, dict[str, object], int | None, list[str]]] = [
    ("full capture", _header(), 4, []),
    (
        "instruction window",
        _header(
            {"instr_begin": 20, "instr_end": 60, "reg_trace_instruction_count": 40}
        ),
        4,
        ["instruction_window"],
    ),
    ("window past the last instruction", _header({"instr_end": 500}), 4, []),
    (
        "filters name the cause of missing instructions",
        _header(
            {"reg_trace_instruction_count": 10},
            trace_pc_offsets=[16],
            trace_instr_categories=["sync"],
        ),
        4,
        ["pc_filter", "instruction_categories"],
    ),
    ("filter that kept every instruction", _header(trace_pc_offsets=[16]), 4, []),
    (
        "unnamed shortfall",
        _header({"reg_trace_instruction_count": 98, "instrumentation_errors": 2}),
        4,
        ["partial_instrumentation"],
    ),
    (
        "opcode_only capture is judged by its filters",
        _header({"reg_trace_instruction_count": 0}, instrument_modes=["opcode_only"]),
        4,
        [],
    ),
    (
        "no instruction trace",
        _header(instrument_modes=["mem_addr_trace"]),
        4,
        ["no_instruction_trace"],
    ),
    (
        "plan capture",
        {
            "instrument_modes": ["reg_trace"],
            "capture_config": {"protocol": "cutracer.plan-runtime/1"},
        },
        4,
        ["plan_capture"],
    ),
    ("warp mask", _header(trace_warp_mask=0x1), 4, ["warp_mask"]),
    ("mask that keeps every warp", _header(trace_warp_mask=0xF), 4, []),
    ("mask of a CTA size 0", _header(trace_warp_mask=0xF), 0, ["warp_mask"]),
    ("mask of an unknown CTA size", _header(trace_warp_mask=0xF), None, ["warp_mask"]),
    ("no mask, unknown CTA size", _header(), None, []),
    ("header without capture_config", {"grid": [1, 1, 1]}, 4, []),
]


class CaptureScopeTest(unittest.TestCase):
    def test_reasons(self) -> None:
        for case, header, warps_per_cta, reasons in _CASES:
            with self.subTest(case):
                self.assertEqual(capture_scope_reasons(header, warps_per_cta), reasons)
