# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict


class TraceRecordType(str, Enum):
    # TraceRecord["type"] value emitted by the debugger pipeline.
    # Matches the "opcode_only" schema in cutracer/validation/schemas/.
    OPCODE_ONLY = "opcode_only"


def is_debugger_trace_record(record: Mapping[str, object]) -> bool:
    record_type = record.get("type")
    source_type = record.get("source_type", "")
    return (
        record_type == TraceRecordType.OPCODE_ONLY
        or record_type == TraceRecordType.OPCODE_ONLY.value
    ) and str(source_type).startswith("gdb_")


class HangVerdict(Enum):
    BARRIER = "barrier"
    LOOPING = "looping"
    MIXED = "mixed"
    NO_ACTIVE_KERNEL = "no_active_kernel"
    NO_HANG = "no_hang"
    OUT_OF_SCOPE = "out_of_scope"


class LoopInstruction(TypedDict):
    pc: str
    sass: str


class CommonLoopSummary(TypedDict):
    signature: int
    period: int
    warp_count: int
    warps: list[str]
    loop_instructions: list[LoopInstruction]


@dataclass(frozen=True)
class CudaWarpIdentity:
    kernel_name: str
    device: int
    sm: int
    cta: tuple[int, int, int]
    warp_id: int
    cuda_warp_slot: int | None = None


@dataclass
class CudaWarpSample:
    identity: CudaWarpIdentity
    sample_index: int
    pc: str
    sass: str = ""
    first_active_threadidx: tuple[int, int, int] | None = None
    active_mask: str | None = None
    # `pc`/`sass` are the instruction fed to analyzers. When resolvable,
    # `pc` is the kernel-relative form of `pc_runtime`; `sass_context` is the
    # raw disassembly window used to prove nearby blockers.
    pc_runtime: str | None = None
    pc_address_space: str | None = None
    observed_pc: str | None = None
    observed_pc_runtime: str | None = None
    observed_sass: str | None = None
    source_type: str | None = None
    pc_semantics: str | None = None
    cuda_focus_status: str | None = None
    sass_context: str = ""
    # Register evidence captured only when cuda-hang-detect is invoked with
    # --capture-operands.
    operand_capture_lane: int | None = None
    registers: dict[str, int] = field(default_factory=dict)
    read_errors: list[str] = field(default_factory=list)
    memory_addresses: list[dict[str, object]] = field(default_factory=list)


@dataclass
class CudaKernelSample:
    sample_index: int
    kernel_name: str | None
    warps: list[CudaWarpSample] = field(default_factory=list)


@dataclass
class HangAnalysisResult:
    verdict: HangVerdict
    kernel_name: str | None
    sample_count: int
    total_warps: int
    status_counts: dict[str, int]
    common_loops: list[CommonLoopSummary] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
