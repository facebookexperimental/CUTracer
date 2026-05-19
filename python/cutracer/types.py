# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Shared type definitions for CUTracer trace analysis.

Provides core types used across multiple analysis modules:
- WarpKey: Unique warp identifier (frozen dataclass, hashable)
- TraceRecord: Typed view of NDJSON trace record (TypedDict, runtime is dict)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict, Union


@dataclass(frozen=True)
class WarpKey:
    """
    Uniquely identifies a warp by CTA coordinates and warp ID.

    frozen=True makes it hashable, usable as dict key / set element.
    Corresponds to C++ WarpKey in analysis.h.
    """

    cta: tuple[int, int, int]
    warp_id: int


class TraceRecord(TypedDict, total=False):
    """
    Type definition for a CUTracer trace record.

    At runtime this is a plain dict (json.loads() output is directly compatible).
    total=False because different record types (reg_trace / mem_trace) have
    different field sets — not all fields are present on every record.

    Some fields are *derived*: they are not present in the raw NDJSON and
    are synthesized by `TraceReader` while iterating (see the "Derived"
    section at the bottom). Downstream consumers can treat them like any
    other field.
    """

    # Common fields across record types
    type: str  # "reg_trace" | "mem_trace" | "kernel_metadata" | "opcode_only" | "tma_trace"
    ctx: str  # CUDA context
    sass: str  # SASS instruction text
    trace_index: int  # Sequence number in trace
    timestamp: int
    grid_launch_id: int
    cta: list[int]  # [x, y, z]
    warp: int  # Warp ID
    opcode_id: int
    pc: str  # Hex string "0x..."

    # cuda-gdb opcode_only specific fields. These are emitted by the debugger
    # sampler, not by NVBit reg_trace/mem_trace/tma_trace records.
    cuda_warp_slot: int
    first_active_threadidx: list[int]
    active_mask: str
    pc_runtime: str
    pc_address_space: str
    sass_context: str

    # reg_trace specific
    regs: list[list[int]]
    regs_indices: list[int]
    uregs: list[int]
    uregs_indices: list[int]

    # mem_trace specific
    addrs: list[int]  # 32 memory addresses (one per thread in warp)

    # tma_trace specific
    desc_addr: str  # TMA descriptor address (hex string "0x...")
    desc_raw: list[Union[int, str]]  # Raw descriptor bytes (16 hex strings or ints)
    tma_transfer_info: dict[str, Any]  # NVBit 1.8 TMA transfer info (src/dst)

    # kernel_metadata specific
    mangled_name: str
    unmangled_name: str
    kernel_checksum: str
    nregs: int
    shmem_static: int
    shmem_dynamic: int
    grid: list[int]
    block: list[int]
    cubin_path: str
    func_addr: str
    sm_family: int  # SM architecture family (e.g. 90 = Hopper, 100 = Blackwell)

    # kernel_launch specific (kernel events file)
    kernel_launch_id: int
    kernel_name: str
    shmem: int
    stream_id: int
    callstack_id: str

    # Derived (synthesized by TraceReader, not present in raw NDJSON)
    caller: str  # Innermost frame from callstack_def resolution (kernel_launch only)
