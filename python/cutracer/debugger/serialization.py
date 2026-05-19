# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import json
from pathlib import Path

from cutracer.debugger.types import CudaKernelSample, TraceRecordType
from cutracer.types import TraceRecord


def samples_to_trace_records(samples: list[CudaKernelSample]) -> list[TraceRecord]:
    # Caller is expected to emit samples in monotonic sample_index order.
    # Within each sample we sort warps for stable trace_index assignment so
    # the analyzer sees per-warp records grouped consistently.
    records: list[TraceRecord] = []
    for sample in samples:
        for warp in sorted(
            sample.warps,
            key=lambda w: (
                w.identity.device,
                w.identity.sm,
                w.identity.cta,
                w.identity.warp_id,
            ),
        ):
            record: TraceRecord = {
                "type": TraceRecordType.OPCODE_ONLY,
                "cta": list(warp.identity.cta),
                "warp": warp.identity.warp_id,
                "pc": _normalize_pc(warp.pc),
                "sass": warp.sass,
                "trace_index": len(records),
            }
            if warp.identity.cuda_warp_slot is not None:
                record["cuda_warp_slot"] = warp.identity.cuda_warp_slot
            if warp.first_active_threadidx is not None:
                record["first_active_threadidx"] = list(warp.first_active_threadidx)
            if warp.active_mask is not None:
                record["active_mask"] = warp.active_mask
            if warp.pc_runtime is not None:
                record["pc_runtime"] = _normalize_pc(warp.pc_runtime)
            if warp.pc_address_space is not None:
                record["pc_address_space"] = warp.pc_address_space
            if warp.sass_context:
                record["sass_context"] = warp.sass_context
            if warp.operand_capture_lane is not None:
                record["operand_capture_lane"] = warp.operand_capture_lane
            if warp.registers:
                record["registers"] = warp.registers
            if warp.read_errors:
                record["read_errors"] = warp.read_errors
            records.append(record)
    return records


def write_samples_trace_file(path: Path, samples: list[CudaKernelSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for record in samples_to_trace_records(samples):
            output.write(json.dumps(record) + "\n")


def _normalize_pc(pc: str) -> str:
    try:
        return hex(int(pc, 16))
    except ValueError:
        return pc.lower()
