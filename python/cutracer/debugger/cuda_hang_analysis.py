# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import re
from typing import cast, TypedDict

from cutracer.analyze.fb.deadlock.detection import DeadlockAnalyzer
from cutracer.analyze.fb.deadlock.types import WarpStatus
from cutracer.debugger.serialization import _normalize_pc, samples_to_trace_records
from cutracer.debugger.types import (
    CommonLoopSummary,
    CudaKernelSample,
    CudaWarpIdentity,
    CudaWarpSample,
    HangAnalysisResult,
    HangVerdict,
    LoopInstruction,
)


class DeadlockSummary(TypedDict):
    total_warps: int
    status_counts: dict[str, int]
    is_potential_hang: bool
    is_soft_hang: bool
    total_records: int
    trace_appears_truncated: bool


COORD_RE: re.Pattern[str] = re.compile(r"\(\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\)")
HEX_RE: re.Pattern[str] = re.compile(r"0x[0-9a-fA-F]+")
DEVICE_SM_RE: re.Pattern[str] = re.compile(
    r"\bDevice\s+(?P<device>\d+)\b.*\bSM\s+(?P<sm>\d+)\b"
)
CUDA_DEBUGGER_ATTACH_ERROR_MARKERS = (
    "cudbgapiattach",
    "hit an internal error while attaching to the application",
)
CUDA_DEBUGGER_INJECTION_ERROR_MARKERS = (
    "cudbgreportdriverinternalerror",
    "initializeinjection",
    "the cuda driver has hit an internal error",
)


def parse_active_kernel_name(kernels_output: str) -> str | None:
    """Best-effort extraction of the currently selected kernel name.

    Example:
        `* 1  Running triton__kernel_42()` -> `triton__kernel_42()`.

    Selected (`* `) lines are preferred over plain data lines.
    """
    candidates = sorted(
        (line.strip() for line in kernels_output.splitlines() if line.strip()),
        key=lambda s: 0 if s.startswith("*") else 1,
    )
    for line in candidates:
        if line.startswith("*") or any(ch.isdigit() for ch in line[:4]):
            name = _extract_kernel_name(line)
            if name is not None:
                return name
    return None


def parse_cuda_warps_output(
    warps_output: str,
    kernel_name: str | None,
    sample_index: int,
) -> CudaKernelSample:
    """Parse `info cuda warps` output into normalized per-warp samples.

    `Wp` is cuda-gdb's physical warp slot. The logical warp id is derived
    from `First Active ThreadIdx.x // 32`.
    """
    current_device = 0
    current_sm = 0
    resolved_kernel = kernel_name or "unknown"
    warps: list[CudaWarpSample] = []

    for line in warps_output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Wp"):
            continue

        device_match = DEVICE_SM_RE.search(stripped)
        if device_match is not None:
            current_device = int(device_match.group("device"))
            current_sm = int(device_match.group("sm"))
            continue

        warp_sample = _parse_warp_line(
            stripped,
            kernel_name=resolved_kernel,
            device=current_device,
            sm=current_sm,
            sample_index=sample_index,
        )
        if warp_sample is not None:
            warps.append(warp_sample)

    return CudaKernelSample(
        sample_index=sample_index,
        kernel_name=kernel_name,
        warps=warps,
    )


def detect_cuda_debugger_internal_error(backtrace_output: str) -> str | None:
    """Recognize cuda-gdb attach/injection failures from a GDB backtrace."""
    normalized = backtrace_output.lower()
    if any(marker in normalized for marker in CUDA_DEBUGGER_ATTACH_ERROR_MARKERS):
        return (
            "cuda-gdb failed while attaching to the live CUDA process; "
            "no active warps could be sampled."
        )
    if any(marker in normalized for marker in CUDA_DEBUGGER_INJECTION_ERROR_MARKERS):
        return (
            "cuda-gdb hit a CUDA driver internal error during debugger injection; "
            "no active warps could be sampled."
        )
    return None


def analyze_cuda_samples(samples: list[CudaKernelSample]) -> HangAnalysisResult:
    """Convert debugger snapshots into CUTracer records and classify the hang."""
    kernel_name = next((s.kernel_name for s in samples if s.kernel_name), None)
    records = samples_to_trace_records(samples)
    if not records:
        return _make_no_records_result(kernel_name, len(samples), bool(samples))

    analyzer = DeadlockAnalyzer()
    for record in records:
        analyzer.process_record(record)

    summary = cast(DeadlockSummary, analyzer.get_summary())
    active_statuses = [
        status
        for status in analyzer.warp_status.values()
        if status != WarpStatus.EXITED
    ]
    if not active_statuses:
        return _make_all_exited_result(kernel_name, len(samples), summary)

    is_hang = summary["is_potential_hang"] or summary["is_soft_hang"]
    notes = _build_analysis_notes(summary, is_hang)

    return HangAnalysisResult(
        verdict=_classify_verdict(active_statuses, is_hang),
        kernel_name=kernel_name,
        sample_count=len(samples),
        total_warps=summary["total_warps"],
        status_counts=summary["status_counts"],
        common_loops=_serialize_common_loops(analyzer),
        notes=notes,
    )


def render_hang_analysis(result: HangAnalysisResult) -> str:
    """Render a concise human-readable summary for debugger output."""
    lines = [f"Verdict: {result.verdict.value}"]
    if result.kernel_name:
        lines.append(f"Kernel: {result.kernel_name}")
    if result.sample_count:
        lines.append(f"Samples: {result.sample_count}")
    lines.append(f"Warps observed: {result.total_warps}")

    if result.status_counts:
        nonzero_counts = ", ".join(
            f"{status}={count}"
            for status, count in result.status_counts.items()
            if count > 0
        )
        lines.append(f"Status counts: {nonzero_counts or 'none'}")

    for note in result.notes:
        lines.append(f"Note: {note}")

    if result.common_loops:
        top_loop = result.common_loops[0]
        lines.append(
            f"Top loop: period={top_loop['period']} warps={top_loop['warp_count']}"
        )
        if top_loop["warps"]:
            lines.append(f"Loop cohort: {', '.join(top_loop['warps'][:3])}")
        if top_loop["loop_instructions"]:
            loop_pcs = ", ".join(
                instruction["pc"] for instruction in top_loop["loop_instructions"]
            )
            lines.append(f"Loop PCs: {loop_pcs}")

    return "\n".join(lines)


def _classify_verdict(
    active_statuses: list[WarpStatus],
    is_hang: bool,
) -> HangVerdict:
    if not is_hang:
        return HangVerdict.NO_HANG

    has_looping = any(status == WarpStatus.LOOPING for status in active_statuses)
    has_barrier = any(status == WarpStatus.BARRIER for status in active_statuses)

    if has_looping and has_barrier:
        return HangVerdict.MIXED
    if has_barrier:
        return HangVerdict.BARRIER
    if has_looping:
        return HangVerdict.LOOPING
    return HangVerdict.OUT_OF_SCOPE


def _make_no_records_result(
    kernel_name: str | None,
    sample_count: int,
    saw_samples: bool,
) -> HangAnalysisResult:
    return HangAnalysisResult(
        verdict=HangVerdict.NO_ACTIVE_KERNEL,
        kernel_name=kernel_name,
        sample_count=sample_count,
        total_warps=0,
        status_counts={},
        notes=[
            (
                "No active CUDA warps were captured during sampling."
                if saw_samples
                else "No debugger samples were collected."
            )
        ],
    )


def _make_all_exited_result(
    kernel_name: str | None,
    sample_count: int,
    summary: DeadlockSummary,
) -> HangAnalysisResult:
    return HangAnalysisResult(
        verdict=HangVerdict.NO_ACTIVE_KERNEL,
        kernel_name=kernel_name,
        sample_count=sample_count,
        total_warps=summary["total_warps"],
        status_counts=summary["status_counts"],
        notes=["All observed warps exited before the sampling window ended."],
    )


def _build_analysis_notes(summary: DeadlockSummary, is_hang: bool) -> list[str]:
    notes: list[str] = []
    if summary["is_soft_hang"]:
        notes.append(
            "Trace view is truncated; hang verdict is based on the stuck active warps."
        )
    elif summary["trace_appears_truncated"]:
        notes.append(
            "Trace view is truncated; some non-barrier warps may still be ambiguous."
        )
    if not is_hang:
        notes.append("At least one active warp still appears to be progressing.")
    return notes


def _serialize_common_loops(analyzer: DeadlockAnalyzer) -> list[CommonLoopSummary]:
    serialized_groups: list[CommonLoopSummary] = []
    for group in analyzer.find_common_loops():
        serialized_groups.append(
            {
                "signature": int(group["signature"]),
                "period": int(group["period"]),
                "warp_count": int(group["warp_count"]),
                "warps": [f"cta={w.cta} warp={w.warp_id}" for w in group["warps"]],
                "loop_instructions": _serialize_loop_instructions(
                    group["loop_instructions"]
                ),
            }
        )
    return serialized_groups


def _serialize_loop_instructions(
    instructions: list[dict[str, object]],
) -> list[LoopInstruction]:
    serialized: list[LoopInstruction] = []
    for instruction in instructions:
        serialized.append(
            {
                "pc": str(instruction["pc"]),
                "sass": str(instruction["sass"]),
            }
        )
    return serialized


def _parse_warp_line(
    line: str,
    kernel_name: str,
    device: int,
    sm: int,
    sample_index: int,
) -> CudaWarpSample | None:
    """Parse a single `info cuda warps` row into a warp sample.

    Example:
        `* 0 0xffffffff 0x00000000 0x0000000000000100 1 (0,0,0) (0,0,0)`
        -> `warp_id=0`, `pc=0x100`, `cta=(0, 0, 0)`.
    """
    normalized = line.lstrip("*").strip()
    parts = normalized.split()
    if not parts or not parts[0].isdigit():
        return None

    hex_values = HEX_RE.findall(normalized)
    coord_values = COORD_RE.findall(normalized)
    if len(hex_values) < 3 or len(coord_values) < 2:
        return None
    first_active_threadidx = _parse_coord_tuple(coord_values[1])

    return CudaWarpSample(
        identity=CudaWarpIdentity(
            kernel_name=kernel_name,
            device=device,
            sm=sm,
            cta=_parse_coord_tuple(coord_values[0]),
            warp_id=first_active_threadidx[0] // 32,
            cuda_warp_slot=int(parts[0]),
        ),
        sample_index=sample_index,
        pc=_normalize_pc(hex_values[2]),
        first_active_threadidx=first_active_threadidx,
        active_mask=_normalize_mask(hex_values[0]),
    )


def _extract_kernel_name(line: str) -> str | None:
    for token in reversed(line.replace(",", " ").split()):
        cleaned = token.strip()
        if cleaned in {"*", "Kernel", "Active", "Inactive", "Running", "Sleeping"}:
            continue
        if cleaned.isdigit():
            continue
        if any(marker in cleaned for marker in ("<", ">", "(", ")", "::")):
            return cleaned
        if any(ch.isalpha() for ch in cleaned):
            return cleaned
    return None


def _parse_coord_tuple(value: str) -> tuple[int, int, int]:
    a, b, c = (int(p) for p in value.strip("()").split(","))
    return (a, b, c)


def _normalize_mask(mask: str) -> str:
    return f"0x{int(mask, 16) & 0xFFFFFFFF:08x}"
