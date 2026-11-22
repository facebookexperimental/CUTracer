# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Read the online deadlock detector's warp status snapshots.

With ``-a deadlock_detection`` and a non-text trace format, CUTracer appends a
JSON snapshot to ``<trace basename>_warp_status_summary.ndjson`` next to the
trace whenever a hang check finds no progressing warp, and at a kernel or
no-data timeout. A snapshot holds the launch's grid and block, its finished,
active and never-executed warp id ranges, and the status, last instruction and
loop body of every active warp. Newer runtimes also record what wrote the
snapshot (``trigger``). The file is appended to across runs and a snapshot
names no run, so a snapshot is matched to a trace by launch id, grid and
block, and only the trailing snapshots of one run are counted.
"""

from __future__ import annotations

import bisect
import json
import re
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WARP_STATUS_SUFFIX = "_warp_status_summary.ndjson"
_TRACE_SUFFIXES = (".ndjson.zst", ".ndjson", ".clp", ".log")
_LAUNCH_ID_RE: re.Pattern[bytes] = re.compile(rb'"kernel_launch_id":\s*(\d+)')

Dims = tuple[int, int, int]
IdRanges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RuntimeWarp:
    """One active warp of a runtime snapshot."""

    warp_id: int
    cta: Dims
    status: str
    last_pc: int | None
    last_sass: str | None
    loop_pcs: tuple[int, ...]
    loop_sass: tuple[str, ...]


@dataclass(frozen=True)
class SnapshotTrigger:
    """What made the runtime write a snapshot.

    ``source`` is ``hang_check``, ``kernel_timeout`` or ``no_data_timeout``;
    ``terminating`` says the runtime then ended the run. ``stalled_checks``
    and ``stalled_ms`` are the consecutive hang checks without progress, and
    how long they lasted, when the snapshot was written.
    """

    source: str
    terminating: bool
    stalled_checks: int
    stalled_ms: int


@dataclass(frozen=True)
class RuntimeWarpStatus:
    """The last snapshot of one launch's runtime warp status.

    ``trigger`` is None for snapshots of runtimes that do not record it.
    """

    path: Path
    snapshot_count: int
    kernel_launch_id: int | None
    grid: Dims | None
    block: Dims | None
    total_warps: int | None
    finished_warps: int | None
    never_executed_warps: int | None
    never_executed_ranges: IdRanges
    warps: tuple[RuntimeWarp, ...]
    trigger: SnapshotTrigger | None = None

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(warp.status.lower() for warp in self.warps))

    def barrier_sites(self) -> list[tuple[int | None, str | None, int]]:
        """``(pc, sass, warps)`` per last instruction of BARRIER warps."""
        sites = Counter(
            (warp.last_pc, warp.last_sass)
            for warp in self.warps
            if warp.status == "BARRIER"
        )
        return [(pc, sass, count) for (pc, sass), count in _by_count(sites)]

    def loop_sites(self) -> list[tuple[tuple[int, ...], tuple[str, ...], int]]:
        """``(pcs, sass, warps)`` per loop body of LOOPING warps.

        The runtime records a body from wherever each warp's history ring
        starts, so bodies are rotated to begin at their lowest PC first.
        """
        sites = Counter(
            _rotate_to_lowest_pc(warp.loop_pcs, warp.loop_sass)
            for warp in self.warps
            if warp.status == "LOOPING"
        )
        return [(pcs, sass, count) for (pcs, sass), count in _by_count(sites)]

    def launch_mismatch(
        self, launch_id: int | None, grid: Dims | None, block: Dims | None
    ) -> str | None:
        """Why this snapshot is not of the given launch, or None if it may be."""
        for name, ours, theirs in (
            ("launch id", self.kernel_launch_id, launch_id),
            ("grid", self.grid, grid),
            ("block", self.block, block),
        ):
            if ours is not None and theirs is not None and ours != theirs:
                return f"{name} {ours} in the warp status file, {theirs} in the trace"
        return None


def _by_count(counter: Counter[Any]) -> list[tuple[Any, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], repr(item[0])))


def _rotate_to_lowest_pc(
    pcs: tuple[int, ...], sass: tuple[str, ...]
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not pcs:
        return pcs, sass
    start = pcs.index(min(pcs))
    return pcs[start:] + pcs[:start], sass[start:] + sass[:start]


def warp_status_summary_path(trace_path: Path) -> Path:
    """The warp status file CUTracer writes next to ``trace_path``."""
    name = trace_path.name
    for suffix in _TRACE_SUFFIXES:
        if name.endswith(suffix):
            return trace_path.with_name(name[: -len(suffix)] + WARP_STATUS_SUFFIX)
    return trace_path.with_name(name + WARP_STATUS_SUFFIX)


def parse_id_ranges(ranges: object) -> IdRanges:
    """Read the runtime's ``["0-7", "12"]`` warp id ranges as ``(first, last)``."""
    if ranges is None:
        return ()
    if not isinstance(ranges, list):
        raise ValueError(f"warp id ranges are not a list: {ranges!r}")
    spans = []
    for item in ranges:
        if not isinstance(item, str):
            raise ValueError(f"warp id range is not a string: {item!r}")
        first, _, last = item.partition("-")
        start, end = int(first), int(last or first)
        if start < 0 or end < start:
            raise ValueError(f"invalid warp id range: {item!r}")
        spans.append((start, end))
    return tuple(spans)


def _object(value: object, what: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} is not an object: {value!r}")
    return value


def _dims(value: object) -> Dims | None:
    if not isinstance(value, Mapping):
        return None
    x, y, z = (value.get(axis) for axis in "xyz")
    if not (isinstance(x, int) and isinstance(y, int) and isinstance(z, int)):
        return None
    return (x, y, z) if min(x, y, z) > 0 else None


def _instruction(value: object) -> tuple[int | None, str | None]:
    if not isinstance(value, Mapping):
        return None, None
    pc = value.get("pc")
    sass = value.get("mnemonic")
    return (
        pc if isinstance(pc, int) else None,
        sass if isinstance(sass, str) and sass != "UNKNOWN" else None,
    )


def _warp(entry: Mapping[str, Any]) -> RuntimeWarp:
    cta = entry["cta"]
    last_pc, last_sass = _instruction(entry.get("last_instruction"))
    body = (entry.get("loop_body") or {}).get("instructions") or []
    loop = [_instruction(instruction) for instruction in body]
    return RuntimeWarp(
        warp_id=int(entry["warp_id"]),
        cta=(int(cta["x"]), int(cta["y"]), int(cta["z"])),
        status=str(entry.get("status", "UNKNOWN")),
        last_pc=last_pc,
        last_sass=last_sass,
        loop_pcs=tuple(pc for pc, _ in loop if pc is not None),
        loop_sass=tuple(sass or "" for pc, sass in loop if pc is not None),
    )


def _trigger(value: object) -> SnapshotTrigger | None:
    if value is None:
        return None
    trigger = _object(value, "trigger")
    source, terminating = trigger.get("source"), trigger.get("terminating")
    checks, stalled_ms = trigger.get("stalled_checks"), trigger.get("stalled_ms")
    if not (
        isinstance(source, str)
        and isinstance(terminating, bool)
        and type(checks) is int
        and type(stalled_ms) is int
    ):
        raise ValueError(f"malformed trigger: {trigger}")
    return SnapshotTrigger(source, terminating, checks, stalled_ms)


def _optional_int(summary: Mapping[str, Any], key: str) -> int | None:
    value = summary.get(key)
    return value if isinstance(value, int) else None


def parse_snapshot(
    snapshot: Mapping[str, Any], path: Path, snapshot_count: int
) -> RuntimeWarpStatus:
    """Raises ValueError when the snapshot does not have the writer's shape."""
    stats = _object(snapshot.get("warp_statistics"), "warp_statistics")
    summary = _object(stats.get("summary"), "warp_statistics.summary")
    ranges = _object(stats.get("warp_id_ranges"), "warp_statistics.warp_id_ranges")
    entries = snapshot.get("active_warps")
    if not isinstance(entries, list):
        raise ValueError("active_warps is not a list")
    try:
        warps = tuple(_warp(entry) for entry in entries)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"malformed active warp entry: {error}") from error
    launch_id = snapshot.get("kernel_launch_id")
    return RuntimeWarpStatus(
        path=path,
        snapshot_count=snapshot_count,
        kernel_launch_id=launch_id if isinstance(launch_id, int) else None,
        grid=_dims(stats.get("grid")),
        block=_dims(stats.get("block")),
        total_warps=_optional_int(summary, "total_warps"),
        finished_warps=_optional_int(summary, "finished_warps"),
        never_executed_warps=_optional_int(summary, "never_executed_warps"),
        never_executed_ranges=parse_id_ranges(ranges.get("never_executed")),
        warps=warps,
        trigger=_trigger(snapshot.get("trigger")),
    )


def _decode(line: bytes) -> dict[str, Any] | None:
    try:
        snapshot = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return (
        snapshot if isinstance(snapshot, dict) and "active_warps" in snapshot else None
    )


def _progress(snapshot: Mapping[str, Any]) -> tuple[float, float]:
    """The longest inactive time and loop repeat count among active warps."""
    inactive, repeat = 0.0, 0.0
    warps = snapshot.get("active_warps")
    for warp in warps if isinstance(warps, list) else []:
        if not isinstance(warp, Mapping):
            continue
        seconds = warp.get("inactive_secs")
        loop = warp.get("loop")
        count = loop.get("repeat") if isinstance(loop, Mapping) else None
        if isinstance(seconds, (int, float)):
            inactive = max(inactive, float(seconds))
        if isinstance(count, (int, float)):
            repeat = max(repeat, float(count))
    return inactive, repeat


def _run_length(lines: list[bytes], last: int, snapshot: Mapping[str, Any]) -> int:
    """How many snapshots, ending at ``lines[last]``, one run wrote.

    Hang checks of one run come at least a second apart, and the stuck warps'
    inactive time or loop count grows between them; an earlier snapshot that
    is not behind the next one, or that ended its run, was written by an
    earlier run.
    """
    count, later = 1, _progress(snapshot)
    for line in reversed(lines[:last]):
        earlier = _decode(line)
        if earlier is None or _progress(earlier) >= later or _ended_run(earlier):
            break
        count, later = count + 1, _progress(earlier)
    return count


def _ended_run(snapshot: Mapping[str, Any]) -> bool:
    trigger = snapshot.get("trigger")
    return isinstance(trigger, Mapping) and trigger.get("terminating") is True


def _launch_id_of(line: bytes) -> int | None:
    found = _LAUNCH_ID_RE.findall(line)
    return int(found[-1]) if found else None


def load_runtime_warp_status(
    path: Path, launch_id: int | None = None
) -> RuntimeWarpStatus:
    """Parse the last well-formed snapshot of ``launch_id`` (of any launch
    when None); later malformed snapshots of the same run are skipped.

    ``snapshot_count`` counts the snapshots the same run wrote up to the one
    returned.

    Raises:
        OSError: The file cannot be read.
        ValueError: No line holds a well-formed snapshot of the launch.
    """
    lines = [
        line
        for line in path.read_bytes().splitlines()
        if launch_id is None or _launch_id_of(line) == launch_id
    ]
    malformed: ValueError | None = None
    later: dict[str, Any] | None = None
    for index in range(len(lines) - 1, -1, -1):
        snapshot = _decode(lines[index])
        if later is not None and (
            snapshot is None
            or _ended_run(snapshot)
            or _progress(snapshot) >= _progress(later)
        ):
            # Another run wrote it, by the rule of _run_length.
            break
        if snapshot is None:
            # A run killed mid-write leaves a torn last line.
            continue
        try:
            return parse_snapshot(snapshot, path, _run_length(lines, index, snapshot))
        except ValueError as error:
            malformed, later = malformed or error, snapshot
    which = "" if launch_id is None else f" of launch {launch_id}"
    if malformed is not None:
        raise ValueError(
            f"{path}: no well-formed warp status snapshot{which}: {malformed}"
        ) from malformed
    raise ValueError(f"{path}: no warp status snapshot{which}")


def find_runtime_warp_status(
    trace_path: Path, launch_id: int | None = None
) -> RuntimeWarpStatus | None:
    """The snapshot next to ``trace_path``, or None when there is no file."""
    path = warp_status_summary_path(trace_path)
    if not path.is_file():
        return None
    return load_runtime_warp_status(path, launch_id)


def header_launch_id(header: Mapping[str, Any] | None) -> int | None:
    """The launch id a trace's kernel_metadata header records."""
    if not header:
        return None
    config = header.get("capture_config")
    launch_id = config.get("grid_launch_id") if isinstance(config, Mapping) else None
    if launch_id is None:
        launch_id = header.get("grid_launch_id")
    return launch_id if isinstance(launch_id, int) else None


def _header_dims(header: Mapping[str, Any], key: str) -> Dims | None:
    value = header.get(key)
    if not isinstance(value, list) or len(value) != 3:
        return None
    x, y, z = value
    if not (isinstance(x, int) and isinstance(y, int) and isinstance(z, int)):
        return None
    return (x, y, z)


def _never_executed(ranges: IdRanges, warps: Collection[int]) -> list[int]:
    starts = [first for first, _ in ranges]
    found = []
    for warp in warps:
        index = bisect.bisect_right(starts, warp) - 1
        if index >= 0 and ranges[index][1] >= warp:
            found.append(warp)
    return found


def runtime_status_for_trace(
    trace_path: Path,
    header: Mapping[str, Any] | None,
    observed_warps: Collection[int] = (),
) -> RuntimeWarpStatus | None:
    """The snapshot of the traced launch next to ``trace_path``, if any.

    A snapshot that counts as never executed a warp the trace shows running
    was written by another run of the launch.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file is malformed or holds another launch or run.
    """
    launch_id = header_launch_id(header)
    runtime = find_runtime_warp_status(trace_path, launch_id)
    if runtime is None or not header:
        return runtime
    mismatch = runtime.launch_mismatch(
        launch_id, _header_dims(header, "grid"), _header_dims(header, "block")
    )
    if mismatch is not None:
        raise ValueError(f"{runtime.path.name} is not of the traced launch: {mismatch}")
    ran = _never_executed(runtime.never_executed_ranges, observed_warps)
    if ran:
        raise ValueError(
            f"{runtime.path.name} is from another run: it counts {len(ran)} warps "
            f"the trace shows running as never executed (e.g. warp {min(ran)})"
        )
    return runtime
