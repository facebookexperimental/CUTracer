# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for reading the online deadlock detector's warp status file."""

import json
import tempfile
import unittest
from pathlib import Path

from cutracer.analyze.runtime_warp_status import (
    find_runtime_warp_status,
    load_runtime_warp_status,
    parse_id_ranges,
    runtime_status_for_trace,
    SnapshotTrigger,
    warp_status_summary_path,
)


def snapshot(
    warps: list[dict[str, object]], launch_id: int = 8, grid_x: int = 4
) -> str:
    """One line as ``write_warp_status_json`` writes it (sorted keys)."""
    return json.dumps(
        {
            "active_warps": warps,
            "kernel_launch_id": launch_id,
            "warp_statistics": {
                "block": {"x": 64, "y": 1, "z": 1},
                "grid": {"x": grid_x, "y": 1, "z": 1},
                "summary": {
                    "active_warps": 3,
                    "finished_warps": 2,
                    "never_executed_warps": 3,
                    "total_warps": 8,
                },
                "warp_id_ranges": {
                    "active": ["2-4"],
                    "finished": ["0-1"],
                    "never_executed": ["5-7"],
                },
            },
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def barrier_warp(warp_id: int, pc: int) -> dict[str, object]:
    return {
        "warp_id": warp_id,
        "cta": {"x": warp_id // 2, "y": 0, "z": 0},
        "status": "BARRIER",
        "loop": {"period": 0, "repeat": 1},
        "last_instruction": {
            "pc": pc,
            "offset_hex": hex(pc),
            "mnemonic": "BAR.SYNC.DEFER_BLOCKING 0x0 ;",
            "has_mem": False,
        },
    }


def looping_warp(warp_id: int, first_pc: int = 0x3F0) -> dict[str, object]:
    body = [
        {"pc": 0x3F0, "mnemonic": "ATOMG.E.CAS ;"},
        {"pc": 0x420, "mnemonic": "@P0 BRA 0x3f0 ;"},
    ]
    if first_pc != body[0]["pc"]:
        body.reverse()
    return {
        "warp_id": warp_id,
        "cta": {"x": warp_id // 2, "y": 0, "z": 0},
        "status": "LOOPING",
        "loop": {"period": 2, "repeat": 40},
        "loop_body": {"instruction_count": 2, "instructions": body},
    }


def stuck(seconds: int) -> str:
    """A snapshot of one warp that has waited ``seconds`` at a barrier."""
    warp = barrier_warp(2, 0x490)
    warp["inactive_secs"] = seconds
    return snapshot([warp])


def header(launch_id: int = 8, grid_x: int = 4) -> dict[str, object]:
    return {
        "type": "kernel_metadata",
        "capture_config": {"grid_launch_id": launch_id},
        "grid": [grid_x, 1, 1],
        "block": [64, 1, 1],
    }


class WarpStatusPathTest(unittest.TestCase):
    def test_file_sits_next_to_every_trace_format(self) -> None:
        for name in ("k.ndjson", "k.ndjson.zst", "k.clp", "k.log"):
            with self.subTest(name=name):
                self.assertEqual(
                    warp_status_summary_path(Path("/d") / name),
                    Path("/d/k_warp_status_summary.ndjson"),
                )


class ParseIdRangesTest(unittest.TestCase):
    def test_reads_ranges_and_single_ids(self) -> None:
        self.assertEqual(parse_id_ranges(["0-2", "5"]), ((0, 2), (5, 5)))

    def test_rejects_malformed_ranges(self) -> None:
        for ranges in (["4-1"], ["x"], "0-2", [3]):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                parse_id_ranges(ranges)


class LoadRuntimeWarpStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace = Path(tmp.name) / "kernel_x_iter0_k.ndjson"

    def _write(self, *lines: str) -> Path:
        path = warp_status_summary_path(self.trace)
        path.write_text("".join(line + "\n" for line in lines))
        return path

    def test_reads_the_last_snapshot(self) -> None:
        self._write(
            snapshot([barrier_warp(2, 0x490)]),
            snapshot([barrier_warp(2, 0x490), barrier_warp(3, 0x490), looping_warp(4)]),
        )
        status = find_runtime_warp_status(self.trace)
        assert status is not None
        self.assertEqual(status.snapshot_count, 2)
        self.assertEqual(status.grid, (4, 1, 1))
        self.assertEqual(status.never_executed_warps, 3)
        self.assertEqual(status.never_executed_ranges, ((5, 7),))
        self.assertEqual(status.status_counts(), {"barrier": 2, "looping": 1})
        self.assertEqual(
            status.barrier_sites(), [(0x490, "BAR.SYNC.DEFER_BLOCKING 0x0 ;", 2)]
        )
        self.assertEqual(status.warps[0].cta, (1, 0, 0))

    def test_one_loop_recorded_from_different_phases_is_one_site(self) -> None:
        self._write(snapshot([looping_warp(2), looping_warp(3, first_pc=0x420)]))
        status = find_runtime_warp_status(self.trace)
        assert status is not None
        self.assertEqual(
            status.loop_sites(),
            [((0x3F0, 0x420), ("ATOMG.E.CAS ;", "@P0 BRA 0x3f0 ;"), 2)],
        )

    def test_only_snapshots_of_the_launch_count(self) -> None:
        self._write(
            snapshot([looping_warp(4)], launch_id=8),
            snapshot([barrier_warp(2, 0x490)], launch_id=9),
        )
        status = load_runtime_warp_status(warp_status_summary_path(self.trace), 8)
        self.assertEqual(status.snapshot_count, 1)
        self.assertEqual(status.status_counts(), {"looping": 1})

    def test_counts_only_the_snapshots_of_the_last_run(self) -> None:
        def spinning(repeat: int) -> str:
            warp = looping_warp(4)
            warp["loop"] = {"period": 2, "repeat": repeat}
            warp["inactive_secs"] = 0
            return snapshot([warp])

        # The file is appended to across runs; a run's checks grow the
        # inactive time or the loop count.
        cases = [
            ([stuck(15), stuck(16), stuck(17)], 3),
            ([stuck(15), stuck(16), stuck(15)], 1),
            ([spinning(40), spinning(80)], 2),
            ([spinning(80), spinning(40)], 1),
        ]
        for lines, count in cases:
            with self.subTest(count=count, lines=len(lines)):
                self._write(*lines)
                status = find_runtime_warp_status(self.trace)
                assert status is not None
                self.assertEqual(status.snapshot_count, count)

    def test_reads_what_wrote_the_snapshot(self) -> None:
        line = json.loads(snapshot([barrier_warp(2, 0x490)]))
        line["trigger"] = {
            "source": "kernel_timeout",
            "terminating": True,
            "stalled_checks": 1,
            "stalled_ms": 900,
        }
        self._write(snapshot([barrier_warp(2, 0x490)]), json.dumps(line))
        status = find_runtime_warp_status(self.trace)
        assert status is not None
        self.assertEqual(
            status.trigger, SnapshotTrigger("kernel_timeout", True, 1, 900)
        )
        self._write(snapshot([barrier_warp(2, 0x490)]))
        status = find_runtime_warp_status(self.trace)
        assert status is not None
        self.assertIsNone(status.trigger)

    def test_a_snapshot_that_ended_its_run_starts_no_later_run(self) -> None:
        def run(seconds: int, terminating: bool) -> str:
            warp = barrier_warp(2, 0x490)
            warp["inactive_secs"] = seconds
            line = json.loads(snapshot([warp]))
            line["trigger"] = {
                "source": "hang_check",
                "terminating": terminating,
                "stalled_checks": 3 if terminating else 1,
                "stalled_ms": seconds * 1000,
            }
            return json.dumps(line)

        # Two runs whose stall times happen to keep growing across the boundary.
        self._write(run(5, False), run(6, True), run(7, False), run(8, True))
        status = find_runtime_warp_status(self.trace)
        assert status is not None
        self.assertEqual(status.snapshot_count, 2)

    def test_malformed_trigger_raises(self) -> None:
        line = json.loads(snapshot([barrier_warp(2, 0x490)]))
        line["trigger"] = {"source": "hang_check", "terminating": "yes"}
        self._write(json.dumps(line))
        with self.assertRaisesRegex(ValueError, "trigger"):
            find_runtime_warp_status(self.trace)

    def test_ignores_a_torn_last_line(self) -> None:
        self._write(snapshot([looping_warp(4)]), '{"active_warps": [{"cta"')
        status = find_runtime_warp_status(self.trace)
        assert status is not None
        self.assertEqual(status.status_counts(), {"looping": 1})

    def test_missing_file_is_none(self) -> None:
        self.assertIsNone(find_runtime_warp_status(self.trace))

    def test_file_without_a_snapshot_raises(self) -> None:
        path = self._write("not json")
        with self.assertRaises(ValueError):
            load_runtime_warp_status(path)

    def test_malformed_snapshots_raise_value_error(self) -> None:
        path = self._write(
            snapshot([{"status": "BARRIER"}]),
            json.dumps({"active_warps": [], "warp_statistics": []}),
            json.dumps({"active_warps": {}}),
            json.dumps({"active_warps": 5}),
        )
        with self.assertRaisesRegex(ValueError, "no well-formed warp status snapshot"):
            load_runtime_warp_status(path)

    def test_a_malformed_last_snapshot_falls_back_to_the_one_before(self) -> None:
        malformed = json.loads(stuck(17))
        malformed["trigger"] = {"source": "hang_check", "terminating": "yes"}
        self._write(stuck(15), stuck(16), json.dumps(malformed))
        status = find_runtime_warp_status(self.trace)
        assert status is not None
        self.assertEqual(status.snapshot_count, 2)
        self.assertEqual(status.status_counts(), {"barrier": 1})

    def test_no_fallback_to_a_snapshot_of_an_earlier_run(self) -> None:
        # The snapshot before a malformed one ended its run, or is not behind it.
        ended = json.loads(stuck(17))
        ended["trigger"] = {
            "source": "hang_check",
            "terminating": True,
            "stalled_checks": 3,
            "stalled_ms": 3000,
        }
        for before in (json.dumps(ended), stuck(18)):
            with self.subTest(before=before[-40:]):
                self._write(stuck(15), stuck(16), before, '{"active_warps": {}}')
                with self.assertRaisesRegex(ValueError, "no well-formed"):
                    find_runtime_warp_status(self.trace)


class RuntimeStatusForTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace = Path(tmp.name) / "kernel_x_iter0_k.ndjson"

    def _write(self, line: str) -> None:
        warp_status_summary_path(self.trace).write_text(line + "\n")

    def test_snapshot_of_the_traced_launch(self) -> None:
        self._write(snapshot([looping_warp(4)]))
        status = runtime_status_for_trace(self.trace, header())
        assert status is not None
        self.assertEqual(status.kernel_launch_id, 8)

    def test_snapshot_of_another_launch_is_rejected(self) -> None:
        self._write(snapshot([looping_warp(4)], launch_id=3))
        with self.assertRaisesRegex(ValueError, "launch 8"):
            runtime_status_for_trace(self.trace, header())

    def test_snapshot_of_another_grid_is_rejected(self) -> None:
        self._write(snapshot([looping_warp(4)], grid_x=2))
        with self.assertRaisesRegex(ValueError, "grid"):
            runtime_status_for_trace(self.trace, header())

    def test_snapshot_of_another_run_is_rejected(self) -> None:
        # Warps 5-7 never executed in the snapshot; the trace shows warp 6.
        self._write(snapshot([looping_warp(4)]))
        with self.assertRaisesRegex(ValueError, "another run"):
            runtime_status_for_trace(self.trace, header(), {0, 6})
        self.assertIsNotNone(runtime_status_for_trace(self.trace, header(), {0, 4}))
