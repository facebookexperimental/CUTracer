# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for cutracer.watchdog."""

import os
import tempfile
import unittest

from cutracer.watchdog import (
    exit_quarantine_changes,
    exit_quarantine_failures,
    main_logs,
    watchdog_terminations,
)

KERNEL_TIMEOUT = (
    "Kernel timeout: running for 61 seconds (limit: 60 seconds); sending SIGTERM. "
    "kernel=my_kernel"
)
NO_DATA = (
    "No-data hang detected: kernel launched 15 seconds ago but never produced "
    "any trace data (limit: 15 seconds); sending SIGTERM."
)
DEADLOCK = (
    "Deadlock sustained for 3 checks at least 1000 ms apart over 2000 ms (minimum "
    "stall 0 ms): launch_id=7, no active warp progressing (looping=12, barrier=4); "
    "sending SIGTERM."
)
LATE_WRITE = (
    "Exit quarantine: shared memory of CTA (1,0,0) changed after its last thread "
    "exited: offset 0x480 poison 0xc5c5c5c500000480 observed 0x0000000100000001 "
    "(launch_id 3, kernel=my_kernel, checksum=abc)"
)
QUARANTINE_SUMMARY = (
    "Exit quarantine: 2 CTA(s) quarantined, 1 changed word(s) (launch_id 3)"
)
POSSIBLE_HANG = (
    "Possible kernel hang: launch_id=7 — state(looping=12, barrier=4, "
    "progressing=0) for 1 seconds."
)


def _write(directory: str, name: str, *lines: str) -> None:
    with open(os.path.join(directory, name), "w") as log:
        log.write("".join(line + "\n" for line in lines))


class WatchdogTest(unittest.TestCase):
    def test_main_logs_lists_only_cutracer_main_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "cutracer_main_20261104_000000_1.log", "started")
            _write(directory, "cutracer_kernel_events_20261104_000000_1.ndjson")
            _write(directory, "run.log")
            self.assertEqual(
                main_logs(directory), {"cutracer_main_20261104_000000_1.log"}
            )

    def test_missing_directory_has_no_logs(self) -> None:
        self.assertEqual(main_logs("/nonexistent/cutracer"), set())

    def test_reads_watchdog_lines_of_new_logs_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "cutracer_main_old.log", KERNEL_TIMEOUT)
            before = main_logs(directory)
            _write(directory, "cutracer_main_rank0.log", "started", KERNEL_TIMEOUT)
            _write(directory, "cutracer_main_rank1.log", NO_DATA)
            _write(directory, "cutracer_main_rank2.log", "started", "finished")

            lines = watchdog_terminations(directory, before=before)

        self.assertEqual(lines, [KERNEL_TIMEOUT, NO_DATA])

    def test_clean_run_has_no_watchdog_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "cutracer_main_rank0.log", "started", "finished")
            self.assertEqual(watchdog_terminations(directory, before=set()), [])

    def test_reads_exit_quarantine_changes_of_new_logs_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "cutracer_main_old.log", LATE_WRITE)
            before = main_logs(directory)
            _write(directory, "cutracer_main_rank0.log", LATE_WRITE, QUARANTINE_SUMMARY)
            _write(directory, "cutracer_main_rank1.log", "started", KERNEL_TIMEOUT)

            self.assertEqual(
                exit_quarantine_changes(directory, before=before), [LATE_WRITE]
            )
            self.assertEqual(
                watchdog_terminations(directory, before=before), [KERNEL_TIMEOUT]
            )

    def test_fatal_reader_uses_fresh_logs_and_process_output(self) -> None:
        fatal = "FATAL: exit quarantine: unsupported graph"
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "cutracer_main_old.log", fatal)
            before = main_logs(directory)
            self.assertEqual(exit_quarantine_failures(directory, before=before), [])
            _write(directory, "cutracer_main_new.log", fatal)
            self.assertEqual(
                exit_quarantine_failures(
                    directory, before=before, output=(fatal.encode(), None)
                ),
                [fatal],
            )
        self.assertEqual(
            exit_quarantine_failures(
                "/nonexistent/cutracer", before=set(), output=(fatal,)
            ),
            [fatal],
        )

    def test_unreadable_log_does_not_drop_other_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.mkdir(os.path.join(directory, "cutracer_main_unreadable.log"))
            _write(directory, "cutracer_main_valid.log", LATE_WRITE, KERNEL_TIMEOUT)
            self.assertEqual(
                exit_quarantine_changes(directory, before=set()), [LATE_WRITE]
            )
            self.assertEqual(
                watchdog_terminations(directory, before=set()), [KERNEL_TIMEOUT]
            )

    def test_deadlock_check_termination_counts_but_a_warning_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "cutracer_main_rank0.log", POSSIBLE_HANG, DEADLOCK)
            self.assertEqual(watchdog_terminations(directory, before=set()), [DEADLOCK])
