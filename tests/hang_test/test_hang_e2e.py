# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
E2E test: CUTracer hang detection + warp status output.

Runs the hanging Triton kernel from test_hang.py under CUTracer's
deadlock_detection analysis in two trace formats:

  - Mode 0 (text):   asserts that ``LOOPING(inactive=...)`` appears in the
                     captured tracer output, guarding the diff that aligned
                     ``inactive_secs`` rendering across non-BARRIER warps.

  - Mode 2 (NDJSON): asserts that the new
                     ``<basename>_warp_status_summary.ndjson`` file is created
                     in append mode (one JSON object per snapshot), and that
                     the legacy ``_warp_status_summary.json`` is no longer
                     produced.

Both modes share the expectation that CUTracer terminates the hung subprocess
with "Deadlock sustained" in its output (non-zero exit code).

This is the Buck equivalent of ``test_hang_test`` /
``test_hang_test_warp_status_ndjson`` from ``.ci/run_tests.sh``.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pkg_resources
from cutracer.runner import _build_cutracer_env, resolve_cutracer_so


class TestHangE2E(unittest.TestCase):
    """E2E: CUTracer hang detection and warp status snapshot output."""

    @classmethod
    def setUpClass(cls):
        # CUDA_INJECTION64_PATH must be unset before resolve_cutracer_so()
        # picks the bundled .so; an externally-set value triggers a hard error.
        os.environ.pop("CUDA_INJECTION64_PATH", None)

        cls.cutracer_so = resolve_cutracer_so()

        # Extract the hanging kernel script from bundled resources
        script_content = pkg_resources.resource_string(
            "hang_test_resources",
            "test_hang.py",
        ).decode()
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, prefix="hang_"
        )
        tmp.write(script_content)
        tmp.flush()
        tmp.close()
        cls.script_path = tmp.name

        # Temp dir for CUTracer trace + warp-status output
        cls.trace_dir = tempfile.mkdtemp(prefix="cutracer_hang_")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "script_path") and os.path.exists(cls.script_path):
            os.unlink(cls.script_path)
        if hasattr(cls, "trace_dir") and os.path.exists(cls.trace_dir):
            shutil.rmtree(cls.trace_dir, ignore_errors=True)

    def _clean_trace_dir(self):
        for f in Path(self.trace_dir).iterdir():
            if f.is_file():
                f.unlink()

    def _run_hang(self, trace_format: int) -> subprocess.CompletedProcess:
        """Run test_hang.py under CUTracer with deadlock_detection.

        The kernel hangs intentionally; CUTracer is expected to detect the
        deadlock and terminate the subprocess (non-zero exit code).
        """
        self._clean_trace_dir()

        env = _build_cutracer_env(
            cutracer_so=self.cutracer_so,
            instrument=None,  # deadlock_detection auto-adds reg_trace
            analysis="deadlock_detection",
            kernel_filters="add_kernel",
            instr_categories=None,
            trace_format=str(trace_format),
            output_dir=self.trace_dir,
            verbose=None,
            zstd_level=None,
            delay_ns=None,
        )

        # Hard wall-clock timeout: hang detection should kick in well before
        # this fires; if it doesn't, the test should fail loudly rather than
        # hang the test runner.
        return subprocess.run(
            [sys.executable, self.script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    @staticmethod
    def _captured_output(result: subprocess.CompletedProcess) -> str:
        return (result.stdout or "") + (result.stderr or "")

    def _assert_deadlock_detected(self, result: subprocess.CompletedProcess):
        """Common precondition: CUTracer terminated the process on hang."""
        output = self._captured_output(result)
        self.assertNotEqual(
            result.returncode,
            0,
            f"Expected hang detection to terminate subprocess, but it exited 0.\n"
            f"OUTPUT:\n{output[-4000:]}",
        )
        self.assertIn(
            "Deadlock sustained",
            output,
            f"Hang detection message not found in subprocess output.\n"
            f"OUTPUT:\n{output[-4000:]}",
        )

    # ------------------------------------------------------------------
    # Test: text mode prints inactive=Xs for LOOPING entries
    # ------------------------------------------------------------------

    def test_text_mode_inactive_secs(self):
        """Mode 0: LOOPING warps must include ``inactive=Xs`` in text output.

        Regression guard for the diff that made ``inactive_secs`` render for
        all warp states (not just BARRIER) so text and JSON outputs agree.
        """
        result = self._run_hang(trace_format=0)
        self._assert_deadlock_detected(result)

        output = self._captured_output(result)
        self.assertIn(
            "LOOPING(inactive=",
            output,
            "Expected 'LOOPING(inactive=...' line in text-mode warp status "
            "summary; the hanging kernel should produce LOOPING warps and "
            "the renderer must include inactive_secs for them.",
        )

    # ------------------------------------------------------------------
    # Test: NDJSON warp status appends one JSON object per snapshot
    # ------------------------------------------------------------------

    def test_ndjson_mode_warp_status(self):
        """Mode 2: warp status is appended as NDJSON across snapshots.

        Validates the diff that switched ``write_warp_status_json`` from a
        truncating ``.json`` to an append-mode ``.ndjson`` so multiple
        hang-detection ticks accumulate as separate lines instead of
        overwriting each other.
        """
        result = self._run_hang(trace_format=2)
        self._assert_deadlock_detected(result)

        # Legacy single-file output must no longer be produced
        legacy_files = sorted(Path(self.trace_dir).glob("*_warp_status_summary.json"))
        self.assertEqual(
            legacy_files,
            [],
            f"Legacy *_warp_status_summary.json should no longer be written, "
            f"found: {legacy_files}",
        )

        ndjson_files = sorted(Path(self.trace_dir).glob("*_warp_status_summary.ndjson"))
        self.assertGreater(
            len(ndjson_files),
            0,
            f"No *_warp_status_summary.ndjson produced in {self.trace_dir}.\n"
            f"Directory contents: {sorted(p.name for p in Path(self.trace_dir).iterdir())}",
        )

        ndjson_path = ndjson_files[0]
        lines = [
            ln.strip() for ln in ndjson_path.read_text().splitlines() if ln.strip()
        ]
        self.assertGreater(
            len(lines),
            0,
            f"NDJSON warp status file is empty: {ndjson_path}",
        )

        records = []
        for i, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                self.fail(
                    f"{ndjson_path.name} line {i + 1} is not valid JSON: {e}\n"
                    f"line content: {line[:200]}"
                )

        # All snapshots in one file must come from the same kernel launch
        kernel_ids = {r.get("kernel_launch_id") for r in records}
        self.assertEqual(
            len(kernel_ids),
            1,
            f"Expected all snapshots to share kernel_launch_id, got: {kernel_ids}",
        )

        # Every warp entry must carry inactive_secs (consistency with text)
        # and at least one entry across all snapshots must be LOOPING.
        saw_looping = False
        for record in records:
            for warp in record.get("active_warps", []):
                self.assertIn(
                    "inactive_secs",
                    warp,
                    f"warp entry missing inactive_secs: {warp}",
                )
                if warp.get("status") == "LOOPING":
                    saw_looping = True
        self.assertTrue(
            saw_looping,
            "Expected at least one LOOPING warp entry across snapshots; "
            "the hanging kernel should be detected as LOOPING.",
        )
