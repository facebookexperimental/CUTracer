# Copyright (c) Meta Platforms, Inc. and affiliates.

"""E2E: non-blocking launches let kernels overlap under CUTracer.

`overlap_workload.py` prints OVERLAPPED only if a kernel launched after a
running kernel starts before that kernel finishes. Natively it does. Under
CUTracer's default launch handling it cannot, because the launching thread
waits for every launch to finish; with --nonblocking-launches it can, both as
a bare launch logger and with the second kernel instrumented.
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


class TestNonblockingLaunchE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("CUDA_INJECTION64_PATH", None)
        cls.cutracer_so = resolve_cutracer_so()
        script = pkg_resources.resource_string(
            "launch_test_resources", "overlap_workload.py"
        ).decode()
        cls.work_dir = tempfile.mkdtemp(prefix="cutracer_launch_")
        cls.script_path = os.path.join(cls.work_dir, "overlap_workload.py")
        Path(cls.script_path).write_text(script)
        cls.queued_script_path = os.path.join(cls.work_dir, "queued_workload.py")
        Path(cls.queued_script_path).write_text(
            pkg_resources.resource_string(
                "launch_test_resources", "queued_workload.py"
            ).decode()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def _run(self, env: dict[str, str], script: str | None = None) -> str:
        env = {**env, "OVERLAP_MAX_POLLS": env.get("OVERLAP_MAX_POLLS", "1000000")}
        result = subprocess.run(
            [sys.executable, script or self.script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output[-4000:])
        return output

    def _cutracer_env(
        self,
        output_dir: str,
        *,
        instrument: str | None = None,
        kernel_filters: str | None = None,
        nonblocking_launches: bool = False,
    ) -> dict[str, str]:
        return _build_cutracer_env(
            cutracer_so=self.cutracer_so,
            instrument=instrument,
            analysis=None,
            kernel_filters=kernel_filters,
            instr_categories=None,
            trace_format="2",
            output_dir=output_dir,
            verbose=None,
            zstd_level=None,
            delay_ns=None,
            nonblocking_launches=nonblocking_launches,
        )

    def test_kernels_overlap_natively(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("CUTRACER_")}
        self.assertIn("OVERLAPPED", self._run(env))

    def test_default_launch_handling_serializes_them(self) -> None:
        output = self._run(self._cutracer_env(tempfile.mkdtemp(dir=self.work_dir)))
        self.assertIn("GAVE_UP", output)

    def test_nonblocking_launch_logger_lets_them_overlap(self) -> None:
        output = self._run(
            self._cutracer_env(
                tempfile.mkdtemp(dir=self.work_dir), nonblocking_launches=True
            )
        )
        self.assertIn("OVERLAPPED", output)

    def test_nonblocking_instrumented_kernel_overlaps_and_completes_its_trace(
        self,
    ) -> None:
        trace_dir = tempfile.mkdtemp(dir=self.work_dir)
        output = self._run(
            self._cutracer_env(
                trace_dir,
                instrument="opcode_only",
                kernel_filters="set_flag",
                nonblocking_launches=True,
            )
        )
        self.assertIn("OVERLAPPED", output)
        traces = sorted(Path(trace_dir).glob("*set_flag*.ndjson"))
        # One trace for the warm-up launch, one for the measured launch.
        self.assertEqual(len(traces), 2, output[-4000:])
        for trace in traces:
            footer = json.loads(trace.read_text().splitlines()[-1])
            with self.subTest(trace=trace.name):
                self.assertEqual(footer["type"], "capture_completion")
                self.assertEqual(footer["status"], "complete")

    def test_two_overlapping_instrumented_kernels_both_complete_their_traces(
        self,
    ) -> None:
        trace_dir = tempfile.mkdtemp(dir=self.work_dir)
        env = self._cutracer_env(
            trace_dir,
            instrument="opcode_only",
            kernel_filters="wait_for_flag,set_flag",
            nonblocking_launches=True,
        )
        env["OVERLAP_MAX_POLLS"] = "20000"
        output = self._run(env)
        self.assertIn("OVERLAPPED", output)
        for kernel in ("wait_for_flag", "set_flag"):
            traces = sorted(Path(trace_dir).glob(f"*{kernel}*.ndjson"))
            self.assertEqual(len(traces), 2, output[-4000:])
            for trace in traces:
                lines = trace.read_text().splitlines()
                instructions = json.loads(lines[0])["instructions"]
                footer = json.loads(lines[-1])
                records = [json.loads(line) for line in lines[1:-1]]
                last_by_warp = {}
                for record in records:
                    last_by_warp[(tuple(record["cta"]), record["warp"])] = record
                with self.subTest(trace=trace.name):
                    self.assertEqual(footer["status"], "complete")
                    self.assertEqual(footer["trace_records_written"], len(records))
                    # The drain waited for the whole kernel: every warp's
                    # last record is its EXIT.
                    self.assertTrue(last_by_warp)
                    for record in last_by_warp.values():
                        sass = instructions[str(record["opcode_id"])]["sass"]
                        self.assertIn("EXIT", sass)

    def test_a_kernel_queued_behind_a_long_one_is_not_timed_out(self) -> None:
        # The instrumented kernel waits in its stream longer than the default
        # 15 s no-data timeout, which must not count that wait.
        trace_dir = tempfile.mkdtemp(dir=self.work_dir)
        env = self._cutracer_env(
            trace_dir,
            instrument="opcode_only",
            kernel_filters="add_one",
            nonblocking_launches=True,
        )
        env["QUEUED_BEHIND_S"] = "20"
        output = self._run(env, self.queued_script_path)
        self.assertIn("DONE", output)
        self.assertNotIn("No-data hang detected", output)
        self.assertEqual(len(list(Path(trace_dir).glob("*add_one*.ndjson"))), 2)
