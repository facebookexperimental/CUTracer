# Copyright (c) Meta Platforms, Inc. and affiliates.

"""E2E: CUTracer captures and runtime identity across CUDA graphs and reloads.

Run locally on a GPU host with::

    buck2 test -c fbcode.re_gpu_tests=False \\
        fbcode//triton/tools/CUTracer/tests/runtime_identity:test_runtime_identity_e2e
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

RELOAD_ITERATIONS = 16
GRAPH_REPLAYS = 1


def _records(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file]


class TestRuntimeIdentityE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("CUDA_INJECTION64_PATH", None)
        cls.cutracer_so = resolve_cutracer_so()
        cls.root = Path(tempfile.mkdtemp(prefix="cutracer_runtime_identity_"))
        cls.workload = cls.root / "runtime_identity_workload.py"
        cls.workload.write_bytes(
            pkg_resources.resource_string(
                "runtime_identity_workload", "runtime_identity_workload.py"
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _run(
        self, instrument: str, kernel_filter: str, *arguments: str
    ) -> tuple[Path, str]:
        output = self.root / self._testMethodName
        output.mkdir()
        env = _build_cutracer_env(
            cutracer_so=self.cutracer_so,
            instrument=instrument,
            analysis=None,
            kernel_filters=kernel_filter,
            instr_categories=None,
            trace_format="2",
            output_dir=str(output),
            verbose=None,
            zstd_level=None,
            delay_ns=None,
            dump_cubin=True,
        )
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        env["CUTRACER_RUNTIME_IDENTITY"] = "1"
        result = subprocess.run(
            [sys.executable, str(self.workload), *arguments],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        log = f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        self.assertEqual(result.returncode, 0, log)
        return output, result.stdout + result.stderr

    def _launches(self, output: Path) -> tuple[list[dict], list[dict]]:
        (manifest,) = output.glob("cutracer_runtime_*.ndjson")
        records = _records(manifest)
        return [r["data"] for r in records if r["type"] == "launch"], records

    def test_graph_replays_of_runtime_kernels_are_traced(self) -> None:
        output, log = self._run("opcode_only", "elementwise_kernel", "graph")
        self.assertNotIn("no capture writer", log)
        launches, _ = self._launches(output)
        graph_launches = [x for x in launches if x["launch_kind"] == "graph"]
        self.assertEqual(len(graph_launches), GRAPH_REPLAYS)
        self.assertNotIn(None, [x["graph_invocation"] for x in graph_launches])
        self.assertIn("direct", {x["launch_kind"] for x in launches})
        for launch in graph_launches:
            trace = _records(output / f"capture_{launch['launch']}.ndjson")
            self.assertEqual(trace[0]["runtime_identity"], launch)
            self.assertGreater(trace[-1]["trace_records_written"], 0, trace[-1])

    def test_reloaded_modules_never_yield_silent_or_misattributed_traces(
        self,
    ) -> None:
        output, _ = self._run(
            "reg_trace", "reload_probe", "reload", str(RELOAD_ITERATIONS)
        )
        launches, records = self._launches(output)
        self.assertGreater(len(launches), 0)
        for launch in launches:
            trace = _records(output / f"capture_{launch['launch']}.ndjson")
            self.assertEqual(trace[0]["runtime_identity"], launch)
            self.assertEqual(trace[-1]["status"], "complete", trace[-1])
            self.assertGreater(trace[-1]["trace_records_written"], 0, trace[-1])
        self.assertEqual(
            len(list(output.glob("capture_*.ndjson"))), len(launches), records
        )
        # The workload alternates two binaries, and CUTracer numbers launches
        # in order, so each parity must map to exactly one distinct CUBIN.
        binaries = [
            {x["cubin_sha256"] for x in launches if x["launch_id"] % 2 == parity}
            for parity in (0, 1)
        ]
        self.assertTrue(all(len(binary) <= 1 for binary in binaries), binaries)
        self.assertFalse(binaries[0] & binaries[1], binaries)
        if len(launches) < RELOAD_ITERATIONS:
            self.assertIn("gap", {r["type"] for r in records})
