# Copyright (c) Meta Platforms, Inc. and affiliates.

"""E2E: CUTracer records a launch's attributes, including PDL.

`pdl_workload.py` launches one Triton kernel with programmatic dependent
launch and once without. The kernel events and the instrumented launch's
trace metadata must say which launch was PDL.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema
import pkg_resources
from cutracer.runner import _build_cutracer_env, resolve_cutracer_so
from cutracer.validation.schema_loader import SCHEMAS_BY_TYPE


def _ndjson(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestLaunchAttributesE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("CUDA_INJECTION64_PATH", None)
        cls.cutracer_so = resolve_cutracer_so()
        cls.work_dir = tempfile.mkdtemp(prefix="cutracer_launch_attrs_")
        cls.script_path = os.path.join(cls.work_dir, "pdl_workload.py")
        Path(cls.script_path).write_text(
            pkg_resources.resource_string(
                "launch_test_resources", "pdl_workload.py"
            ).decode()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def _trace(
        self,
        *args: str,
        instrument: str | None = None,
        kernel_filters: str | None = None,
        kernel_events: str | None = None,
    ) -> Path:
        output_dir = Path(tempfile.mkdtemp(dir=self.work_dir))
        env = _build_cutracer_env(
            cutracer_so=self.cutracer_so,
            instrument=instrument,
            analysis=None,
            kernel_filters=kernel_filters,
            instr_categories=None,
            trace_format="2",
            output_dir=str(output_dir),
            verbose=None,
            zstd_level=None,
            delay_ns=None,
            kernel_events=kernel_events,
        )
        result = subprocess.run(
            [sys.executable, self.script_path, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output[-4000:])
        self.assertIn("DONE", output)
        return output_dir

    def test_kernel_events_mark_the_pdl_launch(self) -> None:
        output_dir = self._trace(kernel_events="nostack")
        events_files = sorted(output_dir.glob("cutracer_kernel_events_*.ndjson"))
        self.assertEqual(len(events_files), 1)
        launches = [
            event
            for event in _ndjson(events_files[0])
            if event.get("type") == "kernel_launch"
            and "scale_kernel" in event["kernel_name"]
        ]
        self.assertEqual(len(launches), 2)
        pdl = [
            event["launch_attributes"].get("programmatic_stream_serialization", False)
            for event in launches
        ]
        self.assertEqual(pdl, [True, False])
        for event in launches:
            self.assertIn("max_dynamic_shared_size_bytes", event["function_attributes"])
            self.assertFalse(event["nonblocking_launches"])

    def test_graph_node_launches_carry_the_captured_attributes(self) -> None:
        output_dir = self._trace("--graph", kernel_events="nostack")
        events_files = sorted(output_dir.glob("cutracer_kernel_events_*.ndjson"))
        self.assertEqual(len(events_files), 1)
        node_launches = [
            event
            for event in _ndjson(events_files[0])
            if event.get("type") == "kernel_launch"
            and "scale_kernel" in event["kernel_name"]
            and "captured_launch_attributes" in event
        ]
        self.assertEqual(len(node_launches), 1)
        self.assertNotIn("launch_attributes", node_launches[0])
        self.assertTrue(
            node_launches[0]["captured_launch_attributes"][
                "programmatic_stream_serialization"
            ]
        )

    def test_trace_metadata_marks_the_pdl_launch(self) -> None:
        output_dir = self._trace(
            instrument="opcode_only", kernel_filters="scale_kernel"
        )
        traces = sorted(output_dir.glob("*scale_kernel*.ndjson"))
        self.assertEqual(len(traces), 2)
        serialization = {}
        for trace in traces:
            metadata = _ndjson(trace)[0]
            self.assertEqual(metadata["type"], "kernel_metadata")
            jsonschema.validate(metadata, SCHEMAS_BY_TYPE["kernel_metadata"])
            serialization[metadata["grid_launch_id"]] = metadata[
                "launch_attributes"
            ].get("programmatic_stream_serialization", False)
            self.assertIn("function_attributes", metadata)
        self.assertEqual(
            [serialization[k] for k in sorted(serialization)], [True, False]
        )
