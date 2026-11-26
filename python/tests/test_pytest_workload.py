# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cutracer.pytest_workload import inspect_trace, project_python
from parameterized import parameterized


class SelectionTest(unittest.TestCase):
    def test_nested_python_keeps_runtime_dependencies_after_env_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime_dependency.py").write_text("VALUE = 42\n")
            with (
                patch.dict(os.environ, {"FB_PAR_RUNTIME_FILES": ""}),
                patch.object(sys, "path", [str(root), *sys.path]),
            ):
                python = project_python(root)
            child = (
                "import os, subprocess\n"
                "os.environ.pop('PYTHONPATH', None)\n"
                f"result = subprocess.run([{python!r}, '-c', "
                "'import runtime_dependency; assert runtime_dependency.VALUE == 42'])\n"
                "raise SystemExit(result.returncode)\n"
            )
            result = subprocess.run(
                [python, "-c", child],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_selection_does_not_run_cases_or_depend_on_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "ran"
            (root / "test_other.py").write_text(
                "import pathlib, pytest\n"
                "@pytest.mark.parametrize('value', [1, 2, 3])\n"
                "def test_other(value):\n"
                f"    pathlib.Path({str(marker)!r}).write_text(str(value))\n"
                "    assert value > 0\n"
            )
            selections = []
            for action in ("stress", "trace"):
                output = root / action
                result = subprocess.run(
                    [
                        project_python(),
                        "-m",
                        "cutracer.pytest_workload",
                        "--rootdir",
                        str(root),
                        "--tests",
                        "test_other.py",
                        "--output-dir",
                        str(output),
                        "--node-filter",
                        "[2]",
                        "--action",
                        action,
                        "--select-only",
                    ],
                    cwd=root,
                    env=dict(os.environ),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                inventory = json.loads((output / "inventory.json").read_text())
                self.assertEqual(
                    [case["nodeid"] for case in inventory["cases"]],
                    [f"test_other.py::test_other[{i}]" for i in (1, 2, 3)],
                )
                selections.append(inventory["selected_case_ids"])
            self.assertEqual(selections[0], selections[1])
            self.assertEqual(len(selections[0]), 1)
            self.assertFalse(marker.exists())


def _blocking_collection(root: Path, address: str) -> None:
    leaf = (
        "import os, signal, socket, time\n"
        "connection = socket.socket(socket.AF_UNIX)\n"
        f"connection.connect({address!r})\n"
        "def ignore_term(signum, frame):\n"
        "    connection.sendall(b'T')\n"
        "signal.signal(signal.SIGTERM, ignore_term)\n"
        "connection.sendall(f'{os.getpid()},{os.getppid()}'.encode())\n"
        "time.sleep(60)\n"
    )
    (root / "test_blocked.py").write_text(
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"subprocess.Popen([sys.executable, '-c', {leaf!r}], start_new_session=True)\n"
        "time.sleep(60)\n"
        "def test_one():\n"
        "    assert True\n"
    )


class TerminationTest(unittest.TestCase):
    def test_outer_termination_stops_owned_nested_collector_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            address = str(root / "ready.sock")
            _blocking_collection(root, address)
            with socket.socket(socket.AF_UNIX) as listener:
                listener.bind(address)
                listener.listen(1)
                listener.settimeout(20)
                process = subprocess.Popen(
                    [
                        project_python(),
                        "-m",
                        "cutracer.pytest_workload",
                        "--rootdir",
                        str(root),
                        "--tests",
                        "test_blocked.py",
                        "--output-dir",
                        str(root / "output"),
                        "--select-only",
                    ],
                    cwd=root,
                    env=dict(os.environ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                owned = []
                try:
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(3)
                        owned = [
                            int(pid) for pid in connection.recv(128).decode().split(",")
                        ]
                        started = time.monotonic()
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            acknowledged = connection.recv(1) == b"T"
                        except TimeoutError:
                            acknowledged = False
                        if acknowledged:
                            os.killpg(process.pid, signal.SIGTERM)
                        stdout, stderr = process.communicate(timeout=4)
                        self.assertEqual(process.returncode, 143, stdout + stderr)
                        self.assertTrue(
                            acknowledged, "TERM did not reach the nested leaf"
                        )
                        self.assertEqual(
                            connection.recv(1), b"", "nested leaf survived cleanup"
                        )
                        self.assertLess(time.monotonic() - started, 5)
                        self.assertFalse((root / "output" / "inventory.json").exists())
                finally:
                    for pid in owned:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=3)


def _capture(path: Path, name: str, *, records: bool, complete: bool = True) -> None:
    metadata = {
        "type": "kernel_metadata",
        "mangled_name": name,
        "unmangled_name": name,
        "kernel_checksum": "a1",
        "func_addr": "0x1",
        "nregs": 8,
        "shmem_static": 0,
        "grid": [1, 1, 1],
        "block": [32, 1, 1],
        "shmem_dynamic": 0,
        "grid_launch_id": 7,
        "instrument_modes": ["reg_trace"],
    }
    instruction = {
        "type": "reg_trace",
        "ctx": "0x1",
        "trace_index": 0,
        "timestamp": 1,
        "grid_launch_id": 7,
        "cta": [0, 0, 0],
        "warp": 0,
        "opcode_id": 0,
        "pc": "0x0",
        "regs": [[0] * 32],
    }
    completion = {
        "type": "capture_completion",
        "version": 1,
        "grid_launch_id": 7,
        "status": "complete" if complete else "incomplete",
        "kernel_completed": complete,
        "channel_drained": complete,
        "trace_records_written": 1 if records else 0,
        "dropped_records": 0,
        "errors": [] if complete else ["kernel_completion_unverified"],
    }
    rows = [metadata, instruction, completion] if records else [metadata, completion]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class TraceEvidenceTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("target_empty", False, True, False),
            ("target_partial", True, False, False),
            ("target_full", True, True, True),
            ("reference", True, True, None),
        ]
    )
    def test_only_completed_target_instruction_capture_counts(
        self, name: str, records: bool, complete: bool, expected: bool | None
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / f"{name}.ndjson"
            _capture(path, name, records=records, complete=complete)
            result = inspect_trace(path, "target")
            if expected is None:
                self.assertIsNone(result)
                return
            if result is None:
                self.fail("target capture was not recognized")
            self.assertTrue(result["validation"]["valid"])
            self.assertEqual(result["complete"], expected)
