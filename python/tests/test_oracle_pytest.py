# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for `cutracer oracle pytest` and `cutracer oracle export`.

A fake interpreter stands in for the project's Python: it plays the pytest
worker (writes the report) and CUTracer (writes the delay dump and the kernel
launch events), so the oracle's classification, coverage and evidence handling
run without pytest, a GPU or cutracer.so. The oracle is a standalone script
shipped as a resource; the tests load it the way the CLI does.
"""

import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from click.testing import CliRunner
from cutracer.oracle.cli import oracle_command, pytest_oracle

# The worker mode imports pytest at run time; RealPytestWorkerTest runs it here.
# @dep=fbsource//third-party/pypi/pytest:pytest

NODE = "tests/test_x.py::test_y"
REPRODUCED, PASSED, INCOMPLETE = 0, 1, 2

_FAKE_WORKER = textwrap.dedent(
    """
    import json, os, sys, time
    args = sys.argv[2:]
    report = args[args.index("--worker-report") + 1]
    script = json.loads(os.environ["FAKE_SCRIPT"])
    if script.get("sleep"):
        time.sleep(script["sleep"])
    dump = os.environ.get("CUTRACER_DELAY_DUMP_PATH")
    if dump and "config" in script:
        with open(dump, "w") as fh:
            json.dump(script["config"], fh)
    out = os.environ.get("CUTRACER_OUTPUT_DIR")
    if out and script.get("launch"):
        name = "cutracer_kernel_events_1_%d.ndjson" % os.getpid()
        with open(os.path.join(out, name), "w") as fh:
            fh.write(json.dumps({"type": "kernel_launch", "kernel_name": "my_kernel",
                                 "kernel_checksum": "beef"}) + "\\n")
    with open(os.environ["FAKE_SEEN"], "w") as fh:
        json.dump({"argv": sys.argv[1:], "dump": dump,
                   "load": os.environ.get("CUTRACER_DELAY_LOAD_PATH"),
                   "events": os.environ.get("CUTRACER_KERNEL_EVENTS")}, fh)
    with open(report, "w") as fh:
        json.dump(script["report"], fh)
    sys.exit(script["report"]["pytest_exit"])
    """
)


def _report(outcome: str = "passed", **kw: Any) -> dict[str, Any]:
    phases = [
        {"nodeid": NODE, "when": "setup", "outcome": "passed", "xfail": False},
        {"nodeid": NODE, "when": "call", "outcome": outcome, "xfail": False},
        {"nodeid": NODE, "when": "teardown", "outcome": "passed", "xfail": False},
    ]
    base: dict[str, Any] = {
        "selected": [NODE],
        "phases": phases,
        "collection_errors": [],
        "pytest_exit": 0 if outcome == "passed" else 1,
    }
    if outcome == "failed":
        base.update(correctness_failure=True, signature="sig-a")
    base.update(kw)
    return base


def _config(points: bool = True) -> dict[str, Any]:
    return {
        "version": "1.0",
        "delay_ns": 1000,
        "kernels": {
            "my_kernel_beef": {
                "kernel_name": "my_kernel",
                "kernel_checksum": "beef",
                "instrumentation_points": (
                    {"16": {"pc": 16, "sass": "SYNCS", "delay": 1000, "on": True}}
                    if points
                    else {}
                ),
            }
        },
    }


class _Fixture:
    def __init__(self, root: Path, oracle: Any) -> None:
        self.root = root
        self.oracle = oracle
        worker = root / "fake_worker.py"
        worker.write_text(_FAKE_WORKER)
        self.python = root / "python"
        self.python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{worker}" "$@"\n')
        self.python.chmod(self.python.stat().st_mode | stat.S_IXUSR)
        self.seen = root / "seen.json"
        self.out = root / "events"
        self.out.mkdir()

    def env(
        self, script: dict[str, Any], *, injected: bool = True, **extra: str
    ) -> dict[str, str]:
        env = {
            "FAKE_SCRIPT": json.dumps(script),
            "FAKE_SEEN": str(self.seen),
            "CUTRACER_OUTPUT_DIR": str(self.out),
            **extra,
        }
        if injected:
            env["CUDA_INJECTION64_PATH"] = "/fake/cutracer.so"
        return env

    def run(
        self,
        script: dict[str, Any],
        *,
        injected: bool = True,
        env: dict[str, str] | None = None,
        **kw: Any,
    ) -> int:
        options: dict[str, Any] = {
            "python": str(self.python),
            "rootdir": self.root,
            "nodeid": NODE,
            "kernels": "my_kernel",
            "output_dir": self.root / "evidence",
            **kw,
        }
        with mock.patch.dict(
            os.environ, self.env(script, injected=injected, **(env or {})), clear=False
        ):
            if not injected:
                os.environ.pop("CUDA_INJECTION64_PATH", None)
            return self.oracle.run_oracle(self.oracle.OracleOptions(**options))

    def outcome(self) -> dict[str, Any]:
        (attempt,) = (self.root / "evidence").glob("attempt-*")
        return json.loads((attempt / "outcome.json").read_text())


class _OracleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.oracle = stack.enter_context(pytest_oracle())
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.fx = _Fixture(self.root, self.oracle)


class RunOracleTest(_OracleTestCase):
    def test_a_passing_covered_node_exits_1_and_copies_the_dump(self) -> None:
        requested = self.root / "requested.json"
        code = self.fx.run(
            {"report": _report(), "config": _config(), "launch": True},
            env={"CUTRACER_DELAY_DUMP_PATH": str(requested)},
        )
        self.assertEqual(code, PASSED)
        self.assertEqual(json.loads(requested.read_text()), _config())
        seen = json.loads(self.fx.seen.read_text())
        # The child dumps into the attempt's evidence, with kernel events on,
        # and runs this script in its worker mode.
        self.assertNotEqual(seen["dump"], str(requested))
        self.assertEqual(seen["events"], "nostack")
        self.assertEqual(Path(seen["argv"][0]).name, "pytest_oracle.py")
        self.assertIn("--pytest-arg=addopts=", seen["argv"])
        outcome = self.fx.outcome()
        self.assertEqual((outcome["exit"], outcome["coverage"]), (PASSED, True))

    def test_a_correctness_failure_reproduces_and_pins_its_signature(self) -> None:
        sig = self.root / "sig"
        script = {"report": _report("failed"), "config": _config(), "launch": True}
        self.assertEqual(self.fx.run(script, signature_file=sig), REPRODUCED)
        self.assertEqual(sig.read_text().strip(), "sig-a")
        other = {
            "report": _report("failed", signature="sig-b"),
            "config": _config(),
            "launch": True,
        }
        self.assertEqual(
            self.fx.run(other, signature_file=sig, output_dir=None), INCOMPLETE
        )

    def test_an_empty_signature_file_pins_the_next_reproduction(self) -> None:
        sig = self.root / "sig"
        sig.write_text("")
        script = {"report": _report("failed"), "config": _config(), "launch": True}
        self.assertEqual(self.fx.run(script, signature_file=sig), REPRODUCED)
        self.assertEqual(sig.read_text(), "sig-a\n")
        self.assertEqual([p.name for p in self.root.glob(".sig*")], [])
        other = {
            "report": _report("failed", signature="sig-b"),
            "config": _config(),
            "launch": True,
        }
        self.assertEqual(
            self.fx.run(other, signature_file=sig, output_dir=None), INCOMPLETE
        )

    def test_the_exit_code_and_its_reason_go_to_stderr(self) -> None:
        script = {"report": _report(), "config": _config(), "launch": True}
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = self.fx.run(script, kernels="other_kernel")
        self.assertEqual(code, INCOMPLETE)
        self.assertIn(
            "cutracer oracle: exit 2: no launched kernel matching 'other_kernel' "
            "has delay points",
            stderr.getvalue(),
        )
        self.assertIn("evidence: ", stderr.getvalue())
        self.assertIn("other_kernel", self.fx.outcome()["reason"])

    def test_an_injected_run_must_instrument_a_launched_kernel(self) -> None:
        cases = {
            "no points": {
                "report": _report(),
                "config": _config(points=False),
                "launch": True,
            },
            "not launched": {"report": _report(), "config": _config()},
            "no dump": {"report": _report(), "launch": True},
        }
        for name, script in cases.items():
            with self.subTest(name=name):
                self.assertEqual(self.fx.run(script, output_dir=None), INCOMPLETE)

    def test_a_kernel_filter_must_match(self) -> None:
        script = {"report": _report(), "config": _config(), "launch": True}
        self.assertEqual(self.fx.run(script, kernels="other_kernel"), INCOMPLETE)

    def test_plain_and_injected_runs_check_the_injection(self) -> None:
        script = {"report": _report()}
        self.assertEqual(self.fx.run(script, injected=False, plain=True), PASSED)
        self.assertEqual(
            self.fx.run(script, injected=True, plain=True, output_dir=None),
            INCOMPLETE,
        )
        self.assertEqual(
            self.fx.run(script, injected=False, output_dir=None), INCOMPLETE
        )

    def test_a_replay_keeps_its_candidate_and_dumps_nothing(self) -> None:
        candidate = self.root / "candidate.json"
        candidate.write_text(json.dumps(_config()))
        code = self.fx.run(
            {"report": _report("failed"), "launch": True},
            env={
                "CUTRACER_DELAY_LOAD_PATH": str(candidate),
                "CUTRACER_DELAY_DUMP_PATH": str(self.root / "x.json"),
            },
        )
        self.assertEqual(code, REPRODUCED)
        seen = json.loads(self.fx.seen.read_text())
        self.assertIsNone(seen["dump"])
        (attempt,) = (self.root / "evidence").glob("attempt-*")
        self.assertEqual(json.loads((attempt / "delay.json").read_text()), _config())

    def test_a_node_past_its_timeout_is_incomplete(self) -> None:
        script = {"report": _report(), "config": _config(), "launch": True, "sleep": 30}
        self.assertEqual(self.fx.run(script, timeout_s=0.5), INCOMPLETE)
        self.assertIn("ran past", self.fx.outcome()["errors"][0])

    def test_without_an_output_dir_nothing_is_kept(self) -> None:
        script = {"report": _report(), "config": _config(), "launch": True}
        scratch = self.root / "tmp"
        scratch.mkdir()
        with mock.patch.object(tempfile, "tempdir", str(scratch)):
            self.assertEqual(self.fx.run(script, output_dir=None), PASSED)
        self.assertEqual(list(scratch.iterdir()), [])


class ClassifyTest(_OracleTestCase):
    def test_outcomes(self) -> None:
        classify = self.oracle.classify
        self.assertEqual(classify(_report(), NODE), (PASSED, "the node passed"))
        site = ["tests/utils.py", "check", 42, "assert out == ref"]
        self.assertEqual(
            classify(_report("failed", assertion_site=site), NODE),
            (
                REPRODUCED,
                "the node failed its correctness check at tests/utils.py::check:42",
            ),
        )
        self.assertEqual(classify(_report("failed"), NODE, "sig-a")[0], REPRODUCED)

    def test_anything_else_measured_nothing(self) -> None:
        setup_error = _report()
        setup_error["phases"][0]["outcome"] = "failed"
        xfail = _report()
        xfail["phases"][1]["xfail"] = True
        not_asserted = _report(
            "failed",
            correctness_failure=False,
            exception="RuntimeError",
            assertion_site=["tests/test_x.py", "test_y", 3, "run()"],
        )
        no_traceback = _report(
            "failed", correctness_failure=False, child_failure=True, exception=""
        )
        cases = {
            "other signature": (_report("failed"), "sig-b", "not the pinned failure"),
            "not an assertion": (
                not_asserted,
                None,
                "failed with RuntimeError at tests/test_x.py::test_y:3",
            ),
            "no worker traceback": (no_traceback, None, "@record"),
            "wrong selection": (
                _report(selected=[NODE, NODE + "2"]),
                None,
                "not exactly the node",
            ),
            "collection error": (
                _report(collection_errors=["boom"]),
                None,
                "could not collect",
            ),
            "setup error": (setup_error, None, "the node's setup failed"),
            "xfail": (xfail, None, "xfail"),
            "skipped": (_report("skipped"), None, "skipped"),
            "exit mismatch": (_report(pytest_exit=2), None, "pytest exited 2"),
            "missing phases": (_report(phases=[]), None, "setup, call and teardown"),
        }
        for name, (report, expected, reason) in cases.items():
            with self.subTest(name=name):
                code, why = self.oracle.classify(report, NODE, expected)
                self.assertEqual(code, INCOMPLETE)
                self.assertIn(reason, why)

    def test_coverage_needs_a_launched_kernel_with_points(self) -> None:
        covered = self.oracle.covered
        launched = {("my_kernel", "beef")}
        self.assertTrue(covered(_config(), launched, "my_kernel"))
        self.assertTrue(covered(_config(), launched, None))
        self.assertFalse(covered(_config(), set(), "my_kernel"))
        self.assertFalse(covered(_config(points=False), launched, "my_kernel"))
        self.assertFalse(covered(_config(), launched, "other_kernel,xyz"))

    def test_the_innermost_frame_under_the_rootdir_is_the_site(self) -> None:
        site_of, signature = self.oracle.assertion_site, self.oracle.signature
        root = self.root
        entries = [
            SimpleNamespace(
                path=f"{root}/tests/test_x.py",
                name="test_y",
                lineno=9,
                statement="check(out)",
            ),
            SimpleNamespace(
                path=f"{root}/tests/utils.py",
                name="check",
                lineno=41,
                statement="assert   out == ref",
            ),
            SimpleNamespace(
                path="/env/site-packages/torch/testing.py",
                name="assert_close",
                lineno=3,
                statement="raise",
            ),
        ]
        site = site_of(entries, root)
        self.assertEqual(site, ["tests/utils.py", "check", 42, "assert out == ref"])
        self.assertIsNone(site_of(entries[2:], root))
        self.assertEqual(signature(NODE, site), signature(NODE, list(site)))
        self.assertNotEqual(signature(NODE, site), signature(NODE + "x", site))


class RealPytestWorkerTest(_OracleTestCase):
    """The worker mode under real pytest (in this process), then the oracle's
    classification of its report."""

    def _classify(
        self,
        body: str,
        sites: tuple[str, ...],
        *,
        prelude: str = "",
        child_last: str | None = None,
    ) -> tuple[int, str]:
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        # A fresh directory per case: pytest caches modules by path.
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        packages = root / "venv/lib/python3/site-packages"
        packages.mkdir(parents=True)
        (packages / "checks.py").write_text("def check():\n    assert 1 == 2\n")
        # A worker's formatted traceback: the test file's helper, then a
        # frame in an installed package, chained after an unrelated error.
        child_stack = (
            "Traceback (most recent call last):\n"
            f'  File "{root}/test_x.py", line 3, in test_x\n'
            "    pass\n"
            "KeyError: 'warmup'\n\n"
            "During handling of the above exception, another exception occurred:\n\n"
            "Traceback (most recent call last):\n"
            f'  File "{root}/test_x.py", line 4, in helper\n'
            "    assert   1 == 2\n"
            f'  File "{packages}/checks.py", line 2, in check\n'
            "    assert 1 == 2\n"
            f"{child_last}\n"
        )
        (root / "test_x.py").write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(packages)!r})\n"
            f"STACK = {child_stack!r}\n"
            f"{prelude}"
            "def helper():\n    assert 1 == 2\n"
            f"def test_x():\n    {body}\n"
        )
        report = root / "report.json"
        args = self.oracle.build_parser().parse_args(
            ["--worker-report", str(report), "--rootdir", str(root)]
            + ["--nodeid", "test_x.py::test_x"]
            + [f"--assertion-site={site}" for site in sites]
            + [f"--pytest-arg={arg}" for arg in self.oracle.DEFAULT_PYTEST_ARGS]
        )
        with mock.patch.dict(os.environ, {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}):
            code = self.oracle.run_worker(args)
        data = json.loads(report.read_text())
        self.assertEqual(code, data["pytest_exit"])
        return self.oracle.classify(data, "test_x.py::test_x")

    def test_a_failed_torchelastic_worker_is_judged_by_its_own_traceback(
        self,
    ) -> None:
        test, helper = "test_x.py::test_x", "test_x.py::helper"
        launcher = (
            "class ChildFailedError(Exception):\n"
            "    def get_first_failure(self):\n"
            "        failure = type('F', (), {})()\n"
            "        failure.message = {'message': 'x', 'extraInfo': {'py_callstack': STACK}}\n"
            "        return 0, failure\n"
        )
        for last, sites, expected in (
            ("AssertionError: mismatch", (helper,), REPRODUCED),
            ("AssertionError: mismatch", (test,), INCOMPLETE),
            ("RuntimeError: NCCL error", (helper,), INCOMPLETE),
            # `torch.testing.assert_close` and Python 3.11+ position markers.
            (
                "AssertionError: Tensor-likes are not close!\n\n"
                "Mismatched elements: 1 / 4 (25.0%)\n"
                "Greatest absolute difference: 1.0 at index (0,)",
                (helper,),
                REPRODUCED,
            ),
            ("    ^^^^^^^^^^^^\nAssertionError: mismatch", (helper,), REPRODUCED),
            # Only the exception line names the exception, not its message.
            ("RuntimeError: NCCL error\nAssertionError: x", (helper,), INCOMPLETE),
        ):
            with self.subTest(last=last, sites=sites):
                body = "raise ChildFailedError('worker 0 failed')"
                code, _ = self._classify(body, sites, prelude=launcher, child_last=last)
                self.assertEqual(code, expected)

    def test_a_worker_without_a_traceback_names_the_missing_record(self) -> None:
        launcher = (
            "class ChildFailedError(Exception):\n"
            "    def get_first_failure(self):\n"
            "        failure = type('F', (), {})()\n"
            "        failure.message = 'To enable traceback see: ...'\n"
            "        return 0, failure\n"
        )
        code, reason = self._classify(
            "raise ChildFailedError('worker 0 failed')",
            ("test_x.py::helper",),
            prelude=launcher,
        )
        self.assertEqual(code, INCOMPLETE)
        self.assertIn("@record", reason)

    def test_the_innermost_in_tree_frame_decides(self) -> None:
        test, helper = "test_x.py::test_x", "test_x.py::helper"
        cases = [
            ("assert 1 == 1", (test,), PASSED),
            ("assert 1 == 2", (test,), REPRODUCED),
            ("raise RuntimeError('CUDA error')", (test,), INCOMPLETE),
            ("import pytest; pytest.skip('no GPU')", (test,), INCOMPLETE),
            # A helper's assertion is the helper's site, not the test's.
            ("helper()", (test,), INCOMPLETE),
            ("helper()", (test, helper), REPRODUCED),
            ("helper()", (), REPRODUCED),
            # Installed packages are never sites; their caller is.
            ("import checks; checks.check()", (test,), REPRODUCED),
            ("import checks; checks.check()", (helper,), INCOMPLETE),
        ]
        for body, sites, expected in cases:
            with self.subTest(body=body, sites=sites):
                self.assertEqual(self._classify(body, sites)[0], expected)


class OracleCliTest(unittest.TestCase):
    def test_arguments_pass_through_and_errors_are_incomplete(self) -> None:
        argv = ["pytest", "--rootdir", ".", "--nodeid", NODE, "--pytest-arg=-x"]
        fake = mock.Mock()
        fake.main.return_value = REPRODUCED

        @contextlib.contextmanager
        def loaded() -> Any:
            yield fake

        with mock.patch("cutracer.oracle.cli.pytest_oracle", loaded):
            invoked = CliRunner().invoke(oracle_command, argv)
        self.assertEqual(invoked.exit_code, REPRODUCED, invoked.output)
        fake.main.assert_called_once_with(
            ["--rootdir", ".", "--nodeid", NODE, "--pytest-arg=-x"],
            prog="cutracer oracle pytest",
        )
        fake.main.side_effect = RuntimeError("x")
        with mock.patch("cutracer.oracle.cli.pytest_oracle", loaded):
            invoked = CliRunner().invoke(oracle_command, argv)
        self.assertEqual(invoked.exit_code, INCOMPLETE)

    def test_usage_errors_exit_2(self) -> None:
        for extra in (["--assertion-site", "x.py"], ["--timeout", "0"]):
            with self.subTest(extra=extra):
                invoked = CliRunner().invoke(
                    oracle_command,
                    ["pytest", "--rootdir", ".", "--nodeid", NODE, *extra],
                )
                self.assertEqual(invoked.exit_code, 2)

    def test_the_exported_script_runs_without_cutracer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "bin" / "oracle.py"
            invoked = CliRunner().invoke(oracle_command, ["export", str(script)])
            self.assertEqual(invoked.exit_code, 0, invoked.output)
            self.assertTrue(os.access(script, os.X_OK))
            fx = _Fixture(root, None)
            env = {k: v for k, v in os.environ.items() if k != "CUDA_INJECTION64_PATH"}
            env.update(fx.env({"report": _report("failed")}, injected=False))
            proc = subprocess.run(
                [sys.executable, str(script), "--python", str(fx.python)]
                + ["--rootdir", tmp, "--nodeid", NODE, "--plain"],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(proc.returncode, REPRODUCED, proc.stderr)

    def test_registered_in_main_cli(self) -> None:
        from cutracer.cli import main

        self.assertIn("oracle", main.commands)
