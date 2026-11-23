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
import shutil
import signal
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
from cutracer.oracle.cli import loaded_oracle, oracle_command

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
    if "signal" in script:
        os.kill(os.getpid(), script["signal"])
    if script.get("sleep"):
        time.sleep(script["sleep"])
    # The capture's files, per numerics mode (a list: one entry per run).
    mode = os.environ.get("CUTRACER_NUMERICS_MODE", "")
    numerics = script.get("numerics", {}).get(mode) or {}
    if isinstance(numerics, list):
        counter = os.environ["FAKE_SEEN"] + "." + mode
        done = int(open(counter).read()) if os.path.exists(counter) else 0
        open(counter, "w").write(str(done + 1))
        numerics = numerics[done]
    for key, folder in (("out", "CUTRACER_NUMERICS_OUT"),
                        ("golden", "CUTRACER_NUMERICS_GOLDEN")):
        for name, value in numerics.get(key, {}).items():
            os.makedirs(os.environ[folder], exist_ok=True)
            with open(os.path.join(os.environ[folder], name), "w") as fh:
                json.dump(value, fh)
    result = numerics.get("report", script["report"])
    shown = {k: v for k, v in os.environ.items()
             if k.startswith("CUTRACER_NUMERICS_")
             or k in ("PYTHONHASHSEED", "PYTHONPATH", "PYTEST_PLUGINS")}
    with open(os.environ["FAKE_SEEN"], "w") as fh:
        json.dump({"argv": sys.argv[1:], "dump": dump,
                   "load": os.environ.get("CUTRACER_DELAY_LOAD_PATH"),
                   "events": os.environ.get("CUTRACER_KERNEL_EVENTS"),
                   "env": shown}, fh)
    with open(report, "w") as fh:
        json.dump(result, fh)
    sys.exit(result["pytest_exit"])
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
        self.oracle = stack.enter_context(loaded_oracle())
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.fx = _Fixture(self.root, self.oracle)


def _summary(verdict: str = "same", proc: str = "main", **kw: Any) -> dict[str, Any]:
    """A capture result file (`numerics-<proc>.json`)."""
    numerics: dict[str, Any] = {
        "verdict": verdict,
        "calls": 2,
        "golden_calls": 2,
        "environment": [],
        "structure": [],
        "first_structure_k": None,
        "first_large_k": None,
        "missing_calls": 0,
        "missing_head": [],
        "first_large": None,
        "changed": [],
        "changed_stable_leaves": 0,
        "stable_changed_head": [],
        "errors": [],
    }
    if verdict == "large":
        change = {"verdict": "large", "call": "t.py:3 #0", "leaf": "a0", "rel_l2": 0.3}
        numerics.update(first_large_k=0, first_large=change, changed=[change])
    if verdict == "recorded":
        numerics = {"verdict": "recorded", "calls": 2, "errors": []}
    numerics.update(kw)
    return {"schema": 1, "proc": proc, "numerics": numerics}


def _leaf_calls(verdict: str) -> dict[str, Any]:
    """A capture calls file whose one tensor compared as `verdict`."""
    cmp = {"verdict": verdict, "rel_l2": 0.001 if verdict != "same" else 0.0}
    leaf = {"name": "a0", "kind": "tensor", "cmp": cmp}
    return {"calls": [_call(leaf)]}


def _call(*leaves: dict[str, Any]) -> dict[str, Any]:
    return {
        "k": 0,
        "nodeid": NODE,
        "fn": "torch.testing.assert_close",
        "site": "tests/test_x.py:3",
        "ordinal": 0,
        "leaves": list(leaves),
    }


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

    def test_a_worker_killed_before_its_report_still_returns_the_dump(self) -> None:
        # CUTracer's kernel timeout ends the process that runs a hung kernel
        # (here the worker) after CUTracer dumped the delay config.
        requested = self.root / "requested.json"
        script = {
            "report": _report(),
            "config": _config(),
            "launch": True,
            "signal": int(signal.SIGTERM),
        }
        code = self.fx.run(script, env={"CUTRACER_DELAY_DUMP_PATH": str(requested)})
        self.assertEqual(code, INCOMPLETE)
        self.assertEqual(json.loads(requested.read_text()), _config())

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
        self.assertFalse((self.root / "x.json").exists())
        (attempt,) = (self.root / "evidence").glob("attempt-*")
        self.assertEqual(json.loads((attempt / "delay.json").read_text()), _config())

    def test_a_node_past_its_timeout_is_incomplete(self) -> None:
        requested = self.root / "requested.json"
        script = {"report": _report(), "config": _config(), "launch": True, "sleep": 30}
        code = self.fx.run(
            script, timeout_s=3, env={"CUTRACER_DELAY_DUMP_PATH": str(requested)}
        )
        self.assertEqual(code, INCOMPLETE)
        self.assertIn("ran past", self.fx.outcome()["errors"][0])
        # The worker ran far enough to dump its delay config; the caller gets it.
        self.assertEqual(json.loads(requested.read_text()), _config())

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


class NumericsCheckTest(_OracleTestCase):
    def golden(self, plain_runs: int = 2, **manifest: Any) -> Path:
        golden = self.root / "golden"
        golden.mkdir(exist_ok=True)
        (golden / "calls-main.json").write_text(json.dumps({"calls": []}))
        values = {
            "oracle": "pytest",
            "nodeid": NODE,
            "rootdir": str(self.root),
            "settings": {
                "CUTRACER_NUMERICS_PIN_HYPOTHESIS": "1",
                "CUTRACER_NUMERICS_WRAP": "pkg.ops:*",
                "PYTHONHASHSEED": "0",
            },
            "plain_runs": plain_runs,
            **manifest,
        }
        (golden / "manifest.json").write_text(json.dumps(values))
        return golden

    def check(
        self,
        out: dict[str, Any],
        report: dict[str, Any] | None = None,
        strict: bool = False,
        **kw: Any,
    ) -> int:
        numerics = self.oracle.NumericsOptions(
            mode="check", golden=kw.pop("golden", None) or self.golden(), strict=strict
        )
        script = {
            "report": report or _report(),
            "numerics": {"check": {"out": out}},
        }
        kw.setdefault("output_dir", None)
        return self.fx.run(script, injected=False, plain=True, numerics=numerics, **kw)

    def test_the_worker_loads_the_capture_with_the_golden_settings(self) -> None:
        golden = self.golden()
        code = self.check(
            {"numerics-main.json": _summary()},
            golden=golden,
            env={"CUTRACER_NUMERICS_FOO": "1", "CUTRACER_NUMERICS_WRAP": "other:x"},
            output_dir=self.root / "evidence",
        )
        self.assertEqual(code, PASSED)
        seen = json.loads(self.fx.seen.read_text())
        module = Path(seen["argv"][seen["argv"].index("--numerics-module") + 1])
        self.assertEqual(module.name, "cutracer_numerics.py")
        self.assertEqual(module.parent.name, "numerics_lib")
        env = seen["env"]
        self.assertEqual(env["CUTRACER_NUMERICS_MODE"], "check")
        self.assertEqual(env["CUTRACER_NUMERICS_GOLDEN"], str(golden.resolve()))
        self.assertEqual(env["CUTRACER_NUMERICS_ROOT"], str(self.root))
        self.assertEqual(env["CUTRACER_NUMERICS_WRAP"], "pkg.ops:*")
        self.assertEqual(env["PYTHONHASHSEED"], "0")
        self.assertNotIn("CUTRACER_NUMERICS_FOO", env)
        self.assertNotIn("PYTEST_PLUGINS", env)
        outcome = self.fx.outcome()
        self.assertEqual(outcome["numerics"]["rule"], "comparable")
        self.assertIn("numerics same over 2 calls", outcome["reason"])

    def test_without_numerics_inherited_capture_settings_are_dropped(self) -> None:
        script = {"report": _report()}
        env = {"CUTRACER_NUMERICS_MODE": "check", "CUTRACER_NUMERICS_GOLDEN": "/x"}
        self.assertEqual(
            self.fx.run(script, injected=False, plain=True, env=env), PASSED
        )
        seen = json.loads(self.fx.seen.read_text())
        self.assertNotIn("--numerics-module", seen["argv"])
        self.assertEqual(
            [k for k in seen["env"] if k.startswith("CUTRACER_NUMERICS_")], []
        )

    def test_a_large_change_of_a_passing_test_reproduces(self) -> None:
        sig = self.root / "sig"
        code = self.check({"numerics-main.json": _summary("large")}, signature_file=sig)
        self.assertEqual(code, REPRODUCED)
        self.assertEqual(
            sig.read_text().strip(), self.oracle.numerics_signature([NODE])
        )
        # A later reproduction by an assertion has the same signature.
        failed = self.check(
            {"numerics-main.json": _summary()}, _report("failed"), signature_file=sig
        )
        self.assertEqual(failed, REPRODUCED)
        sig.write_text("another failure\n")
        self.assertEqual(
            self.check({"numerics-main.json": _summary("large")}, signature_file=sig),
            INCOMPLETE,
        )

    def test_a_large_change_after_calls_stop_lining_up_measured_nothing(self) -> None:
        late = _summary(
            "large", first_large_k=3, first_structure_k=1, structure=["x: extra call"]
        )
        self.assertEqual(self.check({"numerics-main.json": late}), INCOMPLETE)
        early = _summary(
            "large", first_large_k=0, first_structure_k=2, structure=["x: extra call"]
        )
        self.assertEqual(self.check({"numerics-main.json": early}), REPRODUCED)

    def test_a_small_change_of_a_passing_test_passes(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = self.check({"numerics-main.json": _summary("small")})
        self.assertEqual(code, PASSED)
        self.assertIn("the node passed; numerics small over 2 calls", stderr.getvalue())

    def test_strict_counts_a_change_of_a_leaf_stable_in_plain_runs(self) -> None:
        change = {"verdict": "small", "call": "c", "leaf": "a0", "plain_stable": True}
        summary = _summary(
            "small",
            changed_stable_leaves=1,
            changed=[change],
            stable_changed_head=[change],
        )
        out = {"numerics-main.json": summary}
        self.assertEqual(self.check(out), PASSED)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(self.check(out, strict=True), REPRODUCED)
        self.assertIn(
            "bitwise identical in every plain run in main: c a0", stderr.getvalue()
        )
        golden = self.golden(plain_runs=0)
        self.assertEqual(self.check(out, strict=True, golden=golden), INCOMPLETE)

    def test_a_correctness_failure_reproduces_whatever_its_numerics(self) -> None:
        broken = _summary(errors=["boom"], missing_calls=1)
        self.assertEqual(
            self.check({"numerics-main.json": broken}, _report("failed")), REPRODUCED
        )
        self.assertEqual(self.check({}, _report("failed")), REPRODUCED)

    def test_runs_that_cannot_be_compared_measured_nothing(self) -> None:
        cases: dict[str, dict[str, Any]] = {
            "no result": {},
            "an unknown process": {
                "numerics-main.json": _summary(),
                "numerics-rank-1.json": _summary(proc="rank-1"),
            },
            "an unlabeled process": {
                "numerics-main.json": _summary(),
                "unlabeled-7.json": {"pid": 7},
            },
            "a second writer": {
                "numerics-main.json": _summary(),
                "conflict-main-7.json": {},
            },
            "a write failure": {"error-main-7.json": {}},
            "capture errors": {"numerics-main.json": _summary(errors=["boom"])},
            "calls that do not line up": {
                "numerics-main.json": _summary(
                    structure=["x: extra call"], first_structure_k=1
                )
            },
            "missing calls": {
                "numerics-main.json": _summary(missing_calls=1, missing_head=["c"])
            },
            "an unmeasured change": {"numerics-main.json": _summary("unmeasured")},
            "another torch": {
                "numerics-main.json": _summary(environment=["torch 2 differs from 1"])
            },
        }
        for name, out in cases.items():
            with self.subTest(name=name):
                self.assertEqual(self.check(out), INCOMPLETE)

    def test_a_second_writer_discredits_a_large_change_but_an_unlabeled_one_does_not(
        self,
    ) -> None:
        large = _summary("large")
        conflict = {"numerics-main.json": large, "conflict-main-7.json": {}}
        self.assertEqual(self.check(conflict), INCOMPLETE)
        unlabeled = {"numerics-main.json": large, "unlabeled-7.json": {}}
        self.assertEqual(self.check(unlabeled), REPRODUCED)

    def test_a_result_this_oracle_cannot_read_measured_nothing(self) -> None:
        missing_key = _summary("large")
        del missing_key["numerics"]["first_large_k"]
        other_schema = dict(_summary("large"), schema=2)
        no_verdict = _summary("large")
        no_verdict["numerics"]["verdict"] = "weird"
        for name, result in (
            ("a missing key", missing_key),
            ("another schema", other_schema),
            ("an unknown verdict", no_verdict),
        ):
            with self.subTest(name=name):
                self.assertEqual(self.check({"numerics-main.json": result}), INCOMPLETE)

    def test_the_reason_names_the_large_change_that_decided(self) -> None:
        first = {"verdict": "large", "call": "first", "leaf": "a0", "rel_l2": 0.1}
        worst = {"verdict": "large", "call": "worst", "leaf": "b0", "rel_l2": 0.9}
        summary = _summary("large", first_large=first, changed=[worst, first])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(self.check({"numerics-main.json": summary}), REPRODUCED)
        self.assertIn("large numerics change in main: first a0", stderr.getvalue())

    def test_an_expected_signature_applies_to_numerics_reproductions(self) -> None:
        out = {"numerics-main.json": _summary("large")}
        self.assertEqual(self.check(out, expected_signature="other"), INCOMPLETE)
        own = self.oracle.numerics_signature([NODE])
        self.assertEqual(self.check(out, expected_signature=own), REPRODUCED)

    def test_a_golden_whose_plain_runs_reproduced_is_refused(self) -> None:
        golden = self.golden(unstable="plain run 2: large numerics change")
        self.assertEqual(
            self.check({"numerics-main.json": _summary()}, golden=golden), INCOMPLETE
        )

    def test_autotuner_misses_discredit_only_a_large_change_they_precede(
        self,
    ) -> None:
        def misses(first_miss_k: int) -> dict[str, Any]:
            return {
                "pinning": "on",
                "misses": 2,
                "first_miss_k": first_miss_k,
                "miss_head": ["k: not in the record"],
            }

        early = {"numerics-main.json": _summary("large", autotune=misses(0))}
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(self.check(early), INCOMPLETE)
        self.assertIn(
            "2 autotuner call(s) were not in the golden's record", stderr.getvalue()
        )
        # A miss after the change may be the race's own doing.
        late = {"numerics-main.json": _summary("large", autotune=misses(3))}
        self.assertEqual(self.check(late), REPRODUCED)
        small = {"numerics-main.json": _summary("small", autotune=misses(0))}
        self.assertEqual(self.check(small), PASSED)
        # A wrong result is wrong in any config.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(self.check(small, _report("failed")), REPRODUCED)
        self.assertIn("2 autotuner call(s) were not pinned", stderr.getvalue())

    def test_autotuning_that_was_not_pinned_discredits_a_large_change(self) -> None:
        unpinned = {
            "pinning": "missed: triton.runtime.autotuner was imported",
            "misses": 0,
        }
        large = {"numerics-main.json": _summary("large", autotune=unpinned)}
        self.assertEqual(self.check(large), INCOMPLETE)

    def test_a_large_change_with_another_torch_measured_nothing(self) -> None:
        summary = _summary("large", environment=["torch 2 differs from 1"])
        self.assertEqual(self.check({"numerics-main.json": summary}), INCOMPLETE)

    def test_missing_calls_after_a_failure_are_an_early_end(self) -> None:
        summary = _summary(missing_calls=1, calls=1)
        self.assertEqual(
            self.check({"numerics-main.json": summary}, _report("failed")), REPRODUCED
        )

    def test_a_skipped_test_measured_nothing_despite_a_large_change(self) -> None:
        skipped = _report("skipped", pytest_exit=0)
        self.assertEqual(
            self.check({"numerics-main.json": _summary("large")}, skipped), INCOMPLETE
        )

    def test_coverage_is_still_required_of_an_injected_run(self) -> None:
        numerics = self.oracle.NumericsOptions(mode="check", golden=self.golden())
        script = {
            "report": _report(),
            "config": _config(),
            "numerics": {"check": {"out": {"numerics-main.json": _summary("large")}}},
        }
        self.assertEqual(
            self.fx.run(script, numerics=numerics, output_dir=None), INCOMPLETE
        )

    def test_a_check_needs_the_golden_of_the_same_test(self) -> None:
        out = {"numerics-main.json": _summary()}
        no_manifest = self.root / "bare"
        no_manifest.mkdir()
        self.assertEqual(self.check(out, golden=no_manifest), INCOMPLETE)
        other = self.golden(nodeid="tests/test_x.py::test_z")
        self.assertEqual(self.check(out, golden=other), INCOMPLETE)
        numerics = self.oracle.NumericsOptions(
            mode="check", golden=self.golden(nodeid=NODE), wrap=("other:x",)
        )
        script = {"report": _report(), "numerics": {"check": {"out": out}}}
        self.assertEqual(
            self.fx.run(
                script, injected=False, plain=True, numerics=numerics, output_dir=None
            ),
            INCOMPLETE,
        )


class NumericsRecordTest(_OracleTestCase):
    def record(self, script: dict[str, Any], runs: int = 3, **kw: Any) -> int:
        numerics = self.oracle.NumericsOptions(
            mode="record", golden=self.root / "golden", runs=runs
        )
        kw.setdefault("injected", False)
        return self.fx.run(script, numerics=numerics, **kw)

    def script(
        self, *checks: dict[str, Any], calls: int = 2, record_misses: int = 0
    ) -> dict[str, Any]:
        golden_calls = {"schema": 1, "calls": [_call({"name": "a0", "kind": "tensor"})]}
        autotune = {"verdict": "autotune-recorded", "calls": 0, "errors": []}
        recorded = _summary("recorded", calls=calls)
        recorded["numerics"]["autotune"] = {"pinning": "on", "misses": record_misses}
        return {
            "report": _report(),
            "numerics": {
                # Run 0 records the autotuner's choices.
                "autotune": {
                    "out": {"numerics-main.json": {"schema": 1, "numerics": autotune}}
                },
                "record": {
                    "out": {"numerics-main.json": recorded},
                    "golden": {"calls-main.json": golden_calls},
                },
                "check": list(checks),
            },
        }

    def plain_run(self, verdict: str = "small", **kw: Any) -> dict[str, Any]:
        summary = _summary(verdict, **kw)
        return {
            "out": {
                "numerics-main.json": summary,
                "calls-main.json": _leaf_calls(verdict),
            }
        }

    def test_three_plain_runs_record_a_golden_and_its_stability(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = self.record(
                self.script(self.plain_run("same"), self.plain_run("small"))
            )
        self.assertEqual(code, 0, stderr.getvalue())
        verdicts = [
            json.loads(path.read_text())["numerics"]["verdict"]
            for path in (self.root / "evidence").glob(
                "attempt-*/numerics/numerics-main.json"
            )
        ]
        self.assertEqual(verdicts.count("autotune-recorded"), 1)
        golden = self.root / "golden"
        manifest = json.loads((golden / "manifest.json").read_text())
        self.assertEqual(
            (manifest["oracle"], manifest["nodeid"], manifest["plain_runs"]),
            ("pytest", NODE, 2),
        )
        self.assertEqual(manifest["settings"]["CUTRACER_NUMERICS_PIN_HYPOTHESIS"], "1")
        self.assertEqual(manifest["settings"]["CUTRACER_NUMERICS_PIN_AUTOTUNE"], "1")
        self.assertEqual(manifest["settings"]["PYTHONHASHSEED"], "0")
        leaf = json.loads((golden / "calls-main.json").read_text())["calls"][0][
            "leaves"
        ][0]
        self.assertEqual(
            leaf["plain"], {"runs": 2, "changed": 1, "max_rel_l2": 0.001, "max_ulp": 0}
        )
        stability = json.loads((golden / "stability.json").read_text())
        self.assertEqual(
            [r.get("verdict") for r in stability["runs"]], [None, "same", "small"]
        )
        on_disk = [p for p in golden.rglob("*") if p.is_file()]
        self.assertEqual(
            stability["golden_bytes"],
            sum(p.stat().st_size for p in on_disk if p.name != "stability.json"),
        )
        self.assertEqual(list(self.root.glob("golden.partial-*")), [])
        self.assertRegex(
            stderr.getvalue(), r"calls, \d+(\.\d)? [KM]iB; plain runs: same, small"
        )

    def test_a_large_change_between_plain_runs_exits_3(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            code = self.record(
                self.script(self.plain_run("large"), self.plain_run("small"))
            )
        self.assertEqual(code, 3)
        stability = json.loads((self.root / "golden" / "stability.json").read_text())
        self.assertEqual(stability["exit"], 3)

    def test_a_failed_record_leaves_no_golden(self) -> None:
        cases = {
            "nothing compared": self.script(self.plain_run(), calls=0),
            "a plain run that does not line up": self.script(
                self.plain_run("same", structure=["x: extra call"]), self.plain_run()
            ),
            "an injected environment": self.script(self.plain_run(), self.plain_run()),
        }
        for name, script in cases.items():
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                injected = name == "an injected environment"
                self.assertEqual(self.record(script, injected=injected), INCOMPLETE)
                self.assertFalse((self.root / "golden").exists())
                self.assertEqual(list(self.root.glob("golden.partial-*")), [])
                for counter in self.root.glob("seen.json.*"):
                    counter.unlink()

    def test_a_plain_run_that_fails_its_check_exits_3_and_refuses_checks(self) -> None:
        failed = self.plain_run("same")
        failed["report"] = _report("failed")
        with contextlib.redirect_stderr(io.StringIO()):
            code = self.record(self.script(failed, self.plain_run("small")))
        self.assertEqual(code, 3)
        golden = self.root / "golden"
        manifest = json.loads((golden / "manifest.json").read_text())
        self.assertIn("plain run 2", manifest["unstable"])
        leaf = json.loads((golden / "calls-main.json").read_text())["calls"][0][
            "leaves"
        ][0]
        self.assertEqual(leaf["plain"]["runs"], 2)

    def test_an_integer_tensor_that_varies_between_plain_runs_is_not_instability(
        self,
    ) -> None:
        varying = self.plain_run("large")
        varying["out"]["calls-main.json"]["calls"][0]["leaves"][0]["cmp"]["kind"] = (
            "exact"
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = self.record(self.script(varying, self.plain_run("same")))
        self.assertEqual(code, 0, stderr.getvalue())
        golden = self.root / "golden"
        self.assertIsNone(
            json.loads((golden / "manifest.json").read_text())["unstable"]
        )
        leaf = json.loads((golden / "calls-main.json").read_text())["calls"][0][
            "leaves"
        ][0]
        self.assertEqual(leaf["plain"]["changed"], 1)
        self.assertIn("an integer tensor that varies anyway", stderr.getvalue())

    def test_a_complex_tensor_that_varies_between_plain_runs_is_instability(
        self,
    ) -> None:
        varying = self.plain_run("large")
        cmp = varying["out"]["calls-main.json"]["calls"][0]["leaves"][0]["cmp"]
        cmp.update(kind="exact", dtype="complex64")
        with contextlib.redirect_stderr(io.StringIO()):
            code = self.record(self.script(varying, self.plain_run("same")))
        self.assertEqual(code, 3)

    def test_each_run_is_labeled_and_a_failure_keeps_its_evidence(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            ok = self.record(
                self.script(self.plain_run("same"), self.plain_run("same"))
            )
        self.assertEqual(ok, 0)
        self.assertIn("numerics record run 1/3 (golden): exit 1", stderr.getvalue())
        self.assertIn(
            "numerics record run 3/3 (plain check): exit 1", stderr.getvalue()
        )
        shutil.rmtree(self.root / "golden")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            failed = self.record(
                self.script(self.plain_run(), calls=0), output_dir=None
            )
        self.assertEqual(failed, INCOMPLETE)
        kept = stderr.getvalue().split("evidence in ")[-1].strip()
        self.assertTrue(list(Path(kept).glob("attempt-*")), stderr.getvalue())
        shutil.rmtree(kept)

    def test_staging_is_removed_when_a_run_raises(self) -> None:
        numerics = self.oracle.NumericsOptions(
            mode="record", golden=self.root / "golden"
        )

        def broken(mode: str, golden: Path, out: Path, label: str) -> Any:
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError), contextlib.redirect_stderr(io.StringIO()):
            self.oracle.record_golden(numerics, {}, self.root / "evidence", broken)
        self.assertEqual(list(self.root.glob("golden*")), [])

    def test_plain_records_note_a_non_finite_relative_error(self) -> None:
        out = self.root / "run"
        out.mkdir()
        calls = _leaf_calls("small")
        calls["calls"][0]["leaves"][0]["cmp"]["rel_l2"] = float("nan")
        (out / "calls-main.json").write_text(json.dumps(calls))
        stats = self.oracle._plain_stats([out])
        (record,) = stats.values()
        self.assertEqual(
            (record["max_rel_l2"], record.get("nonfinite_rel_l2")), (0.0, True)
        )

    def test_autotuner_misses_while_recording_leave_no_golden(self) -> None:
        missed = self.plain_run("same")
        missed["out"]["numerics-main.json"]["numerics"]["autotune"] = {
            "pinning": "on",
            "misses": 1,
            "miss_head": ["k: not in the golden's record"],
        }
        cases = {
            "run 1": self.script(self.plain_run(), self.plain_run(), record_misses=1),
            "a plain run": self.script(missed, self.plain_run()),
        }
        for name, script in cases.items():
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.record(script), INCOMPLETE)
                self.assertFalse((self.root / "golden").exists())
                for counter in self.root.glob("seen.json.*"):
                    counter.unlink()

    def test_without_autotune_pinning_there_is_no_autotuning_run(self) -> None:
        numerics = self.oracle.NumericsOptions(
            mode="record", golden=self.root / "golden", runs=2, pin_autotune=False
        )
        script = self.script(self.plain_run())
        del script["numerics"]["autotune"]
        with contextlib.redirect_stderr(io.StringIO()):
            code = self.fx.run(script, injected=False, numerics=numerics)
        self.assertEqual(code, 0)
        manifest = json.loads((self.root / "golden" / "manifest.json").read_text())
        self.assertEqual(manifest["settings"]["CUTRACER_NUMERICS_PIN_AUTOTUNE"], "0")

    def test_a_golden_directory_must_be_new_or_empty(self) -> None:
        (self.root / "golden").mkdir()
        (self.root / "golden" / "old").write_text("x")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.record(self.script(self.plain_run())), INCOMPLETE)
        self.assertEqual(
            sorted(p.name for p in (self.root / "golden").iterdir()), ["old"]
        )

    def test_numerics_usage_errors_exit_2(self) -> None:
        base = ["--rootdir", str(self.root), "--nodeid", NODE]
        for extra in (
            ["--numerics", "check"],
            ["--golden", "g"],
            ["--numerics-wrap", "m:f"],
            ["--numerics", "check", "--golden", "g", "--numerics-runs", "2"],
            ["--numerics", "record", "--golden", "g", "--numerics-strict"],
            ["--numerics", "record", "--golden", "g", "--numerics-runs", "0"],
            ["--numerics", "check", "--golden", "g", "--numerics-wrap", "nocolon"],
            ["--numerics", "record", "--golden", "g", "--signature-file", "s"],
            ["--numerics", "record", "--golden", "g", "--expected-signature", "x"],
            ["--numerics", "check", "--golden", "g", "--no-pin-autotune"],
            ["--no-pin-autotune"],
        ):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.oracle.main([*base, *extra])
                self.assertEqual(raised.exception.code, 2)


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
        installed: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        packages = root / "venv/lib/python3/site-packages"
        packages.mkdir(parents=True)
        (packages / "checks.py").write_text("def check():\n    assert 1 == 2\n")
        for name, text in (installed or {}).items():
            (packages / name).parent.mkdir(parents=True, exist_ok=True)
            (packages / name).write_text(text)
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
        # Every case writes its own `test_x.py`, and pytest reuses a module that
        # is already in `sys.modules` under the same name.
        with (
            mock.patch.dict(os.environ, {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}),
            mock.patch.dict(sys.modules),
        ):
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

    def test_modules_the_test_imports_do_not_stop_the_report(self) -> None:
        installed = {
            # Like `torch.classes`: every attribute exists, `__path__` included.
            "import classes": {
                "classes.py": (
                    "import sys, types\n"
                    "class Classes(types.ModuleType):\n"
                    "    __file__ = 'classes.py'\n"
                    "    def __getattr__(self, name):\n"
                    "        return types.ModuleType(name)\n"
                    "sys.modules[__name__] = Classes(__name__)\n"
                )
            },
            # `pkg.ns` is a namespace package: its `__path__` comes from `pkg`.
            "import pkg.ns.mod": {"pkg/__init__.py": "", "pkg/ns/mod.py": ""},
        }
        test = ("test_x.py::test_x",)
        for statement, files in installed.items():
            for body, expected in (
                ("assert 1 == 1", PASSED),
                ("assert 1 == 2", REPRODUCED),
            ):
                with self.subTest(statement, body=body):
                    code, _ = self._classify(
                        body, test, prelude=f"{statement}\n", installed=files
                    )
                    self.assertEqual(code, expected)


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
        # The oracle finds the capture resource, wherever it is unpacked.
        self.assertEqual(Path(fake.NUMERICS_CAPTURE).name, "cutracer_numerics.py")
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

    def test_the_exported_script_checks_numerics_with_the_capture_beside_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "bin" / "oracle.py"
            invoked = CliRunner().invoke(oracle_command, ["export", str(script)])
            self.assertEqual(invoked.exit_code, 0, invoked.output)
            self.assertTrue((root / "bin" / "cutracer_numerics.py").is_file())
            golden = root / "golden"
            golden.mkdir()
            (golden / "calls-main.json").write_text(json.dumps({"calls": []}))
            manifest = {"oracle": "pytest", "nodeid": NODE, "rootdir": tmp}
            (golden / "manifest.json").write_text(json.dumps(manifest))
            fx = _Fixture(root, None)
            env = {k: v for k, v in os.environ.items() if k != "CUDA_INJECTION64_PATH"}
            check = {"out": {"numerics-main.json": _summary("large")}}
            env.update(
                fx.env(
                    {"report": _report(), "numerics": {"check": check}}, injected=False
                )
            )
            proc = subprocess.run(
                [sys.executable, str(script), "--python", str(fx.python)]
                + ["--rootdir", tmp, "--nodeid", NODE, "--plain"]
                + ["--numerics", "check", "--golden", str(golden)],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(proc.returncode, REPRODUCED, proc.stderr)
        self.assertIn("large numerics change in main", proc.stderr)

    def test_export_refuses_the_capture_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "cutracer_numerics.py")
            invoked = CliRunner().invoke(oracle_command, ["export", target])
        self.assertEqual(invoked.exit_code, 2, invoked.output)

    def test_registered_in_main_cli(self) -> None:
        from cutracer.cli import main

        self.assertIn("oracle", main.commands)
