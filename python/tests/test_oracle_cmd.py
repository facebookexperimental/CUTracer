# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for `cutracer oracle cmd`.

A fake command stands in for the workload: it plays CUTracer (writes the
delay dump and the kernel launch events) and the workload itself (prints
output, exits with a code, sleeps, or kills itself with a signal), so the
oracle's exit-code mapping, pattern matching, coverage and evidence
handling run without a GPU or cutracer.so. The oracle is a standalone
script shipped as a resource; the tests load it the way the CLI does.
"""

import contextlib
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from click.testing import CliRunner
from cutracer.oracle.cli import oracle_command, pytest_oracle

REPRODUCED, PASSED, INCOMPLETE = 0, 1, 2

_FAKE_CMD = textwrap.dedent(
    """
    import json, os, sys, time
    script = json.loads(os.environ["FAKE_SCRIPT"])
    if script.get("sleep"):
        time.sleep(script["sleep"])
    if "signal" in script:
        os.kill(os.getpid(), script["signal"])
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
    sys.stdout.write(script.get("output", ""))
    sys.stdout.flush()
    sys.exit(script.get("exit", 0))
    """
)


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
        self.cmd = root / "fake_cmd.py"
        self.cmd.write_text(_FAKE_CMD)
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
        command: list[str] | None = None,
        **kw: Any,
    ) -> int:
        options: dict[str, Any] = {
            "command": (
                command if command is not None else [sys.executable, str(self.cmd)]
            ),
            "kernels": "my_kernel",
            "output_dir": self.root / "evidence",
            **kw,
        }
        with mock.patch.dict(
            os.environ, self.env(script, injected=injected, **(env or {})), clear=False
        ):
            if not injected:
                os.environ.pop("CUDA_INJECTION64_PATH", None)
            return self.oracle.run_cmd_oracle(self.oracle.CmdOptions(**options))

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


class RunCmdOracleTest(_OracleTestCase):
    def test_fail_and_pass_codes_map_to_0_and_1(self) -> None:
        for exit_code, expected in ((1, REPRODUCED), (0, PASSED), (3, INCOMPLETE)):
            with self.subTest(exit=exit_code):
                script = {"exit": exit_code, "config": _config(), "launch": True}
                self.assertEqual(self.fx.run(script, output_dir=None), expected)
        custom = {"exit": 7, "config": _config(), "launch": True}
        self.assertEqual(
            self.fx.run(custom, fail_codes=frozenset({7}), output_dir=None),
            REPRODUCED,
        )

    def test_a_passing_covered_run_copies_the_dump(self) -> None:
        requested = self.root / "requested.json"
        code = self.fx.run(
            {"exit": 0, "config": _config(), "launch": True},
            command=[sys.executable, str(self.fx.cmd), "--foo", "bar"],
            env={"CUTRACER_DELAY_DUMP_PATH": str(requested)},
        )
        self.assertEqual(code, PASSED)
        self.assertEqual(json.loads(requested.read_text()), _config())
        seen = json.loads(self.fx.seen.read_text())
        # The child dumps into the attempt's evidence, with kernel events on,
        # and sees the command's own arguments.
        self.assertNotEqual(seen["dump"], str(requested))
        self.assertEqual(seen["events"], "nostack")
        self.assertEqual(seen["argv"], ["--foo", "bar"])
        outcome = self.fx.outcome()
        self.assertEqual((outcome["exit"], outcome["coverage"]), (PASSED, True))

    def test_a_killed_command_measured_nothing(self) -> None:
        script = {"signal": int(signal.SIGKILL)}
        self.assertEqual(self.fx.run(script, injected=False, plain=True), INCOMPLETE)
        self.assertIn(
            f"killed by signal {int(signal.SIGKILL)}", self.fx.outcome()["reason"]
        )
        injected = {"signal": int(signal.SIGKILL), "config": _config(), "launch": True}
        self.assertEqual(self.fx.run(injected, output_dir=None), INCOMPLETE)

    def test_a_command_past_its_timeout_is_incomplete(self) -> None:
        script = {"exit": 0, "sleep": 30, "config": _config(), "launch": True}
        self.assertEqual(self.fx.run(script, timeout_s=0.5), INCOMPLETE)
        self.assertIn("ran past", self.fx.outcome()["errors"][0])

    def test_a_fail_pattern_must_match(self) -> None:
        matched = {
            "exit": 1,
            "output": "AssertionError: mismatch\n",
            "config": _config(),
            "launch": True,
        }
        self.assertEqual(
            self.fx.run(matched, fail_pattern="AssertionError", output_dir=None),
            REPRODUCED,
        )
        unmatched = {
            "exit": 1,
            "output": "all ok\n",
            "config": _config(),
            "launch": True,
        }
        self.assertEqual(
            self.fx.run(unmatched, fail_pattern="AssertionError", output_dir=None),
            INCOMPLETE,
        )
        # Without a pattern the exit code alone decides.
        self.assertEqual(self.fx.run(unmatched, output_dir=None), REPRODUCED)

    def test_a_reproduction_pins_its_matched_text(self) -> None:
        sig = self.root / "sig"
        script = {
            "exit": 1,
            "output": "AssertionError: x=1\n",
            "config": _config(),
            "launch": True,
        }
        self.assertEqual(
            self.fx.run(script, fail_pattern="AssertionError.*", signature_file=sig),
            REPRODUCED,
        )
        expected = hashlib.sha256(b"AssertionError: x=1").hexdigest()
        self.assertEqual(sig.read_text().strip(), expected)
        other = {
            "exit": 1,
            "output": "AssertionError: x=2\n",
            "config": _config(),
            "launch": True,
        }
        self.assertEqual(
            self.fx.run(
                other,
                fail_pattern="AssertionError.*",
                signature_file=sig,
                output_dir=None,
            ),
            INCOMPLETE,
        )

    def test_without_a_pattern_the_exit_code_is_the_signature(self) -> None:
        sig = self.root / "sig"
        script = {"exit": 1, "config": _config(), "launch": True}
        self.assertEqual(
            self.fx.run(script, fail_codes=frozenset({1, 2}), signature_file=sig),
            REPRODUCED,
        )
        self.assertEqual(sig.read_text().strip(), hashlib.sha256(b"exit:1").hexdigest())
        other = {"exit": 2, "config": _config(), "launch": True}
        self.assertEqual(
            self.fx.run(
                other, fail_codes=frozenset({1, 2}), signature_file=sig, output_dir=None
            ),
            INCOMPLETE,
        )

    def test_an_empty_signature_file_pins_the_next_reproduction(self) -> None:
        sig = self.root / "sig"
        sig.write_text("")
        script = {"exit": 1, "output": "boom\n", "config": _config(), "launch": True}
        self.assertEqual(
            self.fx.run(script, fail_pattern="boom", signature_file=sig), REPRODUCED
        )
        self.assertEqual(sig.read_text(), hashlib.sha256(b"boom").hexdigest() + "\n")
        self.assertEqual([p.name for p in self.root.glob(".sig*")], [])

    def test_the_exit_code_and_its_reason_go_to_stderr(self) -> None:
        script = {"exit": 0, "config": _config(), "launch": True}
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
                "exit": 0,
                "config": _config(points=False),
                "launch": True,
            },
            "not launched": {"exit": 0, "config": _config()},
            "no dump": {"exit": 0, "launch": True},
        }
        for name, script in cases.items():
            with self.subTest(name=name):
                self.assertEqual(self.fx.run(script, output_dir=None), INCOMPLETE)

    def test_a_kernel_filter_must_match(self) -> None:
        script = {"exit": 0, "config": _config(), "launch": True}
        self.assertEqual(self.fx.run(script, kernels="other_kernel"), INCOMPLETE)

    def test_plain_and_injected_runs_check_the_injection(self) -> None:
        script = {"exit": 0}
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
            {"exit": 1, "launch": True},
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

    def test_without_an_output_dir_nothing_is_kept(self) -> None:
        script = {"exit": 0, "config": _config(), "launch": True}
        scratch = self.root / "tmp"
        scratch.mkdir()
        with mock.patch.object(tempfile, "tempdir", str(scratch)):
            self.assertEqual(self.fx.run(script, output_dir=None), PASSED)
        self.assertEqual(list(scratch.iterdir()), [])

    def test_bad_options_measure_nothing(self) -> None:
        script = {"exit": 1, "config": _config(), "launch": True}
        self.assertEqual(
            self.fx.run(
                script,
                fail_codes=frozenset({1}),
                pass_codes=frozenset({1}),
                output_dir=None,
            ),
            INCOMPLETE,
        )
        self.assertEqual(self.fx.run(script, command=[], output_dir=None), INCOMPLETE)
        self.assertEqual(
            self.fx.run(script, fail_codes=frozenset(), output_dir=None), INCOMPLETE
        )
        self.assertEqual(
            self.fx.run(script, fail_pattern="([", output_dir=None), INCOMPLETE
        )


class ClassifyCmdTest(_OracleTestCase):
    def test_outcomes(self) -> None:
        classify = self.oracle.classify_cmd
        code, reason, sig = classify(1, "boom", frozenset({1}), frozenset({0}))
        self.assertEqual((code, reason), (REPRODUCED, "the command exited 1"))
        self.assertEqual(sig, hashlib.sha256(b"exit:1").hexdigest())
        code, reason, sig = classify(1, "boom", frozenset({1}), frozenset({0}), "bo+m")
        self.assertEqual(code, REPRODUCED)
        self.assertIn("matching 'bo+m'", reason)
        self.assertEqual(sig, hashlib.sha256(b"boom").hexdigest())
        self.assertEqual(classify(0, "", frozenset({1}), frozenset({0}))[0], PASSED)

    def test_anything_else_measured_nothing(self) -> None:
        classify = self.oracle.classify_cmd
        fail, passed = frozenset({1}), frozenset({0})
        cases = {
            "signal": (classify(-9, "", fail, passed), "killed by signal 9"),
            "no match": (
                classify(1, "all ok", fail, passed, "AssertionError"),
                "matched no 'AssertionError'",
            ),
            "other code": (classify(3, "", fail, passed), "neither a fail code"),
            "other signature": (
                classify(1, "boom", fail, passed, None, "pinned"),
                "not the pinned failure",
            ),
        }
        for name, ((code, reason, _), fragment) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(code, INCOMPLETE)
                self.assertIn(fragment, reason)


class MainCmdTest(_OracleTestCase):
    def test_the_command_after_dash_dash_runs(self) -> None:
        argv = [
            "--fail-codes",
            "3",
            "--pass-codes",
            "0",
            "--plain",
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(3)",
        ]
        env = {k: v for k, v in os.environ.items() if k != "CUDA_INJECTION64_PATH"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(self.oracle.main_cmd(argv), REPRODUCED)

    def test_usage_errors_exit_2(self) -> None:
        cases = [
            [],
            ["--fail-codes", "1"],
            ["--fail-codes", "1", "--pass-codes", "1", "--", "true"],
            ["--fail-codes", "x", "--", "true"],
            ["--fail-codes", "", "--", "true"],
            ["--pass-codes", ",", "--", "true"],
            ["--fail-pattern", "([", "--", "true"],
            ["--timeout", "0", "--", "true"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as ctx:
                        self.oracle.main_cmd(argv)
                self.assertEqual(ctx.exception.code, 2)

    def test_main_dispatches_cmd(self) -> None:
        with mock.patch.object(self.oracle, "main_cmd", return_value=PASSED) as main:
            self.assertEqual(self.oracle.main(["cmd", "--", "true"]), PASSED)
        main.assert_called_once_with(["--", "true"], prog=None)


class OracleCmdCliTest(unittest.TestCase):
    def test_arguments_pass_through_and_errors_are_incomplete(self) -> None:
        argv = [
            "cmd",
            "--fail-codes",
            "1,2",
            "--fail-pattern",
            "boom",
            "--timeout",
            "5",
            "-k",
            "my_kernel",
            "--",
            "python",
            "test.py",
            "--foo",
        ]
        fake = mock.Mock()
        fake.main_cmd.return_value = REPRODUCED

        @contextlib.contextmanager
        def loaded() -> Any:
            yield fake

        with mock.patch("cutracer.oracle.cli.pytest_oracle", loaded):
            invoked = CliRunner().invoke(oracle_command, argv)
        self.assertEqual(invoked.exit_code, REPRODUCED, invoked.output)
        fake.main_cmd.assert_called_once_with(
            [
                "--fail-codes",
                "1,2",
                "--fail-pattern",
                "boom",
                "--kernels",
                "my_kernel",
                "--timeout",
                "5",
                "--",
                "python",
                "test.py",
                "--foo",
            ],
            prog="cutracer oracle cmd",
        )
        fake.main_cmd.side_effect = RuntimeError("x")
        with mock.patch("cutracer.oracle.cli.pytest_oracle", loaded):
            invoked = CliRunner().invoke(oracle_command, argv)
        self.assertEqual(invoked.exit_code, INCOMPLETE)

    def test_usage_errors_exit_2(self) -> None:
        for argv in (
            ["cmd"],
            ["cmd", "--timeout", "xyz", "--", "true"],
            ["cmd", "--fail-codes", "x", "--", "true"],
            ["cmd", "--fail-codes", "", "--", "true"],
        ):
            with self.subTest(argv=argv):
                invoked = CliRunner().invoke(oracle_command, argv)
                self.assertEqual(invoked.exit_code, 2)

    def test_the_exported_script_runs_cmd_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "bin" / "oracle.py"
            invoked = CliRunner().invoke(oracle_command, ["export", str(script)])
            self.assertEqual(invoked.exit_code, 0, invoked.output)
            env = {k: v for k, v in os.environ.items() if k != "CUDA_INJECTION64_PATH"}
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "cmd",
                    "--fail-codes",
                    "3",
                    "--plain",
                    "--",
                    sys.executable,
                    "-c",
                    "import sys; sys.exit(3)",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(proc.returncode, REPRODUCED, proc.stderr)
