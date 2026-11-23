# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the numerics capture (`cutracer_numerics.py`).

The capture is a standalone script shipped as a resource. Each test loads a
fresh module from it, as `cutracer oracle --numerics` installs it in the
process under test, and restores every function it hooked afterwards. Every
tensor is a CPU tensor.
"""

import atexit
import contextlib
import importlib.util
import inspect
import io
import itertools
import json
import os
import sys
import tempfile
import textwrap
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast, Optional
from unittest import mock

import hypothesis
import numpy
import pytest
import torch
import triton  # @manual=//triton:triton
from cutracer.oracle.cli import numerics_capture_path, pytest_oracle
from triton.runtime.autotuner import Autotuner  # @manual=//triton:triton

_NAMES = itertools.count()
_SCRUBBED = ("PYTEST_CURRENT_TEST", "RANK", "LOCAL_RANK")


class _CaptureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.script = stack.enter_context(numerics_capture_path())
        self.tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.golden = self.tmp / "golden"
        self.out = self.tmp / "out"
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CUTRACER_NUMERICS_") and key not in _SCRUBBED
        }
        stack.enter_context(mock.patch.dict(os.environ, env, clear=True))

    def load(self, **env: str) -> ModuleType:
        """A fresh capture module; it installs itself when `env` names a mode."""
        os.environ.update(env)
        name = f"cutracer_numerics_under_test_{next(_NAMES)}"
        spec = importlib.util.spec_from_file_location(name, self.script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        try:
            spec.loader.exec_module(module)
        finally:
            self.addCleanup(module.uninstall)
        return module

    def run_mode(
        self,
        mode: str,
        body: Callable[[], Any],
        *,
        out: Optional[Path] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ModuleType:
        variables = {
            "CUTRACER_NUMERICS_MODE": mode,
            "CUTRACER_NUMERICS_GOLDEN": str(self.golden),
            **(env or {}),
        }
        if out is not None:
            variables["CUTRACER_NUMERICS_OUT"] = str(out)
        module = self.load(**variables)
        try:
            body()
        finally:
            module.finish()
            module.uninstall()
            for key in variables:
                os.environ.pop(key, None)
        return module

    def summary(self, out: Optional[Path] = None, proc: str = "main") -> dict:
        path = (out or self.out) / f"numerics-{proc}.json"
        return json.loads(path.read_text())["numerics"]

    def record_then_check(
        self,
        recorded: Callable[[], Any],
        checked: Callable[[], Any],
        env: Optional[dict[str, str]] = None,
    ) -> dict:
        self.run_mode("record", recorded, env=env)
        self.run_mode("check", checked, out=self.out, env=env)
        return self.summary()


class ClassifyTest(_CaptureTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.capture = self.load()
        torch.manual_seed(0)

    def verdict(self, cur: Any, gold: Any) -> tuple[str, dict]:
        metrics = self.capture.diff_metrics(cur, gold)
        return self.capture.classify(metrics), metrics

    def test_identical_tensors_are_same(self) -> None:
        x = torch.randn(8, 8)
        self.assertEqual(self.verdict(x.clone(), x)[0], "same")

    def test_one_bf16_ulp_is_small(self) -> None:
        x = torch.randn(64, 64).to(torch.bfloat16)
        y = x.clone()
        y.view(torch.int16)[5, 7] += 1
        verdict, metrics = self.verdict(y, x)
        self.assertEqual(verdict, "small")
        self.assertEqual((metrics["n_diff"], metrics["max_ulp"]), (1, 1))

    def test_fp32_holding_bf16_values_counts_bf16_ulps(self) -> None:
        x = torch.randn(32, 32).to(torch.bfloat16).float()
        y = x.to(torch.bfloat16)
        y.view(torch.int16)[1, 2] += 1
        verdict, metrics = self.verdict(y.float(), x)
        self.assertEqual(verdict, "small")
        self.assertEqual((metrics["ulp_unit"], metrics["max_ulp"]), ("bfloat16", 1))

    def test_a_change_of_the_whole_tensor_is_large(self) -> None:
        x = torch.randn(64, 64)
        verdict, metrics = self.verdict(x * 1.05, x)
        self.assertEqual(verdict, "large")
        self.assertAlmostEqual(metrics["rel_l2"], 0.05, places=5)

    def test_one_changed_row_is_large(self) -> None:
        x = torch.randn(512, 256)
        y = x.clone()
        y[7] *= 1.2
        verdict, metrics = self.verdict(y, x)
        self.assertEqual(verdict, "large")
        self.assertLess(metrics["rel_l2"], 0.02)
        self.assertEqual((metrics["n_bad_rows"], metrics["bad_rows_head"]), (1, [7]))
        self.assertEqual(metrics["n_outliers"], 0)

    def test_one_wrong_element_in_a_wide_row_is_large(self) -> None:
        x = torch.randn(64, 4096)
        x[3, 100] = 0.5
        y = x.clone()
        y[3, 100] += 2.0
        verdict, metrics = self.verdict(y, x)
        self.assertEqual(verdict, "large")
        self.assertEqual(metrics["n_bad_rows"], 0)
        self.assertEqual(
            (metrics["n_outliers"], metrics["outliers_head"]), (1, [3 * 4096 + 100])
        )

    def test_a_change_of_non_finite_class_is_large(self) -> None:
        x = torch.randn(4, 4)
        x[0, 0], x[1, 1], x[2, 2] = float("inf"), float("nan"), float("-inf")
        for position, value in (
            ((0, 0), float("-inf")),
            ((1, 1), float("inf")),
            ((3, 3), float("nan")),
        ):
            y = x.clone()
            y[position] = value
            verdict, metrics = self.verdict(y, x)
            self.assertEqual(verdict, "large", position)
            self.assertEqual(metrics["nonfinite_changed"], 1, position)

    def test_integer_and_bool_changes_are_large(self) -> None:
        ints = torch.arange(10)
        changed = ints.clone()
        changed[4] = 0
        self.assertEqual(self.verdict(changed, ints)[0], "large")
        flags = torch.zeros(5, dtype=torch.bool)
        self.assertEqual(self.verdict(~flags, flags)[0], "large")

    def test_an_integer_change_already_seen_in_plain_runs_is_small(self) -> None:
        ints = torch.arange(10)
        changed = ints.clone()
        changed[4] = 0
        metrics = self.capture.diff_metrics(changed, ints)
        self.assertEqual(
            self.capture.classify(metrics, {"runs": 2, "changed": 1}), "small"
        )
        self.assertEqual(
            self.capture.classify(metrics, {"runs": 2, "changed": 0}), "large"
        )
        # Only a large change is located.
        seen = {"runs": 2, "changed": 1}
        self.assertNotIn("geometry", self.capture.diff_metrics(changed, ints, seen))
        self.assertIn("geometry", metrics)
        # A complex tensor that changed between plain runs is still large.
        z = torch.randn(4, dtype=torch.complex64)
        moved = z.clone()
        moved[0] += 1
        complex_metrics = self.capture.diff_metrics(moved, z, seen)
        self.assertEqual(self.capture.classify(complex_metrics, seen), "large")

    def test_empty_tensors_are_same(self) -> None:
        self.assertEqual(self.verdict(torch.empty(0, 3), torch.empty(0, 3))[0], "same")

    def test_geometry_bounds_the_changed_block(self) -> None:
        x = torch.randn(8, 16)
        y = x.clone()
        y[2:4, 5:9] += 10.0
        _, metrics = self.verdict(y, x)
        self.assertEqual(
            [(g["dim"], g["count"], g["min"], g["max"]) for g in metrics["geometry"]],
            [(0, 2, 2, 3), (1, 4, 5, 8)],
        )
        # A small change is not located: that needs a whole-tensor mask.
        z = x.clone()
        z[2, 5] = torch.nextafter(z[2, 5], torch.tensor(float("inf")))
        verdict, metrics = self.verdict(z, x)
        self.assertEqual(verdict, "small")
        self.assertNotIn("geometry", metrics)

    def test_a_float_format_without_a_value_view_is_compared_by_bytes(self) -> None:
        if not hasattr(torch, "float4_e2m1fn_x2"):
            self.skipTest("this torch has no float4_e2m1fn_x2")
        raw = torch.arange(8, dtype=torch.uint8)
        changed = raw.clone()
        changed[2] ^= 1
        metrics = self.capture.diff_metrics(
            changed.view(torch.float4_e2m1fn_x2), raw.view(torch.float4_e2m1fn_x2)
        )
        self.assertEqual((metrics["kind"], metrics["n_diff"]), ("opaque", 1))
        self.assertEqual(self.capture.classify(metrics), "large")

    def test_float64_values_near_the_limit_do_not_overflow(self) -> None:
        x = 1e200 * torch.randn(64, 64, dtype=torch.float64)
        verdict, metrics = self.verdict(2 * x, x)
        self.assertEqual(verdict, "large")
        self.assertAlmostEqual(metrics["rel_l2"], 1.0, places=6)
        verdict, metrics = self.verdict(-x, x)
        self.assertEqual(verdict, "large")
        self.assertLess(metrics["max_ulp"], 2**63)

    def test_signed_zero_and_nan_payload_changes_are_small(self) -> None:
        x = torch.zeros(4, 4)
        self.assertEqual(self.verdict(-x, x)[0], "small")
        x = torch.randn(4, 4)
        x[1, 1] = float("nan")
        y = x.clone()
        y.view(torch.int32)[1, 1] ^= 1
        verdict, metrics = self.verdict(y, x)
        self.assertEqual((verdict, metrics["nonfinite_changed"]), ("small", 0))

    def test_float8_complex_and_zero_dim_values(self) -> None:
        x8 = torch.randn(16, 16).to(torch.float8_e4m3fn)
        y8 = x8.clone()
        y8.view(torch.uint8)[2, 3] += 1
        self.assertEqual(self.verdict(y8, x8)[0], "small")
        z = torch.randn(8, dtype=torch.complex64)
        self.assertEqual(self.verdict(z * (1 + 1e-7), z)[0], "large")
        # A signed zero, which `!=` does not see.
        signed = torch.zeros(4, dtype=torch.complex64)
        flipped = signed.clone()
        torch.view_as_real(flipped)[1, 0] = -0.0
        verdict, metrics = self.verdict(flipped, signed)
        self.assertEqual((verdict, metrics["n_diff"]), ("large", 1))
        one = torch.tensor(1.0)
        nudged = one.clone()
        nudged.view(torch.int32).add_(1)
        self.assertEqual(self.verdict(nudged, one)[0], "small")
        self.assertEqual(self.verdict(one * 2, one)[0], "large")

    def test_any_change_of_an_all_zero_golden_is_large(self) -> None:
        zeros = torch.zeros(8, 8)
        verdict, metrics = self.verdict(zeros + 1e-30, zeros)
        self.assertEqual((verdict, metrics["rel_l2"]), ("large", float("inf")))

    def test_reduction_order_noise_stays_small(self) -> None:
        a = torch.randn(128, 1024)
        b = torch.randn(1024, 256)
        parts = [a[:, k : k + 128] @ b[k : k + 128] for k in range(0, 1024, 128)]
        forward = sum(parts, torch.zeros(128, 256))
        backward = sum(reversed(parts), torch.zeros(128, 256))
        for dtype in (torch.float16, torch.bfloat16):
            verdict, metrics = self.verdict(backward.to(dtype), forward.to(dtype))
            self.assertEqual(verdict, "small", (dtype, metrics))
        terms = torch.randn(16, 64, 2)
        shuffled = terms[torch.randperm(16)]
        self.assertEqual(self.verdict(shuffled.sum(0), terms.sum(0))[0], "small")

    def test_a_chunked_comparison_matches_a_single_pass(self) -> None:
        keys = (
            "n_diff",
            "n_bad_rows",
            "bad_rows_head",
            "n_outliers",
            "outliers_head",
            "max_ulp",
        )
        x = torch.randn(256, 96)
        y = x + 1e-6 * torch.randn(256, 96)
        y[200] *= 1.3
        y[17, 5] += 50.0
        flat_x, flat_y = x.flatten(), y.flatten()
        # 200 elements per block: whole rows of the 2-D pair, and segments of
        # the single row of the 1-D pair.
        for cur, gold in ((y, x), (flat_y, flat_x)):
            whole = self.capture.diff_metrics(cur, gold)
            with mock.patch.object(self.capture, "_CHUNK_ELEMENTS", 200):
                chunked = self.capture.diff_metrics(cur, gold)
            for key in keys:
                self.assertEqual(chunked.get(key), whole.get(key), (cur.dim(), key))
            self.assertAlmostEqual(chunked["rel_l2"], whole["rel_l2"], places=9)
            self.assertAlmostEqual(
                chunked["max_row_rel_l2"], whole["max_row_rel_l2"], places=9
            )
        self.assertEqual(self.capture.diff_metrics(y, x)["bad_rows_head"], [17, 200])

    def test_a_golden_whose_norm_overflows_is_still_compared(self) -> None:
        x = torch.full((1000,), 1e307, dtype=torch.float64)
        y = x.clone()
        y[5] *= 1.5
        verdict, metrics = self.verdict(y, x)
        self.assertEqual(verdict, "large")
        self.assertEqual(metrics["outliers_head"], [5])

    def test_thresholds_can_be_raised(self) -> None:
        x = torch.randn(64, 64)
        self.assertEqual(self.verdict(x * 1.03, x)[0], "large")
        with mock.patch.dict(
            os.environ,
            {
                "CUTRACER_NUMERICS_LARGE_REL_L2": "0.05",
                "CUTRACER_NUMERICS_LARGE_ROW_REL_L2": "0.05",
            },
        ):
            self.assertEqual(self.verdict(x * 1.03, x)[0], "small")


def _compare_pair(a: Any, b: Any) -> None:
    torch.testing.assert_close(a, b, atol=1e9, rtol=0)


def _compare_close(a: Any, b: Any) -> None:
    torch.allclose(a, b, atol=1e9)


class RoundTripTest(_CaptureTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)
        self.ref = torch.randn(32, 32)
        self.a = self.ref + 1e-3 * torch.randn(32, 32)
        self.b = torch.arange(6)

    def body(self, a: Any) -> Callable[[], None]:
        def run() -> None:
            _compare_pair(a, self.ref)
            _compare_close(self.b, self.b.clone())

        return run

    def test_a_golden_holds_each_distinct_tensor_once(self) -> None:
        self.run_mode("record", self.body(self.a), out=self.out)
        calls = json.loads((self.golden / "calls-main.json").read_text())["calls"]
        self.assertEqual(
            [c["fn"] for c in calls],
            ["torch.testing.assert_close", "torch.allclose"],
        )
        leaves = [leaf for call in calls for leaf in call["leaves"]]
        self.assertTrue(all(leaf["saved"] for leaf in leaves))
        blobs = sorted(p.name for p in (self.golden / "blobs").glob("*/*"))
        # `self.b` and its clone have the same bytes, so they share a blob.
        self.assertEqual(blobs, sorted({leaf["sha256"] for leaf in leaves}))
        self.assertEqual(len(blobs), 3)
        summary = self.summary()
        self.assertEqual((summary["verdict"], summary["reused"]), ("recorded", 1))

    def test_a_second_process_reuses_the_blobs(self) -> None:
        self.run_mode("record", self.body(self.a))
        before = sorted((self.golden / "blobs").glob("*/*"))
        self.run_mode("record", self.body(self.a), out=self.out, env={"RANK": "1"})
        summary = self.summary(proc="rank-1")
        self.assertEqual((summary["reused"], summary["stored_bytes"]), (4, 0))
        self.assertEqual(sorted((self.golden / "blobs").glob("*/*")), before)

    def test_the_budget_counts_the_bytes_written(self) -> None:
        # Four 256 KiB tensors and a 0.4 MiB budget: the zeros and the ones
        # compress to almost nothing, the first noise tensor still fits by its
        # size in memory, the second does not.
        noise = [torch.randint(0, 256, (1 << 18,), dtype=torch.uint8) for _ in "ab"]

        def body() -> None:
            for value in (torch.zeros(1 << 16), torch.ones(1 << 16), *noise):
                torch.equal(value, value)

        budget = {"CUTRACER_NUMERICS_MAX_TOTAL_MB": "0.4"}
        self.run_mode("record", body, out=self.out, env=budget)
        calls = json.loads((self.out / "calls-main.json").read_text())["calls"]
        self.assertEqual(
            [call["leaves"][0]["saved"] for call in calls], [True, True, True, False]
        )
        summary = self.summary()
        self.assertEqual(summary["saved_bytes"], 3 << 18)
        self.assertLess(summary["stored_bytes"], 0.4 * 2**20)

    def test_the_same_values_are_same(self) -> None:
        summary = self.record_then_check(self.body(self.a), self.body(self.a.clone()))
        self.assertEqual(summary["verdict"], "same")
        self.assertEqual((summary["calls"], summary["golden_calls"]), (2, 2))
        self.assertEqual((summary["structure"], summary["missing_calls"]), ([], 0))

    def test_a_one_ulp_change_is_small_and_reports_its_error_ratio(self) -> None:
        nudged = self.a.clone()
        nudged.view(torch.int32)[0, 0] += 1
        summary = self.record_then_check(self.body(self.a), self.body(nudged))
        self.assertEqual(summary["verdict"], "small")
        change = summary["changed"][0]
        self.assertEqual((change["leaf"], change["max_ulp"]), ("a0", 1))
        self.assertAlmostEqual(change["err_ratio"], 1.0, places=3)

    def test_corruption_is_large(self) -> None:
        broken = self.a.clone()
        broken[3] = 0.0
        summary = self.record_then_check(self.body(self.a), self.body(broken))
        self.assertEqual(summary["verdict"], "large")
        self.assertEqual(summary["changed"][0]["leaf"], "a0")

    def test_an_extra_call_is_a_structure_problem(self) -> None:
        def more() -> None:
            self.body(self.a)()
            _compare_pair(self.a, self.ref)

        summary = self.record_then_check(self.body(self.a), more)
        self.assertEqual(len(summary["structure"]), 1)
        self.assertIn("extra call", summary["structure"][0])
        self.assertEqual(
            (summary["first_structure_k"], summary["first_large_k"]), (2, None)
        )

    def test_a_missing_call_is_counted(self) -> None:
        summary = self.record_then_check(
            self.body(self.a), lambda: _compare_pair(self.a, self.ref)
        )
        self.assertEqual((summary["missing_calls"], summary["structure"]), (1, []))

    def test_calls_are_matched_by_site_not_by_order(self) -> None:
        def reordered() -> None:
            _compare_close(self.b, self.b.clone())
            _compare_pair(self.a, self.ref)

        summary = self.record_then_check(self.body(self.a), reordered)
        self.assertEqual((summary["verdict"], summary["structure"]), ("same", []))

    def test_a_comparison_made_inside_torch_is_not_recorded(self) -> None:
        namespace: dict[str, Any] = {
            "__name__": "torch._cutracer_fake_internal",
            "torch": torch,
        }
        source = "def check(a, b):\n    return torch.equal(a, b)\n"
        exec(
            compile(source, "/fake/torch/_cutracer_fake_internal.py", "exec"), namespace
        )

        def body() -> None:
            namespace["check"](self.a, self.a)
            _compare_pair(self.a, self.ref)

        self.run_mode("record", body, out=self.out)
        self.assertEqual(self.summary()["calls"], 1)

    def test_numbers_and_lists_are_compared(self) -> None:
        def body(value: float) -> Callable[[], None]:
            return lambda: torch.testing.assert_close(
                [value, 2.0], [value, 2.0], atol=10.0, rtol=0
            )

        summary = self.record_then_check(body(1.0), body(1.5))
        self.assertEqual(summary["verdict"], "large")
        self.assertEqual(summary["changed"][0]["dtype"], "float64")

    def test_a_tensor_too_big_to_save_is_unmeasured(self) -> None:
        tiny = {"CUTRACER_NUMERICS_MAX_MB": "0.0001"}
        broken = self.a.clone()
        broken[0, 0] += 1.0
        summary = self.record_then_check(self.body(self.a), self.body(broken), tiny)
        self.assertEqual(summary["verdict"], "unmeasured")

    def test_record_refuses_a_golden_that_already_holds_the_process(self) -> None:
        self.run_mode("record", self.body(self.a))
        self.run_mode("record", self.body(self.a), out=self.out)
        summary = self.summary()
        self.assertEqual(summary["verdict"], "refused")
        self.assertIn("already holds calls-main.json", summary["errors"][0])

    def test_a_different_torch_than_the_golden_is_an_environment_problem(self) -> None:
        self.run_mode("record", self.body(self.a))
        calls_path = self.golden / "calls-main.json"
        calls = json.loads(calls_path.read_text())
        calls["env"]["torch"] = "0.0.0"
        calls_path.write_text(json.dumps(calls))
        self.run_mode("check", self.body(self.a), out=self.out)
        summary = self.summary()
        self.assertIn("torch", summary["environment"][0])
        self.assertEqual(summary["structure"], [])

    def test_a_change_that_cannot_be_compared_is_unmeasured(self) -> None:
        self.run_mode("record", self.body(self.a))
        calls = json.loads((self.golden / "calls-main.json").read_text())["calls"]
        sha256 = calls[0]["leaves"][0]["sha256"]
        (self.golden / "blobs" / sha256[:2] / sha256).unlink()
        broken = self.a + 100.0
        self.run_mode("check", self.body(broken), out=self.out)
        summary = self.summary()
        self.assertEqual(summary["verdict"], "unmeasured")
        self.assertIn("FileNotFoundError", " ".join(summary["errors"]))

    def test_a_golden_tensor_the_run_does_not_compare_is_unmeasured(self) -> None:
        def pair(*values: Any) -> Callable[[], None]:
            return lambda: torch.testing.assert_close(values, values)

        summary = self.record_then_check(pair(self.a, self.ref), pair(self.a))
        self.assertEqual(summary["verdict"], "unmeasured")
        self.assertEqual(summary["leaves"]["unmeasured"], 2)

    def test_calls_from_two_threads_at_one_site_cannot_be_matched(self) -> None:
        def threaded() -> None:
            workers = [
                threading.Thread(target=_compare_pair, args=(t, t))
                for t in (self.a, self.ref)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

        summary = self.record_then_check(threaded, threaded)
        self.assertIn("2 threads", summary["structure"][0])
        self.assertEqual(summary["first_structure_k"], 0)

    def test_a_malformed_golden_is_an_error_not_a_crash(self) -> None:
        self.run_mode("record", self.body(self.a))
        calls_path = self.golden / "calls-main.json"
        calls = json.loads(calls_path.read_text())
        del calls["calls"][0]["nodeid"]
        calls_path.write_text(json.dumps(calls))
        self.run_mode("check", self.body(self.a), out=self.out)
        self.assertIn("no usable golden", self.summary()["errors"][0])

    def test_a_write_failure_does_not_reach_the_caller(self) -> None:
        self.out.write_text("not a directory")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.run_mode("record", self.body(self.a), out=self.out)
        self.assertIn("could not write", stderr.getvalue())

    def test_a_second_writer_leaves_a_conflict_file(self) -> None:
        self.out.mkdir()
        (self.out / "numerics-main.json").write_text("{}")
        self.run_mode("record", self.body(self.a), out=self.out)
        names = sorted(p.name for p in self.out.iterdir())
        self.assertEqual(
            names, [f"conflict-main-{os.getpid()}.json", "numerics-main.json"]
        )

    def test_check_needs_an_output_directory(self) -> None:
        before = torch.testing.assert_close
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            capture = self.load(
                CUTRACER_NUMERICS_MODE="check",
                CUTRACER_NUMERICS_GOLDEN=str(self.golden),
            )
            capture.record_tensor("x", self.a)
            capture.record_tensor("y", self.a)
        self.assertIsNone(capture._RECORDER)
        self.assertIs(torch.testing.assert_close, before)
        self.assertEqual(stderr.getvalue().count("is not set"), 1)

    def test_a_driver_run_as_main_records_its_calls(self) -> None:
        namespace: dict[str, Any] = {
            "__name__": "__main__",
            "torch": torch,
            "a": self.a,
        }
        code = compile("torch.testing.assert_close(a, a)\n", "<string>", "exec")
        self.run_mode("record", lambda: exec(code, namespace), out=self.out)
        calls = json.loads((self.out / "calls-main.json").read_text())["calls"]
        self.assertEqual([c["site"] for c in calls], ["<string>:1"])

    def test_call_sites_are_relative_to_the_root(self) -> None:
        namespace: dict[str, Any] = {"__name__": "driver", "torch": torch, "a": self.a}
        code = compile(
            "torch.equal(a, a)\n", str(self.tmp / "sub" / "driver.py"), "exec"
        )
        self.run_mode(
            "record",
            lambda: exec(code, namespace),
            out=self.out,
            env={"CUTRACER_NUMERICS_ROOT": str(self.tmp)},
        )
        calls = json.loads((self.out / "calls-main.json").read_text())["calls"]
        self.assertEqual(
            [c["site"] for c in calls], [os.path.join("sub", "driver.py") + ":1"]
        )

    def test_numpy_operands_passed_by_keyword_are_recorded(self) -> None:
        first, second = list(
            inspect.signature(numpy.testing.assert_allclose).parameters
        )[:2]
        values = numpy.arange(4.0)

        def body() -> None:
            numpy.testing.assert_allclose(**{first: values, second: values})

        self.run_mode("record", body, out=self.out)
        calls = json.loads((self.out / "calls-main.json").read_text())["calls"]
        self.assertEqual([leaf["name"] for leaf in calls[0]["leaves"]], ["a0", "b0"])

    def test_torchscript_still_resolves_a_hooked_builtin(self) -> None:
        capture = self.load(
            CUTRACER_NUMERICS_MODE="record", CUTRACER_NUMERICS_GOLDEN=str(self.golden)
        )

        def same(x: torch.Tensor, y: torch.Tensor) -> bool:
            return torch.equal(x, y)

        self.assertTrue(torch.jit.script(same)(self.a, self.a))
        wrapper = torch.equal
        capture.uninstall()
        from torch.jit import _builtins

        self.assertIsNone(_builtins._find_builtin(wrapper))

    def test_a_forked_child_does_not_write_the_parent_record(self) -> None:
        def body() -> None:
            child = os.fork()
            if child == 0:
                # Little torch work in the child: its thread pools were
                # forked from a parent that used them.
                try:
                    torch.equal(self.b, self.b)
                finally:
                    os._exit(0)
            os.waitpid(child, 0)
            self.notes.append(child)
            _compare_pair(self.a, self.ref)

        self.notes: list[int] = []
        self.run_mode("record", body, out=self.out)
        names = sorted(p.name for p in self.out.iterdir())
        self.assertEqual(
            names,
            [
                "calls-main.json",
                "numerics-main.json",
                f"unlabeled-{self.notes[0]}.json",
            ],
        )
        self.assertEqual(self.summary()["calls"], 1)

    def test_hypothesis_is_pinned_and_restored(self) -> None:
        capture = self.load(
            CUTRACER_NUMERICS_MODE="record",
            CUTRACER_NUMERICS_GOLDEN=str(self.golden),
            CUTRACER_NUMERICS_PIN_HYPOTHESIS="1",
        )
        self.assertTrue(hypothesis.settings().derandomize)
        self.assertIsNone(hypothesis.settings().deadline)
        capture.uninstall()
        self.assertFalse(hypothesis.settings().derandomize)

    def test_a_capture_error_does_not_reach_the_test(self) -> None:
        capture = self.load(
            CUTRACER_NUMERICS_MODE="record",
            CUTRACER_NUMERICS_GOLDEN=str(self.golden),
            CUTRACER_NUMERICS_OUT=str(self.out),
        )
        with mock.patch.object(capture, "_to_cpu", side_effect=RuntimeError("boom")):
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(torch.zeros(2), torch.ones(2))
        capture.finish()
        calls = json.loads((self.out / "calls-main.json").read_text())["calls"]
        self.assertTrue(
            all(leaf["kind"].startswith("unsupported") for leaf in calls[0]["leaves"])
        )
        # A golden that could not copy a tensor can never be checked.
        self.assertIn("boom", self.summary()["errors"][0])

    def test_finish_writes_once(self) -> None:
        capture = self.load(
            CUTRACER_NUMERICS_MODE="record",
            CUTRACER_NUMERICS_GOLDEN=str(self.golden),
            CUTRACER_NUMERICS_OUT=str(self.out),
        )
        capture.record_tensor("x", self.a)
        capture.finish()
        capture.finish()
        self.assertEqual(
            sorted(p.name for p in self.out.iterdir()),
            ["calls-main.json", "numerics-main.json"],
        )
        self.assertEqual(self.summary()["calls"], 1)
        mode = (self.out / "numerics-main.json").stat().st_mode & 0o777
        self.assertEqual(mode, 0o644)

    def test_wrapped_entries_record_their_outermost_results(self) -> None:
        name = f"cutracer_numerics_wrapped_{next(_NAMES)}"
        (self.tmp / f"{name}.py").write_text(
            textwrap.dedent(
                """
                def inner(x):
                    return x + 1

                def kernel(x):
                    return inner(x) * 2
                """
            )
        )
        sys.path.insert(0, str(self.tmp))
        self.addCleanup(sys.path.remove, str(self.tmp))
        __import__(name)
        self.addCleanup(sys.modules.pop, name, None)

        def body() -> None:
            sys.modules[name].kernel(self.a)

        self.run_mode(
            "record", body, out=self.out, env={"CUTRACER_NUMERICS_WRAP": f"{name}:*"}
        )
        calls = json.loads((self.out / "calls-main.json").read_text())["calls"]
        self.assertEqual([c["fn"] for c in calls], [f"result:{name}.kernel"])

    def test_a_descendant_without_a_label_leaves_a_note(self) -> None:
        def body() -> None:
            _compare_pair(self.a, self.ref)

        self.run_mode(
            "record", body, out=self.out, env={"CUTRACER_NUMERICS_ROOT_PID": "1"}
        )
        self.assertEqual(
            [p.name for p in self.out.iterdir()], [f"unlabeled-{os.getpid()}.json"]
        )

    def test_a_rank_labels_its_process(self) -> None:
        self.run_mode("record", self.body(self.a), out=self.out, env={"RANK": "3"})
        self.assertTrue((self.golden / "calls-rank-3.json").is_file())
        self.assertEqual(self.summary(proc="rank-3")["verdict"], "recorded")

    def test_uninstall_restores_the_hooked_functions(self) -> None:
        name = f"cutracer_numerics_wrapped_{next(_NAMES)}"
        (self.tmp / f"{name}.py").write_text("def kernel(x):\n    return x\n")
        sys.path.insert(0, str(self.tmp))
        self.addCleanup(sys.path.remove, str(self.tmp))
        wrapped = __import__(name)
        self.addCleanup(sys.modules.pop, name, None)

        def snapshot() -> tuple:
            return (
                torch.testing.assert_close,
                torch.allclose,
                torch.equal,
                "allclose" in vars(torch.Tensor),
                "equal" in vars(torch.Tensor),
                numpy.testing.assert_allclose,
                numpy.testing.assert_array_equal,
                wrapped.kernel,
                atexit._ncallbacks(),
            )

        before = snapshot()
        capture = self.load(
            CUTRACER_NUMERICS_MODE="record",
            CUTRACER_NUMERICS_GOLDEN=str(self.golden),
            CUTRACER_NUMERICS_WRAP=f"{name}:kernel",
        )
        self.assertIsNot(torch.testing.assert_close, before[0])
        self.assertIsNot(wrapped.kernel, before[7])
        capture.uninstall()
        self.assertEqual(snapshot(), before)


class BlobTest(_CaptureTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.capture = self.load()
        torch.manual_seed(0)

    def store(self, value: torch.Tensor) -> tuple[dict, Path]:
        leaf = {
            "sha256": self.capture._sha256(value),
            "nbytes": value.numel() * value.element_size(),
            "dtype": self.capture._dtype_name(value.dtype),
            "shape": list(value.shape),
        }
        path = self.capture._blob_path(self.tmp, leaf["sha256"])
        self.assertEqual(self.capture._write_blob(path, value), path.stat().st_size)
        return leaf, path

    def codec(self, path: Path) -> str:
        header = self.capture._BLOB_HEADER
        fields = header.unpack(path.read_bytes()[: header.size])
        return self.capture._BLOB_CODECS[fields[1]]

    def test_every_dtype_round_trips_bit_for_bit_across_chunks(self) -> None:
        values = [
            torch.randn(300).to(torch.bfloat16),
            torch.randn(2048),
            torch.zeros(4096),
            torch.arange(4096, dtype=torch.int32),
            torch.randn(64, dtype=torch.float64),
            torch.randn(256).to(torch.float8_e4m3fn),
            torch.rand(100) > 0.5,
            torch.randn(40, dtype=torch.complex64),
            torch.tensor(3.5),
            torch.empty(0, 3),
        ]
        # 1 KiB chunks: the larger values span several, compressed in parallel
        # when a sample compresses and stored as they are otherwise.
        with mock.patch.object(self.capture, "_BLOB_CHUNK", 1024):
            with mock.patch.object(self.capture, "_BLOB_SAMPLE", 256):
                for value in values:
                    with self.subTest(dtype=value.dtype, shape=tuple(value.shape)):
                        leaf, _ = self.store(value)
                        back = self.capture._read_blob(self.tmp, leaf)
                        self.assertEqual(
                            (back.dtype, back.shape), (value.dtype, value.shape)
                        )
                        self.assertEqual(self.capture._sha256(back), leaf["sha256"])

    def test_only_bytes_that_compress_are_compressed(self) -> None:
        zeros = torch.zeros(1 << 16)
        noise = torch.randint(0, 256, (1 << 18,), dtype=torch.uint8)
        _, packed = self.store(zeros)
        _, raw = self.store(noise)
        self.assertNotEqual(self.codec(packed), "raw")
        self.assertLess(packed.stat().st_size, zeros.numel() * 4 // 100)
        self.assertEqual(self.codec(raw), "raw")
        self.assertEqual(
            raw.stat().st_size, noise.numel() + self.capture._BLOB_HEADER.size + 8
        )

    def test_blobs_use_zstd_when_the_interpreter_has_it_and_zlib_otherwise(
        self,
    ) -> None:
        value = torch.zeros(1 << 12)
        _, path = self.store(value)
        self.assertEqual(self.codec(path), "zstd" if self.capture._zstd() else "zlib")
        with mock.patch.object(self.capture, "_zstd", return_value=None):
            leaf, path = self.store(value)
            back = self.capture._read_blob(self.tmp, leaf)
        self.assertEqual(self.codec(path), "zlib")
        self.assertEqual(self.capture._sha256(back), leaf["sha256"])

    def test_a_blob_that_does_not_match_its_hash_is_refused(self) -> None:
        leaf, path = self.store(torch.randint(0, 256, (4096,), dtype=torch.uint8))
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "does not match its sha256"):
            self.capture._read_blob(self.tmp, leaf)


_PLUGIN_TEST = """
import os
from pathlib import Path

import torch


def test_values():
    if os.environ.get("NUMERICS_TEST_SEED_OUT"):
        Path(os.environ["NUMERICS_TEST_SEED_OUT"]).write_text(str(torch.initial_seed()))
    x = torch.arange(16, dtype=torch.float32)
    if os.environ.get("NUMERICS_TEST_PERTURB"):
        x[3] += 100.0
    torch.testing.assert_close(x, torch.arange(16, dtype=torch.float32), atol=1e3, rtol=0)
"""


class PytestPluginTest(_CaptureTestCase):
    def run_pytest(
        self, mode: str, out: Path, configure: bool = False, **env: str
    ) -> int:
        """Run the plugin test file with the capture as a plugin; with
        `configure`, the module is imported inert and pytest_configure
        installs it."""
        test_file = self.tmp / "test_numerics_plugin.py"
        test_file.write_text(_PLUGIN_TEST)
        variables = {
            "CUTRACER_NUMERICS_MODE": mode,
            "CUTRACER_NUMERICS_GOLDEN": str(self.golden),
            "CUTRACER_NUMERICS_OUT": str(out),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            **env,
        }
        if configure:
            capture = self.load()
            self.assertIsNone(capture._RECORDER)
            os.environ.update(variables)
        else:
            capture = self.load(**variables)
        try:
            return int(
                pytest.main(
                    [
                        str(test_file),
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        f"--rootdir={self.tmp}",
                    ],
                    plugins=[capture],
                )
            )
        finally:
            # A fresh import of the test file in the next run; torch's own
            # modules stay loaded (they register operators once per process).
            sys.modules.pop(test_file.stem, None)
            capture.uninstall()
            for key in (
                "CUTRACER_NUMERICS_MODE",
                "NUMERICS_TEST_PERTURB",
                "NUMERICS_TEST_SEED_OUT",
            ):
                os.environ.pop(key, None)

    def test_a_pytest_run_records_and_checks_the_test_comparisons(self) -> None:
        self.assertEqual(self.run_pytest("record", self.tmp / "record"), 0)
        golden_calls = json.loads((self.golden / "calls-main.json").read_text())[
            "calls"
        ]
        self.assertEqual(
            [c["nodeid"] for c in golden_calls],
            ["test_numerics_plugin.py::test_values"],
        )
        self.assertEqual(self.run_pytest("check", self.tmp / "same"), 0)
        self.assertEqual(self.summary(self.tmp / "same")["verdict"], "same")
        self.assertEqual(
            self.run_pytest("check", self.tmp / "large", NUMERICS_TEST_PERTURB="1"), 0
        )
        self.assertEqual(self.summary(self.tmp / "large")["verdict"], "large")

    def test_pytest_configure_installs_the_capture_and_tests_are_seeded(self) -> None:
        torch.manual_seed(1234)
        seed_file = self.tmp / "seed.txt"
        code = self.run_pytest(
            "record",
            self.tmp / "record",
            configure=True,
            NUMERICS_TEST_SEED_OUT=str(seed_file),
        )
        self.assertEqual(code, 0)
        self.assertEqual(seed_file.read_text(), "0")
        self.assertEqual(self.summary(self.tmp / "record")["calls"], 1)


class OracleWorkerTest(_CaptureTestCase):
    """The oracle's pytest worker loads the capture by path under real pytest
    and torch; the oracle then judges the files the capture wrote, so the two
    scripts agree on the format."""

    def setUp(self) -> None:
        super().setUp()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.oracle = stack.enter_context(pytest_oracle())
        (self.tmp / "test_numerics_plugin.py").write_text(_PLUGIN_TEST)

    def in_process_child(
        self,
        command: list[str],
        env: dict[str, str],
        log_path: Path,
        timeout_s: Optional[float],
        what: str = "the node",
    ) -> tuple[int, list[str]]:
        """`_run_child` that runs the worker command in this process."""
        log_path.write_text("")
        args = self.oracle.build_parser().parse_args(command[2:])
        with mock.patch.dict(os.environ, env, clear=True):
            try:
                return self.oracle.run_worker(args), []
            finally:
                capture = sys.modules.pop("cutracer_numerics", None)
                if capture is not None:
                    capture.uninstall()
                sys.modules.pop("test_numerics_plugin", None)

    def test_record_and_check_through_the_oracle(self) -> None:
        options = self.oracle.OracleOptions(
            python=sys.executable,
            rootdir=self.tmp,
            nodeid="test_numerics_plugin.py::test_values",
            plain=True,
        )

        def run(mode: str, **env: str) -> tuple[int, list[dict[str, Any]]]:
            """The oracle's exit code, and the outcome of each of its runs."""
            numerics = self.oracle.NumericsOptions(mode=mode, golden=self.golden)
            evidence = Path(tempfile.mkdtemp(dir=self.tmp))
            with mock.patch.dict(
                os.environ, {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", **env}
            ):
                with mock.patch.object(
                    self.oracle, "_run_child", self.in_process_child
                ):
                    with contextlib.redirect_stderr(io.StringIO()):
                        code = self.oracle.run_oracle(
                            replace(options, numerics=numerics, output_dir=evidence)
                        )
            outcomes = evidence.glob("attempt-*/outcome.json")
            return code, [json.loads(path.read_text()) for path in outcomes]

        self.assertEqual(run("record")[0], 0)
        manifest = json.loads((self.golden / "manifest.json").read_text())
        self.assertEqual(manifest["plain_runs"], 2)
        calls = json.loads((self.golden / "calls-main.json").read_text())["calls"]
        plain = {leaf["name"]: leaf.get("plain") for leaf in calls[0]["leaves"]}
        self.assertEqual(
            plain["a0"], {"runs": 2, "changed": 0, "max_rel_l2": 0.0, "max_ulp": 0}
        )
        code, (same,) = run("check")
        self.assertEqual((code, same["numerics"]["rule"]), (1, "comparable"))
        self.assertIn("numerics same over 1 calls", same["reason"])
        code, (large,) = run("check", NUMERICS_TEST_PERTURB="1")
        self.assertEqual((code, large["numerics"]["rule"]), (0, "large"))
        self.assertIn("test_numerics_plugin.py::test_values", large["reason"])


def _autotuned(x: Any, n: int, BLOCK: int) -> None:
    pass


class _FakeJit:
    """Stands in for a JITFunction under Triton's real autotuner: records the
    BLOCK of every launch instead of running a kernel."""

    def __init__(self) -> None:
        self.fn = _autotuned
        self.launches: list[int] = []

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.launches.append(kwargs["BLOCK"])


class AutotunePinningTest(_CaptureTestCase):
    """Triton's own `Autotuner` on CPU: a fake kernel, and a fake benchmark
    that times each BLOCK as `times` says."""

    def tuner(
        self,
        times: dict[int, float],
        cls: Any = Autotuner,
        arg_names: Optional[list[str]] = None,
        **options: Any,
    ) -> tuple[Any, _FakeJit, list[int]]:
        jit = _FakeJit()
        benchmarked: list[int] = []

        def do_bench(
            kernel_call: Callable[[], Any], quantiles: Any = None
        ) -> list[float]:
            kernel_call()
            benchmarked.append(jit.launches[-1])
            return [times[jit.launches[-1]]] * 3

        configs = [triton.Config({"BLOCK": 16}), triton.Config({"BLOCK": 32})]
        reset = options.pop("reset_to_zero", None)
        tuner = cls(
            jit,
            arg_names or ["x", "n"],
            configs,
            ["n"],
            reset,
            None,
            do_bench=do_bench,
            **options,
        )
        return tuner, jit, benchmarked

    def pinned(self, body: Callable[[], None], **env: str) -> dict:
        """Run `body` in a check that pins against the recorded choices; the
        result's autotune summary."""
        self.run_mode("check", body, out=self.out, env=env or None)
        return self.autotune_summary(self.out)

    def autotune_summary(self, out: Path) -> dict:
        return self.summary(out)["autotune"]

    def record_choices(self) -> None:
        def body() -> None:
            tuner, _, benchmarked = self.tuner({16: 2.0, 32: 1.0})
            tuner.run(torch.zeros(4), 7)
            self.assertEqual(benchmarked, [16, 32])

        self.run_mode("autotune", body, out=self.tmp / "autotune")

    def test_an_autotune_run_records_each_choice_under_triton_s_key(self) -> None:
        self.record_choices()
        (line,) = (self.golden / "autotune-main.jsonl").read_text().splitlines()
        entry = json.loads(line)
        self.assertEqual(entry["key"], repr((7, "torch.float32")))
        self.assertEqual(json.loads(entry["config"])["BLOCK"], 32)
        self.assertIn("_autotuned[n]#", entry["autotuner"])
        summary = self.summary(self.tmp / "autotune")
        self.assertEqual(summary["verdict"], "autotune-recorded")
        self.assertEqual(summary["autotune"]["recorded"], 1)

    def test_the_key_mirror_matches_triton_s_cache_key(self) -> None:
        capture = self.load()
        tuner, _, _ = self.tuner({16: 1.0, 32: 2.0})
        x = torch.zeros(4, dtype=torch.float16)
        tuner.run(x, n=5)
        self.assertEqual(
            list(tuner.cache), [capture._tuning_key(tuner, (x,), {"n": 5})]
        )

    def test_a_pinned_run_takes_the_recorded_config_without_benchmarking(self) -> None:
        self.record_choices()
        seen: dict[str, Any] = {}

        def body() -> None:
            # This benchmark would pick BLOCK 16.
            tuner, jit, benchmarked = self.tuner({16: 1.0, 32: 2.0})
            tuner.run(torch.zeros(4), 7)
            seen.update(launches=list(jit.launches), benchmarked=list(benchmarked))
            tuner.run(torch.zeros(4), 9)
            seen.update(after_miss=list(benchmarked))

        self.run_mode("check", body, out=self.out)
        self.assertEqual((seen["launches"], seen["benchmarked"]), ([32], []))
        self.assertEqual(seen["after_miss"], [16, 32])
        autotune = self.autotune_summary(self.out)
        self.assertEqual(
            (autotune["pinning"], autotune["pinned"], autotune["misses"]), ("on", 1, 1)
        )
        self.assertIn("not in the golden's record", autotune["miss_head"][0])

    def test_a_choice_made_before_the_capture_is_checked_against_the_record(
        self,
    ) -> None:
        self.record_choices()
        for times, misses in (({16: 2.0, 32: 1.0}, 0), ({16: 1.0, 32: 2.0}, 1)):
            # Tuned before the capture was installed.
            tuner, _, _ = self.tuner(times)
            tuner.run(torch.zeros(4), 7)
            out = self.tmp / f"check-{misses}"
            # Looked up when it runs, so it reaches the patched `run`.
            self.run_mode("check", lambda t=tuner: t.run(torch.zeros(4), 7), out=out)
            autotune = self.autotune_summary(out)
            with self.subTest(chosen=min(times, key=times.__getitem__)):
                self.assertEqual((autotune["pinned"], autotune["misses"]), (0, misses))
                if misses:
                    self.assertIn("chose", autotune["miss_head"][0])

    def test_a_choice_made_before_the_capture_misses_before_its_launch(self) -> None:
        self.record_choices()
        tuner, jit, _ = self.tuner({16: 1.0, 32: 2.0})
        tuner.run(torch.zeros(4), 7)

        def launch(*args: Any, **kwargs: Any) -> None:
            # A launch that records a comparison, as a pre-hook can.
            torch.testing.assert_close(torch.zeros(2), torch.zeros(2))

        jit.run = launch
        out = self.tmp / "check"
        self.run_mode("check", lambda: tuner.run(torch.zeros(4), 7), out=out)
        autotune = self.autotune_summary(out)
        self.assertEqual((autotune["misses"], autotune["first_miss_k"]), (1, 0))

    def test_a_choice_made_before_the_capture_is_recorded(self) -> None:
        tuner, _, _ = self.tuner({16: 2.0, 32: 1.0})
        tuner.run(torch.zeros(4), 7)
        self.run_mode(
            "autotune", lambda: tuner.run(torch.zeros(4), 7), out=self.tmp / "autotune"
        )
        (line,) = (self.golden / "autotune-main.jsonl").read_text().splitlines()
        self.assertEqual(json.loads(json.loads(line)["config"])["BLOCK"], 32)

    def test_a_pinned_run_neither_reads_nor_writes_the_disk_cache(self) -> None:
        self.record_choices()
        capture = self.load(
            CUTRACER_NUMERICS_MODE="check",
            CUTRACER_NUMERICS_GOLDEN=str(self.golden),
            CUTRACER_NUMERICS_OUT=str(self.out),
        )
        tuner, _, _ = self.tuner({16: 1.0, 32: 2.0})
        ran: list[bool] = []
        self.assertFalse(
            tuner.check_disk_cache((1,), tuner.configs, lambda: ran.append(True))
        )
        self.assertEqual(ran, [True])
        capture.uninstall()

    def test_without_a_record_or_with_pinning_off_nothing_is_pinned(self) -> None:
        original = Autotuner.run
        self.run_mode("check", lambda: None, out=self.out)
        self.assertEqual(
            self.autotune_summary(self.out)["pinning"], "golden has no record"
        )
        self.assertIs(Autotuner.run, original)
        self.record_choices()
        self.run_mode(
            "check",
            lambda: self.assertIs(Autotuner.run, original),
            out=self.tmp / "off",
            env={"CUTRACER_NUMERICS_PIN_AUTOTUNE": "0"},
        )
        self.assertEqual(self.autotune_summary(self.tmp / "off")["pinning"], "off")

    def test_uninstall_restores_the_autotuner(self) -> None:
        before = (Autotuner.run, Autotuner._bench, Autotuner.check_disk_cache)
        self.record_choices()
        capture = self.load(
            CUTRACER_NUMERICS_MODE="check",
            CUTRACER_NUMERICS_GOLDEN=str(self.golden),
            CUTRACER_NUMERICS_OUT=str(self.out),
        )
        self.assertIsNot(Autotuner.run, before[0])
        capture.uninstall()
        self.assertEqual(
            (Autotuner.run, Autotuner._bench, Autotuner.check_disk_cache), before
        )

    def test_an_autotuner_imported_after_the_capture_is_patched(self) -> None:
        name = f"cutracer_fake_autotuner_{next(_NAMES)}"
        (self.tmp / f"{name}.py").write_text(
            "class Autotuner:\n"
            "    def run(self, *args, **kwargs):\n        return 'run'\n"
            "    def _bench(self, *args, config, **meta):\n        return [1.0]\n"
            "    def check_disk_cache(self, key, configs, bench_fn):\n        return True\n"
        )
        sys.path.insert(0, str(self.tmp))
        self.addCleanup(sys.path.remove, str(self.tmp))
        self.addCleanup(sys.modules.pop, name, None)
        self.record_choices()
        capture = self.load()
        with mock.patch.object(capture, "_AUTOTUNER_MODULE", name):
            os.environ.update(
                CUTRACER_NUMERICS_MODE="check",
                CUTRACER_NUMERICS_GOLDEN=str(self.golden),
                CUTRACER_NUMERICS_OUT=str(self.out),
            )
            capture.install()
            self.assertEqual(capture._AUTOTUNE.state, "triton not imported")
            module = __import__(name)
            self.assertEqual(capture._AUTOTUNE.state, "on")
            self.assertTrue(
                hasattr(module.Autotuner.run, "_cutracer_numerics_original")
            )
            capture.uninstall()
            self.assertFalse(
                hasattr(module.Autotuner.run, "_cutracer_numerics_original")
            )
            self.assertNotIn(capture._IMPORT_HOOK, sys.meta_path)

    def test_a_config_the_autotuner_makes_up_is_pinned_while_benchmarking(
        self,
    ) -> None:
        class Widening(Autotuner):
            def prune_configs(self, kwargs: Any) -> Any:
                return [*self.configs, triton.Config({"BLOCK": 64})]

        def record() -> None:
            tuner, jit, _ = self.tuner({16: 2.0, 32: 2.0, 64: 1.0}, Widening)
            tuner.run(torch.zeros(4), 7)
            self.assertEqual(jit.launches[-1], 64)

        self.run_mode("autotune", record, out=self.tmp / "autotune")
        seen: dict[str, Any] = {}

        def body() -> None:
            tuner, jit, benchmarked = self.tuner({16: 1.0, 32: 2.0, 64: 3.0}, Widening)
            tuner.run(torch.zeros(4), 7)
            seen.update(launch=jit.launches[-1], benchmarked=benchmarked)

        autotune = self.pinned(body)
        self.assertEqual((seen["launch"], seen["benchmarked"]), (64, []))
        self.assertEqual((autotune["pinned"], autotune["misses"]), (1, 0))

    def test_an_autotuner_without_bench_still_pins_the_configs_it_declares(
        self,
    ) -> None:
        self.record_choices()
        seen: dict[str, Any] = {}

        def body() -> None:
            tuner, jit, benchmarked = self.tuner({16: 1.0, 32: 2.0})
            tuner.run(torch.zeros(4), 7)
            seen.update(launch=jit.launches[-1], benchmarked=benchmarked)

        with mock.patch.object(Autotuner, "_bench", None):
            autotune = self.pinned(body)
        self.assertEqual((seen["launch"], seen["benchmarked"]), (32, []))
        self.assertEqual(
            (autotune["pinning"], autotune["pinned"], autotune["misses"]), ("on", 1, 0)
        )

    def test_a_made_up_config_is_a_miss_when_bench_is_not_hooked(self) -> None:
        class Widening(Autotuner):
            def prune_configs(self, kwargs: Any) -> Any:
                return [*self.configs, triton.Config({"BLOCK": 64})]

        self.run_mode(
            "autotune",
            lambda: self.tuner({16: 2.0, 32: 2.0, 64: 1.0}, Widening)[0].run(
                torch.zeros(4), 7
            ),
            out=self.tmp / "autotune",
        )
        capture = self.load()
        pins = capture._AutotunePins("pin", self.golden, "main")
        tuner, _, _ = self.tuner({16: 1.0, 32: 2.0}, Widening)
        args = (torch.zeros(4), 7)
        key = capture._tuning_key(tuner, args, {})
        self.assertIsNone(capture._pin(pins, tuner, key, args, {}))
        self.assertEqual((pins.pinned, pins.n_misses), (0, 1))
        self.assertIn("no _bench", pins.misses[0])

    def test_a_different_choice_than_the_record_is_a_miss(self) -> None:
        class Widening(Autotuner):
            def prune_configs(self, kwargs: Any) -> Any:
                return [*self.configs, triton.Config({"BLOCK": 64})]

        def record() -> None:
            tuner, _, _ = self.tuner({16: 2.0, 32: 2.0, 64: 1.0}, Widening)
            tuner.run(torch.zeros(4), 7)

        self.run_mode("autotune", record, out=self.tmp / "autotune")
        # This run never offers the recorded BLOCK 64.
        autotune = self.pinned(
            lambda: self.tuner({16: 1.0, 32: 2.0})[0].run(torch.zeros(4), 7)
        )
        self.assertEqual(autotune["misses"], 1)
        self.assertIn("chose", autotune["miss_head"][0])

    def test_an_ambiguous_record_is_a_miss(self) -> None:
        self.record_choices()
        record = self.golden / "autotune-main.jsonl"
        (line,) = record.read_text().splitlines()
        other = json.loads(line)
        other["config"] = other["config"].replace('"BLOCK": 32', '"BLOCK": 16')
        record.write_text(line + "\n" + json.dumps(other) + "\n")
        autotune = self.pinned(
            lambda: self.tuner({16: 1.0, 32: 2.0})[0].run(torch.zeros(4), 7)
        )
        self.assertIn("recorded with different configs", autotune["miss_head"][0])

    def test_pinning_resets_as_triton_does_after_benchmarking(self) -> None:
        def record() -> None:
            tuner, _, _ = self.tuner({16: 2.0, 32: 1.0}, reset_to_zero=["x"])
            tuner.run(torch.ones(4), 7)

        self.run_mode("autotune", record, out=self.tmp / "autotune")
        x = torch.ones(4)
        self.pinned(
            lambda: self.tuner({16: 1.0, 32: 2.0}, reset_to_zero=["x"])[0].run(x, 7)
        )
        self.assertEqual(x.tolist(), [0.0] * 4)

    def test_heuristics_over_an_autotuner_are_pinned(self) -> None:
        def run(times: dict[int, float]) -> list[int]:
            tuner, jit, _ = self.tuner(times)
            heuristic = triton.runtime.autotuner.Heuristics(
                tuner, ["x", "n"], {"EXTRA": lambda args: args["n"] + 1}
            )
            heuristic.run(torch.zeros(4), 7)
            return jit.launches

        self.run_mode(
            "autotune", lambda: run({16: 2.0, 32: 1.0}), out=self.tmp / "autotune"
        )
        launches: list[int] = []
        autotune = self.pinned(lambda: launches.extend(run({16: 1.0, 32: 2.0})))
        self.assertEqual((launches, autotune["misses"]), ([32], 0))

    def test_a_pinned_run_with_disk_caching_writes_no_cache_file(self) -> None:
        self.record_choices()
        cache = self.tmp / "triton_cache"

        def body() -> None:
            with mock.patch.dict(os.environ, {"TRITON_CACHE_DIR": str(cache)}):
                tuner, _, benchmarked = self.tuner(
                    {16: 1.0, 32: 2.0}, cache_results=True
                )
                tuner.run(torch.zeros(4), 9)
                self.assertEqual(benchmarked, [16, 32])

        autotune = self.pinned(body)
        self.assertEqual(autotune["misses"], 1)
        self.assertEqual(list(cache.rglob("*.json")) if cache.exists() else [], [])

    def test_autotuners_that_share_a_function_are_told_apart(self) -> None:
        def run(first: dict[int, float], second: dict[int, float]) -> list[int]:
            a, jit_a, _ = self.tuner(first)
            b, jit_b, _ = self.tuner(second)
            a.run(torch.zeros(4), 7)
            b.run(torch.zeros(4), 7)
            return jit_a.launches[-1:] + jit_b.launches[-1:]

        self.run_mode(
            "autotune",
            lambda: self.assertEqual(
                run({16: 1.0, 32: 2.0}, {16: 2.0, 32: 1.0}), [16, 32]
            ),
            out=self.tmp / "autotune",
        )
        launches: list[int] = []
        autotune = self.pinned(
            lambda: launches.extend(run({16: 3.0, 32: 3.0}, {16: 3.0, 32: 3.0}))
        )
        self.assertEqual((launches, autotune["misses"]), ([16, 32], 0))

    def test_the_key_mirror_with_keyword_and_several_dtype_arguments(self) -> None:
        capture = self.load()
        tuner, _, _ = self.tuner({16: 1.0, 32: 2.0}, arg_names=["x", "y", "n"])
        x = torch.zeros(4, dtype=torch.float16)
        y = torch.zeros(4, dtype=torch.int32)
        tuner.run(x, n=5, y=y)
        self.assertEqual(
            list(tuner.cache), [capture._tuning_key(tuner, (x,), {"n": 5, "y": y})]
        )

    def test_an_autotuner_imported_without_the_hook_is_reported(self) -> None:
        name = f"cutracer_fake_autotuner_{next(_NAMES)}"
        self.record_choices()
        capture = self.load()
        with mock.patch.object(capture, "_AUTOTUNER_MODULE", name):
            os.environ.update(
                CUTRACER_NUMERICS_MODE="check",
                CUTRACER_NUMERICS_GOLDEN=str(self.golden),
                CUTRACER_NUMERICS_OUT=str(self.out),
            )
            capture.install()
            with mock.patch.dict(sys.modules, {name: ModuleType(name)}):
                capture.finish()
            capture.uninstall()
        self.assertTrue(self.autotune_summary(self.out)["pinning"].startswith("missed"))

    def test_a_second_recorder_of_one_label_does_not_truncate_the_record(self) -> None:
        self.record_choices()
        record = (self.golden / "autotune-main.jsonl").read_text()
        self.run_mode("autotune", lambda: None, out=self.tmp / "again")
        self.assertIn(
            "unavailable", self.autotune_summary(self.tmp / "again")["pinning"]
        )
        self.assertEqual((self.golden / "autotune-main.jsonl").read_text(), record)

    def test_a_loader_without_create_module_still_imports(self) -> None:
        name = f"cutracer_fake_autotuner_{next(_NAMES)}"

        class OnlyExec:
            def exec_module(self, module: Any) -> None:
                exec(
                    "class Autotuner:\n"
                    "    def run(self, *a, **k):\n        return 'run'\n"
                    "    def _bench(self, *a, config, **m):\n        return [1.0]\n",
                    vars(module),
                )

        class Finder:
            def find_spec(
                self, fullname: str, path: Any = None, target: Any = None
            ) -> Any:
                if fullname != name:
                    return None
                return importlib.util.spec_from_loader(fullname, cast(Any, OnlyExec()))

        finder = Finder()
        sys.meta_path.append(finder)
        self.addCleanup(sys.meta_path.remove, finder)
        self.addCleanup(sys.modules.pop, name, None)
        self.record_choices()
        capture = self.load()
        with mock.patch.object(capture, "_AUTOTUNER_MODULE", name):
            os.environ.update(
                CUTRACER_NUMERICS_MODE="check",
                CUTRACER_NUMERICS_GOLDEN=str(self.golden),
                CUTRACER_NUMERICS_OUT=str(self.out),
            )
            capture.install()
            module = __import__(name)
            self.assertEqual(module.Autotuner().run(), "run")
            self.assertEqual(capture._AUTOTUNE.state, "on")
            capture.uninstall()
