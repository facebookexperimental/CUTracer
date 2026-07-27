# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Tests for cutracer.runner module."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
from cutracer.runner import (
    _build_cutracer_env,
    InstrumentationConfig,
    resolve_cutracer_so,
    run_instrumented_target,
)


class ResolveCutracerSoExplicitPathTest(unittest.TestCase):
    """Tests for resolve_cutracer_so with explicit --cutracer-so path."""

    def test_explicit_path_valid_file(self) -> None:
        """Explicit path to an existing file returns resolved path."""
        with tempfile.NamedTemporaryFile(suffix=".so") as f:
            result = resolve_cutracer_so(explicit_path=f.name)
            self.assertEqual(result, str(Path(f.name).resolve()))

    def test_explicit_path_missing_file(self) -> None:
        """Explicit path to a missing file raises ClickException."""
        with self.assertRaises(click.ClickException) as ctx:
            resolve_cutracer_so(explicit_path="/nonexistent/cutracer.so")
        self.assertIn("not found", ctx.exception.message)


class ResolveCutracerSoCudaInjectionErrorTest(unittest.TestCase):
    """Tests for CUDA_INJECTION64_PATH error."""

    @patch.dict(os.environ, {"CUDA_INJECTION64_PATH": "/some/path.so"}, clear=False)
    def test_cuda_injection_path_raises_error(self) -> None:
        """When CUDA_INJECTION64_PATH is set, raises ClickException."""
        with self.assertRaises(click.ClickException) as ctx:
            resolve_cutracer_so()
        self.assertIn("CUDA_INJECTION64_PATH", ctx.exception.message)
        self.assertIn("is set in your environment", ctx.exception.message)


class ResolveCutracerSoCwdAutoDiscoveryTest(unittest.TestCase):
    """Tests for CWD auto-discovery of ./lib/cutracer.so."""

    def test_cwd_auto_discovery(self) -> None:
        """When ./lib/cutracer.so exists in CWD, it is discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_dir = os.path.join(tmpdir, "lib")
            os.makedirs(lib_dir)
            so_file = os.path.join(lib_dir, "cutracer.so")
            Path(so_file).touch()

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {}, clear=True):
                    with patch("cutracer.runner.resources") as mock_resources:
                        mock_resources.files.side_effect = Exception("no resource")
                        result = resolve_cutracer_so()
                self.assertEqual(result, so_file)
            finally:
                os.chdir(original_cwd)


class ResolveCutracerSoAllFailTest(unittest.TestCase):
    """Tests for when all resolution methods fail."""

    @patch("cutracer.runner.resources")
    def test_all_paths_fail_raises_click_exception(
        self, mock_resources: MagicMock
    ) -> None:
        """When no resolution method works, raises ClickException with help."""
        mock_resources.files.side_effect = Exception("no resource")

        with tempfile.TemporaryDirectory() as tmpdir:
            # CWD has no lib/cutracer.so
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(click.ClickException) as ctx:
                        resolve_cutracer_so()
                    self.assertIn("Could not find cutracer.so", ctx.exception.message)
                    self.assertIn("--cutracer-so", ctx.exception.message)
            finally:
                os.chdir(original_cwd)


class BuildCutracerEnvTest(unittest.TestCase):
    """Tests for _build_cutracer_env."""

    @patch("cutracer.runner.resources")
    def test_sets_cuda_injection_path(self, mock_resources: MagicMock) -> None:
        """CUDA_INJECTION64_PATH is set to the provided cutracer_so path."""
        mock_resources.files.side_effect = Exception("no resource")
        env = _build_cutracer_env(
            cutracer_so="/path/to/cutracer.so",
            instrument=None,
            analysis=None,
            kernel_filters=None,
            instr_categories=None,
            trace_format=None,
            output_dir=None,
            verbose=None,
            zstd_level=None,
            delay_ns=None,
        )
        self.assertEqual(env["CUDA_INJECTION64_PATH"], "/path/to/cutracer.so")

    @patch("cutracer.runner.resources")
    def test_sets_instrument_env_var(self, mock_resources: MagicMock) -> None:
        """CUTRACER_INSTRUMENT is set when instrument is provided."""
        mock_resources.files.side_effect = Exception("no resource")
        env = _build_cutracer_env(
            cutracer_so="/path/to/cutracer.so",
            instrument="tma_trace",
            analysis=None,
            kernel_filters=None,
            instr_categories=None,
            trace_format=None,
            output_dir=None,
            verbose=None,
            zstd_level=None,
            delay_ns=None,
        )
        self.assertEqual(env["CUTRACER_INSTRUMENT"], "tma_trace")

    @patch("cutracer.runner.resources")
    def test_omits_none_values(self, mock_resources: MagicMock) -> None:
        """None-valued parameters are not set in the environment."""
        mock_resources.files.side_effect = Exception("no resource")
        env = _build_cutracer_env(
            cutracer_so="/path/to/cutracer.so",
            instrument=None,
            analysis=None,
            kernel_filters=None,
            instr_categories=None,
            trace_format=None,
            output_dir=None,
            verbose=None,
            zstd_level=None,
            delay_ns=None,
        )
        self.assertNotIn("CUTRACER_INSTRUMENT", env)
        self.assertNotIn("CUTRACER_ANALYSIS", env)
        self.assertNotIn("KERNEL_FILTERS", env)

    @patch("cutracer.runner.resources")
    def test_sets_all_delay_params(self, mock_resources: MagicMock) -> None:
        """All delay-related parameters are set correctly."""
        mock_resources.files.side_effect = Exception("no resource")
        env = _build_cutracer_env(
            cutracer_so="/path/to/cutracer.so",
            instrument="random_delay",
            analysis="random_delay",
            kernel_filters=None,
            instr_categories=None,
            trace_format=None,
            output_dir=None,
            verbose=None,
            zstd_level=None,
            delay_ns=10000,
            delay_min_ns=100,
            delay_mode="random",
            delay_dump_path="/tmp/dump.json",
            delay_load_path="/tmp/load.json",
        )
        self.assertEqual(env["CUTRACER_DELAY_NS"], "10000")
        self.assertEqual(env["CUTRACER_DELAY_MIN_NS"], "100")
        self.assertEqual(env["CUTRACER_DELAY_MODE"], "random")
        self.assertEqual(env["CUTRACER_DELAY_DUMP_PATH"], "/tmp/dump.json")
        self.assertEqual(env["CUTRACER_DELAY_LOAD_PATH"], "/tmp/load.json")

    @patch("cutracer.runner.resources")
    def test_dump_cubin_flag(self, mock_resources: MagicMock) -> None:
        """CUTRACER_DUMP_CUBIN is set to '1' when dump_cubin is True."""
        mock_resources.files.side_effect = Exception("no resource")
        env = _build_cutracer_env(
            cutracer_so="/path/to/cutracer.so",
            instrument=None,
            analysis=None,
            kernel_filters=None,
            instr_categories=None,
            trace_format=None,
            output_dir=None,
            verbose=None,
            zstd_level=None,
            delay_ns=None,
            dump_cubin=True,
        )
        self.assertEqual(env["CUTRACER_DUMP_CUBIN"], "1")

    @patch("cutracer.runner.resources")
    def test_default_timeout_values(self, mock_resources: MagicMock) -> None:
        """Default timeout values are set correctly."""
        mock_resources.files.side_effect = Exception("no resource")
        env = _build_cutracer_env(
            cutracer_so="/path/to/cutracer.so",
            instrument=None,
            analysis=None,
            kernel_filters=None,
            instr_categories=None,
            trace_format=None,
            output_dir=None,
            verbose=None,
            zstd_level=None,
            delay_ns=None,
        )
        self.assertEqual(env["CUTRACER_TRACE_SIZE_LIMIT_MB"], "0")
        self.assertEqual(env["CUTRACER_KERNEL_TIMEOUT_S"], "0")
        self.assertEqual(env["CUTRACER_NO_DATA_TIMEOUT_S"], "15")


class RunInstrumentedTargetTest(unittest.TestCase):
    def test_programmatic_runner_uses_structured_argv_and_explicit_runtime(
        self,
    ) -> None:
        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            return MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.NamedTemporaryFile(suffix=".so") as so:
            result = run_instrumented_target(
                ["python", "test.py"],
                InstrumentationConfig(
                    cutracer_so=so.name,
                    instrument="random_delay",
                    analysis="random_delay",
                    delay_load_path="/tmp/replay.json",
                    base_env={"KEEP": "yes"},
                    cwd="/tmp",
                    timeout=12,
                ),
                runner=fake_runner,
            )

        self.assertEqual(result.returncode, 0)
        command, kwargs = calls[0]
        self.assertEqual(command, ["python", "test.py"])
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["cwd"], "/tmp")
        self.assertEqual(kwargs["timeout"], 12)
        self.assertEqual(kwargs["env"]["KEEP"], "yes")
        self.assertEqual(kwargs["env"]["CUTRACER_INSTRUMENT"], "random_delay")
        self.assertEqual(
            kwargs["env"]["CUTRACER_DELAY_LOAD_PATH"],
            str(Path("/tmp/replay.json").resolve()),
        )

    def test_shell_mode_requires_explicit_shell_for_operators(self) -> None:
        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            return MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.NamedTemporaryFile(suffix=".so") as so:
            run_instrumented_target(
                ["sh", "-c", "producer | consumer"],
                InstrumentationConfig(cutracer_so=so.name, shell=True),
                runner=fake_runner,
            )

        command, kwargs = calls[0]
        self.assertEqual(command, "sh -c 'producer | consumer'")
        self.assertTrue(kwargs["shell"])
