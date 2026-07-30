# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Tests for cutracer.runner module."""

import os
import shutil
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
from cutracer.runner import (
    _build_cutracer_env,
    _BundledCudaTools,
    _persist_extracted_cuda_tools,
    _resolve_bundled_cuda_tools,
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
                        mock_resources.files.side_effect = ModuleNotFoundError(
                            "no resource"
                        )
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
        mock_resources.files.side_effect = ModuleNotFoundError("no resource")

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
        mock_resources.files.side_effect = ModuleNotFoundError("no resource")
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
        mock_resources.files.side_effect = ModuleNotFoundError("no resource")
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
        mock_resources.files.side_effect = ModuleNotFoundError("no resource")
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
        mock_resources.files.side_effect = ModuleNotFoundError("no resource")
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
        mock_resources.files.side_effect = ModuleNotFoundError("no resource")
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
        mock_resources.files.side_effect = ModuleNotFoundError("no resource")
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

    @patch("cutracer.runner._resolve_bundled_cuda_tools")
    def test_sets_persistent_cuda_tool_paths(self, resolve_tools) -> None:
        resolve_tools.return_value = _BundledCudaTools(
            bin_dir=Path("/cache/aarch64/digest/bin"),
            nvdisasm=Path("/cache/aarch64/digest/bin/nvdisasm"),
            cuobjdump=Path("/cache/aarch64/digest/bin/cuobjdump"),
        )
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
            base_env={"PATH": "/usr/bin"},
        )

        self.assertEqual(env["NVDISASM"], "/cache/aarch64/digest/bin/nvdisasm")
        self.assertEqual(env["PATH"], "/cache/aarch64/digest/bin:/usr/bin")


class PersistExtractedCudaToolsTest(unittest.TestCase):
    def _write_tools(self, directory: Path, suffix: bytes = b"v1") -> dict[str, Path]:
        directory.mkdir(parents=True)
        tools = {
            "nvdisasm": directory / "nvdisasm",
            "cuobjdump": directory / "cuobjdump",
        }
        for name, path in tools.items():
            path.write_bytes(name.encode() + suffix)
        return tools

    def test_tools_outlive_extracted_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extracted_dir = root / "extracted"
            extracted = self._write_tools(extracted_dir)
            with patch("cutracer.runner.tempfile.gettempdir", return_value=tmpdir):
                with patch("cutracer.runner.platform.machine", return_value="x86_64"):
                    persisted = _persist_extracted_cuda_tools(extracted)

            shutil.rmtree(extracted_dir)
            self.assertEqual(persisted.nvdisasm.read_bytes(), b"nvdisasmv1")
            self.assertEqual(persisted.cuobjdump.read_bytes(), b"cuobjdumpv1")
            self.assertTrue(os.access(persisted.nvdisasm, os.X_OK))
            self.assertIn("x86_64", persisted.bin_dir.parts)

    def test_host_arch_and_content_have_distinct_cache_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extracted = self._write_tools(root / "extracted")
            with patch("cutracer.runner.tempfile.gettempdir", return_value=tmpdir):
                with patch("cutracer.runner.platform.machine", return_value="x86_64"):
                    x86 = _persist_extracted_cuda_tools(extracted)
                with patch("cutracer.runner.platform.machine", return_value="aarch64"):
                    arm = _persist_extracted_cuda_tools(extracted)

                extracted["nvdisasm"].write_bytes(b"nvdisasmv2")
                with patch("cutracer.runner.platform.machine", return_value="x86_64"):
                    updated = _persist_extracted_cuda_tools(extracted)

            self.assertNotEqual(x86.bin_dir, arm.bin_dir)
            self.assertNotEqual(x86.bin_dir, updated.bin_dir)

    @patch("cutracer.runner.resources")
    def test_missing_bundled_tools_are_optional(self, mock_resources) -> None:
        package = mock_resources.files.return_value
        package.joinpath.return_value.is_file.return_value = False

        self.assertIsNone(_resolve_bundled_cuda_tools())
        mock_resources.as_file.assert_not_called()

    @patch("cutracer.runner._persist_extracted_cuda_tools")
    @patch("cutracer.runner.resources")
    def test_persistence_failure_is_not_swallowed(
        self, mock_resources, persist_tools
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nvdisasm = Path(tmpdir) / "nvdisasm"
            cuobjdump = Path(tmpdir) / "cuobjdump"
            nvdisasm.touch()
            cuobjdump.touch()

            tool_refs = {
                "bin/nvdisasm": MagicMock(),
                "bin/cuobjdump": MagicMock(),
            }
            for ref in tool_refs.values():
                ref.is_file.return_value = True
            package = mock_resources.files.return_value
            package.joinpath.side_effect = tool_refs.__getitem__
            mock_resources.as_file.side_effect = [
                nullcontext(nvdisasm),
                nullcontext(cuobjdump),
            ]
            persist_tools.side_effect = PermissionError("read-only cache")

            with self.assertRaises(click.ClickException) as ctx:
                _resolve_bundled_cuda_tools()

        self.assertIn("Failed to prepare bundled CUDA tools", ctx.exception.message)
        self.assertIn("read-only cache", ctx.exception.message)


class RunInstrumentedTargetTest(unittest.TestCase):
    @patch("cutracer.runner.resolve_cutracer_so", return_value="/current/cutracer.so")
    def test_programmatic_runner_overrides_inherited_injection(self, resolve) -> None:
        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            return MagicMock(returncode=0, stdout="", stderr="")

        run_instrumented_target(
            ["./oracle"],
            InstrumentationConfig(
                instrument="random_delay",
                base_env={"CUDA_INJECTION64_PATH": "/feature/cutracer.so"},
            ),
            runner=fake_runner,
        )

        resolve.assert_called_once_with(None, reject_inherited_injection=False)
        self.assertEqual(
            calls[0][1]["env"]["CUDA_INJECTION64_PATH"], "/current/cutracer.so"
        )

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
