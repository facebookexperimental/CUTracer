# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
E2E test: CUTracer trace format validation with PT2 compiled kernel.

Uses a @torch.compile kernel (test_add.py) to verify that CUTracer correctly
instruments and traces GPU kernels across all three trace formats:
  - Mode 0: Text (.log)
  - Mode 1: NDJSON + Zstd (.ndjson.zst)
  - Mode 2: NDJSON (.ndjson)

Also validates kernel filter functionality and cross-format consistency.

This is the Buck equivalent of the GitHub CI tests:
  - test_py_add_with_kernel_filters (from .ci/run_tests.sh)
  - test_trace_formats (from .ci/run_tests.sh)

Equivalent CLI usage::

    buck2 run fbcode//triton/tools/CUTracer:cutracer -- trace \\
        -i reg_trace,mem_trace -k triton_poi_fused --trace-format 0 \\
        -- python test_add.py
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
import zstandard
from cutracer.runner import _build_cutracer_env, resolve_cutracer_so


class TestPyAddE2E(unittest.TestCase):
    """E2E: CUTracer tracing of a PT2 compiled kernel (py_add)."""

    @classmethod
    def setUpClass(cls):
        # Clear CUDA_INJECTION64_PATH if pre-set in the environment to avoid
        # conflict with resolve_cutracer_so() which rejects pre-set values.
        os.environ.pop("CUDA_INJECTION64_PATH", None)

        cls.cutracer_so = resolve_cutracer_so()

        # Extract the test script from bundled resources
        script_content = pkg_resources.resource_string(
            "py_add_test_resources",
            "test_add.py",
        ).decode()
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, prefix="py_add_"
        )
        tmp.write(script_content)
        tmp.flush()
        tmp.close()
        cls.script_path = tmp.name

        # Temp directory for CUTracer trace output
        cls.trace_dir = tempfile.mkdtemp(prefix="cutracer_py_add_")

        # Shared Triton/Inductor cache directory for all subprocesses.
        # This ensures torch.compile produces the same kernel binary across
        # multiple runs (Mode 0, Mode 1, Mode 2), which is required for
        # cross-format consistency checks.
        cls.triton_cache_dir = tempfile.mkdtemp(prefix="triton_cache_")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "script_path") and os.path.exists(cls.script_path):
            os.unlink(cls.script_path)
        if hasattr(cls, "trace_dir") and os.path.exists(cls.trace_dir):
            shutil.rmtree(cls.trace_dir, ignore_errors=True)
        if hasattr(cls, "triton_cache_dir") and os.path.exists(cls.triton_cache_dir):
            shutil.rmtree(cls.triton_cache_dir, ignore_errors=True)

    def _clean_trace_dir(self):
        """Remove all trace files from the output directory."""
        for f in Path(self.trace_dir).iterdir():
            if f.is_file():
                f.unlink()

    def _run_py_add(
        self,
        trace_format=None,
        instrument="reg_trace",
        kernel_filter="triton_poi_fused",
        dump_cubin=False,
        extra_env=None,
    ):
        """Run test_add.py as a subprocess with CUTracer instrumentation.

        This mirrors the pattern used by cutracer trace CLI and the
        datarace_test E2E test: build environment via _build_cutracer_env,
        then subprocess.run the target script.

        Args:
            trace_format: Trace output format (0=text, 1=ndjson+zstd, 2=ndjson).
            instrument: Instrumentation mode(s), comma-separated.
            kernel_filter: Kernel name filter substring.
            dump_cubin: If True, set CUTRACER_DUMP_CUBIN=1.
            extra_env: Additional env vars to override (applied after
                _build_cutracer_env, so can override any CUTracer var).

        Returns:
            subprocess.CompletedProcess with stdout/stderr captured.
        """
        self._clean_trace_dir()

        env = _build_cutracer_env(
            cutracer_so=self.cutracer_so,
            instrument=instrument,
            analysis=None,
            kernel_filters=kernel_filter,
            instr_categories=None,
            trace_format=str(trace_format) if trace_format is not None else None,
            output_dir=self.trace_dir,
            verbose=None,
            zstd_level=None,
            delay_ns=None,
            dump_cubin=dump_cubin,
        )
        if extra_env:
            env.update(extra_env)
        # Pin Triton/Inductor cache so all subprocesses share the same
        # compiled kernels (deterministic CTA/warp config across runs).
        env["TRITON_CACHE_DIR"] = self.triton_cache_dir
        env["TORCHINDUCTOR_CACHE_DIR"] = self.triton_cache_dir

        result = subprocess.run(
            [sys.executable, self.script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result

    # ------------------------------------------------------------------
    # Test: Kernel filter + reg_trace (mirrors test_py_add_with_kernel_filters)
    # ------------------------------------------------------------------

    def test_kernel_filter_reg_trace(self):
        """Kernel filter ensures only triton_poi_fused kernel is traced."""
        result = self._run_py_add(trace_format=0, instrument="reg_trace")
        self.assertEqual(
            result.returncode,
            0,
            f"test_add.py failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )

        # Verify kernel log files were generated
        log_files = sorted(Path(self.trace_dir).glob("kernel_*.log"))
        self.assertGreater(
            len(log_files),
            0,
            f"No kernel log files generated in {self.trace_dir}",
        )

        # All generated logs must match the kernel filter
        for f in log_files:
            self.assertIn(
                "triton_poi_fused",
                f.name,
                f"Kernel log does not match filter: {f.name}",
            )

        # Verify EXIT pattern exists in the log (proves tracing completed)
        log_content = log_files[0].read_text()
        exit_pattern = r"CTA \d+,\d+,\d+ - warp \d+ - .*EXIT"
        self.assertRegex(
            log_content,
            exit_pattern,
            f"EXIT pattern not found in {log_files[0].name}",
        )

    # ------------------------------------------------------------------
    # Test: Mode 0 text format (mirrors test_trace_formats Mode 0)
    # ------------------------------------------------------------------

    def test_trace_format_mode0_text(self):
        """Mode 0 (text) generates a valid .log trace file."""
        result = self._run_py_add(trace_format=0, instrument="reg_trace,mem_trace")
        self.assertEqual(
            result.returncode,
            0,
            f"Mode 0 run failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )

        log_files = sorted(Path(self.trace_dir).glob("kernel_*triton_poi_fused*.log"))
        self.assertGreater(len(log_files), 0, "No .log file generated for Mode 0")

        from cutracer.validation.text_validator import validate_text_trace

        metadata = validate_text_trace(log_files[0])
        self.assertTrue(
            metadata["valid"],
            f"Text validation failed: {metadata['errors']}",
        )
        self.assertGreater(
            metadata["record_count"],
            0,
            "Mode 0 produced zero records",
        )

    # ------------------------------------------------------------------
    # Test: Mode 2 NDJSON format (mirrors test_trace_formats Mode 2)
    # ------------------------------------------------------------------

    def test_trace_format_mode2_ndjson(self):
        """Mode 2 (NDJSON) generates a valid .ndjson trace file."""
        result = self._run_py_add(trace_format=2, instrument="reg_trace,mem_trace")
        self.assertEqual(
            result.returncode,
            0,
            f"Mode 2 run failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )

        ndjson_files = sorted(
            Path(self.trace_dir).glob("kernel_*triton_poi_fused*.ndjson")
        )
        self.assertGreater(len(ndjson_files), 0, "No .ndjson file generated for Mode 2")

        from cutracer.validation.json_validator import validate_json_trace

        metadata = validate_json_trace(ndjson_files[0])
        self.assertTrue(
            metadata["valid"],
            f"JSON validation failed: {metadata['errors']}",
        )
        self.assertGreater(
            metadata["record_count"],
            0,
            "Mode 2 produced zero records",
        )

    # ------------------------------------------------------------------
    # Test: Mode 1 NDJSON+Zstd format (mirrors test_trace_formats Mode 1)
    # ------------------------------------------------------------------

    def test_trace_format_mode1_ndjson_zstd(self):
        """Mode 1 (NDJSON+Zstd) generates a valid .ndjson.zst file."""
        result = self._run_py_add(trace_format=1, instrument="reg_trace,mem_trace")
        self.assertEqual(
            result.returncode,
            0,
            f"Mode 1 run failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )

        zst_files = sorted(
            Path(self.trace_dir).glob("kernel_*triton_poi_fused*.ndjson.zst")
        )
        self.assertGreater(
            len(zst_files), 0, "No .ndjson.zst file generated for Mode 1"
        )

        # Verify decompression works and content is valid NDJSON.
        # CUTracer uses streaming zstd (multiple frames from periodic flush),
        # so we must use a streaming reader — decompress() only handles
        # a single frame and would silently return partial data.
        dctx = zstandard.ZstdDecompressor()
        chunks = []
        with open(zst_files[0], "rb") as f:
            with dctx.stream_reader(f) as reader:
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
        decompressed = b"".join(chunks)
        lines = [
            line for line in decompressed.decode().strip().split("\n") if line.strip()
        ]
        self.assertGreater(len(lines), 0, "Decompressed file is empty")

        # Each line must be valid JSON with a 'type' field
        for i, line in enumerate(lines[:20]):
            record = json.loads(line)
            self.assertIn(
                "type",
                record,
                f"Line {i} missing 'type' field: {line[:100]}",
            )

    # ------------------------------------------------------------------
    # Test: Cubin dump — explicit enable
    # ------------------------------------------------------------------

    def test_cubin_dump_enabled(self):
        """CUTRACER_DUMP_CUBIN=1 generates .cubin files in output dir."""
        result = self._run_py_add(
            trace_format=0,
            instrument="reg_trace",
            dump_cubin=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Run failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )

        cubin_files = sorted(Path(self.trace_dir).glob("kernel_*.cubin"))
        self.assertGreater(
            len(cubin_files),
            0,
            f"No .cubin files generated in {self.trace_dir}",
        )

        # Verify filename format: kernel_{hex_checksum}_{name}.cubin
        for f in cubin_files:
            self.assertRegex(
                f.name,
                r"^kernel_[0-9a-f]+_.+\.cubin$",
                f"Cubin filename does not match expected format: {f.name}",
            )

        # Verify cubin files are non-empty (actual binary content)
        for f in cubin_files:
            self.assertGreater(
                f.stat().st_size,
                0,
                f"Cubin file is empty: {f.name}",
            )

    # ------------------------------------------------------------------
    # Test: Cubin dump — explicit disable
    # ------------------------------------------------------------------

    def test_cubin_dump_disabled(self):
        """CUTRACER_DUMP_CUBIN=0 suppresses cubin file generation."""
        result = self._run_py_add(
            trace_format=0,
            instrument="reg_trace",
            extra_env={"CUTRACER_DUMP_CUBIN": "0"},
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Run failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )

        cubin_files = sorted(Path(self.trace_dir).glob("kernel_*.cubin"))
        self.assertEqual(
            len(cubin_files),
            0,
            f"Cubin files should NOT be generated when CUTRACER_DUMP_CUBIN=0, "
            f"but found: {[f.name for f in cubin_files]}",
        )

    # ------------------------------------------------------------------
    # Test: Cubin dump — default behavior (auto-enabled with instrumentation)
    # ------------------------------------------------------------------

    def test_cubin_dump_default_with_instrumentation(self):
        """Cubin dump auto-enables when instrumentation is active (default)."""
        result = self._run_py_add(
            trace_format=0,
            instrument="reg_trace",
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Run failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )

        # Without explicit CUTRACER_DUMP_CUBIN, the C++ code defaults to
        # has_any_instrumentation_enabled(), which is true here.
        cubin_files = sorted(Path(self.trace_dir).glob("kernel_*.cubin"))
        self.assertGreater(
            len(cubin_files),
            0,
            f"Cubin files should be auto-generated when instrumentation is "
            f"enabled, but none found in {self.trace_dir}",
        )

    # ------------------------------------------------------------------
    # Test: Cross-format consistency (mirrors test_trace_formats cross-validation)
    # ------------------------------------------------------------------

    def test_cross_format_consistency(self):
        """Mode 0 and Mode 2 produce consistent trace data.

        Runs a warmup subprocess first (without CUTracer) to populate the
        Triton compilation cache, ensuring both Mode 0 and Mode 2 runs
        use the exact same compiled kernel binary.
        """
        # --- Warmup: populate Triton/Inductor cache without CUTracer ---
        # The first torch.compile call triggers Triton compilation + autotuner,
        # which launches benchmark kernels that CUTracer would also trace.
        # Running warmup without CUTracer populates the shared cache so that
        # subsequent Mode 0 and Mode 2 runs only execute the final kernel.
        warmup_env = os.environ.copy()
        warmup_env["TRITON_CACHE_DIR"] = self.triton_cache_dir
        warmup_env["TORCHINDUCTOR_CACHE_DIR"] = self.triton_cache_dir
        warmup = subprocess.run(
            [sys.executable, self.script_path],
            env=warmup_env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            warmup.returncode,
            0,
            f"Warmup failed.\nSTDOUT: {warmup.stdout}\nSTDERR: {warmup.stderr}",
        )

        # --- Run Mode 0 (text) ---
        result_m0 = self._run_py_add(trace_format=0, instrument="reg_trace,mem_trace")
        self.assertEqual(
            result_m0.returncode,
            0,
            f"Mode 0 failed.\nSTDOUT: {result_m0.stdout}\nSTDERR: {result_m0.stderr}",
        )
        log_files = sorted(Path(self.trace_dir).glob("kernel_*triton_poi_fused*.log"))
        self.assertGreater(len(log_files), 0, "No .log file for cross-format test")

        # Save the text file OUTSIDE trace_dir before running Mode 2,
        # because _run_py_add() cleans trace_dir at the start of each run.
        text_backup = tempfile.NamedTemporaryFile(
            suffix=".log", prefix="mode0_backup_", delete=False
        )
        text_backup.close()
        shutil.copy2(log_files[0], text_backup.name)

        try:
            # --- Run Mode 2 (NDJSON) ---
            result_m2 = self._run_py_add(
                trace_format=2, instrument="reg_trace,mem_trace"
            )
            self.assertEqual(
                result_m2.returncode,
                0,
                f"Mode 2 failed.\nSTDOUT: {result_m2.stdout}\nSTDERR: {result_m2.stderr}",
            )
            ndjson_files = sorted(
                Path(self.trace_dir).glob("kernel_*triton_poi_fused*.ndjson")
            )
            self.assertGreater(
                len(ndjson_files), 0, "No .ndjson file for cross-format test"
            )
            json_file = ndjson_files[0]

            # --- Cross-format comparison using cutracer validation API ---
            from cutracer.validation.consistency import compare_trace_formats

            comparison = compare_trace_formats(Path(text_backup.name), json_file)
            self.assertTrue(
                comparison["consistent"],
                f"Cross-format inconsistency detected:\n"
                f"  Text records: {comparison['text_records']}\n"
                f"  JSON records: {comparison['json_records']}\n"
                f"  Differences: {comparison['differences']}",
            )
            self.assertGreater(
                comparison["text_records"],
                0,
                "Cross-format test: text format has zero records",
            )
            self.assertGreater(
                comparison["json_records"],
                0,
                "Cross-format test: JSON format has zero records",
            )
        finally:
            os.unlink(text_backup.name)
