# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the analyze command."""

import json
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner
from cutracer.cli import main
from cutracer.shared_vars import is_fbcode


class AnalyzeWarpSummaryTest(unittest.TestCase):
    """Tests for analyze warp-summary command."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test trace file with warp data
        self.trace_file = Path(self.temp_dir) / "trace.ndjson"
        with open(self.trace_file, "w") as f:
            records = [
                {"warp": 0, "pc": 100, "sass": "MOV R1, R2;"},
                {"warp": 0, "pc": 200, "sass": "ADD R1, R2, R3;"},
                {"warp": 0, "pc": 300, "sass": "EXIT;"},
                {"warp": 1, "pc": 100, "sass": "MOV R1, R2;"},
                {"warp": 1, "pc": 200, "sass": "ADD R1, R2, R3;"},
                {"warp": 2, "pc": 100, "sass": "MOV R1, R2;"},
                {"warp": 2, "pc": 200, "sass": "EXIT;"},
            ]
            for record in records:
                f.write(json.dumps(record) + "\n")

    def test_warp_summary_basic(self):
        """Test basic warp-summary command."""
        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", str(self.trace_file)],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Warp Summary", result.output)
        self.assertIn("Total warps observed", result.output)
        self.assertIn("Completed (EXIT)", result.output)
        self.assertIn("In-progress", result.output)

    def test_warp_summary_with_kernel_metadata_header(self):
        """Regression: a real trace starts with a kernel_metadata record that
        has no 'warp' field, and reg_trace records carry opcode_id (not inline
        sass). Previously the metadata record was bucketed under a None key and
        int(None) aborted the command with a misleading error. The header must
        be skipped, and sass must be resolved via the instruction table so EXIT
        detection still works.
        """
        trace_file = Path(self.temp_dir) / "trace_with_meta.ndjson"
        records = [
            {
                "type": "kernel_metadata",
                "mangled_name": "_Z3fooPf",
                "instructions": {
                    "0": {"sass": "MOV R1, R0 ;"},
                    "1": {"sass": "EXIT ;"},
                },
            },
            {"type": "reg_trace", "warp": 0, "opcode_id": 0, "pc": "0x0"},
            {"type": "reg_trace", "warp": 0, "opcode_id": 1, "pc": "0x10"},
            {"type": "reg_trace", "warp": 1, "opcode_id": 0, "pc": "0x0"},
        ]
        with open(trace_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", str(trace_file), "--format", "json"],
        )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        # Only warps 0 and 1 are counted — the kernel_metadata header is skipped.
        self.assertEqual(output["total_observed"], 2)
        # warp 0's last instruction (opcode_id 1) resolves to EXIT via the
        # instruction table -> completed; warp 1 ends on MOV -> in-progress.
        self.assertEqual(output["completed"]["count"], 1)
        self.assertEqual(output["in_progress"]["count"], 1)

    def test_warp_summary_json_format(self):
        """Test warp-summary with JSON output."""
        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", str(self.trace_file), "--format", "json"],
        )

        self.assertEqual(result.exit_code, 0)

        # Parse JSON output
        output = json.loads(result.output)
        self.assertIn("total_observed", output)
        self.assertIn("completed", output)
        self.assertIn("in_progress", output)
        self.assertIn("missing", output)

        # Verify counts
        self.assertEqual(output["total_observed"], 3)
        self.assertEqual(output["completed"]["count"], 2)  # warp 0 and 2
        self.assertEqual(output["in_progress"]["count"], 1)  # warp 1

    def test_warp_summary_output_file(self):
        """Test warp-summary with output file."""
        output_file = Path(self.temp_dir) / "summary.txt"

        result = self.runner.invoke(
            main,
            [
                "analyze",
                "warp-summary",
                str(self.trace_file),
                "-o",
                str(output_file),
            ],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Output written to", result.output)
        self.assertTrue(output_file.exists())

        content = output_file.read_text()
        self.assertIn("Warp Summary", content)

    def test_warp_summary_json_output_file(self):
        """Test warp-summary with JSON output to file."""
        output_file = Path(self.temp_dir) / "summary.json"

        result = self.runner.invoke(
            main,
            [
                "analyze",
                "warp-summary",
                str(self.trace_file),
                "--format",
                "json",
                "-o",
                str(output_file),
            ],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(output_file.exists())

        with open(output_file) as f:
            output = json.load(f)
        self.assertIn("total_observed", output)

    def test_warp_summary_nonexistent_file(self):
        """Test warp-summary with nonexistent file."""
        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", "/nonexistent/file.ndjson"],
        )

        self.assertNotEqual(result.exit_code, 0)

    def test_warp_summary_help(self):
        """Test warp-summary --help."""
        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", "--help"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Analyze warp execution status", result.output)
        self.assertIn("--format", result.output)
        self.assertIn("--output", result.output)

    def test_analyze_help(self):
        """Test analyze --help shows subcommands."""
        result = self.runner.invoke(
            main,
            ["analyze", "--help"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Analyze trace data", result.output)
        self.assertIn("warp-summary", result.output)

    def test_analyze_no_subcommand(self):
        """Test analyze without subcommand shows help."""
        result = self.runner.invoke(
            main,
            ["analyze"],
        )

        # Click 8+ returns 0 and shows help; Click 7 returns 2
        self.assertIn(result.exit_code, (0, 2))
        self.assertIn("warp-summary", result.output)


# Real Blackwell GEMM TMA descriptor (float16, 8192x8192, box_dim 128x64)
_BLACKWELL_DESC_RAW = [
    "0x00007fb61c000000",
    "0x0000040000246310",
    "0x0000000000000000",
    "0x0000000000000000",
    "0x00001fff00001fff",
    "0x0000000000000000",
    "0x3f00000000000000",
    "0x000000000000007f",
    "0x0000000000004000",
    "0x0000000000000400",
    "0x0000000000000000",
    "0x0000000000000000",
    "0x0000000000000000",
    "0x0000000000000000",
    "0x0000000000000000",
    "0x0000000000000000",
]


@unittest.skipUnless(is_fbcode(), "TMA command only available in fbcode")
class AnalyzeTMAJsonTest(unittest.TestCase):
    """Tests for analyze tma command JSON output fields."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test trace file with TMA records
        self.trace_file = Path(self.temp_dir) / "tma_trace.ndjson"
        records = [
            {
                "type": "tma_trace",
                "desc_addr": "0xABCD",
                "pc": "0x100",
                "cta": [0, 0, 0],
                "desc_raw": _BLACKWELL_DESC_RAW,
                "sass": "UTMALDG.2D [UR16], [UR10] ;",
            },
            {
                "type": "tma_trace",
                "desc_addr": "0xABCD",
                "pc": "0x200",
                "cta": [0, 0, 0],
                "desc_raw": _BLACKWELL_DESC_RAW,
                "sass": "UTMALDG.2D [UR16], [UR10] ;",
            },
            {
                "type": "tma_trace",
                "desc_addr": "0xABCD",
                "pc": "0x100",
                "cta": [1, 0, 0],
                "desc_raw": _BLACKWELL_DESC_RAW,
                "sass": "UTMALDG.2D [UR16], [UR10] ;",
            },
        ]
        with open(self.trace_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def test_tma_json_has_raw_descriptor_fields(self):
        """Test that JSON output includes raw descriptor fields."""
        result = self.runner.invoke(
            main,
            ["analyze", "tma", str(self.trace_file), "--format", "json"],
        )

        self.assertEqual(result.exit_code, 0)
        output = json.loads(result.output)
        desc = output["descriptors"][0]

        self.assertEqual(desc["tensor_rank"], 2)
        self.assertEqual(desc["data_type"], 6)
        self.assertEqual(desc["element_size"], 2)
        self.assertEqual(desc["global_dim"], [8192, 8192, 0, 0, 0])
        self.assertIsInstance(desc["global_stride"], list)
        self.assertEqual(desc["box_dim"], [128, 64, 0, 0, 0])
        self.assertEqual(desc["swizzle_mode"], 4)

    def test_tma_json_has_trace_statistics(self):
        """Test that JSON output includes trace statistics."""
        result = self.runner.invoke(
            main,
            ["analyze", "tma", str(self.trace_file), "--format", "json"],
        )

        self.assertEqual(result.exit_code, 0)
        output = json.loads(result.output)
        desc = output["descriptors"][0]

        # 2 unique PCs: 0x100 and 0x200
        self.assertEqual(desc["unique_pcs"], 2)

        # ops_per_cta stats
        ops = desc["ops_per_cta"]
        self.assertIn("min", ops)
        self.assertIn("max", ops)
        self.assertIn("avg", ops)
        self.assertIn("sample_ctas", ops)
        self.assertEqual(ops["sample_ctas"], 2)

    def test_tma_json_has_inferred_block_shape(self):
        """Test that JSON output includes inferred block shape."""
        result = self.runner.invoke(
            main,
            ["analyze", "tma", str(self.trace_file), "--format", "json"],
        )

        self.assertEqual(result.exit_code, 0)
        output = json.loads(result.output)
        desc = output["descriptors"][0]

        self.assertIn("inferred_block_shape", desc)
        self.assertIsInstance(desc["inferred_block_shape"], list)

    def test_tma_json_has_source_reconstruction(self):
        """Test that JSON output includes source reconstruction."""
        result = self.runner.invoke(
            main,
            ["analyze", "tma", str(self.trace_file), "--format", "json"],
        )

        self.assertEqual(result.exit_code, 0)
        output = json.loads(result.output)

        self.assertIn("source_reconstruction", output)
        self.assertIsInstance(output["source_reconstruction"], str)

    def test_tma_json_preserves_existing_fields(self):
        """Test that existing JSON fields are still present."""
        result = self.runner.invoke(
            main,
            ["analyze", "tma", str(self.trace_file), "--format", "json"],
        )

        self.assertEqual(result.exit_code, 0)
        output = json.loads(result.output)

        # Summary
        self.assertIn("summary", output)
        self.assertEqual(output["summary"]["total_accesses"], 3)
        self.assertEqual(output["summary"]["unique_descriptors"], 1)

        # Existing descriptor fields
        desc = output["descriptors"][0]
        self.assertEqual(desc["desc_addr"], "0xABCD")
        self.assertTrue(desc["is_load"])
        self.assertEqual(desc["dtype"], "float16")
        self.assertEqual(desc["swizzle"], "128B_ATOM_32B")
        self.assertEqual(desc["interleave"], "16B")
        self.assertEqual(desc["access_count"], 3)
        self.assertIn("shape", desc)
        self.assertIn("strides", desc)
        self.assertIn("block_shape", desc)
        self.assertIn("sass", desc)


class AnalyzeIntegrationTest(unittest.TestCase):
    """Integration tests for analyze command with compressed files."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def test_warp_summary_from_zst_file(self):
        """Test warp-summary from Zstd compressed file."""
        import zstandard as zstd

        # Create compressed test file
        trace_file = Path(self.temp_dir) / "trace.ndjson.zst"
        records = [
            {"warp": 0, "pc": 100, "sass": "EXIT;"},
            {"warp": 1, "pc": 100, "sass": "MOV R1, R2;"},
        ]

        cctx = zstd.ZstdCompressor()
        with open(trace_file, "wb") as f:
            with cctx.stream_writer(f) as writer:
                for record in records:
                    writer.write((json.dumps(record) + "\n").encode("utf-8"))

        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", str(trace_file)],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Warp Summary", result.output)


if __name__ == "__main__":
    unittest.main()
