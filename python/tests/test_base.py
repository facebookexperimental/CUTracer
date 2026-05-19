# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Base test class and shared test data for CUTracer validation tests.
"""

import shutil
import tempfile
import unittest
from pathlib import Path


# Example inputs directory
EXAMPLE_INPUTS_DIR = Path(__file__).parent / "example_inputs"

# Real data file paths
REG_TRACE_NDJSON = EXAMPLE_INPUTS_DIR / "reg_trace_sample.ndjson"
REG_TRACE_NDJSON_ZST = EXAMPLE_INPUTS_DIR / "reg_trace_sample.ndjson.zst"
REG_TRACE_LOG = EXAMPLE_INPUTS_DIR / "reg_trace_sample.log"
KERNEL_EVENTS_NDJSON = EXAMPLE_INPUTS_DIR / "kernel_events_sample.ndjson"
INVALID_SYNTAX_NDJSON = EXAMPLE_INPUTS_DIR / "invalid_syntax.ndjson"
INVALID_SCHEMA_NDJSON = EXAMPLE_INPUTS_DIR / "invalid_schema.ndjson"

# CLP archive test files
CLP_ARCHIVE_SAMPLE = EXAMPLE_INPUTS_DIR / "sample.clp"

# Expected record counts for sample files
REG_TRACE_NDJSON_RECORD_COUNT = 100
REG_TRACE_NDJSON_ZST_RECORD_COUNT = 100
REG_TRACE_LOG_RECORD_COUNT = 67
CLP_ARCHIVE_RECORD_COUNT = 100
CLP_ARCHIVE_REG_TRACE_COUNT = 80
CLP_ARCHIVE_MEM_TRACE_COUNT = 20


def count_records_of_type(path: Path, record_type: str) -> int:
    """
    Count NDJSON records in `path` whose "type" field equals `record_type`.

    Used to keep tests decoupled from hardcoded fixture sizes — adding a
    new record line to a fixture should not silently break unrelated tests.
    """
    import json

    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if json.loads(line).get("type") == record_type:
                n += 1
    return n


class BaseValidationTest(unittest.TestCase):
    """Base class for validation tests with shared setup/teardown."""

    @classmethod
    def setUpClass(cls):
        """Verify example input files exist."""
        required_files = [
            REG_TRACE_NDJSON,
            REG_TRACE_LOG,
            INVALID_SYNTAX_NDJSON,
            INVALID_SCHEMA_NDJSON,
        ]
        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(f"Example input file not found: {path}")

    def setUp(self):
        """Create a temporary directory for test files that need to be generated."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def create_temp_file(self, filename: str, content: str) -> Path:
        """Create a temporary file with given content."""
        filepath = self.temp_dir / filename
        filepath.write_text(content)
        return filepath
