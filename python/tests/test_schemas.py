# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for JSON schema definitions.
"""

import json
import unittest

import jsonschema
from cutracer.validation.schema_loader import (
    DEBUGGER_OPCODE_ONLY_SCHEMA,
    DELAY_CONFIG_SCHEMA,
    KERNEL_METADATA_SCHEMA,
    MEM_ACCESS_SCHEMA,
    OPCODE_ONLY_SCHEMA,
    REG_INFO_SCHEMA,
    SCHEMAS_BY_TYPE,
)
from tests.test_base import REG_TRACE_NDJSON


class SchemaTest(unittest.TestCase):
    """Core tests for JSON schema definitions."""

    def test_schemas_contain_all_types(self):
        """Test that SCHEMAS_BY_TYPE contains all expected message types."""
        expected_types = ["reg_trace", "mem_trace", "opcode_only", "kernel_metadata"]
        for msg_type in expected_types:
            self.assertIn(msg_type, SCHEMAS_BY_TYPE)

    def test_schemas_are_valid_json_schema(self):
        """Test that all schemas are valid JSON Schema documents."""
        for schema in SCHEMAS_BY_TYPE.values():
            jsonschema.Draft7Validator.check_schema(schema)

    def test_delay_config_schema_is_valid(self):
        """Test that delay config schema is a valid JSON Schema document."""
        jsonschema.Draft7Validator.check_schema(DELAY_CONFIG_SCHEMA)

    def test_debugger_opcode_only_schema_is_valid(self):
        """Test that the cuda-gdb opcode_only schema is valid JSON Schema."""
        jsonschema.Draft7Validator.check_schema(DEBUGGER_OPCODE_ONLY_SCHEMA)

    def test_real_reg_trace_record_passes(self):
        """Test that real reg_trace records from sample file pass schema validation."""
        with open(REG_TRACE_NDJSON, "r") as f:
            for i, line in enumerate(f):
                if i >= 10:  # Test first 10 records
                    break
                record = json.loads(line.strip())
                # Should not raise any exception
                jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_mem_trace_schema_structure(self):
        """Test that mem_trace schema has required structure."""
        self.assertIn("properties", MEM_ACCESS_SCHEMA)
        self.assertIn("type", MEM_ACCESS_SCHEMA["properties"])
        self.assertIn("addrs", MEM_ACCESS_SCHEMA["properties"])

    def test_opcode_only_schema_structure(self):
        """Test that opcode_only schema has required structure."""
        self.assertIn("properties", OPCODE_ONLY_SCHEMA)
        self.assertIn("type", OPCODE_ONLY_SCHEMA["properties"])
        # opcode_only should NOT have regs or addrs
        self.assertNotIn("regs", OPCODE_ONLY_SCHEMA.get("required", []))
        self.assertNotIn("addrs", OPCODE_ONLY_SCHEMA.get("required", []))

    def test_valid_debugger_opcode_only_record_passes(self):
        """Test that a cuda-gdb opcode_only record passes its schema."""
        valid_record = {
            "type": "opcode_only",
            "trace_index": 0,
            "cta": [0, 0, 0],
            "warp": 4,
            "cuda_warp_slot": 6,
            "pc": "0x15fb0",
            "sass": "BAR.SYNC.DEFER_BLOCKING 0x1",
            "observed_pc": "0x15fc0",
            "observed_sass": "USETMAXREG.DEALLOC.CTAPOOL 0x50",
            "source_type": "gdb_inferred",
            "pc_semantics": "effective_blocking_instruction",
            "registers": {"UR12": 5120},
            "memory_addresses": [
                {
                    "evidence_type": "hgmma_gdesc",
                    "operand": "gdesc[UR12]",
                    "shared_offsets": ["0x1400"],
                }
            ],
        }
        jsonschema.validate(valid_record, DEBUGGER_OPCODE_ONLY_SCHEMA)

    def test_debugger_opcode_only_source_kind_fails(self):
        """Test that the legacy cuda-gdb source_kind field is rejected."""
        invalid_record = {
            "type": "opcode_only",
            "trace_index": 0,
            "cta": [0, 0, 0],
            "warp": 4,
            "pc": "0x15fb0",
            "sass": "BAR.SYNC.DEFER_BLOCKING 0x1",
            "source_kind": "gdb_inferred",
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid_record, DEBUGGER_OPCODE_ONLY_SCHEMA)


class KernelMetadataSchemaTest(unittest.TestCase):
    """Tests for kernel_metadata schema."""

    def test_valid_kernel_metadata_passes(self):
        """Test that a valid kernel_metadata record passes schema validation."""
        valid_record = {
            "type": "kernel_metadata",
            "mangled_name": "_Z6kernelPfS_i",
            "unmangled_name": "kernel(float*, float*, int)",
            "kernel_checksum": "a1b2c3d4e5f67890",
            "func_addr": "0x7f1234567890",
            "nregs": 32,
            "shmem_static": 0,
            "grid": [128, 1, 1],
            "block": [256, 1, 1],
            "shmem_dynamic": 0,
        }
        jsonschema.validate(valid_record, KERNEL_METADATA_SCHEMA)

    def test_valid_kernel_metadata_with_callstack_passes(self):
        """Test that kernel_metadata with cpu_callstack passes validation."""
        valid_record = {
            "type": "kernel_metadata",
            "mangled_name": "_Z6kernelPfS_i",
            "unmangled_name": "kernel(float*, float*, int)",
            "kernel_checksum": "a1b2c3d4e5f67890",
            "func_addr": "0x7f1234567890",
            "nregs": 32,
            "shmem_static": 0,
            "grid": [128, 1, 1],
            "block": [256, 1, 1],
            "shmem_dynamic": 0,
            "cpu_callstack": [
                "cudaLaunchKernel+0x1a2",
                "main+0x45",
            ],
        }
        jsonschema.validate(valid_record, KERNEL_METADATA_SCHEMA)

    def test_unexpected_field_fails(self):
        """Test that unexpected additional properties fail validation."""
        invalid_record = {
            "type": "kernel_metadata",
            "mangled_name": "_Z6kernelPfS_i",
            "unmangled_name": "kernel(float*, float*, int)",
            "kernel_checksum": "a1b2c3d4e5f67890",
            "func_addr": "0x7f1234567890",
            "nregs": 32,
            "shmem_static": 0,
            "grid": [128, 1, 1],
            "block": [256, 1, 1],
            "shmem_dynamic": 0,
            "unknown_field": "should_fail",
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid_record, KERNEL_METADATA_SCHEMA)


class DelayConfigSchemaTest(unittest.TestCase):
    """Tests for delay injection configuration schema."""

    def test_valid_delay_config_passes(self):
        """Test that a valid delay config passes schema validation."""
        valid_config = {
            "version": "1.0",
            "delay_ns": 10000,
            "kernels": {
                "my_kernel_a1b2c3d4e5f67890": {
                    "kernel_name": "my_kernel",
                    "kernel_checksum": "a1b2c3d4e5f67890",
                    "timestamp": "2026-02-03T21:15:21.567",
                    "instrumentation_points": {
                        "10192": {
                            "pc": 10192,
                            "sass": "SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR15+0x38110], R4 ;",
                            "delay": 10000,
                            "on": True,
                        },
                    },
                }
            },
        }
        jsonschema.validate(valid_config, DELAY_CONFIG_SCHEMA)

    def test_missing_required_field_fails(self):
        """Test that missing required fields fail validation."""
        # Missing version
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({"delay_ns": 10000, "kernels": {}}, DELAY_CONFIG_SCHEMA)

    def test_invalid_instrumentation_point_fails(self):
        """Test that incomplete instrumentation point fails validation."""
        invalid_config = {
            "version": "1.0",
            "delay_ns": 10000,
            "kernels": {
                "my_kernel_2026-02-03T21:15:21.567": {
                    "kernel_name": "my_kernel",
                    "timestamp": "2026-02-03T21:15:21.567",
                    "instrumentation_points": {
                        "100": {"pc": 100, "sass": "NOP;"},  # Missing "delay" and "on"
                    },
                }
            },
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid_config, DELAY_CONFIG_SCHEMA)


if __name__ == "__main__":
    unittest.main()
