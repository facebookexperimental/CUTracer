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
from tests.test_base import PLAN_CAPTURE_NDJSON, REG_TRACE_NDJSON


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

    def test_mem_addr_trace_reserves_dsm_attribution_fields(self):
        schema = SCHEMAS_BY_TYPE["mem_addr_trace"]
        for field in (
            "static_memory_space",
            "cluster_shared_mask",
            "issuer_cluster_rank",
            "target_cluster_ranks",
            "cluster_attribution",
        ):
            self.assertIn(field, schema["properties"])
            self.assertNotIn(field, schema["required"])

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
            "grid_launch_id": 7,
            "trace_instr_categories": ["sync", "tma"],
            "trace_launch_id": 7,
            "trace_warp_mask": 240,
            "trace_pc_offsets": [256, 512],
            "trace_pc_filter_valid": True,
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
            "grid_launch_id": 8,
            "cpu_callstack": [
                "cudaLaunchKernel+0x1a2",
                "main+0x45",
            ],
        }
        jsonschema.validate(valid_record, KERNEL_METADATA_SCHEMA)

    def test_valid_kernel_metadata_with_cluster_dimensions_passes(self):
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
            "cluster_dim": [2, 1, 1],
            "cluster_size": 2,
            "cluster_dim_source": "launch_attribute",
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

    def test_instrument_ipoints_accepts_only_resolved_positions(self):
        schema = KERNEL_METADATA_SCHEMA["properties"]["instrument_ipoints"]
        for value in ({}, {"reg_trace": "BEFORE"}, {"reg_trace": "AFTER"}):
            with self.subTest(value=value):
                jsonschema.validate(value, schema)
        for value in (
            {"reg_trace": "DEFAULT"},
            {"reg_trace": "before"},
            {"reg_trace": None},
            {"reg_trace": 0},
            {"other": "BEFORE"},
        ):
            with (
                self.subTest(value=value),
                self.assertRaises(jsonschema.ValidationError),
            ):
                jsonschema.validate(value, schema)

    def test_targeted_trace_selector_fields(self):
        """Test the selector values the runtime writes into every header."""
        base_record = {
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
            "trace_instr_categories": None,
            "trace_launch_id": None,
            "trace_warp_mask": None,
            "trace_pc_offsets": None,
            "trace_pc_filter_valid": True,
        }
        for fields in (
            {},
            {
                "trace_instr_categories": ["sync", "tma"],
                "trace_launch_id": 2**64 - 2,
                "trace_warp_mask": 0xFFFFFFFF,
                "trace_pc_offsets": [0x100, 0x200],
            },
            {"trace_instr_categories": []},
            {"trace_pc_filter_valid": False},
        ):
            with self.subTest(fields=fields):
                jsonschema.validate({**base_record, **fields}, KERNEL_METADATA_SCHEMA)
        for fields in (
            {"trace_instr_categories": ["ldst"]},
            {"trace_launch_id": -1},
            {"trace_warp_mask": 2**32},
            {"trace_pc_offsets": [0x100, 0x100]},
            {"trace_pc_filter_valid": None},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaises(jsonschema.ValidationError),
            ):
                jsonschema.validate({**base_record, **fields}, KERNEL_METADATA_SCHEMA)

    def test_runtime_identity_launch_record(self):
        """Test the launch record attached when runtime identity is enabled."""
        identity = {
            "process": "1" * 64,
            "context": "2" * 64,
            "module": "3" * 64,
            "function": "4" * 64,
            "entry": "_Z6kernelPfS_i",
            "cubin_sha256": "5" * 64,
            "launch_id": 7,
            "kernel_ordinal": 0,
            "launch": "6" * 64,
            "grid": [128, 1, 1],
            "block": [256, 1, 1],
            "cluster_dim": [1, 1, 1],
            "cluster_dim_source": "unknown",
            "stream_handle": 0,
            "launch_kind": "graph",
            "graph_node_handle": 94196315275016,
            "graph_invocation": 3,
            "capture_selected": True,
            "instrumentation_protocol": "legacy/v1",
        }
        record = {
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
        for fields in ({}, {"launch_kind": "direct", "graph_invocation": None}):
            with self.subTest(fields=fields):
                jsonschema.validate(
                    {**record, "runtime_identity": {**identity, **fields}},
                    KERNEL_METADATA_SCHEMA,
                )
        missing = dict(identity)
        del missing["launch"]
        for invalid in (
            missing,
            {**identity, "process": "not-a-digest"},
            {**identity, "launch_kind": "unknown"},
            {**identity, "handle": 1},
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(jsonschema.ValidationError),
            ):
                jsonschema.validate(
                    {**record, "runtime_identity": invalid}, KERNEL_METADATA_SCHEMA
                )


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


def plan_capture():
    """One launch of an instrumentation plan capture, in file order."""
    with PLAN_CAPTURE_NDJSON.open() as source:
        return [json.loads(line) for line in source]


def plan_record(kind):
    """The first record of a type in the plan capture; observations carry R5."""
    return next(
        r
        for r in plan_capture()
        if r["type"] == kind and (kind != "plan_observation" or r["registers"])
    )


class PlanCaptureSchemaTest(unittest.TestCase):
    """Tests for the records of an instrumentation plan capture."""

    def assert_invalid(self, record):
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(record, SCHEMAS_BY_TYPE[record["type"]])

    def test_plan_capture_records_pass(self):
        for record in plan_capture():
            with self.subTest(type=record["type"]):
                jsonschema.validate(record, SCHEMAS_BY_TYPE[record["type"]])

    def test_plan_header_does_not_mix_with_legacy_capture_config(self):
        header = plan_capture()[0]
        header["capture_config"]["instr_begin"] = 0
        self.assert_invalid(header)
        header = plan_capture()[0]
        header["capture_config"]["applied_plan_required"] = False
        self.assert_invalid(header)

    def test_plan_observation_lanes_and_words(self):
        observation = plan_record("plan_observation")
        for registers in (
            {"R5": ["0x2a"] * 31},
            {"R5": ["0x2A"] + [None] * 31},
            {"R5": ["0x100000000"] + [None] * 31},
            {"X5": [None] * 32},
        ):
            with self.subTest(registers=registers):
                self.assert_invalid({**observation, "registers": registers})
        self.assert_invalid({**observation, "selected_mask": "0x09"})

    def test_complete_plan_capture_needs_complete_configs(self):
        completion = plan_record("plan_capture_completion")
        completion["configs"][0]["status"] = "incomplete"
        self.assert_invalid(completion)
        completion["complete"] = False
        jsonschema.validate(completion, SCHEMAS_BY_TYPE["plan_capture_completion"])
        completion = plan_record("plan_capture_completion")
        completion["counters_available"] = False
        self.assert_invalid(completion)

    def test_plan_applied_rejects_unknown_observations(self):
        applied = plan_record("plan_applied")
        point = applied["configs"][0]["configuration"]["points"][0]
        point["observations"] = [{"kind": "reg_trace", "registers": []}]
        self.assert_invalid(applied)
        point["observations"] = [{"kind": "branch_trace"}]
        self.assert_invalid(applied)


if __name__ == "__main__":
    unittest.main()
