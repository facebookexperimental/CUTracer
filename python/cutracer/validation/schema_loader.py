# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
JSON Schema definitions for CUTracer trace formats.

This module loads JSON Schema definitions from external .json files for validating
NDJSON trace files produced by CUTracer. Each schema corresponds to a specific
message type.

Schema files are located in the 'schemas/' subdirectory:
- reg_trace.schema.json: Schema for register trace records
- mem_trace.schema.json: Schema for memory access trace records
- opcode_only.schema.json: Schema for opcode-only trace records
- cuda_gdb_opcode_only.schema.json: Schema for cuda-gdb opcode_only records
"""

import sys
from pathlib import Path
from typing import Any

from tritonparse._json_compat import load, loads

if sys.version_info >= (3, 11):
    import importlib.resources as resources
else:
    import importlib_resources as resources


def _load_schema(schema_name: str) -> dict[str, Any]:
    """
    Load a JSON schema from the schemas directory.

    Args:
        schema_name: Name of the schema file (without .schema.json extension)

    Returns:
        Parsed JSON schema as a dictionary

    Raises:
        FileNotFoundError: If schema file does not exist
        json.JSONDecodeError: If schema file contains invalid JSON
    """
    # Try using importlib.resources first (for installed packages)
    try:
        schema_files = resources.files("cutracer.validation.schemas")
        schema_path = schema_files.joinpath(f"{schema_name}.schema.json")
        schema_text = schema_path.read_text(encoding="utf-8")
        return loads(schema_text)
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        # Fall back to file system loading (for development)
        pass

    # Fallback: load from file system relative to this module
    schema_dir = Path(__file__).parent / "schemas"
    schema_file = schema_dir / f"{schema_name}.schema.json"

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    with open(schema_file, "r", encoding="utf-8") as f:
        return load(f)


# Load schemas from JSON files
REG_INFO_SCHEMA: dict[str, Any] = _load_schema("reg_trace")
MEM_ACCESS_SCHEMA: dict[str, Any] = _load_schema("mem_trace")
OPCODE_ONLY_SCHEMA: dict[str, Any] = _load_schema("opcode_only")
DEBUGGER_OPCODE_ONLY_SCHEMA: dict[str, Any] = _load_schema("cuda_gdb_opcode_only")
DELAY_CONFIG_SCHEMA: dict[str, Any] = _load_schema("delay_config")
KERNEL_METADATA_SCHEMA: dict[str, Any] = _load_schema("kernel_metadata")

# Mapping from type field to schema (for trace records with "type" field)
SCHEMAS_BY_TYPE: dict[str, dict[str, Any]] = {
    "reg_trace": REG_INFO_SCHEMA,
    "mem_trace": MEM_ACCESS_SCHEMA,
    "opcode_only": OPCODE_ONLY_SCHEMA,
    "kernel_metadata": KERNEL_METADATA_SCHEMA,
}
