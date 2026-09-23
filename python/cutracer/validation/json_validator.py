# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
JSON validator for CUTracer NDJSON trace files.

This module provides functions to validate NDJSON trace files produced by
CUTracer for syntax correctness and schema compliance. Supports both
uncompressed (.ndjson) and Zstd-compressed (.ndjson.zst) files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import jsonschema
from tritonparse._json_compat import JSONDecodeError, loads
from zstandard import ZstdError

from .compression import detect_compression, open_trace_file
from .schema_loader import SCHEMAS_BY_TYPE


class JsonValidationError(Exception):
    """Exception raised for JSON validation errors."""

    pass


@dataclass
class _JsonSyntaxSummary:
    valid_count: int = 0
    metadata_count: int = 0
    message_type: str | None = None
    errors: list[str] = field(default_factory=list)
    type_error: str | None = None

    def observe(self, record: Any) -> None:
        self.valid_count += 1
        if not isinstance(record, dict):
            self.type_error = self.type_error or "JSON trace record must be an object"
            return
        message_type = record.get("type")
        if message_type in ("kernel_metadata", "capture_completion"):
            self.metadata_count += 1
        elif not isinstance(message_type, str) or message_type not in SCHEMAS_BY_TYPE:
            self.type_error = (
                self.type_error or f"Unknown message type in file: {message_type}"
            )
        elif self.message_type is None:
            self.message_type = message_type


def validate_json_syntax(
    filepath: str | Path,
) -> tuple[int, list[str]]:
    """
    Validate JSON syntax line-by-line for NDJSON file.

    Supports both uncompressed (.ndjson) and Zstd-compressed (.ndjson.zst) files.

    Args:
        filepath: Path to NDJSON trace file

    Returns:
        Tuple of (valid_count, errors) where:
            - valid_count: Number of valid JSON lines parsed
            - errors: List of error messages with line numbers

    Raises:
        FileNotFoundError: If file does not exist
    """
    summary = _scan_json_syntax(Path(filepath))
    return summary.valid_count, summary.errors


def _scan_json_syntax(filepath: Path) -> _JsonSyntaxSummary:
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    summary = _JsonSyntaxSummary()

    try:
        with open_trace_file(filepath, encoding_errors="replace") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    summary.observe(loads(line))
                except JSONDecodeError as e:
                    summary.errors.append(
                        f"Line {line_num}: JSON decode error - {e.msg}"
                    )

    except (OSError, ValueError, EOFError, ZstdError) as e:
        summary.errors.append(f"File reading error: {str(e)}")

    return summary


def validate_json_schema(
    filepath: Union[str, Path],
    message_type: str = "reg_trace",
    max_errors: int = 10,
    allow_mixed_types: bool = False,
) -> bool:
    """
    Validate JSON schema against TraceRecord definition.

    Supports both uncompressed (.ndjson) and Zstd-compressed (.ndjson.zst) files.

    Args:
        filepath: Path to NDJSON trace file
        message_type: Expected message type (default: "reg_trace")
        max_errors: Maximum number of schema errors to collect (default: 10)
        allow_mixed_types: If True, validate each record against its own schema
                          based on the 'type' field. If False, all records must
                          match the specified message_type. (default: False)

    Returns:
        True if all records pass schema validation

    Raises:
        JsonValidationError: If schema validation fails (includes detailed errors)
        FileNotFoundError: If file does not exist
        ValueError: If message_type is not recognized
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if message_type not in SCHEMAS_BY_TYPE:
        valid_types = ", ".join(SCHEMAS_BY_TYPE.keys())
        raise ValueError(
            f"Unknown message type: {message_type}. Valid types: {valid_types}"
        )

    # Pre-create validators for all types if allowing mixed types
    validators = {
        msg_type: jsonschema.Draft7Validator(schema)
        for msg_type, schema in SCHEMAS_BY_TYPE.items()
    }
    default_validator = validators[message_type]

    errors: List[str] = []
    line_num = 0

    try:
        with open_trace_file(filepath, encoding_errors="replace") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = loads(line)
                except JSONDecodeError as e:
                    errors.append(f"Line {line_num}: JSON syntax error - {e.msg}")
                    if len(errors) >= max_errors:
                        break
                    continue

                # Get the record's type
                record_type = record.get("type", "")

                if allow_mixed_types:
                    # Validate against the appropriate schema for this record's type
                    if record_type not in SCHEMAS_BY_TYPE:
                        errors.append(
                            f"Line {line_num}: Unknown message type '{record_type}'"
                        )
                        if len(errors) >= max_errors:
                            break
                        continue
                    validator = validators[record_type]
                else:
                    # Check if type field matches expected type
                    if record_type != message_type:
                        errors.append(
                            f"Line {line_num}: Expected type '{message_type}', "
                            f"got '{record_type}'"
                        )
                        if len(errors) >= max_errors:
                            break
                        continue
                    validator = default_validator

                # Validate against schema
                validation_errors = list(validator.iter_errors(record))
                if validation_errors:
                    for error in validation_errors[
                        :3
                    ]:  # Show first 3 errors per record
                        field_path = ".".join(str(p) for p in error.path)
                        errors.append(
                            f"Line {line_num}: Schema error at '{field_path}': "
                            f"{error.message}"
                        )
                    if len(errors) >= max_errors:
                        break

    except Exception as e:
        errors.append(f"File reading error: {str(e)}")

    if errors:
        error_summary = "\n".join(errors)
        if len(errors) >= max_errors:
            error_summary += f"\n... (showing first {max_errors} errors)"
        raise JsonValidationError(f"Schema validation failed:\n{error_summary}")

    return True


def validate_json_trace(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Complete validation of NDJSON trace file.

    Performs both syntax and schema validation, with auto-detection of
    message type from the first record. Supports both uncompressed (.ndjson)
    and Zstd-compressed (.ndjson.zst) files.

    Args:
        filepath: Path to NDJSON trace file

    Returns:
        Dictionary containing:
            - valid: bool - Whether validation passed
            - record_count: int - Number of valid records
            - file_size: int - File size in bytes (compressed size for .zst files)
            - compression: str - Compression type ("zstd" or "none")
            - message_type: str - Detected message type
            - errors: List[str] - Error messages (empty if valid)

    Raises:
        FileNotFoundError: If file does not exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    compression = detect_compression(filepath)

    result: Dict[str, Any] = {
        "valid": False,
        "record_count": 0,
        "file_size": filepath.stat().st_size,
        "compression": compression,
        "message_type": None,
        "errors": [],
    }

    # Step 1: Validate syntax
    try:
        syntax = _scan_json_syntax(filepath)
        result["record_count"] = syntax.valid_count

        if syntax.errors:
            result["errors"].extend(syntax.errors)
            return result

        if syntax.valid_count == 0:
            result["errors"].append("No valid JSON records found in file")
            return result

    except Exception as e:
        result["errors"].append(f"Syntax validation error: {str(e)}")
        return result

    if syntax.type_error:
        result["errors"].append(syntax.type_error)
        return result
    result["message_type"] = syntax.message_type
    result["record_count"] -= syntax.metadata_count

    # Metadata-only captures still have a schema, even with no instruction type.
    try:
        validate_json_schema(
            filepath,
            message_type=syntax.message_type or "kernel_metadata",
            allow_mixed_types=True,
        )
        result["valid"] = True
    except JsonValidationError as e:
        result["errors"].append(str(e))
    except Exception as e:
        result["errors"].append(f"Schema validation error: {str(e)}")

    return result
