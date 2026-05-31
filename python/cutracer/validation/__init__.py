# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer validation module.

This module provides validation functions for trace files in different formats:
- JSON validation (syntax and schema)
- Text format validation
- Cross-format consistency checking
- Compression handling (Zstd)
"""

from .compression import (
    detect_compression,
    get_file_size,
    get_trace_format,
    iter_lines,
    open_trace_file,
)
from .consistency import (
    compare_record_counts,
    compare_trace_content,
    compare_trace_formats,
    get_trace_statistics,
)
from .json_validator import (
    JsonValidationError,
    validate_json_schema,
    validate_json_syntax,
    validate_json_trace,
)
from .schema_loader import (
    DEBUGGER_OPCODE_ONLY_SCHEMA,
    DELAY_CONFIG_SCHEMA,
    MEM_ACCESS_SCHEMA,
    MEM_ADDR_TRACE_SCHEMA,
    MEM_VALUE_TRACE_SCHEMA,
    OPCODE_ONLY_SCHEMA,
    REG_INFO_SCHEMA,
    SCHEMAS_BY_TYPE,
    TMA_TRACE_SCHEMA,
)
from .text_validator import (
    parse_text_trace_record,
    TextValidationError,
    validate_text_format,
    validate_text_trace,
)

__all__ = [
    # Compression utilities
    "detect_compression",
    "get_trace_format",
    "open_trace_file",
    "iter_lines",
    "get_file_size",
    # JSON validation
    "validate_json_syntax",
    "validate_json_schema",
    "validate_json_trace",
    "JsonValidationError",
    # Text validation
    "validate_text_format",
    "validate_text_trace",
    "parse_text_trace_record",
    "TextValidationError",
    # Cross-format consistency
    "compare_record_counts",
    "compare_trace_content",
    "compare_trace_formats",
    "get_trace_statistics",
    # Schemas
    "REG_INFO_SCHEMA",
    "MEM_ACCESS_SCHEMA",
    "MEM_VALUE_TRACE_SCHEMA",
    "MEM_ADDR_TRACE_SCHEMA",
    "TMA_TRACE_SCHEMA",
    "OPCODE_ONLY_SCHEMA",
    "DEBUGGER_OPCODE_ONLY_SCHEMA",
    "DELAY_CONFIG_SCHEMA",
    "SCHEMAS_BY_TYPE",
]
