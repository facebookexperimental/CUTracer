# CUTracer Python Module

Python tools for CUTracer trace validation, parsing, and analysis.

## Overview

The `cutracer` Python package provides a comprehensive framework for working with CUTracer trace files. This module is designed to be:
- **Reusable**: Import and use in your own Python scripts
- **Testable**: Full unittest suite with real trace data
- **Type-safe**: Type hints and mypy compatibility
- **Extensible**: Plugin architecture for future enhancements

## Installation

### For Development

```bash
cd /path/to/CUTracer/python
pip install -e ".[dev]"
```

### For Production Use

```bash
cd /path/to/CUTracer/python
pip install .
```

### Optional CLP Support

`cutracer query` on `.clp` archives needs the `clp` extra. Its
`yscope_clp_core` dependency only ships x86_64 wheels, so the extra cannot be
installed on aarch64:

```bash
pip install ".[clp]"
```

## Features

### Trace Validation (Current)

- **JSON Validation**: Validate NDJSON trace files (mode 2) for syntax and schema compliance
- **Text Validation**: Validate text-format trace files (mode 0) for format compliance
- **Cross-Format Consistency**: Compare different trace formats for data consistency

### Planned Features

- **Trace Parsing**: Parse trace files into structured Python objects
- **Analysis Tools**: Instruction histograms, performance metrics, trace comparison
- **Format Conversion**: Convert between different trace formats
- **Compression Support**: Handle zstd-compressed traces (mode 1)

## Usage

### CLI

After installation, the `cutracer` command provides a unified interface for
tracing, querying, and analyzing CUDA traces:

```bash
# Trace a CUDA application (requires cutracer.so — see main README for build)
export CUTRACER_LIB_PATH=/path/to/CUTracer/lib
cutracer trace -i tma_trace -- python test.py

# Validate trace files
cutracer validate kernel_trace.ndjson

# Query and filter trace data
cutracer query trace.ndjson --filter "warp=24"

# Analyze traces
cutracer analyze warp-summary trace.ndjson
```

### Python API

```python
from cutracer.validation import (
    validate_json_trace,
    validate_text_trace,
    compare_trace_formats,
)

# Validate JSON trace
result = validate_json_trace("kernel_trace.ndjson")
if result["valid"]:
    print(f"✓ Valid JSON trace with {result['record_count']} records")
else:
    print(f"✗ Validation failed: {result['errors']}")

# Validate text trace
result = validate_text_trace("kernel_trace.log")
if result["valid"]:
    print(f"✓ Valid text trace")
else:
    print(f"✗ Validation failed: {result['errors']}")

# Compare two formats
result = compare_trace_formats("kernel_trace.log", "kernel_trace.ndjson")
if result["consistent"]:
    print("✓ Formats are consistent")
else:
    print(f"✗ Inconsistencies found: {result['differences']}")
```

## Module Structure

```
python/
├── cutracer/                        # Main package
│   ├── __init__.py                  # Package entry point with version
│   └── validation/                  # Validation framework
│       ├── __init__.py              # Validation API exports
│       ├── schema_loader.py         # JSON Schema loader
│       ├── json_validator.py        # JSON syntax & schema validation
│       ├── text_validator.py        # Text format validation
│       ├── consistency.py           # Cross-format consistency checks
│       └── schemas/                 # JSON Schema definitions
│           ├── __init__.py
│           ├── reg_trace.schema.json
│           ├── mem_trace.schema.json
│           ├── opcode_only.schema.json
│           └── delay_config.schema.json
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_base.py                 # Base test class and utilities
│   ├── test_schemas.py              # Schema loading tests
│   ├── test_json_validator.py       # JSON validation tests
│   ├── test_text_validator.py       # Text validation tests
│   ├── test_consistency.py          # Consistency check tests
│   └── example_inputs/              # Real trace data for tests
│       ├── reg_trace_sample.ndjson
│       ├── reg_trace_sample.log
│       ├── invalid_syntax.ndjson
│       └── invalid_schema.ndjson
├── pyproject.toml                   # Modern Python project config
└── README.md                        # This file
```

## Development

### Running Tests

```bash
cd python/

# Run all tests
python -m unittest discover -s tests -v

# Run specific test file
python -m unittest tests.test_json_validator -v
```

### Type Checking

```bash
cd python/
mypy cutracer/
```

### Code Formatting

```bash
# From project root directory
./format.sh format

# Or manually with ufmt
ufmt format python/
usort format python/
```

### Running All Checks

```bash
# Format code
./format.sh format

# Type check
mypy cutracer/

# Run tests
python -m unittest discover -s tests -v
```

## Validation Details

### JSON Trace Validation

The JSON validator checks:

- **Syntax**: Valid JSON format on each line (NDJSON)
- **Schema**: Correct field types and structure per JSON Schema
- **Required Fields**: `message_type`, `ctx`, `kernel_launch_id`, `trace_index`, `timestamp`, `sass`, etc.
- **Register Values**: Arrays of integers with proper format
- **CTA/Warp IDs**: Valid integer ranges

### Text Trace Validation

The text validator checks:

- **Format Patterns**: Correct CTX/CTA/warp header patterns
- **Register Output**: Proper hex format (e.g., `Reg0_T00: 0x...`)
- **Memory Access**: Valid memory address patterns

### Consistency Validation

The consistency validator compares:

- **Record Counts**: Same number of records in both formats
- **Content Matching**: Same kernel IDs, trace indices, SASS strings
- **Timestamp Order**: Consistent ordering between formats

## Trace Format Reference

### JSON Format (NDJSON - Mode 2)

Each line is a JSON object with the following structure:

```json
{
  "message_type": "reg_trace",
  "ctx": "0x58a0c0",
  "kernel_launch_id": 0,
  "trace_index": 0,
  "timestamp": 1762026820167834792,
  "sass": "LDC R1, c[0x0][0x28] ;",
  "pc": 0,
  "opcode_id": 0,
  "warp": 0,
  "cta": [0, 0, 0],
  "regs": [[0, 0, 0, ...]]
}
```

### Text Format (Mode 0)

Human-readable format with CTX headers and register values:

```
CTX 0x58a0c0 - CTA 0,0,0 - warp 0 - LDC R1, c[0x0][0x28] ;:
    * Reg0_T00: 0x0000000000000000  Reg0_T01: 0x0000000000000000 ...
```

## Contributing

1. Install development dependencies: `pip install -e ".[dev]"`
2. Make your changes
3. Run tests: `python -m unittest discover -s tests -v`
4. Run type checker: `mypy cutracer/`
5. Format code: `./format.sh format`
6. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Support

For issues and questions, please open an issue on the CUTracer GitHub repository.
