# CUTracer Python Module

The `cutracer` Python package is the `cutracer` command and the Python APIs behind it.

## Overview

The package captures traces by running a program under `cutracer.so`; reads, queries and validates the traces; drives random-delay stress, oracles and reduction; runs NVIDIA Compute Sanitizer; and hosts the automated diagnosis service behind `cutracer diagnose`. It does not contain `cutracer.so`: build that from the repository root as the [main README](https://github.com/facebookexperimental/CUTracer#installation) describes.

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

`pip install cutracer` installs the latest release from PyPI instead, and `pip install --pre cutracer` the nightly build.

### Optional CLP Support

`cutracer query` on `.clp` archives needs the `clp` extra. Its `yscope_clp_core` dependency only ships x86_64 wheels, so the extra cannot be installed on aarch64:

```bash
pip install ".[clp]"
```

## Package Layout

| Module | Contents |
|--------|----------|
| `cutracer.cli` | Entry point of the `cutracer` command |
| `cutracer.runner` | `cutracer trace`: finds `cutracer.so`, sets its environment variables and runs the target |
| `cutracer.query` | Trace readers (NDJSON, Zstd, CLP), `cutracer query`, the SASS dumper behind `cutracer sass`, and the warp summary |
| `cutracer.validation` | JSON Schemas and the validators behind `cutracer validate` and `cutracer compare` |
| `cutracer.analyze` | The `cutracer analyze` group (`warp-summary`) |
| `cutracer.stress` | `cutracer stress`: delay-injection search with control runs and verdicts |
| `cutracer.oracle` | `cutracer oracle`: runs a pytest node as a stress or reduce oracle |
| `cutracer.reduce` | `cutracer reduce`: delta debugging of delay configs |
| `cutracer.compute_sanitizer` | `cutracer compute-sanitizer` and the racecheck log parser |
| `cutracer.service` | The automated diagnosis service behind `cutracer diagnose` and the `diagnose-*` commands |
| `cutracer.debugger` | Hang analysis of `cuda-gdb` samples: warp states and the instruction each warp is blocked on |
| `cutracer.isa` | Instruction decoders, including the AMDGPU replay helpers in `cutracer.isa.amdgcn` |
| `cutracer.watchdog` | Recognizes runs that CUTracer's own watchdog ended |
| `cutracer.kernel_config` | Kernel-level configuration extracted from trace metadata |
| `cutracer.file_utils`, `cutracer.runtime_version`, `cutracer.types`, `cutracer.shared_vars`, `cutracer.cutracer_logger` | Shared helpers: file identities, the runtime version, type definitions, build detection and logging |

## Usage

### CLI

After installation, the `cutracer` command traces, queries, validates and analyzes CUDA kernels. The [main README](https://github.com/facebookexperimental/CUTracer#commands) lists every command, and `cutracer <command> --help` shows its flags:

```bash
# Trace a CUDA application (needs cutracer.so; run from the repository root or pass --cutracer-so)
cutracer trace --cutracer-so /path/to/CUTracer/lib/cutracer.so -i reg_trace -- python test.py

# Validate trace files
cutracer validate kernel_trace.ndjson

# Query and filter trace data
cutracer query trace.ndjson --filter "warp=24"

# Analyze traces
cutracer analyze warp-summary trace.ndjson

# Search for a delay config that triggers a bug
cutracer stress -k my_kernel -o /tmp/stress -- ./oracle.sh

# Run NVIDIA Compute Sanitizer
cutracer compute-sanitizer --tool racecheck -- python test.py
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

Trace capture, stress, replay and Compute Sanitizer runs also have typed configs for driving CUTracer without its command line: `cutracer.runner.InstrumentationConfig`, `cutracer.stress.stress.StressConfig`, `cutracer.reduce.reduce.ReplayConfig` and `cutracer.compute_sanitizer.run.ComputeSanitizerConfig`. The diagnosis service uses them.

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

[Outputs and File Formats](https://github.com/facebookexperimental/CUTracer/blob/main/docs/Outputs-and-File-Formats.md) documents every record type and field. The two most common formats look like this.

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
