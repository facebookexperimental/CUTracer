# CLP Archive Format Validator - Design Document

## Overview

This document describes the design for adding CLP (Compressed Log Processor) archive format validation support to CUTracer. CLP archives represent Mode 3 trace compression, containing NDJSON trace data compressed using the CLP-S format.

## Background

CUTracer supports multiple trace output formats:
- **Mode 0**: Plain NDJSON (uncompressed)
- **Mode 1**: Zstd-compressed NDJSON
- **Mode 3**: CLP archive format (highest compression ratio)

The existing validation module handles Modes 0 and 1. This design adds support for Mode 3 (CLP archives).

## Components

### 1. New File: `clp_validator.py`

**Location**: `python/cutracer/validation/clp_validator.py`

**Purpose**: Validate CLP archive files and provide iteration over contained records.

#### Classes and Functions

```python
class ClpValidationError(Exception):
    """Raised when CLP archive validation fails."""
    pass

def detect_clp_archive(filepath: Union[str, Path]) -> bool:
    """
    Detect if a file is a CLP archive.
    
    Args:
        filepath: Path to the file to check
        
    Returns:
        True if file is a CLP archive, False otherwise
        
    Raises:
        FileNotFoundError: If file does not exist
    """

def iter_clp_records(filepath: Union[str, Path]) -> Iterator[str]:
    """
    Iterate over decompressed NDJSON records from a CLP archive.
    
    Args:
        filepath: Path to the CLP archive
        
    Yields:
        NDJSON record strings (one per line)
        
    Raises:
        FileNotFoundError: If file does not exist
        ClpValidationError: If file is not a valid CLP archive
    """

def validate_clp_archive(filepath: Union[str, Path]) -> dict:
    """
    Validate a CLP archive file.
    
    Args:
        filepath: Path to the CLP archive
        
    Returns:
        Dictionary containing:
        {
            "valid": bool,
            "record_count": int,
            "message_types": dict[str, int],  # e.g., {"reg_trace": 100, "mem_trace": 50}
            "compressed_size": int,
            "errors": list[str]
        }
        
    Raises:
        FileNotFoundError: If file does not exist
    """

def get_clp_statistics(filepath: Union[str, Path]) -> dict:
    """
    Get statistics about a CLP archive without full validation.
    
    Args:
        filepath: Path to the CLP archive
        
    Returns:
        Dictionary with record counts and size information
    """
```

### 2. Updates to `compression.py`

**Modifications**:

1. **`detect_compression()`**: Add CLP detection
   ```python
   # Add CLP detection (check .clp extension)
   if filepath.suffix.lower() == ".clp":
       return "clp"
   ```

2. **`get_trace_format()`**: Handle CLP archives
   ```python
   # CLP archives contain NDJSON data
   if suffixes.endswith(".clp"):
       return ("ndjson", "clp")
   ```

### 3. Updates to `validation/__init__.py`

**New exports** (add around line 55):

```python
from .clp_validator import (
    ClpValidationError,
    detect_clp_archive,
    iter_clp_records,
    validate_clp_archive,
    get_clp_statistics,
)

__all__ = [
    # ... existing exports ...
    # CLP validation
    "ClpValidationError",
    "detect_clp_archive",
    "iter_clp_records",
    "validate_clp_archive",
    "get_clp_statistics",
]
```

### 4. New Test File: `test_clp.py`

**Location**: `python/tests/test_clp.py`

**Test Classes**:

| Class | Purpose |
|-------|---------|
| `ClpDetectionTest` | Tests for `detect_clp_archive()` |
| `ClpIteratorTest` | Tests for `iter_clp_records()` |
| `ClpValidationTest` | Tests for `validate_clp_archive()` |
| `ClpCompressionIntegrationTest` | Integration tests with `compression.py` |

### 5. Updates to `test_base.py`

**New constants**:

```python
# CLP archive test files
CLP_ARCHIVE_SAMPLE = EXAMPLE_INPUTS_DIR / "sample.clp"
CLP_ARCHIVE_RECORD_COUNT = 100  # Expected count for sample file
CLP_ARCHIVE_REG_TRACE_COUNT = 80
CLP_ARCHIVE_MEM_TRACE_COUNT = 20
```

## Dependencies

### External Tools

- **`clp-s` CLI**: Required for decompressing CLP archives
  - Detection: `shutil.which("clp-s")`
  - Usage: `clp-s s <archive> '*'` to query all records

### Fallback Strategy

If `clp-s` is unavailable:
1. Tests will be skipped with informative message
2. Functions will raise `ClpValidationError` with installation instructions

## API Design Decisions

### 1. Return Dict vs Raise Exception

**Decision**: `validate_clp_archive()` returns a dict with `valid: bool` rather than raising exceptions.

**Rationale**: Matches existing `validate_json_trace()` pattern. Allows collecting multiple errors and returning partial results.

### 2. Separate Detection Function

**Decision**: Provide `detect_clp_archive()` as separate function.

**Rationale**: Allows cheap detection without full validation. Useful for format routing.

### 3. Iterator Pattern

**Decision**: Provide `iter_clp_records()` for streaming access.

**Rationale**: CLP archives can be large. Streaming avoids loading entire archive into memory.

## Error Handling

| Error Condition | Behavior |
|-----------------|----------|
| File not found | Raise `FileNotFoundError` |
| Not a CLP archive | Raise `ClpValidationError` |
| Corrupted archive | Return `{"valid": False, "errors": [...]}` |
| `clp-s` unavailable | Raise `ClpValidationError` with install instructions |
| Invalid JSON record | Collect in errors list, continue processing |

## Test Data Requirements

### Sample CLP Archive

Need to create/obtain a sample `.clp` archive containing:
- Mixed `reg_trace` and `mem_trace` records
- Known record counts for validation
- Small enough for fast tests (~100 records)

**Generation** (from existing test infrastructure):
```bash
cd tests/py_add
TRACE_FORMAT_NDJSON=3 \
CUDA_INJECTION64_PATH="../../lib/cutracer.so" \
CUTRACER_INSTRUMENT=reg_trace,mem_trace \
python ./test_add.py
# Copy generated .clp file to example_inputs/
```

## Open Questions

1. **CLP Python bindings**: Should we use Python bindings instead of CLI subprocess?
   - Pro: Faster, no subprocess overhead
   - Con: Additional dependency, may not be available

2. **Schema validation**: Should CLP validator also validate JSON schema of records?
   - Current design: Delegates to existing `validate_json_schema()` if needed
   - Alternative: Integrate schema validation into `validate_clp_archive()`

3. **Partial decompression**: Can we validate without full decompression?
   - CLP-S supports queries; could count records without extracting all data
