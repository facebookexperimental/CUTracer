# CLP Archive Validator - Implementation Checklist

Track implementation progress by checking off items as completed.

---

## Pre-Implementation

- [ ] Confirm `clp-s` CLI availability in test environment
- [ ] Obtain/generate sample CLP archive for `example_inputs/`
- [ ] Determine expected record counts for sample file
- [ ] Review DESIGN.md and confirm approach

---

## Phase 1: CLP Detection & Test Infrastructure

**Goal**: Establish detection capability and test fixtures

### Test Fixtures
- [ ] Create/add sample CLP archive to `tests/example_inputs/sample.clp`
- [ ] Update `tests/test_base.py`:
  - [ ] Add `CLP_ARCHIVE_SAMPLE` path constant
  - [ ] Add `CLP_ARCHIVE_RECORD_COUNT` constant
  - [ ] Add `CLP_ARCHIVE_REG_TRACE_COUNT` constant
  - [ ] Add `CLP_ARCHIVE_MEM_TRACE_COUNT` constant

### Implementation
- [ ] Create `python/cutracer/validation/clp_validator.py`:
  - [ ] Add file header and docstring
  - [ ] Implement `ClpValidationError` exception class
  - [ ] Implement `detect_clp_archive(filepath) -> bool`

### Tests
- [ ] Create `python/tests/test_clp.py`:
  - [ ] Add file header and imports
  - [ ] Implement `ClpDetectionTest` class:
    - [ ] `test_detect_clp_archive_by_extension`
    - [ ] `test_detect_clp_archive_not_clp_ndjson`
    - [ ] `test_detect_clp_archive_not_clp_zst`
    - [ ] `test_detect_clp_archive_nonexistent`

### Verification
- [ ] Run: `python -m pytest tests/test_clp.py::ClpDetectionTest -v`
- [ ] All Phase 1 tests pass

---

## Phase 2: CLP Record Iteration

**Goal**: Read and iterate over records from CLP archives

**Prerequisite**: Phase 1 complete and verified

### Implementation
- [ ] Add to `python/cutracer/validation/clp_validator.py`:
  - [ ] Implement `_check_clp_tool_available() -> bool`
  - [ ] Implement `iter_clp_records(filepath) -> Iterator[str]`

### Tests
- [ ] Add to `python/tests/test_clp.py`:
  - [ ] Implement `ClpIteratorTest` class:
    - [ ] `test_clp_tool_available`
    - [ ] `test_iter_clp_records_yields_lines`
    - [ ] `test_iter_clp_records_valid_json`
    - [ ] `test_iter_clp_records_nonexistent`
    - [ ] `test_iter_clp_records_not_clp`
  - [ ] Add skip decorator for tests requiring `clp-s`

### Verification
- [ ] Run: `python -m pytest tests/test_clp.py::ClpIteratorTest -v`
- [ ] All Phase 2 tests pass (or skip appropriately if clp-s unavailable)

---

## Phase 3: CLP Validation Logic

**Goal**: Implement full validation with statistics

**Prerequisite**: Phase 2 complete and verified

### Implementation
- [ ] Add to `python/cutracer/validation/clp_validator.py`:
  - [ ] Implement `get_clp_statistics(filepath) -> dict`
  - [ ] Implement `validate_clp_archive(filepath) -> dict`

### Tests
- [ ] Add to `python/tests/test_clp.py`:
  - [ ] Implement `ClpValidationTest` class:
    - [ ] `test_validate_clp_archive_valid`
    - [ ] `test_validate_clp_archive_record_count`
    - [ ] `test_validate_clp_archive_message_types`
    - [ ] `test_validate_clp_archive_compressed_size`
    - [ ] `test_validate_clp_archive_empty`
    - [ ] `test_validate_clp_archive_invalid_json_collected`

### Verification
- [ ] Run: `python -m pytest tests/test_clp.py::ClpValidationTest -v`
- [ ] All Phase 3 tests pass

---

## Phase 4: Integration with Existing Module

**Goal**: Integrate CLP support into existing compression/validation infrastructure

**Prerequisite**: Phase 3 complete and verified

### Implementation
- [ ] Update `python/cutracer/validation/compression.py`:
  - [ ] Modify `detect_compression()` to return `"clp"` for CLP archives
  - [ ] Modify `get_trace_format()` to handle `.clp` extension

- [ ] Update `python/cutracer/validation/__init__.py`:
  - [ ] Add imports from `clp_validator`
  - [ ] Add CLP functions to `__all__` list

### Tests
- [ ] Add to `python/tests/test_clp.py`:
  - [ ] Implement `ClpCompressionIntegrationTest` class:
    - [ ] `test_detect_compression_clp`
    - [ ] `test_get_trace_format_clp`
    - [ ] `test_module_exports_clp_validation_error`
    - [ ] `test_module_exports_detect_clp_archive`
    - [ ] `test_module_exports_iter_clp_records`
    - [ ] `test_module_exports_validate_clp_archive`

### Regression Tests
- [ ] Run: `python -m pytest tests/test_compression.py -v`
- [ ] Verify existing compression tests still pass

### Verification
- [ ] Run: `python -m pytest tests/test_clp.py::ClpCompressionIntegrationTest -v`
- [ ] All Phase 4 tests pass

---

## Final Verification

- [ ] Run full CLP test suite: `python -m pytest tests/test_clp.py -v`
- [ ] Run full validation test suite: `python -m pytest tests/test_*.py -v`
- [ ] All tests pass
- [ ] Code review complete
- [ ] Documentation updated (if needed)

---

## Summary

| Phase | Status | Tests Passing |
|-------|--------|---------------|
| Pre-Implementation | ⬜ Not Started | N/A |
| Phase 1: Detection | ⬜ Not Started | ⬜ |
| Phase 2: Iteration | ⬜ Not Started | ⬜ |
| Phase 3: Validation | ⬜ Not Started | ⬜ |
| Phase 4: Integration | ⬜ Not Started | ⬜ |
| Final Verification | ⬜ Not Started | ⬜ |

**Legend**: ⬜ Not Started | 🟡 In Progress | ✅ Complete
