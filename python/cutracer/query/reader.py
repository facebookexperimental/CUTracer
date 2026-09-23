# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Trace reader for CUTracer analysis.

This module provides the TraceReader class for reading and iterating
over trace records from NDJSON files (plain or Zstd-compressed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

from cutracer.types import TraceRecord
from cutracer.validation.compression import detect_compression, open_trace_file
from tritonparse._json_compat import JSONDecodeError, loads


def _parse_single_filter(expr: str) -> Callable[[TraceRecord], bool]:
    """
    Parse a single 'field=value' filter expression and return a predicate.

    Values are automatically converted to int if possible.

    Args:
        expr: Single filter expression (e.g., "warp=24", "type=mem_trace")

    Returns:
        Predicate function that takes a record and returns bool

    Raises:
        ValueError: If the filter expression is invalid
    """
    if "=" not in expr:
        raise ValueError(
            f"Invalid filter expression: '{expr}'. Expected format: 'field=value'"
        )

    field, value = expr.split("=", 1)
    field = field.strip()
    value = value.strip()

    if not field:
        raise ValueError("Filter field name cannot be empty")

    # Try to convert value to int (supports hex with 0x, octal with 0o, binary with 0b)
    try:
        int_value: Any = int(value, 0)
        # Match both string and int representations for backward compatibility
        return (
            lambda record: record.get(field) == value or record.get(field) == int_value
        )
    except ValueError:
        pass

    # Try JSON parsing for complex types (list/dict), e.g. cta=[0,0,0]
    try:
        json_value = loads(value)
        if isinstance(json_value, (list, dict)):
            return lambda record: record.get(field) == json_value
    except (JSONDecodeError, ValueError):
        pass

    return lambda record: record.get(field) == value


def parse_filter_expr(filter_expr: str) -> Callable[[TraceRecord], bool]:
    """
    Parse a filter expression and return a predicate function.

    Supports simple equality filters like "field=value" and
    semicolon-separated multiple conditions combined with AND logic.

    Semicolons are used as the separator (not commas) because field values
    may contain commas (e.g., "cta=[5, 0, 0]").

    Args:
        filter_expr: Filter expression. Can be a single condition
            (e.g., "warp=24") or semicolon-separated multiple conditions
            (e.g., "pc=0x1030;warp=64").

    Returns:
        Predicate function that takes a record and returns bool

    Raises:
        ValueError: If any filter expression is invalid

    Examples:
        >>> pred = parse_filter_expr("warp=24")
        >>> pred({"warp": 24})
        True
        >>> pred = parse_filter_expr("pc=0x100;warp=24")
        >>> pred({"pc": 256, "warp": 24})
        True
        >>> pred({"pc": 256, "warp": 25})
        False
    """
    parts = [p.strip() for p in filter_expr.split(";") if p.strip()]

    if not parts:
        raise ValueError("Filter expression cannot be empty")

    if len(parts) == 1:
        return _parse_single_filter(parts[0])

    predicates = [_parse_single_filter(p) for p in parts]
    return lambda record: all(p(record) for p in predicates)


def build_filter_predicate(
    filter_exprs: tuple[str, ...],
) -> Callable[[TraceRecord], bool]:
    """
    Build a combined AND predicate from multiple filter expressions.

    Each expression is parsed via parse_filter_expr (which itself supports
    semicolon-separated conditions). Multiple expressions are combined
    with AND logic.

    Args:
        filter_exprs: One or more filter expressions (e.g., from multiple -f flags)

    Returns:
        A single predicate function that returns True only when all conditions match

    Raises:
        ValueError: If any filter expression is invalid
    """
    predicates = [parse_filter_expr(expr) for expr in filter_exprs]

    if len(predicates) == 1:
        return predicates[0]

    def combined(record: TraceRecord) -> bool:
        return all(p(record) for p in predicates)

    return combined


def select_records(
    records: Iterator[TraceRecord],
    head: Optional[int] = None,
    tail: Optional[int] = None,
) -> list[TraceRecord]:
    """
    Memory-efficient record selection using streaming.

    This function processes records in a streaming fashion:
    - For head: Uses itertools.islice to stop early after N records
    - For tail: Uses collections.deque(maxlen=N) to keep only last N records

    Memory complexity:
    - head N: O(N) - only stores N records
    - tail N: O(N) - deque automatically discards older records

    Note: The records iterator is consumed after calling this function.

    Args:
        records: Iterator of trace records (consumed once!)
        head: Number of records from the beginning (default: 10)
        tail: Number of records from the end (overrides head)

    Returns:
        Selected subset of records as a list
    """
    if tail is not None:
        if tail <= 0:
            return []
        # deque with maxlen automatically discards oldest items
        return list(deque(records, maxlen=tail))

    # Default head to 10 if not specified
    head = head if head is not None else 10
    if head <= 0:
        return []
    # islice stops iteration after head items
    return list(islice(records, head))


def _resolve_record_references(
    record: TraceRecord,
    sass_table: dict[str, str],
    callstack_table: dict[str, list[str]],
) -> None:
    """Resolve SASS and caller references without replacing inline values."""
    if "sass" not in record and "opcode_id" in record and sass_table:
        opcode_key = str(record["opcode_id"])
        if opcode_key in sass_table:
            record["sass"] = sass_table[opcode_key]

    # The innermost frame is the direct call site. It may still be framework
    # code (e.g., torch dispatch), depending on how the callstack was captured.
    if (
        record.get("type") == "kernel_launch"
        and "callstack_id" in record
        and callstack_table
        and "caller" not in record
    ):
        frames = callstack_table.get(record["callstack_id"])
        if frames:
            record["caller"] = frames[-1]


class TraceReaderBase(ABC):
    """
    Reader for CUTracer trace files.
    Example:
        >>> reader = TraceReader("trace.ndjson.zst")
        >>> for record in reader.iter_records():
        ...     print(record["sass"])
    """

    @abstractmethod
    def __init__(self, file_path: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def iter_records(
        self, filter_exprs: Optional[tuple[str, ...]] = None
    ) -> Iterator[TraceRecord]:
        pass


class TraceReader(TraceReaderBase):
    """
    Reader for CUTracer trace files.

    Supports NDJSON format with optional Zstd compression.
    Provides efficient iteration over trace records.

    Example:
        >>> reader = TraceReader("trace.ndjson.zst")
        >>> for record in reader.iter_records():
        ...     print(record["sass"])
    """

    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Initialize the TraceReader.

        Args:
            file_path: Path to the trace file (.ndjson or .ndjson.zst)

        Raises:
            FileNotFoundError: If the file does not exist
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.compression = detect_compression(self.file_path)
        self.capture_completion: TraceRecord | None = None
        self.capture_completion_count = 0

    def iter_records(
        self, filter_exprs: Optional[tuple[str, ...]] = None
    ) -> Iterator[TraceRecord]:
        """
        Iterate over all trace records in the file.

        If the file contains a kernel_metadata record with an "instructions"
        table, the SASS string is injected into each subsequent record via
        opcode_id lookup, so downstream consumers always see a "sass" field
        regardless of whether it was stored inline (legacy) or in the
        instruction table (new format).

        Completion records are retained in `capture_completion` and counted
        separately, without yielding them as instructions. This property is
        raw metadata; consuming it does not verify capture completeness.

        Yields:
            dict: Each trace record as a dictionary

        Raises:
            json.JSONDecodeError: If a line contains invalid JSON

        Example:
            >>> reader = TraceReader("trace.ndjson")
            >>> for record in reader.iter_records():
            ...     process(record)
        """
        # Maps opcode_id (str) -> sass string, populated from kernel_metadata
        sass_table: dict[str, str] = {}
        # Maps callstack_id -> frames list, populated from callstack_def
        callstack_table: dict[str, list[str]] = {}
        self.capture_completion = None
        self.capture_completion_count = 0

        with open_trace_file(self.file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = loads(line)

                record_type = record.get("type")

                if record_type == "capture_completion":
                    self.capture_completion = record
                    self.capture_completion_count += 1
                    continue

                # Cache instruction table from kernel_metadata
                if record_type == "kernel_metadata":
                    instructions = record.get("instructions", {})
                    for opcode_id, info in instructions.items():
                        if isinstance(info, dict) and "sass" in info:
                            sass_table[opcode_id] = info["sass"]

                # Cache callstack definitions (don't yield — metadata only)
                if record_type == "callstack_def":
                    cid = record.get("callstack_id")
                    frames = record.get("frames")
                    if cid and isinstance(frames, list):
                        callstack_table[cid] = frames
                    continue

                _resolve_record_references(record, sass_table, callstack_table)
                yield record
