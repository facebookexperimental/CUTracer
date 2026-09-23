# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Cross-format consistency checker for CUTracer traces.

This module provides functions to compare trace files in different formats
(text vs NDJSON) for data consistency. Supports Zstd-compressed files.

Because Mode 0 (text) and Mode 2 (NDJSON) are produced by separate GPU
executions, per-record ordering is non-deterministic due to GPU warp
scheduling. Instead of per-record comparison, this module uses statistical
aggregation (record counts, unique CTA/warp/SASS sets) to verify that
both code paths serialize the same logical data correctly.
"""

import re
from pathlib import Path
from typing import Any, Dict, Set, Tuple, Union

from tritonparse._json_compat import JSONDecodeError, loads

from .compression import detect_compression, open_trace_file
from .text_validator import MEM_ACCESS_HEADER_PATTERN, REG_INFO_HEADER_PATTERN


def compare_record_counts(
    text_metadata: Dict[str, Any], json_metadata: Dict[str, Any], tolerance: float = 0.0
) -> bool:
    """
    Compare record counts between text and JSON formats.

    Args:
        text_metadata: Metadata from validate_text_trace()
        json_metadata: Metadata from validate_json_trace()
        tolerance: Allowed difference as fraction (default 0%)

    Returns:
        True if counts are within tolerance

    Raises:
        ValueError: If metadata is invalid or missing required fields
    """
    if "record_count" not in text_metadata:
        raise ValueError("text_metadata missing 'record_count' field")
    if "record_count" not in json_metadata:
        raise ValueError("json_metadata missing 'record_count' field")

    text_count = text_metadata["record_count"]
    json_count = json_metadata["record_count"]

    if text_count == 0 and json_count == 0:
        return True

    if text_count == 0 or json_count == 0:
        return False

    # Calculate relative difference
    max_count = max(text_count, json_count)
    diff = abs(text_count - json_count)
    relative_diff = diff / max_count

    return relative_diff <= tolerance


def _extract_json_stats(
    json_file: Path,
) -> Dict[str, Any]:
    """
    Extract statistical summary from an NDJSON trace file.

    Supports Zstd-compressed files via open_trace_file().
    Handles both legacy traces (inline "sass" per record) and new traces
    where SASS is stored in the kernel_metadata "instructions" table and
    looked up via opcode_id.

    Returns:
        Dictionary with type_counts, unique_ctas, unique_warps, unique_sass.
    """
    type_counts: Dict[str, int] = {}
    unique_ctas: Set[Tuple[int, ...]] = set()
    unique_warps: Set[int] = set()
    unique_sass: Set[str] = set()

    # Maps opcode_id (str) -> sass string, populated from kernel_metadata
    sass_table: Dict[str, str] = {}

    with open_trace_file(json_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = loads(line)

            msg_type = record.get("type", "unknown")
            if msg_type == "kernel_metadata":
                # Cache instruction table for sass lookup by opcode_id
                instructions = record.get("instructions", {})
                for opcode_id, info in instructions.items():
                    if isinstance(info, dict) and "sass" in info:
                        sass_table[opcode_id] = info["sass"]
                continue
            if msg_type == "capture_completion":
                # Completion describes the capture, not an executed instruction.
                # compare_trace_formats() validates its schema before comparison.
                continue

            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
            if "cta" in record:
                unique_ctas.add(tuple(record["cta"]))
            if "warp" in record:
                unique_warps.add(record["warp"])
            if "sass" in record:
                unique_sass.add(record["sass"])
            elif "opcode_id" in record and sass_table:
                sass = sass_table.get(str(record["opcode_id"]))
                if sass:
                    unique_sass.add(sass)

    return {
        "type_counts": type_counts,
        "unique_ctas": unique_ctas,
        "unique_warps": unique_warps,
        "unique_sass": unique_sass,
    }


# Regex with capture groups for extracting fields from reg_info header lines
_REG_HEADER_FIELDS = re.compile(
    r"^CTX\s+0x[0-9a-fA-F]+\s+-\s+CTA\s+(\d+),(\d+),(\d+)\s+-\s+"
    r"warp\s+(\d+)\s+-\s+(.+):$"
)

# Regex with capture groups for extracting fields from mem_access header lines
_MEM_HEADER_FIELDS = re.compile(
    r"^CTX\s+0x[0-9a-fA-F]+\s+-\s+kernel_launch_id\s+\d+\s+-\s+"
    r"CTA\s+(\d+),(\d+),(\d+)\s+-\s+warp\s+(\d+)\s+-\s+PC\s+\d+\s+-\s+(.+):$"
)


def _extract_text_stats(
    text_file: Path,
) -> Dict[str, Any]:
    """
    Extract statistical summary from a text-format trace file.

    Returns:
        Dictionary with type_counts, unique_ctas, unique_warps, unique_sass.
    """
    type_counts: Dict[str, int] = {}
    unique_ctas: Set[Tuple[int, ...]] = set()
    unique_warps: Set[int] = set()
    unique_sass: Set[str] = set()

    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            m = _REG_HEADER_FIELDS.match(line)
            if m:
                cx, cy, cz, warp, sass = m.groups()
                type_counts["reg_trace"] = type_counts.get("reg_trace", 0) + 1
                unique_ctas.add((int(cx), int(cy), int(cz)))
                unique_warps.add(int(warp))
                unique_sass.add(sass.strip())
                continue

            m = _MEM_HEADER_FIELDS.match(line)
            if m:
                cx, cy, cz, warp, sass = m.groups()
                type_counts["mem_trace"] = type_counts.get("mem_trace", 0) + 1
                unique_ctas.add((int(cx), int(cy), int(cz)))
                unique_warps.add(int(warp))
                unique_sass.add(sass.strip())

    return {
        "type_counts": type_counts,
        "unique_ctas": unique_ctas,
        "unique_warps": unique_warps,
        "unique_sass": unique_sass,
    }


def compare_trace_content(
    text_file: Union[str, Path],
    json_file: Union[str, Path],
) -> Dict[str, Any]:
    """
    Compare trace content using statistical aggregation.

    Because Mode 0 (text) and Mode 2 (NDJSON) come from separate GPU
    executions, per-record ordering is non-deterministic. This function
    compares aggregate statistics that must be identical across runs of
    the same kernel binary:
      - Record counts by type
      - Set of unique CTA coordinates
      - Set of unique warp IDs
      - Set of unique SASS instructions

    Supports Zstd-compressed JSON files.

    Args:
        text_file: Path to text trace file
        json_file: Path to NDJSON trace file (supports .ndjson.zst)

    Returns:
        Dictionary containing:
            - consistent: bool - Whether statistics are consistent
            - differences: List[str] - Differences found
            - text_stats: Dict - Statistics extracted from text file
            - json_stats: Dict - Statistics extracted from JSON file

    Raises:
        FileNotFoundError: If either file does not exist
    """
    text_file = Path(text_file)
    json_file = Path(json_file)

    if not text_file.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    result: Dict[str, Any] = {
        "consistent": True,
        "differences": [],
        "text_stats": {},
        "json_stats": {},
    }

    try:
        text_stats = _extract_text_stats(text_file)
        json_stats = _extract_json_stats(json_file)
        result["text_stats"] = text_stats
        result["json_stats"] = json_stats

        # 1. Compare record counts by type
        if text_stats["type_counts"] != json_stats["type_counts"]:
            result["differences"].append(
                f"Type counts mismatch: text={text_stats['type_counts']}, "
                f"json={json_stats['type_counts']}"
            )
            result["consistent"] = False

        # 2. Compare unique CTA set
        if text_stats["unique_ctas"] != json_stats["unique_ctas"]:
            text_only = text_stats["unique_ctas"] - json_stats["unique_ctas"]
            json_only = json_stats["unique_ctas"] - text_stats["unique_ctas"]
            result["differences"].append(
                f"CTA set mismatch: text_only={text_only}, json_only={json_only}"
            )
            result["consistent"] = False

        # 3. Compare unique warp set
        if text_stats["unique_warps"] != json_stats["unique_warps"]:
            text_only = text_stats["unique_warps"] - json_stats["unique_warps"]
            json_only = json_stats["unique_warps"] - text_stats["unique_warps"]
            result["differences"].append(
                f"Warp set mismatch: text_only={text_only}, json_only={json_only}"
            )
            result["consistent"] = False

        # 4. Compare unique SASS instruction set
        if text_stats["unique_sass"] != json_stats["unique_sass"]:
            text_only = text_stats["unique_sass"] - json_stats["unique_sass"]
            json_only = json_stats["unique_sass"] - text_stats["unique_sass"]
            result["differences"].append(
                f"SASS set mismatch: text_only={text_only}, json_only={json_only}"
            )
            result["consistent"] = False

    except Exception as e:
        result["differences"].append(f"Error during comparison: {str(e)}")
        result["consistent"] = False

    return result


def compare_trace_formats(
    text_file: Path, json_file: Path, tolerance: float = 0.0
) -> Dict[str, Any]:
    """
    Comprehensive comparison of two trace formats.

    Performs record count comparison and statistical content comparison.

    Args:
        text_file: Path to text trace file
        json_file: Path to NDJSON trace file
        tolerance: Allowed difference for record count (default: 0%)

    Returns:
        Dictionary containing:
            - consistent: bool - Overall consistency
            - record_count_match: bool - Whether counts match
            - content_match: bool - Whether statistical content matches
            - text_records: int - Number of text records
            - json_records: int - Number of JSON records
            - unique_ctas_count: int - Number of unique CTAs
            - unique_warps_count: int - Number of unique warps
            - unique_sass_count: int - Number of unique SASS instructions
            - differences: List[str] - All differences found

    Raises:
        FileNotFoundError: If either file does not exist
    """
    from .json_validator import validate_json_trace
    from .text_validator import validate_text_trace

    if not text_file.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    result: Dict[str, Any] = {
        "consistent": False,
        "record_count_match": False,
        "content_match": False,
        "text_records": 0,
        "json_records": 0,
        "unique_ctas_count": 0,
        "unique_warps_count": 0,
        "unique_sass_count": 0,
        "differences": [],
    }

    try:
        # Validate both files
        text_metadata = validate_text_trace(text_file)
        json_metadata = validate_json_trace(json_file)

        result["text_records"] = text_metadata["record_count"]
        result["json_records"] = json_metadata["record_count"]

        # Check if validation passed
        if not text_metadata["valid"]:
            result["differences"].append(
                f"Text file validation failed: {text_metadata['errors']}"
            )
            return result

        if not json_metadata["valid"]:
            result["differences"].append(
                f"JSON file validation failed: {json_metadata['errors']}"
            )
            return result

        # Compare record counts
        count_match = compare_record_counts(
            text_metadata, json_metadata, tolerance=tolerance
        )
        result["record_count_match"] = count_match

        if not count_match:
            diff = abs(result["text_records"] - result["json_records"])
            result["differences"].append(
                f"Record count mismatch: text={result['text_records']}, "
                f"json={result['json_records']}, diff={diff}"
            )

        # Compare content via statistical aggregation
        content_result = compare_trace_content(text_file, json_file)
        result["content_match"] = content_result["consistent"]

        # Populate stats counts from content result
        json_stats = content_result.get("json_stats", {})
        result["unique_ctas_count"] = len(json_stats.get("unique_ctas", set()))
        result["unique_warps_count"] = len(json_stats.get("unique_warps", set()))
        result["unique_sass_count"] = len(json_stats.get("unique_sass", set()))

        if not content_result["consistent"]:
            result["differences"].extend(content_result["differences"])

        # Overall consistency
        result["consistent"] = count_match and content_result["consistent"]

    except Exception as e:
        result["differences"].append(f"Comparison error: {str(e)}")

    return result


def get_trace_statistics(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract statistics from a trace file.

    Works with both text and JSON formats. Supports Zstd-compressed JSON files.

    Args:
        filepath: Path to trace file

    Returns:
        Dictionary containing:
            - format: str - "text" or "json"
            - compression: str - "zstd" or "none"
            - record_count: int
            - file_size: int
            - message_types: Dict[str, int] - Count by message type
            - unique_ctxs: int - Number of unique contexts
            - unique_warps: int - Number of unique warps

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not recognized
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    compression = detect_compression(filepath)

    stats: Dict[str, Any] = {
        "format": None,
        "compression": compression,
        "record_count": 0,
        "file_size": filepath.stat().st_size,
        "message_types": {},
        "unique_ctxs": set(),
        "unique_warps": set(),
    }

    # Determine format by extension
    suffixes = "".join(filepath.suffixes).lower()

    if ".ndjson" in suffixes:
        stats["format"] = "json"
        with open_trace_file(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = loads(line)
                    stats["record_count"] += 1

                    msg_type = record.get("type", "unknown")
                    stats["message_types"][msg_type] = (
                        stats["message_types"].get(msg_type, 0) + 1
                    )

                    if "ctx" in record:
                        stats["unique_ctxs"].add(record["ctx"])
                    if "warp" in record:
                        stats["unique_warps"].add(record["warp"])

                except JSONDecodeError:
                    pass

    elif filepath.suffix == ".log":
        stats["format"] = "text"
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if REG_INFO_HEADER_PATTERN.match(line):
                    stats["record_count"] += 1
                    stats["message_types"]["reg_info"] = (
                        stats["message_types"].get("reg_info", 0) + 1
                    )

                    # Extract context
                    ctx_match = re.search(r"CTX\s+(0x[0-9a-fA-F]+)", line)
                    if ctx_match:
                        stats["unique_ctxs"].add(ctx_match.group(1))

                    # Extract warp
                    warp_match = re.search(r"warp\s+(\d+)", line)
                    if warp_match:
                        stats["unique_warps"].add(int(warp_match.group(1)))

                elif MEM_ACCESS_HEADER_PATTERN.match(line):
                    stats["record_count"] += 1
                    stats["message_types"]["mem_access"] = (
                        stats["message_types"].get("mem_access", 0) + 1
                    )

                    ctx_match = re.search(r"CTX\s+(0x[0-9a-fA-F]+)", line)
                    if ctx_match:
                        stats["unique_ctxs"].add(ctx_match.group(1))

                    warp_match = re.search(r"warp\s+(\d+)", line)
                    if warp_match:
                        stats["unique_warps"].add(int(warp_match.group(1)))
    else:
        raise ValueError(f"Unknown file format: {filepath.suffix}")

    # Convert sets to counts
    stats["unique_ctxs"] = len(stats["unique_ctxs"])
    stats["unique_warps"] = len(stats["unique_warps"])

    return stats
