# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Report generator for reduction results.

Generates human-readable and JSON reports from reduction results.
"""

from datetime import datetime
from typing import Any, Optional

from cutracer.reduce.reduce import ReduceResult
from tritonparse._json_compat import dump


def generate_report(
    result: ReduceResult,
    config_path: str,
    test_script: str,
    source_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate a detailed report from reduction results.

    Args:
        result: The reduction result.
        config_path: Path to the original config file.
        test_script: Path to the test script used.
        source_path: Optional path to source code.

    Returns:
        Dictionary containing the full report.
    """
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config_file": str(config_path),
            "test_script": str(test_script),
            "source_file": source_path,
        },
        "summary": {
            "total_points": result.total_points,
            "essential_points": len(result.essential_points),
            "iterations": result.iterations,
        },
        "essential_delay_points": [
            {
                "kernel_key": p.kernel_key,
                "kernel_name": p.kernel_name,
                "pc_offset": p.pc_offset,
                "sass": p.sass,
                "delay_ns": p.delay_ns,
            }
            for p in result.essential_points
        ],
        "minimal_config_path": result.minimal_config_path,
    }

    return report


def save_report(report: dict[str, Any], output_path: str) -> None:
    """
    Save report to a JSON file.

    Args:
        report: The report dictionary.
        output_path: Path to save the report.
    """
    with open(output_path, "w") as f:
        dump(report, f, indent=True)


def format_report_text(report: dict[str, Any]) -> str:
    """
    Format report as human-readable text.

    Args:
        report: The report dictionary.

    Returns:
        Formatted text string.
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append(" CUTRACER REDUCE REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Metadata
    # pyrefly: ignore [bad-argument-type]
    lines.append(f"Timestamp: {report['metadata']['timestamp']}")
    # pyrefly: ignore [bad-argument-type]
    lines.append(f"Config: {report['metadata']['config_file']}")
    # pyrefly: ignore [bad-argument-type]
    lines.append(f"Test script: {report['metadata']['test_script']}")
    if report["metadata"].get("source_file"):
        # pyrefly: ignore [bad-argument-type]
        lines.append(f"Source: {report['metadata']['source_file']}")
    lines.append("")

    # Summary
    summary = report["summary"]
    lines.append("SUMMARY")
    lines.append("-" * 40)
    # pyrefly: ignore [bad-argument-type]
    lines.append(f"  Total points tested: {summary['total_points']}")
    # pyrefly: ignore [bad-argument-type]
    lines.append(f"  Essential points found: {summary['essential_points']}")
    # pyrefly: ignore [bad-argument-type]
    lines.append(f"  Iterations: {summary['iterations']}")
    lines.append("")

    # Essential points
    essential = report.get("essential_delay_points", [])
    if essential:
        lines.append("ESSENTIAL DELAY POINTS")
        lines.append("-" * 40)
        for i, point in enumerate(essential, 1):
            # pyrefly: ignore [bad-argument-type]
            lines.append(f"  {i}. {point['sass']}")
            # pyrefly: ignore [bad-argument-type]
            lines.append(f"     PC: {point['pc_offset']}")
            # pyrefly: ignore [bad-argument-type]
            lines.append(f"     Kernel: {point['kernel_name']}")
            lines.append("")

    # Minimal config
    if report.get("minimal_config_path"):
        # pyrefly: ignore [bad-argument-type]
        lines.append(f"Minimal config saved to: {report['minimal_config_path']}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
