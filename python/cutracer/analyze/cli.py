# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the analyze command group.

Provides command-line interface for trace analysis commands.
"""

from pathlib import Path
from typing import Any, Optional

import click
from cutracer.analyze.capture_scope import capture_scope_reasons
from cutracer.analyze.runtime_warp_status import runtime_status_for_trace
from cutracer.kernel_config import parse_kernel_metadata
from cutracer.query.reader import TraceReader
from cutracer.query.warp_summary import (
    compute_warp_summary,
    format_warp_summary_text,
    warp_completed,
    warp_summary_to_dict,
)
from cutracer.shared_vars import is_fbcode
from cutracer.types import TraceRecord
from tritonparse._json_compat import dumps


def _launch_scope(header: Optional[TraceRecord]) -> tuple[Optional[int], list[str]]:
    """Warps the grid launched, and the capture filters that hide some of them."""
    config = parse_kernel_metadata(header) if header is not None else None
    if config is None or config.total_ctas <= 0 or config.threads_per_cta <= 0:
        return None, []
    return config.total_ctas * config.warps_per_cta, list(
        capture_scope_reasons(header, config.warps_per_cta)
    )


def _group_warp_records(
    reader: TraceReader,
) -> tuple[dict[Any, list[TraceRecord]], Optional[TraceRecord]]:
    """One record per warp, and the kernel_metadata header that gives the grid.

    The summary only asks whether a warp exited, so each warp keeps its first
    EXIT record, else its latest record. Records without a "warp" field (the
    header, callstack or launch metadata) are excluded: bucketed under a None
    key, they would make the integer warp-id handling in compute_warp_summary
    treat the whole trace as unusable.
    """
    groups: dict[Any, list[TraceRecord]] = {}
    header: Optional[TraceRecord] = None
    for record in reader.iter_records():
        warp = record.get("warp")
        if record.get("type") == "kernel_metadata":
            header = header or record
        elif warp is not None and not warp_completed(groups.get(warp) or []):
            groups[warp] = [record]
    return groups, header


def _runtime_never_executed(
    trace: Path, header: Optional[TraceRecord], observed_warps: set[int]
) -> Optional[int]:
    """Never-executed count from the traced launch's warp status file."""
    try:
        runtime = runtime_status_for_trace(trace, header, observed_warps)
    except (OSError, ValueError) as e:
        click.echo(f"warning: ignoring warp status file: {e}", err=True)
        return None
    return runtime.never_executed_warps if runtime is not None else None


@click.group(name="analyze")
def analyze_command() -> None:
    """
    Analyze trace data for patterns and insights.

    Available subcommands:
      warp-summary      Analyze warp execution status
    """
    pass


@click.command(name="warp-summary")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: stdout).",
)
def warp_summary_command(
    file: Path,
    output_format: str,
    output_file: Optional[Path],
) -> None:
    """
    Analyze warp execution status from a trace file.

    Identifies completed, in-progress, and missing warps by analyzing
    whether each warp executed an EXIT instruction. Missing warps are counted
    against the launch's grid from the trace header, unless a capture filter
    (warp mask, instruction window, PC or category filter) hides warps that
    ran. The online detector's never-executed count is shown when its warp
    status file for the same launch sits next to the trace.

    \b
    Examples:
      cutracer analyze warp-summary trace.ndjson
      cutracer analyze warp-summary trace.ndjson --format json
      cutracer analyze warp-summary trace.ndjson -o summary.json -f json
    """
    try:
        reader = TraceReader(file)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    groups, header = _group_warp_records(reader)
    launched, scope = _launch_scope(header)
    if not groups:
        launched_note = f" (the header launched {launched} warps)" if launched else ""
        raise click.ClickException(
            f"No warp records found in trace file{launched_note}."
        )

    summary = compute_warp_summary(groups, launched_warps=None if scope else launched)
    if summary is None:
        raise click.ClickException(
            "No warp records found in trace file "
            "(no records with an integer 'warp' field)."
        )
    summary.capture_scope = scope
    summary.runtime_never_executed_warps = _runtime_never_executed(
        file, header, {warp for warp in groups if isinstance(warp, int)}
    )

    if output_format == "json":
        output = dumps(warp_summary_to_dict(summary), indent=True)
    else:
        output = format_warp_summary_text(summary)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(output + "\n")
        click.echo(f"Output written to {output_file}", err=True)
    else:
        click.echo(output)


# Register subcommands
analyze_command.add_command(warp_summary_command)

# Internal commands are absent from the exported source/package.
if is_fbcode():
    from cutracer.analyze.fb.cli import create_analyze_command

    analyze_command = create_analyze_command(warp_summary_command)
