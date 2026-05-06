# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the analyze command group.

Provides command-line interface for trace analysis commands.
"""

from pathlib import Path
from typing import Optional

import click
from cutracer.query.grouper import StreamingGrouper
from cutracer.query.reader import TraceReader
from cutracer.query.warp_summary import (
    compute_warp_summary,
    format_warp_summary_text,
    warp_summary_to_dict,
)
from cutracer.shared_vars import is_fbcode
from tritonparse._json_compat import dumps


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
    whether each warp executed an EXIT instruction.

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

    # Group records by warp and get all records per group
    grouper = StreamingGrouper(reader.iter_records(), "warp")
    groups = grouper.all_per_group()

    if not groups:
        raise click.ClickException("No records found in trace file.")

    summary = compute_warp_summary(groups)
    if summary is None:
        raise click.ClickException(
            "Could not compute warp summary. "
            "Make sure records have integer 'warp' field."
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

# Conditionally register internal commands (fb/ modules not synced to OSS)
if is_fbcode():
    from cutracer.analyze.fb.ai.cli import all_command, deadlock_command
    from cutracer.analyze.fb.data_race.cli import data_race_command
    from cutracer.analyze.fb.dataflow.cli import mma_command, tma_command

    analyze_command.add_command(data_race_command)
    analyze_command.add_command(tma_command)
    analyze_command.add_command(mma_command)
    analyze_command.add_command(deadlock_command)
    analyze_command.add_command(all_command)
