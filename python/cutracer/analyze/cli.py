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


class _AnalyzeGroup(click.Group):
    """Click group that runs a default subcommand for a bare trace path.

    ``cutracer analyze`` is the unified entry point for schedule-sensitive
    concurrency-defect analysis. When the first argument is a trace file
    rather than a known subcommand, we inject ``default_command`` so that
    ``cutracer analyze <trace>`` runs the default detector bundle instead of
    erroring with "no such command". Explicit subcommands, options (e.g.
    ``--help``), and the no-argument case (which still shows usage/help) are
    all left untouched.

    ``default_command`` stays ``None`` in open-source builds, where the
    concurrency detectors are not shipped; it is set to ``"all"`` only when the
    internal detectors are registered below, so this shared file carries no
    hard dependency on the internal ``all`` command.
    """

    default_command: Optional[str] = None

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if (
            self.default_command is not None
            and args
            and not args[0].startswith("-")
            and args[0] not in self.commands
        ):
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


@click.group(name="analyze", cls=_AnalyzeGroup)
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

    # Group records by warp and get all records per group. Records without a
    # "warp" field (e.g. the leading kernel_metadata header, or callstack/
    # launch metadata) must be excluded: otherwise they get bucketed under a
    # None key, and the integer warp-id handling in compute_warp_summary would
    # treat the whole trace as unusable.
    warp_records = (r for r in reader.iter_records() if r.get("warp") is not None)
    grouper = StreamingGrouper(warp_records, "warp")
    groups = grouper.all_per_group()

    if not groups:
        raise click.ClickException("No records found in trace file.")

    summary = compute_warp_summary(groups)
    if summary is None:
        raise click.ClickException(
            "No warp records found in trace file "
            "(no records with an integer 'warp' field)."
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

    # Unified entry point: a bare ``cutracer analyze <trace>`` (no subcommand)
    # runs the full schedule-sensitive concurrency-defect bundle (deadlock +
    # data-race), equivalent to ``analyze all``. Explicit subcommands remain
    # available as single-detector opt-ins / back-compat filters.
    analyze_command.default_command = "all"
    analyze_command.help = (
        (analyze_command.help or "").rstrip()
        + "\n\nDefault: running 'cutracer analyze <trace>' with no subcommand "
        "runs all applicable schedule-sensitive concurrency-defect detectors "
        "(deadlock + data-race) together and labels each finding by bug class "
        "(equivalent to 'analyze all'). Pass a subcommand to run a single "
        "detector."
    )
