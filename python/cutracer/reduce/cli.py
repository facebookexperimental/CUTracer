# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI for CUTracer reduce command.

Provides command-line interface for reducing delay injection configurations.
"""

import logging
import os
import shlex
import sys
from typing import NoReturn

import click
from cutracer.cutracer_logger import get_logger
from cutracer.reduce.config_mutator import DelayPoint
from cutracer.reduce.reduce import (
    reduce_bisect,
    reduce_delay_points,
    ReduceConfig,
    ReduceResult,
    ReplayConfig,
)
from cutracer.reduce.report import generate_report, save_report

logger = get_logger("reduce")


def progress_callback_linear(
    current: int, total: int, point: DelayPoint, is_essential: bool
) -> None:
    """Print progress during linear reduction."""
    status = "ESSENTIAL" if is_essential else "  not essential"
    click.echo(f"[{current}/{total}] {point.sass[:50]:<50} {status}")


def progress_callback_bisect(
    message: str, current_size: int, original_size: int
) -> None:
    """Print progress during bisection reduction."""
    click.echo(f"  [{current_size}/{original_size} points] {message}")


def _exit_with_error(
    message: str, verbose: bool = False, exception: Exception | None = None
) -> NoReturn:
    """Print error message and exit."""
    logger.error(message)
    click.echo(f"Error: {message}", err=True)
    if verbose and exception:
        import traceback

        traceback.print_exc()
    sys.exit(1)


def _run_reduction(
    reduce_config: ReduceConfig,
    strategy: str,
    confidence_runs: int,
    verbose: bool,
) -> ReduceResult:
    """Run the reduction algorithm and handle errors."""
    try:
        if strategy == "bisect":
            return reduce_bisect(
                config=reduce_config,
                confidence_runs=confidence_runs,
                progress_callback=progress_callback_bisect if not verbose else None,
            )
        else:
            return reduce_delay_points(
                config=reduce_config,
                progress_callback=progress_callback_linear if not verbose else None,
            )
    except ValueError as e:
        _exit_with_error(str(e))
    except Exception as e:
        _exit_with_error(f"Unexpected error: {e}", verbose=verbose, exception=e)


@click.command(name="reduce")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to delay config JSON file (repro config).",
)
@click.option(
    "--test",
    "-t",
    required=True,
    type=str,
    help="Oracle command, parsed with shlex.split and run directly as an argv "
    "list (NOT through a shell). Exit 0 if the race occurs, 1 if it does not. "
    "Shell composition (&&, |, globs, $VAR) must live inside a wrapper script "
    "that exits 0 on manifest, e.g. the issue1996_pytest_race.sh convention.",
)
@click.option(
    "--cutracer-so",
    default=None,
    help="Path to cutracer.so (default: bundled / auto-discovered).",
)
@click.option("-k", "--kernel-filters", default=None, help="Kernel name filter.")
@click.option(
    "--timeout",
    type=click.IntRange(min=1),
    default=300,
    show_default=True,
    help="Per-replay timeout (s).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="reduce_report.json",
    help="Output path for the report JSON.",
)
@click.option(
    "--minimal-config",
    "-m",
    type=click.Path(),
    default="minimal_config.json",
    help="Output path for the minimal config.",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["linear", "bisect"]),
    default="linear",
    help="Reduction strategy: 'linear' tests each point one by one (O(N)), "
    "'bisect' uses ddmin bisection (much faster, O(N log N) worst case).",
)
@click.option(
    "--confidence-runs",
    type=int,
    default=1,
    help="Number of test runs for majority voting (bisect strategy only). "
    "Use odd numbers (3, 5) for probabilistic races (e.g., random delay mode). "
    "Default: 1 (single run, for deterministic races).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output.",
)
@click.option(
    "--no-validate-schema",
    is_flag=True,
    help="Skip JSON schema validation of the config file.",
)
def reduce_command(
    config: str,
    test: str,
    cutracer_so: str | None,
    kernel_filters: str | None,
    timeout: int,
    output: str,
    minimal_config: str,
    strategy: str,
    confidence_runs: int,
    verbose: bool,
    no_validate_schema: bool,
) -> None:
    """
    Reduce delay injection config to find minimal race trigger.

    Uses delta debugging to find the minimal set of delay points
    that trigger a data race.

    Strategies:
      - linear: Tests each point one by one. Simple but O(N) test runs.
      - bisect: ddmin-style bisection. Much faster, splits points in half
        and recursively narrows down. Use --confidence-runs for probabilistic races.

    Oracle convention (same as llvm-reduce):

    \b
      - Exit 0: Interesting (race occurred)
      - Exit 1: Not interesting (no race)
      - Other exits: Infrastructure error

    The oracle (-t) is parsed with shlex.split and run directly as an argv list,
    NOT through a shell. Shell composition (&&, |, globs, $VAR) must live inside
    a wrapper script that exits 0 on manifest (the issue1996_pytest_race.sh
    convention).

    Examples:

    \b
      cutracer reduce -c repro.json -t ./test_race.sh
      cutracer reduce -c repro.json -t ./test_race.sh --strategy bisect
      cutracer reduce -c repro.json -t ./test_race.sh -s bisect --confidence-runs 3
    """
    if verbose:
        get_logger().setLevel(logging.DEBUG)

    click.echo("=" * 60)
    click.echo(" CUTRACER REDUCE")
    click.echo("=" * 60)
    click.echo(f"Config: {config}")
    click.echo(f"Test script: {test}")
    click.echo(f"Strategy: {strategy}")
    if strategy == "bisect" and confidence_runs > 1:
        click.echo(f"Confidence runs: {confidence_runs}")
    click.echo("")

    oracle_argv = shlex.split(test)
    if not oracle_argv:
        raise click.UsageError("--test must contain a non-empty oracle command")

    # The oracle runs under this package's CUTracer runtime. It must not invoke
    # another `cutracer` binary to load the replay config.
    reduce_config = ReduceConfig(
        config_path=config,
        output_path=minimal_config,
        verbose=verbose,
        validate_schema=not no_validate_schema,
        replay=ReplayConfig(
            oracle_argv=oracle_argv,
            cutracer_so=cutracer_so,
            kernel_filters=kernel_filters,
            output_dir=os.path.dirname(os.path.abspath(minimal_config)),
            timeout=timeout,
            capture_output=not verbose,
        ),
    )

    # Run reduction
    click.echo(f"\nStarting {strategy} reduction...\n")
    result = _run_reduction(reduce_config, strategy, confidence_runs, verbose)

    # Generate report
    click.echo("\n" + "=" * 60)
    click.echo(" REDUCTION COMPLETE")
    click.echo("=" * 60)

    if result.essential_points:
        click.echo(f"\nFound {len(result.essential_points)} essential point(s):\n")
        for i, point in enumerate(result.essential_points, 1):
            click.echo(f"  {i}. {point.sass}")
            click.echo(f"     PC: {point.pc_offset}")
            click.echo(f"     Kernel: {point.kernel_name}")
            click.echo("")
    else:
        click.echo("\nNo essential points found.")
        click.echo("    This may indicate the race is not deterministically triggered")
        click.echo("    by the delay injection points in this config.")

    # Generate and save report
    report = generate_report(
        result=result,
        config_path=config,
        test_script=test,
    )
    save_report(report, output)
    click.echo(f"\nReport saved to: {output}")

    if result.minimal_config_path:
        click.echo(f"Minimal config saved to: {result.minimal_config_path}")

    # Print summary stats
    click.echo("\nStats:")
    click.echo(f"   Strategy: {result.strategy}")
    click.echo(f"   Total points: {result.total_points}")
    click.echo(f"   Minimal set: {len(result.essential_points)}")
    click.echo(f"   Iterations: {result.iterations}")

    click.echo("")
