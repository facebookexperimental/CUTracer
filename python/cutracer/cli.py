# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer CLI entry point.

Provides command-line interface for trace validation, query, and analysis.
"""

import sys
from importlib.metadata import PackageNotFoundError, version

import click
from cutracer.analyze.cli import analyze_command
from cutracer.query.cli import query_command, sass_command
from cutracer.reduce.cli import reduce_command
from cutracer.runner import trace_command
from cutracer.sanitize.cli import sanitize_command
from cutracer.validation.cli import compare_command, validate_command


def _get_package_version() -> str:
    """Get package version from metadata."""
    try:
        return version("cutracer")
    except PackageNotFoundError:
        pass
    # Buck-built binaries carry no .dist-info for importlib.metadata to read;
    # fall back to the fbpkg build stamp (internal only).
    from cutracer.shared_vars import is_fbcode

    if is_fbcode():
        from cutracer.fb.version import get_build_version

        build_version = get_build_version()
        if build_version:
            return build_version
    return "0+unknown"


EXAMPLES = """
Examples:
  cutracer trace -i tma_trace -- ./vectoradd
  cutracer trace -i tma_trace --instr-categories=tma -- python my_test.py
  cutracer validate kernel_trace.ndjson
  cutracer validate kernel_trace.ndjson.zst --verbose
  cutracer validate trace.log --format text
  cutracer query trace.ndjson --filter "warp=24"
  cutracer query trace.ndjson -f "pc=0x43d0;warp=24"
  cutracer query trace.ndjson --group-by warp --count
  cutracer analyze warp-summary trace.ndjson
"""


@click.group(epilog=EXAMPLES, invoke_without_command=True)
@click.version_option(version=_get_package_version(), prog_name="cutracer")
@click.pass_context
def main(ctx: click.Context) -> None:
    """CUTracer: CUDA trace validation, query, and analysis tools."""
    from cutracer.shared_vars import is_fbcode

    if is_fbcode():
        from cutracer.fb.usage import usage_report_logger

        usage_report_logger()

    if ctx.invoked_subcommand is None:
        raise click.UsageError("Missing command. Run 'cutracer --help' for usage.")


# Register subcommands
main.add_command(analyze_command)
main.add_command(compare_command)
main.add_command(query_command)
main.add_command(reduce_command)
main.add_command(sanitize_command)
main.add_command(sass_command)
main.add_command(trace_command)
main.add_command(validate_command)


if __name__ == "__main__":
    sys.exit(main())
