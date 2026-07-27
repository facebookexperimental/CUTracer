# Copyright (c) Meta Platforms, Inc. and affiliates.

"""``cutracer compute-sanitizer`` — run NVIDIA Compute Sanitizer.

Capture-mode (Phase 2) counterpart to ``cutracer trace``. Instead of injecting
NVBit (``cutracer.so`` via ``CUDA_INJECTION64_PATH``), it runs the target UNDER
compute-sanitizer in a separate process. NVBit and compute-sanitizer cannot
share a process, so this is always a distinct run from ``trace``; the captured
log feeds ``cutracer analyze data-race --sanitizer-log`` (the ingest side).

Execution uses an **argv list, not** ``shell=True``: the sanitizer prefix is
concatenated with the user's command as argv, avoiding the quoting / injection
risk of string-joining a vendor wrapper with user input. Because argv mode
loses the shell ``VAR=val cmd`` form that ``trace`` supports, environment
variables for the target are passed explicitly via repeatable ``--env KEY=VAL``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
from cutracer.compute_sanitizer.run import ComputeSanitizerConfig, run_compute_sanitizer

# compute-sanitizer --tool choices. racecheck is the default because the
# analyze-side ingest (``--sanitizer-log``) currently consumes racecheck.
_TOOLS = ["racecheck", "memcheck", "initcheck", "synccheck"]


def _parse_env(pairs: tuple[str, ...]) -> dict[str, str]:
    """Layer ``--env KEY=VAL`` overrides on top of the current environment."""
    env = os.environ.copy()
    for pair in pairs:
        if "=" not in pair:
            raise click.UsageError(f"--env expects KEY=VAL, got: {pair!r}")
        key, value = pair.split("=", 1)
        env[key] = value
    return env


@click.command(
    name="compute-sanitizer",
    context_settings={"ignore_unknown_options": True},
)
@click.option(
    "--tool",
    type=click.Choice(_TOOLS),
    default="racecheck",
    show_default=True,
    help="Compute Sanitizer tool to run.",
)
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write the sanitizer log (<tool>.log); created if "
    "missing. If omitted, sanitizer output goes to stdout/stderr.",
)
@click.option(
    "--compute-sanitizer",
    "compute_sanitizer",
    default=None,
    help="Path to the compute-sanitizer binary (default: search PATH). Not "
    "bundled — requires a system CUDA toolkit install.",
)
@click.option(
    "--env",
    "env_pairs",
    multiple=True,
    metavar="KEY=VAL",
    help="Set an environment variable for the target (repeatable). Use this "
    "instead of the shell `VAR=val cmd` form; compute-sanitizer capture runs "
    "argv directly, not through a shell.",
)
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED, required=True)
def compute_sanitizer_command(
    tool: str,
    output_dir: Path | None,
    compute_sanitizer: str | None,
    env_pairs: tuple[str, ...],
    cmd: tuple[str, ...],
) -> None:
    """Run a command under NVIDIA Compute Sanitizer (capture mode).

    \b
    Examples:
      cutracer compute-sanitizer --tool racecheck -- python my_test.py
      cutracer compute-sanitizer --tool memcheck -o /tmp/sani -- ./my_kernel
      cutracer compute-sanitizer --env CUDA_VISIBLE_DEVICES=0 -- python my_test.py

    The captured log feeds ``cutracer analyze data-race --sanitizer-log``.
    NVBit (``cutracer trace``) and compute-sanitizer cannot share a process,
    so this is a separate run from tracing.
    """
    if cmd and cmd[0].startswith("-"):
        raise click.UsageError(
            f"First token of the wrapped command looks like a flag: {cmd[0]!r}. "
            "Put `compute-sanitizer` options before `--` and the target "
            "command after it."
        )

    env = _parse_env(env_pairs)
    result = run_compute_sanitizer(
        ComputeSanitizerConfig(
            argv=list(cmd),
            tool=tool,
            output_dir=None if output_dir is None else str(output_dir),
            compute_sanitizer=compute_sanitizer,
            base_env=env,
            capture_output=False,
        ),
        runner=subprocess.run,
    )
    if not result.tensor_ops_enabled:
        click.echo(
            "warning: this compute-sanitizer lacks --check-tensor-ops; TMA/"
            "TensorOps hazards on Hopper+ will not be checked.",
            err=True,
        )
    sys.exit(result.process.returncode)
