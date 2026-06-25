# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""``cutracer sanitize`` — run a target under NVIDIA Compute Sanitizer.

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
import shutil
import subprocess
import sys
from pathlib import Path

import click

# compute-sanitizer --tool choices. racecheck is the default because the
# analyze-side ingest (``--sanitizer-log``) currently consumes racecheck.
_TOOLS = ["racecheck", "memcheck", "initcheck", "synccheck"]


def resolve_compute_sanitizer(override: str | None) -> str:
    """Locate the ``compute-sanitizer`` binary: explicit override, else PATH.

    Not bundled (it is a multi-file ~42MB toolkit directory, and the buck CUDA
    platform ships none) — a system install on PATH is the supported default.
    Raises ``ClickException`` with guidance when it cannot be found.
    """
    if override is not None:
        if not Path(override).exists():
            raise click.ClickException(
                f"--compute-sanitizer path does not exist: {override}"
            )
        return override
    found = shutil.which("compute-sanitizer")
    if found is None:
        raise click.ClickException(
            "compute-sanitizer not found on PATH. Install the CUDA toolkit's "
            "Compute Sanitizer and put `compute-sanitizer` on PATH, or pass "
            "--compute-sanitizer <path>."
        )
    return found


def build_sanitizer_argv(
    sanitizer_bin: str,
    tool: str,
    cmd: list[str],
    log_file: str | None = None,
) -> list[str]:
    """Build the compute-sanitizer argv (pure — spawns nothing).

    Factored out so the command construction is unit-testable without a real
    sanitizer binary. Flags mirror the validated wrapper in
    ``meta_triton_sanitizer_plan.md`` §5: follow child processes, check TMA
    tensor ops (off by default, but the Hopper/Blackwell hot path), and report
    only explicit API errors.
    """
    argv = [
        sanitizer_bin,
        "--tool",
        tool,
        "--target-processes",
        "all",
        "--check-tensor-ops",
        "yes",
        "--report-api-errors",
        "explicit",
    ]
    if log_file is not None:
        argv += ["--log-file", log_file]
    return argv + list(cmd)


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
    name="sanitize",
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
    "instead of the shell `VAR=val cmd` form — sanitize runs argv directly, "
    "not through a shell.",
)
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED, required=True)
def sanitize_command(
    tool: str,
    output_dir: Path | None,
    compute_sanitizer: str | None,
    env_pairs: tuple[str, ...],
    cmd: tuple[str, ...],
) -> None:
    """Run a command under NVIDIA Compute Sanitizer (capture mode).

    \b
    Examples:
      cutracer sanitize --tool racecheck -- python my_test.py
      cutracer sanitize --tool memcheck -o /tmp/sani -- ./my_kernel
      cutracer sanitize --env CUDA_VISIBLE_DEVICES=0 -- python my_test.py

    The captured log feeds ``cutracer analyze data-race --sanitizer-log``.
    NVBit (``cutracer trace``) and compute-sanitizer cannot share a process,
    so this is a separate run from tracing.
    """
    if cmd and cmd[0].startswith("-"):
        raise click.UsageError(
            f"First token of the wrapped command looks like a flag: {cmd[0]!r}. "
            "Put `sanitize` options before `--` and the target command after it."
        )

    sanitizer_bin = resolve_compute_sanitizer(compute_sanitizer)

    log_file: str | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(output_dir / f"{tool}.log")

    argv = build_sanitizer_argv(sanitizer_bin, tool, list(cmd), log_file=log_file)
    env = _parse_env(env_pairs)

    click.echo(f"Running: {' '.join(argv)}", err=True)
    result = subprocess.run(argv, env=env)
    sys.exit(result.returncode)
