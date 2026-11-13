# Copyright (c) Meta Platforms, Inc. and affiliates.

"""``cutracer oracle``: test adapters for ``cutracer stress`` and ``reduce``.

The oracle is a standalone script (``pytest_oracle.py``, shipped as a
resource) so the project's interpreter can run it without CUTracer; these
commands run it in-process or write it out.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from types import ModuleType

import click

_PYTEST_ORACLE = "pytest_oracle.py"


@contextlib.contextmanager
def pytest_oracle() -> Iterator[ModuleType]:
    """The pytest oracle script, loaded as a module. Its file stays in place
    while the context is open: the oracle starts its worker from it."""
    source = resources.files("cutracer.oracle").joinpath(_PYTEST_ORACLE)
    with resources.as_file(source) as path:
        spec = importlib.util.spec_from_file_location("cutracer_pytest_oracle", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        # Dataclasses resolve their string annotations through sys.modules.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.modules.pop(spec.name, None)


@click.group(name="oracle")
def oracle_command() -> None:
    """Turn a test into a stress/reduce oracle.

    The oracle exits 0 when the bug reproduced, 1 when the test passed, and 2
    when the run measured nothing.
    """


@oracle_command.command(
    name="pytest",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def oracle_pytest_command(args: tuple[str, ...]) -> None:
    """Run one pytest node as a stress/reduce oracle (`--help` for options).

    Exit 0 when the node fails its correctness check, 1 when it passes, and 2
    for anything else.
    """
    try:
        with pytest_oracle() as oracle:
            code = oracle.main(list(args), prog="cutracer oracle pytest")
    except Exception as error:  # noqa: B902
        # Python's own exit code for an uncaught exception is 1, which a
        # search reads as a clean pass; anything unexpected must be 2.
        click.echo(f"oracle infrastructure error: {error!r}", err=True)
        code = 2
    sys.exit(code)


@oracle_command.command(name="cmd")
@click.option(
    "--fail-codes",
    default=None,
    metavar="N,...",
    help="Command exit codes that reproduce the bug (default: 1).",
)
@click.option(
    "--pass-codes",
    default=None,
    metavar="N,...",
    help="Command exit codes that pass (default: 0).",
)
@click.option(
    "--fail-pattern",
    default=None,
    metavar="REGEX",
    help="Require this regex in the command's output for a fail-code run to count.",
)
@click.option(
    "-k",
    "--kernels",
    default=None,
    help="Comma-separated kernel-name substrings; an injected run must have "
    "launched one of them with delay points.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Keep each run's evidence in a new attempt-* directory here.",
)
@click.option(
    "--signature-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Pin the first reproduction's signature here; later runs count only it.",
)
@click.option(
    "--timeout",
    default=None,
    metavar="S",
    help="End the command after this many seconds (exit 2).",
)
@click.option(
    "--plain",
    is_flag=True,
    default=False,
    help="Run without CUTracer (a preflight).",
)
@click.argument("command", nargs=-1, type=click.UNPROCESSED, required=True)
def oracle_cmd_command(
    fail_codes: str | None,
    pass_codes: str | None,
    fail_pattern: str | None,
    kernels: str | None,
    output_dir: Path | None,
    signature_file: Path | None,
    timeout: str | None,
    plain: bool,
    command: tuple[str, ...],
) -> None:
    """Run any command as a stress/reduce oracle.

    Exit 0 when the command exits with a fail code (and its output matches
    --fail-pattern, if given), 1 when it exits with a pass code, and 2 for
    anything else.
    """
    # Only forward what the user gave: the standalone script owns the
    # defaults and all validation, so they live in exactly one place.
    argv: list[str] = []
    if fail_codes is not None:
        argv += ["--fail-codes", fail_codes]
    if pass_codes is not None:
        argv += ["--pass-codes", pass_codes]
    if fail_pattern is not None:
        argv += ["--fail-pattern", fail_pattern]
    if kernels is not None:
        argv += ["--kernels", kernels]
    if output_dir is not None:
        argv += ["--output-dir", str(output_dir)]
    if signature_file is not None:
        argv += ["--signature-file", str(signature_file)]
    if timeout is not None:
        argv += ["--timeout", timeout]
    if plain:
        argv.append("--plain")
    argv += ["--", *command]
    try:
        with pytest_oracle() as oracle:
            code = oracle.main_cmd(argv, prog="cutracer oracle cmd")
    except Exception as error:  # noqa: B902
        # Python's own exit code for an uncaught exception is 1, which a
        # search reads as a clean pass; anything unexpected must be 2.
        click.echo(f"oracle infrastructure error: {error!r}", err=True)
        code = 2
    sys.exit(code)


@oracle_command.command(name="export")
@click.argument("path", type=click.Path(dir_okay=False, path_type=Path))
def oracle_export_command(path: Path) -> None:
    """Write the oracle to PATH as a standalone script.

    It needs only the standard library (and pytest in the project's
    interpreter, for pytest mode), so a search can run it as its oracle
    instead of starting CUTracer for every run: `PYTHON PATH --rootdir ...
    --nodeid ...` for one pytest node, or `PYTHON PATH cmd --fail-codes ...
    -- <command>` for any command. `--python` defaults to the interpreter
    that runs it.
    """
    text = resources.files("cutracer.oracle").joinpath(_PYTEST_ORACLE).read_text()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.chmod(0o755)
    os.replace(tmp, path)
    click.echo(str(path))
