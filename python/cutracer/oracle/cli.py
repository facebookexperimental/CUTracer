# Copyright (c) Meta Platforms, Inc. and affiliates.

"""``cutracer oracle``: test adapters for ``cutracer stress`` and ``reduce``.

The oracle is a standalone script (``pytest_oracle.py``, shipped as a
resource) so the project's interpreter can run it without CUTracer; these
commands run it in-process or write it out. ``cutracer_numerics.py``, also a
resource, is the numerics capture the process under test imports.
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
_NUMERICS_CAPTURE = "cutracer_numerics.py"


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


@contextlib.contextmanager
def numerics_capture_path() -> Iterator[Path]:
    """The numerics capture script, which the process under test imports as
    `cutracer_numerics`. Its file stays in place while the context is open."""
    source = resources.files("cutracer.oracle").joinpath(_NUMERICS_CAPTURE)
    with resources.as_file(source) as path:
        yield path


@contextlib.contextmanager
def loaded_oracle() -> Iterator[ModuleType]:
    """`pytest_oracle()`, pointed at the numerics capture resource (an
    exported oracle finds the capture beside itself instead)."""
    with pytest_oracle() as oracle, numerics_capture_path() as capture:
        # A module attribute the script reads when no --numerics-capture is given.
        setattr(oracle, "NUMERICS_CAPTURE", capture)  # noqa: B010
        yield oracle


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
        with loaded_oracle() as oracle:
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
@click.option(
    "--numerics",
    type=click.Choice(["record", "check"]),
    default=None,
    help="check: a large change of a tensor the command compares counts as a "
    "reproduction; record: write the golden from plain runs.",
)
@click.option(
    "--golden",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="The numerics golden directory.",
)
@click.option(
    "--numerics-runs",
    default=None,
    metavar="N",
    help="record: plain runs in all, the first recording (default: 3).",
)
@click.option(
    "--numerics-wrap",
    multiple=True,
    metavar="MODULE:FUNCTION",
    help="Also compare this function's tensor results; repeatable.",
)
@click.option(
    "--numerics-strict",
    is_flag=True,
    default=False,
    help="check: count any change of a tensor stable in every plain run.",
)
@click.option(
    "--no-pin-autotune",
    is_flag=True,
    default=False,
    help="record: do not pin Triton autotuning to a first plain run's choices.",
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
    numerics: str | None,
    golden: Path | None,
    numerics_runs: str | None,
    numerics_wrap: tuple[str, ...],
    numerics_strict: bool,
    no_pin_autotune: bool,
    command: tuple[str, ...],
) -> None:
    """Run any command as a stress/reduce oracle.

    Exit 0 when the command exits with a fail code (and its output matches
    --fail-pattern, if given), 1 when it exits with a pass code, and 2 for
    anything else.
    """
    # Only forward what the user gave: the standalone script owns the
    # defaults and all validation, so they live in exactly one place.
    valued = (
        ("--fail-codes", fail_codes),
        ("--pass-codes", pass_codes),
        ("--fail-pattern", fail_pattern),
        ("--kernels", kernels),
        ("--output-dir", output_dir),
        ("--signature-file", signature_file),
        ("--timeout", timeout),
        ("--numerics", numerics),
        ("--golden", golden),
        ("--numerics-runs", numerics_runs),
    )
    argv = [
        part
        for flag, value in valued
        if value is not None
        for part in (flag, str(value))
    ]
    argv += [
        flag
        for flag, on in (
            ("--plain", plain),
            ("--numerics-strict", numerics_strict),
            ("--no-pin-autotune", no_pin_autotune),
        )
        if on
    ]
    argv += [f"--numerics-wrap={entry}" for entry in numerics_wrap]
    argv += ["--", *command]
    try:
        with loaded_oracle() as oracle:
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
    that runs it. The numerics capture (`--numerics`) is written beside it,
    as cutracer_numerics.py.
    """
    if path.name == _NUMERICS_CAPTURE:
        raise click.BadParameter(
            f"{_NUMERICS_CAPTURE} is the capture's own name", param_hint="PATH"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    files = resources.files("cutracer.oracle")
    for name, target, mode in (
        (_NUMERICS_CAPTURE, path.with_name(_NUMERICS_CAPTURE), 0o644),
        (_PYTEST_ORACLE, path, 0o755),
    ):
        tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        tmp.write_text(files.joinpath(name).read_text())
        tmp.chmod(mode)
        os.replace(tmp, target)
    click.echo(str(path))
