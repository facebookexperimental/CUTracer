# Copyright (c) Meta Platforms, Inc. and affiliates.

"""``cutracer oracle``: test adapters for ``cutracer stress`` and ``reduce``.

The pytest oracle is a standalone script (``pytest_oracle.py``, shipped as a
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


@oracle_command.command(name="export")
@click.argument("path", type=click.Path(dir_okay=False, path_type=Path))
def oracle_export_command(path: Path) -> None:
    """Write the pytest oracle to PATH as a standalone script.

    It needs only the standard library (and pytest in the project's
    interpreter), so a search can run `PYTHON PATH --rootdir ... --nodeid ...`
    as its oracle instead of starting CUTracer for every run. `--python`
    defaults to the interpreter that runs it.
    """
    text = resources.files("cutracer.oracle").joinpath(_PYTEST_ORACLE).read_text()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.chmod(0o755)
    os.replace(tmp, path)
    click.echo(str(path))
