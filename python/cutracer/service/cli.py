"""Local automated-diagnosis command integrated into the CUTracer CLI."""

from __future__ import annotations

import tempfile
from typing import Callable, Optional

import click
from cutracer.runtime_version import get_runtime_version
from cutracer.service.contracts import SessionReport, WorkUnit
from cutracer.service.runner import (
    FileReportSink,
    FileSessionStore,
    LocalExperimentExecutor,
    LocalRuntimeConfig,
    run_local_session,
)

RunLocalFn = Callable[..., SessionReport]


def run_diagnosis(
    target: str,
    *,
    out: Optional[str],
    state: Optional[str],
    out_dir: Optional[str],
    compute_sanitizer: Optional[str],
    cutracer_so: Optional[str],
    toolchain_version: str,
    run_local_fn: RunLocalFn,
) -> SessionReport:
    """Load a work unit and drive the shared session loop on this host."""
    with open(target) as fh:
        unit = WorkUnit.from_json(fh)

    artifact_dir = out_dir
    if artifact_dir is None:
        artifact_dir = tempfile.mkdtemp(prefix="cutracer_diagnose_")
        click.echo(f"cutracer diagnose: writing artifacts to {artifact_dir}", err=True)

    executor = LocalExperimentExecutor(
        LocalRuntimeConfig(
            out_dir=artifact_dir,
            cutracer_so=cutracer_so,
            compute_sanitizer=compute_sanitizer,
            cutracer_version=get_runtime_version(),
            toolchain_version=toolchain_version,
        )
    )
    store = None if state is None else FileSessionStore(state)
    report = run_local_fn(unit, executor=executor, sink=None, store=store)
    if out is not None:
        FileReportSink(out).publish(report)
    return report


@click.command(name="diagnose")
@click.argument("target", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--local",
    is_flag=True,
    help="Run on this GPU host (the default; retained for explicit callers).",
)
@click.option(
    "--distributed",
    is_flag=True,
    help="Use a configured distributed deployment adapter.",
)
@click.option("--out", type=click.Path(), default=None)
@click.option("--state", type=click.Path(), default=None)
@click.option("--out-dir", type=click.Path(), default=None)
@click.option("--compute-sanitizer", type=click.Path(), default=None)
@click.option(
    "--cutracer-so",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Override the CUTracer library bundled with this command.",
)
@click.option("--toolchain-version", default="")
def diagnose_command(
    target: str,
    local: bool,
    distributed: bool,
    out: Optional[str],
    state: Optional[str],
    out_dir: Optional[str],
    compute_sanitizer: Optional[str],
    cutracer_so: Optional[str],
    toolchain_version: str,
) -> None:
    """Run a bounded sanitizer, stress, trace, and reasoning session."""
    if local and distributed:
        raise click.UsageError("--local and --distributed are mutually exclusive")
    if distributed:
        raise click.ClickException(
            "distributed launch requires a configured MAST/Sandcastle deployment adapter"
        )

    report = run_diagnosis(
        target,
        out=out,
        state=state,
        out_dir=out_dir,
        compute_sanitizer=compute_sanitizer,
        cutracer_so=cutracer_so,
        toolchain_version=toolchain_version,
        run_local_fn=run_local_session,
    )
    if out is None:
        click.echo(report.to_json(indent=2))
