"""Local automated-diagnosis command integrated into the CUTracer CLI."""

from __future__ import annotations

import tempfile
from typing import Callable, Optional

import click
from cutracer.runner import resolve_cutracer_so
from cutracer.service.artifacts import local_artifact, runtime_identity_is_missing
from cutracer.service.contracts import SessionReport, WorkUnit
from cutracer.service.runner import (
    FileReportSink,
    FileSessionStore,
    LocalExperimentExecutor,
    LocalRuntimeConfig,
    run_local_session,
)

RunLocalFn = Callable[..., SessionReport]


def _cutracer_runtime_identity(cutracer_so: str) -> str:
    artifact = local_artifact(cutracer_so, include_sha256=True)
    if artifact.sha256 is None:
        raise click.ClickException(
            f"could not identify the CUTracer runtime library: {cutracer_so}"
        )
    return f"sha256:{artifact.sha256}"


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
    toolchain_identity = toolchain_version.strip()
    if runtime_identity_is_missing(toolchain_identity):
        raise click.ClickException(
            "--toolchain-version must identify the immutable target toolchain"
        )

    with open(target) as fh:
        unit = WorkUnit.from_json(fh)

    artifact_dir = out_dir
    if artifact_dir is None:
        artifact_dir = tempfile.mkdtemp(prefix="cutracer_diagnose_")
        click.echo(f"cutracer diagnose: writing artifacts to {artifact_dir}", err=True)

    resolved_cutracer_so = resolve_cutracer_so(cutracer_so)

    executor = LocalExperimentExecutor(
        LocalRuntimeConfig(
            out_dir=artifact_dir,
            cutracer_so=resolved_cutracer_so,
            compute_sanitizer=compute_sanitizer,
            cutracer_version=_cutracer_runtime_identity(resolved_cutracer_so),
            toolchain_version=toolchain_identity,
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
@click.option(
    "--toolchain-version",
    required=True,
    help="Immutable identity of the target CUDA/Triton toolchain.",
)
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
