"""Local automated-diagnosis command integrated into the CUTracer CLI."""

from __future__ import annotations

import datetime
import os
import tempfile
from typing import Callable, Optional

import click
from cutracer.runner import resolve_cutracer_so
from cutracer.service.artifacts import local_artifact, runtime_identity_is_missing
from cutracer.service.baseline import FileBaselineStore
from cutracer.service.contracts import SessionReport, WorkUnit
from cutracer.service.policy import DEFAULT_LOCAL_SANITIZER_TOOLS, InitialCampaignPolicy
from cutracer.service.runner import (
    FileReportSink,
    FileSessionStore,
    LocalExperimentExecutor,
    LocalRuntimeConfig,
    run_local_session,
)
from cutracer.service.suppression import load_suppression_policy, SuppressionPolicy
from cutracer.service.triage import TriagePolicy

RunLocalFn = Callable[..., SessionReport]


def _default_baseline_dir() -> str:
    state_root = os.environ.get("XDG_STATE_HOME")
    if not state_root:
        state_root = os.path.join(os.path.expanduser("~"), ".local", "state")
    return os.path.join(state_root, "cutracer", "baselines")


def _load_cli_suppression_policy(path: Optional[str]) -> SuppressionPolicy:
    if path is None:
        return SuppressionPolicy()
    try:
        return load_suppression_policy(path)
    except (OSError, ValueError) as error:
        raise click.ClickException(
            f"could not load suppression policy {path}: {error}"
        ) from error


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
    sanitizer_tools: tuple[str, ...] = DEFAULT_LOCAL_SANITIZER_TOOLS,
    baseline_dir: Optional[str] = None,
    suppression_policy_path: Optional[str] = None,
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
    triage_policy = TriagePolicy(
        baselines=FileBaselineStore(baseline_dir or _default_baseline_dir()),
        suppression=_load_cli_suppression_policy(suppression_policy_path),
        today=datetime.date.today().isoformat(),
    )
    report = run_local_fn(
        unit,
        executor=executor,
        campaign_policy=InitialCampaignPolicy(sanitizer_tools=sanitizer_tools),
        triage_policy=triage_policy,
        sink=None,
        store=store,
    )
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
@click.option(
    "--baseline-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Persistent per-scope baseline directory (defaults to the user state dir).",
)
@click.option(
    "--suppression-policy",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="JSON policy whose ordered rules replace the built-in suppressions.",
)
@click.option("--compute-sanitizer", type=click.Path(), default=None)
@click.option(
    "--sanitizer-tool",
    "sanitizer_tools",
    type=click.Choice(DEFAULT_LOCAL_SANITIZER_TOOLS),
    multiple=True,
    default=DEFAULT_LOCAL_SANITIZER_TOOLS,
    show_default=True,
    help="Compute Sanitizer tools in the initial campaign; repeat to select a subset.",
)
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
    baseline_dir: Optional[str],
    suppression_policy: Optional[str],
    compute_sanitizer: Optional[str],
    sanitizer_tools: tuple[str, ...],
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
        sanitizer_tools=sanitizer_tools,
        baseline_dir=baseline_dir,
        suppression_policy_path=suppression_policy,
    )
    if out is None:
        click.echo(report.to_json(indent=2))
