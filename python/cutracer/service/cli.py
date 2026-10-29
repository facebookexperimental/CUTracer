"""Local automated-diagnosis command integrated into the CUTracer CLI."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import importlib
import json
import os
import tempfile
import uuid
from typing import Callable, cast, Optional, Protocol

import click
from cutracer.runner import resolve_cutracer_so
from cutracer.service.artifacts import local_artifact, runtime_identity_is_missing
from cutracer.service.baseline import FileBaselineStore
from cutracer.service.contracts import (
    DEFAULT_MAX_DISCOVERED_KERNELS,
    DiagnosisReport,
    SessionReport,
    WorkUnit,
)
from cutracer.service.kernel_discovery import discover_work_units, KernelDiscoveryError
from cutracer.service.local_workflow import PartialDiagnosisError, run_local_diagnosis
from cutracer.service.policy import DEFAULT_LOCAL_SANITIZER_TOOLS, InitialCampaignPolicy
from cutracer.service.runner import (
    FileReportSink,
    LocalExperimentExecutor,
    LocalRuntimeConfig,
    run_local_session,
)
from cutracer.service.suppression import load_suppression_policy, SuppressionPolicy
from cutracer.service.triage import TriagePolicy

RunLocalFn = Callable[..., SessionReport]
_DISTRIBUTED_CLI_MODULE = "cutracer.service.fb.distributed_cli"


class _JsonOutput(Protocol):
    def to_json(self, **kwargs: object) -> str: ...


DistributedFn = Callable[..., _JsonOutput]
_COMMAND_MODE_KEY = "cutracer_diagnose_command_mode"


@dataclasses.dataclass(frozen=True)
class _LocalDiagnosisContext:
    runtime: LocalRuntimeConfig
    campaign_policy: InitialCampaignPolicy
    triage_policy: TriagePolicy


class _DiagnoseCommand(click.Command):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        try:
            separator = args.index("--")
        except ValueError:
            separator = len(args)
        ctx.meta[_COMMAND_MODE_KEY] = separator < len(args) - 1
        return super().parse_args(ctx, args)


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


def _distributed_fn(name: str) -> DistributedFn:
    try:
        module = importlib.import_module(_DISTRIBUTED_CLI_MODULE)
    except ModuleNotFoundError as error:
        if error.name is not None and (
            error.name == _DISTRIBUTED_CLI_MODULE
            or _DISTRIBUTED_CLI_MODULE.startswith(f"{error.name}.")
        ):
            raise click.ClickException(
                "this CUTracer build does not include the distributed adapter"
            ) from error
        raise
    function = getattr(module, name, None)
    if not callable(function):
        raise click.ClickException(f"distributed adapter has no {name} entry point")
    return cast(DistributedFn, function)


def _run_distributed_start(
    target: WorkUnit,
    *,
    config_path: str,
    session_id: str,
    toolchain_version: str,
    sanitizer_tools: tuple[str, ...],
    max_kernels: int,
) -> _JsonOutput:
    return _distributed_fn("start_distributed_session")(
        target,
        config_path=config_path,
        session_id=session_id,
        toolchain_version=toolchain_version,
        sanitizer_tools=sanitizer_tools,
        max_kernels=max_kernels,
    )


def _run_distributed_status(
    *,
    config_path: str,
    session_id: str,
) -> _JsonOutput:
    return _distributed_fn("status_distributed_session")(
        config_path=config_path,
        session_id=session_id,
    )


def _run_distributed_resume(
    *,
    config_path: str,
    session_id: str,
    result_limit: int,
) -> _JsonOutput:
    return _distributed_fn("resume_distributed_session")(
        config_path=config_path,
        session_id=session_id,
        result_limit=result_limit,
    )


def _emit_json(value: _JsonOutput, out: Optional[str]) -> None:
    payload = value.to_json(indent=2)
    if out is None:
        click.echo(payload)
        return
    with click.open_file(out, "w", atomic=True) as output:
        output.write(payload)
        output.write("\n")


def _validate_distributed_options(
    *,
    config_path: Optional[str],
    state: Optional[str],
    out_dir: Optional[str],
    baseline_dir: Optional[str],
    suppression_policy: Optional[str],
    compute_sanitizer: Optional[str],
    cutracer_so: Optional[str],
) -> str:
    if config_path is None:
        raise click.UsageError("--distributed requires --distributed-config")
    local_options = {
        "--state": state,
        "--out-dir": out_dir,
        "--baseline-dir": baseline_dir,
        "--suppression-policy": suppression_policy,
        "--compute-sanitizer": compute_sanitizer,
        "--cutracer-so": cutracer_so,
    }
    supplied = [name for name, value in local_options.items() if value is not None]
    if supplied:
        raise click.UsageError(
            "distributed mode does not accept local options: " + ", ".join(supplied)
        )
    return config_path


def _local_only_options(
    state: Optional[str],
    out_dir: Optional[str],
    baseline_dir: Optional[str],
    suppression_policy: Optional[str],
    compute_sanitizer: Optional[str],
    cutracer_so: Optional[str],
) -> dict[str, Optional[str]]:
    return {
        "state": state,
        "out_dir": out_dir,
        "baseline_dir": baseline_dir,
        "suppression_policy": suppression_policy,
        "compute_sanitizer": compute_sanitizer,
        "cutracer_so": cutracer_so,
    }


def _start_distributed_from_cli(
    target: WorkUnit,
    *,
    config_path: Optional[str],
    session_id: Optional[str],
    toolchain_version: str,
    sanitizer_tools: tuple[str, ...],
    max_kernels: int,
    out: Optional[str],
    local_options: dict[str, Optional[str]],
) -> None:
    resolved_config = _validate_distributed_options(
        config_path=config_path,
        state=local_options["state"],
        out_dir=local_options["out_dir"],
        baseline_dir=local_options["baseline_dir"],
        suppression_policy=local_options["suppression_policy"],
        compute_sanitizer=local_options["compute_sanitizer"],
        cutracer_so=local_options["cutracer_so"],
    )
    resolved_session_id = session_id or f"cutracer-{uuid.uuid4().hex}"
    click.echo(
        f"cutracer diagnose: distributed session {resolved_session_id}",
        err=True,
    )
    session = _run_distributed_start(
        target,
        config_path=resolved_config,
        session_id=resolved_session_id,
        toolchain_version=toolchain_version,
        sanitizer_tools=sanitizer_tools,
        max_kernels=max_kernels,
    )
    _emit_json(session, out)


def _write_diagnosis_report(
    report: SessionReport | DiagnosisReport,
    out: str,
) -> None:
    if isinstance(report, SessionReport):
        FileReportSink(out).publish(report)
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with click.open_file(out, "w", atomic=True) as output:
        output.write(report.to_json(indent=2))
        output.write("\n")


def _local_diagnosis_context(
    *,
    artifact_dir: str,
    compute_sanitizer: str | None,
    cutracer_so: str | None,
    toolchain_identity: str,
    sanitizer_tools: tuple[str, ...],
    baseline_dir: str | None,
    suppression_policy_path: str | None,
) -> _LocalDiagnosisContext:
    resolved_cutracer_so = resolve_cutracer_so(cutracer_so)
    runtime = LocalRuntimeConfig(
        out_dir=artifact_dir,
        cutracer_so=resolved_cutracer_so,
        compute_sanitizer=compute_sanitizer,
        cutracer_version=_cutracer_runtime_identity(resolved_cutracer_so),
        toolchain_version=toolchain_identity,
    )
    return _LocalDiagnosisContext(
        runtime=runtime,
        campaign_policy=InitialCampaignPolicy(sanitizer_tools=sanitizer_tools),
        triage_policy=TriagePolicy(
            baselines=FileBaselineStore(baseline_dir or _default_baseline_dir()),
            suppression=_load_cli_suppression_policy(suppression_policy_path),
            today=datetime.date.today().isoformat(),
        ),
    )


def run_diagnosis(
    target: str | WorkUnit,
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
    max_kernels: int = DEFAULT_MAX_DISCOVERED_KERNELS,
) -> SessionReport | DiagnosisReport:
    """Load a work unit and drive the shared session loop on this host."""
    toolchain_identity = toolchain_version.strip()
    if runtime_identity_is_missing(toolchain_identity):
        raise click.ClickException(
            "--toolchain-version must identify the immutable target toolchain"
        )

    unit = target if isinstance(target, WorkUnit) else _load_work_unit(target)

    artifact_dir = out_dir
    if artifact_dir is None:
        artifact_dir = tempfile.mkdtemp(prefix="cutracer_diagnose_")
        click.echo(f"cutracer diagnose: writing artifacts to {artifact_dir}", err=True)

    context = _local_diagnosis_context(
        artifact_dir=artifact_dir,
        compute_sanitizer=compute_sanitizer,
        cutracer_so=cutracer_so,
        toolchain_identity=toolchain_identity,
        sanitizer_tools=sanitizer_tools,
        baseline_dir=baseline_dir,
        suppression_policy_path=suppression_policy_path,
    )
    try:
        report = run_local_diagnosis(
            unit,
            runtime=context.runtime,
            state=state,
            campaign_policy=context.campaign_policy,
            triage_policy=context.triage_policy,
            run_local_fn=run_local_fn,
            max_kernels=max_kernels,
            executor_factory=LocalExperimentExecutor,
            discovery_fn=discover_work_units,
        )
    except PartialDiagnosisError as error:
        if out is not None:
            _write_diagnosis_report(error.report, out)
        raise
    if out is not None:
        _write_diagnosis_report(report, out)
    return report


def _run_local_from_cli(
    target: WorkUnit,
    *,
    out: Optional[str],
    state: Optional[str],
    out_dir: Optional[str],
    baseline_dir: Optional[str],
    suppression_policy: Optional[str],
    compute_sanitizer: Optional[str],
    sanitizer_tools: tuple[str, ...],
    cutracer_so: Optional[str],
    toolchain_version: str,
    max_kernels: int,
) -> None:
    try:
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
            max_kernels=max_kernels,
        )
    except KernelDiscoveryError as error:
        raise click.ClickException(f"kernel discovery failed: {error}") from error
    except PartialDiagnosisError as error:
        if out is None:
            click.echo(error.report.to_json(indent=2))
        raise click.ClickException(str(error)) from error
    if out is None:
        click.echo(report.to_json(indent=2))


def _load_work_unit(path: str) -> WorkUnit:
    try:
        with open(path, encoding="utf-8") as source:
            return WorkUnit.from_json(source)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise click.ClickException(
            f"could not load target {path!r}: {error}"
        ) from error


def _command_work_unit(
    argv: tuple[str, ...],
    *,
    unit_id: Optional[str],
    arch: Optional[str],
    source_revision: Optional[str],
    cwd: Optional[str],
) -> WorkUnit:
    if not argv or not argv[0].strip():
        raise click.UsageError("`-- <cmd>` requires a nonempty command")
    if arch is None or not arch.strip():
        raise click.UsageError("`-- <cmd>` requires --arch")
    revision = "" if source_revision is None else source_revision.strip()
    if runtime_identity_is_missing(revision):
        raise click.UsageError("`-- <cmd>` requires an immutable --source-revision")
    identity = json.dumps(
        {"arch": arch.strip(), "argv": argv, "cwd": cwd or "", "revision": revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    resolved_id = (
        f"command-{hashlib.sha256(identity).hexdigest()[:16]}"
        if unit_id is None
        else unit_id.strip()
    )
    if not resolved_id:
        raise click.UsageError("--unit-id must not be empty")
    return WorkUnit(
        unit_id=resolved_id,
        argv=list(argv),
        workload="command",
        test_id=resolved_id,
        cwd=cwd or "",
        arch=arch.strip(),
        source_revision=revision,
    )


def _resolve_work_unit(
    inputs: tuple[str, ...],
    *,
    command_mode: bool,
    unit_id: Optional[str],
    arch: Optional[str],
    source_revision: Optional[str],
    cwd: Optional[str],
) -> WorkUnit:
    if command_mode:
        return _command_work_unit(
            inputs,
            unit_id=unit_id,
            arch=arch,
            source_revision=source_revision,
            cwd=cwd,
        )
    if len(inputs) != 1:
        raise click.UsageError("provide one target JSON or use `-- <cmd>`")
    command_options = (unit_id, arch, source_revision, cwd)
    if any(value is not None for value in command_options):
        raise click.UsageError(
            "--unit-id, --arch, --source-revision, and --cwd require `-- <cmd>`"
        )
    return _load_work_unit(inputs[0])


@click.command(name="diagnose", cls=_DiagnoseCommand)
@click.argument("inputs", nargs=-1, type=click.UNPROCESSED)
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
@click.option(
    "--distributed-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Versioned distributed deployment configuration "
        "(defaults to CUTRACER_DISTRIBUTED_CONFIG in distributed mode)."
    ),
)
@click.option(
    "--session-id",
    default=None,
    help="Stable distributed session id; generated when omitted.",
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
@click.option("--unit-id", default=None, help="Stable id for a `-- <cmd>` target.")
@click.option(
    "--max-kernels",
    type=click.IntRange(min=1),
    default=DEFAULT_MAX_DISCOVERED_KERNELS,
    show_default=True,
    help="Maximum discovered kernels to fan out into pinned sessions.",
)
@click.option("--arch", default=None, help="GPU architecture for a `-- <cmd>` target.")
@click.option(
    "--source-revision",
    default=None,
    help="Immutable source revision for a `-- <cmd>` target.",
)
@click.option("--cwd", default=None, help="Working directory for a `-- <cmd>` target.")
@click.pass_context
def diagnose_command(
    ctx: click.Context,
    inputs: tuple[str, ...],
    local: bool,
    distributed: bool,
    distributed_config: Optional[str],
    session_id: Optional[str],
    out: Optional[str],
    state: Optional[str],
    out_dir: Optional[str],
    baseline_dir: Optional[str],
    suppression_policy: Optional[str],
    compute_sanitizer: Optional[str],
    sanitizer_tools: tuple[str, ...],
    cutracer_so: Optional[str],
    toolchain_version: str,
    unit_id: Optional[str],
    max_kernels: int,
    arch: Optional[str],
    source_revision: Optional[str],
    cwd: Optional[str],
) -> None:
    """Run a bounded sanitizer, stress, trace, and reasoning session."""
    target = _resolve_work_unit(
        inputs,
        command_mode=bool(ctx.meta.get(_COMMAND_MODE_KEY)),
        unit_id=unit_id,
        arch=arch,
        source_revision=source_revision,
        cwd=cwd,
    )
    if local and distributed:
        raise click.UsageError("--local and --distributed are mutually exclusive")
    if distributed:
        _start_distributed_from_cli(
            target,
            config_path=distributed_config
            or os.environ.get("CUTRACER_DISTRIBUTED_CONFIG")
            or None,
            session_id=session_id,
            toolchain_version=toolchain_version,
            sanitizer_tools=sanitizer_tools,
            max_kernels=max_kernels,
            out=out,
            local_options=_local_only_options(
                state,
                out_dir,
                baseline_dir,
                suppression_policy,
                compute_sanitizer,
                cutracer_so,
            ),
        )
        return
    if distributed_config is not None or session_id is not None:
        raise click.UsageError(
            "--distributed-config and --session-id require --distributed"
        )

    _run_local_from_cli(
        target,
        out=out,
        state=state,
        out_dir=out_dir,
        compute_sanitizer=compute_sanitizer,
        cutracer_so=cutracer_so,
        toolchain_version=toolchain_version,
        max_kernels=max_kernels,
        sanitizer_tools=sanitizer_tools,
        baseline_dir=baseline_dir,
        suppression_policy=suppression_policy,
    )


def _distributed_config_option(
    function: Callable[..., object],
) -> Callable[..., object]:
    return click.option(
        "--distributed-config",
        type=click.Path(exists=True, dir_okay=False),
        required=True,
        envvar="CUTRACER_DISTRIBUTED_CONFIG",
        help="Versioned distributed deployment configuration.",
    )(function)


@click.command(name="diagnose-status")
@click.argument("session_id")
@_distributed_config_option
@click.option("--out", type=click.Path(), default=None)
def diagnose_status_command(
    session_id: str,
    distributed_config: str,
    out: Optional[str],
) -> None:
    """Read one durable distributed diagnosis session without mutating it."""
    _emit_json(
        _run_distributed_status(
            config_path=distributed_config,
            session_id=session_id,
        ),
        out,
    )


@click.command(name="diagnose-resume")
@click.argument("session_id")
@_distributed_config_option
@click.option(
    "--result-limit",
    type=click.IntRange(min=1),
    default=256,
    show_default=True,
)
@click.option("--out", type=click.Path(), default=None)
def diagnose_resume_command(
    session_id: str,
    distributed_config: str,
    result_limit: int,
    out: Optional[str],
) -> None:
    """Run one idempotent reconciliation pass for a distributed session."""
    _emit_json(
        _run_distributed_resume(
            config_path=distributed_config,
            session_id=session_id,
            result_limit=result_limit,
        ),
        out,
    )
