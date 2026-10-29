"""Thin Run-level commands; backend imports follow the selected profile."""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from typing import Any, cast

import click
from cutracer.service import api
from cutracer.service.checkpoint import read_json
from cutracer.service.file_run_store import FileRunControlStore
from cutracer.service.run_contracts import (
    canonical_input_digest,
    DeploymentProfile,
    RunInput,
    RunPublicationStatus,
    RunResult,
    RunStatus,
    RunSummary,
)
from cutracer.service.storage import SessionStoreError


def _backend_options(function: Callable[..., Any]) -> Callable[..., Any]:
    for name in ("state-dir", "local-config", "distributed-config", "hosted-config"):
        function = click.option(f"--{name}", type=click.Path(), default=None)(function)
    return function


def _backend(
    *,
    state_dir: str | None,
    local_config: str | None,
    distributed_config: str | None,
    hosted_config: str | None,
    request: RunInput | None,
    execute: bool,
) -> tuple[api.DiagnosisBackend, str]:
    if (state_dir is None) == (distributed_config is None):
        raise click.UsageError(
            "provide either --state-dir (local) or --distributed-config (remote)"
        )
    if state_dir is not None:
        if hosted_config is not None or (
            request is not None and request.profile != DeploymentProfile.LOCAL
        ):
            raise click.UsageError("local state cannot be used for a remote profile")
        try:
            module = importlib.import_module("cutracer.service.local_workflow")
        except ModuleNotFoundError as error:
            if error.name != "cutracer.service.local_workflow":
                raise
            raise click.UsageError(
                "local execution requires the full cutracer CLI; "
                "service_run is the CPU-only remote control entry point"
            ) from error
        with redirect_stdout(sys.stderr):
            backend = module.create_local_backend(
                state_dir=state_dir,
                request=request,
                config_path=local_config,
                execute=execute,
            )
        return cast(api.DiagnosisBackend, backend), ""
    if local_config is not None or (
        request is not None and request.profile == DeploymentProfile.LOCAL
    ):
        raise click.UsageError(
            "remote configuration cannot be used for a local profile"
        )
    config_module = importlib.import_module("cutracer.service.fb.distributed_config")
    config = config_module.load_distributed_config(distributed_config)
    hosted = None
    if hosted_config is not None:
        hosted_module = importlib.import_module("cutracer.service.fb.hosted_controller")
        hosted = hosted_module.HostedControllerConfig.from_dict(
            read_json(hosted_config)
        )
    backend_module = importlib.import_module("cutracer.service.fb.run_backend")
    backend = backend_module.create_remote_backend(config, hosted=hosted)
    return cast(api.DiagnosisBackend, backend), canonical_input_digest(
        dataclasses.asdict(config)
    )


def _unsuccessful(output: object) -> bool:
    summary = output.summary if isinstance(output, RunResult) else output
    return isinstance(summary, RunSummary) and (
        summary.status in (RunStatus.FAILED, RunStatus.EXPIRED)
        or summary.publication == RunPublicationStatus.FAILED
    )


def _start(request_path: str, *, submit: bool, **options: Any) -> None:
    try:
        request = RunInput.from_dict(read_json(request_path))
        if submit and request.profile != DeploymentProfile.HOSTED:
            raise click.UsageError("diagnose-submit requires the hosted profile")
        backend, config_digest = _backend(request=request, execute=True, **options)
        if (
            request.profile == DeploymentProfile.HOSTED
            and not request.deployment_config_digest
        ):
            request = dataclasses.replace(
                request, deployment_config_digest=config_digest
            )
        output = (
            api.submit(request, backend=backend)
            if submit
            else api.run(request, backend=backend)
        )
    except (OSError, KeyError, TypeError, ValueError, SessionStoreError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(output.to_json(indent=2))
    if _unsuccessful(output):
        sys.exit(1)


def _existing(run_id: str, operation: str, **options: Any) -> None:
    execute = operation in ("resume", "advance")
    try:
        request = None
        state_dir = options.get("state_dir")
        if execute and state_dir is not None:
            request = (
                FileRunControlStore(os.path.join(state_dir, "runs")).load(run_id).input
            )
        backend, _digest = _backend(request=request, execute=execute, **options)
        function = {
            "resume": api.resume,
            "advance": api.advance,
            "status": api.status,
            "result": api.result,
        }[operation]
        output = function(run_id, backend=backend)
    except (OSError, KeyError, TypeError, ValueError, SessionStoreError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(output.to_json(indent=2))
    if execute and _unsuccessful(output):
        sys.exit(1)


@click.command(name="diagnose-run")
@click.argument("request_path", type=click.Path(exists=True, dir_okay=False))
@_backend_options
def diagnose_run_command(request_path: str, **options: Any) -> None:
    """Run a frozen request in the foreground, returning a RunResult.

    Exits 1 after printing the result when the Run failed, expired, or could
    not publish its output.
    """
    _start(request_path, submit=False, **options)


@click.command(name="diagnose-submit")
@click.argument("request_path", type=click.Path(exists=True, dir_okay=False))
@_backend_options
def diagnose_submit_command(request_path: str, **options: Any) -> None:
    """Submit a hosted request and return its RunHandle."""
    _start(request_path, submit=True, **options)


@click.command(name="diagnose-run-resume")
@click.argument("run_id")
@_backend_options
def diagnose_run_resume_command(run_id: str, **options: Any) -> None:
    """Resume this same Run in the foreground without resetting its deadline.

    Exits 1 after printing the result when the Run failed, expired, or could
    not publish its output.
    """
    _existing(run_id, "resume", **options)


@click.command(name="diagnose-run-advance")
@click.argument("run_id")
@_backend_options
def diagnose_run_advance_command(run_id: str, **options: Any) -> None:
    """Explicitly advance a Run when the caller owns scheduling.

    Exits 1 after printing the summary when the Run failed, expired, or could
    not publish its output.
    """
    _existing(run_id, "advance", **options)


@click.command(name="diagnose-run-status")
@click.argument("run_id")
@_backend_options
def diagnose_run_status_command(run_id: str, **options: Any) -> None:
    """Read Run state without executing GPU or model work."""
    _existing(run_id, "status", **options)


@click.command(name="diagnose-run-result")
@click.argument("run_id")
@_backend_options
def diagnose_run_result_command(run_id: str, **options: Any) -> None:
    """Read available reports, including failed and incomplete child status."""
    _existing(run_id, "result", **options)


@click.group(
    commands={
        "diagnose-run": diagnose_run_command,
        "diagnose-submit": diagnose_submit_command,
        "diagnose-run-resume": diagnose_run_resume_command,
        "diagnose-run-advance": diagnose_run_advance_command,
        "diagnose-run-status": diagnose_run_status_command,
        "diagnose-run-result": diagnose_run_result_command,
    }
)
def main() -> None:
    """CPU-only control of remote CUTracer diagnosis Runs."""
