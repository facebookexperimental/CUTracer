"""Kernel discovery before creating pinned analysis sessions."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence

from click import ClickException
from cutracer.runner import InstrumentationConfig, run_instrumented_target, RunTarget
from cutracer.service.artifacts import local_artifact, publish_local_artifact
from cutracer.service.checkpoint import atomic_write_json
from cutracer.service.contracts import (
    ArtifactRef,
    DEFAULT_MAX_DISCOVERED_KERNELS,
    ExecutionProvenance,
    ExecutionStatus,
    KernelDiscoveryManifest,
    KernelDiscoveryOutcome,
    KernelDiscoveryResult,
    KernelId,
    WorkUnit,
)
from cutracer.service.discovery_plan import (
    discovery_child_session_id as discovery_child_session_id,
    pin_discovered_work_units as pin_discovered_work_units,
)


__all__ = [
    "KernelDiscovery",
    "KernelDiscoveryError",
    "discover_work_units",
    "discovery_child_session_id",
    "pin_discovered_work_units",
]


_EVENT_PREFIX = "cutracer_kernel_events_"
_EVENT_SUFFIX = ".ndjson"
_TARGET_OUTPUT_TAIL_CHARS = 4096


class KernelDiscoveryError(RuntimeError):
    """Discovery failed without producing a trustworthy kernel set."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.detail = message
        super().__init__(f"{code}: {message}")


@dataclasses.dataclass(frozen=True)
class KernelDiscovery:
    result: KernelDiscoveryResult
    manifest_artifact: ArtifactRef
    work_units: tuple[WorkUnit, ...]

    @property
    def manifest(self) -> KernelDiscoveryManifest:
        if self.result.discovery is None:
            raise ValueError("successful kernel discovery has no manifest")
        return self.result.discovery


@dataclasses.dataclass(frozen=True)
class _ObservedDiscovery:
    process: subprocess.CompletedProcess[str]
    event_paths: tuple[str, ...]
    kernels: tuple[KernelId, ...]


def _event_paths(directory: str) -> list[str]:
    try:
        names = os.listdir(directory)
    except OSError as error:
        raise KernelDiscoveryError(
            "unreadable_output", f"cannot list discovery output: {error}"
        ) from error
    paths = []
    for name in names:
        if not name.startswith(_EVENT_PREFIX) or not name.endswith(_EVENT_SUFFIX):
            continue
        path = os.path.join(directory, name)
        if os.path.islink(path) or not os.path.isfile(path):
            raise KernelDiscoveryError(
                "malformed_event_file",
                f"kernel event path is not a regular file: {name}",
            )
        paths.append(path)
    if not paths:
        raise KernelDiscoveryError(
            "missing_event_files", "CUTracer emitted no kernel event files"
        )
    return sorted(paths, key=lambda path: (os.path.basename(path), path))


def _target_failure_detail(process: subprocess.CompletedProcess[str]) -> str:
    detail = f"kernel discovery target exited with code {process.returncode}"
    outputs = [
        (name, output.strip())
        for name, output in (("stdout", process.stdout), ("stderr", process.stderr))
        if output.strip()
    ]
    if not outputs:
        return detail
    tail_chars = _TARGET_OUTPUT_TAIL_CHARS // len(outputs)
    return (
        detail
        + "\n"
        + "\n".join(f"{name}: {output[-tail_chars:]}" for name, output in outputs)
    )


def _kernel_from_event(
    event: object, *, path: str, line_number: int
) -> KernelId | None:
    if not isinstance(event, dict):
        raise KernelDiscoveryError(
            "malformed_event", f"{path}:{line_number} is not a JSON object"
        )
    if event.get("type") != "kernel_launch":
        return None
    name = event.get("kernel_name")
    mangled_name = event.get("mangled_name")
    checksum = event.get("kernel_checksum")
    if not isinstance(name, str) or not name.strip():
        raise KernelDiscoveryError(
            "malformed_event", f"{path}:{line_number} has no kernel_name"
        )
    if mangled_name is not None and (
        not isinstance(mangled_name, str) or not mangled_name.strip()
    ):
        raise KernelDiscoveryError(
            "malformed_event", f"{path}:{line_number} has invalid mangled_name"
        )
    if not isinstance(checksum, str) or not checksum.strip():
        raise KernelDiscoveryError(
            "malformed_event", f"{path}:{line_number} has no kernel_checksum"
        )
    return KernelId(
        name=name,
        cubin_hash=checksum,
        mangled_name=mangled_name,
    )


def _read_kernels(paths: Sequence[str]) -> tuple[KernelId, ...]:
    kernels: dict[tuple[str, str], KernelId] = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise KernelDiscoveryError(
                            "malformed_event",
                            f"{path}:{line_number} is not valid JSON: {error}",
                        ) from error
                    kernel = _kernel_from_event(
                        event,
                        path=path,
                        line_number=line_number,
                    )
                    if kernel is not None:
                        key = (kernel.name, kernel.cubin_hash)
                        existing = kernels.get(key)
                        if existing is None or existing.mangled_name is None:
                            kernels[key] = kernel
                        elif (
                            kernel.mangled_name is not None
                            and kernel.mangled_name != existing.mangled_name
                        ):
                            raise KernelDiscoveryError(
                                "malformed_event",
                                f"{path}:{line_number} has conflicting mangled_name",
                            )
        except (OSError, UnicodeError) as error:
            raise KernelDiscoveryError(
                "unreadable_event_file", f"cannot read {path}: {error}"
            ) from error
    return tuple(kernels[key] for key in sorted(kernels))


def _matches_filters(kernel: KernelId, filters: Sequence[str]) -> bool:
    names = (kernel.name, kernel.mangled_name or "")
    return not filters or any(value in name for value in filters for name in names)


def _remote_artifact(
    artifact: ArtifactRef,
    path: str,
    artifact_uri_for_path: Callable[[str], str | None] | None,
) -> ArtifactRef:
    if artifact_uri_for_path is None:
        return artifact
    uri = artifact_uri_for_path(path)
    return artifact if uri is None else dataclasses.replace(artifact, uri=uri)


def _write_manifest(
    path: str,
    manifest: KernelDiscoveryManifest,
    artifact_uri_for_path: Callable[[str], str | None] | None,
) -> ArtifactRef:
    atomic_write_json(path, manifest.to_dict())
    return _remote_artifact(
        local_artifact(path, include_sha256=True, media_type="application/json"),
        path,
        artifact_uri_for_path,
    )


def _run_discovery_target(
    target: WorkUnit,
    *,
    attempt_dir: str,
    cutracer_so: str,
    timeout: int,
    runner: RunTarget | None,
    resolved_argv: Sequence[str] | None,
    resolved_cwd: str | None,
    resolved_base_env: Mapping[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    argv = target.argv if resolved_argv is None else list(resolved_argv)
    cwd = (target.cwd or None) if resolved_cwd is None else resolved_cwd
    base_env = (
        {**os.environ, **target.env}
        if resolved_base_env is None
        else dict(resolved_base_env)
    )
    try:
        return run_instrumented_target(
            argv,
            InstrumentationConfig(
                cutracer_so=cutracer_so,
                output_dir=attempt_dir,
                kernel_events="nostack",
                cwd=cwd,
                timeout=timeout,
                base_env=base_env,
            ),
            runner=runner,
        )
    except subprocess.TimeoutExpired as error:
        raise KernelDiscoveryError(
            "execution_timed_out", f"kernel discovery timed out: {error}"
        ) from error
    except (ClickException, OSError, ValueError) as error:
        raise KernelDiscoveryError(
            "execution_failed", f"kernel discovery could not run: {error}"
        ) from error


def _selected_kernels(
    discovered: Sequence[KernelId],
    filters: Sequence[str],
    max_kernels: int,
) -> tuple[KernelId, ...]:
    if not discovered:
        raise KernelDiscoveryError(
            "no_kernel_launches", "kernel event files contain no kernel launches"
        )
    selected = tuple(
        kernel for kernel in discovered if _matches_filters(kernel, filters)
    )
    if not selected:
        raise KernelDiscoveryError(
            "no_matching_kernels",
            "no discovered kernel matched the requested kernel filters",
        )
    if len(selected) > max_kernels:
        raise KernelDiscoveryError(
            "kernel_limit_exceeded",
            f"discovered {len(selected)} kernels; limit is {max_kernels}",
        )
    return selected


def _publish_event_artifacts(
    paths: Sequence[str],
    out_dir: str,
    artifact_uri_for_path: Callable[[str], str | None] | None,
) -> tuple[ArtifactRef, ...]:
    artifacts = []
    for path in paths:
        destination = os.path.join(out_dir, os.path.basename(path))
        artifact = publish_local_artifact(
            path,
            destination,
            media_type="application/x-ndjson",
        )
        artifacts.append(_remote_artifact(artifact, destination, artifact_uri_for_path))
    return tuple(artifacts)


def _observe_discovery(
    target: WorkUnit,
    *,
    attempt_dir: str,
    cutracer_so: str,
    timeout: int,
    max_kernels: int,
    runner: RunTarget | None,
    resolved_argv: Sequence[str] | None,
    resolved_cwd: str | None,
    resolved_base_env: Mapping[str, str] | None,
) -> _ObservedDiscovery:
    process = _run_discovery_target(
        target,
        attempt_dir=attempt_dir,
        cutracer_so=cutracer_so,
        timeout=timeout,
        runner=runner,
        resolved_argv=resolved_argv,
        resolved_cwd=resolved_cwd,
        resolved_base_env=resolved_base_env,
    )
    try:
        event_paths = tuple(_event_paths(attempt_dir))
        kernels = _selected_kernels(
            _read_kernels(event_paths),
            target.kernel_filters,
            max_kernels,
        )
    except KernelDiscoveryError as error:
        code = error.code
        if error.code == "missing_event_files" and process.returncode != 0:
            code = "target_failed"
        raise KernelDiscoveryError(
            code, f"{error.detail}; {_target_failure_detail(process)}"
        ) from error
    return _ObservedDiscovery(process, event_paths, kernels)


def _discovery_manifest(
    target: WorkUnit,
    observed: _ObservedDiscovery,
    event_artifacts: tuple[ArtifactRef, ...],
) -> KernelDiscoveryManifest:
    return KernelDiscoveryManifest(
        target_unit_id=target.unit_id,
        source_revision=target.source_revision,
        arch=target.arch,
        kernels=observed.kernels,
        event_artifacts=event_artifacts,
        target_returncode=observed.process.returncode,
        kernel_filters=tuple(target.kernel_filters),
    )


def _discovery_result(
    target: WorkUnit,
    manifest: KernelDiscoveryManifest,
    artifact: ArtifactRef,
    *,
    experiment_id: str,
    cutracer_version: str,
    toolchain_version: str,
    duration_s: float,
) -> KernelDiscoveryResult:
    return KernelDiscoveryResult(
        experiment_id=experiment_id,
        execution_status=ExecutionStatus.SUCCEEDED,
        outcome=KernelDiscoveryOutcome.DISCOVERED,
        provenance=ExecutionProvenance(
            source_revision=target.source_revision,
            arch=target.arch,
            tool_version=cutracer_version,
            toolchain_version=toolchain_version,
        ),
        discovery=manifest,
        manifest_artifact=artifact,
        duration_s=duration_s,
    )


def _publish_discovery(
    target: WorkUnit,
    observed: _ObservedDiscovery,
    *,
    out_dir: str,
    experiment_id: str,
    cutracer_version: str,
    toolchain_version: str,
    duration_s: float,
    artifact_uri_for_path: Callable[[str], str | None] | None,
) -> KernelDiscovery:
    event_artifacts = _publish_event_artifacts(
        observed.event_paths,
        out_dir,
        artifact_uri_for_path,
    )
    manifest = _discovery_manifest(target, observed, event_artifacts)
    artifact = _write_manifest(
        os.path.join(out_dir, "kernel_discovery_manifest.json"),
        manifest,
        artifact_uri_for_path,
    )
    result = _discovery_result(
        target,
        manifest,
        artifact,
        experiment_id=experiment_id,
        cutracer_version=cutracer_version,
        toolchain_version=toolchain_version,
        duration_s=duration_s,
    )
    return KernelDiscovery(
        result=result,
        manifest_artifact=artifact,
        work_units=pin_discovered_work_units(target, observed.kernels),
    )


def discover_work_units(
    target: WorkUnit,
    *,
    out_dir: str,
    cutracer_so: str,
    timeout: int = 1800,
    max_kernels: int = DEFAULT_MAX_DISCOVERED_KERNELS,
    runner: RunTarget | None = None,
    experiment_id: str | None = None,
    cutracer_version: str = "",
    toolchain_version: str = "",
    resolved_argv: Sequence[str] | None = None,
    resolved_cwd: str | None = None,
    resolved_base_env: Mapping[str, str] | None = None,
    artifact_uri_for_path: Callable[[str], str | None] | None = None,
) -> KernelDiscovery:
    """Run one metadata-only launch pass and return stable pinned child units."""
    if target.kernel is not None:
        raise ValueError("kernel discovery requires an unpinned WorkUnit")
    if max_kernels <= 0:
        raise ValueError("max_kernels must be positive")
    os.makedirs(out_dir, exist_ok=True)
    attempt_dir = tempfile.mkdtemp(prefix=".kernel_discovery_", dir=out_dir)
    started = time.monotonic()
    try:
        observed = _observe_discovery(
            target,
            attempt_dir=attempt_dir,
            cutracer_so=cutracer_so,
            timeout=timeout,
            max_kernels=max_kernels,
            runner=runner,
            resolved_argv=resolved_argv,
            resolved_cwd=resolved_cwd,
            resolved_base_env=resolved_base_env,
        )
        return _publish_discovery(
            target,
            observed,
            out_dir=out_dir,
            experiment_id=experiment_id or f"discover-{target.unit_id}",
            cutracer_version=cutracer_version,
            toolchain_version=toolchain_version,
            duration_s=time.monotonic() - started,
            artifact_uri_for_path=artifact_uri_for_path,
        )
    finally:
        shutil.rmtree(attempt_dir, ignore_errors=True)
