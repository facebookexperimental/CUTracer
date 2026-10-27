"""Register and memory-value trace experiment adapter."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, cast, Mapping, Optional, Sequence, Union

from click import ClickException
from cutracer.runner import InstrumentationConfig, run_instrumented_target, RunTarget
from cutracer.service.artifacts import (
    local_artifact,
    local_file_uri,
    local_path,
    stage_verified_local_artifact,
)
from cutracer.service.contracts import (
    ArtifactRef,
    ExecutionProvenance,
    ExecutionStatus,
    ExperimentKind,
    ExperimentSpec,
    KernelId,
    ResultCompleteness,
    TraceAnalysisEvidence,
    TraceExperimentResult,
    TraceInstructionCategory,
    TraceSelector,
)
from cutracer.service.trace_analysis import run_trace_analysis
from cutracer.validation.compression import open_trace_file

TraceLocator = Callable[[str], Union[str, Sequence[str], None]]
TraceAnalysisRunner = Callable[..., TraceAnalysisEvidence]
logger: logging.Logger = logging.getLogger(__name__)

_TRACE_SUFFIXES = (".ndjson", ".ndjson.zst")
_RESERVED_CAPTURE_BASENAMES = frozenset(
    {
        "_SUCCESS",
        "experiment_envelope.json",
        "index.json",
        "trace.log",
        "trace_analysis.json",
        "trace_progress.json",
    }
)


def _validate_trace_spec(spec: ExperimentSpec) -> None:
    if spec.kind not in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
        raise ValueError(f"not a trace experiment: {spec.kind}")
    if spec.trace is None:
        raise ValueError("trace options are required")


def _validate_trace_runtime_identity(spec: ExperimentSpec) -> None:
    assert spec.trace is not None
    options = spec.trace
    if options.mode != spec.kind:
        raise ValueError("trace options mode differs from the experiment kind")
    if not 0 < options.trace_size_limit_mb <= 0xFFFFFFFF:
        raise ValueError("trace size limit must be in [1, UINT32_MAX] MB")
    kernel = spec.unit.kernel
    if kernel is None or not kernel.name.strip() or not kernel.cubin_hash.strip():
        raise ValueError("targeted trace requires a pinned WorkUnit kernel")


def _validate_trace_selector_scalars(selector: TraceSelector) -> None:
    if (
        selector.launch_id is not None
        and not 0 <= selector.launch_id <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("trace launch_id must fit an unsigned 64-bit integer")
    if selector.warp_id is not None and not 0 <= selector.warp_id < 32:
        raise ValueError("trace warp_id must be in [0, 31]")
    if selector.warpgroup_id is not None and not 0 <= selector.warpgroup_id < 8:
        raise ValueError("trace warpgroup_id must be in [0, 7]")
    if selector.warp_id is not None and selector.warpgroup_id is not None:
        raise ValueError("trace selector cannot combine warp_id and warpgroup_id")


def _validate_trace_selector_lists(selector: TraceSelector) -> None:
    if len(selector.pc_offsets) > 32:
        raise ValueError("trace selector accepts at most 32 PC offsets")
    for offset in selector.pc_offsets:
        value = int(offset, 0)
        if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("trace PC offset must fit an unsigned 64-bit integer")
    if any(
        not isinstance(category, TraceInstructionCategory)
        for category in selector.instruction_categories
    ):
        raise ValueError("trace instruction category is not allowlisted")


def _validate_trace_config_binding(spec: ExperimentSpec) -> None:
    assert spec.trace is not None
    selector = spec.trace.selector
    artifact = spec.trace.config_artifact
    if selector.config_source is not None and artifact is None:
        raise ValueError("config-targeted trace requires a resolved config artifact")
    if selector.config_source is None and artifact is not None:
        raise ValueError("trace config artifact requires a config_source selector")


def _validate_trace_runtime_spec(spec: ExperimentSpec) -> None:
    _validate_trace_spec(spec)
    assert spec.trace is not None
    _validate_trace_runtime_identity(spec)
    _validate_trace_selector_scalars(spec.trace.selector)
    _validate_trace_selector_lists(spec.trace.selector)
    _validate_trace_config_binding(spec)


def _default_trace_locator(out_dir: str) -> Sequence[str]:
    """Return every raw trace emitted by one isolated capture attempt."""
    if not os.path.isdir(out_dir):
        return []
    paths = []
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        # Kernel-events files are metadata-only launch logs whose first record
        # is ``kernel_launch``, not an instruction trace beginning with
        # ``kernel_metadata``. They are never deterministic-analyzer inputs.
        if (
            name.startswith("cutracer_kernel_events_")
            or name in _RESERVED_CAPTURE_BASENAMES
            or not name.endswith(_TRACE_SUFFIXES)
        ):
            continue
        if os.path.islink(path):
            raise OSError(f"capture emitted a symlinked trace candidate: {name}")
        if not os.path.isfile(path):
            raise OSError(f"capture emitted a non-regular trace candidate: {name}")
        paths.append(path)
    return sorted(paths, key=lambda path: (os.path.basename(path), path))


def _located_trace_paths(
    out_dir: str,
    locator: TraceLocator,
) -> list[str]:
    located = locator(out_dir)
    if located is None:
        return []
    raw_paths = [located] if isinstance(located, str) else list(located)
    paths = sorted(
        {os.path.abspath(path) for path in raw_paths},
        key=lambda path: (os.path.basename(path), path),
    )
    expected_root = os.path.abspath(out_dir)
    for path in paths:
        if os.path.commonpath((expected_root, path)) != expected_root:
            raise ValueError(
                f"trace locator returned a path outside its attempt: {path}"
            )
        if not os.path.isfile(path) or os.path.islink(path):
            raise OSError(f"trace locator returned a non-regular file: {path}")
        if not path.endswith(_TRACE_SUFFIXES):
            raise ValueError(f"trace locator returned an unsupported trace: {path}")
    return paths


def _trace_metadata(path: str) -> tuple[dict[str, object], bool]:
    """Read the portable header and report whether trace data follows it."""
    metadata: Optional[dict[str, object]] = None
    has_trace_records = False
    with open_trace_file(path) as trace_file:
        for raw_line in trace_file:
            line = raw_line.strip()
            if not line:
                continue
            parsed: Any = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"trace record is not a JSON object: {path}")
            record = cast(dict[str, object], parsed)
            record_type = record.get("type")
            if metadata is None:
                if record_type != "kernel_metadata":
                    raise ValueError(
                        f"trace does not begin with kernel_metadata: {path}"
                    )
                metadata = record
                continue
            if record_type not in ("kernel_metadata", "callstack_def"):
                has_trace_records = True
                break
    if metadata is None:
        raise ValueError(f"trace is empty: {path}")
    return metadata, has_trace_records


def _expected_targeting_metadata(
    spec: ExperimentSpec,
    *,
    resolved_config_path: Optional[str],
) -> dict[str, object]:
    assert spec.trace is not None
    warp_mask = _trace_warp_mask(spec)
    pc_offsets = _trace_pc_offsets(
        spec,
        resolved_config_path=resolved_config_path,
    )
    requested_categories = {
        category.value for category in spec.trace.selector.instruction_categories
    }
    # Enum declaration order mirrors the C++ serializer's stable enum order.
    # Deriving the list makes a future Python enum addition fail closed against
    # older C++ metadata instead of silently dropping the new category here.
    categories = [
        category.value
        for category in TraceInstructionCategory
        if category.value in requested_categories
    ]
    return {
        "trace_launch_id": spec.trace.selector.launch_id,
        "trace_warp_mask": None if warp_mask is None else int(warp_mask, 0),
        "trace_pc_offsets": (
            None if not pc_offsets else [int(offset, 0) for offset in pc_offsets]
        ),
        "trace_pc_filter_valid": True,
        "trace_instr_categories": None if not categories else categories,
    }


def _matches_pinned_capture(
    metadata: Mapping[str, object],
    pinned: KernelId,
    *,
    launch_matches: bool,
) -> bool:
    names = (metadata.get("mangled_name"), metadata.get("unmangled_name"))
    if pinned.name not in names:
        return False
    if metadata.get("kernel_checksum") != pinned.cubin_hash:
        if launch_matches:
            raise ValueError(
                "captured trace kernel checksum differs from the pinned WorkUnit"
            )
        return False
    return True


def _validate_effective_targeting(
    metadata: Mapping[str, object],
    expected_targeting: Mapping[str, object],
    instrument_modes: Sequence[str],
) -> None:
    for field, expected in expected_targeting.items():
        if metadata.get(field) != expected:
            raise ValueError(f"captured trace targeting metadata differs for {field}")
    actual_modes = metadata.get("instrument_modes")
    if (
        not isinstance(actual_modes, list)
        or any(not isinstance(mode, str) for mode in actual_modes)
        or sorted(actual_modes) != sorted(instrument_modes)
    ):
        raise ValueError(
            "captured trace instrumentation mode differs from the trace job"
        )


def _portable_cubin_name(metadata: Mapping[str, object]) -> str:
    cubin_name = metadata.get("cubin_path")
    if (
        not isinstance(cubin_name, str)
        or not cubin_name.endswith(".cubin")
        or os.path.basename(cubin_name) != cubin_name
    ):
        raise ValueError("captured trace has no portable cubin_path")
    return cubin_name


def _selected_capture_metadata(
    metadata: Mapping[str, object],
    *,
    spec: ExperimentSpec,
    pinned: KernelId,
    expected_targeting: Mapping[str, object],
) -> Optional[tuple[int, str]]:
    assert spec.trace is not None
    launch_id = metadata.get("grid_launch_id")
    requested_launch = spec.trace.selector.launch_id
    launch_matches = requested_launch is None or launch_id == requested_launch
    if not _matches_pinned_capture(
        metadata,
        pinned,
        launch_matches=launch_matches,
    ):
        return None
    if not isinstance(launch_id, int) or isinstance(launch_id, bool):
        raise ValueError("captured trace is missing a valid grid_launch_id")
    if not launch_matches:
        return None
    _validate_effective_targeting(
        metadata, expected_targeting, spec.trace.effective_instrument_modes
    )
    return launch_id, _portable_cubin_name(metadata)


def _selected_capture_outputs(
    paths: Sequence[str],
    *,
    attempt_dir: str,
    spec: ExperimentSpec,
    resolved_config_path: Optional[str],
) -> tuple[list[str], list[str], bool]:
    """Select only exact pinned-kernel launches and their declared cubins."""
    pinned = _selected_kernel(spec)
    if pinned is None:
        raise ValueError("targeted trace requires a pinned WorkUnit kernel")
    assert spec.trace is not None
    expected_targeting = _expected_targeting_metadata(
        spec,
        resolved_config_path=resolved_config_path,
    )
    selected: list[str] = []
    cubin_names = set()
    launch_ids = set()
    all_have_records = True
    for path in paths:
        metadata, has_records = _trace_metadata(path)
        selected_metadata = _selected_capture_metadata(
            metadata,
            spec=spec,
            pinned=pinned,
            expected_targeting=expected_targeting,
        )
        if selected_metadata is None:
            continue
        launch_id, cubin_name = selected_metadata
        if launch_id in launch_ids:
            raise ValueError(f"captured duplicate grid_launch_id: {launch_id}")
        launch_ids.add(launch_id)
        cubin_names.add(cubin_name)
        selected.append(path)
        all_have_records = all_have_records and has_records

    if not selected:
        launch = spec.trace.selector.launch_id
        suffix = "" if launch is None else f" at launch_id {launch}"
        raise ValueError(f"pinned kernel did not produce a targeted trace{suffix}")

    supporting = []
    for name in sorted(cubin_names):
        path = os.path.join(attempt_dir, name)
        if not os.path.isfile(path) or os.path.islink(path):
            raise OSError(f"captured trace cubin is unavailable: {name}")
        supporting.append(path)
    return selected, supporting, all_have_records


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


def _capture_to_log(
    log_path: str,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> Optional[ArtifactRef]:
    """Persist captured stdout/stderr before the attempt directory is removed."""
    try:
        with open(log_path, "w") as fh:
            fh.write(_text(stdout) + "\n" + _text(stderr))
    except OSError:
        logger.warning("failed to capture trace log to %s", log_path, exc_info=True)
        return None
    return local_artifact(log_path)


def _sealed_log_if_available(out_dir: str) -> Optional[ArtifactRef]:
    path = os.path.join(out_dir, "trace.log")
    if not os.path.isfile(path) or os.path.islink(path):
        return None
    return local_artifact(path, media_type="text/plain", include_sha256=True)


def _promote_artifact(
    source: str,
    destination: str,
    *,
    media_type: str,
) -> ArtifactRef:
    """Atomically copy one attempt artifact to its stable, hashed path."""
    descriptor = -1
    temporary_path: Optional[str] = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(destination)}_",
            dir=os.path.dirname(destination),
        )
        with open(source, "rb") as source_file, os.fdopen(descriptor, "wb") as staged:
            descriptor = -1
            shutil.copyfileobj(source_file, staged)
            staged.flush()
            os.fsync(staged.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        directory = os.open(os.path.dirname(destination), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        artifact = local_artifact(
            destination,
            media_type=media_type,
            include_sha256=True,
        )
        if artifact.sha256 is None or artifact.size_bytes is None:
            raise OSError(f"artifact disappeared before it was hashed: {destination}")
        return artifact
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _promote_paths(
    paths: Sequence[str],
    *,
    out_dir: str,
    media_type: str,
    promoted: Optional[list[ArtifactRef]] = None,
) -> list[ArtifactRef]:
    artifacts = [] if promoted is None else promoted
    seen = set()
    for path in paths:
        basename = os.path.basename(path)
        if not basename or basename in seen:
            raise ValueError(
                f"capture emitted duplicate artifact basename: {basename!r}"
            )
        if basename in _RESERVED_CAPTURE_BASENAMES:
            raise ValueError(f"capture emitted reserved artifact basename: {basename}")
        seen.add(basename)
    for path in paths:
        basename = os.path.basename(path)
        artifacts.append(
            _promote_artifact(
                path,
                os.path.join(out_dir, basename),
                media_type=(
                    "application/zstd"
                    if media_type == "application/x-ndjson" and path.endswith(".zst")
                    else media_type
                ),
            )
        )
    return artifacts


def _selected_kernel(spec: ExperimentSpec) -> Optional[KernelId]:
    assert spec.trace is not None
    requested = spec.trace.selector.kernel
    if requested is not None and requested != spec.unit.kernel:
        raise ValueError("trace selector kernel does not match the pinned WorkUnit")
    return spec.unit.kernel


def _parse_config_pc(value: object) -> int:
    text = str(value)
    try:
        return int(text, 0)
    except ValueError:
        if text.isdecimal():
            return int(text, 10)
        raise


def _resolved_trace_config_path(
    spec: ExperimentSpec,
    *,
    resolved_config_path: Optional[str],
) -> Optional[str]:
    assert spec.trace is not None
    artifact = spec.trace.config_artifact
    if artifact is None:
        if spec.trace.selector.config_source is not None:
            raise ValueError(
                "config-targeted trace requires a resolved config artifact"
            )
        if resolved_config_path is not None:
            raise ValueError("resolved trace config has no config artifact binding")
        return None
    if spec.trace.selector.config_source is None:
        raise ValueError("trace config artifact requires a config_source selector")
    if resolved_config_path is None:
        return local_path(artifact)
    if not os.path.isabs(resolved_config_path):
        raise ValueError("resolved trace config path must be absolute")
    return resolved_config_path


def _enabled_point_offset(key: object, point: object) -> Optional[str]:
    # Match the delay-config schema/loader exactly: only an explicit boolean
    # true enables a point. Missing, false, or merely truthy values must not
    # silently widen a targeted trace.
    if not isinstance(point, dict) or not point.get("on", True):
        return None
    raw_offset = point["pc"] if "pc" in point else key
    return hex(_parse_config_pc(raw_offset))


def _matching_config_offsets(
    kernels: Mapping[object, object],
    pinned: KernelId,
) -> tuple[list[str], bool]:
    offsets = []
    matched_kernel = False
    for raw_kernel in kernels.values():
        if not isinstance(raw_kernel, dict):
            continue
        if raw_kernel.get("kernel_checksum") != pinned.cubin_hash:
            continue
        # A delay config may store a mangled CUDA name while the WorkUnit uses
        # the demangled launch name; the cubin checksum is shared by both.
        matched_kernel = True
        points = raw_kernel.get("instrumentation_points")
        if not isinstance(points, dict):
            continue
        offsets.extend(
            offset
            for key, point in points.items()
            if (offset := _enabled_point_offset(key, point)) is not None
        )
    return offsets, matched_kernel


def _config_pc_offsets(
    spec: ExperimentSpec,
    *,
    resolved_config_path: Optional[str],
) -> list[str]:
    """Resolve enabled PCs from the caller's verified config snapshot.

    MAST verifies/materializes the input artifact before entering this adapter
    and passes that immutable snapshot via ``resolved_config_path``.  Local
    callers retain the legacy direct-local-artifact fallback.
    """
    config_path = _resolved_trace_config_path(
        spec,
        resolved_config_path=resolved_config_path,
    )
    if config_path is None:
        return []
    with open(config_path) as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or not isinstance(payload.get("kernels"), dict):
        raise ValueError("selected trace config has no kernels mapping")

    pinned = _selected_kernel(spec)
    if pinned is None:
        raise ValueError("config-targeted trace requires a pinned WorkUnit kernel")
    offsets, matched_kernel = _matching_config_offsets(payload["kernels"], pinned)
    if not matched_kernel:
        raise ValueError(
            "selected trace config has no entry for the pinned kernel checksum "
            f"{pinned.cubin_hash}; config names may be mangled while the WorkUnit "
            f"name is {pinned.name}"
        )
    if not offsets:
        raise ValueError(
            "selected trace config has no enabled PCs for the pinned kernel"
        )
    return offsets


def _trace_pc_offsets(
    spec: ExperimentSpec,
    *,
    resolved_config_path: Optional[str],
) -> list[str]:
    assert spec.trace is not None
    offsets = spec.trace.selector.pc_offsets or _config_pc_offsets(
        spec,
        resolved_config_path=resolved_config_path,
    )
    normalized = sorted(
        {hex(int(offset, 0)) for offset in offsets}, key=lambda x: int(x, 0)
    )
    if any(not 0 <= int(offset, 0) <= 0xFFFFFFFFFFFFFFFF for offset in normalized):
        raise ValueError("trace PC offset must fit an unsigned 64-bit integer")
    if len(normalized) > 32:
        raise ValueError("targeted trace accepts at most 32 resolved PC offsets")
    return normalized


def _trace_warp_mask(spec: ExperimentSpec) -> Optional[str]:
    assert spec.trace is not None
    selector = spec.trace.selector
    if selector.warp_id is not None:
        if not 0 <= selector.warp_id < 32:
            raise ValueError("trace warp_id must be in [0, 31]")
        return hex(1 << selector.warp_id)
    if selector.warpgroup_id is not None:
        if not 0 <= selector.warpgroup_id < 8:
            raise ValueError("trace warpgroup_id must be in [0, 7]")
        return hex(0xF << (selector.warpgroup_id * 4))
    return None


def _validate_resume(
    resume_from: TraceExperimentResult,
    *,
    spec: ExperimentSpec,
    provenance: ExecutionProvenance,
) -> None:
    if (
        resume_from.experiment_id != spec.experiment_id
        or resume_from.mode != spec.kind
        or resume_from.provenance != provenance
        or resume_from.execution_status != ExecutionStatus.SUCCEEDED
        or resume_from.completeness != ResultCompleteness.PARTIAL
        or resume_from.analysis is not None
        or not resume_from.traces
        or not resume_from.supporting_artifacts
        or resume_from.log is None
    ):
        raise ValueError("resumed trace progress does not match captured experiment")


def _stage_resume_artifacts(
    artifacts: Sequence[ArtifactRef],
    *,
    source_dir: str,
    destination_dir: str,
) -> list[str]:
    os.makedirs(destination_dir, exist_ok=True)
    paths = []
    seen = set()
    for artifact in artifacts:
        basename = os.path.basename(artifact.relative_path)
        if not basename or artifact.relative_path != basename:
            raise ValueError(
                "resumed trace artifact must use a portable basename: "
                f"{artifact.relative_path!r}"
            )
        if basename in seen:
            raise ValueError(f"resumed trace has duplicate artifact: {basename!r}")
        seen.add(basename)
        # A sealed MAST checkpoint intentionally rewrites ``uri`` to its
        # Manifold location.  Restore has already materialized each declared
        # basename under ``out_dir``; reconstruct a local reference while
        # retaining the sealed hash/size, then verify and snapshot those bytes
        # before analysis.
        source_path = os.path.abspath(os.path.join(source_dir, basename))
        if os.path.islink(source_path):
            raise ValueError(f"resumed trace artifact is a symlink: {basename}")
        local_ref = ArtifactRef(
            uri=local_file_uri(source_path),
            relative_path=basename,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            media_type=artifact.media_type,
        )
        staged = stage_verified_local_artifact(
            local_ref,
            destination=os.path.join(destination_dir, basename),
        )
        paths.append(staged.path)
    return paths


def _run_trace_capture(
    spec: ExperimentSpec,
    *,
    attempt_dir: str,
    runner: Optional[RunTarget],
    cutracer_so: Optional[str],
    timeout: int,
    resolved_argv: Optional[Sequence[str]],
    resolved_cwd: Optional[str],
    resolved_base_env: Optional[Mapping[str, str]],
    resolved_config_path: Optional[str],
    use_bundled_cuda_tools: bool,
) -> subprocess.CompletedProcess[str]:
    assert spec.trace is not None
    selected_kernel = _selected_kernel(spec)
    kernel_filter = (
        selected_kernel.name
        if selected_kernel is not None and selected_kernel.name
        else None
    )
    pc_offsets = _trace_pc_offsets(
        spec,
        resolved_config_path=resolved_config_path,
    )
    categories = ",".join(
        category.value for category in spec.trace.selector.instruction_categories
    )
    if resolved_base_env is None:
        base_env = os.environ.copy()
        base_env.update(spec.unit.env)
    else:
        base_env = dict(resolved_base_env)
    argv = list(spec.unit.argv if resolved_argv is None else resolved_argv)
    cwd = resolved_cwd if resolved_cwd is not None else (spec.unit.cwd or None)
    return run_instrumented_target(
        argv,
        InstrumentationConfig(
            cutracer_so=cutracer_so,
            instrument=",".join(spec.trace.effective_instrument_modes),
            kernel_filters=kernel_filter,
            trace_instr_categories=categories or None,
            trace_launch_id=spec.trace.selector.launch_id,
            trace_warp_mask=_trace_warp_mask(spec),
            trace_pc_offsets=",".join(pc_offsets) or None,
            trace_format="ndjson",
            output_dir=attempt_dir,
            dump_cubin=True,
            trace_size_limit_mb=spec.trace.trace_size_limit_mb,
            cwd=cwd,
            timeout=timeout,
            base_env=base_env,
            use_bundled_cuda_tools=use_bundled_cuda_tools,
        ),
        runner=runner,
    )


def _seal_capture_artifacts(
    *,
    attempt_dir: str,
    out_dir: str,
    trace_locator: TraceLocator,
    spec: ExperimentSpec,
    resolved_config_path: Optional[str],
    traces: list[ArtifactRef],
    supporting: list[ArtifactRef],
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> tuple[ArtifactRef, bool]:
    attempt_log = os.path.join(attempt_dir, "trace.log")
    _capture_to_log(attempt_log, stdout, stderr)
    log = _promote_artifact(
        attempt_log,
        os.path.join(out_dir, "trace.log"),
        media_type="text/plain",
    )
    selected, supporting_paths, all_have_records = _selected_capture_outputs(
        _located_trace_paths(attempt_dir, trace_locator),
        attempt_dir=attempt_dir,
        spec=spec,
        resolved_config_path=resolved_config_path,
    )
    _promote_paths(
        selected,
        out_dir=out_dir,
        media_type="application/x-ndjson",
        promoted=traces,
    )
    _promote_paths(
        supporting_paths,
        out_dir=out_dir,
        media_type="application/x-cubin",
        promoted=supporting,
    )
    return log, all_have_records


def _capture_error(
    proc: subprocess.CompletedProcess[str],
    traces: Sequence[ArtifactRef],
    supporting: Sequence[ArtifactRef],
    all_have_records: bool,
) -> Optional[tuple[ExecutionStatus, str]]:
    if proc.returncode != 0:
        return ExecutionStatus.FAILED, _target_failure_detail(proc)
    if not traces:
        return (
            ExecutionStatus.INFRA_ERROR,
            "CUTracer completed without producing a trace",
        )
    if not supporting:
        return (
            ExecutionStatus.INFRA_ERROR,
            "CUTracer completed without producing a cubin artifact",
        )
    output = f"{_text(proc.stdout)}\n{_text(proc.stderr)}"
    if "Trace size limit exceeded" in output:
        return (
            ExecutionStatus.INFRA_ERROR,
            "CUTracer stopped capture after the trace size limit was exceeded",
        )
    if not all_have_records:
        return (
            ExecutionStatus.INFRA_ERROR,
            "one or more targeted traces contain metadata but no selected trace records",
        )
    return None


def _target_failure_detail(proc: subprocess.CompletedProcess[str]) -> str:
    return _text(proc.stderr) or _text(proc.stdout) or "instrumented target failed"


def _run_trace_analyzer(
    analysis_paths: Sequence[str],
    *,
    arch: str,
    out_dir: str,
    trace_analyzer: Optional[TraceAnalysisRunner],
) -> tuple[Optional[TraceAnalysisEvidence], Optional[str]]:
    analysis_runner = trace_analyzer or run_trace_analysis
    try:
        analysis = analysis_runner(
            analysis_paths,
            arch=arch,
            out_dir=out_dir,
        )
    except Exception as exc:
        logger.exception("deterministic trace analysis runner failed")
        candidate = getattr(analysis_runner, "__name__", "")
        analyzer_name = candidate.strip() if isinstance(candidate, str) else ""
        if not analyzer_name:
            analyzer_name = type(analysis_runner).__name__.strip() or "trace_analyzer"
        error = f"deterministic trace analysis failed: {exc}"
        return (
            TraceAnalysisEvidence(
                analyzer=analyzer_name,
                execution_status=ExecutionStatus.INFRA_ERROR,
                detector_summaries=[],
                findings=[],
                total_findings=0,
                error=error,
            ),
            error,
        )
    artifact = analysis.artifact
    artifact_is_hashed = (
        artifact is not None
        and artifact.sha256 is not None
        and artifact.size_bytes is not None
    )
    if analysis.execution_status != ExecutionStatus.SUCCEEDED or not artifact_is_hashed:
        error = (
            analysis.error
            or "deterministic trace analysis did not produce a hashed report"
        )
        # Keep partial findings, but normalize the nested evidence so the
        # outer INFRA_ERROR/PARTIAL result remains valid at the service
        # boundary. An unhashed artifact is not portable evidence.
        analysis = dataclasses.replace(
            analysis,
            execution_status=ExecutionStatus.INFRA_ERROR,
            artifact=artifact if artifact_is_hashed else None,
            error=error,
        )
        return (
            analysis,
            error,
        )
    return analysis, None


def _seal_timed_out_capture(
    *,
    attempt_dir: str,
    out_dir: str,
    trace_locator: TraceLocator,
    spec: ExperimentSpec,
    resolved_config_path: Optional[str],
    exc: subprocess.TimeoutExpired,
) -> tuple[
    list[ArtifactRef],
    list[ArtifactRef],
    Optional[ArtifactRef],
    Optional[str],
]:
    traces: list[ArtifactRef] = []
    supporting: list[ArtifactRef] = []
    try:
        log, _all_have_records = _seal_capture_artifacts(
            attempt_dir=attempt_dir,
            out_dir=out_dir,
            trace_locator=trace_locator,
            spec=spec,
            resolved_config_path=resolved_config_path,
            traces=traces,
            supporting=supporting,
            stdout=exc.stdout,
            stderr=exc.stderr,
        )
        return traces, supporting, log, None
    except Exception as promotion_error:
        logger.exception("failed to seal timed-out trace capture")
        return (
            traces,
            supporting,
            _sealed_log_if_available(out_dir),
            "trace timed out and partial evidence could not be sealed: "
            f"{promotion_error}",
        )


@dataclasses.dataclass(frozen=True)
class _TraceRunOptions:
    runner: Optional[RunTarget]
    trace_locator: TraceLocator
    trace_analyzer: Optional[TraceAnalysisRunner]
    cutracer_so: Optional[str]
    timeout: int
    resolved_argv: Optional[Sequence[str]]
    resolved_cwd: Optional[str]
    resolved_base_env: Optional[Mapping[str, str]]
    resolved_config_path: Optional[str]
    use_bundled_cuda_tools: bool


@dataclasses.dataclass
class _TraceRunState:
    spec: ExperimentSpec
    out_dir: str
    provenance: ExecutionProvenance
    started: float
    capture_duration_s: Optional[float] = None
    analysis_duration_s: float = 0.0
    traces: list[ArtifactRef] = dataclasses.field(default_factory=list)
    supporting: list[ArtifactRef] = dataclasses.field(default_factory=list)
    log: Optional[ArtifactRef] = None
    analysis: Optional[TraceAnalysisEvidence] = None

    def result(
        self,
        status: ExecutionStatus,
        *,
        error: Optional[str],
        completeness: ResultCompleteness,
    ) -> TraceExperimentResult:
        return TraceExperimentResult(
            experiment_id=self.spec.experiment_id,
            execution_status=status,
            mode=self.spec.kind,
            provenance=self.provenance,
            trace=self.traces[0] if self.traces else None,
            traces=list(self.traces),
            supporting_artifacts=list(self.supporting),
            log=self.log,
            analysis=self.analysis,
            duration_s=(
                self.capture_duration_s
                if self.capture_duration_s is not None
                else time.monotonic() - self.started
            ),
            error=error,
            completeness=completeness,
            analysis_duration_s=self.analysis_duration_s,
        )


def _new_trace_run_state(
    spec: ExperimentSpec,
    out_dir: str,
    *,
    cutracer_version: str,
    toolchain_version: str,
) -> _TraceRunState:
    return _TraceRunState(
        spec=spec,
        out_dir=out_dir,
        provenance=ExecutionProvenance(
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
            tool_version=cutracer_version,
            toolchain_version=toolchain_version,
        ),
        started=time.monotonic(),
    )


def _seal_fresh_capture(
    state: _TraceRunState,
    attempt_dir: str,
    options: _TraceRunOptions,
    proc: subprocess.CompletedProcess[str],
) -> Optional[TraceExperimentResult]:
    try:
        state.log, all_have_records = _seal_capture_artifacts(
            attempt_dir=attempt_dir,
            out_dir=state.out_dir,
            trace_locator=options.trace_locator,
            spec=state.spec,
            resolved_config_path=options.resolved_config_path,
            traces=state.traces,
            supporting=state.supporting,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except (OSError, ValueError) as seal_error:
        state.log = _sealed_log_if_available(state.out_dir)
        if proc.returncode != 0:
            return state.result(
                ExecutionStatus.FAILED,
                error=(
                    f"{_target_failure_detail(proc)}; "
                    f"partial evidence could not be sealed: {seal_error}"
                ),
                completeness=ResultCompleteness.PARTIAL,
            )
        raise
    state.capture_duration_s = time.monotonic() - state.started
    capture_error = _capture_error(
        proc,
        state.traces,
        state.supporting,
        all_have_records,
    )
    if capture_error is None:
        return None
    status, error = capture_error
    return state.result(
        status,
        error=error,
        completeness=ResultCompleteness.PARTIAL,
    )


def _fresh_analysis_paths(
    state: _TraceRunState,
    attempt_dir: str,
    options: _TraceRunOptions,
) -> tuple[list[str], Optional[TraceExperimentResult]]:
    proc = _run_trace_capture(
        state.spec,
        attempt_dir=attempt_dir,
        runner=options.runner,
        cutracer_so=options.cutracer_so,
        timeout=options.timeout,
        resolved_argv=options.resolved_argv,
        resolved_cwd=options.resolved_cwd,
        resolved_base_env=options.resolved_base_env,
        resolved_config_path=options.resolved_config_path,
        use_bundled_cuda_tools=options.use_bundled_cuda_tools,
    )
    early_result = _seal_fresh_capture(state, attempt_dir, options, proc)
    return [local_path(trace) for trace in state.traces], early_result


def _analyze_trace_paths(
    state: _TraceRunState,
    analysis_paths: Sequence[str],
    options: _TraceRunOptions,
) -> TraceExperimentResult:
    analysis_started = time.monotonic()
    try:
        state.analysis, error = _run_trace_analyzer(
            analysis_paths,
            arch=state.spec.unit.arch,
            out_dir=state.out_dir,
            trace_analyzer=options.trace_analyzer,
        )
    finally:
        state.analysis_duration_s += time.monotonic() - analysis_started
    if error is not None:
        return state.result(
            ExecutionStatus.INFRA_ERROR,
            error=error,
            completeness=ResultCompleteness.PARTIAL,
        )
    return state.result(
        ExecutionStatus.SUCCEEDED,
        error=None,
        completeness=ResultCompleteness.COMPLETE,
    )


def _timeout_trace_result(
    state: _TraceRunState,
    attempt_dir: str,
    options: _TraceRunOptions,
    exc: subprocess.TimeoutExpired,
) -> TraceExperimentResult:
    state.traces, state.supporting, state.log, seal_error = _seal_timed_out_capture(
        attempt_dir=attempt_dir,
        out_dir=state.out_dir,
        trace_locator=options.trace_locator,
        spec=state.spec,
        resolved_config_path=options.resolved_config_path,
        exc=exc,
    )
    error = str(exc) if seal_error is None else f"{exc}; {seal_error}"
    return state.result(
        ExecutionStatus.TIMED_OUT,
        error=error,
        completeness=ResultCompleteness.PARTIAL,
    )


def run_trace_experiment(
    spec: ExperimentSpec,
    *,
    out_dir: str,
    runner: Optional[RunTarget] = None,
    trace_locator: Optional[TraceLocator] = None,
    trace_analyzer: Optional[TraceAnalysisRunner] = None,
    cutracer_so: Optional[str] = None,
    timeout: int = 1800,
    cutracer_version: str = "",
    toolchain_version: str = "",
    resolved_argv: Optional[Sequence[str]] = None,
    resolved_cwd: Optional[str] = None,
    resolved_base_env: Optional[Mapping[str, str]] = None,
    resolved_config_path: Optional[str] = None,
    use_bundled_cuda_tools: bool = True,
) -> TraceExperimentResult:
    """Capture, seal, and analyze every targeted trace from one attempt."""
    _validate_trace_spec(spec)
    os.makedirs(out_dir, exist_ok=True)
    attempt_dir = tempfile.mkdtemp(prefix=".trace_attempt_", dir=out_dir)
    state = _new_trace_run_state(
        spec,
        out_dir,
        cutracer_version=cutracer_version,
        toolchain_version=toolchain_version,
    )
    options = _TraceRunOptions(
        runner=runner,
        trace_locator=trace_locator or _default_trace_locator,
        trace_analyzer=trace_analyzer,
        cutracer_so=cutracer_so,
        timeout=timeout,
        resolved_argv=resolved_argv,
        resolved_cwd=resolved_cwd,
        resolved_base_env=resolved_base_env,
        resolved_config_path=resolved_config_path,
        use_bundled_cuda_tools=use_bundled_cuda_tools,
    )
    try:
        _validate_trace_runtime_spec(spec)
        analysis_paths, early_result = _fresh_analysis_paths(
            state, attempt_dir, options
        )
        if early_result is not None:
            return early_result
        return _analyze_trace_paths(state, analysis_paths, options)
    except subprocess.TimeoutExpired as exc:
        return _timeout_trace_result(state, attempt_dir, options, exc)
    except (ClickException, OSError, ValueError) as exc:
        return state.result(
            ExecutionStatus.INFRA_ERROR,
            error=str(exc),
            completeness=ResultCompleteness.PARTIAL,
        )
    finally:
        shutil.rmtree(attempt_dir, ignore_errors=True)
