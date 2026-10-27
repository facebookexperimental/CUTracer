"""Register and memory-value trace experiment adapter."""

from __future__ import annotations

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
    mode: ExperimentKind,
) -> None:
    for field, expected in expected_targeting.items():
        if metadata.get(field) != expected:
            raise ValueError(f"captured trace targeting metadata differs for {field}")
    if metadata.get("instrument_modes") != [mode.value]:
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
    _validate_effective_targeting(metadata, expected_targeting, spec.kind)
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


def _run_trace_analyzer(
    trace_path: str,
    *,
    arch: str,
    out_dir: str,
    trace_analyzer: Optional[TraceAnalysisRunner],
) -> TraceAnalysisEvidence:
    analysis_runner = trace_analyzer or run_trace_analysis
    try:
        return analysis_runner(trace_path, arch=arch, out_dir=out_dir)
    except Exception as exc:
        logger.exception("deterministic trace analysis runner failed")
        candidate = getattr(analysis_runner, "__name__", "")
        name = candidate.strip() if isinstance(candidate, str) else ""
        if not name:
            name = type(analysis_runner).__name__.strip() or "trace_analyzer"
        return TraceAnalysisEvidence(
            analyzer=name,
            execution_status=ExecutionStatus.INFRA_ERROR,
            detector_summaries=[],
            findings=[],
            total_findings=0,
            error=f"deterministic trace analysis failed: {exc}",
        )


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
    if selector.warp_id is not None and selector.warpgroup_id is not None:
        raise ValueError("trace selector cannot set both warp_id and warpgroup_id")
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
) -> TraceExperimentResult:
    """Capture a targeted trace with this package's CUTracer runtime."""
    if spec.kind not in (ExperimentKind.REG_TRACE, ExperimentKind.MEM_VALUE_TRACE):
        raise ValueError(f"not a trace experiment: {spec.kind}")
    if spec.trace is None:
        raise ValueError("trace options are required")

    os.makedirs(out_dir, exist_ok=True)
    locate = trace_locator or _default_trace_locator
    base_env = os.environ.copy()
    base_env.update(spec.unit.env)
    started = time.monotonic()
    capture_finished: Optional[float] = None
    analysis_duration_s = 0.0
    log_path = os.path.join(out_dir, "trace.log")
    try:
        selected_kernel = _selected_kernel(spec)
        kernel_filter = (
            selected_kernel.name
            if selected_kernel is not None and selected_kernel.name
            else None
        )
        pc_offsets = _trace_pc_offsets(spec, resolved_config_path=None)
        categories = ",".join(
            category.value for category in spec.trace.selector.instruction_categories
        )
        proc = run_instrumented_target(
            spec.unit.argv,
            InstrumentationConfig(
                cutracer_so=cutracer_so,
                instrument=spec.kind.value,
                kernel_filters=kernel_filter,
                trace_instr_categories=categories or None,
                trace_launch_id=spec.trace.selector.launch_id,
                trace_warp_mask=_trace_warp_mask(spec),
                trace_pc_offsets=",".join(pc_offsets) or None,
                output_dir=out_dir,
                trace_size_limit_mb=spec.trace.trace_size_limit_mb,
                cwd=spec.unit.cwd or None,
                timeout=timeout,
                base_env=base_env,
            ),
            runner=runner,
        )
        log = _capture_to_log(log_path, proc.stdout, proc.stderr)

        trace_paths = _located_trace_paths(out_dir, locate)
        trace_path = max(trace_paths, key=os.path.getmtime) if trace_paths else None
        if proc.returncode == 0 and trace_path is not None:
            status = ExecutionStatus.SUCCEEDED
            error = None
            media_type = (
                "application/zstd"
                if trace_path.endswith(".zst")
                else "application/x-ndjson"
            )
            trace = local_artifact(trace_path, media_type=media_type)
            capture_finished = time.monotonic()
            analysis_started = capture_finished
            analysis = _run_trace_analyzer(
                trace_path,
                arch=spec.unit.arch,
                out_dir=out_dir,
                trace_analyzer=trace_analyzer,
            )
            analysis_duration_s = time.monotonic() - analysis_started
        elif proc.returncode == 0:
            status = ExecutionStatus.INFRA_ERROR
            error = "CUTracer completed without producing a trace"
            trace = None
            analysis = None
        else:
            status = ExecutionStatus.FAILED
            error = proc.stderr or "instrumented target failed"
            trace = None
            analysis = None
    except subprocess.TimeoutExpired as exc:
        status = ExecutionStatus.TIMED_OUT
        error = str(exc)
        trace = None
        analysis = None
        # The child is killed at the deadline, but subprocess still captured
        # whatever it emitted beforehand; persist it instead of dropping it.
        log = _capture_to_log(log_path, exc.stdout, exc.stderr)
    except (ClickException, OSError, ValueError) as exc:
        status = ExecutionStatus.INFRA_ERROR
        error = str(exc)
        trace = None
        analysis = None
        log = None

    return TraceExperimentResult(
        experiment_id=spec.experiment_id,
        execution_status=status,
        mode=spec.kind,
        provenance=ExecutionProvenance(
            source_revision=spec.unit.source_revision,
            arch=spec.unit.arch,
            kernel=spec.unit.kernel,
            tool_version=cutracer_version,
            toolchain_version=toolchain_version,
        ),
        trace=trace,
        log=log,
        analysis=analysis,
        duration_s=(
            (capture_finished if capture_finished is not None else time.monotonic())
            - started
        ),
        analysis_duration_s=analysis_duration_s,
        error=error,
    )
