# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer trace runner.

Wraps user commands with CUTracer environment variables for trace collection.
Resolves cutracer.so automatically via buck resource or explicit path.

Usage:
    cutracer trace --instrument=tma_trace -- ./vectoradd
    cutracer trace --instrument=tma_trace --instr-categories=tma -- python -m pytest test.py
"""

import hashlib
import importlib.resources as resources
import os
import platform
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import click

RunTarget = Callable[..., "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class InstrumentationConfig:
    """Programmatic configuration for one CUTracer-instrumented target run."""

    cutracer_so: Optional[str] = None
    instrument: Optional[str] = None
    analysis: Optional[str] = None
    kernel_filters: Optional[str] = None
    instr_categories: Optional[str] = None
    trace_format: Optional[str] = None
    output_dir: Optional[str] = None
    verbose: Optional[int] = None
    zstd_level: Optional[int] = None
    delay_ns: Optional[int] = None
    delay_min_ns: Optional[int] = None
    delay_enable_prob: Optional[float] = None
    delay_mode: Optional[str] = None
    delay_cluster_cta_id: Optional[int] = None
    delay_warpgroup_id: Optional[int] = None
    delay_warp_mask: Optional[str] = None
    delay_patterns: Optional[str] = None
    delay_dump_path: Optional[str] = None
    delay_load_path: Optional[str] = None
    cpu_callstack: Optional[str] = None
    channel_records: Optional[int] = None
    kernel_events: Optional[str] = None
    dump_cubin: bool = False
    trace_size_limit_mb: int = 0
    kernel_timeout_s: int = 0
    no_data_timeout_s: int = 15
    cwd: Optional[str] = None
    timeout: Optional[int] = None
    base_env: Optional[Mapping[str, str]] = None
    capture_output: bool = True
    shell: bool = False


@dataclass(frozen=True)
class _BundledCudaTools:
    """Filesystem paths for CUDA host tools bundled with CUTracer."""

    bin_dir: Path
    nvdisasm: Path
    cuobjdump: Path


def _is_under_tempdir(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path(tempfile.gettempdir()).resolve())
        return True
    except ValueError:
        return False


def _persist_extracted_so(src: Path) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "cutracer_so_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / "cutracer.so"
    src_size = src.stat().st_size
    if not dst.is_file() or dst.stat().st_size != src_size:
        shutil.copy2(src, dst)
        dst.chmod(0o755)
    return dst


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_arch() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def _persist_extracted_cuda_tools(
    extracted: Mapping[str, Path],
) -> _BundledCudaTools:
    """Copy temporary fbpkg resources into an immutable runtime cache.

    ``importlib.resources.as_file`` removes files extracted from a zip/PEX when
    its context exits. NVBit starts later in a child process, so the bundled
    host tools must outlive that context. The host architecture and content
    digest keep x86_64/aarch64 packages and CUTracer releases isolated.
    """
    tool_digests = {name: _file_sha256(path) for name, path in extracted.items()}
    bundle_digest = hashlib.sha256()
    for name in sorted(tool_digests):
        bundle_digest.update(name.encode())
        bundle_digest.update(b"\0")
        bundle_digest.update(tool_digests[name].encode())
        bundle_digest.update(b"\0")

    cache_root = (
        Path(tempfile.gettempdir())
        / f"cutracer_runtime_{os.getuid()}"
        / _host_arch()
        / bundle_digest.hexdigest()
        / "bin"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    for name, source in extracted.items():
        destination = cache_root / name
        if destination.is_file() and _file_sha256(destination) == tool_digests[name]:
            continue

        fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=str(cache_root))
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            temporary.chmod(0o755)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    return _BundledCudaTools(
        bin_dir=cache_root,
        nvdisasm=cache_root / "nvdisasm",
        cuobjdump=cache_root / "cuobjdump",
    )


def _resolve_bundled_cuda_tools() -> Optional[_BundledCudaTools]:
    """Resolve bundled nvdisasm/cuobjdump to paths valid for child processes."""
    try:
        package = resources.files("cutracer")
    except (FileNotFoundError, ModuleNotFoundError):
        return None

    tool_refs = {
        name: package.joinpath(f"bin/{name}") for name in ("nvdisasm", "cuobjdump")
    }
    if not all(ref.is_file() for ref in tool_refs.values()):
        return None

    try:
        with ExitStack() as stack:
            extracted = {
                name: stack.enter_context(resources.as_file(ref))
                for name, ref in tool_refs.items()
            }

            parents = {path.parent for path in extracted.values()}
            if len(parents) == 1 and not any(
                _is_under_tempdir(path) for path in extracted.values()
            ):
                bin_dir = parents.pop()
                return _BundledCudaTools(
                    bin_dir=bin_dir,
                    nvdisasm=bin_dir / "nvdisasm",
                    cuobjdump=bin_dir / "cuobjdump",
                )

            return _persist_extracted_cuda_tools(extracted)
    except OSError as exc:
        raise click.ClickException(
            f"Failed to prepare bundled CUDA tools: {exc}"
        ) from exc


def resolve_cutracer_so(
    explicit_path: Optional[str] = None,
    *,
    reject_inherited_injection: bool = True,
) -> str:
    """Resolve cutracer.so path.

    Resolution order:
    1. Explicit --cutracer-so path
    2. Buck resource (bundled with python_library via resources = {})
    3. CWD auto-discovery: ./lib/cutracer.so

    If CUDA_INJECTION64_PATH is already set, raises ClickException
    (it conflicts with cutracer trace's automatic configuration).

    If all fail, raises ClickException with clear instructions.
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_file():
            raise click.ClickException(f"cutracer.so not found at: {explicit_path}")
        return str(p.resolve())

    # Fail if CUDA_INJECTION64_PATH is already set — cutracer trace sets
    # this variable itself, and a conflicting value indicates a misconfiguration.
    env_injection = (
        os.environ.get("CUDA_INJECTION64_PATH") if reject_inherited_injection else None
    )
    if env_injection:
        raise click.ClickException(
            f"CUDA_INJECTION64_PATH is set in your environment:\n"
            f"  Path: {env_injection}\n\n"
            f"This conflicts with CUTracer's bundled cutracer.so and may cause\n"
            f"stale or mismatched behavior. Please either:\n"
            f"  1. Unset it:  unset CUDA_INJECTION64_PATH\n"
            f"  2. Use --cutracer-so to explicitly specify a path:\n"
            f"     cutracer trace --cutracer-so {env_injection} -i tma_trace -- ./app"
        )

    # Buck resource (internal): cutracer.so bundled via python_library resources.
    # Use as_file() to get a proper filesystem Path from the Traversable.
    # For on-disk resources (buck2 run, pip install), the path persists after
    # context exit and is returned directly. For zip-packaged resources (fbpkg
    # fetch, PEX), the temp file is removed when the context exits — copy the
    # bytes to a persistent cache path inside the system temp dir so the .so
    # is still on disk when libcuda dlopen()s it via CUDA_INJECTION64_PATH.
    try:
        so_ref = resources.files("cutracer").joinpath("cutracer.so")
        with resources.as_file(so_ref) as so_path:
            if so_path.is_file():
                if so_path.is_absolute() and not _is_under_tempdir(so_path):
                    click.echo(f"Using bundled cutracer.so: {so_path}")
                    return str(so_path)
                cached = _persist_extracted_so(so_path)
                click.echo(f"Using bundled cutracer.so: {cached}")
                return str(cached)
    except Exception:
        pass

    # CWD auto-discovery: supports running from the CUTracer project root
    # after `make`, which produces lib/cutracer.so.
    cwd_candidate = Path.cwd() / "lib" / "cutracer.so"
    if cwd_candidate.is_file():
        click.echo(f"Using cutracer.so found at: {cwd_candidate}")
        return str(cwd_candidate)

    raise click.ClickException(
        "Could not find cutracer.so. Options:\n"
        "  1. Use --cutracer-so /path/to/cutracer.so\n"
        "  2. Run from the CUTracer project root after 'make':\n"
        "     cd CUTracer && cutracer trace ...\n"
        "  3. (Internal) Use buck2 run (cutracer.so is bundled automatically):\n"
        "     buck2 run fbcode//triton/tools/CUTracer:cutracer -- trace ..."
    )


def build_cutracer_env(
    cutracer_so: str,
    instrument: Optional[str],
    analysis: Optional[str],
    kernel_filters: Optional[str],
    instr_categories: Optional[str],
    trace_format: Optional[str],
    output_dir: Optional[str],
    verbose: Optional[int],
    zstd_level: Optional[int],
    delay_ns: Optional[int],
    delay_min_ns: Optional[int] = None,
    delay_enable_prob: Optional[float] = None,
    delay_mode: Optional[str] = None,
    delay_cluster_cta_id: Optional[int] = None,
    delay_warpgroup_id: Optional[int] = None,
    delay_warp_mask: Optional[str] = None,
    delay_patterns: Optional[str] = None,
    delay_dump_path: Optional[str] = None,
    delay_load_path: Optional[str] = None,
    cpu_callstack: Optional[str] = None,
    channel_records: Optional[int] = None,
    kernel_events: Optional[str] = None,
    dump_cubin: bool = False,
    trace_size_limit_mb: int = 0,
    kernel_timeout_s: int = 0,
    no_data_timeout_s: int = 15,
    base_env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build environment dict with CUTracer variables."""
    env = dict(os.environ if base_env is None else base_env)
    env["CUDA_INJECTION64_PATH"] = cutracer_so

    if instrument is not None:
        env["CUTRACER_INSTRUMENT"] = instrument

    if analysis is not None:
        env["CUTRACER_ANALYSIS"] = analysis
    if kernel_filters is not None:
        env["KERNEL_FILTERS"] = kernel_filters
    if instr_categories is not None:
        env["CUTRACER_INSTR_CATEGORIES"] = instr_categories
    if trace_format is not None:
        env["CUTRACER_TRACE_FORMAT"] = str(trace_format)
    if output_dir is not None:
        # Resolve to absolute path so subprocesses that chdir (e.g. tests
        # using `cwd=tmp_dir`) still write logs to the user-visible location.
        env["CUTRACER_OUTPUT_DIR"] = str(Path(output_dir).resolve())
    if verbose is not None:
        env["TOOL_VERBOSE"] = str(verbose)
    if zstd_level is not None:
        env["CUTRACER_ZSTD_LEVEL"] = str(zstd_level)
    if delay_ns is not None:
        env["CUTRACER_DELAY_NS"] = str(delay_ns)
    if delay_min_ns is not None:
        env["CUTRACER_DELAY_MIN_NS"] = str(delay_min_ns)
    if delay_enable_prob is not None:
        env["CUTRACER_DELAY_ENABLE_PROB"] = str(delay_enable_prob)
    if delay_mode is not None:
        env["CUTRACER_DELAY_MODE"] = delay_mode
    if delay_cluster_cta_id is not None:
        env["CUTRACER_CLUSTER_CTA_ID"] = str(delay_cluster_cta_id)
    if delay_warpgroup_id is not None:
        env["CUTRACER_DELAY_WARPGROUP_ID"] = str(delay_warpgroup_id)
    if delay_warp_mask is not None:
        env["CUTRACER_DELAY_WARP_MASK"] = delay_warp_mask
    if delay_patterns is not None:
        env["CUTRACER_DELAY_PATTERNS"] = delay_patterns
    if delay_dump_path is not None:
        env["CUTRACER_DELAY_DUMP_PATH"] = str(Path(delay_dump_path).resolve())
    if delay_load_path is not None:
        env["CUTRACER_DELAY_LOAD_PATH"] = str(Path(delay_load_path).resolve())
    if cpu_callstack is not None:
        env["CUTRACER_CPU_CALLSTACK"] = str(cpu_callstack)
    if channel_records is not None:
        env["CUTRACER_CHANNEL_RECORDS"] = str(channel_records)
    if kernel_events is not None:
        env["CUTRACER_KERNEL_EVENTS"] = kernel_events
    if dump_cubin:
        env["CUTRACER_DUMP_CUBIN"] = "1"
    env["CUTRACER_TRACE_SIZE_LIMIT_MB"] = str(trace_size_limit_mb)
    env["CUTRACER_KERNEL_TIMEOUT_S"] = str(kernel_timeout_s)
    env["CUTRACER_NO_DATA_TIMEOUT_S"] = str(no_data_timeout_s)

    # NVBit launches nvdisasm after importlib's resource context has exited.
    # Zip/PEX resources therefore need a persistent path, not the temporary
    # path returned directly by resources.as_file(). Set the absolute override
    # as well as PATH so both NVBit and other CUDA-tool consumers resolve the
    # host-architecture binaries bundled in this CUTracer package. NVBit has an
    # explicit NVDISASM override but invokes cuobjdump by name, so cuobjdump is
    # intentionally provided through PATH.
    cuda_tools = _resolve_bundled_cuda_tools()
    if cuda_tools is not None:
        env["NVDISASM"] = str(cuda_tools.nvdisasm)
        env["PATH"] = os.pathsep.join(
            p for p in (str(cuda_tools.bin_dir), env.get("PATH")) if p
        )

    return env


# Compatibility for existing callers while the public API rolls through the stack.
_build_cutracer_env = build_cutracer_env


def _resolve_and_build_env(
    config: InstrumentationConfig,
    *,
    reject_inherited_injection: bool = True,
) -> tuple[str, dict]:
    """Resolve cutracer.so and assemble the CUTracer environment for a config.

    Shared by the ``trace`` CLI path and the programmatic run path so both
    resolve the library and build env vars identically. Programmatic callers
    pass ``reject_inherited_injection=False`` because they intentionally own
    the child environment and overwrite CUDA_INJECTION64_PATH; the CLI path
    keeps the stricter inherited-environment diagnostic for direct invocations.
    """
    so_path = resolve_cutracer_so(
        config.cutracer_so,
        reject_inherited_injection=reject_inherited_injection,
    )
    run_env = build_cutracer_env(
        cutracer_so=so_path,
        instrument=config.instrument,
        analysis=config.analysis,
        kernel_filters=config.kernel_filters,
        instr_categories=config.instr_categories,
        trace_format=config.trace_format,
        output_dir=config.output_dir,
        verbose=config.verbose,
        zstd_level=config.zstd_level,
        delay_ns=config.delay_ns,
        delay_min_ns=config.delay_min_ns,
        delay_enable_prob=config.delay_enable_prob,
        delay_mode=config.delay_mode,
        delay_cluster_cta_id=config.delay_cluster_cta_id,
        delay_warpgroup_id=config.delay_warpgroup_id,
        delay_warp_mask=config.delay_warp_mask,
        delay_patterns=config.delay_patterns,
        delay_dump_path=config.delay_dump_path,
        delay_load_path=config.delay_load_path,
        cpu_callstack=config.cpu_callstack,
        channel_records=config.channel_records,
        kernel_events=config.kernel_events,
        dump_cubin=config.dump_cubin,
        trace_size_limit_mb=config.trace_size_limit_mb,
        kernel_timeout_s=config.kernel_timeout_s,
        no_data_timeout_s=config.no_data_timeout_s,
        base_env=config.base_env,
    )
    return so_path, run_env


def run_instrumented_target(
    argv: Sequence[str],
    config: InstrumentationConfig,
    *,
    runner: Optional[RunTarget] = None,
) -> "subprocess.CompletedProcess[str]":
    """Run a target with the CUTracer library bundled with this Python package."""
    if not argv:
        raise ValueError("instrumented target argv must not be empty")

    _so_path, run_env = _resolve_and_build_env(config, reject_inherited_injection=False)
    command: object
    if config.shell:
        import shlex

        command = shlex.join(argv)
    else:
        command = list(argv)
    run = runner or subprocess.run
    return run(
        command,
        shell=config.shell,
        env=run_env,
        cwd=config.cwd,
        timeout=config.timeout,
        capture_output=config.capture_output,
        text=True,
        check=False,
    )


def _print_config_summary(env: dict) -> None:
    """Print a summary of the active CUTracer configuration."""
    cutracer_keys = [
        "CUDA_INJECTION64_PATH",
        "NVDISASM",
        "CUTRACER_INSTRUMENT",
        "CUTRACER_ANALYSIS",
        "KERNEL_FILTERS",
        "CUTRACER_INSTR_CATEGORIES",
        "CUTRACER_TRACE_FORMAT",
        "CUTRACER_OUTPUT_DIR",
        "CUTRACER_DUMP_CUBIN",
        "TOOL_VERBOSE",
        "CUTRACER_ZSTD_LEVEL",
        "CUTRACER_DELAY_NS",
        "CUTRACER_DELAY_MIN_NS",
        "CUTRACER_DELAY_ENABLE_PROB",
        "CUTRACER_DELAY_MODE",
        "CUTRACER_CLUSTER_CTA_ID",
        "CUTRACER_DELAY_WARPGROUP_ID",
        "CUTRACER_DELAY_WARP_MASK",
        "CUTRACER_DELAY_PATTERNS",
        "CUTRACER_DELAY_DUMP_PATH",
        "CUTRACER_DELAY_LOAD_PATH",
        "CUTRACER_CPU_CALLSTACK",
        "CUTRACER_CHANNEL_RECORDS",
        "CUTRACER_KERNEL_EVENTS",
        "CUTRACER_TRACE_SIZE_LIMIT_MB",
        "CUTRACER_KERNEL_TIMEOUT_S",
        "CUTRACER_NO_DATA_TIMEOUT_S",
    ]
    click.echo("=" * 60)
    click.echo("CUTracer Configuration:")
    for key in cutracer_keys:
        if key in env:
            click.echo(f"  {key} = {env[key]}")
    click.echo("=" * 60)


# Common options shared between trace and report commands
_CUTRACER_OPTIONS = [
    click.option(
        "--instrument",
        "-i",
        default=None,
        help="Instrumentation type(s): opcode_only, reg_trace, mem_addr_trace, "
        "mem_value_trace, tma_trace, random_delay. "
        "If omitted, CUTracer acts as a kernel launch logger (no trace files).",
    ),
    click.option(
        "--analysis",
        "-a",
        default=None,
        help="Analysis type(s): proton_instr_histogram, deadlock_detection, random_delay",
    ),
    click.option(
        "--kernel-filters",
        "-k",
        default=None,
        help="Comma-separated kernel name substring filters",
    ),
    click.option(
        "--instr-categories",
        default=None,
        help="Instruction category filters: mma, tma, sync",
    ),
    click.option(
        "--trace-format",
        default=None,
        help="Trace format: text (0), zstd (1), ndjson (2, default), clp (3)",
    ),
    click.option(
        "--output-dir",
        "--trace-output-dir",
        "-o",
        default=None,
        help="Output directory for trace files",
    ),
    click.option(
        "--verbose",
        "-v",
        type=int,
        default=None,
        help="Verbosity level (0/1/2)",
    ),
    click.option(
        "--cutracer-so",
        default=None,
        help="Explicit path to cutracer.so (overrides buck2 build)",
    ),
    click.option(
        "--zstd-level",
        type=int,
        default=None,
        help="Zstd compression level (1-22)",
    ),
    click.option(
        "--delay-ns",
        type=int,
        default=None,
        help="Max delay in nanoseconds for random_delay instrumentation",
    ),
    click.option(
        "--delay-min-ns",
        type=int,
        default=None,
        help="Min delay in nanoseconds (floor for random mode, default: 0). "
        "Setting min > 0 ensures every thread gets at least this much delay.",
    ),
    click.option(
        "--delay-enable-prob",
        type=click.FloatRange(0.0, 1.0),
        default=None,
        help="Probability per PC of enabling delay during recording (0.0-1.0, default 0.5). "
        "Set to 1.0 for deterministic injection (strongly recommended with "
        "--delay-warpgroup-id / --delay-warp-mask, where the default 0.5 gate halves "
        "the active warpgroup PCs). No effect in replay mode.",
    ),
    click.option(
        "--delay-mode",
        type=click.Choice(["random", "fixed", "cluster", "cluster_fixed"]),
        default=None,
        help="Delay mode (combines distribution and CTA targeting): "
        "'random' = per-thread random delay, all CTAs (default); "
        "'fixed' = same delay for all threads, all CTAs (often masks races); "
        "'cluster' = per-thread random delay, one CTA per cluster (exposes inter-CTA sync issues); "
        "'cluster_fixed' = fixed delay, one CTA per cluster (per-CTA timing skew without intra-CTA jitter).",
    ),
    click.option(
        "--delay-cluster-cta-id",
        type=int,
        default=None,
        help="Cluster mode only: force every instrumentation point to delay this CTA "
        "index in every cluster (e.g. 0 = always slow CTA 0). Default: random per point. "
        "Useful for deterministic A/B bisection of inter-CTA sync issues. "
        "Precedence: when set together with --delay-load-path, this override wins over "
        "the per-point cluster_seed in the replay config — replay is no longer bit-identical "
        "to the recording. Unset (or omit) for exact replay.",
    ),
    click.option(
        "--delay-warpgroup-id",
        type=int,
        default=None,
        help="Warp-targeted delay: warpgroup index (>= 0 selects warps [4N..4N+3]). "
        "Use to stall one warpgroup while peers race ahead (exposes warpgroup-scheduler races, "
        "e.g. TMEM dealloc on Blackwell). Resolved to a 32-bit warp mask on the host side. "
        "Wins over --delay-warp-mask when both are set (a startup warning is emitted).",
    ),
    click.option(
        "--delay-warp-mask",
        type=str,
        default=None,
        help="Warp-targeted delay: explicit bitmask of CTA-local warp ids (hex or decimal, "
        "e.g. '0xF' for warps 0-3, '0xF0' for warps 4-7). Bit N == 1 means warp N is delayed. "
        "Warps >= 32 are silently skipped. Accepts strings so '0xF' works; C++ side parses "
        "via strtoul(_, nullptr, 0). Ignored when --delay-warpgroup-id is also set.",
    ),
    click.option(
        "--delay-patterns",
        default=None,
        help="Comma-separated SASS instruction substrings for delay injection "
        "(overrides built-in patterns). Example: 'SYNCS.EXCH' for mbarrier init only",
    ),
    click.option(
        "--delay-dump-path",
        default=None,
        help="Output path to dump delay config JSON for replay",
    ),
    click.option(
        "--delay-load-path",
        default=None,
        help="Load delay config JSON for replay mode",
    ),
    click.option(
        "--cpu-callstack",
        type=click.Choice(["auto", "auto_gil", "pytorch", "backtrace", "0", "1"]),
        default=None,
        help="CPU call stack mode: auto (default), auto_gil (acquire GIL for Triton), "
        "pytorch, backtrace, 0=disabled",
    ),
    click.option(
        "--channel-records",
        type=int,
        default=None,
        help="Channel buffer capacity in records (default: auto/4MB). "
        "Set to 1 for per-record flush (useful for hang debugging)",
    ),
    click.option(
        "--kernel-events",
        type=click.Choice(["0", "dedup", "full", "nostack"]),
        default=None,
        help="Kernel events recording: 0=disabled (default), dedup=callstack dedup, "
        "full=full callstack per launch, nostack=metadata only",
    ),
    click.option(
        "--dump-cubin/--no-dump-cubin",
        default=None,
        help="Dump cubin files for instrumented kernels (for SASS disassembly via nvdisasm). "
        "Auto-enabled when --instrument is set; use --no-dump-cubin to override.",
    ),
    click.option(
        "--trace-size-limit-mb",
        type=int,
        default=0,
        show_default=True,
        help="Maximum trace file size in MB (0 = disabled, default: 0). "
        "Stops tracing when any file exceeds this limit; "
        "kernel execution continues normally.",
    ),
    click.option(
        "--kernel-timeout-s",
        type=int,
        default=0,
        show_default=True,
        help="Kernel execution timeout in seconds (0 = disabled, default: 0). "
        "Auto-terminates any kernel running longer than this value. "
        "Independent of deadlock detection.",
    ),
    click.option(
        "--no-data-timeout-s",
        type=int,
        default=15,
        show_default=True,
        help="No-data timeout in seconds for silent hang detection (default: 15). "
        "Terminates the process when no trace data arrives for this duration. "
        "Independent of deadlock detection (does not require -a deadlock_detection). "
        "Set to 0 to disable.",
    ),
]


def cutracer_options(func):
    """Decorator to apply all common CUTracer options to a click command."""
    for option in reversed(_CUTRACER_OPTIONS):
        func = option(func)
    return func


@click.command(
    name="trace",
    context_settings={"ignore_unknown_options": True},
)
@cutracer_options
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED, required=True)
def trace_command(
    instrument: Optional[str],
    analysis: Optional[str],
    kernel_filters: Optional[str],
    instr_categories: Optional[str],
    trace_format: Optional[str],
    output_dir: Optional[str],
    verbose: Optional[int],
    cutracer_so: Optional[str],
    zstd_level: Optional[int],
    delay_ns: Optional[int],
    delay_min_ns: Optional[int],
    delay_enable_prob: Optional[float],
    delay_mode: Optional[str],
    delay_cluster_cta_id: Optional[int],
    delay_warpgroup_id: Optional[int],
    delay_warp_mask: Optional[str],
    delay_patterns: Optional[str],
    delay_dump_path: Optional[str],
    delay_load_path: Optional[str],
    cpu_callstack: Optional[str],
    channel_records: Optional[int],
    kernel_events: Optional[str],
    dump_cubin: Optional[bool],
    trace_size_limit_mb: int,
    kernel_timeout_s: int,
    no_data_timeout_s: int,
    cmd: tuple,
) -> None:
    """Trace a CUDA application with CUTracer instrumentation.

    Sets up CUTracer environment variables and runs the specified command.
    The cutracer.so shared library is bundled via buck2 resources if not
    explicitly provided.

    When --instrument is omitted, CUTracer acts as a lightweight kernel launch
    logger: kernel names, grid/block dims, and shared memory usage are printed
    but no trace files are created and no instrumentation overhead is added.

    \b
    Examples:
      cutracer trace --instrument=tma_trace -- ./vectoradd
      cutracer trace -i tma_trace --instr-categories=tma --trace-format=2 -- ./my_app
      cutracer trace -i reg_trace --kernel-filters=matmul_kernel -- python -m pytest test.py
      cutracer trace -i tma_trace --cutracer-so=/path/to/cutracer.so -- ./app
      cutracer trace -- ./my_app  # kernel launch logger only (no instrumentation)
    """
    if not cmd:
        raise click.UsageError(
            "No command specified. Usage: cutracer trace [OPTIONS] -- COMMAND"
        )

    # `ignore_unknown_options=True` lets us forward arbitrary args to the wrapped
    # command after `--`, but it also silently turns a typo'd flag (e.g. `-o`
    # before `-o` was added) into the first token of `cmd`. The wrapped shell
    # then chokes with a cryptic message like `/bin/sh: - : invalid option`.
    # Catch that case here and point the user at the likely fix.
    if cmd[0].startswith("-"):
        raise click.UsageError(
            f"Unrecognized option {cmd[0]!r} before '--'. "
            "If this is meant for the wrapped command, separate CUTracer flags "
            "from the command with '--', e.g.: "
            "cutracer trace [OPTIONS] -- COMMAND [ARGS...]. "
            "Run `cutracer trace --help` to see available options."
        )

    # `buck2 test` dispatches the actual test binary through TPX in a sandbox
    # that scrubs environment variables, so CUDA_INJECTION64_PATH never reaches
    # the GPU process and cutracer.so is never loaded — no logs, silent no-op.
    # Warn early so users don't waste a full test run wondering where the logs
    # went. `buck2 run <test_target>` works because the test binary inherits
    # our env directly.
    if cmd[0] in ("buck", "buck2") and len(cmd) > 1 and cmd[1] == "test":
        click.echo(
            f"WARNING: Wrapping `{cmd[0]} test` is unsupported. TPX runs the "
            "test binary in a sandbox that strips CUDA_INJECTION64_PATH, so "
            "cutracer.so will not be loaded and no traces will be produced. "
            f"Use `{cmd[0]} run <test_target>` instead, or run the test binary "
            "from buck-out directly.",
            err=True,
        )

    # Auto-enable cubin dumping when instrumentation is active, unless
    # the user explicitly passed --no-dump-cubin.
    if dump_cubin is None:
        dump_cubin = instrument is not None

    config = InstrumentationConfig(
        cutracer_so=cutracer_so,
        instrument=instrument,
        analysis=analysis,
        kernel_filters=kernel_filters,
        instr_categories=instr_categories,
        trace_format=trace_format,
        output_dir=output_dir,
        verbose=verbose,
        zstd_level=zstd_level,
        delay_ns=delay_ns,
        delay_min_ns=delay_min_ns,
        delay_enable_prob=delay_enable_prob,
        delay_mode=delay_mode,
        delay_cluster_cta_id=delay_cluster_cta_id,
        delay_warpgroup_id=delay_warpgroup_id,
        delay_warp_mask=delay_warp_mask,
        delay_patterns=delay_patterns,
        delay_dump_path=delay_dump_path,
        delay_load_path=delay_load_path,
        cpu_callstack=cpu_callstack,
        channel_records=channel_records,
        kernel_events=kernel_events,
        dump_cubin=dump_cubin,
        trace_size_limit_mb=trace_size_limit_mb,
        kernel_timeout_s=kernel_timeout_s,
        no_data_timeout_s=no_data_timeout_s,
        capture_output=False,
        shell=True,
    )

    so_path, run_env = _resolve_and_build_env(config)

    _print_config_summary(run_env)

    import shlex

    # The CLI preserves shell execution for environment-assignment compatibility.
    # Operators are intentionally quoted as argv tokens; callers that need pipes
    # or chaining must pass an explicit shell command such as `sh -c 'a | b'`.
    import sys

    cmd_string = shlex.join(cmd)
    click.echo(f"Running: {cmd_string}")
    click.echo("=" * 60)

    result = run_instrumented_target(cmd, replace(config, cutracer_so=so_path))
    sys.exit(result.returncode)
