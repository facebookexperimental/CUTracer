# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer CLI entry point.

Provides command-line interface for trace validation, query, and analysis.
"""

import sys

import click
from cutracer.analyze.cli import analyze_command
from cutracer.compute_sanitizer.cli import compute_sanitizer_command
from cutracer.oracle.cli import oracle_command
from cutracer.query.cli import query_command, sass_command
from cutracer.reduce.cli import reduce_command
from cutracer.runtime_version import get_runtime_version
from cutracer.service.cli import (
    diagnose_command,
    diagnose_resume_command,
    diagnose_status_command,
)
from cutracer.service.run_cli import (
    diagnose_run_advance_command,
    diagnose_run_command,
    diagnose_run_result_command,
    diagnose_run_resume_command,
    diagnose_run_status_command,
    diagnose_submit_command,
)
from cutracer.shared_vars import is_fbcode
from cutracer.stress.cli import stress_command
from cutracer.validation.cli import compare_command, validate_command

if is_fbcode():
    from cutracer.fb.runner import trace_command
else:
    from cutracer.runner import trace_command

# The top-level help text is selected at import time by build flavor.
# The public text lives here; the internal text lives in
# `cutracer.fb.help`, which the public package excludes (see
# pyproject.toml), so internal devices, install paths, and skill
# bundle details never publish. Never inline internal help text in
# this file. Keep both texts in sync with the commands registered
# below.
_OSS_HELP = """\
CUTracer: SASS-level GPU kernel tracing and debugging for NVIDIA CUDA.

CUTracer injects lightweight instrumentation into CUDA kernels at the
SASS level with NVBit (no application rebuild), records per-warp
execution traces, and analyzes them offline. Use it to reconstruct
TMA/MMA operations, expose data races with random-delay injection,
diagnose hangs and deadlocks, localize illegal memory accesses, and
inspect where wrong or NaN values come from.

AMD ROCm and MTIA workflows ship in Meta's internal build; this build
covers NVIDIA. Run `cutracer <command> --help` for a command's flags,
and use the same cutracer build for capture and analysis in one
session: trace files are versioned with the tool that wrote them.
Never set CUDA_INJECTION64_PATH yourself; `cutracer trace` resolves
and injects cutracer.so for you.
"""

_OSS_EPILOG = """\
\b
Supported devices
  NVIDIA CUDA, SM90 (Hopper) and SM100 (Blackwell) and later.
  Capture with `cutracer trace`; analysis of saved traces is CPU-only.

\b
What you can do
  Trace capture - record opcode, register, memory-address,
    memory-value, or TMA execution with -i; omit -i for a
    kernel-launch logger that writes no trace files.
  Inspection - `query` filters, groups, and formats traces;
    `validate` checks schema; `compare` cross-checks text vs JSON;
    `sass` disassembles a cubin.
  TMA/MMA - `analyze tma` rebuilds tensor descriptors (shape,
    strides, dtype, swizzle; optional GEMM inference and Python
    source); `analyze mma` decodes tensor-core operations.
  Instruction histogram - `trace -a proton_instr_histogram` counts
    SASS mnemonics per warp per clock-delimited region (Proton IPC).
  Data races - inject per-thread random delays before sync
    instructions (`trace -a random_delay`) so timing-dependent races
    fail observably; find a triggering config with `stress`, replay
    it deterministically, and shrink it with `reduce`.
  Hangs - `trace -a deadlock_detection` classifies warps and ends a
    hung run; --kernel-timeout-s and --no-data-timeout-s (default
    15s) are watchdogs even without that analysis.
  Sanitizer - `compute-sanitizer` runs NVIDIA Compute Sanitizer in a
    separate run (it cannot share a process with NVBit tracing).
  Diagnosis - `diagnose` runs a bounded sanitizer, stress, trace,
    and reasoning session and writes one report.

\b
Install
  Nightly CLI (matches a cutracer.so built from main):
    pip install --pre cutracer
  Stable CLI (build cutracer.so from the matching tag, e.g. v0.3.0):
    pip install cutracer==0.3.0
  From a source checkout: `cd python && pip install .`
    (CLP traces, x86_64 only: pip install 'cutracer[clp]')
  cutracer.so is required by `trace` and is not in the wheel:
    ./install_third_party.sh && make -j$(nproc)   # lib/cutracer.so
    Needs libzstd. Point the CLI at it with --cutracer-so,
    $CUTRACER_LIB_PATH, or run from the checkout root.
  No CUDA toolkit? `./scripts/install_cuda.sh 13.0` installs one
  without sudo (12.6, 12.8, 12.9, 13.0, 13.2, 13.4; x86_64/aarch64).

\b
Agent skills
  Canonical skills are in agentic/skills/ (investigate-kernel,
  debug-hanging-kernel, debug-illegal-access, audit-memory-model,
  report-case-study). Symlink one into your agent's skill directory
  or register agentic/skills/ as a search path; see
  agentic/README.md ("Agent discovery").

\b
Documentation and examples
  Wiki: https://github.com/facebookexperimental/CUTracer/wiki
    (edited through docs/ in this repo)
  Worked reproducers: examples/ (proton_trace, tlx_tmem_war,
    tlx_gemm_clc, ck_fmha_kv_lds_war)
  Issues: https://github.com/facebookexperimental/CUTracer/issues

\b
Examples
  cutracer trace -i tma_trace -- ./vectoradd
  cutracer trace -i tma_trace --instr-categories=tma -- python my_test.py
  cutracer trace -- ./your_app  # kernel launch logger, no trace files
  cutracer validate kernel_trace.ndjson
  cutracer validate kernel_trace.ndjson.zst --verbose
  cutracer validate trace.log --format text
  cutracer query trace.ndjson --filter "warp=24"
  cutracer query trace.ndjson -f "pc=0x43d0;warp=24"
  cutracer query trace.ndjson --group-by warp --count
  cutracer sass kernel.cubin -o kernel.sass
  cutracer analyze warp-summary trace.ndjson
  cutracer trace -a deadlock_detection --channel-records 1 -- python my_test.py
  cutracer trace -a random_delay --delay-ns 10000 --delay-dump-path cfg.json -- ./app
  cutracer trace -a random_delay --delay-ns 10000 --delay-load-path cfg.json -- ./app
  cutracer stress -k my_kernel -o /tmp/stress -- ./oracle.sh
  cutracer reduce -c cfg.json -t ./test_race.sh --strategy bisect --confidence-runs 3
  cutracer compute-sanitizer --tool racecheck -- python my_test.py
  cutracer diagnose work_unit.json --toolchain-version <immutable-id> --out report.json
"""

if is_fbcode():
    from cutracer.fb.help import FB_EPILOG as _MAIN_EPILOG, FB_HELP as _MAIN_HELP
else:
    _MAIN_HELP, _MAIN_EPILOG = _OSS_HELP, _OSS_EPILOG


@click.group(help=_MAIN_HELP, epilog=_MAIN_EPILOG, invoke_without_command=True)
@click.version_option(version=get_runtime_version(), prog_name="cutracer")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Run the CUTracer CLI (help text is selected as `_MAIN_HELP`)."""
    from cutracer.shared_vars import is_fbcode

    if is_fbcode():
        from cutracer.fb.usage import usage_report_logger

        usage_report_logger()

    if ctx.invoked_subcommand is None:
        raise click.UsageError("Missing command. Run 'cutracer --help' for usage.")


# Register subcommands
main.add_command(analyze_command)
main.add_command(compare_command)
main.add_command(compute_sanitizer_command)
main.add_command(diagnose_command)
main.add_command(diagnose_resume_command)
main.add_command(diagnose_status_command)
main.add_command(diagnose_run_command)
main.add_command(diagnose_submit_command)
main.add_command(diagnose_run_resume_command)
main.add_command(diagnose_run_advance_command)
main.add_command(diagnose_run_status_command)
main.add_command(diagnose_run_result_command)
main.add_command(oracle_command)
main.add_command(query_command)
main.add_command(reduce_command)
main.add_command(sass_command)
main.add_command(stress_command)
main.add_command(trace_command)
main.add_command(validate_command)


if __name__ == "__main__":
    sys.exit(main())
