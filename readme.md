# CUTracer

CUTracer is a CUDA binary instrumentation tool built on [NVBit](https://github.com/NVlabs/NVBit). It cleanly separates lightweight data collection (instrumentation) from host-side processing (analysis). Typical workflows include per-warp instruction histograms (delimited by GPU clock reads) and kernel hang detection.

## Features

-   NVBit-powered, runtime attach via `CUDA_INJECTION64_PATH` (no app rebuild needed)
-   Multiple instrumentation modes: opcode-only, register trace, memory trace, random delay
-   Built-in analyses:
    -   Instruction Histogram (for Proton/Triton workflows)
    -   Deadlock/Hang Detection
    -   Data Race Detection
-   CUDA Graph and stream-capture aware flows
-   Deterministic kernel log file naming and CSV outputs

## Requirements

All requirements are aligned with NVBit.

Unique requirements:
- **libzstd**: Required for trace compression

## Installation

1. Clone the repository:

```bash
cd ~
git clone git@github.com:facebookexperimental/CUTracer.git
cd CUTracer
```

2. Install system dependencies (libzstd static library for self-contained builds):

```bash
# Ubuntu/Debian
# On most Ubuntu/Debian systems, libzstd-dev provides both shared and static libs (libzstd.a).
# You can verify this with: dpkg -L libzstd-dev | grep 'libzstd.a'
# If your distribution does not ship the static library in libzstd-dev, you may need to
# build zstd from source or install a distro-specific static libzstd package.
sudo apt-get install libzstd-dev

# CentOS/RHEL/Fedora (static library for portable builds)
sudo dnf install libzstd-static

# If static library is not available, the build will fall back to dynamic linking
# and display a warning. The resulting binary will not be self-contained.
```

3. Download third-party dependencies:

```bash
./install_third_party.sh
```

This will download:
- NVBit (NVIDIA Binary Instrumentation Tool)
- nlohmann/json (JSON library for C++)

4. Build the tool:

```bash
make -j$(nproc)
```

## Quickstart

### 1. Install the Python CLI

```bash
cd ~/CUTracer/python
pip install .
```

### 2. Run your CUDA app with CUTracer

```bash
# Option A: Set CUTRACER_LIB_PATH once (recommended)
export CUTRACER_LIB_PATH=~/CUTracer/lib
cutracer trace -i tma_trace -- ./your_app

# Option B: Specify cutracer.so explicitly
cutracer trace -i tma_trace --cutracer-so ~/CUTracer/lib/cutracer.so -- ./your_app

# Option C: Run from the CUTracer project root (auto-discovers ./lib/cutracer.so)
cd ~/CUTracer
cutracer trace -i tma_trace -- ./your_app

# Option D: Kernel launch logger only (no instrumentation, no trace files)
cutracer trace -- ./your_app
```

### 3. Analyze the output

```bash
cutracer analyze warp-summary output.ndjson
cutracer query output.ndjson --filter "warp=24"
cutracer validate output.ndjson
```

> **Note**: You can also use CUTracer without the Python CLI by setting the
> `CUDA_INJECTION64_PATH` environment variable directly:
> ```bash
> CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so ./your_app
> ```

## Configuration (env vars)

-   `CUTRACER_INSTRUMENT`: comma-separated modes: `opcode_only`, `reg_trace`, `mem_trace`, `random_delay`
-   `CUTRACER_ANALYSIS`: comma-separated analyses: `proton_instr_histogram`, `deadlock_detection`, `random_delay`
    -   Enabling `proton_instr_histogram` auto-enables `opcode_only`
    -   Enabling `deadlock_detection` auto-enables `reg_trace`
    -   Enabling `random_delay` auto-enables `random_delay` instrumentation; also requires `CUTRACER_DELAY_NS` to be set
-   `KERNEL_FILTERS`: comma-separated substrings matching unmangled or mangled kernel names
-   `INSTR_BEGIN`, `INSTR_END`: static instruction index gate during instrumentation
-   `TOOL_VERBOSE`: 0/1/2
-   `CUTRACER_TRACE_FORMAT`: trace output format. Accepts string names or numeric values (replaces the legacy `TRACE_FORMAT_NDJSON` env var, which is still accepted for backward compatibility)
    -   **ndjson** or 2 (default): NDJSON uncompressed (`.ndjson`)
    -   text (or 0): Plain text (`.log`, legacy format, verbose)
    -   zstd (or 1): NDJSON+Zstd compressed (`.ndjson.zst`, ~12x compression, 92% space savings)
    -   clp (or 3): CLP Archive (`.clp`)
-   `CUTRACER_ZSTD_LEVEL`: Zstd compression level (1-22, default 9)
    -   Lower values (1-3): Faster compression, slightly larger output
    -   Higher values (19-22): Maximum compression, slower but smallest output
    -   Default of 9 provides balanced compression speed and ratio
- `CUTRACER_DELAY_NS`: Max delay value in nanoseconds for `random_delay` analysis (required when `random_delay` is enabled)
- `CUTRACER_DELAY_MIN_NS`: Minimum delay in nanoseconds — floor for random mode (default: 0). Must be ≤ `CUTRACER_DELAY_NS`
- `CUTRACER_DELAY_MODE`: Delay mode: `random` (per-thread random, default), `fixed` (same for all threads), `cluster` (one CTA per cluster), `cluster_fixed` (fixed delay, one CTA per cluster)
- `CUTRACER_DELAY_DUMP_PATH`: Output path for delay config JSON file (for recording instrumentation patterns)
- `CUTRACER_DELAY_LOAD_PATH`: Input path for delay config JSON file (for replay mode - deterministic reproduction)
- `CUTRACER_DELAY_PATTERNS`: Comma-separated SASS instruction substrings for delay injection (overrides built-in patterns). Use `"*"` to match all instructions
- `CUTRACER_DELAY_ENABLE_PROB`: Per-PC enable probability (0.0-1.0, default: 0.5). Use `1.0` with warp targeting
- `CUTRACER_DELAY_WARPGROUP_ID`: Warp-targeted delay: warpgroup index (>= 0 selects warps `[4N..4N+3]`, -1 = disabled)
- `CUTRACER_DELAY_WARP_MASK`: Warp-targeted delay: hex bitmask of CTA-local warp IDs (e.g. `0xF` for warps 0-3, 0 = disabled)
- `CUTRACER_OUTPUT_DIR`: Output directory for all CUTracer files (trace files and log files). Defaults to the current directory. The directory must exist and be writable.
- `CUTRACER_CPU_CALLSTACK`: CPU call stack capture mode at each kernel launch (default: `auto`)
    - `auto` (default): Prefer PyTorch CapturedTraceback for Python frames, fallback to C++ backtrace if Python/PyTorch is unavailable
    - `pytorch`: Force PyTorch CapturedTraceback only (returns empty if unavailable)
    - `backtrace`: Force C++ backtrace only (original behavior)
    - `1`: Same as `auto` (backward compatible)
    - `0`: Disable call stack capture
    - When enabled, the `kernel_metadata` trace event includes a `cpu_callstack` array and a `cpu_callstack_source` field (`"pytorch"` or `"backtrace"`) indicating the capture method used
- `CUTRACER_KERNEL_TIMEOUT_S`: Kernel execution time limit in seconds (default: 0 = disabled)
    - Terminates the process with SIGTERM when a kernel runs longer than this value
    - Acts as a general safety valve, independent of deadlock detection (does not require `-a deadlock_detection`)
- `CUTRACER_NO_DATA_TIMEOUT_S`: No-data hang detection timeout in seconds (default: 15)
    - Terminates the process with SIGTERM when no trace data arrives for this duration
    - Acts as a general safety valve, independent of deadlock detection (does not require `-a deadlock_detection`)
    - Catches "silent" hangs where all warps are blocked on synchronization primitives with zero trace output
    - Works whether the kernel went silent after producing some data, or never produced any data at all
    - When `-a deadlock_detection` is also active, prints detailed warp status summary before termination
    - Set to 0 to disable
- `CUTRACER_TRACE_SIZE_LIMIT_MB`: Maximum trace file size in MB (default: 0 = disabled)
    - When any trace file exceeds this limit, tracing is stopped for that kernel; kernel execution continues normally
    - Useful for preventing runaway trace files from filling disk (e.g., during deadlocked kernels)

**Notes:**
- The tool sets `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` to simplify channel memory handling.
- Multiple analyses can be combined (e.g., `CUTRACER_ANALYSIS=proton_instr_histogram,deadlock_detection`). Each analysis auto-enables its required instrumentation mode.

## Analyses

### Instruction Histogram (proton_instr_histogram)

-   Counts SASS instruction mnemonics per warp within regions delimited by clock reads (start/stop model; nested regions not supported)
-   Output: one CSV per kernel launch with columns `warp_id,region_id,instruction,count`

Example (Triton/Proton + IPC):

```bash
cd ~/CUTracer/tests/proton_tests

# 1) Collect histogram with CUTracer
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_ANALYSIS=proton_instr_histogram \
KERNEL_FILTERS=add_kernel \
python ./vector-add-instrumented.py

# 2) Run without CUTracer to generate a clean Chrome trace
python ./vector-add-instrumented.py

# 3) Merge and compute IPC
python ~/CUTracer/scripts/parse_instr_hist_trace.py \
  --chrome-trace ./vector.chrome_trace \
  --cutracer-trace ./kernel_*_add_kernel_hist.csv \
  --cutracer-log ./cutracer_main_*.log \
  --output vectoradd_ipc.csv
```

### Deadlock / Hang Detection (deadlock_detection)

-   Detects sustained hangs by identifying warps stuck in stable PC loops; logs and issues SIGTERM→SIGKILL if sustained
-   Requires `reg_trace` (auto-enabled)

Example (intentional loop):

```bash
cd ~/CUTracer/tests/hang_test
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_ANALYSIS=deadlock_detection \
python ./test_hang.py
```

### Data Race Detection (random_delay)

-   Data races depend on thread scheduling and timing — buggy code may appear correct by luck.
    This analysis exposes hidden races by injecting random delays before **synchronization-related SASS instructions** (e.g., `BAR`, `MEMBAR`, `ATOM`, `RED`), disrupting the normal timing and forcing latent races to manifest as observable failures.
-   Each instrumentation point is randomly enabled/disabled (50% probability)
-   Two delay modes:
    -   **`random` (default):** Each thread gets a random delay in `[0, CUTRACER_DELAY_NS]` using GPU-side xorshift32 PRNG seeded with `threadIdx/blockIdx/clock`. Creates per-thread timing skew that amplifies data races. **Recommended.**
    -   **`fixed`:** All threads get the same delay. Preserves relative timing between threads and often *masks* races rather than exposing them. Not recommended for race detection.
-   Requires `CUTRACER_DELAY_NS` to be set. The `random_delay` instrumentation mode is auto-enabled.

Example:

```bash
CUTRACER_DELAY_NS=100000 \
CUTRACER_ANALYSIS=random_delay \
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
python3 your_kernel.py
```

#### Delay Dump and Replay

CUTracer supports dumping delay configurations to JSON for deterministic reproduction of data races:

- **Dump mode**: Set `CUTRACER_DELAY_DUMP_PATH` to save the random instrumentation pattern to a JSON file
- **Replay mode**: Set `CUTRACER_DELAY_LOAD_PATH` to load a saved config and reproduce the exact same delay pattern

**Note**: You cannot use both at the same time.

**Workflow**:
1. Run with `CUTRACER_DELAY_DUMP_PATH=/tmp/config.json` to record the delay pattern
2. When a failure occurs, save the config file
3. Replay with `CUTRACER_DELAY_LOAD_PATH=/tmp/config.json` to reproduce deterministically
4. Reduce with `cutracer reduce` to find the minimal set of delay points (see below)

#### Reduce (Delta Debugging)

The `reduce` subcommand finds the minimal set of delay injection points that trigger a data race. Two strategies:

-   **`linear`**: Tests each point one by one. O(N) test runs. Simple but slow.
-   **`bisect`**: ddmin-style bisection. Splits points in half and recursively narrows down. Typically O(log N) iterations. **Recommended for large configs.**

Use `--confidence-runs N` (odd number) for majority voting when the race is probabilistic.

```bash
# Bisection reduction (fast)
cutracer reduce -c config.json -t ./test_race.sh --strategy bisect --confidence-runs 3
```

The test script convention follows `llvm-reduce`: exit 0 = interesting (race occurred), exit 1+ = not interesting (no race).

## Examples

The [`examples/`](examples/) directory contains reference trace outputs for common workflows:

-   **[Proton Trace](examples/proton_trace/)** -- sample instruction histogram CSV, CUTracer log, and a README explaining the end-to-end proton instrumentation workflow for a Triton vector-add kernel

## Troubleshooting

-   No CSV/log: check `CUDA_INJECTION64_PATH`, `KERNEL_FILTERS`, and write permissions
-   Empty histogram: ensure kernels emit clock instructions (e.g., Triton `pl.scope`)
-   High overhead: prefer opcode-only; narrow filters; use `INSTR_BEGIN/INSTR_END`
-   CUDA Graph/stream capture: data is flushed at `cuGraphLaunch` exit; ensure stream sync
-   IPC merge issues: resolve warp mismatches and kernel hash ambiguity with parser flags

## License

This repository contains code under the MIT license (Meta) and the BSD-3-Clause license (NVIDIA). See [LICENSE](LICENSE) and [LICENSE-BSD](LICENSE-BSD) for details.

## 📚 Documentation

The full project documentation lives in [`docs/`](docs/) and is automatically
synced to the [GitHub Wiki](https://github.com/facebookexperimental/CUTracer/wiki)
on every push to `main` via
[`.github/workflows/sync-wiki.yml`](.github/workflows/sync-wiki.yml).

**Edit `docs/*.md`, not the wiki directly** — direct wiki edits will be
overwritten on the next sync.

Key topics: [Quickstart](docs/Quickstart.md), [Concepts](docs/Concepts.md),
[Configuration](docs/Configuration.md),
[Instrumentation Modes](docs/Instrumentation-Modes.md),
[Outputs and File Formats](docs/Outputs-and-File-Formats.md),
[API and Data Structures](docs/API-and-Data-Structures.md),
[Analyses](docs/Analyses.md),
[Post-processing: IPC Merge](docs/Post-processing-IPC-Merge.md),
[Triton/Proton Integration](docs/Triton_Proton-Integration.md),
[Architecture](docs/Architecture.md),
[Developer Guide](docs/Developer-Guide.md),
[Build, Test, and CI](docs/Build-Test-and-CI.md),
[Troubleshooting](docs/Troubleshooting.md), [FAQ](docs/FAQ.md).
