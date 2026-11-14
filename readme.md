<p align="center"><img src="logo.svg" alt="CUTracer logo" width="120"></p>

# CUTracer

[![PyPI](https://img.shields.io/pypi/v/cutracer)](https://pypi.org/project/cutracer/)

CUTracer is a debugging toolkit for NVIDIA GPU kernels. Its runtime library, `cutracer.so`, uses [NVBit](https://github.com/NVlabs/NVBit) to instrument CUDA kernels at the SASS level as they load, so the application does not have to be rebuilt. The `cutracer` command runs a program under that library to record per-warp traces, to inject delays that make timing-dependent races fail visibly, or to stop a hung kernel and report what each warp was doing. It also runs NVIDIA Compute Sanitizer, searches for and shrinks delay configs that trigger a race, and queries and validates traces.

CUTracer supports Hopper (SM90), Blackwell (SM100) and later GPUs. Analyzing saved traces needs no GPU.

## What it is for

| Problem | Start with |
|---|---|
| A kernel hangs | `cutracer trace -a deadlock_detection` classifies each warp as waiting at a barrier, looping or progressing, prints a warp report and ends the run |
| Results change between runs, or a race is suspected | `cutracer stress` injects delays until a test fails and saves the delay config that made it fail; `cutracer reduce` shrinks that config to the delay points that matter |
| Out-of-bounds or misaligned accesses, shared-memory races, uninitialized reads, barrier misuse | `cutracer compute-sanitizer` runs NVIDIA Compute Sanitizer's memcheck, racecheck, initcheck or synccheck |
| A failing test needs a first diagnosis | `cutracer diagnose` runs Compute Sanitizer (and stress, given a correctness oracle) on it, follows up with traces and writes one report |
| A kernel computes wrong values | `cutracer trace -i reg_trace` or `-i mem_value_trace` records what each instruction read or wrote, and `cutracer query` filters the records |
| You want the instruction mix of a Triton kernel region | `cutracer trace -a proton_instr_histogram` counts SASS instructions per warp between clock reads (see [Triton/Proton Integration](docs/Triton_Proton-Integration.md)) |

## Requirements

- Linux on x86_64 or aarch64, with a Hopper or newer NVIDIA GPU for capture
- A CUDA toolkit with `nvcc` to build `cutracer.so`; [`scripts/install_cuda.sh`](scripts/install_cuda.sh) installs one without root (see [Quickstart](docs/Quickstart.md))
- libzstd, preferably with its static library so that `cutracer.so` is self-contained
- NVBit 1.8.1 or newer, which `./install_third_party.sh` downloads
- Python 3.10 or newer for the `cutracer` command
- NVIDIA Compute Sanitizer, part of the CUDA toolkit, for `cutracer compute-sanitizer` and `cutracer diagnose`

## Installation

### 1. Build `cutracer.so`

```bash
git clone https://github.com/facebookexperimental/CUTracer.git
cd CUTracer

# libzstd on Ubuntu/Debian (libzstd-dev normally includes libzstd.a)
sudo apt-get install libzstd-dev
# libzstd on CentOS/RHEL/Fedora
sudo dnf install libzstd-static

./install_third_party.sh   # NVBit, nlohmann/json, RapidJSON, PicoSHA2
make -j$(nproc)            # writes lib/cutracer.so
```

`make` builds for every GPU architecture; `make ARCH=sm_90` builds for one and is faster. Without the static libzstd, the build links libzstd dynamically and prints a warning.

### 2. Install the `cutracer` command

The [`cutracer`](https://pypi.org/project/cutracer/) package on PyPI holds the Python command only, not `cutracer.so`. Trace files are versioned with the `cutracer.so` that wrote them, so use a command from the same version:

```bash
pip install --pre cutracer   # nightly build, for a cutracer.so built from main
pip install cutracer         # latest release, for a cutracer.so built from its tag (git checkout v<version>)
pip install ./python         # the command from this checkout
```

`cutracer --version` prints the installed version. Reading CLP archives (`--trace-format clp`) needs `pip install 'cutracer[clp]'`, which is available on x86_64 only.

### 3. Smoke test

`cutracer trace` finds `lib/cutracer.so` when it runs from the repository root; from anywhere else, pass `--cutracer-so /path/to/CUTracer/lib/cutracer.so`. From the repository root:

```bash
pip install torch==2.13.0 numpy --index-url https://download.pytorch.org/whl/cu130
cutracer trace -i mem_addr_trace -k triton_poi_fused -o /tmp/ct_quickstart -- python tests/py_add/test_add.py
cutracer validate "$(ls /tmp/ct_quickstart/*.ndjson | head -1)"
```

The test compiles a small kernel with `torch.compile`, and `-k triton_poi_fused` selects it.

## Usage

### Trace a kernel

`cutracer trace [flags] -- <command>` runs the command with `cutracer.so` injected. `-i` selects what to record:

| Mode | Records |
|---|---|
| `opcode_only` | PC and opcode per warp |
| `reg_trace` | Register values, before the instruction executes |
| `mem_addr_trace` | The 32 lane addresses of each memory instruction |
| `mem_value_trace` | Addresses and the values loaded or stored, after the instruction executes |
| `tma_trace` | The descriptor address and transfer parameters of each TMA instruction |
| `random_delay` | Nothing; it injects delays before synchronization instructions (see below) |

Without `-i`, CUTracer only logs kernel launches (name, grid and block dimensions, shared memory) and writes no trace files.

```bash
cutracer trace -i reg_trace -k my_kernel -o traces -- python my_test.py
cutracer query traces/<trace>.ndjson --filter "warp=24"
cutracer query traces/<trace>.ndjson --group-by sass --count --top 20
```

Each launch of a matching kernel writes `kernel_<checksum>_iter<N>_pid<pid>_<name>.ndjson`, next to the kernel's `.cubin` (`cutracer sass` disassembles it) and CUTracer's log, `cutracer_main_<time>_<pid>.log`. `-k` takes comma-separated kernel name substrings. `--instr-categories mma,tma,sync`, `INSTR_BEGIN`/`INSTR_END` and the targeted capture variables in [Configuration](docs/Configuration.md) narrow a capture further. An [instrumentation plan](docs/Instrumentation-Plans.md) captures chosen registers at exact PCs of a known CUBIN (SM100a), and `CUTRACER_RUNTIME_IDENTITY=1` writes a manifest that binds each launch to its process, module and CUBIN.

### Debug a hang

```bash
cutracer trace -a deadlock_detection --channel-records 1 -k my_kernel -o hang -- python my_test.py
cutracer analyze warp-summary hang/<trace>.ndjson
```

`-a deadlock_detection` turns on `reg_trace`, classifies each warp as `BARRIER`, `LOOPING` or `PROGRESSING`, prints a warp report and ends a hung run with SIGTERM, then SIGKILL (exit status 241 or 247). `--channel-records 1` flushes every record, so the trace reaches the moment of the hang. Without this analysis, `--no-data-timeout-s` (default 15 s) still ends a run whose kernels stop sending records, and `--kernel-timeout-s` ends a kernel that runs too long; `0` turns either off.

With the defaults the check ends a run after 3 checks about 1 s apart in which no warp made progress, so a correct kernel whose warps spin longer, for example while waiting on another rank, is ended too. Raise `--hang-min-stall-s` above the longest such wait, and `--no-data-timeout-s` with it if warps can block without spinning.

### Find and shrink a data race

A data race depends on timing, so a buggy kernel can pass by luck. `random_delay` injects random per-thread delays before synchronization instructions, which makes such races fail visibly. `cutracer stress` automates the search: it runs an oracle command (exit 0 when the bug reproduced, non-zero otherwise) under several delay strategies, after control runs with every delay turned off, and saves the first delay config that reproduces the bug.

```bash
cutracer stress -k my_kernel --confirm -o /tmp/stress -- ./oracle.sh
```

`/tmp/stress/stress_report.json` gives the verdict: `finding` when the injected delay made the difference, `suspect` when the failure cannot be credited to the delay (for example, it also happens without one), `clean` when every planned attempt ran and none reproduced, and `incomplete` otherwise. `--confirm` checks a finding by shrinking it to the delay points that matter and replaying the result, and `cutracer stress` prints the `cutracer reduce` command for the saved config. `cutracer oracle pytest` turns one pytest node into an oracle without a wrapper script.

To record, replay and shrink a delay config by hand:

```bash
cutracer trace -a random_delay --delay-ns 10000 --delay-dump-path cfg.json -- ./app   # record which delay points were on
cutracer trace -a random_delay --delay-ns 10000 --delay-load-path cfg.json -- ./app   # replay the same points
cutracer reduce -c cfg.json -t ./oracle.sh --strategy bisect --confidence-runs 3      # keep only the points that trigger the bug
```

[Analyses](docs/Analyses.md) describes the delay modes, the stress strategies and the config format.

### Run Compute Sanitizer

```bash
cutracer compute-sanitizer --tool memcheck -o /tmp/sani -- python my_test.py
```

`--tool` is `racecheck` (the default), `memcheck`, `initcheck` or `synccheck`, and the log goes to `/tmp/sani/memcheck.log`. Compute Sanitizer and NVBit cannot share a process, so this is a separate run from `cutracer trace`. Pass the target's environment with `--env KEY=VAL`, because the command runs without a shell.

### Diagnose a failing test

```bash
cutracer diagnose --toolchain-version <toolchain-id> --arch sm_100a --source-revision <commit> --out report.json -- python my_test.py
```

`cutracer diagnose` finds the kernels the command launches (up to `--max-kernels`, default 32) and gives each its own session: all four Compute Sanitizer tools by default, random-delay stress when the target has a correctness oracle, and follow-up traces, replays and reductions. When the `claude` CLI is on `PATH`, an AI turn plans each follow-up; without it, the session falls back to deterministic decisions. Suppressions and per-scope baselines (`--baseline-dir`) limit AI turns to new findings and stress reproductions. A target can also be a work-unit JSON file or an approved `--preset`. See [Automated Diagnosis Service](docs/Automated-Diagnosis-Service.md).

## Commands

| Command | What it does |
|---|---|
| `trace` | Run a command with CUTracer injected: record traces, inject delays, detect hangs or log kernel launches |
| `query` | Filter, group and format trace records and kernel events |
| `validate` | Check the syntax and schema of a trace file |
| `compare` | Check that a text trace and an NDJSON trace of the same run agree |
| `sass` | Disassemble a dumped `.cubin` with `nvdisasm` |
| `analyze warp-summary` | List the completed, in-progress and missing warps of a trace |
| `stress` | Search for a delay config that triggers a bug |
| `oracle` | Run a pytest node as a `stress` or `reduce` oracle (`oracle pytest`), or write that oracle out as a script (`oracle export`) |
| `reduce` | Shrink a delay config to the delay points that trigger a bug |
| `compute-sanitizer` | Run NVIDIA Compute Sanitizer on a command |
| `diagnose` | Run a bounded diagnosis session on a command, a work unit or a preset |
| `diagnose-submit`, `diagnose-run`, `diagnose-status`, `diagnose-resume`, `diagnose-run-*` | Submit, run and inspect distributed diagnosis runs |

`cutracer <command> --help` shows a command's flags and examples, and `cutracer analyze --help` lists the analyses in your build.

## Configuration

Most `cutracer trace` flags set an environment variable that `cutracer.so` reads. Some features exist only as variables; `cutracer trace` passes them through to the command. The ones you are most likely to need:

| Variable | Flag | Purpose |
|---|---|---|
| `CUTRACER_INSTRUMENT` | `-i` | Instrumentation modes |
| `CUTRACER_ANALYSIS` | `-a` | `deadlock_detection`, `random_delay` or `proton_instr_histogram` |
| `KERNEL_FILTERS` | `-k` | Kernel name substrings to instrument |
| `CUTRACER_OUTPUT_DIR` | `-o` | Output directory, created if missing |
| `CUTRACER_TRACE_FORMAT` | `--trace-format` | `ndjson` (default), `zstd`, `text` or `clp` |
| `CUTRACER_NO_DATA_TIMEOUT_S`, `CUTRACER_KERNEL_TIMEOUT_S` | `--no-data-timeout-s`, `--kernel-timeout-s` | Hang watchdogs |
| `CUTRACER_HANG_CHECK_INTERVAL_MS`, `CUTRACER_HANG_CHECK_HITS`, `CUTRACER_HANG_MIN_STALL_S` | `--hang-check-interval-ms`, `--hang-check-hits`, `--hang-min-stall-s` | When `-a deadlock_detection` ends a run as hung |
| `CUTRACER_NONBLOCKING_LAUNCHES` | `--nonblocking-launches` | Let kernels overlap as they do natively (PDL, kernels that wait for a later launch); not with `-a deadlock_detection`, `-a proton_instr_histogram` or a plan |
| `CUTRACER_DELAY_*`, `CUTRACER_CLUSTER_CTA_ID` | `--delay-*` | Delay injection, dump and replay |
| `INSTR_BEGIN`, `INSTR_END` | | Instrument only a range of static instruction indices |
| `CUTRACER_EXACT_KERNEL_NAME`, `CUTRACER_EXACT_KERNEL_CHECKSUM` | | Instrument exactly one kernel, by full name and checksum |
| `CUTRACER_TRACE_LAUNCH_ID`, `CUTRACER_TRACE_WARP_MASK`, `CUTRACER_TRACE_PC_OFFSETS` | | Record only one launch, some warps or some PCs |
| `CUTRACER_INSTRUMENTATION_PLAN` | | Capture registers at the points a plan file names |
| `CUTRACER_RUNTIME_IDENTITY` | | `1` writes the runtime identity manifest |

[Configuration](docs/Configuration.md) lists every variable and its default. To run without the `cutracer` command, set `CUDA_INJECTION64_PATH=/path/to/CUTracer/lib/cutracer.so` and the variables yourself. `cutracer trace` sets `CUDA_INJECTION64_PATH` itself, and without `--cutracer-so` it refuses to start when the variable is already set.

## Agent skills

Reusable CUTracer workflows for Codex, Claude, Muse, and other coding agents live under [`agentic/skills/`](agentic/README.md). The directory is the canonical source. Agents install or register matching skills with their own supported loader before use.

| Skill | Use it when |
|---|---|
| [`investigate-kernel`](agentic/skills/investigate-kernel/SKILL.md) | A GPU kernel misbehaves and no workflow has been chosen yet; it routes to the matching skill |
| [`debug-hanging-kernel`](agentic/skills/debug-hanging-kernel/SKILL.md) | A kernel hangs or never returns |
| [`debug-illegal-access`](agentic/skills/debug-illegal-access/SKILL.md) | A kernel faults with CUDA error 700 or an illegal or misaligned address |
| [`audit-memory-model`](agentic/skills/audit-memory-model/SKILL.md) | A kernel (NVIDIA Hopper or Blackwell, AMD CDNA3 or CDNA4) must be checked against the vendor memory model without a failing run |
| [`report-case-study`](agentic/skills/report-case-study/SKILL.md) | An investigation is complete and needs an auditable, reusable report |

## Examples

[`examples/`](examples/) holds reproducers and reference outputs of real investigations:

| Example | What it shows |
|---|---|
| [Proton trace](examples/proton_trace/) | The instruction histogram workflow for a Triton vector-add kernel, with a sample histogram CSV and CUTracer log |
| [Conv backward-weight IMA](examples/conv_bwd_weight_ima/) | An illegal memory access in a TorchInductor convolution kernel, localized to one SASS instruction, with the bad address traced back to its source registers and the bug pinned on `ptxas` |
| [TLX named-barrier deadlock](examples/tlx_named_barrier_deadlock/) | A kernel hang in a TLX compiler whose named-barrier lowering was later fixed |
| [TLX TMEM WAR](examples/tlx_tmem_war/) | Matched failing and fixed reproducers for an aliased-TMEM race in the 2-CTA attention-backward kernel |
| [TLX CLC GEMM](examples/tlx_gemm_clc/) | A missing async-proxy fence after the CLC response read in the TLX Blackwell GEMM, with three blind agent audits using `audit-memory-model` |
| [CK FMHA K/V LDS WAR](examples/ck_fmha_kv_lds_war/) | A missing workgroup barrier before the V tile reuses the K buffer in LDS, in the Composable Kernel FMHA forward example on MI350, with a delay patch that exposes it and three blind agent audits |

Each example's README lists its setup. The TLX examples need a Triton with TLX, such as Meta's `fbtriton` package on PyPI, installed in place of the `triton` package that PyTorch brings.

## Troubleshooting

- No trace files: `-i` was not given (CUTracer then only logs launches), or `-k` matched no kernel. Run once without `-i` to see the names of the kernels that launch.
- The run ends with exit status 241 or 247: CUTracer ended it as hung, through `deadlock_detection` or a timeout. A slow but healthy kernel can need a larger `--no-data-timeout-s` or `--kernel-timeout-s`; `0` turns either off.
- `cutracer validate` rejects a new trace: check that the `cutracer` command and `cutracer.so` come from the same version.
- High overhead: `opcode_only` is the lightest mode; narrow the capture with `-k`, `--instr-categories` or `INSTR_BEGIN`/`INSTR_END`.

[Troubleshooting](docs/Troubleshooting.md) and the [FAQ](docs/FAQ.md) cover more cases.

## Documentation

The full documentation lives in [`docs/`](docs/) and is synced to the [GitHub Wiki](https://github.com/facebookexperimental/CUTracer/wiki) on every push to `main` by [`.github/workflows/sync-wiki.yml`](.github/workflows/sync-wiki.yml). Edit `docs/*.md`, not the wiki: the next sync overwrites direct wiki edits.

- Getting started: [Quickstart](docs/Quickstart.md), [Concepts](docs/Concepts.md), [Examples](docs/Examples.md)
- Reference: [Configuration](docs/Configuration.md), [Instrumentation Modes](docs/Instrumentation-Modes.md), [Instrumentation Plans](docs/Instrumentation-Plans.md), [Outputs and File Formats](docs/Outputs-and-File-Formats.md), [API and Data Structures](docs/API-and-Data-Structures.md)
- Workflows: [Analyses](docs/Analyses.md), [Automated Diagnosis Service](docs/Automated-Diagnosis-Service.md), [Post-processing: IPC Merge](docs/Post-processing-IPC-Merge.md), [Triton/Proton Integration](docs/Triton_Proton-Integration.md)
- Development: [Architecture](docs/Architecture.md), [Developer Guide](docs/Developer-Guide.md), [Build, Test, and CI](docs/Build-Test-and-CI.md), [Code Formatting](docs/Code-Formatting.md), [Python package](python/README.md)
- Help: [Troubleshooting](docs/Troubleshooting.md), [FAQ](docs/FAQ.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This repository contains code under the MIT license (Meta) and the BSD-3-Clause license (NVIDIA). See [LICENSE](LICENSE) and [LICENSE-BSD](LICENSE-BSD) for details.
