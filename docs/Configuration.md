CUTracer is configured entirely through environment variables. This page is the complete reference.

## Core Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUDA_INJECTION64_PATH` | path | — | Path to `lib/cutracer.so`. The CUDA driver uses this to load CUTracer |
| `CUTRACER_INSTRUMENT` | string (comma-separated) | `""` (none) | Instrumentation types to enable. See values below |
| `CUTRACER_ANALYSIS` | string (comma-separated) | `""` (none) | Analysis types to enable. See values below |
| `KERNEL_FILTERS` | string (comma-separated) | `""` (all kernels) | Kernel name substring filters (matched against both mangled and unmangled names). Example: `KERNEL_FILTERS=add,_Z2_gemm,reduce` |
| `INSTR_BEGIN` | uint32 | `0` | Start of the static instruction index interval for instrumentation |
| `INSTR_END` | uint32 | `UINT32_MAX` | End of the static instruction index interval for instrumentation |
| `TOOL_VERBOSE` | int | `0` | Tool log verbosity level (0 = quiet, 1 = verbose, 2 = more verbose) |

### `CUTRACER_INSTRUMENT` Values

| Value | Overhead | Description |
|-------|----------|-------------|
| `opcode_only` | Lightest 🪶 | Collects opcode ID, warp ID, PC, CTA IDs, kernel_launch_id. Used for instruction histograms |
| `reg_trace` | Medium 🔬 | Collects per-thread register values including uniform registers |
| `mem_addr_trace` | Heavy 🧠 | Collects 32-lane memory access addresses |
| `mem_value_trace` | Heaviest 💾 | Collects memory addresses + values (up to 128-bit per lane), plus memory space, load/store direction, and access width |
| `tma_trace` | Medium 📦 | TMA transfer tracing (NVBit-parsed TMA parameter handle) for TMA instructions on Hopper/Blackwell GPUs |
| `random_delay` | Low 🔀 | Injects delays before synchronization instructions for data race detection |

> ⚠️ Enabling both `mem_addr_trace` and `mem_value_trace` triggers a warning. `mem_value_trace` already includes address information, so there is usually no need to enable both.

### `CUTRACER_ANALYSIS` Values

| Value | Auto-enables Instrumentation | Description |
|-------|------------------------------|-------------|
| `proton_instr_histogram` | `opcode_only` | Per-warp instruction histogram, segmented by clock regions |
| `deadlock_detection` | `reg_trace` | Kernel hang detection based on warp loop signatures |
| `random_delay` | `random_delay` instrumentation | Random delay injection on synchronization instructions. Requires `CUTRACER_DELAY_NS` |

When an analysis auto-enables an instrumentation type, you do **not** need to repeat it in `CUTRACER_INSTRUMENT`.

---

## Output Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_OUTPUT_DIR` | path | `""` (current working directory) | Output directory for trace files. Created, with its parents, if it does not exist; must be a writable directory |
| `CUTRACER_TRACE_FORMAT` | string or int | `ndjson` (`2`) | Trace output format. Accepts string names (case-insensitive) or numeric values. See values below |
| `CUTRACER_ZSTD_LEVEL` | int | `9` | Zstd compression level (1–22). Only takes effect when format is `zstd` (`1`) |

> **Deprecated:** `TRACE_FORMAT_NDJSON` is still accepted as a fallback if `CUTRACER_TRACE_FORMAT` is not set, but it is deprecated and will print a deprecation notice. Use `CUTRACER_TRACE_FORMAT` instead.

### `CUTRACER_TRACE_FORMAT` Values

| String | Numeric | Format | Description |
|--------|---------|--------|-------------|
| `text` | `0` | Text | Plain-text format, human-readable |
| `zstd` | `1` | NDJSON + Zstd | Compressed NDJSON (`.ndjson.zst`) |
| `ndjson` | `2` | NDJSON | **Default**. Uncompressed NDJSON, useful for debugging |
| `clp` | `3` | CLP Archive | CLP archive format |

Unrecognized string names or out-of-range numeric values cause a fatal error.

### `CUTRACER_ZSTD_LEVEL` Range

- **1–3**: Faster compression, slightly larger output
- **19–22**: Maximum compression, slower but smallest output
- **Default: 9** (balanced compression speed and ratio)
- Invalid values (< 1 or > 22) fall back to `9` with a warning

---

## Instruction Category Filtering

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_INSTR_CATEGORIES` | string (comma-separated) | `""` (no filter — all instructions instrumented) | Only instrument instructions belonging to the specified categories |

### Supported Categories

| Category | Matched SASS Instructions | Description |
|----------|---------------------------|-------------|
| `mma` | HGMMA, UTCHMMA, UTCIMMA, UTCQMMA, UTCOMMA | Matrix Multiply-Accumulate |
| `tma` | UTMALDG, UTMASTG, UTMAREDG | Tensor Memory Access |
| `sync` | WARPGROUP.DEPBAR, SYNCS.\*, BAR.SYNC, BAR.ARV, DEPBAR.\* | Synchronization |

- Case-insensitive
- Empty or unset means all instructions are instrumented (no filtering)
- Setting the variable with no valid category names prints a WARNING

**Example** — trace only MMA and TMA instructions:

```bash
CUTRACER_INSTR_CATEGORIES=mma,tma \
CUTRACER_INSTRUMENT=reg_trace \
CUDA_INJECTION64_PATH=./lib/cutracer.so \
python3 your_kernel.py
```

---

## Targeted Capture

These variables narrow a trace capture to one kernel, one launch, some warps or some instructions. `cutracer trace` has no flags for them but passes them through from the environment, and the diagnosis service sets them for its follow-up traces. Every `kernel_metadata` header records the trace selectors in effect.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_EXACT_KERNEL_NAME` | string | unset | Instrument only the launch root whose mangled or unmangled name is exactly this and whose checksum is `CUTRACER_EXACT_KERNEL_CHECKSUM`, plus its device callees. Set both or neither: one without the other is a fatal error. `KERNEL_FILTERS` is ignored while they are set |
| `CUTRACER_EXACT_KERNEL_CHECKSUM` | string | unset | Kernel checksum of that launch root, as in trace file names (`kernel_<checksum>_...`) |
| `CUTRACER_TRACE_LAUNCH_ID` | uint64 | unset (all launches) | Write `reg_trace`, `mem_addr_trace` and `mem_value_trace` records only for the launch with this `kernel_launch_id` |
| `CUTRACER_TRACE_WARP_MASK` | uint32 (hex or decimal) | `0` (all warps) | Write those records only for the CTA-local warps whose bits are set, e.g. `0xF` for warps 0–3 |
| `CUTRACER_TRACE_PC_OFFSETS` | string (comma-separated, hex or decimal) | `""` (all instructions) | Instrument only the instructions at these function-relative SASS PC offsets, at most 32. An invalid token or more than 32 offsets disables trace records |
| `CUTRACER_TRACE_INSTR_CATEGORIES` | string (comma-separated) | `""` (all instructions) | Instrument only instructions of these categories (`mma`, `tma`, `sync`), on top of `CUTRACER_INSTR_CATEGORIES` |

Random-delay targeting has its own variables (`CUTRACER_DELAY_WARPGROUP_ID`, `CUTRACER_DELAY_WARP_MASK`); these selectors do not change where delays are injected.

---

## Delay Injection Configuration (random_delay analysis)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_DELAY_NS` | uint64 | `0` | Delay value in nanoseconds. Must be positive (≤ UINT32_MAX) when `random_delay` analysis is enabled |
| `CUTRACER_DELAY_MIN_NS` | uint64 | `0` | Lower bound for sampled delays in `random` mode. Must be `≤ CUTRACER_DELAY_NS` |
| `CUTRACER_DELAY_MODE` | string | `random` | One of `random`, `fixed`, `cluster`, `cluster_fixed`. Selects the delay sampling strategy used by the `random_delay` analysis |
| `CUTRACER_CLUSTER_CTA_ID` | int | `-1` (disabled) | Restrict delay injection to a specific CTA ID (used with `cluster*` modes). Setting this together with `CUTRACER_DELAY_LOAD_PATH` produces non-bit-identical replays |
| `CUTRACER_DELAY_WARP_MASK` | uint32 | `0` (disabled) | Delay only the CTA-local warps whose bits are set (hex or decimal), with a fixed delay. With a `cluster*` mode, only those warps of the selected CTA of each cluster are delayed |
| `CUTRACER_DELAY_WARPGROUP_ID` | int | `-1` (disabled) | Delay only warpgroup N (warps `[4N..4N+3]`); wins over `CUTRACER_DELAY_WARP_MASK`. Combines with `cluster*` modes the same way |
| `CUTRACER_DELAY_ENABLE_PROB` | float | `0.5` | Probability that each instrumentation point is enabled while recording (0.0–1.0; a value outside the range is FATAL). Use `1.0` with warp targeting. No effect in replay mode |
| `CUTRACER_DELAY_PATTERNS` | string (comma-separated) | `""` (built-in patterns) | SASS instruction substrings to inject delays before, replacing the built-in patterns. `*` matches every instruction |
| `CUTRACER_DELAY_DUMP_PATH` | path | `""` (disabled) | Output path for delay config JSON (dump mode — records random delay pattern) |
| `CUTRACER_DELAY_LOAD_PATH` | path | `""` (disabled) | Input path for delay config JSON (replay mode — deterministically reproduces a recorded pattern) |
| `CUTRACER_DELAY_SEED` | uint64 | random | Seed of the per-point on/off choices. Processes that share it make the same choices and merge their writes to `CUTRACER_DELAY_DUMP_PATH`; `cutracer trace --delay-dump-path` and `cutracer stress` set one per run |

> ⚠️ `CUTRACER_DELAY_DUMP_PATH` and `CUTRACER_DELAY_LOAD_PATH` are **mutually exclusive**. Setting both causes a FATAL exit.

See [Analyses](Analyses.md#data-race-detection-random_delay-) for full usage details including dump/replay workflow.

---

## Debug Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_DUMP_CUBIN` | int | auto | Set to `1` to dump cubin files for instrumented kernels, `0` to disable. When not set, **auto-enabled** whenever any instrumentation is active. The log indicates `enabled (auto)` vs `enabled` |

---

## Advanced Configuration

These knobs are not needed for typical workflows. They control instrumentation
points, host-side callstack capture, kernel-level metadata events, channel
buffer sizing, and runtime safety limits.

### Instrumentation Point (IPOINT) Selection

NVBit lets each instrumentation be inserted *before* (`a`, the default) or
*after* (`b`) the target instruction. These two variables control that
choice. Setting both is a FATAL error.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_INSTRUMENT_IPOINT_UNIFORM` | string (`a`/`b`) | `a` | Apply a single IPOINT to **all** enabled instruments |
| `CUTRACER_INSTRUMENT_IPOINT` | string (comma-separated, `a`/`b`) | — | Per-instrument IPOINT list. Length **must equal** the number of values in `CUTRACER_INSTRUMENT`, in the same order. Example: `CUTRACER_INSTRUMENT=opcode_only,reg_trace` + `CUTRACER_INSTRUMENT_IPOINT=a,b` |

### CPU-Side Callstack Capture

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_CPU_CALLSTACK` | string | `auto` | Host callstack capture strategy. Values: `auto`, `auto_gil`, `pytorch`, `backtrace`, `0` (disable), `1` (enable, default backend) |

### Kernel Metadata Events (NDJSON)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_KERNEL_EVENTS` | string | `0` | Per-kernel metadata events writer. Values: `0` (disabled), `dedup`, `full`, `nostack`. Anything other than `0` requires `CUTRACER_CPU_CALLSTACK ≠ 0` |

When enabled, CUTracer writes a separate `kernel_events_*.ndjson` file. See
[Outputs and File Formats](Outputs-and-File-Formats.md) for the file layout.

### GPU Channel Buffer Sizing

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_CHANNEL_RECORDS` | int | `0` (use built-in default) | Number of records the GPU→CPU channel buffer should hold. The on-device buffer size is computed from this value. Increase for high-throughput kernels if you see drops |

### Runtime Identity Manifest

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_RUNTIME_IDENTITY` | string | unset | `1` writes a runtime identity manifest that binds each launch to its process, CUDA context, module, CUBIN bytes and function-relative instruction PCs |
| `CUTRACER_RUN_ID` | string | unset | Recorded in the manifest as an unauthenticated run hint |
| `CUTRACER_WORKER_ID` | string | unset | Recorded in the manifest as an unauthenticated worker hint |

The manifest also records `RANK` and `WORLD_SIZE` as unauthenticated hints.
An instrumentation plan enables the manifest automatically.
Instrumented launches get trace files named after their launch record, and the
trace header carries that record. See
[Outputs and File Formats](Outputs-and-File-Formats.md#runtime-identity-manifest-).

### Instrumentation Plans

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_INSTRUMENTATION_PLAN` | path | unset | Capture register values at the exact instructions, launches, CTAs, warps and lanes named by a plan file. Excludes the instruction selection variables above |

See [Instrumentation Plans](Instrumentation-Plans.md) for the file format and outputs.

### Runtime Safety Limits

These guardrails terminate the run early if a kernel is hung or a trace is
runaway-large. All accept `0` to disable.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_KERNEL_TIMEOUT_S` | uint32 | `0` (disabled) | Max wall-clock seconds a single kernel is allowed to run, counted from its first trace record, or from its launch when it sends none (`random_delay` alone, instrumentation plans) |
| `CUTRACER_NO_DATA_TIMEOUT_S` | uint32 | `15` | Abort if no records arrive on the channel for this many seconds. With `random_delay` alone no records ever arrive, so this is a per-kernel wall-clock limit counted from launch; CUTracer warns at startup when it is the only limit in that mode |
| `CUTRACER_TRACE_SIZE_LIMIT_MB` | uint32 | `0` (disabled) | Stop tracing once the on-disk trace reaches this size (MB) |

### Launch Handling

By default CUTracer waits for the device to go idle before every launch that
is not being captured into a CUDA graph, and waits for each launch to finish
and its trace to drain before returning to the application. Separate launches
therefore never overlap under CUTracer (nodes of one CUDA graph still can),
which hides cross-kernel races (for example a PDL consumer that reads before its
producer finishes) and can hang programs in which a running kernel waits for a
later launch.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_NONBLOCKING_LAUNCHES` | `0`/`1` | `0` | Return to the application right after each launch: the launch's trace drains on a background thread without the launching thread waiting for it, and CUTracer waits for the device to go idle before a launch only when the launch changes code an earlier launch may still be running (a switch between instrumented and original code, or instrumenting a kernel or callee that already ran). Not compatible with `deadlock_detection`, `proton_instr_histogram` or `CUTRACER_INSTRUMENTATION_PLAN`, which assume one kernel at a time; CUTracer refuses the combination |

With non-blocking launches:
- Each instrumented launch is followed in its stream by the event its drain waits on. So is every identified launch when `CUTRACER_RUNTIME_IDENTITY` is on. Other uninstrumented launches (the bare launch logger, or kernels outside `-k`) add nothing. On GB200, a kernel launched right after an instrumented one with programmatic dependent launch (PDL) still overlapped it, and the instrumented kernel's trace was complete; other GPUs and drivers are untested.
- PDL inside a CUDA graph still does not overlap. On GB200 a graph-replayed PDL pair that races natively was clean under CUTracer, with or without this setting and with the bare launch logger, so the cause is in how graph launches run under NVBit, not in CUTracer's waits.
- The wait before a launch that changes code covers direct launches only: launches captured into a graph switch code without it, as in the default mode. If a running kernel waits for the very launch that needs the wait, the program hangs as in the default mode.
- The timers start at a kernel's first trace record, not at its launch, so a kernel waiting in its stream is not timed out. They judge only while a kernel that has sent records is still open. The kernel timeout restarts whenever the trace switches between overlapping kernels and can fire late or not at all. A kernel that hangs before its first record, and a delay-only kernel (which sends none), are not timed out.
- A launching thread that gets far ahead waits: for the oldest queued drain once 1024 are queued, and for all of them once 256 traces are open. If a queued kernel waits for a launch that thread has not issued yet, the program hangs there.
- At context teardown CUTracer waits for every queued drain, that is, for every instrumented kernel still running.

### Deadlock Check Policy

With `CUTRACER_ANALYSIS=deadlock_detection`, the online check ends the run after
a number of consecutive checks in which no active warp makes progress. See
[Analyses](Analyses.md) for the rule.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUTRACER_HANG_CHECK_INTERVAL_MS` | uint32 | `1000` | Minimum time between two checks. Must be positive |
| `CUTRACER_HANG_CHECK_HITS` | uint32 | `3` | Consecutive checks without a progressing warp that end the run. Must be positive |
| `CUTRACER_HANG_MIN_STALL_S` | uint32 | `0` (hits alone) | The stall must also have lasted this long, counted from the first check without progress. Set it above the longest wait a correct kernel can spend spinning; keep it below `CUTRACER_NO_DATA_TIMEOUT_S` if warps may block without spinning |

---

## Implicitly Set Variables

These are set automatically by CUTracer at startup. Users should **not** set them manually:

| Variable | Value | Description |
|----------|-------|-------------|
| `CUDA_MANAGED_FORCE_DEVICE_ALLOC` | `1` | Simplifies channel memory handling |

CUDA/NVBit/driver versions must be compatible with your GPU.

---

## Quick Reference (alphabetical)

| Variable | Section |
|----------|---------|
| `CUDA_INJECTION64_PATH` | Core |
| `CUDA_MANAGED_FORCE_DEVICE_ALLOC` | Implicit |
| `CUTRACER_ANALYSIS` | Core |
| `CUTRACER_CHANNEL_RECORDS` | Advanced — Channel |
| `CUTRACER_CLUSTER_CTA_ID` | Delay Injection |
| `CUTRACER_CPU_CALLSTACK` | Advanced — Callstack |
| `CUTRACER_DELAY_DUMP_PATH` | Delay Injection |
| `CUTRACER_DELAY_ENABLE_PROB` | Delay Injection |
| `CUTRACER_DELAY_LOAD_PATH` | Delay Injection |
| `CUTRACER_DELAY_MIN_NS` | Delay Injection |
| `CUTRACER_DELAY_MODE` | Delay Injection |
| `CUTRACER_DELAY_NS` | Delay Injection |
| `CUTRACER_DELAY_PATTERNS` | Delay Injection |
| `CUTRACER_DELAY_SEED` | Delay Injection |
| `CUTRACER_DELAY_WARP_MASK` | Delay Injection |
| `CUTRACER_DELAY_WARPGROUP_ID` | Delay Injection |
| `CUTRACER_DUMP_CUBIN` | Debug |
| `CUTRACER_EXACT_KERNEL_CHECKSUM` | Targeted Capture |
| `CUTRACER_EXACT_KERNEL_NAME` | Targeted Capture |
| `CUTRACER_HANG_CHECK_HITS` | Advanced — Deadlock Check Policy |
| `CUTRACER_HANG_CHECK_INTERVAL_MS` | Advanced — Deadlock Check Policy |
| `CUTRACER_HANG_MIN_STALL_S` | Advanced — Deadlock Check Policy |
| `CUTRACER_INSTR_CATEGORIES` | Instruction Category Filtering |
| `CUTRACER_INSTRUMENTATION_PLAN` | Advanced — Instrumentation Plans |
| `CUTRACER_INSTRUMENT` | Core |
| `CUTRACER_INSTRUMENT_IPOINT` | Advanced — IPOINT |
| `CUTRACER_INSTRUMENT_IPOINT_UNIFORM` | Advanced — IPOINT |
| `CUTRACER_KERNEL_EVENTS` | Advanced — Kernel Events |
| `CUTRACER_KERNEL_TIMEOUT_S` | Advanced — Runtime Limits |
| `CUTRACER_NO_DATA_TIMEOUT_S` | Advanced — Runtime Limits |
| `CUTRACER_NONBLOCKING_LAUNCHES` | Advanced — Launch Handling |
| `CUTRACER_OUTPUT_DIR` | Output |
| `CUTRACER_RUN_ID` | Advanced — Runtime Identity |
| `CUTRACER_RUNTIME_IDENTITY` | Advanced — Runtime Identity |
| `CUTRACER_TRACE_FORMAT` | Output |
| `CUTRACER_TRACE_INSTR_CATEGORIES` | Targeted Capture |
| `CUTRACER_TRACE_LAUNCH_ID` | Targeted Capture |
| `CUTRACER_TRACE_PC_OFFSETS` | Targeted Capture |
| `CUTRACER_TRACE_SIZE_LIMIT_MB` | Advanced — Runtime Limits |
| `CUTRACER_TRACE_WARP_MASK` | Targeted Capture |
| `CUTRACER_WORKER_ID` | Advanced — Runtime Identity |
| `CUTRACER_ZSTD_LEVEL` | Output |
| `INSTR_BEGIN` | Core |
| `INSTR_END` | Core |
| `KERNEL_FILTERS` | Core |
| `TOOL_VERBOSE` | Core |

---

## Notes 📝

- When `proton_instr_histogram` is enabled, `opcode_only` is forced internally to minimize overhead and ensure required data is available.
- When `deadlock_detection` is enabled, `reg_trace` is forced internally because loop detection relies on PC and opcode correlation per warp.
- When `random_delay` analysis is enabled, the tool will FATAL exit if `CUTRACER_DELAY_NS` is not set or is zero.
- `KERNEL_FILTERS` uses substring matching against both unmangled and mangled names; any match enables instrumentation for that function and related device functions.
- Enabling additional instrumentation modes increases overhead and output volume; prefer the minimal set that satisfies your analysis.
