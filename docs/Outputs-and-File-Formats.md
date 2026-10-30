## Main Tool Log 🗒️

- Filename: `cutracer_main_YYYYMMDD_HHMMSS.log`
- Contents: Initialization, kernel launch lines (ctx, grid/block sizes, kernel checksum), analysis messages.

## Per-kernel Trace Log 🧾

- Filename: `kernel_<checksum>_iter<idx>_<mangled_name>.{log,ndjson,ndjson.zst,clp}`
- Purpose: Detailed per-kernel trace output (register, memory, TMA, or opcode records) when relevant instrumentation modes are active.
- The file extension depends on the trace format selected via `CUTRACER_TRACE_FORMAT` (see below).
- With the [runtime identity manifest](#runtime-identity-manifest-) enabled, a launch that has a launch record is written to `capture_<launch>.{log,ndjson,ndjson.zst,clp}` instead, where `<launch>` is that record's key.

## Histogram CSV 📄

- Filename: `kernel_<checksum>_iter<idx>_<mangled_name>_hist.csv`
- Columns:
  - `warp_id` (int): Global warp ID.
  - `region_id` (int): Region ordinal within the warp (clock-to-clock segment).
  - `instruction` (str): SASS mnemonic (with dot modifiers), extracted from static decoding.
  - `count` (int): Occurrence count in the region.

Example rows:
```
warp_id,region_id,instruction,count
0,0,LDG.E.SYS,64
0,0,IMAD.MOV.U32,32
```

## Trace Format 📁

CUTracer supports multiple trace output formats, controlled by `CUTRACER_TRACE_FORMAT` (accepts string names or numeric values). See [Configuration](Configuration.md#output-configuration) for full details.

| String | Numeric | Format | Extension | Description |
|--------|---------|--------|-----------|-------------|
| `text` | `0` | Text | `.log` | Plain-text, human-readable |
| `zstd` | `1` | NDJSON + Zstd | `.ndjson.zst` | Compressed Newline-Delimited JSON. Compression level controlled by `CUTRACER_ZSTD_LEVEL` (default 9) |
| `ndjson` | `2` | NDJSON | `.ndjson` | **Default**. Uncompressed NDJSON, useful for debugging format issues |
| `clp` | `3` | CLP Archive | `.clp` | First writes `.ndjson`, then converts to CLP archive via `clp-s` and removes the intermediate `.ndjson` file |

> **Deprecated:** The old `TRACE_FORMAT_NDJSON` env var is still accepted as a fallback but will print a deprecation notice.

### NDJSON Record Types

Each line in an NDJSON trace file is a self-contained JSON object. The `type` field identifies the record kind:

| `type` value | Instrumentation Mode | Key Fields |
|--------------|---------------------|------------|
| `"reg_trace"` | `reg_trace` | `regs[reg][thread]`, `regs_indices`, `uregs`, `uregs_indices` |
| `"mem_addr_trace"` | `mem_addr_trace` | `addrs[32]`, `ipoint: "B"`, optional DSM attribution fields |
| `"mem_value_trace"` | `mem_value_trace` | `addrs[32]`, `values[32][N]`, `mem_space`, `is_load`, `access_size`, `ipoint: "A"` |
| `"opcode_only"` | `opcode_only` | (common fields only) |
| `"tma_trace"` | `tma_trace` | `tma_param_size`, `tma_transfer_info`, `desc_addr` (older traces: `desc_raw[16]`) |

All records share common fields: `ctx`, `grid_launch_id`, `cta`, `warp`, `opcode_id`, `pc`, `sass`, `trace_index`, `timestamp`.

[Instrumentation plan](Instrumentation-Plans.md#output) captures contain
`plan_observation` records instead, between a `plan_applied` record after the
header and a `plan_capture_completion` record before the footer.

### `mem_addr_trace` DSM attribution

Current SM90+ captures add these backward-compatible fields:

| Field | Meaning |
| --- | --- |
| `static_memory_space` | NVBit's static memory-space value; diagnostic only for generic `ST.E` |
| `cluster_attribution` | `supported`, `unsupported_arch`, or `unavailable` |
| `cluster_shared_mask` | Active lanes whose effective address belongs to cluster shared memory |
| `issuer_cluster_rank` | Issuing CTA's linear rank in the dynamic cluster |
| `target_cluster_ranks[32]` | Target rank per lane; inactive/non-DSM lanes use `4294967295` |

The rank vector is serialized only when attribution is `supported`. It remains
fixed-width so element `N` always describes lane `N`; use `active_mask` and
`cluster_shared_mask` to select meaningful lanes. Older traces omit all of
these fields and remain valid.

Kernel metadata records cluster geometry once per launch as `cluster_dim`,
`cluster_size`, and `cluster_dim_source` (`launch_attribute`,
`required_function_attribute`, or `unknown`). Direct and CUDA Graph launches
share this contract.

When the runtime identity manifest is enabled, the kernel metadata header also
carries `runtime_identity`: the same launch record that the manifest writes.

### NDJSON Example

```json
{"type":"reg_trace","ctx":"0x5591abcd","grid_launch_id":1,"cta":[0,0,0],"warp":0,"opcode_id":42,"pc":"0x43d0","sass":"HGMMA.64x128x32 ...","trace_index":7,"timestamp":1234567890,"regs":[[1,2,3,...]],"regs_indices":[4,5],"uregs":[100],"uregs_indices":[7]}
```

## Kernel Events NDJSON 🚀

Enabled by `CUTRACER_KERNEL_EVENTS` (see [Configuration](Configuration.md#kernel-metadata-events-ndjson)).
This is a **separate** NDJSON file (not the per-kernel trace) that records
metadata about every kernel launch — useful for offline correlation, callstack
analysis, and dedup studies.

- Filename: `cutracer_kernel_events_YYYYMMDD_HHMMSS.ndjson`
  (placed under `CUTRACER_OUTPUT_DIR` if set, otherwise the working directory)
- Writer: `g_kernel_events_writer` in `src/cutracer.cu`
  (output produced by `src/trace_writer.cpp`)
- Format: NDJSON, one JSON object per line, always uncompressed

### Record types

| `type` value | When emitted | Key fields |
|--------------|--------------|------------|
| `"kernel_launch"` | Every kernel launch | `kernel_launch_id`, `kernel_name` (demangled), `mangled_name`, `kernel_checksum`, `grid` (`[x,y,z]`), `block` (`[x,y,z]`), `nregs`, `shmem`, `stream_id` |
| `"callstack_def"` | First time a unique CPU callstack is seen (mode=`dedup`) | `callstack_id` (hex hash), `frames` (array of strings), `source` |

`kernel_name` remains the human-readable demangled name. `mangled_name` is the
exact CUDA symbol used when another tool, such as Compute Sanitizer, requires a
vendor-facing kernel filter. Readers accept legacy events without this field,
but such events cannot safely drive a pinned Compute Sanitizer run.

Mode-specific fields on `kernel_launch`:

- `dedup` — adds `callstack_id` (or `null`); `callstack_def` records are
  emitted exactly once per distinct callstack.
- `full` — embeds the entire callstack inline as `cpu_callstack` (array of
  strings) plus `cpu_callstack_source`.
- `nostack` — no callstack fields are added.

## Runtime Identity Manifest 🪪

Enabled by `CUTRACER_RUNTIME_IDENTITY=1` (see [Configuration](Configuration.md#runtime-identity-manifest)).
The manifest records what actually ran, from CUDA and NVBit callbacks, so a
trace can be bound to the exact binary and launch that produced it.

- Filename: `cutracer_runtime_<pid>_<nonce>.ndjson`, created exclusively
  under `CUTRACER_OUTPUT_DIR` if set, otherwise the working directory
- Format: NDJSON with sorted keys. Every line is
  `{"schema": "cutracer.runtime/v1", "process": <key>, "sequence": <n>, "type": <type>, "data": {...}}`
- Keys (`process`, `context`, `module`, `function`, `launch`) are SHA-256
  digests of the identifying record, so equal handles reused by the driver
  get distinct keys

| `type` value | When emitted | Key fields in `data` |
|--------------|--------------|----------------------|
| `"process_start"` | Tool initialization | `pid`, `parent_pid`, `start_ticks`, `boot_id`, `nonce`, `tool_sha256` (SHA-256 of the loaded tool), unauthenticated `run_hint`, `worker_hint`, `rank_hint`, `world_size_hint` |
| `"context_begin"` / `"context_end"` | CUDA context creation and teardown | `context`, `handle`; `incomplete_launches` at the end |
| `"module_begin"` / `"module_end"` | Module load, or the first launch from a module loaded before the tool | `module`, `context`, `handle`, `load_observed`, `api` |
| `"function"` | First launch of a function | `function`, `function_handle`, `entry`, `cubin_sha256` of the dumped CUBIN bytes, `instructions` (`pc`, `bytes`, `opcode_index`, `sass` per instruction), `module_load_observed`, `device_uuid`, `sm_family`, CUDA binary and driver versions |
| `"launch"` | Every direct launch or graph node launch | `launch`, `launch_id`, `kernel_ordinal`, `grid`, `block`, `cluster_dim`, `cluster_dim_source`, `launch_kind`, `graph_invocation`, `capture_selected` |
| `"driver_completion"` | The launch's CUDA completion was observed | `launch`, `launch_id`, `context`, `kernel_completed`, `channel_drained`, `trace_footer_required` |
| `"launch_incomplete"` | A context ended before a launch completed | `launch`, `reason` |
| `"gap"` | Identity could not be established | `reason` |
| `"plan_loaded"` | An [instrumentation plan](Instrumentation-Plans.md) was loaded | `plan_id`, `canonical`, `canonical_sha256`, `file_sha256` |
| `"plan_applied"`, `"plan_capture_completion"` | A plan launch was bound or finished | Same as the trace records of the same name |
| `"plan_gap"`, `"plan_error"` | A context ended with unfinished plan captures, or plan capture was disabled | `reason` |
| `"process_end"` | Tool termination | `pending_launches`, `live_contexts`, `has_gaps` |

The manifest contains host and device identifiers (process IDs, boot ID and
GPU UUID). Driver completion is recorded separately from trace completion: a
trace is complete only when its own `capture_completion` footer says so.

- A launch made with a CUDA runtime `CUkernel` is recorded under the
  context's `CUfunction`, the handle NVBit reports for graph nodes.
- A graph node takes its cluster geometry from graph construction. When one
  function was captured with different cluster shapes, its graph launches
  report `cluster_dim_source: "unknown"`.
- After a module unload, the driver can hand the same `CUfunction` value to a
  function of a later module. NVBit keeps per-handle state across that reuse,
  so such a launch runs uninstrumented, gets no trace and no launch record,
  and the manifest records a `gap`.
- Every launched function is inspected once per module lifetime, selected or
  not: its instructions are walked and its CUBIN is dumped.

## Delay Config JSON 🔀

When using the `random_delay` analysis with `CUTRACER_DELAY_DUMP_PATH`, CUTracer writes a delay configuration JSON file for deterministic replay. This file is consumed by `CUTRACER_DELAY_LOAD_PATH`.

- Filename: user-specified via `CUTRACER_DELAY_DUMP_PATH`
- See [Analyses](Analyses.md#data-race-detection-random_delay) for the full schema and dump/replay workflow.

Top-level structure:
```json
{
  "version": "1.0",
  "delay_ns": 10000,
  "kernels": {
    "<kernel_name>_<checksum>": {
      "kernel_name": "matmul_kernel",
      "kernel_checksum": "7fa21c3",
      "timestamp": "2026-02-03T21:15:21.567",
      "instrumentation_points": {
        "10192": { "pc": 10192, "sass": "SYNCS.PHASECHK...", "delay": 10000, "on": true },
        "10864": { "pc": 10864, "sass": "WARPGROUP.DEPBAR...", "delay": 10000, "on": false }
      }
    }
  }
}
```

## Output Directory 📂

By default, all output files are written to the traced process's current working directory.

Set `CUTRACER_OUTPUT_DIR` to redirect trace files (per-kernel logs, histograms) to a custom directory. The directory must already exist and be writable. See [Configuration](Configuration.md#output-configuration).

## Notes 📝
- When `CUTRACER_OUTPUT_DIR` is set, per-kernel trace files and histogram CSVs are prefixed with that directory path. The main tool log (`cutracer_main_*.log`) is always written to the current working directory.
- `<checksum>` is an FNV-1a hash of the kernel name + all SASS instructions (hex string), providing robust kernel identification across recompilations.
- `<idx>` is a per-kernel iteration counter that distinguishes repeated launches of the same kernel.
- `<mangled_name>` is truncated to 150 characters in filenames for readability.
- Use the Python CLI tools (`cutracer validate`, `cutracer query`) to read and process NDJSON and Zstd-compressed trace files.
