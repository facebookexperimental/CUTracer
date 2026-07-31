## Main Tool Log 🗒️

- Filename: `cutracer_main_YYYYMMDD_HHMMSS.log`
- Contents: Initialization, kernel launch lines (ctx, grid/block sizes, kernel checksum), analysis messages.

## Per-kernel Trace Log 🧾

- Filename: `kernel_<checksum>_iter<idx>_<mangled_name>.{log,ndjson,ndjson.zst,clp}`
- Purpose: Detailed per-kernel trace output (register, memory, TMA, or opcode records) when relevant instrumentation modes are active.
- The file extension depends on the trace format selected via `CUTRACER_TRACE_FORMAT` (see below).

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
| `"tma_trace"` | `tma_trace` | `desc_addr`, `desc_raw[16]` |

All records share common fields: `ctx`, `grid_launch_id`, `cta`, `warp`, `opcode_id`, `pc`, `sass`, `trace_index`, `timestamp`.

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
| `"kernel_launch"` | Every kernel launch | `kernel_launch_id`, `kernel_name`, `kernel_checksum`, `grid` (`[x,y,z]`), `block` (`[x,y,z]`), `nregs`, `shmem`, `stream_id` |
| `"callstack_def"` | First time a unique CPU callstack is seen (mode=`dedup`) | `callstack_id` (hex hash), `frames` (array of strings), `source` |

Mode-specific fields on `kernel_launch`:

- `dedup` — adds `callstack_id` (or `null`); `callstack_def` records are
  emitted exactly once per distinct callstack.
- `full` — embeds the entire callstack inline as `cpu_callstack` (array of
  strings) plus `cpu_callstack_source`.
- `nostack` — no callstack fields are added.

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
