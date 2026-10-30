# Instrumentation Plans

An instrumentation plan captures register values at exact instructions of a
known CUBIN, for chosen launches, CTAs, warps and lanes, within explicit
budgets. Each point names its function-relative byte PC and the instruction
bytes it expects, so a plan only applies to the binary it was written for.

## Running a plan

Set `CUTRACER_INSTRUMENTATION_PLAN` to the plan file and trace as usual,
without any other instruction selection:

```bash
CUTRACER_INSTRUMENTATION_PLAN=/path/to/plan.json \
  cutracer trace --trace-format ndjson --output-dir ./capture -- python app.py
```

- The plan enables the [runtime identity manifest](Outputs-and-File-Formats.md#runtime-identity-manifest-),
  which binds every capture to the launch and CUBIN that produced it.
- It cannot be combined with `CUTRACER_INSTRUMENT`, `CUTRACER_ANALYSIS`,
  `KERNEL_FILTERS`, `INSTR_BEGIN`, `INSTR_END`, `CUTRACER_INSTR_CATEGORIES`,
  `CUTRACER_INSTRUMENT_IPOINT` or `CUTRACER_INSTRUMENT_IPOINT_UNIFORM`.
- The trace format must be `zstd` (`1`) or `ndjson` (`2`).
- A plan error disables plan capture for the rest of the process. The workload
  keeps running, affected traces are marked incomplete, and the manifest
  records a `plan_error` event.
- Only selected warps at planned points send records, so a capture can stay
  silent for a long time. `CUTRACER_KERNEL_TIMEOUT_S` still applies; raise or
  disable `CUTRACER_NO_DATA_TIMEOUT_S` for long kernels.

## Plan file

A plan is one JSON document with schema `cutracer.instrumentation_plan/v1`,
or an NDJSON file with one `cutracer.kernel_plan/v1` record (a single
`kernel` instead of `kernels`) per line, all with the same `plan_id` and
`policy`. Files are limited to 16 MiB. Duplicate keys, unknown fields, and
integers that are negative or above 2^53 - 1 are rejected.

```json
{
  "schema": "cutracer.instrumentation_plan/v1",
  "plan_id": "r5-after-load",
  "policy": {
    "unmatched_kernel": "skip",
    "unsupported_request": "error",
    "identity_mismatch": "skip",
    "conflicts": "error"
  },
  "kernels": [
    {
      "match": {
        "entry": "identity_kernel",
        "target": "sm_100a",
        "identity": {"kind": "cubin_sha256", "value": "<SHA-256 of the dumped CUBIN>"}
      },
      "pc_namespace": "function_relative_bytes",
      "configs": [
        {
          "config_id": "subset",
          "requests": ["r5-after-load"],
          "launches": {"mode": "all"},
          "capture": {"ctas": "all", "warps": [1], "lanes": [0, 3]},
          "budget": {"max_records": 64, "max_bytes": 65536, "on_limit": "mark_incomplete"},
          "points": [
            {
              "function": "identity_kernel",
              "pc": "0x10",
              "expected_bytes": "197905000000000000210000002e0e00",
              "phase": "after",
              "predicate_policy": "executed_lanes",
              "observations": [{"kind": "reg_trace", "registers": ["R5"]}]
            }
          ]
        }
      ]
    }
  ]
}
```

`R5` holds `threadIdx.x` after the `S2R` at `0x10`.
[`plan_capture_sample.ndjson`](../python/tests/example_inputs/plan_capture_sample.ndjson)
is one launch of this kernel captured with this plan on a B200.

| Field | Meaning |
|-------|---------|
| `policy.unsupported_request` | `error` rejects the plan when a point cannot be captured; `report_gap` marks only that configuration as not selected and records the reason |
| `policy.identity_mismatch` | A launch of a planned entry from a binary the plan does not name: `error` disables the plan; `skip` leaves that binary's launches uninstrumented (their manifest launch records show `capture_selected: false`), as when autotuning compiles one entry name several times. `unmatched_kernel` must be `skip` and `conflicts` must be `error` |
| `match.entry` | Kernel entry name as NVBit reports it (mangled). Kernels not named in the plan are not instrumented |
| `match.target`, `match.identity` | Target (`sm_<N>`, optionally with an `a` or `f` suffix) and SHA-256 of the dumped CUBIN |
| `configs[].requests` | Names of the consumers that asked for this configuration; carried through to the outputs |
| `launches` | `{"mode": "all"}`, or `{"mode": "per_context_kernel_ordinal", "indices": [...]}` where the ordinal counts earlier launches of the same CUBIN and entry in the CUDA context |
| `capture` | `ctas` (`[x, y, z]` coordinates), `warps` (CTA-local indices) and `lanes` (0 to 31): each `"all"` or a nonempty list |
| `budget` | Records and bytes the configuration may write for one launch. `max_bytes` bounds both the device records, 2120 bytes each whatever their registers, and the JSON lines written; entry marker records count too. Reaching any limit marks the capture incomplete |
| `points[].function` | Function that contains the instruction: the entry or a function it calls |
| `points[].pc` | Function-relative byte offset in lowercase hex |
| `points[].expected_bytes` | Instruction bytes in lowercase hex. A loaded binary with different bytes at that PC is an error |
| `points[].phase` | Capture `before` or `after` the instruction |
| `points[].predicate_policy` | `executed_lanes` captures lanes whose guard predicate is true; `reached_lanes` captures every lane that reached the instruction |
| `points[].observations` | `reg_trace` with `R<n>`, `UR<n>`, `P<n>` or `UP<n>` registers, or `opcode_only`. The format also accepts `mem_addr_trace` and `mem_value_trace` (with `mref`), `tma_trace` and `random_delay` (with `delay_ns`), which the runtime does not apply yet |

Loading normalizes a plan: equivalent files produce the same canonical JSON
and `canonical_sha256`, and points with the same function, PC and phase are
merged.

To find PCs and bytes, run the workload once with `CUTRACER_RUNTIME_IDENTITY=1`.
Each `function` record of the manifest lists the CUBIN SHA-256 and every
instruction's `pc`, `bytes` and `sass`.

## What the runtime applies

- SM100a CUBINs only.
- `reg_trace` and `opcode_only` observations.
- Up to 16 distinct registers per point: `R0`–`R254` within the function's
  register allocation, `UR0`–`UR62`, `P0`–`P6` and `UP0`–`UP6`.
- `after` points on predicated instructions or on control transfers (`BRA`,
  `BRX`, `JMP`, `JMX`, `CALL`, `RET`, `EXIT`, `KILL`, `RTT`, `BPT` and their
  uniform forms), and `after` points with `reached_lanes`, are not supported.
- Up to 256 configurations and 8192 instrumented points per process
  (including one entry point per kernel, added to count the warps that enter
  it), up to 1024 unfinished captured launches per context, and blocks of at
  most 1024 threads.
- Direct launches and CUDA Graph node launches. `cuLaunch`, `cuLaunchGrid`,
  `cuLaunchGridAsync` and `cuGraphLaunchWithArguments` disable plan capture.

## Output

Each launch of a planned binary writes one trace file,
`capture_<launch>.ndjson` (or `.ndjson.zst`), even when no configuration
selects that launch. It contains, in order:

1. The `kernel_metadata` header, with `capture_config.protocol` set to
   `cutracer.plan-runtime/1` and the launch record in `runtime_identity`.
2. `plan_applied`: the configurations bound to this launch, whether the launch
   selected each one, and the instructions actually instrumented
   (`actual_points`, with their loaded bytes).
3. `plan_observation` records, one per selected warp, point and
   configuration. `reached_mask`, `executed_mask`, `selected_mask` and
   `valid_mask` describe the lanes; `registers` maps each register to 32
   per-lane values, `null` outside `valid_mask`.
4. `plan_capture_completion`: per-configuration device, receiver and writer
   counts. `complete` is `true` only when every selected configuration
   received and wrote every device record within budget and the expected
   number of warps entered the kernel. A selected configuration whose CTA,
   warp and lane selection contains no thread of the launch is incomplete
   with the error `no_selected_warps_in_launch`.
5. The usual `capture_completion` footer.

The manifest adds `plan_loaded`, `plan_applied`, `plan_capture_completion`,
`plan_gap` and `plan_error` events. `cutracer validate` checks the trace
records above, not the manifest.

## Checking a plan without a GPU

`instrumentation_plan_validate PLAN.json` checks the file format and prints
the canonical plan and its digests, or the first error with exit status 1.
The runtime limits above are checked only when a plan is applied. The
validator needs only the plan parser:

```bash
g++ -std=c++17 -Iinclude -Ithird_party -Ithird_party/picosha2 \
  tools/instrumentation_plan_validate.cpp src/instrumentation_plan.cpp \
  -o instrumentation_plan_validate
```
