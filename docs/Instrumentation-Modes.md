## opcode_only (lightweight) 🪶
- Emits: opcode ID, warp ID, PC, kernel_launch_id, CTA IDs
- Use for: Proton instruction histogram; lowest overhead
- Instrument via: `CUTRACER_INSTRUMENT=opcode_only`
- May be auto-enabled by analyses (see below)

## reg_trace (medium) 🔬
- Emits: per-thread register values (plus unified registers), opcode ID, PC
- Use for: register value tracing and dataflow inspection
- Instrument via: `CUTRACER_INSTRUMENT=reg_trace`

## mem_addr_trace (heavy) 🧠
- Emits: 32-lane memory addresses for memory-reference instructions at
  `IPOINT_BEFORE`
- On SM90+: also emits static memory space, a cluster-shared lane mask, issuer
  cluster rank, and 32 lane-indexed target CTA ranks. CUDA 12.8 is the minimum
  supported toolkit for this runtime DSM attribution path.
- On pre-SM90: emits `cluster_attribution="unsupported_arch"`; the device
  variant does not reference cluster PTX.
- Use for: memory access pattern analysis
- Instrument via: `CUTRACER_INSTRUMENT=mem_addr_trace`
- Binary channel record size: 456 bytes (320 bytes before DSM attribution).
  NDJSON size varies with numeric values; a supported record retains all 32
  target-rank slots so lane indices remain stable.

## mem_value_trace (heaviest) 💾
- Emits: 32-lane memory addresses **and** values (up to 128-bit per lane)
- Additional fields: memory space (`GLOBAL`=1, `SHARED`=4, `LOCAL`=5), load/store indicator, access size in bytes
- Use for: detailed data flow analysis, value-level tracing
- Instrument via: `CUTRACER_INSTRUMENT=mem_value_trace`
- Record size: ~820 bytes per record
- Always captured at `IPOINT_AFTER` for consistent timing semantics
- ⚠️ `mem_value_trace` already includes address information; usually no need to enable both `mem_addr_trace` and `mem_value_trace`

## tma_trace (medium) 📦
- Emits: 128-byte TMA descriptor content and descriptor address
- Targets: `UTMALDG` (TMA load), `UTMASTG` (TMA store), `UTMAREDG` (TMA reduction)
- Use for: analyzing Tensor Memory Access patterns on Hopper (sm_90) and Blackwell (sm_100+) GPUs
- Instrument via: `CUTRACER_INSTRUMENT=tma_trace`
- Note: OSS build provides a stub; full implementation is internal

## random_delay (low overhead) 🔀
- Injects a fixed delay before synchronization instructions (mbarrier try_wait, mbarrier arrive, TMA loads, warpgroup depbar)
- Each instrumentation point is randomly enabled/disabled (50% probability)
- Use for: data race detection by disrupting timing
- Instrument via: `CUTRACER_INSTRUMENT=random_delay` (usually auto-enabled by `CUTRACER_ANALYSIS=random_delay`)
- Requires: `CUTRACER_DELAY_NS` to be set to a positive value

## Combining modes ➕
- `CUTRACER_INSTRUMENT` accepts comma-separated values. Analyses may also enable required modes implicitly.
- `proton_instr_histogram` auto-enables `opcode_only`.
- `deadlock_detection` auto-enables `reg_trace`.
- `random_delay` analysis auto-enables `random_delay` instrumentation.
- ⚠️ Enabling both `mem_addr_trace` and `mem_value_trace` produces a warning; prefer `mem_value_trace` alone if you need value data.

## Notes 📝
- When an analysis auto-enables a mode, you do not need to repeat it in `CUTRACER_INSTRUMENT`.
- Enabling additional modes increases overhead and output volume; prefer the minimal set that satisfies your analysis.
- Use `CUTRACER_INSTR_CATEGORIES` to further narrow instrumentation to specific instruction categories (MMA, TMA, SYNC). See [Configuration](Configuration.md#instruction-category-filtering).
