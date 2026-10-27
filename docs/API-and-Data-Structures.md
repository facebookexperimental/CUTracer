## Device→Host Message Types (from `include/common.h`) 📨

All message structs share a common `message_header_t` with a `type` field that identifies the record kind.

- `reg_info_t` (type = `MSG_TYPE_REG_INFO`)
  - Fields: `cta_id_{x,y,z}`, `warp_id`, `opcode_id`, `num_regs`, `reg_vals[32][MAX_REG_OPERANDS]`, `kernel_launch_id`, `pc`, `num_uregs`, `ureg_vals[MAX_UREG_OPERANDS]`.
  - `MAX_REG_OPERANDS` = 16, `MAX_UREG_OPERANDS` = 16 (configurable buffer limits, not hardware constraints).
- `mem_addr_access_t` (type = `MSG_TYPE_MEM_ADDR_ACCESS`)
  - Fields: `kernel_launch_id`, `cta_id_{x,y,z}`, `pc`, `warp_id`, `opcode_id`, `addrs[32]`.
  - ~300 bytes per record.
- `mem_value_access_t` (type = `MSG_TYPE_MEM_VALUE_ACCESS`)
  - Fields: `kernel_launch_id`, `cta_id_{x,y,z}`, `pc`, `warp_id`, `opcode_id`, `mem_space`, `is_load`, `access_size`, `addrs[32]`, `values[32][4]`.
  - `mem_space`: GLOBAL=1, SHARED=4, LOCAL=5 (matches `InstrType::MemorySpace`).
  - `is_load`: 1=load, 0=store.
  - `access_size`: bytes per lane (1, 2, 4, 8, or 16).
  - `values`: max 128-bit per lane (4×32-bit). Only `ceil(access_size/4)` registers are meaningful.
  - ~820 bytes per record. Always captured at `IPOINT_AFTER`.
- `opcode_only_t` (type = `MSG_TYPE_OPCODE_ONLY`)
  - Fields: `kernel_launch_id`, `cta_id_{x,y,z}`, `pc`, `warp_id`, `opcode_id`.
  - Lightweight; used for instruction histogram analysis.
- `tma_access_t` (type = `MSG_TYPE_TMA_ACCESS`)
  - Fields: `kernel_launch_id`, `cta_id_{x,y,z}`, `pc`, `warp_id`, `opcode_id`, `desc_addr`, `desc_raw[16]`.
  - `desc_addr`: 64-bit address of the TMA descriptor.
  - `desc_raw`: 128-byte raw TMA descriptor content (16×`uint64_t`).
  - Targets UTMALDG, UTMASTG, UTMAREDG instructions on Hopper/Blackwell GPUs.

### Host-only Structures

- `RegIndices` (C++ only, not transmitted over GPU channel)
  - `reg_indices`: vector of R register numbers (0–254) mapping `reg_vals` array positions to actual register numbers.
  - `ureg_indices`: vector of UR register numbers (0–62) mapping `ureg_vals` positions.
  - Collected at instrumentation time to avoid runtime overhead.

## Instruction Categories (from `include/instr_category.h`) 🏷️

### `InstrCategory` Enum

| Value | Description |
|-------|-------------|
| `NONE` | No category / unknown instruction |
| `MMA` | Matrix Multiply-Accumulate (HGMMA, UTCHMMA, UTCIMMA, UTCQMMA, UTCOMMA) |
| `TMA` | Tensor Memory Access (UTMALDG, UTMASTG, UTMAREDG) |
| `SYNC` | Synchronization (WARPGROUP.DEPBAR) |

### Category Functions

- `detect_instr_category(sass)` → `InstrCategory`: detects category from SASS string via substring matching against `INSTR_CATEGORY_PATTERNS`.
- `is_instr_category(sass, category)` → `bool`: checks if a SASS instruction belongs to a specific category.
- `should_instrument_category(category)` → `bool`: checks against `CUTRACER_INSTR_CATEGORIES` env filter. Returns `true` if no filter is set (all categories pass) or if the category is explicitly enabled.
- `has_category_filter_enabled()` → `bool`: returns `true` if `CUTRACER_INSTR_CATEGORIES` was set to a non-empty value.
- `get_patterns_for_category(category)` → `vector<const char*>`: returns all SASS patterns for a given category.
- `get_instr_category_name(category)` → `const char*`: human-readable name.
- `get_instr_pattern_description(sass)` → `const char*`: description of the matched pattern (e.g., "Hopper GMMA (sm_90)").

## Instrumentation Types (from `include/instrument.h`) 🔧

### `InstrumentType` Enum

| Value | Description |
|-------|-------------|
| `OPCODE_ONLY` | Lightweight: only collect opcode information |
| `REG_TRACE` | Medium: collect register values |
| `MEM_ADDR_TRACE` | Heavy: collect memory access addresses |
| `MEM_VALUE_TRACE` | Heavy: collect memory access with values |
| `TMA_TRACE` | TMA descriptor tracing |
| `RANDOM_DELAY` | Inject random delays on synchronization instructions |

### `OperandLists` Struct

Groups operand data for instrumentation functions:
- `reg_nums`: regular register numbers
- `ureg_nums`: uniform register numbers

## Runtime State (from `include/analysis.h`) 🧠

- `WarpState`
  - Tracks region collection (`is_collecting`), `region_counter`, and per-region `histogram`.
- `RegionHistogram`
  - Finalized histogram with `warp_id`, `region_id`, `histogram`.
- `CTXstate`
  - Channel: `channel_dev`, `channel_host`.
  - SASS maps: `id_to_sass_map`, `clock_opcode_ids`, `exit_opcode_ids`.
  - Deadlock detection state: `loop_states`, `active_warps`, `pending_mem_by_warp`, timestamps, and termination counters.

## Logging Helpers (from `include/log.h`) 📝

- `lprintf` (log file only), `oprintf` (stdout only), `loprintf` (both), `trace_lprintf` (kernel trace log only)
- Verbose macros: `loprintf_v` (verbose ≥ 1), `loprintf_vl(level, ...)` (verbose ≥ level), `lprintf_v`, `oprintf_v`
- `log_open_kernel_file(ctx, func, iteration, kernel_checksum)`, `log_close_kernel_file()`
- `generate_kernel_log_basename(ctx, func, iteration, kernel_checksum)` — builds `kernel_<checksum>_iter<idx>_<name>` prefix
- `init_log_handle()`, `cleanup_log_handle()`

## Key Host Functions 🧵

- `recv_thread_fun(void* args)`
  - Receives typed messages, dispatches analyses, writes outputs, and performs hang checks.
- `dump_histograms_to_csv(ctx, func, iteration, histograms)`
  - Writes `kernel_*_hist.csv` with `warp_id,region_id,instruction,count`.

## Key Instrumentation Helpers (from `src/instrument.cu`) 🧰

- `instrument_opcode_only(instr, opcode_id, ctx_state)`
- `instrument_register_trace(instr, opcode_id, ctx_state, operands)`
- `instrument_memory_addr_trace(instr, opcode_id, ctx_state, mref_idx)`
- `instrument_memory_value_trace(instr, opcode_id, ctx_state, mref_idx, mem_space)`
- `instrument_tma_trace(instr, opcode_id, ctx_state, ctx)`
- `instrument_delay_injection(instr, delay_ns)`
- `shouldInjectDelay(instr, patterns)` → `bool` — checks SASS against `DELAY_INJECTION_PATTERNS`

### Delay Injection Patterns

Instructions targeted for random delay injection (from `DELAY_INJECTION_PATTERNS`):
- `SYNCS.PHASECHK.TRANS64.TRYWAIT` — mbarrier try_wait
- `SYNCS.ARRIVE.TRANS64.RED.A1T0` — mbarrier arrive
- `UTMALDG.2D` — TMA load
- `WARPGROUP.DEPBAR.LE` — MMA wait
