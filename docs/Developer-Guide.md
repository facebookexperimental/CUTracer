## Code map 🗺️
- `src/cutracer.cu`: NVBit callbacks, kernel filtering, SASS iteration, instrumentation dispatch, kernel-events writer wiring
- `src/instrument.cu`: NVBit call injection helpers (opcode/reg/mem/tma)
- `src/inject_funcs.cu`: device-side `instrument_*` functions pushing packets
- `src/analysis.cu`: `recv_thread_fun`, analysis dispatch, histogram CSV dump
- `src/env_config.cu`: env parsing, forcing modes, filters, intervals
- `src/log.cu`: `loprintf`/log file management for the main tool log
- `src/python_callstack.cpp`: host-side Python/PyTorch callstack capture used by `CUTRACER_CPU_CALLSTACK` and the kernel-events writer
- `src/trace_writer.cpp`: NDJSON / Zstd / CLP writer backend (also used by the kernel-events stream)
- `src/delay_inject_config.cu`: parsing/serialization of the delay config JSON for the `random_delay` analysis (dump/replay)
- `src/tool_func/flush_channel.cu`: precompiled NVBit device function used to flush the GPU→CPU channel; built into a `.fatbin` and embedded via `bin2c`
- `include/`: public headers for types and interfaces

## Add a new analysis 🧩
1) Define/extend message types if needed, or reuse existing (`opcode_only`, `reg_info`, `mem_access`).
2) In `recv_thread_fun`, branch on header type and implement the analysis state machine.
3) Emit results using a stable file format and naming (`generate_kernel_log_basename`).
4) Add an env switch under `CUTRACER_ANALYSIS` to enable it (consider auto-enabling minimal instrumentation).

## Add a new instrumentation 🔗
1) In `instrument_function_if_needed`, collect needed operands/metadata when iterating SASS.
2) Implement an `instrument_*` helper in `src/instrument.cu` using `nvbit_insert_call` and `nvbit_add_call_arg_*`.
3) Add the device-side callee in `src/inject_funcs.cu` that constructs and pushes a packet.
4) Gate enabling via `CUTRACER_INSTRUMENT` and document interplay with analyses.

## Clock and regions ⏱️
- Identify `CS2R SR_CLOCKLO` opcode IDs during instrumentation per function.
- Use them at analysis time to toggle collection per warp.

## Mapping opcode_id to SASS mnemonic 🔤
- During instrumentation, CUTracer records a per-function mapping `opcode_id -> SASS string` and the set of clock and EXIT opcode IDs.
- Analyses can look up the mnemonic via `ctx_state->id_to_sass_map[f][opcode_id]` after retrieving the current `CUfunction` from `kernel_launch_id`.

## Kernel boundaries and iteration 📌
- Each kernel launch is assigned a `kernel_launch_id`. When the receiver observes a change, it finalizes and dumps data for the previous kernel.
- Per-kernel iteration indices are tracked to build deterministic filenames.
