# AMD gfx950 (CDNA4, MI350)

Read [amd-common.md](amd-common.md) first. This file adds what is specific to
gfx950. Sources: the AMD Instinct CDNA4 ISA guide and LLVM `AMDGPUUsage`.

## Target facts

| Feature | gfx950 |
|---|---|
| LDS per CU | 160 KiB (CDNA4 ISA guide, state overview) |
| Direct global-to-LDS loads | As on gfx942, plus `dwordx3` and `dwordx4` forms: `buffer_load_dwordx3/x4` with LDS=1 and `global_load_lds_dwordx3/x4`. There is no `dwordx2` form. A `dwordx3` load "writes 3 dwords and skips the 4th" (CDNA4 ISA guide, 9.1.9). |
| Cache-coherence bits on memory instructions | `sc0`, `sc1`, `nt`, with the same scope mapping as gfx942 (LLVM cache-policy tables) |

## Differences that matter for an audit

- The wider direct-to-LDS loads count on `vmcnt` like the narrower ones:
  one count per instruction, whatever its width. Check `vmcnt(N)` arithmetic
  by counting instructions, and apply the partial-wait rule in the common
  file when a kernel mixes operation types.
- With 160 KiB of LDS, kernels keep more stages or staging buffers in LDS.
  Each extra buffer that is reused across iterations adds read-then-write
  edges to check.

## Agent-scope ordering

LLVM `AMDGPUUsage` has no separate gfx950 memory-model section; its
cache-policy tables give gfx950 the same `sc0`/`sc1` scope bits as gfx942.
Use the gfx942 code sequences, mark them inferred for gfx950, and confirm
against the AMDGCN the compiler actually emits for your fences and atomics.
