# AMD gfx942 (CDNA3, MI300)

Read [amd-common.md](amd-common.md) first. This file adds what is specific to
gfx942. Sources: the AMD Instinct MI300 (CDNA3) ISA guide and the LLVM
`AMDGPUUsage` sections "Memory Model GFX942" and "AMDHSA Memory Model Code
Sequences GFX942".

## Target facts

| Feature | gfx942 |
|---|---|
| LDS per CU | 64 KiB (ISA guide, state overview) |
| Direct global-to-LDS loads | `buffer_load_{ubyte,sbyte,ushort,sshort,dword,format_x}` with LDS=1, and `global_load_lds_*` / `scratch_load_lds_*` up to dword (ISA guide, 9.1.9 and the FLAT opcode list) |
| Cache-coherence bits on memory instructions | `sc0`, `sc1`, `nt` |
| L2 caches | "The gfx942 can be configured as a number of smaller agents with each having a single L2 shared by all CUs on the same agent, or as fewer (possibly one) larger agents with groups of CUs on each agent each sharing separate L2 caches." (LLVM). So one agent can have several L2 caches. |

## Agent-scope ordering

Communication between workgroups whose CUs use different L2 caches goes
through memory that another L2 may cache. From the LLVM GFX942 code-sequence
table:

| Operation | Sequence |
|---|---|
| `fence release`, agent scope | `buffer_wbl2 sc1=1`, then `s_waitcnt lgkmcnt(0) & vmcnt(0)` |
| `fence acquire`, agent scope | `s_waitcnt lgkmcnt(0) & vmcnt(0)`, then `buffer_inv sc1=1` |
| `load atomic acquire`, agent scope, global | the load with `sc1=1`, `s_waitcnt vmcnt(0)`, then `buffer_inv sc1=1` |
| System scope | adds `sc0` (for example `buffer_inv sc0=1 sc1=1`) |

The table lists variants by address space and execution mode (for example,
OpenCL local memory and tgsplit mode); take the row for your case. Ordering
that covers only one L2 is not enough between CUs on different L2 caches.
