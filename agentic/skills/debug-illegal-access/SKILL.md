---
name: debug-illegal-access
description: Use when a CUDA or Triton kernel on an NVIDIA GPU faults, for example "an illegal memory access was encountered" (CUDA error 700), "unspecified launch failure", a misaligned address, an out-of-bounds load or store, or an error reported by a later CUDA call that leaves the context unusable.
license: MIT
compatibility: NVIDIA GPU with the cutracer CLI and cutracer.so (pass --cutracer-so to trace commands if the CLI cannot find it). compute-sanitizer from a CUDA toolkit is optional.
---

# Debug an illegal memory access

Find the instruction that faulted and the address it used, then explain where that address came from.

The error usually surfaces at a later synchronizing call, not at the launch that faulted, so name the kernel first.

## 1. Localize the fault

If a CUDA toolkit is installed, the sanitizer reports the kernel, the PC, the access size, and the address:

```bash
mkdir -p <dir>
CUDA_VISIBLE_DEVICES=<gpu> timeout 900 cutracer compute-sanitizer --tool memcheck -o <dir> -- <command>
```

The report also names the nearest allocation and its size, which gives the bad address's offset from that allocation.
When the address is near 0, that allocation is meaningless; subtract the index times the access size to recover the base.
The sanitizer cannot share a process with `cutracer trace`, so it is a separate run.
The program's error text can change between runs and tools ("unspecified launch failure", "operation not supported on global/shared address space", "illegal memory access"); it is the same fault.

If the report and the SASS around the PC (step 4) already explain the address, for example a 32-bit index that wraps, go to step 5.
Trace only when the cause of the address is still open, typically a value loaded from memory or passed by the host.
When the faulting kernel reads an index or pointer that a smaller kernel wrote just before, trace that kernel first (step 3).
It completes, so its trace is whole.

Without the sanitizer, the last record of the kernel-events file names the faulting launch, because CUTracer synchronizes after each direct launch (a CUDA graph is synchronized only after the whole graph, so the record names the graph's last node):

```bash
mkdir -p <dir>
timeout -s INT 120 cutracer trace --kernel-events nostack -o <dir> -- <command> > <dir>/run.log 2>&1
tail -n 1 <dir>/cutracer_kernel_events_*.ndjson
```

`run.log` then shows the error inside a CUTracer call such as `cudaStreamIsCapturing ... src/cutracer.cu`; that is where the sticky error surfaced, not a CUTracer fault.

Without the sanitizer you also lack the faulting PC.
List the memory instructions of the faulting kernel (`cuobjdump -sass <binary>`, or step 4) and capture those, narrowed as in step 2.
In a large kernel, find the bad path statically: list every instruction that writes the register holding the address or the stored value, and compare their constants (masks, strides, bounds) across branches; an outlier is the candidate.

## 2. Capture the addresses and values

```bash
cutracer trace -i mem_addr_trace --channel-records 1 -k <kernel> -o <dir> -- <command>
```

`--channel-records 1` flushes each record as it is produced.
Without it, a faulting kernel loses every record still buffered when the context dies.
For a short kernel that is often all of them, and for a long one the records near the fault.
Even with it, the faulting access itself is often missing, and records from unrelated CTAs can also be lost; treat absence as weak evidence.
CUTracer prints `incomplete capture ... kernel_completed=0`; that is expected for a faulting kernel.

`-k` matches every launch of the kernel, and CUTracer writes each launch to its own `..._iter<N>_...ndjson` file.
The short last file is the faulting launch.

For a large grid, the trace grows with every warp.
`--trace-size-limit-mb` stops recording once the file is full, which is usually before the faulting CTAs run.
Instead, instrument only the instructions you need.
`INSTR_BEGIN` and `INSTR_END` select static instruction indices from `INSTR_BEGIN` up to but not including `INSTR_END`, and the index is the PC divided by 16.

```bash
INSTR_BEGIN=<pc/16> INSTR_END=<pc/16 + 1> cutracer trace -i mem_addr_trace --channel-records 1 -k <kernel> -o <dir> -- <command>
```

Which records survive the fault changes from run to run; if the record you need is missing, repeat the narrowed capture.
There is no capture filter by CTA.
`reg_trace` over a whole large grid can take many minutes; narrow it the same way.

To see values, not only addresses, use `-i mem_value_trace`.
It records each lane's address and the value each load returned or each store wrote (`is_load` tells which), so a bad pointer or index loaded from a table shows up in the trace.
Each lane's value is a list of 32-bit words, `[v]`, or `[lo, hi]` for 64 bits.
This prints, for each CTA, how many distinct values one PC loaded or stored and the smallest and largest; a null pointer shows as a minimum of 0, an out-of-range value as the maximum:

```bash
python3 - <trace>.ndjson <pc> <<'EOF'
import json, sys, collections
seen = collections.defaultdict(set)
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("type") != "mem_value_trace" or r["pc"] != sys.argv[2]:
        continue
    mask = int(r["active_mask"], 16)
    for i, v in enumerate(r["values"]):
        if mask >> i & 1:
            seen[tuple(r["cta"])].add(v[0] | (v[1] << 32 if len(v) > 1 else 0))
for cta, vs in sorted(seen.items()):
    print(cta, len(vs), hex(min(vs)), hex(max(vs)))
EOF
```

A run accepts one `INSTR_BEGIN`/`INSTR_END` range; to relate a store to an earlier load in a large grid, capture each in its own run and join the records on `cta`, `warp`, and lane, which works when the program is deterministic.

`-i reg_trace` records each instruction's register operands before the instruction executes.
Destination registers are listed too, and hold stale values at that point; read only the source operands.
The faulting instruction may have no record.
Then read the instruction that formed the address, for example `IMAD.WIDE R2, R7, 0x10, R2`, whose inputs give the base and the index.
For a memory operand such as `[R2.64]`, or a uniform pair such as `UR8:UR9`, CUTracer records only the first register, the low half.
When the high half matters, take the full address from `mem_addr_trace` or `mem_value_trace`.

Record format:

- A raw trace file starts with a `kernel_metadata` record and ends with a `capture_completion` record; skip both when scripting.
- The metadata's `instructions` table holds the SASS of the instrumented instructions (all of them, or only the `INSTR_BEGIN`/`INSTR_END` range), keyed by instruction index (the PC divided by 16).
- Memory records have `pc` (hex string), `cta` (`[x, y, z]`), `warp` (a global warp id, not CTA-local), `active_mask` (hex string), and `addrs`, 32 per-lane addresses, 0 for an inactive lane.
- Records have no `sass` field; `cutracer query` adds it from the metadata, which is why `--group-by sass` also shows a blank row with count 1 for the metadata record.
- Register records have `regs_indices` (register numbers) and `regs`, one list of 32 lane values per register in the same order.
- Uniform registers are under `uregs_indices` and `uregs`, one value each; an instruction with only uniform operands has no `regs` keys.

## 3. Find the bad address and where it came from

For a misaligned-address error, check `address % access_size` first.
When the base itself is misaligned, every CTA faults the same way and there is no outlier to find.

Otherwise compare each instruction's address range across CTAs, grouping by the full CTA coordinate (the batch is often `cta[1]`):

```bash
cutracer query <trace>.ndjson --group-by sass --count
python3 - <trace>.ndjson <pc> <<'EOF'
import json, sys, collections
per_cta = collections.defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("type") != "mem_addr_trace" or r["pc"] != sys.argv[2]:
        continue
    mask = int(r["active_mask"], 16)
    per_cta[tuple(r["cta"])] += [a for i, a in enumerate(r["addrs"]) if mask >> i & 1]
if not per_cta:
    sys.exit("no records at this PC")
mid = sorted(min(v) for v in per_cta.values())[len(per_cta) // 2]
for cta, v in sorted(per_cta.items(), key=lambda kv: -abs(min(kv[1]) - mid))[:5]:
    print(cta, hex(min(v)), hex(max(v)))
EOF
```

When the faulting access is missing from the trace, look for the CTAs that reached the previous PC but have no record at the faulting one.
When the faulting launch kept almost nothing, compare each PC's addresses and values with the earlier launches of the same kernel that completed.

| What the address shows | Likely cause |
|---|---|
| 0 plus a small offset | Null or uninitialized pointer, often an unfilled entry of a pointer table |
| `2^32 x element size` below where the index should land, so below the base, only for large shapes | 32-bit index overflow: a 32-bit `IMAD` feeding a sign-extending `IMAD.WIDE` |
| Exactly `2^32` (or another power of two) from a valid pointer | A 32-bit store landed on the high half of a 64-bit pointer |
| A constant multiple (often 4x) of the correct offset | Byte offsets used as element indices, or the reverse |
| Far outside the range of the same PC in other CTAs | Bad offset arithmetic, a missing mask, or a corrupt index |
| A valid range that keeps going past the buffer's end, from one thread | A loop bound loaded from a table (for example a CSR row offset) is wrong; the fault comes where the run leaves the last adjacent mapping |
| `address % access_size != 0` | Misaligned vector access (`.64`, `.128`) on an unaligned base |
| Inside no live allocation, but a buffer an earlier kernel used | The buffer was freed or reused before this kernel ran |

Before naming a cause, test it against the trace.
Compute the address your explanation predicts for a CTA that has records and for one that stopped: it must match the recorded addresses where records exist, and leave every buffer where the CTA stopped.
Judge the access the stopped CTA made, not the instruction: the same instruction can be in range for the CTAs that completed and out of range for a later CTA, lane, or iteration.

If the test knows the tensors' base pointers and sizes, check the address against each `[base, base + nbytes)`.
Without them, the address ranges the other PCs touched show roughly where each buffer starts; they are a lower bound on its size, not its size.

If the address comes from a loaded value (an index table, a pointer array), the load is usually correct and whoever wrote the value got it wrong.
Find that writer.

- If an earlier kernel wrote it, trace that kernel with `-i mem_value_trace -k <that kernel>`, and compare the values it loaded with the values it stored to tell a bad input from a bad computation.
  Check every stored value against what the reader assumes (offsets increase, indices stay below the buffer length), not a sample: one bad entry among millions is typical.
- If the host wrote it, the value reaches the kernel as an argument: a base loaded with `LDC` from `c[0x0][...]`, or a pointer loaded from a table the argument points to (`LDC.64`, then `LDG.E.64`).

Scalar arguments that shape an index (sizes, moduli, strides) are kernel arguments too.
To read an argument without the source, take `reg_trace` at the first instruction that reads the destination of the `LDC`, `ULDC`, or `LDCU`; at the load itself the destination still holds a stale value.
Comparing that value across launches (one file per launch) finds the launch whose argument is wrong.
When a debugger is available, breakpoints on the driver's copy and allocation calls (`cuMemcpyHtoD_v2`, `cuMemAlloc_v2`) show what the host copied and allocated.

To see whether a pointer was valid earlier, trace the earlier kernel that used it in the same run (`-k <earlier>,<faulting>`) and compare the addresses; a host call between the two launches (a free, a reuse) is then the cause.

## 4. Map the PC to source

```bash
cutracer sass <dir>/<cubin> --stdout | grep -B3 '/\*0a40\*/'
```

CUTracer names the dumped cubin `kernel_<hash>_<sha256>_<mangled name>.cubin`; `cutracer sass` takes one file, so pick the one for the kernel.
The cubin holds every kernel of the module, and PCs restart at 0 for each function.
Find the function's section first, or read the traced kernel's SASS from the `instructions` table of the trace's `kernel_metadata` record.
`cuobjdump -sass <binary>` also lists every kernel, including ones you did not trace.
When the code has line info, the `//## File ..., line N` comments give the source line.
PCs in the trace are hex, padded to at least four digits in the SASS listing (`/*0a40*/`, `/*628f0*/`).
The `desc[URx]` operand of `LDG`/`STG` is a memory access descriptor (cache policy), not a bounds check; sharing it across buffers is normal and does not cause a fault.

## 5. Fix and confirm

When the bad value came from memory or from the host, fix the writer; a bounds check or clamp in the faulting kernel hides the bug.
Rerun the reproducer without CUTracer, and `compute-sanitizer --tool memcheck` when available, and show that the fault is gone and the kernel still does all of its work (for example every batch still runs).
If you cannot rebuild the program, say that the fix is unconfirmed, or confirm it on a small reproducer whose SASS matches the faulting kernel.
