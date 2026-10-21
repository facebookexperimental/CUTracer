---
name: debug-hanging-kernel
description: Use when a CUDA or Triton kernel on an NVIDIA GPU hangs, never returns, times out, or is suspected of deadlock, for example a barrier that never completes, an mbarrier or flag wait that spins forever, or warps that stop making progress.
license: MIT
compatibility: NVIDIA GPU with the cutracer CLI and cutracer.so.
---

# Debug a hanging kernel

Find which warps are stuck, where each one stopped, and which dependency is
never satisfied: a peer that never arrives, or a value that never becomes
what the wait expects.

## 1. Confirm the hang and name the kernel

Run the program without CUTracer first, then with `--kernel-events`, which
records launches without instrumenting instructions:

```bash
mkdir -p <dir>
timeout -s INT 60 stdbuf -oL <command> > <dir>/native.log 2>&1; echo $?
timeout -s INT 60 cutracer trace --kernel-events nostack -o <dir> -- <command> > <dir>/run.log 2>&1
tail -n 1 <dir>/cutracer_kernel_events_*.ndjson
```

Exit status 124 means `timeout` stopped a hang. Nothing else stops a hung run
in either mode, so keep the timeout a few times longer than a normal run.
Redirect the output to a file; piped into another command, the interrupt
also stops the pipe and the output is lost. `stdbuf -oL` keeps a C or C++
program's progress lines, which the interrupt otherwise discards with the
unflushed buffer. CUTracer writes each launch to
the events file as it happens, so the last record names the kernel that was
running when the run stopped. If the program exits normally, it does not
hang under these conditions. Run it a few more times, because a hang that
depends on timing may need several runs; if it never hangs, go back to
`investigate-kernel`. Report whether every run hangs or only some.

If the program hangs only under CUTracer, CUTracer caused it: it waits for
each kernel to finish before the program launches the next one, so a kernel
that waits for a later kernel (for example a flag set by a kernel on another
stream) waits forever. Lazy module loading can do the same without CUTracer;
`CUDA_MODULE_LOADING=EAGER` rules it out.

Run every command that starts the program under `timeout -s INT` and with
`CUDA_VISIBLE_DEVICES`, including a first `--help` probe: an unknown binary
may ignore its arguments and launch the hanging kernel. Afterwards, list GPU
processes with `nvidia-smi --query-compute-apps=pid --format=csv,noheader`
and kill only your own, by PID (`/proc/<pid>/cwd` shows where it started);
on a shared machine, killing by name stops other people's runs.

## 2. Capture the stuck state

```bash
cutracer trace -i reg_trace -a deadlock_detection --channel-records 1 \
    --kernel-timeout-s 60 --trace-size-limit-mb 500 -k <kernel> -o <dir> -- <command>
```

- `--channel-records 1` flushes after every record, so the trace holds each
  warp's last instructions, though the very last record can be lost.
- CUTracer ends a hung run itself and prints a warp report: after 15 seconds
  without trace data (`--no-data-timeout-s`) when warps block, after
  `Deadlock sustained for 3 checks` when every active warp is classified
  `LOOPING` or `BARRIER`, or at `--kernel-timeout-s`, whichever comes first.
  `--trace-size-limit-mb` caps the trace. The run exits nonzero (241 after
  SIGTERM, 247 after SIGKILL); that is expected.
- The report lists every active warp, so with a large grid `run.log` runs to
  megabytes. Read the `WARP STATISTICS` block first
  (`grep -A14 'WARP STATISTICS' <dir>/run.log`), then single warps.
- If the hang needs more than one launch to appear, keep `-k` narrow or the
  trace grows with every earlier launch. If the value the warps wait on may
  come from an earlier kernel, capture both in the same run:
  `-i reg_trace,mem_value_trace -k <writer>,<waiter>` (`-k` takes
  comma-separated name substrings). Addresses then match between the two.
- Instrumenting every instruction slows a kernel many times over. When
  the hang depends on timing across many CTAs (CTAs that wait on each other
  through global memory), a full capture may not reach the hang before the
  timeout; even a correct kernel can look hung. If the traced run stops far
  earlier in the work than the native one, instrument a few instructions
  only: find them with `cutracer sass` and prefix the trace command with
  `INSTR_BEGIN=<first> INSTR_END=<last + 1>`, static instruction indices
  (the PC divided by 16) in decimal, counted per function. Start with the
  store that publishes the value the waiters need (one record per CTA),
  then the waiting loop's load. If a window on a busy spin loop is still
  too slow, drop `--channel-records 1` for that run. With a window, the
  report's finished and never-executed counts mean nothing.

The report looks like this:

```
==> WARP STATISTICS for kernel_launch_id=0:
      Total warps:              64 (100.0%)
      Finished warps:           60 ( 93.8%)
      Active warps:              4 (  6.2%)
==> WARP STATUS SUMMARY for kernel_launch_id=0 (4 active warps):
    Warp18[4,0,0]: BARRIER(inactive=15s) no_loop(period=0, repeat=0)
    Warp19[4,0,0]: BARRIER(inactive=15s) no_loop(period=0, repeat=0)
    Warp41[10,0,0]: PROGRESSING(inactive=15s) no_loop(period=0, repeat=0)
    Warp57[14,0,0]: LOOPING(inactive=0s, period=6, repeat=812) last_seen=0s
```

Under each warp the report prints its `Last:` instruction, or the loop body
for a `LOOPING` warp. `Last:` is often wrong; take each warp's last
instructions from the trace (step 3).

The same data is in `<dir>/*_warp_status_summary.ndjson`.
`cutracer analyze warp-summary <trace>.ndjson` recomputes completed and
in-progress warps from the trace alone. It cannot see warps numbered above
the highest one in the trace, so take never-executed counts from the report.

| Status | Meaning |
|---|---|
| `BARRIER` | The warp's last instruction is a blocking barrier (`BAR.SYNC.DEFER_BLOCKING`) |
| `LOOPING` | The warp repeats a short instruction sequence, typically a spin wait on an mbarrier or a flag |
| `PROGRESSING` | Neither; check the last instructions. A slow spin loop (for example around an atomic) can show here instead of `LOOPING` |
| never executed | No record of the warp. In a hang, warps that never started are expected: the stuck resident CTAs keep the rest from launching, whatever the cause. It is an oversubscribed grid only when the resident CTAs wait for something that unstarted CTAs must do (for example a counter every CTA increments); first find what they wait for |

## 3. Read each active warp's last instructions

The trace is authoritative for what it recorded; a missing record proves
nothing when the trace was capped or windowed. Confirm the instruction the warp report prints
for each warp (`Last:`, `last_instruction`) against the trace; it has named
instructions near the start of the kernel for warps stuck much later.

```bash
cutracer query <trace>.ndjson -f "warp=<id>" --tail 5 --fields pc,sass,cta,warp
cutracer query <trace>.ndjson -f "cta=[4,0,0]" --group-by warp --count
# which warps executed a PC (query prints 10 rows unless given --all-lines)
cutracer query <trace>.ndjson -f "pc=<pc>" --fields cta,warp --all-lines
```

Write `<pc>` as the trace does (`0x1a0`, no leading zeros); other spellings
match nothing and print no error.

Warp IDs in the report and the trace are global. In a CTA of `W` warps, CTA
`c` holds warps `c*W` to `c*W + W-1`.

An active warp whose last record is past its synchronization (for example
the final store before `EXIT`) is not stuck. Its `EXIT` was not captured
before the run stopped. Set it aside.

## 4. Find the missing partner

For a warp stuck at a barrier, list what the barrier expects and who arrived:

- `BAR.SYNC.DEFER_BLOCKING 0x2, 0x80` waits on barrier 2 for 0x80 = 128
  threads. `BAR.ARV` arrives without waiting. A `BAR.SYNC` without a count
  waits for the whole CTA. When the count is a register (`BAR.SYNC 0x1, R2`),
  read its value in the stuck CTA with `--fields cta,warp,sass,regs` at that
  PC, and compare it with the number of threads that reach the barrier. If
  that record has no value for the register, read it at the peer's `BAR.ARV`,
  or trace the load that set it.
- In a warp-specialized kernel, check which warps take which role. A role
  that gets no warps, or a barrier count written for a different split of
  warps, leaves the barrier short.
- Query the tail of every other warp in the same CTA. A peer that exited, or
  that took a branch around its `BAR.ARV` or `BAR.SYNC`, is the missing
  arrival. A peer stuck at a different barrier means a circular wait.
- To see why a peer branched away, find the compare that last set the
  branch's predicate before the branch (predicates are reused, so do not stop
  at an earlier compare of the same register), and read its operand values.
  `-f "pc=<pc>" --fields cta,warp,sass,regs,regs_indices,uregs,uregs_indices`
  prints them (uniform registers such as `UR6` are in `uregs`). `regs[i]`
  holds the 32 lane values of register `regs_indices[i]`; only lanes set in
  `active_mask` are valid. Values are
  recorded before the instruction executes, so the value a load or an atomic
  returned appears at the next instruction that reads its register. Comparing them across CTAs shows
  which input differs in the CTA that hangs.
- For a `LOOPING` warp, find the event that should end the loop. A load,
  compare, and branch on a flag in shared or global memory (`LDS` or `LDG`,
  `ISETP`, `BRA`) waits for a store to that flag. First read the value the
  loop loads (`mem_value_trace` at the load's PC) and check it against the
  loop's exit test: a store may have happened with a value the loop rejects.
  Otherwise find the store's PC in the SASS, list which CTAs executed it, and
  in a CTA that did not, follow the storing warp's last instructions to the
  branch or `EXIT` that skipped it. A predicated instruction leaves no record
  where its guard is false, so a CTA missing from the list at `@P0 STS` may
  have reached it and skipped it. When CTAs take work items from an atomic
  counter, the item is not the CTA index; recover it from the address the
  CTA reads or writes.
  If the flag is a global counter that every CTA increments and every CTA
  waits on (a hand-written grid barrier), the CTAs that never executed the
  increment were never scheduled: the grid is larger than the GPU holds at
  once. That limit is the SM count times the CTAs one SM holds for this
  kernel, so the same grid can pass on one GPU and hang on a smaller one.
  An mbarrier wait (`SYNCS.PHASECHK` or a `try_wait` loop) waits for the
  arrive or the transaction count that completes the phase, from the warp or
  the copy that owns it. Transaction counts exist from sm_90 on: compare the
  bytes the arrive expects (the register operand of `SYNCS.ARRIVE.TRANS64`)
  with the sizes of the copies the stuck CTA issued (`UBLKCP` or TMA
  instructions; a `UBLKCP` size is in a uniform register).
- If the value the warp waits on was written by an earlier kernel (a flag, a
  count, a table entry), the hung launch's code can be correct. Find the
  earlier launches in the kernel-events file, capture the writer together
  with the waiter (step 2), and match the store to the stuck CTA's load by
  address.

Map PCs to source with the cubin CUTracer dumped next to the trace:

```bash
cutracer sass <dir>/<cubin> --stdout | grep -B3 '/\*0a70\*/'
```

`cutracer sass` takes one cubin. The SASS pads PCs to at least four digits
(`/*0a70*/`). The `//## File ..., line N` comments give the source line when the code was
compiled with line info (`-lineinfo` for CUDA; Triton keeps it by default).
Without them, map by PC and by the instruction pattern.

## 5. Fix and confirm

| Cause | Fix |
|---|---|
| A peer skips its arrive on some path | Arrive on every path, or skip the wait on the same predicate when the waiter's work is not needed |
| Thread count in the barrier differs from the threads that arrive | Make the count match the participating warps |
| Two groups wait on each other's barriers | Reorder so that one group arrives before it waits |
| A barrier inside divergent control flow | Move it out of the divergent region |
| The source places the barrier correctly but the SASS has it inside a branch (inline-asm barriers without `convergent`, or compiler scheduling) | Use the compiler's barrier intrinsic or mark the asm `convergent`; if the source is right and the SASS is not, report a compiler bug |
| A grid-wide spin wait on a grid larger than the GPU holds at once | Split the work into two kernels, or use a cooperative launch with the grid capped at the resident limit |
| An mbarrier phase that is never completed | Add the missing arrive or the expected transaction bytes; check the phase parity each waiter uses |

Rerun the reproducer with the fix, without CUTracer. The program must exit, with the expected output.
If you cannot rebuild the program, say that the fix is unconfirmed.

## Common mistakes

| Mistake | Instead |
|---|---|
| Treating every active warp as stuck | Read each one's last instruction |
| Instrumenting before confirming the hang | Instrumentation changes timing; confirm without it first |
| Reading no trace files after a `-k` run as "no kernel ran" | `-k` matched nothing; rerun without it and take the name from the kernel-events file |
