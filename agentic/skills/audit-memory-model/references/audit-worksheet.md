# Audit worksheet

## 1. Collect the artifacts

- Compile the exact configuration that runs, with a fresh compile cache. For
  Triton, set `TRITON_CACHE_DIR` to an empty directory. On NVIDIA the cache
  entry holds `.ttgir`, `.llir`, `.ptx`, and `.cubin`. On AMD it holds `.ttgir`,
  `.llir`, `.amdgcn`, and `.hsaco`.
- Disassemble the binary: `nvdisasm -c <kernel>.cubin` or `cutracer sass` on
  NVIDIA, `llvm-objdump -d --mcpu=<target> <kernel>.hsaco` on AMD.
- Note the GPU, driver, and compiler version. Triage results (step 5) depend
  on them.

## 2. Buffer map

**NVIDIA, from a trace.** Capture one launch with `cutracer trace -i
reg_trace,mem_addr_trace`, with enough work that each CTA reuses its buffers,
then run `scripts/trace_edges.py <trace> --json edges.json`. Its candidate
edges name the PCs that touch each shared buffer and the kind of access (generic
or async, read or write). Each witness gives a sample CTA, warps, and lane
masks, plus the chain the script matched (fence, releases, acquires, and any
warp-wide sync, each with its warp, PC, and mask) so you can check the link by
hand. The script assumes every arrive is a release and every successful wait
an acquire; for each link you rely on, confirm in the PTX that neither is
`.relaxed`. Build the buffer map from those PCs and the SASS operands. Add what the
script cannot show:

- buffers that MMA instructions read through descriptors, which the script
  lists separately;
- the full extent of TMA and bulk-copy buffers, of which the script sees only
  the start address (a `tma_trace` capture or the descriptors give the box
  size);
- code paths the launch did not execute.

**AMD, or without a trace.** Build one row per buffer that more than one role
touches: shared memory or LDS, and global buffers used for communication
between workgroups. Read offsets from the intermediate code and confirm them
in the machine ISA.

| Buffer | Offset, size | Accesses (instruction, PC, path, thread or wave role) | Reused across iterations? |
|---|---|---|---|

In either case, include every buffer, not only the obvious data tiles:

- operand stages filled by copy engines or direct-to-LDS loads;
- staging buffers written by threads and read by bulk stores;
- buffers that library helpers or compiler passes allocate, which do not
  appear in the kernel source;
- hardware barrier objects (for example NVIDIA mbarriers), with their init and
  their signalling operations.

## 3. Edge list

**NVIDIA, from a trace.** The same script prints one row per candidate edge:
the direction, the two PCs, the number of dynamic occurrences, and which
mechanisms it saw for them. Every row is a candidate for step 4, whatever
mechanisms it reports, and so is every operation listed as unresolved.

**AMD, or without a trace.** For each buffer, list each ordered pair that must
not overlap. For a buffer reused in a loop, include the pair between the last
access of iteration *i* and the first access of iteration *i+1*.

List every access to the buffer between two synchronization points, on every
path, in program order. The second access of a pair is the earliest one after
the first access, not the first one you find: a write placed before the
barrier you expected to protect it is the defect, and stopping at a later
write hides it.

| Buffer | First access (PC, path) | Second access (PC, path) | Kind (RAW, WAR, WAW) | Different paths or waves? |
|---|---|---|---|---|

## 4. Ordering per edge

For each edge, name the mechanism from the "What orders what" table in the
hardware references (the vendor's common file and the target file),
with the instruction and its PC. Write "none" when there is none. Write down
any argument you are tempted to accept that appears in the reference's
"Arguments that do not establish ordering" list, and reject it.

| Edge | Mechanism found (instruction, PC) | Accepted? |
|---|---|---|

Repeat for the reference's other required items, such as barrier
initialization, matrix-unit ordering, and cluster or agent scope.

## 5. Classify, then test the fix

Classify each gap from the rule and the ordering chain in step 4:

| Classification | Meaning |
|---|---|
| **Violation** | The rule requires a mechanism on this edge, and no accepted mechanism connects the two accesses. |
| **Unresolved** | The edge may be ordered by a mechanism you could not confirm (wrong barrier, unknown phase, unmatched operand). Say what would settle it. |
| **Ordered** | An accepted mechanism connects the two accesses. |

Then test the fix for each violation:

1. Add the instruction the rules require (fence, wait, or completion check)
   to a copy of the intermediate code, at the position the rules give. For a
   read followed by a write through another path, that is after the reads and
   before the release.
2. Recompile the copy with the same compiler and flags: `ptxas` for PTX, LLVM
   with the same `--mcpu` for AMDGPU. For Triton, the kernel override
   mechanism can load the modified code.
3. Diff the machine ISA, including scheduling or control fields if you can
   decode them.

Report the diff as evidence, not as the classification. An added instruction
shows that the shipped binary lacked it. An instruction that compiles to
nothing shows only that the modified input produced no additional instruction;
it does not show that the lowering provides the ordering, or
that other compilers or GPUs tolerate the gap.

On NVIDIA, trace the fixed build and re-run the script: the edge must now
show the required mechanism, and no other edge may get worse. Optionally run
the fixed build to confirm that results do not change. Do not expect the fix
to remove a failure. This class often has none.

To show that a violation can change results, widen the gap: delay one
participant between the last synchronization and its access (for example a
loop of sleep instructions in one warp or wave), then run the test on the
original and on the fixed code with the same delay. A delay adds latency only,
so it is a legal schedule. First check that the test can detect a corrupted
buffer: its inputs must make the output depend on the buffer's contents, and
its tolerance must be smaller than the change. A failure that the fix removes
under the same delay is causal evidence. No failure shows nothing.

## 6. Report

For each finding:

| Field | Content |
|---|---|
| Buffer | Name, offset, size |
| Edge | The two instructions with PCs, paths, and roles; the direction |
| Missing mechanism | The rule and the fence or completion that is missing |
| Fix position | The source location, and where the fix appears in the intermediate code and machine ISA |
| Classification | Violation, unresolved, or ordered, with the rule |
| ISA evidence | The diff with the fix applied |
| Trace observations | Candidate occurrences and the mechanisms seen, before and after the fix (NVIDIA) |
| Evidence state | Observed, inferred, or causal for each claim |
| Not covered | Specializations, cluster shapes, and paths that were not audited |

Order violations by the contract gap and the behavior it can affect, not by
whether the fix adds instructions. Keep the contract violation, the generated code, and any runtime
consequence in separate fields.
