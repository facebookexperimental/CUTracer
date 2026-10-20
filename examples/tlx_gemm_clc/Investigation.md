# CUTracer investigates a TLX CLC GEMM that passes its tests on B200: missing async-proxy fence before the CLC response buffer is reused

**Status:** Confirmed. The defect is a violation of the PTX ordering rules. No runtime failure was observed. **Example:** [`examples/tlx_gemm_clc`](README.md).

**Tags:** CUTracer, `audit-memory-model`, `trace_edges.py`, TLX, Triton, cluster launch control (CLC), `clusterlaunchcontrol.try_cancel`, `UGETNEXTWORKID`, `fence.proxy.async`, generic proxy, async proxy, write-after-read, warp specialization, persistent GEMM, B200, sm_100a, blind investigation, Claude, Codex, Muse.

## TL;DR

- **Symptom:** none at runtime. The kernel passes its reproducer on four shapes, with 0 mismatched iterations out of 50 for each.
- **Conclusion:** in TLX `clc_consumer`, each consumer reads the CLC response from shared memory and then releases the buffer. The next `try_cancel` writes the same 16 bytes through the async proxy. No `fence.proxy.async` orders the read before that write.
- **Outcome:** three coding agents audited the kernel blind, side by side, with the CUTracer `audit-memory-model` skill installed. Two of the three found the defect and fixed it at the position of the merged upstream fix. The third agent saw the buffer reuse but left it unresolved.
- **CUTracer contribution:** the two successful agents found the defect in the PTX first. A CUTracer trace then confirmed that the three reads and the `try_cancel` write use the same bytes, with the buffer reused across CTAs. A trace of the patched build showed the fence on every read in the MMA and TMA warps.

## Investigation timeline

Each agent ran once, with the same prompt, its own workspace, and its own B200 GPU. Times are from each agent's timestamped `timeline.md`.

| Agent (client, model) | Total | Root cause localized | Defect boundary confirmed |
|---|---|---|---|
| Claude Code 2.1.291, `claude-opus-5-5` | 18 min | **T+3 min**: the PTX shows a generic response read, then the release, with no proxy fence, in all three consumer roles | **T+16 min**: the patched PTX adds 3 × `MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S`. The trace of the patched build shows the fence on every MMA and TMA read. A source fix in `clc_consumer` gives the same SASS. |
| Codex CLI 0.162.1, `gpt-6.1-sol` | 12 min | **T+2 min**: the same PTX observation, after a CLC hypothesis at T+1 min | **T+6 min**: the patched PTX adds the same 3 fence pairs. The trace comparison shows the fence on every MMA and TMA read. |
| Muse Code 1.4.11, `muse-spark-1.3-internal` | 11 min | Not localized. At T+9 min the agent saw that the read and the next `try_cancel` share one offset, and recorded the edge as unresolved. | n/a |

## Case identity

| Field | Value |
|---|---|
| Reproducer | [`repro.py`](repro.py) and [`affected.patch`](affected.patch) in this directory, with `fbtriton` 3.7.4 from PyPI |
| Affected component | `triton.language.extra.tlx.dynamic_launch.clc_consumer`, used by `matmul_kernel_tma_ws_blackwell_clc` (TLX tutorial `blackwell_gemm_clc`) |
| Recognition signature | SASS: each `LDS.128 [..+0x200e0]` (response read) is followed by `SYNCS.ARRIVE.TRANS64.A1T0 [..+0x20100]` (release) with no `FENCE.VIEW.ASYNC.S` between them. `trace_edges.py`: `generic_read->async_write` rows from the MMA-warp and TMA-warp reads to `0x920` with no `fence_seen`. |
| Validated action | Fix: `fence("async_shared")` in `clc_consumer` after the response is decoded and before the release. Upstream: [`facebookexperimental/triton#1723`](https://github.com/facebookexperimental/triton/pull/1723), included in `fbtriton` 3.7.4. |

## Symptom and reproducer

The [README](README.md) gives the setup and the commands for both states. The affected state is `fbtriton` 3.7.4 with [`affected.patch`](affected.patch) applied. That patch removes the fence and restores the source before the fix.

The agents got the earlier reproducer, v1. It sets `atol=rtol=1e-2` and scales the inputs to `(randn+1)/K`. The [prompt](evidence/prompt.md) gives its result: 0/50 mismatched iterations on each of four shapes. Codex found that v1 cannot reject wrong output at `K=512`. There, the reference maximum is about 0.0028, so an all-zero output passes the 0.01 tolerance. The [`repro.py`](repro.py) in this directory uses unscaled inputs and a tolerance that scales with the output, so a zeroed or stale tile fails.

| Variant | Changed variable | Result | Evidence |
|---|---|---|---|
| Affected | `affected.patch` applied | v1: 0/50 on 4 shapes. `repro.py`: 0/100 on 4 shapes. | [`run_results.txt`](evidence/run_results.txt) |
| Fixed | stock `fbtriton` 3.7.4 | `repro.py`: 0/100 on 4 shapes | [`run_results.txt`](evidence/run_results.txt) |

The reproducer does not discriminate the two states. The defect is visible in the PTX, in the SASS, and in a CUTracer trace.

## Evidence summary

| Claim | State | Producing tool | Evidence | Boundary |
|---|---|---|---|---|
| The kernel passes its reproducer | observed | `repro.py` | [`run_results.txt`](evidence/run_results.txt) | B200, four shapes, one configuration |
| Each consumer releases the response buffer with no proxy fence after reading it | observed | PTX and SASS reading (all three agents); `nvdisasm` | [`sass_excerpt.txt`](evidence/sass_excerpt.txt); [Claude excerpt](evidence/runs/claude/conclusion_excerpt.md); [Codex excerpt](evidence/runs/codex/conclusion_excerpt.md) | The configuration in `repro.py` |
| The reads and the next `try_cancel` write use the same 16 bytes, and CTAs reuse the buffer | observed | CUTracer `reg_trace,mem_addr_trace` capture, then `trace_edges.py` | Claude: 108 CTAs reuse the response in a 256-tile launch ([edges](evidence/runs/claude/edges_affected.txt)). Codex: a 512-tile launch ([edges](evidence/runs/codex/edges_affected_full.txt)). | One launch each |
| The write-after-read across proxies is unordered | inferred | PTX ISA 9.4 rules, as quoted in the skill references | Section 8.6: "A proxy fence is required to synchronize memory operations across different proxies." Section 9.7.15.18: the `.multicast::cluster::all` form of `try_cancel`, which this kernel uses, writes the response with "weak async-proxy writes". | Rule-based; no failure observed |
| The fence is the missing mechanism | causal | PTX patch, `ptxas`, `nvdisasm`, CUTracer trace | The patched PTX adds exactly 3 × `MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S`, one between each response read and its release ([Codex PTX patch](evidence/runs/codex/clc_fence_ptx.patch)). In the patched trace, the MMA and TMA rows show `fence_seen` on every occurrence: Claude 108/108 each ([edges](evidence/runs/claude/edges_fixed.txt)); Codex 36/36 each in 3 CTAs ([edges](evidence/runs/codex/edges_fixed_3cta.txt)). | ptxas 13.1, sm_100a |

The chain for the successful runs:

```text
PTX: ld.shared.b128 of the response -> mbarrier.arrive, no fence.proxy.async (3 roles)
  -> CUTracer trace: same 16 bytes as the next UGETNEXTWORKID, reused across CTAs; fence_seen=0 in the MMA and TMA warps
  -> PTX patch adds the fence -> SASS gains 3 fence pairs -> patched trace shows fence_seen on every MMA and TMA read
```

Two details matter when reading the `trace_edges.py` rows:

- **The MMA and TMA rows also report `sync_lanes_unresolved`, in both states.** All 32 lanes read the response, but one elected lane arrives. The PTX joins the lanes with `bar.warp.sync -1` before the arrive, and PTX ISA 9.7.15.2 says that `bar.warp.sync` orders memory among the participating threads. ptxas emits no `WARPSYNC` for it, so the trace cannot show the join. Claude and Codex both checked the PTX and classified these rows from it. These rows are not a second defect.
- **The epilogue row is partly `fence_seen` in the affected state of Codex's trace** (870 of 2616 occurrences). An unrelated epilogue fence, for the output staging buffer, executes between the read and the release on some paths. The MMA and TMA rows have no such fence. They carry the finding.

## How CUTracer helped

- CUTracer **confirmed** the mechanism at runtime. The trace showed that the three response reads (PCs `0x1b40`, `0x2b90`, `0x3730`) and the `try_cancel` write (PC `0x920`) use the same 16 bytes. It also showed that CTAs reuse the buffer, which is the condition for the write-after-read.
- CUTracer **verified** the fix. `trace_edges.py` on the patched build showed `fence_seen` on every MMA and TMA read, and no other edge got worse.
- CUTracer did **not** localize the defect first. Both successful agents found it in the PTX, by following the skill's audit procedure, before any trace existed.
- Muse traced a 512 × 512 × 128 launch. That launch has 8 tiles for 148 SMs, so no CTA reused the response buffer, and the trace had no edge to show. The skill tells the agent to trace "more tiles than SMs" for a persistent kernel. The miss came from that skipped step.

## Resolution and validation

The fix is [`facebookexperimental/triton#1723`](https://github.com/facebookexperimental/triton/pull/1723). It adds `fence("async_shared")` in `clc_consumer` after `_clc_query` and before `barrier_arrive`. `fbtriton` 3.7.4 includes it.

| Gate | Affected | Fixed | Acceptance criterion |
|---|---:|---:|---|
| `repro.py`, 4 shapes | 0/100 | 0/100 | No regression. Not discriminating. |
| SASS: `FENCE.VIEW.ASYNC.S` count | 4 | 7 | 3 added, one between each response read and its release ([`sass_excerpt.txt`](evidence/sass_excerpt.txt)) |
| `trace_edges.py`, MMA and TMA rows, `fence_seen` | 0 of 108 each | 108 of 108 each | Every read is fenced (Claude's traces: [affected](evidence/runs/claude/edges_affected.txt), [patched](evidence/runs/claude/edges_fixed.txt)) |

Claude's patched trace came from a PTX override with the same fence. Claude then applied a source patch ([`clc_consumer_fence.patch`](evidence/runs/claude/clc_consumer_fence.patch)), recompiled, and observed the same 3 fence pairs. The reproducer passed 0/50 on all four shapes. Claude restored the source afterwards.

## Blind investigation and later validation

### Blind phase

| Input or milestone | Evidence or decision available at that time |
|---|---|
| Starting prompt and artifacts | The [prompt](evidence/prompt.md): the v1 reproducer, the installed `fbtriton` 3.7.4 with the affected source, the OSS `cutracer` CLI (commit `b9cdb15`), standard NVIDIA tools, and "0/50 mismatched iterations on four shapes". The prompt names no defect, fix, or skill. It says only that CUTracer's agent skills are installed. |
| Skills available | The CUTracer OSS skills `audit-memory-model` and `report-case-study`, found through each client's own skill mechanism. All three agents used `audit-memory-model` without being told to. |
| Initial hypotheses | Claude: audit against the memory model, because the reproducer passes. Codex: "passing comparisons may hide missing cross-proxy ordering", and then CLC response reuse. Muse: shared-memory edges, then Tensor Memory hand-offs. |
| Tool results | Claude and Codex: the PTX localization, trace, patch, and patched trace above. Muse: a trace with no buffer reuse, and edges listed as ordered where checkable. |
| Blind conclusions | Claude and Codex: a violation, with a fix and an ISA diff. Muse: the CLC reuse edge left unresolved. Excerpts: [Claude](evidence/runs/claude/conclusion_excerpt.md), [Codex](evidence/runs/codex/conclusion_excerpt.md), [Muse](evidence/runs/muse/conclusion_excerpt.md). |

The agents' conclusions also discuss other edges. The published excerpts cover only the CLC response finding.

**What this blind run does and does not test.** The `audit-memory-model` skill was written while this case was studied. Its sm_100 reference describes `clusterlaunchcontrol.try_cancel` and states that reusing the response buffer is a generic-read-then-async-write edge. These runs therefore show that agents apply the procedure correctly on this kernel. They do not show that the skill finds a defect class that its authors did not study.

### Post-investigation validation: comparison with the merged fix

| Topic | Agents | Merged fix (`#1723`) | Gap |
|---|---|---|---|
| Root cause | Claude and Codex: a missing `fence.proxy.async` between the response read and the release, in all three consumer roles | The same | None |
| Fix position | Claude: `fence("async_shared")` after `_clc_query`, before `barrier_arrive`, tested at source level. Codex: the same position ([proposed patch](evidence/runs/codex/proposed_source.patch)), tested only as a PTX patch. Codex's patch omits `tl.constexpr` around the fence argument. | `fence(tl.constexpr("async_shared"))` at the same position | Codex's source patch was not compiled |
| Finding unit | One static site per consumer role (3 sites). The trace counts are dynamic occurrences per CTA and per reuse. | One function, `clc_consumer`, shared by all roles | Counts differ by unit, not by conclusion |
| Missed | Muse did not localize the defect, because its trace had no buffer reuse | n/a | Trace a launch with more tiles than SMs, as the skill says |
