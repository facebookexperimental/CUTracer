# Codex: CLC response finding

Excerpt of the agent's frozen `conclusion.md`, limited to the CLC response finding and the evidence it relies on. Text is verbatim except that markdown links were removed and workspace paths and GPU indices were replaced with placeholders. File names refer to the agent's workspace; only the files in this directory are published.

## Finding 2: CLC response reuse lacks a proxy fence

Observed. CLC writes the 16-byte response at `0x920`, `UGETNEXTWORKID.BROADCAST`. The generic reads are `LDS.128` at `0x1b40` in the epilogue, `0x2b90` in MMA and `0x3730` in TMA. Their CLC-empty releases are at `0x1bc0`, `0x2c20` and `0x37c0`. The next request waits on the three-arrival empty barrier at `0x830`. Each reader path lacks a shared-memory proxy fence after its response read and before release. The epilogue's existing fences at `0x1370` and `0x1a90` precede its response read. MMA's initialization fence at `0x21e0` also precedes its response read. See helper snapshot, PTX context, PC map and original witnesses.

Observed rule; inferred violation. PTX ISA 9.4 section 9.7.15.18 specifies weak async-proxy response writes for the exact multicast CLC form emitted here. Its example explicitly fences the prior generic response read before the next async response write. Section 9.7.15.4 explains that proxy ordering composes with synchronization within each participating thread. Thread barriers and a value-dependent elected arrival alone do not provide that proxy ordering. See CLC contract and example and proxy fence rules. The return-`-1` path issues no next request; the violation concerns successful responses followed by reuse.

Inferred fix. Execute `fence.proxy.async.shared::cta` in every reading thread after response decoding and before the release synchronization. Preserve the existing role-wide thread barriers before the elected arrival. clc-fix.patch is the exact tested PTX intervention. proposed-clc-source.patch inserts `fence("async_shared")` in the installed helper's corresponding source position. The source patch was proposed only; its generated PTX was not separately tested.

Causal code-generation result. The tested CLC-only patch adds exactly three `MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S` pairs, plus padding. See SASS diff, instruction counts and fixed trace.

Observed trace result. The full original analyzer reports 654 MMA-read-to-CLC-write candidate pairs and 654 TMA-read-to-CLC-write pairs, with zero `fence_seen` on each. Candidates include later requests, not only adjacent iterations; they are not counts of failures. See full original analysis.

Observed before/after subset; inferred ordering improvement. Complete streams for three CTAs with the most CLC requests were retained from each launch. The CTAs differ between launches because scheduling differs. Each selected launch covers 16 requests across three CTAs. The counts below describe analyzer candidates in those subsets. See comparison, combined analysis and the `*.selection.json` provenance files.

| Reader role | Candidate pairs | Original fence_seen | CLC-only fence_seen | Combined fence_seen |
|---|---:|---:|---:|---:|
| MMA | 36 | 0 | 36 | 36 |
| TMA | 36 | 0 | 36 | 36 |
| Other epilogue warps | 108 | 69 | 108 | 108 |

Observed limitation. The analyzer still reports single-warp lane joins unresolved: each load covers 32 lanes, while the release and CLC issue use elected lanes. The original and patched PTX contain `bar.warp.sync` on these reader release paths, but ptxas emits no corresponding WARPSYNC there. The PTX supplies the required role synchronization; the trace does not independently prove its machine-level lane joining. The issuing epilogue warp's 36 same-warp candidates remain in a separate unresolved category. Other shared-memory candidate completion and descriptor limitations remain; none of the compared candidate classes loses a previously observed fence or synchronization mechanism. See the before/after analyzer files and PTX context.
