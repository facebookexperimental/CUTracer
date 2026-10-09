# CUTracer investigates nondeterministic gradient corruption in TLX 2-CTA attention backward on NVIDIA B200: missing task-wide barrier before aliased TMEM overwrite

**Status:** Confirmed · **Tags:** `CUTracer`, `TLX`, `TMEM`, `WAR`, `data race`,
`B200`, `attention backward`

## TL;DR

- **Symptom:** `_attn_bwd_ws` intermittently corrupts `dq`, `dk`, and `dv` for
  shape `(4, 8, 1024, 128)` on NVIDIA B200. The first verification run failed
  on iteration 2.
- **Conclusion:** The eight compute warps do not rendezvous after reading QK
  from TMEM and before overwriting the aliased region with P. CUTracer found
  this intra-warpgroup TMEM WAR in all 256 launched CTAs.
- **Outcome:** A barrier at the proven boundary passed 100/100 natural runs and
  5/5 targeted-delay runs. The same CUTracer detector changed from 256 findings
  to zero on complete captures. The later historical fix includes this barrier,
  a second DP/dS barrier, and additional lifetime-order changes.
- **CUTracer contribution:** CUTracer established schedule sensitivity, reduced
  a reliable trigger, falsified two nearby release-order hypotheses, and
  localized the conflict to `LDTM.x64` at PC `0x1320` and `STTM.x32` at PC
  `0x1c60`.

## Investigation timeline

**Coding agent:** Codex · **Model:** GPT-5.6 · **Total investigation time:**
52 minutes to confirm the defect boundary.

Elapsed times are rounded from contemporaneous notes. `T+0` is the first
baseline run.

| Elapsed | Action and evidence | Decision or next action |
|---|---|---|
| T+0 min | Reproduced corruption on iteration 2. Recorded synchronization, incomplete-store, TMEM-lifetime, and compiler hypotheses before source inspection. | Establish whether scheduling changes the outcome. |
| T+1 min | CUTracer isolated `_attn_bwd_ws`, checksum `0x5351267c2d3ba402`, with a 512-thread block and 2-CTA cluster. | Target this launch. |
| T+16 min | Stress found a reproducing 50-μs WG0 delay after 5/5 clean baselines. | Inspect synchronization and reuse. |
| T+18 min | Filtered `reg_trace` returned zero findings with incomplete detector inputs. | Reject the negative result and collect a complete trace. |
| T+39 min | Corrected reduction found one reliable trigger at PC `0x5000`; 3/3 replays failed. | Use the trigger to test source hypotheses. |
| T+42–45 min | Two nearby release-order controls still failed. | Reject both explanations. Do not label the trigger PC as root cause. |
| **T+49 min — root cause localized** | Complete opcode trace and CUTracer detector found the QK-read/P-store WAR in 256/256 CTAs. | Add a collective barrier at the exact alias boundary. |
| **T+52 min — defect boundary confirmed** | One-barrier control passed 100/100 natural and 5/5 targeted runs. Complete detector result changed 256→0. | Preserve the claim for this specialization and compare with history. |

## Case identity

| Field | Value |
|---|---|
| Reproducer | [`repro.py`](repro.py) with the bundled [`d4e09c9e` kernel source](kernel/blackwell_fa_ws_pipelined_persistent.py) |
| Affected configuration | `_attn_bwd_ws` on NVIDIA B200; input `(4,8,1024,128)`; grid `(8,8,4)`; 512 threads; cluster `(2,1,1)` |
| Recognition signature | `MISSING_WARPGROUP_BARRIER`; `LDTM.x64` PC `0x1320` → aliased `STTM.x32` PC `0x1c60`; warps 0–7; 256/256 CTAs |
| Validated action | Use the complete upstream fix in [`facebookexperimental/triton#1992`](https://github.com/facebookexperimental/triton/pull/1992); the two-barrier subset is a validated control, and CUTracer directly proved the QK/P barrier for this configuration |

## Symptom and reproducer

The PyPI `fbtriton` distribution contains TLX and provides the `triton` Python
import. The reproducer uses that compiler wheel and loads only the affected
historical kernel source bundled with this example. The historical compiler
does not need to be built.

The following tested setup installs the current wheel in an isolated
environment. The kernel source, not the compiler version, defines the affected
artifact.

```bash
mkdir cutracer-tlx-tmem-war
cd cutracer-tlx-tmem-war

git clone https://github.com/facebookexperimental/CUTracer.git

uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python \
  torch numpy \
  --index-url https://download.pytorch.org/whl/cu130
uv pip uninstall --python .venv/bin/python triton
uv pip install --python .venv/bin/python fbtriton

.venv/bin/python CUTracer/examples/tlx_tmem_war/repro.py --iterations 30
```

Exit code `1` means that corruption was detected. Exit code `0` means that all
requested iterations passed. A passing run does not prove that a probabilistic
race is absent.

The first verification run produced:

```text
iteration 1: pass
RACE REPRODUCED on iteration 2: dq: mismatches=1393, max_abs=0.135401, dk: mismatches=1453, max_abs=0.159302, dv: mismatches=1038, max_abs=0.0756836
```

The current PyPI wheel at verification time reproduced the same mismatch
signature on iteration 4. This confirms the portable setup. It does not make
that wheel version part of the defect boundary.

Matched variants produced these results. The compact run record is in
[`evidence/run_results.txt`](evidence/run_results.txt).

| Variant | Changed variable | Result | Interpretation |
|---|---|---|---|
| Original | None | Failed on natural iteration 2 | Establishes the symptom |
| Original with CUTracer delay | Delay WG0 by 50 μs at reduced PC `0x5000` | Failed 1/1 targeted run | Provides a reliable schedule perturbation, not the conflicting instruction |
| Blind QK/P control | Add one eight-warp barrier before the aliased P store | Passed 100/100 natural and 5/5 targeted runs | Addresses the proven QK/P ordering edge |
| Two-barrier control derived from the historical fix | Apply [`fix.patch`](fix.patch) | Passed 100/100 natural runs after reveal | Validates the barrier subset for this reproducer; it is not a complete backport of the public fix |

## Evidence summary

After [installing CUTracer](../../readme.md#installation), collect and analyze
the decisive detector input with:

```bash
mkdir -p traces
cutracer trace \
  --instrument opcode_only \
  --kernel-filters _attn_bwd_ws \
  --trace-format zstd \
  --output-dir traces \
  -- .venv/bin/python CUTracer/examples/tlx_tmem_war/repro.py \
    --iterations 1

cutracer analyze data-race traces/*_attn_bwd_ws.ndjson.zst \
  --detector intra_wg_tmem_war \
  --no-ai \
  --format json \
  --output intra_wg_tmem_war.json
```

The reproducer exits with status `0` when the numerical race is not observed.
That status does not prove the kernel is race-free. Check the
capture-completion record before interpreting zero detector findings.

| Claim | State | Producing tool | Evidence | Boundary |
|---|---|---|---|---|
| Fixed inputs intermittently corrupt all three gradients. | observed | PyTorch and `repro.py` | [`run_results.txt`](evidence/run_results.txt) | One B200 and the pinned shape/toolchain |
| Relative WG0 progress changes the outcome. | causal | `cutracer stress` and saved-config replay | 5/5 baselines passed; a 50-μs WG0 delay reproduced; exact replay reproduced | Establishes schedule sensitivity, not the conflicting resource |
| PC `0x5000` is a reliable delay trigger. | observed | `cutracer reduce` with enable probability 1.0 and three confidence runs | 600 points reduced to one; 3/3 independent replays failed | A trigger can widen a race window without being the race location |
| QK read and aliased P overwrite lack a collective ordering edge. | causal | Complete CUTracer opcode trace and `intra_wg_tmem_war` detector | [`cutracer_detector_summary.json`](evidence/cutracer_detector_summary.json); [`sass_excerpt.txt`](evidence/sass_excerpt.txt) | 256/256 CTAs for this specialization; all eight compute warps |
| One barrier removes the proven hazard for this configuration. | causal | Matched runtime control and the same CUTracer detector | 100/100 natural passes; 5/5 targeted passes; detector 256→0 | Does not prove all shapes or equivalence to the complete historical fix |

An earlier category-filtered `reg_trace` produced zero race findings. Its
detector contracts reported missing address and TMA events plus incomplete
static and exit coverage. The zero result was therefore indeterminate. The
investigation required a complete opcode capture.

### Decisive evidence chain

CUTracer stress produced this result:

```text
5/5 instrumented baselines clean; WG0 at 50,000 ns reproduced
```

This result established schedule sensitivity. It changed the next action from
output initialization checks to synchronization and resource-reuse analysis.

The corrected reducer then produced:

```text
600 delay points -> one WG0 point at PC 0x5000 -> 3/3 replays reproduced
```

The delayed instruction was an SMEM staging store. Two controls moved nearby
`dq_empties` and `dp_empties` releases. Both controls still failed. These
results falsified the release-order hypotheses and prevented the delayed PC
from being misidentified as the conflict.

The complete opcode capture contained 16,267,429 records. Its completion
record reported `status=complete`, `kernel_completed=true`,
`channel_drained=true`, zero dropped records, and zero errors. CUTracer then
reported:

```text
intra_wg_tmem_war: 256
PC 0x1320 LDTM.x64, base UR8 -> PC 0x1c60 STTM.x32, base UR6
warps: 0,1,2,3,4,5,6,7
```

Source and SASS mapping connected PC `0x1320` to the QK TMEM load at
[line 1955](https://github.com/facebookexperimental/triton/blob/d4e09c9e8c9c1ba3b1dbc974756b32d613e94942/third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py#L1955).
PC `0x1c60` maps to the aliased P TMEM store at
[line 1964](https://github.com/facebookexperimental/triton/blob/d4e09c9e8c9c1ba3b1dbc974756b32d613e94942/third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py#L1964).
The eight compute warps can progress independently. A fast warp can therefore
overwrite the aliased TMEM region before another warp completes its QK read.

The blind control added one eight-warp rendezvous immediately before the P
store. The generated SASS placed `BAR.SYNC ... 0x9, 0x100` directly before the
`STTM`. Its complete capture contained 16,262,914 records with zero drops and
zero errors. The same isolated detector returned zero findings. This zero is
meaningful only for the QK/P detector and this complete capture.

## How CUTracer changed the investigation

CUTracer stress proved that WG0 timing affected the result. Reduction created
a reliable one-point trigger. That trigger made negative source controls
repeatable and falsified two nearby release-order explanations.

The complete opcode trace then exposed the exact TMEM read and overwrite pair.
The detector preserved its aggregation unit: one violation per CTA, for
256/256 launched CTAs. The treatment trace verified that the barrier removed
the same detector finding. CUTracer therefore changed both the diagnosis and
the next source edit.

## Resolution and validation

The blind control inserted an eight-compute-warp rendezvous after all use of
`qkT` and before `tlx.local_store(p_tiles[...], ppT)`. This is a discriminating
control for the proven hazard.

The public fix in
[`facebookexperimental/triton#1992`](https://github.com/facebookexperimental/triton/pull/1992)
includes the two named barriers and additional lifetime-order changes. The
checked-in [`fix.patch`](fix.patch) contains only the two barriers. Apply this
matched control with:

```bash
(cd CUTracer/examples/tlx_tmem_war && patch -p1 < fix.patch)
.venv/bin/python CUTracer/examples/tlx_tmem_war/repro.py --iterations 100
```

| Gate | Original | Blind QK/P control | Two-barrier control | Acceptance criterion |
|---|---:|---:|---:|---|
| Natural reproducer | Failed on iteration 2 | 100/100 passed | 100/100 passed | No mismatch |
| Targeted 50-μs WG0 perturbation | 1/1 failed | 5/5 passed | Not rerun | No mismatch |
| CUTracer `intra_wg_tmem_war` | 256 CTA findings | 0 findings | Not rerun | Complete capture and zero QK/P findings |
| Capture completeness | Complete; 0 drops/errors | Complete; 0 drops/errors | Not rerun | Complete and drained |
| Other shapes and performance | Not measured | Not measured | Not measured | Open follow-up |

## Blind investigation and post-investigation validation

### Blind phase

The investigator received the affected kernel source and `repro.py`. The
investigator did not receive the known diagnosis, later commits, public pull
request, or `fix.patch`. Initial hypotheses were recorded before kernel
inspection.

The blind phase proved the QK-read/P-store hazard. It proposed one task-wide
barrier at that boundary. The phase ended before the historical fix was
revealed.

### Post-investigation validation

The historical fix agreed with the independently diagnosed QK/P barrier. It
uses `QK_READ_DONE_BAR` and `NUM_COMPUTE_THREADS = 8 * 32` instead of the
blind control's literal barrier arguments.

The historical fix also adds a second barrier between the DP TMEM load at
[line 1978](https://github.com/facebookexperimental/triton/blob/d4e09c9e8c9c1ba3b1dbc974756b32d613e94942/third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py#L1978)
and the aliased dS store at
[line 1985](https://github.com/facebookexperimental/triton/blob/d4e09c9e8c9c1ba3b1dbc974756b32d613e94942/third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py#L1985).
The affected SASS contains this `LDTM`/`STTM` pair. However, the blind detector
did not report it for this capture. The one-barrier control also passed the
tested runtime gates without the second barrier. Thus this investigation gives
structural support, but not runtime-causal proof, for the DP/dS barrier.

The public change also contains lifetime-order corrections outside the two
barriers. This investigation did not backport or isolate those corrections.
The production recommendation remains the complete public fix. The post-reveal
two-barrier control passed 100/100 natural iterations. The report does not
claim that either control is equivalent to the complete public fix.

## Comparison with the human investigation

Human report: *The barrier your mbarrier doesn't cover*. The source is
internal, so this public report omits its URL and private-only measurements.
The human report remained hidden until the blind conclusion was complete.
The public code reference is
[`facebookexperimental/triton#1992`](https://github.com/facebookexperimental/triton/pull/1992).

| Topic | Agent with CUTracer | Human report | Gap or next test |
|---|---|---|---|
| Immediate cause | Proved that the QK `LDTM` races with the aliased P `STTM` without an eight-warp rendezvous. | Describes the same intra-task TMEM WAR and explains that mbarriers order partitions, not warps inside one partition. | Agreement. No corrective test is required for this boundary. |
| TMEM alias coverage | Dynamically proved QK→P. It observed DP→dS structurally but did not prove that pair causal. | Identifies both QK→P and DP→dS as the same hazard shape. | The agent missed the second dynamic hazard. A detector run on a specialization that exposes DP→dS, followed by a DP-barrier-only A/B, would close this gap. |
| Warp-task localization | A 50-μs delay on WG0 reproduced once in 61 stress trials. | Targeted delays localized the hazard to both warpgroups of the eight-warp compute task. | The agent localized timing to part of the compute task but did not map the complete task. Repeat per-warpgroup trials with one delay configuration and retain each denominator. |
| Finding unit | Reported 256 violations: one instance in each of 256 CTAs, all for the same PC pair. | Reports one real WAR site. | The counts use different units. The agent report preserves both the static site and per-CTA instance counts. |
| Natural reproduction | Failed on iteration 2 but did not estimate a rate. | Uses a larger fixed-count 2-CTA baseline campaign to estimate the natural failure rate. | The agent confirmed the symptom but missed the stable frequency estimate. Run a fixed-count baseline campaign and report its denominator. |
| Sibling coverage | Tested only the naturally failing 2-CTA specialization. | Connects the 2-CTA case to a 1-CTA sibling that needs timing perturbation. | The agent missed the sibling survey. Run the same detector and barrier A/B on the 1-CTA path. |
| Independent oracle | Used numerical comparison, CUTracer stress, complete opcode traces, and the CUTracer detector. | Also used Compute Sanitizer before and after the barrier change. | The agent missed cross-tool corroboration. Run Compute Sanitizer on original and treatment artifacts as separate processes. |
| Fix scope | Proved one barrier sufficient for the supplied specialization. It later tested the two-barrier subset for 100 natural iterations. | Uses both barriers and records additional lifetime-order changes in the public fix. | The agent controls are not equivalent to the complete public fix. Test or review each additional lifetime invariant separately. |
| Broader engineering result | Did not test performance or compiler insertion. | Reports no measurable barrier cost and identifies automatic compiler insertion as the preferred long-term fix. | These items extend beyond immediate diagnosis. Add a benchmark and an alias-aware compiler regression before making broader adoption claims. |

The agent produced one useful artifact that the human narrative reports at a
different level: a complete trace with one finding per CTA and a matched
detector result of 256→0. This gives explicit capture completeness and dynamic
coverage for the QK/P boundary.

The largest agent omission was the second DP/dS hazard. The agent's conclusion
was therefore correct but incomplete. The missing sibling survey and
Compute-Sanitizer A/B also prevented the agent from reaching the human report's
broader claim that one mechanism explains both 1-CTA and 2-CTA behavior across
two independent perturbation tools.
