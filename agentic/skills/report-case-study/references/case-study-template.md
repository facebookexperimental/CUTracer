# CUTracer case-study template

Use the required core for every report. Include optional sections only when
they help prove the conclusion. Remove all bracketed guidance before delivery.

```markdown
# CUTracer investigates <exact symptom> in <kernel/operator/component> [on <platform>]: <confirmed cause or narrowest proven boundary>

**Status:** <confirmed / bounded / unresolved> · **Tags:** <tool, subsystem,
failure class, platform, and kernel/operator terms>

## TL;DR

- **Symptom:** <failure, trigger, affected kernel, and impact>.
- **Conclusion:** <causal mechanism or narrowest proven boundary>.
- **Outcome:** <fix, workaround, control, or open follow-up and its validation>.
- **CUTracer contribution:** <isolated / observed / confirmed / falsified /
  bounded; name the decisive CUTracer evidence>.

## Investigation timeline

<Optional. Remove this section when the sequence does not teach a reusable
lesson. Use rounded elapsed times only when artifact timestamps or
contemporaneous notes support them. Mark root-cause localization and later
defect-boundary confirmation separately. Do not invent timing.>

**Coding agent:** <client or agent name> · **Model:** <exact model, or the
narrowest runtime-provided family with the limitation stated> · **Total
investigation time:** <T+N minutes to the final reported milestone>.

| Elapsed | Action and evidence | Decision or next action |
|---|---|---|
| T+0 min | <starting evidence and hypotheses> | <first test> |
| T+N min | <localizing evidence> | <changed next action> |
| **T+N min — root cause localized** | <decisive mechanism evidence> | <bounded causal conclusion> |
| **T+N min — defect boundary confirmed** | <matched discriminating test> | <narrowest proven boundary> |

## Case identity

| Field | Value |
|---|---|
| Reproducer | <durable path/link and revision> |
| Affected component/configuration | <exact symbol and smallest configuration needed to recognize the case> |
| Recognition signature | <error, PC, instruction, address/value pattern, or verdict> |
| Validated action | <fix, workaround, control, or open follow-up> |

Add platform, software, or artifact identity rows only when they distinguish
the affected path or compared artifacts. Label historical environment facts
separately from the environment used for current verification.

## Symptom and reproducer

<State the smallest discriminating input and the observed failure. Include
frequency or denominator only when measured. Give the intended audience a
portable setup. Include dependency or wheel installation, a durable source
link, and the run command. Use a public repository for a public report. Do not
expose private repository paths. Do not use a host-specific virtual environment
or session-local script as the reproduction instruction. For TLX, install the
`fbtriton` PyPI wheel. The `fbtriton` distribution provides the `triton`
import. Pin it only when the compiler version affects the case.>

```bash
<create environment and install pinned dependencies or wheels>
<fetch or select the durable reproducer>
<exact run command>
```

| Variant | Changed variable | Result | Evidence |
|---|---|---|---|
| Original | <value> | <exit status, verdict, count, or metric> | <artifact> |
| Treatment/control | <value> | <exit status, verdict, count, or metric> | <artifact> |

## Evidence summary

| Claim | State | Producing tool | Evidence | Boundary |
|---|---|---|---|---|
| <symptom> | observed | <tool> | <decisive line/artifact> | <scope> |
| <runtime manifestation> | <state> | <tool> | <trace/log/ISA> | <scope> |
| <root cause> | <state> | <tool or source inspection> | <A/B or mapping> | <remaining uncertainty> |

<Explain the shortest auditable chain from source or IR to ISA/runtime
evidence. Keep observation, bounded inference, causal findings, and unresolved
questions visibly separate.>

For each decisive CUTracer result, show:

```text
<raw CUTracer output> -> <bounded inference> -> <changed next action>
```

## How CUTracer helped

<State exactly what CUTracer isolated, observed, confirmed, falsified, or
bounded. Attribute profiler, debugger, sanitizer, compiler, and source results
to those tools. Keep this section focused on decisive contributions. Place
each material capture, coverage, or generalization limit beside the evidence
that it constrains. Mention a non-contribution here only when readers could
otherwise misattribute a decisive result to CUTracer.>

## Resolution and validation

<Describe the fix, workaround, control, or open follow-up. Do not call a
workaround a fix.>

| Gate | Original | Treatment | Acceptance criterion |
|---|---:|---:|---|
| Reproducer | <result> | <result> | <criterion> |
| Independent oracle | <result> | <result> | <criterion> |
| CUTracer | <result> | <result> | <criterion or not rerun> |
| Regression/performance | <result> | <result> | <criterion or not measured> |
```

## Optional sections

### Blind investigation and later validation

Use this when the investigator did not know the historical diagnosis or fix.
Keep the phases separate. Do not rewrite the blind phase with hindsight.

```markdown
#### Blind phase

| Input or milestone | Evidence or decision available at that time |
|---|---|
| Starting prompt and artifacts | <actual inputs; no later root cause or fix> |
| Initial hypotheses | <ranked hypotheses> |
| Tool result | <verdict and completeness> |
| Blind conclusion | <conclusion supported before reveal> |

#### Post-investigation validation

<Introduce the historical diagnosis or fix here. Compare it with the blind
conclusion, and state what the earlier CUTracer evidence did or did not prove.>
```

### Human report comparison

Use this after the blind or historical validation when a human-authored report
exists. Keep the human report hidden until the blind conclusion is frozen.
Compare evidence and investigation coverage. Do not use the human report to
rewrite the blind phase. If the case study is public and the human report is
private, omit the private link and private-only details. Use a public source
when one exists.

```markdown
## Comparison with the human investigation

Human report: <durable public link and title, or title only when private>

| Topic | Agent with CUTracer | Human report | Gap or next test |
|---|---|---|---|
| Root cause | <agent conclusion and evidence> | <human conclusion and evidence> | <agreement or contradiction> |
| Hazard or failure coverage | <what was proven> | <additional cases or mechanisms> | <what the agent missed> |
| Finding unit | <per event/warp/CTA/site> | <human aggregation> | <reconcile counts> |
| Independent validation | <A/B and tools> | <other tools or sibling cases> | <test that would close the gap> |

<State what the agent found that the human report did not record. Then state
the material evidence, mechanism, or scope that the agent missed or left
unproven. Distinguish a correct narrow conclusion from an incorrect one.>
```

### Detector or analysis design

Include this for a new detector. Define preconditions, aggregation key,
sufficient-condition rule, verdict semantics, completeness gates, positive
controls, legal counterexamples, and implementation diffs.

### Performance

Include this only when measured. State hardware, warmup, iterations, statistic,
variance, and the compared artifact identities.

### Sibling survey

Include this when near-copy kernels, shapes, or alternate dispatch paths may
share the defect. Name what was inspected and what remains open.

### Tool friction

Record only reusable friction. Give the observable symptom, investigation
impact, workaround, and concrete improvement.

## Editing rules

- Use only facts from inspected inputs and produced artifacts.
- Link the relevant `examples/` or `tests/` code beside the status when it
  exists. Omit the field when no such code exists.
- Prefer one evidence table over repeated prose.
- Keep raw logs in artifacts. Quote only decisive lines.
- Put each material limitation beside the claim it constrains.
- Link each decisive artifact where the report uses it. Do not add an artifact
  dump or generic limitations section.
- Use "in this capture" or "for this reproducer" for bounded results.
- State why a zero finding is meaningful or vacuous.
- Make reproduction setup portable. Include pinned dependencies or wheels.
- For TLX, use the `fbtriton` PyPI wheel. Pin it only when the compiler version
  affects the case.
- Link public reports to public source. Do not expose private paths.
- Keep the CUTracer contribution section focused on what changed the diagnosis.
- When a human report exists, compare it only after the blind conclusion is
  frozen. Name material omissions and the evidence needed to close them.
- Do not link or quote a private human report from a public case study.
- Preserve literal commands, logs, identifiers, and persisted verdict strings.
- Write approximately 80% toward ASD-STE100. Use short, direct sentences and
  one main claim per sentence. Preserve technical terms and bounded qualifiers.
- Use `T+N min` only when recorded evidence supports the elapsed time.
- Close unresolved reports as unresolved. Do not manufacture a success story.
