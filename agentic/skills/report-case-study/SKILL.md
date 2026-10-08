---
name: report-case-study
description: Write an evidence-backed CUTracer GPU debugging case report after an investigation. Use when asked to document, publish, or summarize a CUTracer case for humans and future debugging agents without overstating incomplete evidence or CUTracer's contribution.
license: MIT
compatibility: Requires the investigation's reproducer, environment, tool outputs, and validation artifacts.
---

# Report a CUTracer case study

## Invocation boundary

Register this skill when the repository or agent session is prepared. Invoke
it only at the investigation's completion boundary, when all of the following
are true:

- the reproducer is stable;
- decisive artifacts are available;
- the conclusion is causal, bounded, or explicitly unresolved;
- the fix, workaround, or control was tested when applicable; and
- the user requested documentation, or documentation was part of the original
  task.

Do not invoke this skill during the initial diagnosis. Continue with the
relevant CUTracer debugging workflow until the completion conditions are met.
Publishing is a separate action and requires its own authorization.

Read [references/case-study-template.md](references/case-study-template.md)
before drafting. Use its required core. Add optional sections only when they
help prove the conclusion.

## 1. Inventory evidence before writing

Build a private evidence ledger for each headline claim:

| State | Meaning |
|---|---|
| observed | Directly present in a run, trace, log, IR, disassembly, or source file |
| inferred | Best explanation consistent with the evidence, but not independently confirmed |
| causal | A matched intervention changes the outcome, or complete evidence proves the mechanism |
| unresolved | Required evidence is missing, incomplete, unsupported, or ambiguous |

For every observation, record:

- the exact artifact or command;
- revision, kernel identity, specialization, GPU, and toolchain;
- capture mode and completeness;
- result, unit, denominator, and exit status when relevant;
- a durable link or a session-local path, labeled correctly.

Keep current verification separate from historical documentation. A current
run can confirm the symptom on the current host. It does not retroactively
verify a historical driver, compiler, timing, or failure rate.

### Preserve blind investigations

When the investigation was blind, preserve two phases:

1. Record the starting prompt, initial artifacts, hypotheses, tool results,
   and conclusion without exposing the known root cause or fix.
2. Reveal the historical diagnosis or fix only in a separate
   post-investigation validation phase.

Do not rewrite the first phase as if the investigator knew the answer. Do not
use later evidence to strengthen an earlier CUTracer verdict.

When a human-authored report exists, keep it hidden during the blind phase.
Compare it with the frozen agent conclusion only after the post-investigation
reveal. Record agreements, contradictions, and evidence that the agent missed.
Do not import a human claim into the agent evidence ledger.

## 2. Choose the narrowest defensible conclusion

Select one report status:

- **Confirmed:** Evidence establishes the mechanism, and a discriminating
  intervention changes the result.
- **Bounded:** Evidence localizes or excludes part of the failure, but does not
  establish the full mechanism.
- **Unresolved:** The reproducer is valid, but required evidence is missing or
  contradictory.

Separate these claims:

1. the user-visible symptom;
2. the runtime manifestation;
3. the root cause;
4. the detector or analyzer verdict;
5. the resolution: fix, workaround, control, or open follow-up.

A workaround is not a fix. A schedule change that hides a failure is not proof
of a synchronization repair. A passing numerical run is not proof of memory or
synchronization safety.

## 3. Attribute every result to its producing tool

Name the tool that produced each fact. A CUTracer wrapper around Compute
Sanitizer does not make a sanitizer finding a CUTracer finding. A profiler that
selects a dispatch, a debugger that captures raw events, and a CUTracer analyzer
that rejects an incomplete trace have different contributions.

Call the case a CUTracer success only when a specific CUTracer artifact
materially localized the defect, changed the diagnosis, confirmed or falsified
a hypothesis, or verified the resolution. Merely running CUTracer is not a
success claim. Show the chain explicitly:

```text
raw CUTracer output -> bounded inference -> changed next action
```

Use exact verbs:

- **isolated** the kernel or dispatch;
- **observed** an address, value, or instruction;
- **confirmed** a mechanism with complete evidence;
- **falsified** a hypothesis;
- **bounded** the conclusion by rejecting incomplete evidence.

Keep the CUTracer contribution section focused on what changed the
investigation. Put a non-contribution there only when readers could otherwise
misattribute a decisive result to CUTracer. Put each material capture,
coverage, or generalization limit beside the claim that it constrains.

## 4. Require matched validation for causal claims

A strong A/B changes one relevant variable while holding the input, GPU,
toolchain, launch configuration, and oracle constant. Record both commands and
both outcomes.

For a fix, prefer a regression that fails without the change and passes with
it. If only a workaround or negative control exists, report that narrower
result. Do not turn a setup failure, missing dependency, or failed capture into
evidence about the kernel.

Negative evidence is meaningful only with a stated boundary. Never translate
zero findings from an incomplete capture into "no race" or "safe." Include all
completeness reasons and the detector's aggregation unit.

## 5. Draft at the smallest useful depth

Every report needs:

1. a title in this form: `CUTracer investigates <exact symptom> in
   <kernel/operator/component> [on <platform>]: <confirmed cause or narrowest
   proven boundary>`;
2. a status plus concise tags for the tool, subsystem, failure class, platform,
   and kernel or operator terms that agents will search;
3. a short human TL;DR in this order: symptom, conclusion, outcome, and CUTracer
   contribution;
4. a compact case identity table with the durable reproducer, affected
   component or configuration, recognition signature, and validated action;
5. portable reproduction instructions;
6. an evidence summary from observation to conclusion;
7. resolution and matched validation.

When a human-authored report exists, add a comparison section after the blind
or historical validation. Compare the investigation boundaries and evidence,
not writing quality. State:

- which conclusions agree;
- which finding units differ, such as one static site versus one finding per
  CTA;
- what the agent found that the human report did not record;
- what the agent missed or left unproven; and
- which additional test, detector, sibling case, or tool would have closed
  each material gap.

Respect the target report's audience. A public case study must not link a
private human report or copy private-only paths and measurements. Name the
report without a link, use a public equivalent when one exists, and include
only facts that are public or cleared for publication.

Keep omissions distinct from errors. A narrower agent conclusion can be
correct and incomplete. Do not call unrequested performance work or a broader
sibling survey a diagnostic failure, but record the missing coverage when it
changes how readers use the result.

Do not add a generic scope, limitations, or artifacts section. Put each useful
limit beside the affected claim. Link each decisive artifact where the report
uses it. Omit limitations and artifact inventories that do not change how a
reader interprets or reproduces the result.

Keep the case identity table small. Add a platform only when it selects the
affected path. Add software versions or artifact checksums only when they are
part of the defect boundary or are needed to distinguish compared artifacts.
Do not repeat details that the symptom, evidence, or validation sections make
clear.

Reproduction instructions must work for the intended audience. Include the
dependency or wheel installation commands and the run command. Link the
reproducer to a durable location that the audience can access. Use a public
repository for a public report. Do not expose private repository paths in a
public report. Do not present a host-specific virtual environment or
session-local script path as the reproduction command.

For a TLX case, install the `fbtriton` wheel from PyPI. The distribution name
is `fbtriton`, and it provides the `triton` Python import. Install it after
PyTorch if PyTorch installs another Triton distribution. Pin `fbtriton` only
when the compiler version is part of the defect boundary or reproducibility
requires it. Otherwise install the current wheel and record the tested version
as environment evidence. Do not tell the audience to build fbtriton from
source or install the upstream `triton` PyPI package unless the case requires a
specific compiler revision that is unavailable as an `fbtriton` wheel.

Add a timeline only when sequence or elapsed time teaches something. When
artifact timestamps are available, use a table with rounded `T+N min` values.
Place the timeline immediately after the TL;DR so readers can scan the result
and investigation sequence together.
State the total investigation time above the table. Tie that duration to the
last milestone that the report claims.
List the coding-agent client and model beside the total time. If the runtime
does not expose the model in the agent response, inspect the session trajectory
or runtime metadata for its exact model identifier. Use the narrowest available
model family and state that limitation only when the trajectory also omits the
exact model. Do not infer an exact model.

If a subagent performed the investigation, inspect that subagent's trajectory
and record the coding agent and model shown there. Do not infer them from the
parent agent or the later report-writing session. If the trajectory does not
contain an exact model, state that it is unavailable.

Mark when the immediate root cause was localized. Mark later confirmation of
the defect boundary separately. Do not invent elapsed times.

Write approximately 80% toward ASD-STE100. Use short, direct sentences. State
one main claim per sentence. Use consistent terms. Make cause, contrast,
condition, and sequence explicit. Avoid vague pronouns, idioms, and
promotional language. Preserve GPU terms, identifiers, commands, literal logs,
and bounded qualifiers. Do not claim formal ASD-STE100 compliance.

## 6. Audit before delivery

Verify all of the following:

- The title, TL;DR, body, and status use the same conclusion and scope.
- Each case-specific fact comes from an inspected input or produced artifact.
- Current and historical evidence are labeled separately.
- A human report, when available, is compared only after the blind conclusion
  and does not retroactively change the agent evidence.
- The human comparison identifies material agent omissions and the test or
  evidence needed to close them.
- A public report does not link or quote a private human report.
- Root cause, runtime manifestation, detector verdict, and resolution are not
  conflated.
- Every number has a unit, a denominator when applicable, and an evidence
  location.
- Original and treatment artifacts have distinct identities.
- Negative evidence states its completeness and coverage boundary.
- Tool attribution distinguishes CUTracer from profilers, debuggers,
  sanitizers, compilers, and source inspection.
- The title contains the exact symptom, affected component, platform when
  useful, and the narrowest proven conclusion.
- The status links the relevant `examples/` or `tests/` code when it exists.
- The TL;DR uses symptom, conclusion, outcome, and CUTracer contribution order.
- Commands include portable setup and are runnable or explicitly illustrative.
- TLX setup uses the PyPI `fbtriton` wheel. It pins the wheel only when the
  compiler version is relevant to the defect boundary.
- Public reports contain no private repository paths or source links.
- Session-local paths are not presented as durable links.
- The CUTracer contribution section emphasizes decisive contributions.
- Useful caveats appear beside the evidence they constrain.
- The report does not use a generic limitations section or artifact dump.
- A timed investigation marks root-cause localization and later boundary
  confirmation separately.
- Placeholders and unsupported optional sections are removed.

If a headline claim fails this audit, narrow it or mark it unresolved. Do not
fill the gap with a plausible narrative.
