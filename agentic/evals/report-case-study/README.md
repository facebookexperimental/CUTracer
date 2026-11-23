# Evals for report-case-study

Short investigation packets for testing the channel-title rules in the [`report-case-study`](../../skills/report-case-study/SKILL.md) skill: the Workplace post title prefix, the matching first tag, and a conclusion that keeps the packet's scope. Cases E1 and E2 test another rule of the skill, with their own prompt: see [Compiler-build cases](#compiler-build-cases-e1-and-e2).

Each of cases A-C holds a `packet.md`, which is the only file the agent under test may see, and an `answer.md` with the expected title prefix, first tag, and conclusion wording. The packets are fictional and need no GPU.

## Cases

| Case | What it tests |
|---|---|
| A | An agent-drafted, confirmed case gets `[Bot🤖][Case Report]` and `#casereport` |
| B | A human-written, bounded case gets `[Case Report]` with no Bot marker, and the title does not upgrade a bounded conclusion |
| C | A case whose resolution adds a CUTracer feature is still a Case Report, because the reader's main takeaway is the root cause |

## Run a blind trial

Give an agent one `packet.md` and `prompt.md`, filled in. Run each case with and without the skill on the same agent, and report both.

Keep the agent away from this directory: the `answer.md` files hold the answer keys. Copy each `packet.md` to its own directory before the run.

## Grade

A run passes a case only when all three checks hold:

1. The post title starts with the exact prefix in `answer.md`, with exactly one type prefix.
2. The first tag is the hashtag in `answer.md`.
3. The title keeps the packet's symptom and conclusion scope. A bounded packet stays bounded, and no case gains a stronger claim than the packet states.

Fix this grading rule before looking at the runs. Leave out runs that break the prompt's rules.

## Results so far

No blind runs yet. This eval was added with the channel-title revision of the skill; run it before citing the skill as measured on these cases.

## Compiler-build cases: E1 and E2

Two blind case bundles for the skill's rule that a report reads each arm's compiler build from that arm's own logs: an A/B across two builds does not validate a fix, and a build that does not change the outcome stays out of the defect boundary. Each bundle is a finished investigation (notes, a fix, command outputs with their environment, disassembly excerpts); its `answer.md` holds the answer key and the grading checks.

### Cases

| Case | Bundle | Trap |
|---|---|---|
| E1 | A warp-specialized attention backward on B200; a delay search finds a large dK/dV change, and a barrier fixes it. | The fixed tree was measured only under another compiler build. Only each run's environment block shows it, and the investigator's notes call the fix validated. |
| E2 | A persistent GEMM epilogue on H100 with a shared-memory staging hazard. | Negative control: both compiler builds reproduce, the fix holds on both, and the build is not part of the boundary. |

### Run a blind trial

Copy `cases/<case>/bundle/` outside this repository, and give the agent `compiler-build-prompt.md` with `{BUNDLE}` and `{OUTPUT}` filled in. `{SKILL_CLAUSE}` is empty for a run without the skill; for a run with it, copy the skill directory at the revision under test outside the repository and use `` and the skill in `<dir>` (read its SKILL.md and the template it references, and follow them)``. Never give the agent `answer.md` or this directory.

### Grade

Use the checks in each case's `answer.md`, which were written before any run. E1 passes when E1-a and E1-b hold, E2 when E2-a and E2-b hold; E1-c and E2-c are reported beside them.

### Results so far

Runs on 2026-11-18 with Claude Haiku as the agent under test, 2 runs per cell, graded by the author of the revision. "current" is the skill before the compiler-build revision, "revised" the skill with it.

```
case  pass (extra check)   no skill    current     revised
E1    a+b (c)              2/2 (2/2)   2/2 (2/2)   2/2 (2/2)
E2    a+b (c)              2/2 (2/2)   2/2 (2/2)   2/2 (2/2)
```

No cell differs: this agent already catches E1's confound and keeps E2's build out of the boundary without the skill, so the revision is not yet shown to help on these cases. One E1 run without the skill passes as graded but still calls the fix "validated by the stress oracle" in its summary, with the build difference only among its caveats. One revised E1 run also passes but calls the build part of the defect boundary, which E1 cannot show (the unfixed tree never ran on 3.9.0); the skill now says the build is part of the boundary only when the same source was measured under both builds, and the two revised E1 runs repeated after that change (counted in the table) say the boundary is untested. The visible difference is the template's compiler-build column: all four revised reports give each variant's build in the variant table, against one of the four current reports; none of the reports written without the skill used the template's variant table. Builds in the case identity and wheel pins appear in every cell.

Two earlier rounds were discarded because the bundles were inconsistent: a disassembly PC disagreed with a stress log, and the patch file was named `fix.diff`, which the repository ignores. A case that separates the cells needs a subtler confound, for example a build that differs only in a wheel's local version, or a build recorded in a file the notes never cite.
