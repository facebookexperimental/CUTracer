# Evals for debug-illegal-access

Planted-fault CUDA programs for testing the [`debug-illegal-access`](../../skills/debug-illegal-access/SKILL.md) skill.
Each program faults on every run, and the comment at the top of each source file states the bug: it is the answer key.

## Cases

| Case | Symptom the owner reports | Build |
|---|---|---|
| S1 | A bucket sort fails with an illegal memory access | line info, PTX |
| S2 | A traversal fails after a metadata refresh | no line info |
| S3 | A row scale fails on a 40000 x 65536 matrix; smaller ones work | line info, PTX |
| S4 | A 12-layer forward pass fails at the end | line info, PTX |
| S5 | A batched axpy over 24 batches fails | no line info |
| S6 | A float4 sum fails with a misaligned address | line info, PTX |
| V1 | An SpMV over a 4M-row CSR matrix fails; smaller ones work | release: SASS only |
| V2 | A routing pipeline fails at the final download | release: SASS only |

`symptoms.json` holds the exact symptom text for each case.
S2, S3 and V1 need several GB of GPU memory; S3 allocates about 10.5 GB.

## Build

```bash
./build.sh sm_90a out-h100    # or sm_100a for B200
```

The script strips the comments, so the binaries carry no answer key, and strips symbols.

## Run a blind trial

Give an agent one binary, its symptom, and `prompt.md`, filled in.
Run each case with and without the skill, so the comparison is within one agent.

Keep the agent away from this directory: it holds the answer keys.
Copy the binaries somewhere else, and do not give the agent this repository unless its own instructions forbid reading `agentic/evals/`.

## Grade

A run counts as found when it names the faulting value and the code that produced it, as the top comment of the case states.
Grade the fix separately: it must remove the fault and keep all of the kernel's work.
Record runs that break the prompt's rules (for example, using compute-sanitizer when the prompt says it is unavailable) and leave them out of the counts.

## Results so far

Blind runs with the OSS CUTracer build on H100 and B200, without compute-sanitizer or cuda-gdb unless noted, graded by the skill's author against the answer keys.
Some runs used earlier drafts of the skill.

```
case                                agent         with skill  without
S1-S6                               Claude Opus   6/6         6/6
S1-S6, sanitizer and cuda-gdb on    Claude Opus   6/6         6/6
V1, V2                              Claude Opus   2/2         2/2
S5                                  Claude Haiku  3/3         0/4
S3                                  Claude Haiku  2/7         0/3
```

A strong agent finds every cause with or without the skill.
A smaller agent finds S5 with the skill and misnames the cause without it.
