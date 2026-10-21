# Evals for debug-hanging-kernel

Planted-hang CUDA programs for testing the [`debug-hanging-kernel`](../../skills/debug-hanging-kernel/SKILL.md) skill.
Each program hung in 10 of 10 runs on B200 and H100 (on A100, the six that build for sm_80), and the comment at the top of each source file states the bug: it is the answer key.
Each source also builds a fixed program with `-DFIXED`, which exits 0 and checks its own output.

## Cases

| Case | Symptom the owner reports | Build |
|---|---|---|
| S1 | A handoff kernel never returns | line info, PTX |
| S2 | A staging kernel never finishes on one batch | SASS and PTX, no line info |
| S3 | A two-phase update never finishes on a 200M-element array; small arrays work | line info, PTX |
| S4 | A row-sum kernel hangs on one batch of tiles | line info, PTX |
| S5 | A tile sum hangs on one dataset (sm_90 and newer only) | SASS and PTX, no line info |
| S6 | A step loop hangs in its fourth step | line info, PTX |
| V1 | A single-pass scan of 4M ints never finishes; 1M works | release: SASS only |

`symptoms.json` holds the exact symptom text for each case.
S3 allocates about 800 MB of GPU memory.

## Build and check

```bash
./build.sh sm_100a out-b200              # or sm_90a (H100), sm_80 (A100, which skips S5)
FIXED=1 ./build.sh sm_100a out-b200-fixed
./reliability.sh out-b200 out-b200-fixed 10 0,1,2,3,4,5,6 > reliability.txt
```

The script strips the comments, so the binaries carry no answer key, and strips symbols.
`reliability.sh` runs every case N times without CUTracer, buggy and fixed, and prints how many buggy runs hung and how many fixed runs passed.

## Run a blind trial

Give an agent one binary, its symptom, and `prompt.md`, filled in.
Run each case with and without the skill on the same agent.

Keep the agent away from this directory: it holds the answer keys.
Copy each binary to its own directory under a unique name; an agent that stops its program with `pkill -x <name>` otherwise stops other runs too.

## Grade

A run counts as found when it names the mechanism and the runtime facts in the case's top comment: which CTAs wait, and the launch and input value that cause the hang.
Grade the fix separately: it must end the hang and keep all of the kernel's work, as the fixed program does.
Leave out runs that break the prompt's rules in a way that gives them information or disturbs other runs, and record the rest; the runs below counted several that started the program once without `timeout`.

## Results so far

Blind runs with the OSS CUTracer build on B200, without cuda-gdb or compute-sanitizer, graded by one grader that did not know each run's condition.
The runs used the previous revision of the skill; the current one adds fixes from those runs and was not rerun blind.

```
case             agent            with skill  without
S1-S6, V1        Claude Opus      7/7         -
S4, S5, S6, V1   Codex            4/4         4/4
S4, S5, S6, V1   another agent    3/4         3/4
```

Runs with the skill also answer an extra question (where the skill fell short), so their times run longer. In this comparison, both agents found as many causes without the skill as with it: the CLI's help names the right capture flags, and the SASS shows each mechanism.
The previous revision's commands were also checked by hand on H100 and A100.
