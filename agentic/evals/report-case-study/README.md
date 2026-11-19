# Evals for report-case-study

Short investigation packets for testing the channel-title rules in the [`report-case-study`](../../skills/report-case-study/SKILL.md) skill: the Workplace post title prefix, the matching first tag, and a conclusion that keeps the packet's scope.

Each case directory holds a `packet.md`, which is the only file the agent under test may see, and an `answer.md` with the expected title prefix, first tag, and conclusion wording. The packets are fictional and need no GPU.

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
