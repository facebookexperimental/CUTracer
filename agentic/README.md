# CUTracer agent skills

This directory contains reusable CUTracer workflows for coding agents. Each
skill is a self-contained package:

```text
agentic/skills/<skill-name>/
├── SKILL.md
└── references/
```

Each `SKILL.md` uses `name` and `description` frontmatter. Skill bodies describe
decisions and evidence requirements without depending on a specific agent,
IDE, repository host, or tool-call syntax.

New skills come with an eval under `agentic/evals/<skill-name>/` that shows
the skill helps and keeps it working when the skill changes; existing skills
gain one when they are next revised.

## Available skills

| Skill | Use it when |
|---|---|
| [`investigate-kernel`](skills/investigate-kernel/SKILL.md) | A GPU kernel misbehaves and no workflow has been chosen yet. It routes to the matching debugging skill. |
| [`debug-hanging-kernel`](skills/debug-hanging-kernel/SKILL.md) | A kernel hangs or never returns (NVIDIA). |
| [`debug-illegal-access`](skills/debug-illegal-access/SKILL.md) | A kernel faults: CUDA error 700, an illegal or misaligned address (NVIDIA). |
| [`report-case-study`](skills/report-case-study/SKILL.md) | A CUTracer investigation is complete and needs an auditable, reusable report. |
| [`audit-memory-model`](skills/audit-memory-model/SKILL.md) | A GPU kernel (NVIDIA Hopper or Blackwell, AMD CDNA3 or CDNA4) must be checked against the vendor memory model without a failing run, for example when the reproducer passes or a fence or wait may be missing. |

Registration makes a skill discoverable. It does not start the workflow. Each
skill defines its own invocation boundary. For example, `report-case-study`
runs only after an investigation reaches a stable conclusion.

## Agent discovery

`agentic/skills/` is the canonical catalog. Before acting on a matching task,
an agent must install or register the skill with its own supported loader:

1. Select the matching package from the table above.
2. Determine the current client's supported skill directory or registration
   mechanism. Do not assume that another client's path is valid.
3. Prefer a symlink from that location to the canonical package. If the client
   accepts arbitrary search paths, register `agentic/skills/` directly.
4. Verify that `SKILL.md` and its relative `references/` links resolve through
   the installed location.
5. Read the complete `SKILL.md` and the references it selects before acting.

Claude-, Codex-, Muse-, and other repository-aware agents must perform this
installation themselves when their policy and filesystem permissions allow it.
If persistent installation is not allowed, load the canonical `SKILL.md`
directly for the current task and tell the user that installation was skipped.

Do not copy or fork skill content per client. A copied package can drift from
the canonical instructions.

## Adding a skill

1. Create `agentic/skills/<name>/SKILL.md` with `name` and `description`
   frontmatter.
2. Keep the main file short. Put detailed templates and procedures in
   `references/`.
3. Avoid client-specific tools, hidden repository paths, and organization-only
   terminology.
4. Document any capability that requires a specific client without embedding
   that client's installation path in the canonical skill.
5. Validate every relative link and test installation through at least one
   client-supported skill loader.
6. Add an eval in `agentic/evals/<name>/` in the same change (see Evals).

## Evals

An eval is a set of cases with a known answer, a way to run an agent on them
blind, and a rule for grading the result. A typical eval directory holds:

```text
agentic/evals/<skill-name>/
├── README.md     cases, how to build and run them, grading rule, results
├── prompt.md     the blind prompt, with placeholders
└── cases/        inputs with the answer key, for example planted-bug programs
```

- Run each case with and without the skill on the same agent, and report
  both. A skill that does not change the result for any agent tested is not
  yet shown to help.
- Fix the grading rule before looking at the runs, and leave out runs that
  break the prompt's rules.
- Keep the answer keys away from the agent under test: give it built inputs,
  not this repository, and never link a skill to its eval.
- When a change to the skill is meant to fix a failure, rerun the case that
  showed it.
