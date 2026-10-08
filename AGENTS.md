# Agent guidance

## Reusable skills

Canonical agent skills live under `agentic/skills/`.

Before a task, inspect [agentic/README.md](agentic/README.md). When a listed
skill matches the request, install or register the canonical skill directory
with the current client's supported skill loader. Prefer a symlink over a
copied package. Then read its complete `SKILL.md` and the references it selects
before acting.

If client policy or filesystem permissions prevent installation, read the
canonical `SKILL.md` directly for the current task and report that persistent
installation was not completed.

Treat `agentic/skills/` as the source of truth. Do not create an independent
client-specific copy.
