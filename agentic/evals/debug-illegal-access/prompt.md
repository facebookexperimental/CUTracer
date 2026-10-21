You are debugging a GPU program on this machine. Use only GPU {GPU}: prefix every GPU command with `CUDA_VISIBLE_DEVICES={GPU}`. Put long runs under `timeout`.

Program: {BINARY} (prebuilt; the source code is not available). Symptom report from its owner: "{SYMPTOM}"

Task: find the root cause (which instruction or code location, and why), with evidence, and propose a fix.

Environment: the `cutracer` CLI is installed (`cutracer --help`), and its cutracer.so is at {CUTRACER_SO}.
{CONDITION}

Write scratch files only under {WORKDIR}. Do not read the program's source or any evals directory. Never leave a hung process on the GPU.

Return as your final message: (1) root cause and evidence, (2) proposed fix, (3) every command you ran that mattered and whether it worked, (4) how many commands you ran in total and which step was the hardest{SKILL_FEEDBACK}.

<!--
Fill in:
- {CONDITION}, with the skill: "Use the CUTracer agent skills in {SKILLS_DIR}: start by reading investigate-kernel/SKILL.md and follow the skills it routes you to."
- {CONDITION}, without the skill: "Do not read any agent skill files (anything named SKILL.md or under an agentic/ directory)."
- Append to {CONDITION} when testing without a sanitizer: "compute-sanitizer and cuda-gdb are not available for this task: do not run them, directly or through `cutracer compute-sanitizer`."
- {SKILL_FEEDBACK}, with the skill: ", and (5) every place where a skill was wrong, unclear, or missing something you needed, quoting the skill text". Without the skill: empty.
-->
