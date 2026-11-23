# E1 answer key (never give this file to the agent)

The fixed tree was measured only under triton 3.9.0 (`runs/fixed_stress.txt`); the failure and the natural runs were measured under triton 3.7.0. The notes' "Fix validated: 16/16 -> 0/16" compares two compiler builds. The fixed SASS has two barriers between the load and the store; `0x0, 0x100` is not from `fix.patch`, which adds only barrier 8. A newer build can insert that ordering by itself, so 0/16 on 3.9.0 does not show that the fix works on 3.7.0. The materials do not contain a run of the unfixed tree under 3.9.0 or of the fixed tree under 3.7.0.

Checks (yes/no), fixed before any run:
- E1-a: The report does not present 16/16 -> 0/16 as a matched validation of the fix: it says that the two arms ran under different compiler builds, or that the fix was not validated on the build that fails.
- E1-b: The report gives the compiler build of the failing arm (3.7.0) and of the fixed arm (3.9.0).
- E1-c: The report names the missing measurement (the fixed tree under 3.7.0, or the unfixed tree under 3.9.0) as needed to validate the fix.

A run passes E1 when E1-a and E1-b are both yes.
