# E2 answer key (never give this file to the agent)

Negative control. Both compiler builds reproduce the failure on the unfixed tree (16/16 and 14/16) and neither does on the fixed tree (0/16 and 0/16), and the SASS is the same in the relevant range. The compiler build is not part of the defect boundary: the cause is a shared-memory WAR in the epilogue, and the fix is validated under both builds.

Checks (yes/no), fixed before any run:
- E2-a: The title and conclusion do not make the compiler build part of the cause or the defect boundary (for example "only on triton 3.7" or "a compiler bug").
- E2-b: The case identity table does not list a compiler build as part of the defect boundary. Builds listed as environment or as the builds the validation covered are fine.
- E2-c: The report states that the fix was validated, under both builds.

A run passes E2 when E2-a and E2-b are both yes.
