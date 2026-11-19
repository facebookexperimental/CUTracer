# Packet B

Authorship: a human engineer wrote this report. An agent is only formatting it for posting.

Status: bounded.

Symptom: the `scan_prefix` kernel hangs on B200 for one input shape. Other shapes complete.

Conclusion: the hang is in the second of the kernel's three phases. A CUTracer trace shows 0 progressing warps in that phase and excludes the first phase. The trace does not establish which barrier blocks, and no fix has been tested.

CUTracer contribution: the trace bounded the hang to the second phase and falsified a first-phase cause.
