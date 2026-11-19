# Packet C

Authorship: a coding agent ran the investigation and drafted this report.

Status: confirmed.

Symptom: the `gather_index` kernel reads the wrong table row on H100 when two thread blocks update the index table at the same time.

Conclusion: the kernel reads the index table without checking the version counter, so a reader can take a row that a concurrent update has half overwritten. A matched run with the version check reads the correct row in 20 of 20 runs. The resolution also added a CUTracer check that flags this read pattern in other kernels, and that check is new in this change.

CUTracer contribution: the trace observed the unguarded read, and the same run confirmed the version check.
