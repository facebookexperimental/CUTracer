# Packet A

Authorship: a coding agent ran the investigation and drafted this report. A human reviewed the draft and fixed two typos.

Status: confirmed.

Symptom: the `tile_reduce` kernel returns a wrong sum on H100 whenever the input has more than 4096 rows. Small inputs pass.

Conclusion: warp 0 writes the partial sum to shared memory, and warp 1 reads it before the barrier that orders the two accesses. A CUTracer trace shows both accesses with no barrier between them. Adding the barrier makes 20 of 20 previously failing runs pass, with the input, GPU, and toolchain unchanged.

CUTracer contribution: the trace observed both shared-memory accesses and confirmed that no barrier orders them.
