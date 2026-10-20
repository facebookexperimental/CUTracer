#!/usr/bin/env python3
"""Count s_barrier instructions in each qr_async FMHA forward kernel of a CK example binary.

Usage: count_barriers.py <tile_example_fmha_fwd> [kernel-name filter ...]

Extracts the gfx950 code objects with roc-obj, disassembles them, and prints the
s_barrier count of every BlockFmhaPipelineQRKSVSAsync kernel whose mangled name
contains all filters (default: the d96 bf16 tile 128x128x32x128x32x96).
"""

import collections
import glob
import os
import re
import subprocess
import sys
import tempfile

binary = os.path.abspath(sys.argv[1])
filters = sys.argv[2:] or ["Li128ELi128ELi32ELi128ELi32ELi96", "DF16b"]
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(
        ["/opt/rocm/bin/roc-obj", "-d", binary],
        cwd=tmp,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    counts = {}
    for path in glob.glob(os.path.join(tmp, "*.s")):
        kernel = None
        for line in open(path):
            m = re.match(r"^[0-9a-f]+ <(.+)>:", line)
            if m:
                kernel = m.group(1)
                counts.setdefault(kernel, 0)
            elif kernel and re.match(r"^\s+s_barrier\b", line):
                counts[kernel] += 1
selected = {
    k: v
    for k, v in counts.items()
    if "BlockFmhaPipelineQRKSVSAsync" in k and all(f in k for f in filters)
}
print(
    f"{len(selected)} kernels; s_barrier count -> number of kernels:",
    dict(sorted(collections.Counter(selected.values()).items())),
)
