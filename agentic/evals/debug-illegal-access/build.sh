#!/bin/bash
# Build the blind binaries for one GPU architecture.
# Usage: build.sh <arch> <outdir>    e.g. build.sh sm_90a out-h100, build.sh sm_100a out-b200
# Strips comments (the answer key), builds with the per-case flags below, and strips symbols.
set -euo pipefail
ARCH=$1; OUT=$(realpath -m "$2"); CASES=$(dirname "$0")/cases
mkdir -p "$OUT"; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
build() {  # name source flags...
  local name=$1 src=$2; shift 2
  python3 - "$CASES/$src" "$T/app.cu" <<'PY'
import re, sys
out = [re.sub(r"\s*//.*$", "", l.rstrip("\n")) for l in open(sys.argv[1]) if not l.lstrip().startswith("//")]
open(sys.argv[2], "w").write("\n".join(out) + "\n")
PY
  (cd "$T" && nvcc -std=c++17 -O2 "$@" -o "$OUT/$name" app.cu)
  strip "$OUT/$name"
}
LI="-lineinfo -arch=$ARCH"            # debug-style: SASS + PTX + line info
REL="-gencode arch=compute_${ARCH#sm_},code=$ARCH"   # release-style: SASS only, no PTX, no line info
build S1 S1_offsets.cu   $LI
build S2 S2_highhalf.cu  -arch=$ARCH -lcuda
build S3 S3_overflow.cu  $LI
build S4 S4_nthlaunch.cu $LI
build S5 S5_ptrtable.cu  -arch=$ARCH
build S6 S6_misarg.cu    $LI
build V1 V1_spmv_emptyrow.cu   $REL
build V2 V2_scatter_history.cu $REL -lcuda
ls -la "$OUT"
