#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# NVBit injected objects must not depend on CUDA device assertion support.

set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <injected-object>..." >&2
  exit 2
fi

for object in "$@"; do
  # Capture the output first so a cuobjdump failure cannot look like a clean
  # symbol table, and grep cannot terminate cuobjdump early via a pipe.
  symbols=$(cuobjdump --dump-elf-symbols "$object")
  if grep -qw '__assertfail' <<< "$symbols"; then
    echo "❌ $object depends on __assertfail; NVBit cannot load injected device assertions." >&2
    echo "   CUDA 13.0 requires -DNDEBUG when compiling injected functions." >&2
    exit 1
  fi
  echo "✅ No device assertion dependency in $object"
done
