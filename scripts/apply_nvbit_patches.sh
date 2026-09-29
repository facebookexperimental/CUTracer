#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Apply the producer publication fix missing from the NVBit 1.8 release.
# Usage: bash scripts/apply_nvbit_patches.sh [path/to/nvbit]

set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [path/to/nvbit]" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NVBIT_DIR=${1:-third_party/nvbit}
PATCH_FILE="$SCRIPT_DIR/patches/nvbit-channel-producer-fence.patch"

if [ ! -f "$NVBIT_DIR/core/utils/channel.hpp" ]; then
  echo "Error: NVBit channel.hpp not found in $NVBIT_DIR. Run ./install_third_party.sh first." >&2
  exit 1
fi
if ! command -v patch >/dev/null 2>&1; then
  echo "Error: Install the 'patch' utility to apply the NVBit channel fix." >&2
  exit 1
fi

PATCH_ARGS=(--batch --fuzz=0 --no-backup-if-mismatch -p1 -d "$NVBIT_DIR" -i "$PATCH_FILE")

# Recognize the exact fix, not just a fence elsewhere in ChannelDev::flush().
if patch "${PATCH_ARGS[@]}" --dry-run --reverse --force >/dev/null 2>&1; then
  echo "NVBit channel producer publication fix is already applied."
  exit 0
fi

# Check first so an incompatible upstream channel is left unchanged.
if ! patch "${PATCH_ARGS[@]}" --dry-run --forward; then
  echo "Error: NVBit channel does not match the producer publication patch. Review the upstream channel before building CUTracer." >&2
  exit 1
fi

patch "${PATCH_ARGS[@]}" --forward
echo "Applied NVBit channel producer publication fix."
