#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Apply CUTracer's patches in scripts/patches/ to an NVBit release tree.
# Usage: bash scripts/apply_nvbit_patches.sh [path/to/nvbit]

set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [path/to/nvbit]" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NVBIT_DIR=${1:-third_party/nvbit}

if [ ! -f "$NVBIT_DIR/core/utils/channel.hpp" ]; then
  echo "Error: NVBit channel.hpp not found in $NVBIT_DIR. Run ./install_third_party.sh first." >&2
  exit 1
fi
if ! command -v patch >/dev/null 2>&1; then
  echo "Error: Install the 'patch' utility to apply CUTracer's NVBit patches." >&2
  exit 1
fi

shopt -s nullglob
for PATCH_FILE in "$SCRIPT_DIR"/patches/*.patch; do
  PATCH_NAME=$(basename "$PATCH_FILE")
  PATCH_ARGS=(--batch --fuzz=0 --no-backup-if-mismatch -p1 -d "$NVBIT_DIR" -i "$PATCH_FILE")

  if patch "${PATCH_ARGS[@]}" --dry-run --reverse --force >/dev/null 2>&1; then
    echo "NVBit patch $PATCH_NAME is already applied."
    continue
  fi

  # Check first so an incompatible upstream file is left unchanged.
  if ! patch "${PATCH_ARGS[@]}" --dry-run --forward; then
    echo "Error: NVBit does not match $PATCH_NAME. Review the upstream change before building CUTracer." >&2
    exit 1
  fi

  patch "${PATCH_ARGS[@]}" --forward
  echo "Applied NVBit patch $PATCH_NAME."
done
