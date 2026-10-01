#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Preserve the H100 entrypoint and its CUDA 13.0 default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/setup-gpu.sh" "$@"
