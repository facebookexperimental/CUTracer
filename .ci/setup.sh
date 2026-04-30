#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Setup script for CUTracer CI environment
# This script sets up CUDA and system dependencies for building and testing CUTracer

set -e

# Default values
export CUDA_VERSION=${CUDA_VERSION:-"12.8"}
export DEBUG=${DEBUG:-"0"}
export CONDA_ENV=${CONDA_ENV:-"cutracer"}

echo "Setting up CUTracer environment..."
echo "CUDA_VERSION: $CUDA_VERSION"
echo "DEBUG: $DEBUG"
echo "CONDA_ENV: $CONDA_ENV"

# Setup conda + CUDA + cuDNN + PyTorch nightly via the base script
# (formerly curled from tritonparse; now vendored locally — see
# .ci/setup-base.sh header for the migration story).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "⬇️ Setting up base dependencies via $SCRIPT_DIR/setup-base.sh..."
if bash "$SCRIPT_DIR/setup-base.sh"; then
    echo "✅ Base dependencies setup complete."
else
    echo "❌ Base dependencies setup failed."
    exit 1
fi

# Install zstd library for compression support
echo "📦 Installing zstd library..."
sudo apt-get update && sudo apt-get install -y libzstd-dev
echo "✅ zstd installed"

echo "Setup completed successfully!"
