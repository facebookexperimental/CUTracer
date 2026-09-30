#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Setup for the uv-based H100 and B200 runners. CUDA_VERSION selects the
# toolkit (13.0 by default); INSTALL_CUDA_COMPAT=1 also installs the matching
# CUDA forward-compatibility libraries for PTX JIT on an older host driver.
# Keep this separate from the conda-based T4 setup in .ci/setup.sh.
#
# Preconditions: /workspace/setup_instance.sh has already been sourced in
# a previous step, so:
#   - VIRTUAL_ENV points at /workspace/uv_venvs/$CONDA_ENV
#   - $PATH contains /workspace/uv_venvs/$CONDA_ENV/bin and
#     /home/runner/.local/bin (uv lives there)
#   - $LD_LIBRARY_PATH points at the venv's nvidia/cu13/lib etc.
#
# Probed on 2026-05-06 (linux-gcp-h100, NVIDIA H100 80GB, driver 580.126.09):
#   - cuda-toolkit-13-0 install: ~75s, ~4.9 GB on /usr/local/cuda-13.0
#   - meta-triton venv already ships PyTorch 2.13.0.dev+cu130 +
#     Triton 3.6.0+fb.beta editable from /workspace/meta-triton, so we do
#     NOT reinstall those.
#   - The runner is an ephemeral k8s pod (containerd CRI), so apt installs
#     and /usr/local/cuda-13.0 do not persist across runs.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_VERSION="${CUDA_VERSION:-13.0}"
INSTALL_CUDA_COMPAT="${INSTALL_CUDA_COMPAT:-0}"
case "$CUDA_VERSION" in
    13.0|13.3|13.4) ;;
    *) echo "::error::Unsupported CUDA_VERSION: $CUDA_VERSION"; exit 1 ;;
esac
case "$INSTALL_CUDA_COMPAT" in
    0|1) ;;
    *) echo "::error::INSTALL_CUDA_COMPAT must be 0 or 1"; exit 1 ;;
esac
CUDA_ROOT="/usr/local/cuda-${CUDA_VERSION}"
CUDA_PACKAGES=("cuda-toolkit-${CUDA_VERSION//./-}")
if [ "$INSTALL_CUDA_COMPAT" = "1" ]; then
    CUDA_PACKAGES+=("cuda-compat-${CUDA_VERSION//./-}")
fi

# Sanity: SETUP_SCRIPT must have been sourced upstream.
if [ -z "$VIRTUAL_ENV" ]; then
    echo "::error::VIRTUAL_ENV not set. Source \$SETUP_SCRIPT in a previous step."
    exit 1
fi
if ! command -v uv >/dev/null; then
    echo "::error::uv not on PATH. /home/runner/.local/bin should be on PATH after sourcing SETUP_SCRIPT."
    exit 1
fi

echo "Active venv: $VIRTUAL_ENV"
echo "uv:          $(command -v uv) ($(uv --version))"
echo "python:      $(command -v python) ($(python -V 2>&1))"

# ─────────────────────────────────────────────────────────────────────
# 1. apt: zstd CLI + libzstd-dev + GoogleTest
#    run_tests.sh's trace verification uses `zstd -d` to inspect
#    compressed traces, and CUTracer compression code links libzstd.
#    libzstd.so.1 is preinstalled but no -dev headers and no CLI.
# ─────────────────────────────────────────────────────────────────────
echo "::group::apt: compression and unit-test dependencies"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends bc zstd libzstd-dev libgtest-dev
echo "::endgroup::"

# ─────────────────────────────────────────────────────────────────────
# 2. apt: the requested CUDA toolkit and optional compatibility libraries.
#    The NVIDIA cuda apt repo is not preconfigured on this k8s pod
#    image, so add the keyring deb (also writes
#    /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list) every job.
#    The runner-provided PyTorch build may target a different CUDA minor
#    version. Record its build version separately from the selected toolkit.
# ─────────────────────────────────────────────────────────────────────
echo "::group::apt: ${CUDA_PACKAGES[*]}"
KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
KEYRING_DEB="${TMPDIR:-/tmp}/cuda-keyring.deb"
curl -fsSL -o "$KEYRING_DEB" "$KEYRING_URL"
sudo dpkg -i "$KEYRING_DEB"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${CUDA_PACKAGES[@]}"
"$CUDA_ROOT/bin/nvcc" --version
echo "::endgroup::"

# ─────────────────────────────────────────────────────────────────────
# 3. cutracer Python package + pandas (used by run_tests.sh).
#    With SKIP_CONDA=1, run_tests.sh skips its own pip block, so we
#    have to install here.
# ─────────────────────────────────────────────────────────────────────
echo "::group::uv pip install"
uv pip install pandas
uv pip install -e "$PROJECT_ROOT/python"
echo "::endgroup::"

# ─────────────────────────────────────────────────────────────────────
# 4. Persist CUDA env to subsequent workflow steps.
#    GITHUB_ENV / GITHUB_PATH are GitHub Actions' magic files; lines
#    appended to them get exported to every later step in the same job.
#    No-op when run outside Actions (those vars are unset).
# ─────────────────────────────────────────────────────────────────────
CUDA_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [ "$INSTALL_CUDA_COMPAT" = "1" ]; then
    if [ ! -f "$CUDA_ROOT/compat/libcuda.so.1" ]; then
        echo "::error::CUDA compatibility library missing from $CUDA_ROOT/compat"
        exit 1
    fi
    CUDA_LIBRARY_PATH="$CUDA_ROOT/compat:$CUDA_LIBRARY_PATH"
fi
if [ -n "$GITHUB_ENV" ]; then
    {
        echo "CUDA_HOME=$CUDA_ROOT"
        echo "LD_LIBRARY_PATH=$CUDA_LIBRARY_PATH"
    } >> "$GITHUB_ENV"
fi
if [ -n "$GITHUB_PATH" ]; then
    echo "$CUDA_ROOT/bin" >> "$GITHUB_PATH"
fi

echo "✅ setup-gpu.sh complete (CUDA $CUDA_VERSION, compatibility libraries: $INSTALL_CUDA_COMPAT)"
