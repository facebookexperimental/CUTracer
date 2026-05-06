#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# H100 runner setup. Installs CUDA 13.0 toolkit + system deps + cutracer
# Python package on top of the meta-triton uv venv that the linux-gcp-h100
# runner ships. Keep this separate from .ci/setup.sh — that one is conda-
# based and targets the T4 (4-core-ubuntu-gpu-t4) runner.
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
# 1. apt: zstd CLI + libzstd-dev
#    run_tests.sh's trace verification uses `zstd -d` to inspect
#    compressed traces, and CUTracer compression code links libzstd.
#    libzstd.so.1 is preinstalled but no -dev headers and no CLI.
# ─────────────────────────────────────────────────────────────────────
echo "::group::apt: zstd"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends zstd libzstd-dev
echo "::endgroup::"

# ─────────────────────────────────────────────────────────────────────
# 2. apt: cuda-toolkit-13-0
#    The NVIDIA cuda apt repo is not preconfigured on this k8s pod
#    image, so add the keyring deb (also writes
#    /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list) every job.
#    Match the cu130 PyTorch runtime in the meta-triton venv.
# ─────────────────────────────────────────────────────────────────────
echo "::group::apt: cuda-toolkit-13-0"
KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
curl -fsSL -o /tmp/cuda-keyring.deb "$KEYRING_URL"
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13-0
/usr/local/cuda-13.0/bin/nvcc --version
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
if [ -n "$GITHUB_ENV" ]; then
    {
        echo "CUDA_HOME=/usr/local/cuda-13.0"
        echo "LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    } >> "$GITHUB_ENV"
fi
if [ -n "$GITHUB_PATH" ]; then
    echo "/usr/local/cuda-13.0/bin" >> "$GITHUB_PATH"
fi

echo "✅ setup-h100.sh complete"
