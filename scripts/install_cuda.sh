#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Usage: CUDA_INSTALL_PREFIX=/home/yhao/opt ./install_cuda.sh 12.8
#        TARGETARCH=aarch64 CUDA_INSTALL_PREFIX=/opt ./install_cuda.sh 12.8
#
# Notice: Part of this script is synced with https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cuda.sh
#
# Architecture support:
#   - x86_64: Standard x86-64 systems
#   - aarch64 (sbsa): ARM64 server platforms including NVIDIA GB200 / Grace Hopper
#
# Architecture is auto-detected from `uname -m` but can be overridden via:
#   TARGETARCH=<arch>  Set target architecture (x86_64|aarch64|arm64)
#                      Useful for cross-compilation or container builds targeting
#                      GB200 (aarch64) from an x86_64 host.
#
set -ex

# Debug settings - log all execution steps
# Use user-specific directory to avoid permission issues in multi-user environments
DEBUG_LOG_DIR="/tmp/${USER}"
mkdir -p "${DEBUG_LOG_DIR}"
exec 5>"${DEBUG_LOG_DIR}/cuda_install_debug.txt"
BASH_XTRACEFD="5"
PS4='${LINENO}: '

# Record script start time for duration calculation
SCRIPT_START_TIME=$(date +%s)

echo "🚀 ===== CUDA Installation Script Started ====="

# Basic settings
CUDA_INSTALL_PREFIX=${CUDA_INSTALL_PREFIX:-$HOME/opt}
CUDA_INSTALL_PREFIX=${CUDA_INSTALL_PREFIX%/}
CUDA_VERSION=${CUDA_VERSION:-12.8}
NVSHMEM_VERSION=${NVSHMEM_VERSION:-3.4.5}
INSTALL_NCCL=${INSTALL_NCCL:-1}
NCCL_VERSION=${NCCL_VERSION:-v2.29.7-1}

# Minimum required disk space in GB (can be overridden)
REQUIRED_DISK_SPACE_GB=${REQUIRED_DISK_SPACE_GB:-15}

echo "CUDA_INSTALL_PREFIX=${CUDA_INSTALL_PREFIX}"
echo "CUDA_VERSION=${CUDA_VERSION}"
echo "INSTALL_NCCL=${INSTALL_NCCL}"
echo "NCCL_VERSION=${NCCL_VERSION}"

# Version configuration using associative arrays
declare -A CUDA_FULL_VERSION=(
  ["12.6"]="12.6.3"
  ["12.8"]="12.8.1"
  ["12.9"]="12.9.1"
  ["13.0"]="13.0.2"
  ["13.2"]="13.2.1"
)

declare -A CUDA_RUNFILE=(
  ["12.6"]="cuda_12.6.3_560.35.05_linux"
  ["12.8"]="cuda_12.8.1_570.124.06_linux"
  ["12.9"]="cuda_12.9.1_575.57.08_linux"
  ["13.0"]="cuda_13.0.2_580.95.05_linux"
  ["13.2"]="cuda_13.2.1_595.58.03_linux"
)

declare -A CUDNN_VERSIONS=(
  ["12.6"]="9.10.2.21"
  ["12.8"]="9.23.1.3"
  ["12.9"]="9.23.1.3"
  ["13.0"]="9.23.1.3"
  ["13.2"]="9.23.1.3"
)

declare -A CUDA_MAJOR=(
  ["12.6"]="12"
  ["12.8"]="12"
  ["12.9"]="12"
  ["13.0"]="13"
  ["13.2"]="13"
)

# Create temporary directory
USER_TMPDIR="/tmp/${USER}/cuda_install"
mkdir -p "${USER_TMPDIR}"
export TMPDIR="${USER_TMPDIR}"

# Error handling function
function error_exit {
  echo "❌ ERROR: $1" >&2
  # Save error message to log file
  echo "$(date): ERROR - $1" >>"${USER_TMPDIR}/error.log"
  # Leave error marker file for debugging
  echo "$1" >"${USER_TMPDIR}/error_message"
  touch "${USER_TMPDIR}/error_occurred"
  exit 1
}

# Cleanup function for temporary directories
# Uses absolute paths based on USER_TMPDIR to ensure cleanup works regardless of current directory
function cleanup_temp_dirs {
  echo "Cleaning up temporary directories..."
  local base_dir
  base_dir=$(pwd)

  # Clean up directories in the current working directory (where script was started)
  [ -d "${base_dir}/tmp_cusparselt" ] && rm -rf "${base_dir}/tmp_cusparselt"
  [ -d "${base_dir}/tmp_cudnn" ] && rm -rf "${base_dir}/tmp_cudnn"
  [ -d "${base_dir}/nccl" ] && rm -rf "${base_dir}/nccl"
  [ -d "${base_dir}/tmp_nvshmem" ] && rm -rf "${base_dir}/tmp_nvshmem"

  # Also clean up in USER_TMPDIR if different from base_dir
  if [ "${USER_TMPDIR}" != "${base_dir}" ]; then
    [ -d "${USER_TMPDIR}/tmp_cusparselt" ] && rm -rf "${USER_TMPDIR}/tmp_cusparselt"
    [ -d "${USER_TMPDIR}/tmp_cudnn" ] && rm -rf "${USER_TMPDIR}/tmp_cudnn"
    [ -d "${USER_TMPDIR}/nccl" ] && rm -rf "${USER_TMPDIR}/nccl"
    [ -d "${USER_TMPDIR}/tmp_nvshmem" ] && rm -rf "${USER_TMPDIR}/tmp_nvshmem"
  fi

  touch "${USER_TMPDIR}/cleanup_completed"
}

# Cleanup function called on script exit (trap handler)
function cleanup_on_exit {
  local exit_code=$?
  echo "Script exiting with code: ${exit_code}"
  cleanup_temp_dirs
  if [ ${exit_code} -ne 0 ]; then
    echo "⚠️  Script exited with errors. Check ${USER_TMPDIR}/error.log for details."
  fi
}

# Set trap after functions are defined to avoid calling undefined functions
trap cleanup_on_exit EXIT
trap 'error_exit "Script interrupted"' INT TERM

# Detect architecture
# TARGETARCH can be set via environment variable to override auto-detection.
# On GB200 / Grace Hopper systems, uname -m returns 'aarch64' and the NVIDIA
# download paths use the 'sbsa' (Server Base System Architecture) identifier.
export TARGETARCH=${TARGETARCH:-$(uname -m)}

# resolve_arch: normalize architecture names and set ARCH_PATH
function resolve_arch {
  if [ "${TARGETARCH}" = 'aarch64' ] || [ "${TARGETARCH}" = 'arm64' ]; then
    ARCH_PATH='sbsa'
  elif [ "${TARGETARCH}" = 'x86_64' ]; then
    ARCH_PATH='x86_64'
  else
    error_exit "Unsupported architecture: ${TARGETARCH}. Supported: x86_64, aarch64, arm64"
  fi
}

resolve_arch
echo "🏗️  Architecture: ${TARGETARCH} (download path: ${ARCH_PATH})"

# Check if command exists
function command_exists {
  command -v "$1" &>/dev/null
}

# Check dependencies
function check_dependencies {
  echo "Checking dependencies..."
  local missing_deps=()

  for cmd in wget git make; do
    if ! command_exists "$cmd"; then
      missing_deps+=("$cmd")
    fi
  done

  if [ ${#missing_deps[@]} -gt 0 ]; then
    error_exit "Missing required dependencies: ${missing_deps[*]}"
  fi

  echo "✅ All dependency checks passed"
}

# Check disk space (CUDA toolkit requires approximately 10-15GB)
function check_disk_space {
  local required_gb=${1:-15}
  local target_dir="${CUDA_INSTALL_PREFIX}"

  echo "Checking disk space (need ${required_gb}GB)..."

  local available_kb
  available_kb=$(df "${target_dir}" | awk 'NR==2 {print $4}')
  local available_gb=$((available_kb / 1024 / 1024))

  if [ "${available_gb}" -lt "${required_gb}" ]; then
    error_exit "Insufficient disk space: need ${required_gb}GB, only ${available_gb}GB available at ${target_dir}"
  fi

  echo "✅ Disk space check passed (${available_gb}GB available)"
}

# Check network connectivity to NVIDIA download server
function check_network {
  echo "Checking network connectivity to NVIDIA servers..."

  local test_url="https://developer.nvidia.com/cuda/toolkit"
  local http_code

  http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "${test_url}" 2>/dev/null || echo "000")

  if [[ "${http_code}" == "000" ]]; then
    error_exit "Cannot connect to NVIDIA download server (${test_url}). Please check your network connection."
  elif [[ "${http_code}" =~ ^(2|3)[0-9][0-9]$ ]]; then
    echo "✅ Network connectivity check passed (HTTP ${http_code})"
  else
    echo "⚠️  Warning: NVIDIA server returned HTTP ${http_code}, continuing anyway..."
  fi
}

# Get disk space information for a given path
# Returns: total_gb used_gb available_gb use_percent
function get_disk_info {
  local target_path="${1:-/}"
  df -BG "${target_path}" | awk 'NR==2 {
    gsub(/G/, "", $2); gsub(/G/, "", $3); gsub(/G/, "", $4); gsub(/%/, "", $5);
    print $2, $3, $4, $5
  }'
}

# Print disk space summary
function print_disk_summary {
  local label="$1"
  local target_path="${2:-/}"

  local disk_info
  disk_info=$(get_disk_info "${target_path}")
  local total_gb
  local used_gb
  local available_gb
  local use_percent
  total_gb=$(echo "${disk_info}" | awk '{print $1}')
  used_gb=$(echo "${disk_info}" | awk '{print $2}')
  available_gb=$(echo "${disk_info}" | awk '{print $3}')
  use_percent=$(echo "${disk_info}" | awk '{print $4}')

  echo "💾 ${label}:"
  echo "   Total: ${total_gb}GB | Used: ${used_gb}GB | Available: ${available_gb}GB | Usage: ${use_percent}%"
}

# Check if file or directory exists
function check_exists {
  if [ ! -e "$1" ]; then
    error_exit "Path does not exist: $1"
  fi
}

# Check directory
check_exists "${CUDA_INSTALL_PREFIX}"

# Check dependencies
check_dependencies

# Check disk space
check_disk_space "${REQUIRED_DISK_SPACE_GB}"

# Check network connectivity
check_network

# Validate that a remote URL is reachable (HTTP HEAD check)
# Returns 0 if the URL returns HTTP 2xx/3xx, 1 otherwise.
function validate_download_url {
  local url=$1
  local http_code

  http_code=$(curl -s -o /dev/null -w "%{http_code}" --head --connect-timeout 10 "${url}" 2>/dev/null || echo "000")

  if [[ "${http_code}" =~ ^(2|3)[0-9][0-9]$ ]]; then
    return 0
  else
    return 1
  fi
}

# Real CUDA installation function
function install_cuda {
  local version=$1
  local runfile=$2
  local major_minor=${version%.*}

  echo "Installing CUDA ${version}..."
  rm -rf "${CUDA_INSTALL_PREFIX}/cuda-${major_minor}" "${CUDA_INSTALL_PREFIX}/cuda"

  # Append the _sbsa suffix for aarch64 platforms (GB200, Grace Hopper, etc.)
  if [[ ${ARCH_PATH} == 'sbsa' ]]; then
    runfile="${runfile}_sbsa"
  fi
  runfile="${runfile}.run"

  local download_url="https://developer.download.nvidia.com/compute/cuda/${version}/local_installers/${runfile}"

  # Pre-validate the download URL to provide a clear error message before
  # attempting a large download, especially useful for aarch64/sbsa where
  # some CUDA versions may not have runfiles available.
  echo "Validating CUDA download URL for ${ARCH_PATH}..."
  if ! validate_download_url "${download_url}"; then
    error_exit "CUDA runfile not available at: ${download_url}
  This may indicate that CUDA ${version} does not have a runfile for ${ARCH_PATH}.
  If running on GB200/aarch64, verify the CUDA version supports sbsa."
  fi
  echo "✅ Download URL validated"

  echo "Downloading CUDA installation file: ${runfile}"
  # Use -c for resume support, -t 3 for retry, -q for quiet mode
  if ! wget -c -t 3 -q "${download_url}" -O "${runfile}"; then
    error_exit "CUDA installation file download failed: ${runfile}"
  fi

  echo "CUDA installation file download complete, preparing to install..."
  chmod +x "${runfile}"

  echo "Executing CUDA installation script..."
  if ! ./"${runfile}" --toolkit --silent --toolkitpath="${CUDA_INSTALL_PREFIX}/cuda-${major_minor}"; then
    echo "CUDA installation failed, keeping installation file for troubleshooting"
    error_exit "CUDA ${version} installation failed"
  fi

  # Create installation complete marker
  touch "${USER_TMPDIR}/cuda_${version}_install_complete"

  echo "CUDA installation complete, creating symbolic link..."
  rm -f "${CUDA_INSTALL_PREFIX}/cuda" && ln -s "${CUDA_INSTALL_PREFIX}/cuda-${major_minor}" "${CUDA_INSTALL_PREFIX}/cuda"

  echo "Removing installation file..."
  rm -f "${runfile}"

  echo "CUDA ${version} installation completed"
  return 0
}

# Real cuDNN installation function
function install_cudnn {
  local cuda_major_version=$1
  local cudnn_version=$2

  echo "Installing cuDNN ${cudnn_version} for CUDA ${cuda_major_version}..."

  # Create temporary directory and enter it using pushd for consistent style
  mkdir -p tmp_cudnn
  pushd tmp_cudnn || error_exit "Failed to enter cuDNN temporary directory"

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  local filepath="cudnn-linux-${ARCH_PATH}-${cudnn_version}_cuda${cuda_major_version}-archive"
  echo "Downloading cuDNN: ${filepath}.tar.xz"
  # Use -c for resume support, -t 3 for retry
  if ! wget -c -t 3 -q "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-${ARCH_PATH}/${filepath}.tar.xz"; then
    popd
    rm -rf tmp_cudnn
    error_exit "cuDNN download failed: ${filepath}.tar.xz"
  fi

  echo "cuDNN download complete, preparing to extract..."
  if ! tar xf "${filepath}.tar.xz"; then
    popd
    rm -rf tmp_cudnn
    error_exit "cuDNN extraction failed: ${filepath}.tar.xz"
  fi

  echo "cuDNN extraction complete, installing to CUDA directory..."
  cp -a "${filepath}/include/"* "${CUDA_INSTALL_PREFIX}/cuda/include/"
  cp -a "${filepath}/lib/"* "${CUDA_INSTALL_PREFIX}/cuda/lib64/"

  echo "cuDNN installation complete, cleaning up temporary files..."
  popd
  rm -rf tmp_cudnn

  # Create cuDNN installation complete marker
  touch "${USER_TMPDIR}/cudnn_${cudnn_version}_installed"

  echo "cuDNN ${cudnn_version} installation completed"
  return 0
}

# Real NCCL installation function
function install_nccl {
  if [ "$INSTALL_NCCL" -eq 0 ]; then
    echo "NCCL installation skipped as per INSTALL_NCCL=${INSTALL_NCCL}"
    return 0
  fi

  echo "Installing NCCL for CUDA ${CUDA_VERSION}..."

  # Use the NCCL version for CUDA 12.6 due to sm50 support.
  # Synced with PyTorch .ci/docker/common/install_nccl.sh
  if [[ ${CUDA_VERSION:0:4} == "12.6" ]]; then
    NCCL_VERSION="v2.29.3-1"
  fi

  if [[ -z "${NCCL_VERSION}" ]]; then
    error_exit "NCCL_VERSION is empty"
  fi

  echo "Using NCCL version: ${NCCL_VERSION}"
  echo "${NCCL_VERSION}" >"${USER_TMPDIR}/nccl_version.txt"

  # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
  # Follow build: https://github.com/NVIDIA/nccl/tree/master?tab=readme-ov-file#build
  echo "Cloning NCCL repository..."
  if ! git clone -b "${NCCL_VERSION}" --depth 1 https://github.com/NVIDIA/nccl.git; then
    error_exit "NCCL repository clone failed"
  fi

  echo "Starting NCCL compilation..."
  pushd nccl || error_exit "Failed to enter NCCL directory"

  if ! make -j src.build CUDA_HOME="${CUDA_INSTALL_PREFIX}/cuda"; then
    popd
    rm -rf nccl
    error_exit "NCCL compilation failed"
  fi

  echo "NCCL compilation complete, installing to CUDA directory..."
  cp -a build/include/* "${CUDA_INSTALL_PREFIX}/cuda/include/"
  cp -a build/lib/* "${CUDA_INSTALL_PREFIX}/cuda/lib64/"

  popd
  rm -rf nccl

  # Note: ldconfig is called once at the end of install_cuda_version
  # No need to call it here

  # Create NCCL installation complete marker
  touch "${USER_TMPDIR}/nccl_installed"

  echo "NCCL installation completed"
  return 0
}

# Real cuSparseLt installation function
function install_cusparselt {
  echo "Installing cuSparseLt for CUDA ${CUDA_VERSION}..."
  mkdir -p tmp_cusparselt
  pushd tmp_cusparselt || error_exit "Failed to create cuSparseLt temporary directory"

  local cusparselt_version
  local arch_path=${ARCH_PATH}
  local cuda_major_version=${CUDA_VERSION%%.*}

  # Determine cuSparseLt version based on CUDA version
  # Synced with PyTorch .ci/docker/common/install_cusparselt.sh
  if [[ ${CUDA_VERSION} == "12.6" ]] || [[ ${CUDA_VERSION} == "12.8" ]]; then
    cusparselt_version="0.7.1.0"
  else
    cusparselt_version="0.8.1.1"
  fi

  # Starting from 0.8.0, NVIDIA ships separate archives per CUDA major version
  local CUSPARSELT_NAME
  local cusparselt_minor
  cusparselt_minor=$(echo "${cusparselt_version}" | cut -d. -f2)
  if [[ "${cusparselt_minor}" -ge 8 ]]; then
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-${cusparselt_version}_cuda${cuda_major_version}-archive"
  else
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-${cusparselt_version}-archive"
  fi

  echo "${cusparselt_version}" >"${USER_TMPDIR}/cusparselt_version.txt"

  echo "Downloading cuSparseLt: ${CUSPARSELT_NAME}.tar.xz"
  # Use -c for resume support, -t 3 for retry
  if ! wget -c -t 3 -q "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz"; then
    popd
    rm -rf tmp_cusparselt
    error_exit "cuSparseLt download failed"
  fi

  echo "cuSparseLt download complete, preparing to extract..."
  if ! tar xf "${CUSPARSELT_NAME}.tar.xz"; then
    popd
    rm -rf tmp_cusparselt
    error_exit "cuSparseLt extraction failed"
  fi

  echo "cuSparseLt extraction complete, installing to CUDA directory..."
  cp -a "${CUSPARSELT_NAME}/include/"* "${CUDA_INSTALL_PREFIX}/cuda/include/"
  cp -a "${CUSPARSELT_NAME}/lib/"* "${CUDA_INSTALL_PREFIX}/cuda/lib64/"

  popd
  rm -rf tmp_cusparselt

  touch "${USER_TMPDIR}/cusparselt_installed"

  echo "cuSparseLt installation completed"
  return 0
}

# nvSHMEM installation function
function install_nvshmem {
  local cuda_major_version=$1 # e.g. "12"
  local nvshmem_version=${NVSHMEM_VERSION}
  local arch_path=${ARCH_PATH}

  local tmpdir="tmp_nvshmem"
  mkdir -p "${tmpdir}"
  pushd "${tmpdir}" || error_exit "Failed to enter nvSHMEM temporary directory"

  # nvSHMEM license: https://docs.nvidia.com/nvshmem/api/sla.html
  local filename="libnvshmem-linux-${arch_path}-${nvshmem_version}_cuda${cuda_major_version}-archive"
  local suffix=".tar.xz"
  local url="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-${arch_path}/${filename}${suffix}"

  echo "Downloading nvSHMEM: ${filename}${suffix}"
  # Use -c for resume support, -t 3 for retry
  if ! wget -c -t 3 -q "${url}"; then
    popd
    rm -rf "${tmpdir}"
    error_exit "nvSHMEM download failed: ${filename}${suffix}"
  fi

  echo "Extracting nvSHMEM..."
  if ! tar xf "${filename}${suffix}"; then
    popd
    rm -rf "${tmpdir}"
    error_exit "nvSHMEM extraction failed: ${filename}${suffix}"
  fi

  echo "Installing nvSHMEM to CUDA directory..."
  cp -a "${filename}/include/"* "${CUDA_INSTALL_PREFIX}/cuda/include/"
  cp -a "${filename}/lib/"* "${CUDA_INSTALL_PREFIX}/cuda/lib64/"

  popd
  rm -rf "${tmpdir}"

  # Create nvshmem installation complete marker
  touch "${USER_TMPDIR}/nvshmem_${nvshmem_version}_installed"

  echo "nvSHMEM ${nvshmem_version} for CUDA ${cuda_major_version} (${arch_path}) installed."
  return 0
}

# Generic CUDA version installation function
# Uses version configuration from associative arrays defined at the top
function install_cuda_version {
  local version=$1

  # Get configuration for this version
  local cuda_full=${CUDA_FULL_VERSION[$version]}
  local cudnn_ver=${CUDNN_VERSIONS[$version]}
  local cuda_major=${CUDA_MAJOR[$version]}

  local runfile=${CUDA_RUNFILE[$version]}

  if [[ -z "$cuda_full" ]]; then
    error_exit "No configuration found for CUDA version: $version"
  fi

  # Export CUDNN_VERSION for summary display
  export INSTALLED_CUDNN_VERSION="${cudnn_ver}"

  echo "Starting installation for CUDA ${version}..."
  echo "  Full version: ${cuda_full}"
  echo "  Runfile: ${runfile}"
  echo "  cuDNN version: ${cudnn_ver}"
  echo "  CUDA major: ${cuda_major}"

  # All installation functions use consistent error handling:
  # - Functions return 0 on success, non-zero on failure
  # - Functions call error_exit internally for detailed error messages
  # - The || error_exit pattern provides a fallback if the function returns non-zero without calling error_exit

  echo "📦 STEP 1: Installing CUDA toolkit..."
  if ! install_cuda "${cuda_full}" "${runfile}"; then
    error_exit "CUDA ${cuda_full} toolkit installation failed"
  fi

  echo "🧠 STEP 2: Installing cuDNN..."
  if ! install_cudnn "${cuda_major}" "${cudnn_ver}"; then
    error_exit "cuDNN installation failed"
  fi

  echo "🔗 STEP 3: Installing NCCL..."
  if ! install_nccl; then
    error_exit "NCCL installation failed"
  fi

  echo "⚡ STEP 4: Installing cuSparseLt..."
  if ! install_cusparselt; then
    error_exit "cuSparseLt installation failed"
  fi

  echo "💾 STEP 5: Installing nvSHMEM..."
  if ! install_nvshmem "${cuda_major}"; then
    error_exit "nvSHMEM installation failed"
  fi

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  else
    echo "⚠️  Note: Running as non-root user, ldconfig was not executed."
    echo "   You may need to set LD_LIBRARY_PATH to use the installed libraries."
  fi

  echo "✅ CUDA ${version} installation completed"
  return 0
}



# Main execution logic
echo "🔧 ===== Parsing command line arguments ====="
# Generate VALID_VERSIONS from the configuration arrays to keep them in sync
VALID_VERSIONS=("${!CUDA_FULL_VERSION[@]}")
# Sort the versions for consistent ordering
mapfile -t VALID_VERSIONS < <(printf '%s\n' "${VALID_VERSIONS[@]}" | sort -V)

# Parse positional CUDA version argument
if [ $# -gt 0 ]; then
  if [[ " ${VALID_VERSIONS[*]} " =~ " $1 " ]]; then
    CUDA_VERSION=$1
    echo "Setting CUDA version to: $CUDA_VERSION"
  else
    error_exit "Invalid CUDA version: $1. Must be one of: ${VALID_VERSIONS[*]}"
  fi
fi

# Note: Version validation is already done in the while loop above
# No need for redundant validation here
echo "Using CUDA version: $CUDA_VERSION"

# Perform cleanup before installation
cleanup_temp_dirs

# Perform installation
echo "⚙️  ===== Starting installation of CUDA ${CUDA_VERSION} ====="
echo "Calling install_cuda_version function with version: ${CUDA_VERSION}"

# Use set +e to temporarily disable error exit, for capturing errors and logging
set +e
install_cuda_version "${CUDA_VERSION}"
INSTALL_RESULT=$?
set -e

if [ $INSTALL_RESULT -ne 0 ]; then
  error_exit "Installation failed, exit code: $INSTALL_RESULT"
fi



# Final cleanup
cleanup_temp_dirs

# Version verification
echo "🔍 ===== Verifying CUDA installation ====="
if [ -f "${CUDA_INSTALL_PREFIX}/cuda/bin/nvcc" ]; then
  echo "✅ nvcc found - checking version:"
  echo "📋 $( "${CUDA_INSTALL_PREFIX}/cuda/bin/nvcc" --version | head -1 )"
  echo ""
else
  echo "⚠️  Warning: nvcc not found at ${CUDA_INSTALL_PREFIX}/cuda/bin/nvcc"
fi

# Check nvidia-smi from system PATH (it's part of the driver, not toolkit)
if command -v nvidia-smi &>/dev/null; then
  echo "✅ nvidia-smi found in system PATH - checking GPU status:"
  echo "📋 $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1)"
  echo ""
else
  echo "ℹ️  Note: nvidia-smi not found in PATH (GPU driver may not be installed)"
fi

echo "🎉 ===== Script execution completed ====="
touch "${USER_TMPDIR}/script_completed_successfully"
# Calculate and display installation duration
SCRIPT_END_TIME=$(date +%s)
INSTALL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
INSTALL_MINUTES=$((INSTALL_DURATION / 60))
INSTALL_SECONDS=$((INSTALL_DURATION % 60))

echo "✅ CUDA ${CUDA_VERSION} has been successfully installed to ${CUDA_INSTALL_PREFIX}"
echo "⏱️  Total installation time: ${INSTALL_MINUTES}m ${INSTALL_SECONDS}s"
echo "📊 ========================================="
echo " 📋 CUDA & Related Libraries Installation Summary"
echo "📊 ========================================="
echo "  Architecture: ${TARGETARCH} (${ARCH_PATH})"
echo "  CUDA        : ${CUDA_VERSION}"
echo "  cuDNN       : ${INSTALLED_CUDNN_VERSION:-(not available)}"
if [ "$INSTALL_NCCL" -eq 1 ]; then
  if [ -f "${USER_TMPDIR}/nccl_version.txt" ]; then
    NCCL_VERSION_PRINT=$(cat "${USER_TMPDIR}/nccl_version.txt")
    echo "  NCCL        : ${NCCL_VERSION_PRINT}"
  else
    echo "  NCCL        : (not found)"
  fi
else
  echo "  NCCL        : (skipped)"
fi
if [ -f "${USER_TMPDIR}/cusparselt_version.txt" ]; then
  CUSPARSELT_VERSION_PRINT=$(cat "${USER_TMPDIR}/cusparselt_version.txt")
  echo "  cuSparseLt  : ${CUSPARSELT_VERSION_PRINT}"
else
  echo "  cuSparseLt  : (not found)"
fi
echo "  nvSHMEM     : ${NVSHMEM_VERSION}"
echo "🔗 -----------------------------------------"
echo "📁  Install Path : ${CUDA_INSTALL_PREFIX}/cuda"
echo "💡  Usage:"
echo "    export PATH=${CUDA_INSTALL_PREFIX}/cuda/bin:\$PATH"
echo "    export LD_LIBRARY_PATH=${CUDA_INSTALL_PREFIX}/cuda/lib64:\$LD_LIBRARY_PATH"
echo "📊 ========================================="
