#!/bin/bash
# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# See LICENSE file for details.
# A script to check or apply code formatting using clang-format.
# It automatically finds all relevant source files in the project.

# --- Go to script's directory ---
# This ensures that the script can be called from any location and that all
# subsequent paths are relative to the project root.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd -- "$SCRIPT_DIR" || exit

# --- Create the pinned formatter environment on first use ---
FORMAT_REQUIREMENTS="$SCRIPT_DIR/requirements-format.txt"
FORMAT_VENV="${CUTRACER_FORMAT_VENV:-$SCRIPT_DIR/.venv-format}"
FORMAT_PYTHON="${CUTRACER_FORMAT_PYTHON:-python3}"
FORMAT_REQUIREMENTS_ID=$(cksum "$FORMAT_REQUIREMENTS" | awk '{print $1 "-" $2}')
FORMAT_ENV_MARKER="$FORMAT_VENV/.requirements-$FORMAT_REQUIREMENTS_ID"

if [ ! -x "$FORMAT_VENV/bin/python" ] || [ ! -f "$FORMAT_ENV_MARKER" ]; then
  echo "🔧  Preparing formatter environment in $FORMAT_VENV..."
  if command -v uv &>/dev/null; then
    if [ ! -x "$FORMAT_VENV/bin/python" ]; then
      uv venv --python "$FORMAT_PYTHON" "$FORMAT_VENV" || exit 1
    fi
    uv pip install --python "$FORMAT_VENV/bin/python" --requirement "$FORMAT_REQUIREMENTS" || exit 1
  else
    if [ ! -x "$FORMAT_VENV/bin/python" ]; then
      "$FORMAT_PYTHON" -m venv "$FORMAT_VENV" || exit 1
    fi
    "$FORMAT_VENV/bin/python" -m pip install --requirement "$FORMAT_REQUIREMENTS" || exit 1
  fi
  touch "$FORMAT_ENV_MARKER"
fi

export PATH="$FORMAT_VENV/bin:$PATH"

# --- Check for dependencies ---
CLANG_FORMAT_VERSION="21.1.2"
RUFF_LINT_RULES="E4,E7,E9,F"

if ! command -v clang-format &>/dev/null; then
  echo "❌ Error: clang-format is not installed or not in your PATH." >&2
  echo "Please install clang-format v${CLANG_FORMAT_VERSION} (must match CI version):" >&2
  echo "  - Recommended: pip install clang-format==${CLANG_FORMAT_VERSION}" >&2
  echo "  - On Debian/Ubuntu: sudo apt install clang-format (⚠️ version may differ)" >&2
  echo "  - On Fedora/CentOS: sudo dnf install clang-tools-extra (⚠️ version may differ)" >&2
  echo "  - On macOS (Homebrew): brew install clang-format (⚠️ version may differ)" >&2
  exit 1
fi

# --- Configuration ---
# Define the directories to be processed.
# Add or remove directories as needed for your project.
DIRECTORIES_TO_SCAN=(src include tests tutorial .ci)

# Define the file extensions to be processed.
# Add or remove extensions as needed for your project.
FILE_PATTERNS=(-name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh")

# (Optional) Python formatting configuration can be supplied via pyproject.toml

# --- Helper Functions ---

# Function to process a single file.
# It checks if a file needs formatting, formats it if necessary,
# and prints the filename to stdout if it was changed.
# shellcheck disable=SC2317  # Invoked by xargs through an exported function.
format_and_report_changes() {
  file="$1"
  # Use diff to compare the original file with clang-format's output.
  # If they differ, format the file in-place and print its name.
  if ! diff -q "${file}" <(clang-format "${file}") >/dev/null; then
    clang-format -i "${file}" || return 1
    echo "${file}"
  fi
}
# Export the function so it's available to the subshells created by xargs.
export -f format_and_report_changes

# Python checks/format: ufmt (sorter + formatter from pyproject.toml) -> ruff linting
python_check() {
  local failed=0

  if command -v ufmt &>/dev/null; then
    if ! ufmt check .; then
      failed=1
    fi
  else
    echo "❌ ufmt not found (required for Python formatting)." >&2
    failed=1
  fi

  if command -v ruff &>/dev/null; then
    if ! ruff format --check .; then
      failed=1
    fi

    if ! ruff check --select "$RUFF_LINT_RULES" . --diff; then
      failed=1
    fi
  else
    echo "❌ ruff not found (required for Python linting)." >&2
    failed=1
  fi

  return $failed
}

python_format() {
  local failed=0

  if command -v ufmt &>/dev/null; then
    echo "🎨  Formatting Python code with ufmt..."
    if ! ufmt format .; then
      failed=1
    fi
  else
    echo "❌ ufmt not found (required for Python formatting)." >&2
    failed=1
  fi

  if command -v ruff &>/dev/null; then
    echo "🔧  Fixing Python linting issues with ruff..."
    if ! ruff check --select "$RUFF_LINT_RULES" . --fix; then
      failed=1
    fi
    if ! ruff format .; then
      failed=1
    fi
  else
    echo "❌ ruff not found (required for Python linting)." >&2
    failed=1
  fi

  return $failed
}

# Function to print usage instructions
usage() {
  echo "Usage: $0 {check|format}"
  echo "  check      Check for formatting issues without modifying files."
  echo "             Exits with an error code if issues are found."
  echo "  format     Apply formatting to files in-place."
  exit 1
}

# --- Main Script Logic ---

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
  echo "Error: No mode specified."
  usage
fi

MODE=$1

# Filter for directories that actually exist to prevent 'find' errors.
EXISTING_DIRS=()
for dir in "${DIRECTORIES_TO_SCAN[@]}"; do
  if [ -d "$dir" ]; then
    EXISTING_DIRS+=("$dir")
  fi
done

if [ ${#EXISTING_DIRS[@]} -eq 0 ]; then
  echo "⏩  No source directories found to process. Searched for: ${DIRECTORIES_TO_SCAN[*]}. Exiting."
  exit 0
fi

# We pipe the output of find directly to xargs.
# This avoids issues with storing null-delimited strings in shell variables.
# The -r flag for xargs prevents it from running the command if find returns no files.

case "$MODE" in
check)
  echo "🔎  Checking code formatting in: ${EXISTING_DIRS[*]}..."

  # Step 1: C/C++ check with --Werror to get a reliable exit code
  find "${EXISTING_DIRS[@]}" -type f \( "${FILE_PATTERNS[@]}" \) -print0 | xargs -0 -r clang-format --dry-run --Werror
  CXX_STATUS=$?

  # Step 2: Python check using ufmt or black/usort
  python_check
  PY_STATUS=$?

  if [ $CXX_STATUS -ne 0 ] || [ $PY_STATUS -ne 0 ]; then
    echo "----------------------------------------------------"
    echo "Please run './format.sh format' to fix them."
    exit 1
  fi

  echo "✅  All files are correctly formatted (or no files were found to check)."
  exit 0
  ;;

format)
  echo "🎨  Applying code formatting to: ${EXISTING_DIRS[*]}..."

  # Use xargs to run the formatting function in parallel.
  # -P 0 tells xargs to use as many processes as available CPU cores.
  # The output will be a list of files that were actually changed.
  CHANGED_FILES=$(find "${EXISTING_DIRS[@]}" -type f \( "${FILE_PATTERNS[@]}" \) -print0 | \
    xargs -0 -P 0 -I {} bash -c 'format_and_report_changes "{}"')
  CXX_STATUS=$?

  if [ -n "$CHANGED_FILES" ]; then
    echo "✨ Changed files:"
    # Use printf to format the list of changed files neatly.
    printf "  - %s\n" $CHANGED_FILES
  else
    echo "No files needed formatting."
  fi

  # Python formatting
  python_format
  PY_STATUS=$?

  if [ $CXX_STATUS -ne 0 ] || [ $PY_STATUS -ne 0 ]; then
    exit 1
  fi

  echo "✅  Formatting complete."
  ;;

*)
  echo "Error: Invalid mode '$MODE'."
  usage
  ;;
esac

exit 0
