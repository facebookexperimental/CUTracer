#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# Script to list all GPU processes with detailed process information

# Add to ~/.local/bin and make executable: chmod +x listgpu.sh
# ln -s `realpath listgpu.sh` ~/.local/bin/listgpu

# Check if nvidia-smi is available
if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi command not found. Are NVIDIA drivers installed?"
  exit 1
fi

echo "Listing all GPU processes..."
echo "============================================="

# Get all process IDs using GPUs
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

if [ -z "$GPU_PIDS" ]; then
  echo "No GPU processes found."
else
  # Print header
  printf "%-8s %-12s %-15s %-20s %-10s %-s\n" "PID" "USER" "COMMAND" "START_TIME" "CPU%" "FULL_COMMAND"
  printf "%-8s %-12s %-15s %-20s %-10s %-s\n" "----" "----" "-------" "----------" "----" "------------"

  # Process each GPU PID
  for PID in $GPU_PIDS; do
    # Get process information
    PROCESS_USER=$(ps -o user= -p $PID 2>/dev/null)
    PROCESS_COMMAND=$(ps -o comm= -p $PID 2>/dev/null)
    PROCESS_START=$(ps -o lstart= -p $PID 2>/dev/null)
    PROCESS_CPU=$(ps -o pcpu= -p $PID 2>/dev/null)
    PROCESS_FULL_CMD=$(ps -o cmd= -p $PID 2>/dev/null)

    # Check if process still exists
    if [ -z "$PROCESS_USER" ]; then
      printf "%-8s %-12s %-15s %-20s %-10s %-s\n" "$PID" "N/A" "N/A" "N/A" "N/A" "Process not found"
    else
      # Truncate long commands for display
      TRUNCATED_CMD=$(echo "$PROCESS_FULL_CMD" | cut -c1-50)
      if [ ${#PROCESS_FULL_CMD} -gt 50 ]; then
        TRUNCATED_CMD="${TRUNCATED_CMD}..."
      fi

      printf "%-8s %-12s %-15s %-20s %-10s %-s\n" "$PID" "$PROCESS_USER" "$PROCESS_COMMAND" "$PROCESS_START" "${PROCESS_CPU}%" "$TRUNCATED_CMD"
    fi
  done
fi

echo ""
echo "============================================="
echo "GPU Status Overview:"
echo "============================================="

# Display nvidia-smi output but only the GPU information part (exclude processes)
nvidia-smi | sed '/| Processes:/,$d'

# Only show memory usage and detailed process info if there are GPU processes
if [ -n "$GPU_PIDS" ]; then
  echo ""
  echo "============================================="
  echo "GPU Memory Usage:"
  echo "============================================="

  # Get GPU memory usage in a formatted way
  GPU_MEMORY_INFO=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)

  # Print header
  printf "%-5s %-40s %-12s %-12s %-10s\n" "GPU" "Name" "Used (MiB)" "Total (MiB)" "GPU (%)"
  printf "%-5s %-40s %-12s %-12s %-10s\n" "====" "========================================" "============" "============" "=========="

  # Process each GPU
  while IFS=',' read -r gpu_index gpu_name memory_used memory_total gpu_util; do
    # Clean up whitespace from fields
    gpu_index=$(echo "$gpu_index" | xargs)
    gpu_name=$(echo "$gpu_name" | xargs)
    memory_used=$(echo "$memory_used" | xargs)
    memory_total=$(echo "$memory_total" | xargs)
    gpu_util=$(echo "$gpu_util" | xargs)

    # Truncate GPU name if too long (limit to 39 characters)
    if [ ${#gpu_name} -gt 39 ]; then
      gpu_name="${gpu_name:0:36}..."
    fi

    printf "%-5s %-40s %-12s %-12s %-10s\n" "$gpu_index" "$gpu_name" "$memory_used" "$memory_total" "$gpu_util%"
  done <<<"$GPU_MEMORY_INFO"

  echo ""
  echo "============================================="
  echo "Detailed GPU Process Information:"
  echo "============================================="

  # Get GPU UUID to index mapping
  GPU_MAPPING=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)

  # Get compute apps with GPU UUID
  COMPUTE_APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits)

  if [ -n "$COMPUTE_APPS" ]; then
    # Print header with better alignment
    printf "%-6s %-8s %-s %12s\n" "GPU" "PID" "Process" "Memory Used"
    printf "%-6s %-8s %-s %12s\n" "===" "========" "=======" "============"

    while IFS=',' read -r gpu_uuid pid process_name memory_used; do
      # Find GPU index for this UUID
      gpu_index=$(echo "$GPU_MAPPING" | grep "$gpu_uuid" | cut -d',' -f1)

      # Clean up whitespace from fields
      gpu_index=$(echo "$gpu_index" | xargs)
      pid=$(echo "$pid" | xargs)
      process_name=$(echo "$process_name" | xargs)
      memory_used=$(echo "$memory_used" | xargs)

      # Get full command line for this process
      full_cmd=$(ps -o cmd= -p "$pid" 2>/dev/null)

      if [ -n "$full_cmd" ]; then
        # Extract the first argument (executable name) from the command line
        exec_path=$(echo "$full_cmd" | awk '{print $1}')
      else
        # Fallback to nvidia-smi process name if ps fails
        exec_path="$process_name"
        full_cmd="$process_name"
      fi

      # Print first line: GPU, PID, full executable path, memory (no truncation)
      printf "%-6s %-8s %-s %12s\n" "$gpu_index" "$pid" "$exec_path" "$memory_used"

      # Print second line: full command line (indented)
      printf "%15s└─ %s\n" "" "$full_cmd"
      printf "\n"
    done <<<"$COMPUTE_APPS"

    echo "============================================="
  else
    echo "No compute applications found."
    echo "============================================="
  fi
fi

# Optional: run CUDA sample vectorAdd to sanity-check default GPU availability
VECTOR_ADD_PATH="$HOME/cuda-samples/Samples/0_Introduction/vectorAdd/vectorAdd"
if [ -x "$VECTOR_ADD_PATH" ]; then
  # Run with timeout to avoid hanging; suppress output
  echo "Running CUDA sample vectorAdd to sanity-check default GPU availability..."
  timeout 10s "$VECTOR_ADD_PATH" >/dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "❌ The current GPU is not available."
  else
    echo "✅ Passed."
  fi
fi
