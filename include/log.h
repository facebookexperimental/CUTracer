/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#ifndef LOG_HANDLE_H
#define LOG_HANDLE_H

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include <string>

#include "cuda.h"

// Forward declarations - use void* to avoid conflicts with CUDA types
using CUcontext_ptr = void*;
using CUfunction_ptr = void*;

/* ===== Logging Functions ===== */

/**
 * lprintf - print to the currently active log file only (log print)
 */
void lprintf(const char* format, ...);

/* ===== Verbose-conditional Logging Macros ===== */

// Forward declaration for verbose variable
extern int verbose;

/**
 * loprintf_v - verbose-conditional loprintf
 * Only prints when verbose is non-zero.
 */
#define loprintf_v(fmt, ...)                   \
  do {                                         \
    if (verbose) loprintf(fmt, ##__VA_ARGS__); \
  } while (0)

/**
 * loprintf_vl - verbose level-conditional loprintf
 * Only prints when verbose >= level.
 */
#define loprintf_vl(level, fmt, ...)                      \
  do {                                                    \
    if (verbose >= (level)) loprintf(fmt, ##__VA_ARGS__); \
  } while (0)

/**
 * lprintf_v - verbose-conditional lprintf
 * Only prints to log file when verbose is non-zero.
 */
#define lprintf_v(fmt, ...)                   \
  do {                                        \
    if (verbose) lprintf(fmt, ##__VA_ARGS__); \
  } while (0)

/**
 * oprintf_v - verbose-conditional oprintf
 * Only prints to stdout when verbose is non-zero.
 */
#define oprintf_v(fmt, ...)                   \
  do {                                        \
    if (verbose) oprintf(fmt, ##__VA_ARGS__); \
  } while (0)

/**
 * oprintf - print to stdout only (output print)
 */
void oprintf(const char* format, ...);

/**
 * loprintf - print to the currently active log file and stdout (log and output print)
 */
void loprintf(const char* format, ...);

/**
 * trace_lprintf - print to the kernel trace log file only
 */
void trace_lprintf(const char* format, ...);

/* ===== File Management Functions ===== */

/**
 * Opens a new log file for a specific kernel invocation.
 * This should be called on kernel entry.
 * @param ctx CUDA context
 * @param func CUfunction representing the kernel
 * @param iteration Current iteration of the kernel execution
 * @param kernel_checksum FNV-1a hash of kernel name + SASS (hex string)
 */
void log_open_kernel_file(CUcontext_ptr ctx, CUfunction_ptr func, uint32_t iteration,
                          const std::string& kernel_checksum);

/**
 * Closes the kernel-specific log file.
 * This should be called on kernel exit.
 */
void log_close_kernel_file();

/**
 * Initializes the log handle system. Creates the main process log file.
 */
void init_log_handle();

/**
 * Cleans up the log handle system. Closes the main process log file.
 */
void cleanup_log_handle();

/**
 * Flushes all open log file buffers to disk.
 * Called periodically from recv_thread to ensure log data is persisted
 * during kernel hangs (when normal cleanup paths are blocked).
 */
void flush_log_files();

/**
 * Builds a deterministic base filename for a kernel's trace log.
 *
 * Format:
 *   "kernel_<kernel_checksum>_iter<iteration>_<truncated_mangled_name>"
 *
 * Details:
 * - Uses the kernel_checksum (FNV-1a hash of kernel name + SASS instructions)
 *   for robust kernel identification across recompilations.
 * - Appends the kernel iteration number to distinguish repeated launches.
 * - Includes a truncated (up to 150 chars) copy of the mangled name to aid
 *   human readability while keeping the filename manageable.
 *
 * Args:
 *   ctx: CUDA context associated with the kernel function.
 *   func: The CUfunction handle of the kernel.
 *   iteration: Per-kernel iteration counter maintained by the caller.
 *   kernel_checksum: FNV-1a hash of kernel name + SASS (hex string).
 *
 * Returns:
 *   The base filename (without extension) for the kernel-specific log file.
 */
std::string generate_kernel_log_basename(CUcontext ctx, CUfunction func, uint32_t iteration,
                                         const std::string& kernel_checksum);

#endif /* LOG_HANDLE_H */
