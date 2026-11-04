/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <sstream>
#include <string>

/* nvbit interface file */
#include "nvbit.h"

/* include environment configuration */
#include "env_config.h"

/* include log handle header */
#include "log.h"

/**
 * Builds a deterministic base filename for a kernel's trace log.
 *
 * The resulting string embeds:
 *   - The kernel_checksum (FNV-1a hash of kernel name + SASS) for identification
 *   - The iteration number (decimal)
 *   - A truncated copy (first 150 chars) of the mangled name for readability
 *
 * If output_dir is set (via CUTRACER_OUTPUT_DIR), the filename
 * will be prefixed with that directory path.
 *
 * Example: "kernel_7fa21c3_iter42__Z23my_kernelPiS_..."
 * With path: "/tmp/traces/kernel_7fa21c3_iter42__Z23my_kernelPiS_..."
 */
std::string generate_kernel_log_basename(CUcontext ctx, CUfunction func, uint32_t iteration,
                                         const std::string& kernel_checksum) {
  const char* mangled_name_raw = nvbit_get_func_name(ctx, func, true);
  if (!mangled_name_raw) {
    mangled_name_raw = "unknown_kernel";
  }

  std::string mangled_name(mangled_name_raw);

  // Truncate the name for the filename string part
  std::string truncated_name = mangled_name.substr(0, 150);

  std::stringstream ss;

  // Prepend output_dir if specified
  if (!output_dir.empty()) {
    ss << output_dir;
    // Ensure path ends with a separator
    if (output_dir.back() != '/') {
      ss << "/";
    }
  }

  // Format to hex for the hash
  ss << "kernel_" << kernel_checksum << "_iter" << std::dec << iteration << "_" << truncated_name;

  return ss.str();
}

/* ===== Global Variables ===== */

// The main log file for the entire process run
static FILE* g_main_log_file = NULL;
// The currently active log file for kernel traces
static FILE* g_kernel_log_file = NULL;

/* ===== Utility Functions for Logging ===== */

/**
 * @brief Base function for formatted output. Uses va_list to avoid re-formatting.
 *
 * @param file_output if true, output to the active log file
 * @param stdout_output if true, output to stdout
 * @param format format string
 * @param args variable argument list
 */
static void vfprintf_base(bool file_output, bool stdout_output, const char* format, va_list args) {
  if (!file_output && !stdout_output) {
    return;
  }

  va_list args_copy;
  va_copy(args_copy, args);

  int needed = vsnprintf(nullptr, 0, format, args);
  if (needed < 0) {
    va_end(args_copy);
    return;
  }

  std::string output_buffer(needed + 1, '\0');
  vsnprintf(output_buffer.data(), needed + 1, format, args_copy);
  va_end(args_copy);

  if (stdout_output) {
    fprintf(stdout, "%s", output_buffer.c_str());
  }

  if (file_output && g_main_log_file) {
    fprintf(g_main_log_file, "%s", output_buffer.c_str());
  }
}

void lprintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf_base(true, false, format, args);
  va_end(args);
}

void oprintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf_base(false, true, format, args);
  va_end(args);
}

void loprintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf_base(true, true, format, args);
  // Flush the main log file if it exists
  if (!g_main_log_file) {
    oprintf("ERROR: Main log file not initialized before loprintf\n");
  }

  va_end(args);
}

void trace_lprintf(const char* format, ...) {
  if (!g_kernel_log_file) {
    oprintf("ERROR: Kernel trace log file not initialized before trace_lprintf\n");
    return;
  }

  va_list args;
  va_start(args, format);
  vfprintf(g_kernel_log_file, format, args);
  va_end(args);
}

/* ===== File Management Functions ===== */

void log_open_kernel_file(CUcontext_ptr ctx, CUfunction_ptr func, uint32_t iteration,
                          const std::string& kernel_checksum) {
  // close previous log file if it's open
  log_close_kernel_file();

  std::string basename = generate_kernel_log_basename((CUcontext)ctx, (CUfunction)func, iteration, kernel_checksum);
  std::string log_filename = basename + ".log";

  g_kernel_log_file = fopen(log_filename.c_str(), "w");
  if (g_kernel_log_file) {
    loprintf("Opened kernel trace log: %s\n", log_filename.c_str());
  } else {
    oprintf("ERROR: Failed to open kernel trace log file: %s\n", log_filename.c_str());
  }
}

void log_close_kernel_file() {
  if (g_kernel_log_file) {
    fclose(g_kernel_log_file);
    g_kernel_log_file = NULL;
  }
}

void init_log_handle() {
  // Get current timestamp for filename
  time_t now = time(0);
  struct tm* timeinfo = localtime(&now);
  char timestamp[32];
  strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", timeinfo);

  // Build the log filename, prepending output_dir if set
  std::string main_log_filename;
  if (!output_dir.empty()) {
    main_log_filename = output_dir;
    if (main_log_filename.back() != '/') {
      main_log_filename += "/";
    }
  }
  main_log_filename += "cutracer_main_";
  main_log_filename += timestamp;
  main_log_filename += "_" + std::to_string(getpid());
  main_log_filename += ".log";

  g_main_log_file = fopen(main_log_filename.c_str(), "w");
  if (!g_main_log_file) {
    // Fallback to stdout if file creation fails
    g_main_log_file = stdout;
    oprintf("WARNING: Failed to create main log file. Falling back to stdout.\n");
  }

  loprintf("Log handle system initialized. Main log is %s.\n",
           (g_main_log_file == stdout) ? "stdout" : main_log_filename.c_str());
}

void flush_log_files() {
  if (g_main_log_file && g_main_log_file != stdout) {
    fflush(g_main_log_file);
  }
  if (g_kernel_log_file) {
    fflush(g_kernel_log_file);
  }
}

void cleanup_log_handle() {
  log_close_kernel_file();

  if (g_main_log_file && g_main_log_file != stdout) {
    fclose(g_main_log_file);
  }

  g_main_log_file = NULL;

  oprintf_v("Log handle system cleaned up.\n");
}
