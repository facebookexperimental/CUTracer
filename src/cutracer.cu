/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * This source code contains modifications by Meta Platforms, Inc. licensed
 * under MIT, based on original NVIDIA NVBit sample code licensed under
 * BSD-3-Clause. See LICENSE file in the root directory for Meta's license
 * terms. See LICENSE-BSD file in the root directory for NVIDIA's license terms.
 */

#include <assert.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* contains definition of the reg_info_t and mem_access_t structure */
#include "common.h"

/* analysis functionality */
#include "analysis.h"

/* precompiled flush_channel fatbin (bin2c byte array) */
#include "tool_func/flush_channel.c"

/* instrumentation functionality */
#include "instrument.h"

/* env config */
#include "env_config.h"
#include "python_callstack.h"
#include "trace_writer.h"

/* logging functionality */
#include "log.h"

/* delay configuration */
#include "delay_inject_config.h"

/* implicit registers collection */
#include "implicit_regs.h"

#define CUDA_CHECK_LAST_ERROR()                                                                       \
  do {                                                                                                \
    cudaError_t err = cudaGetLastError();                                                             \
    if (err != cudaSuccess) {                                                                         \
      loprintf("FATAL: CUDA Last Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      fflush(stderr);                                                                                 \
      assert(err == cudaSuccess);                                                                     \
    }                                                                                                 \
  } while (0)

/* lock */
pthread_mutex_t mutex;
pthread_mutex_t cuda_event_mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* The kernel launch is identified by a global id */
uint64_t global_kernel_launch_id = 0;

/* Kernel events writer for structured launch event recording */
static TraceWriter* g_kernel_events_writer = nullptr;

/* Callstack dedup table: FNV-1a hash → already emitted */
static std::unordered_map<uint64_t, bool> callstack_seen;

/* Random number generator for delay injection on/off control.
 * Uses std::random_device to seed std::mt19937 for true randomness
 * across different runs/processes. */
static std::mt19937 g_delay_rng;
static bool g_delay_rng_initialized = false;

/* Lazily initialize the RNG with true randomness */
static void ensure_delay_rng_initialized() {
  if (!g_delay_rng_initialized) {
    std::random_device rd;
    g_delay_rng.seed(rd());
    g_delay_rng_initialized = true;
  }
}

/* Generate random enabled state with configurable probability (g_delay_enable_prob, default 0.5).
 * Short-circuits at the endpoints so --delay-enable-prob 1.0 / 0.0 are bit-deterministic and the
 * dumped replay config truly contains all-enabled / all-disabled points. */
static bool generate_random_delay_enabled() {
  ensure_delay_rng_initialized();
  if (g_delay_enable_prob >= 1.0f) return true;
  if (g_delay_enable_prob <= 0.0f) return false;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(g_delay_rng) < g_delay_enable_prob;
}

/* Generate a random seed for cluster mode CTA selection */
static uint32_t generate_cluster_seed() {
  ensure_delay_rng_initialized();
  return g_delay_rng();
}

/**
 * @brief Extract the cluster_dim attribute from a cuLaunchKernelEx params
 *        struct, if present. Returns true and fills (cx, cy, cz) on hit;
 *        returns false (and leaves outputs unchanged) if either the cbid is
 *        not a cuLaunchKernelEx variant or the attribute is absent.
 *
 * Centralizes the launch-attr scan that's needed in two places: at every
 * direct launch (so graph-node launches can later look up the dim) and at the
 * direct-launch logging site itself.
 */
static bool extract_cluster_dim_from_launch_ex(nvbit_api_cuda_t cbid, void* params, unsigned int& cx, unsigned int& cy,
                                               unsigned int& cz) {
  if (cbid != API_CUDA_cuLaunchKernelEx_ptsz && cbid != API_CUDA_cuLaunchKernelEx) {
    return false;
  }
  cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
  for (unsigned int i = 0; i < p->config->numAttrs; ++i) {
    if (p->config->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
      cx = p->config->attrs[i].value.clusterDim.x;
      cy = p->config->attrs[i].value.clusterDim.y;
      cz = p->config->attrs[i].value.clusterDim.z;
      return true;
    }
  }
  return false;
}

/**
 * @brief Log cluster dimensions for a kernel under --delay-mode cluster, once
 *        per CUfunction.
 *
 * If the caller has a launch-time cluster dim (via CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
 * on cuLaunchKernelEx), pass it in via dyn_*; otherwise pass 0 and we fall back
 * to the kernel-declared __cluster_dims__ via cuFuncGetAttribute. When the
 * resulting cluster size is <= 1 we emit a WARNING because cluster mode is a
 * no-op for non-cluster-launched kernels (the device function short-circuits).
 */
static void log_cluster_info_once(CUcontext /*ctx*/, CUfunction func, const char* func_name, unsigned int dyn_cx,
                                  unsigned int dyn_cy, unsigned int dyn_cz) {
  // The set tracks "have we already emitted the [CLUSTER] log for this
  // CUfunction?". Both call sites (enter_kernel_launch, nvbit_at_graph_node_launch)
  // hold *different* outer mutexes (cuda_event_mutex vs mutex), so we need our
  // own lock to serialize concurrent inserts safely.
  static std::unordered_set<CUfunction> g_cluster_logged_funcs;
  static pthread_mutex_t g_cluster_logged_mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&g_cluster_logged_mutex);
  bool inserted = g_cluster_logged_funcs.insert(func).second;
  pthread_mutex_unlock(&g_cluster_logged_mutex);
  if (!inserted) {
    return;
  }
  unsigned int cx = dyn_cx, cy = dyn_cy, cz = dyn_cz;
  if (cx == 0 || cy == 0 || cz == 0) {
    int rcw = 0, rch = 0, rcd = 0;
    if (cuFuncGetAttribute(&rcw, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, func) != CUDA_SUCCESS) rcw = 0;
    if (cuFuncGetAttribute(&rch, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, func) != CUDA_SUCCESS) rch = 0;
    if (cuFuncGetAttribute(&rcd, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, func) != CUDA_SUCCESS) rcd = 0;
    cx = rcw > 0 ? (unsigned int)rcw : 1;
    cy = rch > 0 ? (unsigned int)rch : 1;
    cz = rcd > 0 ? (unsigned int)rcd : 1;
  }
  unsigned int cluster_size = cx * cy * cz;
  if (cluster_size <= 1) {
    loprintf(
        "[CLUSTER] WARNING: kernel=%s cluster_dim=(%u,%u,%u) size=%u - "
        "cluster delay mode is a NO-OP for non-cluster-launched kernels; "
        "use --delay-mode random to expose intra-CTA races instead\n",
        func_name, cx, cy, cz, cluster_size);
  } else {
    loprintf(
        "[CLUSTER] kernel=%s cluster_dim=(%u,%u,%u) size=%u - "
        "cluster delay will affect 1 of %u CTAs per cluster\n",
        func_name, cx, cy, cz, cluster_size, cluster_size);
  }
}

/**
 * @brief Capture the current CPU call stack using backtrace() and dladdr().
 *
 * Returns a vector of human-readable frame strings containing:
 * - Function name (demangled via abi::__cxa_demangle when possible)
 * - Shared object path
 * - Offset within the symbol
 *
 * Skips the first `skip` frames (default 2) to exclude this function
 * and internal NVBit callback frames from the output.
 *
 * @param max_frames Maximum number of frames to capture
 * @param skip Number of leading frames to skip
 * @return Vector of frame description strings
 */
__attribute__((noinline)) static std::vector<std::string> capture_cpu_callstack_backtrace(int max_frames = 64,
                                                                                          int skip = 2) {
  std::vector<std::string> result;
  std::vector<void*> buffer(max_frames);

  int nframes = backtrace(buffer.data(), max_frames);
  if (nframes <= skip) {
    return result;
  }

  result.reserve(nframes - skip);
  for (int i = skip; i < nframes; i++) {
    Dl_info info;
    std::ostringstream oss;

    if (dladdr(buffer[i], &info)) {
      const char* soname = info.dli_fname ? info.dli_fname : "??";
      ptrdiff_t offset = 0;
      if (info.dli_saddr) {
        offset = static_cast<const char*>(buffer[i]) - static_cast<const char*>(info.dli_saddr);
      }

      // Attempt C++ name demangling for readable output
      std::string display_name = info.dli_sname ? info.dli_sname : "??";
      if (info.dli_sname) {
        int status = 0;
        char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
        if (status == 0 && demangled) {
          display_name = demangled;
          free(demangled);
        }
      }

      oss << display_name << "+0x" << std::hex << offset << " (" << soname << ")";
    } else {
      // Fallback: use backtrace_symbols for this single frame
      char** symbols = backtrace_symbols(&buffer[i], 1);
      if (symbols && symbols[0]) {
        oss << symbols[0];
      } else {
        oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(buffer[i]);
      }
      free(symbols);
    }
    result.push_back(oss.str());
  }

  return result;
}

/**
 * @brief Entry point for CPU call stack capture.
 *
 * Selects the call stack capture method based on cpu_callstack_mode:
 * - AUTO: Try PyTorch CapturedTraceback first, fallback to backtrace()
 * - PYTORCH: Use PyTorch only (empty if unavailable)
 * - BACKTRACE: Use backtrace() only (original behavior)
 * - DISABLED: No capture
 *
 * Returns {frames, source} where source identifies the capture method used.
 *
 * Uses skip=3 to account for the extra wrapper frame introduced by this
 * function, so the output starts from the caller's caller (the NVBit
 * callback), matching the original behavior before the refactor.
 * Frames skipped: capture_cpu_callstack_backtrace, capture_cpu_callstack,
 * enter_kernel_launch (or nvbit_at_graph_node_launch).
 */
__attribute__((noinline)) static std::pair<std::vector<std::string>, std::string> capture_cpu_callstack() {
  switch (cpu_callstack_mode) {
    case CpuCallstackMode::DISABLED:
      return {{}, ""};
    case CpuCallstackMode::BACKTRACE:
      return {capture_cpu_callstack_backtrace(64, 3), "backtrace"};
    case CpuCallstackMode::PYTORCH: {
      auto result = capture_cpu_callstack_pytorch();
      if (result.empty()) {
        loprintf_v(
            "PyTorch callstack capture returned empty "
            "(Python/GIL unavailable on this thread)\n");
      }
      return {std::move(result), "pytorch"};
    }
    case CpuCallstackMode::AUTO: {
      auto result = capture_cpu_callstack_pytorch();
      if (!result.empty()) {
        return {std::move(result), "pytorch"};
      }
      return {capture_cpu_callstack_backtrace(64, 3), "backtrace"};
    }
    case CpuCallstackMode::AUTO_GIL: {
      auto result = capture_cpu_callstack_pytorch_acquire_gil();
      if (!result.empty()) {
        return {std::move(result), "cpython_gil"};
      }
      return {capture_cpu_callstack_backtrace(64, 3), "backtrace"};
    }
  }
  // Unreachable for known enum values; fallback defensively for future additions
  return {capture_cpu_callstack_backtrace(64, 3), "backtrace"};
}

/**
 * @brief Compute FNV-1a hash of a callstack (vector of frame strings).
 *
 * Uses the same FNV-1a constants as compute_kernel_checksum().
 * Includes an inter-frame separator byte (0xFF) to prevent collisions
 * between e.g. {"a","bc"} and {"ab","c"}.
 */
static uint64_t hash_callstack(const std::vector<std::string>& frames) {
  const uint64_t FNV_OFFSET = 14695981039346656037ULL;
  const uint64_t FNV_PRIME = 1099511628211ULL;
  uint64_t hash = FNV_OFFSET;
  for (const auto& frame : frames) {
    for (char c : frame) {
      hash ^= static_cast<uint64_t>(c);
      hash *= FNV_PRIME;
    }
    hash ^= 0xFF;
    hash *= FNV_PRIME;
  }
  return hash;
}

/**
 * @brief Write a kernel launch event to the kernel events file.
 *
 * In DEDUP mode, emits a callstack_def record on first occurrence and
 * references it by callstack_id in subsequent launch events.
 * In FULL mode, inlines the complete callstack in every launch event.
 * In NOSTACK mode, omits callstack information entirely.
 */
static void write_kernel_launch_event(const KernelFuncMetadata& meta, const KernelDimensions& dims,
                                      unsigned int dynamic_shmem, uint64_t stream_id, uint64_t launch_id,
                                      const std::vector<std::string>& callstack, const std::string& callstack_source) {
  if (!g_kernel_events_writer || kernel_events_mode == KernelEventsMode::DISABLED) {
    return;
  }

  nlohmann::json event;
  event["type"] = "kernel_launch";
  event["kernel_launch_id"] = launch_id;
  event["kernel_name"] = meta.unmangled_name;
  event["kernel_checksum"] = meta.kernel_checksum;
  event["grid"] = {dims.gridDimX, dims.gridDimY, dims.gridDimZ};
  event["block"] = {dims.blockDimX, dims.blockDimY, dims.blockDimZ};
  event["nregs"] = meta.nregs;
  event["shmem"] = meta.shmem_static_nbytes + dynamic_shmem;
  event["stream_id"] = stream_id;

  if (kernel_events_mode == KernelEventsMode::DEDUP) {
    if (callstack.empty()) {
      event["callstack_id"] = nullptr;
    } else {
      uint64_t cs_hash = hash_callstack(callstack);
      std::ostringstream oss;
      oss << std::hex << cs_hash;
      std::string callstack_id = oss.str();
      event["callstack_id"] = callstack_id;

      if (callstack_seen.emplace(cs_hash, true).second) {
        nlohmann::json def;
        def["type"] = "callstack_def";
        def["callstack_id"] = callstack_id;
        def["frames"] = callstack;
        if (!callstack_source.empty()) {
          def["source"] = callstack_source;
        }
        g_kernel_events_writer->write_metadata(def);
      }
    }
  } else if (kernel_events_mode == KernelEventsMode::FULL) {
    if (!callstack.empty()) {
      event["cpu_callstack"] = callstack;
      if (!callstack_source.empty()) {
        event["cpu_callstack_source"] = callstack_source;
      }
    }
  }
  // NOSTACK mode: no callstack fields added

  g_kernel_events_writer->write_metadata(event);

  // Flush after each event for reliability — kernel events are metadata-only
  // (low volume), so the overhead is acceptable. Without explicit flush,
  // write_metadata() data stays in memory (no threshold-based auto-flush)
  // and would be lost on abnormal process exit.
  g_kernel_events_writer->flush();
}

// Global mapping tables for kernel launch tracking
std::map<uint64_t, std::pair<CUcontext, CUfunction>> kernel_launch_to_func_map;
std::map<uint64_t, uint32_t> kernel_launch_to_iter_map;
std::map<uint64_t, KernelDimensions> kernel_launch_to_dimensions_map;

// Cluster dim observed at the most recent cuLaunchKernelEx for a given
// CUfunction (including stream-captured launches). This lets the graph-node
// launch path see the launch-time cluster dim that Triton sets via
// CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, which cuFuncGetAttribute can't see
// because Triton doesn't use the static __cluster_dims__ kernel attribute.
struct ObservedClusterDim {
  unsigned int x = 0, y = 0, z = 0;
};

std::unordered_map<CUfunction, ObservedClusterDim> g_func_to_observed_cluster;
pthread_mutex_t g_cluster_obs_mutex = PTHREAD_MUTEX_INITIALIZER;

// map to store the iteration count for each kernel
static std::map<CUfunction, uint32_t> kernel_iter_map;

// Per-function static metadata (populated during instrumentation)
std::unordered_map<CUfunction, KernelFuncMetadata> kernel_metadata_by_func;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/**
 * @brief Compute a kernel checksum for robust kernel identification.
 *
 * Uses FNV-1a algorithm to hash the kernel name AND all SASS instruction strings.
 * This provides robust identification across recompilations - same kernel name
 * with different SASS (due to different compilation options, optimizations, etc.)
 * will produce different checksums.
 *
 * NOTE: This uses the ORIGINAL SASS from nvbit_get_instrs() before any
 * instrumentation is applied. The delay injection happens later and does
 * not affect the checksum.
 *
 * @param kernel_name The kernel's mangled name
 * @param instrs Vector of instructions from nvbit_get_instrs()
 * @return Hex string of the FNV-1a hash
 */
static std::string compute_kernel_checksum(const std::string& kernel_name, const std::vector<Instr*>& instrs) {
  // FNV-1a constants
  const uint64_t FNV_OFFSET = 14695981039346656037ULL;
  const uint64_t FNV_PRIME = 1099511628211ULL;

  uint64_t hash = FNV_OFFSET;

  // First, hash the kernel name
  for (char c : kernel_name) {
    hash ^= static_cast<uint64_t>(c);
    hash *= FNV_PRIME;
  }

  // Then hash all SASS instructions
  for (const auto& instr : instrs) {
    const char* sass = instr->getSass();
    if (sass) {
      // Hash each character of the SASS string
      for (const char* p = sass; *p; ++p) {
        hash ^= static_cast<uint64_t>(*p);
        hash *= FNV_PRIME;
      }
    }
    // Also include offset for additional uniqueness
    uint64_t offset = instr->getOffset();
    hash ^= offset;
    hash *= FNV_PRIME;
  }

  // Convert to hex string
  std::ostringstream oss;
  oss << std::hex << std::nouppercase << hash;
  return oss.str();
}

/**
 * @brief Ensure kernel_checksum is populated for the given metadata.
 *
 * Idempotent: if checksum is already computed, this is a no-op.
 * Also ensures mangled_name is set (needed for checksum computation).
 *
 * @param meta The kernel metadata entry to populate
 * @param ctx The CUDA context
 * @param func The CUfunction handle
 */
static void ensure_kernel_checksum(KernelFuncMetadata& meta, CUcontext ctx, CUfunction func) {
  if (!meta.kernel_checksum.empty()) return;
  if (meta.mangled_name.empty()) {
    meta.mangled_name = nvbit_get_func_name(ctx, func, true);
  }
  const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, func);
  meta.kernel_checksum = compute_kernel_checksum(meta.mangled_name, instrs);
}

/**
 * @brief Check if an instruction should be instrumented based on category filtering.
 *
 * When category filtering is enabled (CUTRACER_INSTR_CATEGORIES is set),
 * only instructions belonging to enabled categories will be instrumented.
 * When category filtering is disabled, all instructions are instrumented.
 *
 * @param instr The instruction to check
 * @return true if the instruction should be instrumented
 */
static bool should_instrument_instr_by_category(Instr* instr) {
  if (!has_category_filter_enabled()) {
    return true;
  }

  const char* sass_str = instr->getSass();
  InstrCategory category = detect_instr_category(sass_str);

  if (category == InstrCategory::NONE) {
    // Not a categorized instruction - skip when category filtering is enabled
    return false;
  }

  // Check if this category is enabled
  bool should_instrument = should_instrument_category(category);
  if (should_instrument) {
    loprintf_v("Category filter: instrumenting %s instruction at pc=0x%lx: %s\n", get_instr_category_name(category),
               instr->getOffset(), sass_str);
  }
  return should_instrument;
}

/**
 * @brief Check if an instruction should receive delay injection.
 *
 * Uses user-specified patterns (g_delay_patterns / --delay-patterns) if provided,
 * otherwise falls back to the built-in DELAY_INJECTION_PATTERNS list.
 * Special case: "*" matches all instructions.
 */
static bool shouldInjectDelayWithPatterns(Instr* instr) {
  static std::vector<const char*> user_patterns_cstr;
  static bool user_patterns_built = false;
  static bool delay_all_instructions = false;
  if (!user_patterns_built && !g_delay_patterns.empty()) {
    delay_all_instructions = (g_delay_patterns.size() == 1 && g_delay_patterns[0] == "*");
    if (!delay_all_instructions) {
      for (const auto& p : g_delay_patterns) {
        user_patterns_cstr.push_back(p.c_str());
      }
    }
    user_patterns_built = true;
  }
  if (delay_all_instructions) {
    return true;
  }
  const auto& active_patterns = g_delay_patterns.empty() ? DELAY_INJECTION_PATTERNS : user_patterns_cstr;
  return shouldInjectDelay(instr, active_patterns);
}

/* ===== Main Functionality ===== */

/**
 * @brief Get or create per-function metadata entry.
 *
 * If the function is seen for the first time, inserts a new entry and
 * populates all fields (name, func_addr, nregs, shmem, checksum) via
 * CUDA driver API and NVBit.  Subsequent calls return the existing entry
 * directly.
 */
static KernelFuncMetadata& get_or_create_kernel_func_metadata(CUfunction func, CUcontext ctx) {
  auto [it, inserted] = kernel_metadata_by_func.emplace(func, KernelFuncMetadata{});
  if (inserted) {
    it->second.unmangled_name = nvbit_get_func_name(ctx, func, false);
    it->second.func_addr = nvbit_get_func_addr(ctx, func);
    CUDA_SAFECALL(cuFuncGetAttribute(&it->second.nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));
    CUDA_SAFECALL(cuFuncGetAttribute(&it->second.shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));
    it->second.sm_family = nvbit_get_sm_family(ctx);
    ensure_kernel_checksum(it->second, ctx, func);
  }
  return it->second;
}

/**
 * @brief Conditionally instruments a CUDA function by delegating to specialized
 * instrumentation functions.
 *
 * This function is based on the `instrument_kernel` function from NVIDIA's
 * `mem_trace` example. It inspects a function and, if it matches the filters,
 * iterates through its SASS instructions, delegating the actual instrumentation
 * to functions in `instrument.cu`.
 *
 * Meta's enhancements include:
 *  - Combining memory (MREF) and register (REG) tracing.
 *  - Adding support for Unified Registers (UREG).
 *  - Kernel Filtering: A system (`kernel_filters`) to selectively
 *    instrument kernels based on environment variables, avoiding unnecessary
 *    overhead.
 *  - Modular Instrumentation API: The core instrumentation logic is refactored
 *    into `instrument.cu`, providing a clear API for different tracing types.
 *
 * @param ctx The CUDA context of the function.
 * @param func The `CUfunction` to inspect and potentially instrument.
 * @return `true` if instrumentation should be enabled, `false` otherwise.
 */
bool instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  if (!has_any_instrumentation_enabled()) {
    return false;
  }
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  // Static regex pattern for extracting UR register numbers from GENERIC operands
  static std::regex ureg_pattern(R"(UR(\d+))");

  /* Get related functions of the kernel (device function that can be
   * called by the kernel) */
  std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  // No filters specified implies we want to instrument all functions
  bool should_instrument = kernel_filters.empty();
  /* iterate on function */
  for (auto f : related_functions) {
    const char* unmangled_name = nvbit_get_func_name(ctx, f, false);
    const char* mangled_name = nvbit_get_func_name(ctx, f, true);

    // Check if the current function matches any kernel filter
    bool kernel_filter_matched = kernel_filters.empty();
    for (const auto& filter : kernel_filters) {
      if ((unmangled_name && strstr(unmangled_name, filter.c_str()) != NULL) ||
          (mangled_name && strstr(mangled_name, filter.c_str()) != NULL)) {
        kernel_filter_matched = true;
        // Update return value (determines if instrumented code should run)
        should_instrument = true;
        loprintf_v("Found matching kernel for filter '%s': %s (mangled: %s)\n", filter.c_str(),
                   unmangled_name ? unmangled_name : "unknown", mangled_name ? mangled_name : "unknown");
        break;
      }
    }

    // Skip if filter not matched or already instrumented
    if (!kernel_filter_matched || !already_instrumented.insert(f).second) {
      continue;
    }

    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
    loprintf_v("Inspecting kernel %s at address 0x%lx\n", nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));

    // Get or create per-function metadata (checksum already computed by helper)
    auto& meta = get_or_create_kernel_func_metadata(f, ctx);
    loprintf_v("Kernel checksum for %s: %s\n", meta.mangled_name.c_str(), meta.kernel_checksum.c_str());

    if (dump_cubin) {
      // Use the same "kernel_{checksum}_{name}" prefix as trace files
      // so Python analyze can derive cubin path from trace filename
      // (trace: kernel_{checksum}_iter{N}_{name}.ndjson → cubin: kernel_{checksum}_{name}.cubin)
      std::string truncated_name = std::string(mangled_name).substr(0, 150);
      std::string cubin_filename = "kernel_" + meta.kernel_checksum + "_" + truncated_name + ".cubin";
      if (!output_dir.empty()) {
        meta.cubin_path = output_dir + (output_dir.back() != '/' ? "/" : "") + cubin_filename;
      } else {
        meta.cubin_path = cubin_filename;
      }
      nvbit_dump_cubin(ctx, f, meta.cubin_path.c_str());
    }

    // Create kernel delay config if dump path is set (for exporting to JSON)
    // Note: dump mode and replay mode are mutually exclusive
    KernelDelayInjectConfig* kernel_delay_config = nullptr;
    if (!delay_dump_path.empty() && is_instrument_type_enabled(InstrumentType::RANDOM_DELAY)) {
      assert(!is_delay_replay_mode() && "Dump and replay modes are mutually exclusive");
      // Reuse the kernel_checksum already computed above for trace writer
      kernel_delay_config = create_kernel_delay_config(mangled_name, meta.kernel_checksum);
    }

    // Get replay instrumentation points once for replay mode
    const std::map<uint64_t, DelayInstrumentationPoint>* replay_points = nullptr;
    if (is_delay_replay_mode() && is_instrument_type_enabled(InstrumentType::RANDOM_DELAY)) {
      assert(delay_dump_path.empty() && "Dump and replay modes are mutually exclusive");
      // Reuse the kernel_checksum already computed above for trace writer
      replay_points = get_replay_instrumentation_points(mangled_name, meta.kernel_checksum);
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }
      if (verbose) {
        instr->printDecoded();
      }

      OperandLists operands;
      OperandContext op_ctx;  // Context for implicit register collection
      int mref_idx = 0;
      int opcode_id = instr->getIdx();
      ctx_state->id_to_sass_map[f][opcode_id] = std::string(instr->getSass());

      // Collect SASS binary encoding (nvbit 1.8+)
      {
        std::vector<uint8_t> binary;
        instr->getSassBinary([](uint8_t b, void* ud) { static_cast<std::vector<uint8_t>*>(ud)->push_back(b); },
                             &binary);
        if (!binary.empty()) {
          ctx_state->id_to_sass_binary_map[f][opcode_id] = std::move(binary);
        }
      }

      /* iterate on the operands */
      for (int i = 0; i < instr->getNumOperands(); i++) {
        /* get the operand "i" */
        const InstrType::operand_t* op = instr->getOperand(i);
        if (op->type == InstrType::OperandType::REG) {
          for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
            operands.reg_nums.push_back(op->u.reg.num + reg_idx);
          }
        } else if (op->type == InstrType::OperandType::UREG) {
          for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
            operands.ureg_nums.push_back(op->u.reg.num + reg_idx);
          }
        } else if (op->type == InstrType::OperandType::GENERIC) {
          loprintf_v("  GENERIC operand[%d]: '%s'\n", i, op->u.generic.array);

          // Extract UR register numbers from GENERIC operand using regex
          try {
            std::string generic_str(op->u.generic.array);
            std::sregex_iterator begin(generic_str.begin(), generic_str.end(), ureg_pattern);
            std::sregex_iterator end;

            int match_count = 0;
            for (auto it = begin; it != end; ++it) {
              std::smatch match = *it;
              // match[0] is the full match "URxx"
              // match[1] is the captured group (the number part)
              int ureg_num = std::stoi(match[1].str());

              operands.ureg_nums.push_back(ureg_num);
              op_ctx.generic_urs.push_back(ureg_num);
              op_ctx.generic_strs.push_back(generic_str);  // Store raw string for prefix analysis
              match_count++;

              loprintf_v("    Extracted UREG: UR%d\n", ureg_num);
            }

            if (match_count == 0) {
              loprintf_v("    No UREG found in GENERIC operand\n");
            }
          } catch (const std::exception& e) {
            loprintf("ERROR: Failed to parse GENERIC operand: %s\n", e.what());
          }
        } else if (op->type == InstrType::OperandType::MEM_DESC) {
          loprintf_v("  MEM_DESC operand[%d]: ureg_num=%d\n", i, op->u.mem_desc.ureg_num);
          int desc_lo = op->u.mem_desc.ureg_num;
          op_ctx.desc_urs.push_back(desc_lo);
          // NVBit 1.8+: TMA descriptor pointer is exposed as MEM_DESC instead of MREF.has_ur.
          // Mirror it into operands.desc_urs (consumed by instrument_tma_trace) and
          // operands.ureg_nums (so reg_trace captures both 32-bit halves of the descriptor address).
          operands.desc_urs.push_back(desc_lo);
          operands.ureg_nums.push_back(desc_lo);
          operands.ureg_nums.push_back(desc_lo + 1);
        } else if (op->type == InstrType::OperandType::MREF) {
          if (op->u.mref.has_ra) {
            operands.reg_nums.push_back(op->u.mref.ra_num);
            op_ctx.mref_ras.push_back(op->u.mref.ra_num);
          }
          if (op->u.mref.has_ur) {
            operands.ureg_nums.push_back(op->u.mref.ur_num);
            op_ctx.mref_urs.push_back(op->u.mref.ur_num);
          }
          if (op->u.mref.has_desc) {
            operands.ureg_nums.push_back(op->u.mref.desc_ureg_num);
            operands.ureg_nums.push_back(op->u.mref.desc_ureg_num + 1);
          }

          // Use new instrumentation interface for memory tracing
          if (is_instrument_type_enabled(InstrumentType::MEM_ADDR_TRACE)) {
            instrument_memory_addr_trace(instr, opcode_id, ctx_state, mref_idx);
          }
          // Memory value tracing (captures addresses + values at IPOINT_AFTER)
          if (is_instrument_type_enabled(InstrumentType::MEM_VALUE_TRACE)) {
            int mem_space = (int)instr->getMemorySpace();
            instrument_memory_value_trace(instr, opcode_id, ctx_state, mref_idx, mem_space);
          }
          mref_idx++;
        } else if (op->type == InstrType::OperandType::TMA_PARAM_HANDLE) {
          // NVBit 1.8 packs the TMA URs (URb at slot 0, URa at slot 1) into a
          // single TMA_PARAM_HANDLE operand. The actual handle bytes are
          // pushed to the device function via
          // nvbit_add_call_arg_tma_param_handle_and_size() inside
          // instrument_tma_trace(); here we only mirror the UR numbers into
          // operands.ureg_nums so reg_trace can capture their values.
          loprintf_v("  TMA_PARAM_HANDLE operand[%d]: ureg_nums=[%d,%d,%d,%d]\n", i, op->u.tma_param_handle.ureg_num[0],
                     op->u.tma_param_handle.ureg_num[1], op->u.tma_param_handle.ureg_num[2],
                     op->u.tma_param_handle.ureg_num[3]);
          for (int t = 0; t < InstrType::MAX_TMA_REGS; t++) {
            int ur = op->u.tma_param_handle.ureg_num[t];
            if (ur < 0 || ur == InstrType::URZ) {
              continue;
            }
            operands.ureg_nums.push_back(ur);
          }
        } else {
          loprintf_v("  Unhandled operand[%d] type=%s\n", i, InstrType::OperandTypeStr[(int)op->type]);
        }
      }

      // Collect implicit registers based on instruction semantics
      // (e.g., UTMALDG uses URb+1 for barrier address, not exposed by NVBit)
      collect_implicit_regs(instr->getSass(), op_ctx, operands);

      // Store register indices in CPU-side static map (no GPU overhead)
      // Must be AFTER collect_implicit_regs() since implicit regs modify operands
      {
        RegIndices reg_idx;
        reg_idx.reg_indices.reserve(operands.reg_nums.size());
        for (int num : operands.reg_nums) {
          reg_idx.reg_indices.push_back(static_cast<uint8_t>(num));
        }
        reg_idx.ureg_indices.reserve(operands.ureg_nums.size());
        for (int num : operands.ureg_nums) {
          reg_idx.ureg_indices.push_back(static_cast<uint8_t>(num));
        }
        ctx_state->id_to_reg_indices_map[f][opcode_id] = std::move(reg_idx);
      }

      // Choose instrumentation based on enabled types (only if this instruction passes category filter)
      if (should_instrument_instr_by_category(instr)) {
        if (is_instrument_type_enabled(InstrumentType::OPCODE_ONLY)) {
          // Lightweight instrumentation for instruction histogram analysis.
          // This sends minimal data (opcode_id, warp_id) to the CPU, reducing
          // overhead.
          instrument_opcode_only(instr, opcode_id, ctx_state);
        } else if (is_instrument_type_enabled(InstrumentType::REG_TRACE)) {
          // Full register tracing.
          instrument_register_trace(instr, opcode_id, ctx_state, operands);
        }

        // TMA_TRACE: tensor memory accelerator tracing (independent of REG_TRACE/OPCODE_ONLY).
        // NVBit 1.8 exposes first-class classification via isTMAMem()/isTMATensor(),
        // replacing the SASS-substring check used previously.
        if (is_instrument_type_enabled(InstrumentType::TMA_TRACE)) {
          if (instr->isTMAMem() || instr->isTMATensor()) {
            instrument_tma_trace(instr, opcode_id, ctx_state, ctx);
          }
        }
      }

      // Delay instrumentation for synchronization instructions
      if (is_instrument_type_enabled(InstrumentType::RANDOM_DELAY) && shouldInjectDelayWithPatterns(instr)) {
        bool enabled;
        uint32_t delay_ns;
        uint32_t cluster_seed = 0;
        bool found = true;

        if (is_delay_replay_mode()) {
          // Replay mode: always use config values, no random
          uint64_t pc_offset = instr->getOffset();
          found = lookup_replay_config(replay_points, pc_offset, enabled, delay_ns, cluster_seed);
          if (!found) {
            // Point not found in config - skip instrumentation for this point
            loprintf("Replay mode: kernel=%s pc=0x%lx NOT FOUND in config, skipping\n", unmangled_name, pc_offset);
          }
        } else {
          // Random mode: generate random on/off with 50% probability
          enabled = generate_random_delay_enabled();
          delay_ns = g_delay_ns;

          // Generate cluster seed only when cluster targeting is active —
          // it's the per-instrumentation-point selector for which CTA gets
          // delayed and is persisted to config for deterministic replay.
          if (g_delay_cta_target == 1) {
            cluster_seed = generate_cluster_seed();
          }

          // Register the point for config output if kernel_delay_config was created
          if (kernel_delay_config) {
            register_delay_instrumentation_point(kernel_delay_config, instr, delay_ns, enabled, cluster_seed);
          }
        }

        // Dispatcher: 5-branch combinatorial over orthogonal targeting axes.
        // Axes: distribution (g_delay_mode), CTA targeting (g_delay_cta_target),
        // warp targeting (g_delay_warp_mask / g_delay_warpgroup_id). The
        // cluster+warp combo is FATAL'd at env-parse; the assert here is
        // defense-in-depth and a placeholder for a future
        // instrument_delay_cluster_warpgroup.
        const bool warp_active = (g_delay_warp_mask != 0) || (g_delay_warpgroup_id >= 0);
        const bool cluster_active = (g_delay_cta_target == 1);

        if (found && enabled) {
          if (cluster_active && warp_active) {
            assert(false && "cluster + warp targeting unsupported (see CUTRACER_DELAY_* validation)");
          } else if (cluster_active) {
            // Cluster targeting: delay one CTA per cluster (random per point via
            // cluster_seed, or forced via g_cluster_cta_id). Distribution within
            // the selected CTA is fixed (g_delay_mode==0) or per-thread random
            // (g_delay_mode==1) — orthogonal to targeting.
            instrument_cluster_delay_injection(instr, g_delay_min_ns, delay_ns, cluster_seed, g_cluster_cta_id,
                                               /*use_fixed_delay=*/(g_delay_mode == 0) ? 1 : 0);
          } else if (warp_active) {
            // Warp targeting: stall only the warps selected by warp_mask /
            // warpgroup_id within every CTA. Fixed delay per thread (lock-step
            // within selected warps); g_delay_mode=random is silently treated
            // as fixed (a startup WARNING is emitted in parse_delay_config).
            instrument_warpgroup_delay_injection(instr, delay_ns, g_delay_warp_mask, g_delay_warpgroup_id);
          } else if (g_delay_mode == 1) {
            // All-CTA random: per-thread random delay in [min_delay_ns, delay_ns]
            instrument_random_delay_injection(instr, g_delay_min_ns, delay_ns);
          } else {
            // All-CTA fixed: same delay for all threads (original behavior)
            instrument_delay_injection(instr, delay_ns);
          }
        }
        if (found) {
          char warp_tag[64] = {0};
          if (warp_active) {
            uint32_t effective_mask =
                (g_delay_warpgroup_id >= 0) ? (0xFu << (g_delay_warpgroup_id * 4)) : g_delay_warp_mask;
            snprintf(warp_tag, sizeof(warp_tag), " [WARPGROUP mask=0x%x id=%d]", effective_mask, g_delay_warpgroup_id);
          }
          loprintf("Delay injection: kernel=%s pc=0x%lx sass='%s' enabled=%s delay=%u%s%s%s\n", unmangled_name,
                   instr->getOffset(), instr->getSass(), enabled ? "true" : "false", enabled ? delay_ns : 0,
                   cluster_active ? " [CLUSTER]" : "", warp_tag, is_delay_replay_mode() ? " [REPLAY]" : "");
        }
      }
    }

    // Statically identify special instructions for this function:
    // - "clock" (CS2R SR_CLOCKLO) used for histogram region boundaries
    // - "EXIT" used as a candidate signal for warp completion on host side
    // - Categorized instructions (MMA, TMA, SYNC, etc.) for tracing
    for (std::map<int, std::string>::const_iterator it_sass = ctx_state->id_to_sass_map[f].begin();
         it_sass != ctx_state->id_to_sass_map[f].end(); ++it_sass) {
      const char* sass_cstr = it_sass->second.c_str();
      if (strstr(sass_cstr, "CS2R") && strstr(sass_cstr, "SR_CLOCKLO")) {
        ctx_state->clock_opcode_ids[f].insert(it_sass->first);
      }
      // Simple substring detection for EXIT mnemonic (predication handled by extract on host side)
      if (strstr(sass_cstr, "EXIT")) {
        ctx_state->exit_opcode_ids[f].insert(it_sass->first);
      }
      // Detect instruction category and log if matched
      InstrCategory category = detect_instr_category(sass_cstr);
      if (category != InstrCategory::NONE) {
        const char* desc = get_instr_pattern_description(sass_cstr);
        loprintf("%s detected: kernel=%s opcode_id=%d sass='%s' (%s)\n", get_instr_category_name(category),
                 unmangled_name, it_sass->first, sass_cstr, desc ? desc : "");
      }
    }

    /* ============================================ */
  }

  return should_instrument;
}

// Reference code from NVIDIA nvbit mem_trace tool
// Reference code from NVIDIA nvbit mem_trace tool
void init_context_state(CUcontext ctx) {
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];
  ctx_state->recv_thread_done = RecvThreadState::WORKING;
  cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
  ctx_state->channel_host.init((int)ctx_state_map.size() - 1, channel_buffer_size, ctx_state->channel_dev,
                               recv_thread_fun, ctx);
  nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}

/**
 * @brief Build a kernel_metadata JSON object for the trace file header.
 *
 * Combines per-function static attributes (from KernelFuncMetadata) with
 * per-launch dynamic attributes (grid/block dims, dynamic shared memory).
 * Optionally includes the CPU-side call stack captured at kernel launch time.
 *
 * When func and ctx_state are provided, includes a per-instruction table with
 * SASS strings, binary encodings, and register indices — all static data
 * collected once during instrumentation.
 */
static nlohmann::json build_kernel_metadata_json(const KernelFuncMetadata& meta, const KernelDimensions& dims,
                                                 unsigned int dynamic_shmem,
                                                 const std::vector<std::string>& cpu_callstack = {},
                                                 const std::string& callstack_source = "", CUfunction func = nullptr,
                                                 CTXstate* ctx_state = nullptr) {
  auto md = meta.to_json();
  md["type"] = "kernel_metadata";
  md["grid"] = {dims.gridDimX, dims.gridDimY, dims.gridDimZ};
  md["block"] = {dims.blockDimX, dims.blockDimY, dims.blockDimZ};
  md["shmem_dynamic"] = dynamic_shmem;
  if (!cpu_callstack.empty()) {
    md["cpu_callstack"] = cpu_callstack;
    if (!callstack_source.empty()) {
      md["cpu_callstack_source"] = callstack_source;
    }
  }

  // Embed per-instruction static table (SASS, binary encoding, register indices)
  if (func && ctx_state && ctx_state->id_to_sass_map.count(func)) {
    nlohmann::json instr_table = nlohmann::json::object();
    for (const auto& [opcode_id, sass] : ctx_state->id_to_sass_map[func]) {
      nlohmann::json entry;
      entry["sass"] = sass;

      // Binary encoding (hex string, from getSassBinary())
      auto bin_func_it = ctx_state->id_to_sass_binary_map.find(func);
      if (bin_func_it != ctx_state->id_to_sass_binary_map.end()) {
        auto bin_it = bin_func_it->second.find(opcode_id);
        if (bin_it != bin_func_it->second.end()) {
          const auto& bytes = bin_it->second;
          std::string hex;
          hex.reserve(bytes.size() * 2);
          for (uint8_t b : bytes) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", b);
            hex += buf;
          }
          entry["binary"] = hex;
        }
      }

      // Register indices (static data from instrumentation time)
      auto reg_func_it = ctx_state->id_to_reg_indices_map.find(func);
      if (reg_func_it != ctx_state->id_to_reg_indices_map.end()) {
        auto reg_it = reg_func_it->second.find(opcode_id);
        if (reg_it != reg_func_it->second.end()) {
          const auto& ri = reg_it->second;
          if (!ri.reg_indices.empty()) {
            entry["regs"] = ri.reg_indices;
          }
          if (!ri.ureg_indices.empty()) {
            entry["uregs"] = ri.ureg_indices;
          }
        }
      }

      instr_table[std::to_string(opcode_id)] = entry;
    }
    md["instructions"] = instr_table;
  }

  return md;
}

// Reference code from NVIDIA nvbit mem_trace tool
/**
 * @brief Prepares for a kernel launch, conditionally instrumenting and logging.
 *
 * Based on `enter_kernel_launch` from NVIDIA's `mem_trace` example, this
 * function is called just before a kernel launch. It determines if a kernel
 * should be instrumented based on active filters.
 *
 * Key modifications by Meta include making instrumentation and log file creation
 * conditional on the filtering result. This prevents empty log files for
 * skipped kernels.
 *
 * @param ctx The current CUDA context.
 * @param func The kernel function being launched.
 * @param kernel_launch_id A reference to the global kernel launch counter.
 * @param cbid The NVBit callback ID for the CUDA API call.
 * @param params A pointer to the parameters of the CUDA launch call.
 * @param stream_capture True if the launch is part of a CUDA stream capture.
 * @param build_graph True if the launch is part of a manual graph build.
 * @return `true` if the kernel was instrumented, `false` otherwise.
 */
static bool enter_kernel_launch(CUcontext ctx, CUfunction func, uint64_t& kernel_launch_id, nvbit_api_cuda_t cbid,
                                void* params, bool stream_capture = false, bool build_graph = false) {
  // Capture the CPU call stack early, before synchronization, so it reflects
  // the actual application call site that triggered this kernel launch.
  auto [cpu_callstack, callstack_source] = capture_cpu_callstack();

  CTXstate* ctx_state = ctx_state_map[ctx];

  // Cluster targeting only: record cluster dim from launch attrs (including
  // during stream capture) so graph-node launches can later look up the runtime
  // cluster dim Triton set via CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION. The map
  // is only read inside other `g_delay_cta_target == 1` branches, so skipping
  // the mutex/insert when targeting all CTAs saves work on every launch.
  if (g_delay_cta_target == 1) {
    unsigned int cx = 0, cy = 0, cz = 0;
    if (extract_cluster_dim_from_launch_ex(cbid, params, cx, cy, cz)) {
      ObservedClusterDim d{cx, cy, cz};
      pthread_mutex_lock(&g_cluster_obs_mutex);
      g_func_to_observed_cluster[func] = d;
      pthread_mutex_unlock(&g_cluster_obs_mutex);
    }
  }

  // no need to sync during stream capture or manual graph build, since no
  // kernel is actually launched.
  if (!stream_capture && !build_graph) {
    /* Make sure GPU is idle */
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();
  }

  bool should_instrument = instrument_function_if_needed(ctx, func);

  // Get or create per-function metadata (basic fields auto-populated on first access)
  const auto& meta = get_or_create_kernel_func_metadata(func, ctx);
  const char* func_name = meta.unmangled_name.c_str();
  uint64_t pc = meta.func_addr;

  // Per-launch dynamic attributes, populated in the normal launch path below.
  // Declared here so they are accessible when writing kernel_metadata after TraceWriter creation.
  KernelDimensions dims{};
  unsigned int dynamic_shmem = 0;

  // during stream capture or manual graph build, no kernel is launched, so
  // do not set launch argument, do not print kernel info, do not increase
  // grid_launch_id. All these should be done at graph node launch time.
  if (!stream_capture && !build_graph) {
    /* set kernel launch id at launch time */
    nvbit_set_at_launch(ctx, func, (uint64_t)kernel_launch_id);

    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
      loprintf(
          "CUTracer: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
          "Kernel name %s - kernel hash 0x%s - kernel launch id %ld - grid size %d,%d,%d "
          "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
          "id %ld\n",
          (uint64_t)ctx, pc, func_name, meta.kernel_checksum.c_str(), kernel_launch_id, p->config->gridDimX,
          p->config->gridDimY, p->config->gridDimZ, p->config->blockDimX, p->config->blockDimY, p->config->blockDimZ,
          meta.nregs, meta.shmem_static_nbytes + p->config->sharedMemBytes, (uint64_t)p->config->hStream);
    } else {
      cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
      loprintf(
          "CUTracer: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
          "Kernel name %s - kernel hash 0x%s - kernel launch id %ld - grid size %d,%d,%d "
          "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
          "id %ld\n",
          (uint64_t)ctx, pc, func_name, meta.kernel_checksum.c_str(), kernel_launch_id, p->gridDimX, p->gridDimY,
          p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ, meta.nregs,
          meta.shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream);
    }

    // Cluster targeting: log cluster dimensions once per function (see
    // log_cluster_info_once for details).
    if (should_instrument && g_delay_cta_target == 1) {
      unsigned int dyn_cx = 0, dyn_cy = 0, dyn_cz = 0;
      extract_cluster_dim_from_launch_ex(cbid, params, dyn_cx, dyn_cy, dyn_cz);
      log_cluster_info_once(ctx, func, func_name, dyn_cx, dyn_cy, dyn_cz);
    }

    // For histogram analysis, we need to map the kernel launch ID back to its
    // function and context to access the correct SASS and clock opcode maps.
    // We also track the iteration count for unique file naming.
    uint32_t current_iter = kernel_iter_map[func];
    kernel_launch_to_func_map[kernel_launch_id] = {ctx, func};
    kernel_launch_to_iter_map[kernel_launch_id] = current_iter;

    // Store kernel dimensions and dynamic shared memory for warp statistics tracking
    // (dims and dynamic_shmem declared at function scope for metadata writing)
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
      dims = {p->config->gridDimX,  p->config->gridDimY,  p->config->gridDimZ,
              p->config->blockDimX, p->config->blockDimY, p->config->blockDimZ};
      dynamic_shmem = p->config->sharedMemBytes;
    } else {
      cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
      dims = {p->gridDimX, p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ};
      dynamic_shmem = p->sharedMemBytes;
    }
    kernel_launch_to_dimensions_map[kernel_launch_id] = dims;

    // Write structured kernel launch event (applies to ALL kernels, not just instrumented)
    if (kernel_events_mode != KernelEventsMode::DISABLED) {
      uint64_t stream_id = 0;
      if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
        stream_id = (uint64_t)((cuLaunchKernelEx_params*)params)->config->hStream;
      } else {
        stream_id = (uint64_t)((cuLaunchKernel_params*)params)->hStream;
      }
      write_kernel_launch_event(meta, dims, dynamic_shmem, stream_id, kernel_launch_id, cpu_callstack,
                                callstack_source);
    }

    // increment kernel launch id for next launch
    // kernel id can be changed here, since nvbit_set_at_launch() has copied
    // its value above.
    kernel_launch_id++;
  }

  if (should_instrument && !stream_capture && !build_graph) {
    // Record kernel start time at launch (before any GPU data arrives).
    // This enables no-data timeout detection even for immediate-silence hangs
    // where the kernel deadlocks from the very first instruction and the
    // recv thread never receives any trace data.
    ctx_state->kernel_start_time = time(nullptr);

    // Create TraceWriter for each kernel launch (per-launch trace files)
    std::string base_filename = generate_kernel_log_basename(ctx, func, kernel_iter_map[func]++, meta.kernel_checksum);
    uint64_t current_launch_id = kernel_launch_id - 1;  // kernel_launch_id was already incremented

    {
      std::unique_lock<std::shared_mutex> lock(ctx_state->writers_mutex);
      ctx_state->trace_writers[current_launch_id] = new TraceWriter(base_filename, trace_format);
    }

    // Initialize trace_index for this kernel
    ctx_state->trace_index_by_kernel[current_launch_id] = 0;

    // Write kernel_metadata as the first JSON line in the trace file.
    auto metadata =
        build_kernel_metadata_json(meta, dims, dynamic_shmem, cpu_callstack, callstack_source, func, ctx_state);
    ctx_state->trace_writers[current_launch_id]->write_metadata(metadata);

    loprintf_v("Created TraceWriter for launch_id %lu, mode %d, file: %s\n", current_launch_id, trace_format,
               base_filename.c_str());
  }

  nvbit_enable_instrumented(ctx, func, should_instrument);
  return should_instrument;
}

// the function is only called for non cuda graph launch cases.
static void leave_kernel_launch(CUcontext ctx, CTXstate* ctx_state, uint64_t& grid_launch_id) {
  // make sure user kernel finishes to avoid deadlock
  cudaDeviceSynchronize();
  /* push a flush channel kernel using preloaded function */
  void* flush_args[] = {&ctx_state->channel_dev};
  nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0, nullptr, flush_args, nullptr);

  /* Make sure GPU is idle */
  cudaDeviceSynchronize();
  CUDA_CHECK_LAST_ERROR();

  // Flush current launch's TraceWriter buffer (but don't close it yet)
  // The recv_thread will close writers when it detects launch_id changes
  uint64_t current_launch_id = grid_launch_id - 1;
  {
    std::shared_lock<std::shared_mutex> lock(ctx_state->writers_mutex);
    auto it = ctx_state->trace_writers.find(current_launch_id);
    if (it != ctx_state->trace_writers.end() && it->second) {
      it->second->flush();
    }
  }
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char* name, void* params,
                         CUresult* pStatus) {
  pthread_mutex_lock(&cuda_event_mutex);

  /* we prevent re-entry on this callback when issuing CUDA functions inside
   * this function */
  if (skip_callback_flag) {
    pthread_mutex_unlock(&cuda_event_mutex);
    return;
  }
  skip_callback_flag = true;

  CTXstate* ctx_state = ctx_state_map[ctx];

  switch (cbid) {
    // Identify all the possible CUDA launch events without stream
    // parameters, they will not get involved with cuda graph
    case API_CUDA_cuLaunch:
    case API_CUDA_cuLaunchGrid: {
      cuLaunch_params* p = (cuLaunch_params*)params;
      CUfunction func = p->f;
      if (!is_exit) {
        ctx_state->need_sync = true;
        enter_kernel_launch(ctx, func, global_kernel_launch_id, cbid, params);
      } else {
        leave_kernel_launch(ctx, ctx_state, global_kernel_launch_id);
      }
    } break;
    // To support kernel launched by cuda graph (in addition to existing kernel
    // launche method), we need to do:
    //
    // 1. instrument kernels at cudaGraphAddKernelNode event. This is for cases
    // that kernels are manually added to a cuda graph.
    // 2. distinguish captured kernels when kernels are recorded to a graph
    // using stream capture. cudaStreamIsCapturing() tells us whether a stream
    // is capturiong.
    // 3. per-kernel instruction counters, since cuda graph can launch multiple
    // kernels at the same time.
    //
    // Three cases:
    //
    // 1. original kernel launch:
    //     1a. for any kernel launch without using a stream, we instrument it
    //     before it is launched, call cudaDeviceSynchronize after it is
    //     launched and read the instruction counter of the kernel.
    //     1b. for any kernel launch using a stream, but the stream is not
    //     capturing, we do the same thing as 1a.
    //
    //  2. cuda graph using stream capturing: if a kernel is launched in a
    //  stream and the stream is capturing. We instrument the kernel before it
    //  is launched and do nothing after it is launched, because the kernel is
    //  not running until cudaGraphLaunch. Instead, we issue a
    //  cudaStreamSynchronize after cudaGraphLaunch is done and reset the
    //  instruction counters, since a cloned graph might be launched afterwards.
    //
    //  3. cuda graph manual: we instrument the kernel added by
    //  cudaGraphAddKernelNode and do the same thing for cudaGraphLaunch as 2.
    //
    // The above method should handle most of cuda graph launch cases.
    // kernel launches with stream parameter, they can be used for cuda graph
    case API_CUDA_cuLaunchKernel_ptsz:
    case API_CUDA_cuLaunchKernel:
    case API_CUDA_cuLaunchCooperativeKernel:
    case API_CUDA_cuLaunchCooperativeKernel_ptsz:
    case API_CUDA_cuLaunchKernelEx:
    case API_CUDA_cuLaunchKernelEx_ptsz:
    case API_CUDA_cuLaunchGridAsync: {
      CUfunction func;
      CUstream hStream;

      if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
        cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
        func = p->f;
        hStream = p->config->hStream;
      } else if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel ||
                 cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz || cbid == API_CUDA_cuLaunchCooperativeKernel) {
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
        func = p->f;
        hStream = p->hStream;
      } else {
        cuLaunchGridAsync_params* p = (cuLaunchGridAsync_params*)params;
        func = p->f;
        hStream = p->hStream;
      }

      cudaStreamCaptureStatus streamStatus;
      /* check if the stream is capturing, if yes, do not sync */
      CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
      if (!is_exit) {
        bool stream_capture = (streamStatus == cudaStreamCaptureStatusActive);
        if (!stream_capture) {
          ctx_state->need_sync = true;
        }
        enter_kernel_launch(ctx, func, global_kernel_launch_id, cbid, params, stream_capture);
      } else {
        if (streamStatus != cudaStreamCaptureStatusActive) {
          loprintf_vl(1, "kernel %s not captured by cuda graph\n", nvbit_get_func_name(ctx, func));
          leave_kernel_launch(ctx, ctx_state, global_kernel_launch_id);
        } else {
          loprintf_vl(1, "kernel %s captured by cuda graph\n", nvbit_get_func_name(ctx, func));
        }
      }
    } break;
    case API_CUDA_cuGraphAddKernelNode: {
      cuGraphAddKernelNode_params* p = (cuGraphAddKernelNode_params*)params;
      CUfunction func = p->nodeParams->func;

      if (!is_exit) {
        // cuGraphAddKernelNode_params->nodeParams is the same as
        // cuLaunchKernel_params up to sharedMemBytes
        enter_kernel_launch(ctx, func, global_kernel_launch_id, cbid, (void*)p->nodeParams, false, true);
      }
    } break;
    case API_CUDA_cuGraphLaunch: {
      // if we are exiting a cuda graph launch:
      // Wait until the graph is completed using
      // cudaStreamSynchronize()
      if (is_exit) {
        cuGraphLaunch_params* p = (cuGraphLaunch_params*)params;

        CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
        CUDA_CHECK_LAST_ERROR();
        /* push a flush channel kernel using preloaded function */
        void* flush_args[] = {&ctx_state->channel_dev};
        nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0, p->hStream, flush_args, nullptr);
        CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
        CUDA_CHECK_LAST_ERROR();
      }

    } break;
    default:
      break;
  };

  skip_callback_flag = false;
  pthread_mutex_unlock(&cuda_event_mutex);
}

// Reference NVIDIA record_reg_vals example
void nvbit_tool_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  init_context_state(ctx);
  pthread_mutex_unlock(&mutex);
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_ctx_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  if (verbose) {
    printf("CUTracer: STARTING CONTEXT %p\n", ctx);
  }
  assert(ctx_state_map.find(ctx) == ctx_state_map.end());
  CTXstate* ctx_state = new CTXstate;
  ctx_state_map[ctx] = ctx_state;

  nvbit_load_tool_module(ctx, (const void*)flush_channel_bin, &ctx_state->tool_module);
  assert(ctx_state->tool_module != nullptr);
  nvbit_find_function_by_name(ctx, ctx_state->tool_module, "flush_channel", &ctx_state->flush_channel_func);
  assert(ctx_state->flush_channel_func != nullptr);

  pthread_mutex_unlock(&mutex);
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_ctx_term(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  skip_callback_flag = true;
  loprintf_v("CUTracer: TERMINATING CONTEXT %p\n", ctx);
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  /* Final flush: ensure last kernel's data is not lost */
  if (ctx_state->need_sync) {
    void* flush_args[] = {&ctx_state->channel_dev};
    nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0, nullptr, flush_args, nullptr);
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();
  }

  /* Notify receiver thread and wait for receiver thread to
   * notify back */
  if (ctx_state->recv_thread_done != RecvThreadState::INIT) {
    ctx_state->recv_thread_done = RecvThreadState::STOP;
    while (ctx_state->recv_thread_done != RecvThreadState::FINISHED);
  }

  // Cleanup all TraceWriters (fallback - most should be closed by recv_thread)
  {
    std::unique_lock<std::shared_mutex> lock(ctx_state->writers_mutex);
    for (auto& [launch_id, writer] : ctx_state->trace_writers) {
      if (writer) {
        loprintf_v("Cleaning up remaining writer for launch_id %lu\n", launch_id);
        writer->flush();
        delete writer;
      }
    }
    ctx_state->trace_writers.clear();
  }

  // Clean up any remaining kernel mapping entries
  // (in case there were kernels launched but no data received)
  kernel_launch_to_func_map.clear();
  kernel_launch_to_iter_map.clear();

  if (ctx_state->channel_dev != nullptr) {
    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
  }
  skip_callback_flag = false;
  ctx_state_map.erase(ctx);
  delete ctx_state;

  // Global cleanup: only when the last context is destroyed.
  // g_kernel_events_writer is shared across all contexts, so we must not
  // delete it while other contexts may still record events.
  if (ctx_state_map.empty()) {
    if (g_kernel_events_writer) {
      g_kernel_events_writer->flush();
      delete g_kernel_events_writer;
      g_kernel_events_writer = nullptr;
      callstack_seen.clear();
    }

    cleanup_log_handle();
    finalize_delay_config();
  }

  pthread_mutex_unlock(&mutex);
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_graph_node_launch(CUcontext ctx, CUfunction func, CUstream stream, uint64_t launch_handle) {
  func_config_t config = {0};
  const char* func_name = nvbit_get_func_name(ctx, func);
  uint64_t pc = nvbit_get_func_addr(ctx, func);

  pthread_mutex_lock(&mutex);

  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  nvbit_set_at_launch(ctx, func, (uint64_t)global_kernel_launch_id, stream, launch_handle);
  nvbit_get_func_config(ctx, func, &config);

  loprintf(
      "CUTracer: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
      "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
      "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
      "id %ld\n",
      (uint64_t)ctx, pc, func_name, global_kernel_launch_id, config.gridDimX, config.gridDimY, config.gridDimZ,
      config.blockDimX, config.blockDimY, config.blockDimZ, config.num_registers,
      config.shmem_static_nbytes + config.shmem_dynamic_nbytes, (uint64_t)stream);

  // Cluster targeting: log cluster dims once per function. Look up the
  // runtime cluster dim observed at the most recent cuLaunchKernelEx for this
  // CUfunction (including the stream-capture launch that built this graph
  // node). Falls back to the kernel-declared __cluster_dims__ inside the
  // helper if neither source has dynamic info.
  if (g_delay_cta_target == 1) {
    unsigned int dyn_cx = 0, dyn_cy = 0, dyn_cz = 0;
    pthread_mutex_lock(&g_cluster_obs_mutex);
    auto it = g_func_to_observed_cluster.find(func);
    if (it != g_func_to_observed_cluster.end()) {
      dyn_cx = it->second.x;
      dyn_cy = it->second.y;
      dyn_cz = it->second.z;
    }
    pthread_mutex_unlock(&g_cluster_obs_mutex);
    log_cluster_info_once(ctx, func, func_name, dyn_cx, dyn_cy, dyn_cz);
  }

  // Store kernel launch mapping for graph node launches
  kernel_launch_to_func_map[global_kernel_launch_id] = {ctx, func};
  kernel_launch_to_iter_map[global_kernel_launch_id] = kernel_iter_map[func];

  // Store kernel dimensions
  KernelDimensions dims = {config.gridDimX,  config.gridDimY,  config.gridDimZ,
                           config.blockDimX, config.blockDimY, config.blockDimZ};
  kernel_launch_to_dimensions_map[global_kernel_launch_id] = dims;

  // Capture the CPU call stack for graph node launches as well
  auto [cpu_callstack, callstack_source] = capture_cpu_callstack();

  // Write structured kernel launch event for graph node launches
  if (kernel_events_mode != KernelEventsMode::DISABLED) {
    const auto& meta_ev = get_or_create_kernel_func_metadata(func, ctx);
    write_kernel_launch_event(meta_ev, dims, config.shmem_dynamic_nbytes, (uint64_t)stream, global_kernel_launch_id,
                              cpu_callstack, callstack_source);
  }

  // Create TraceWriter for this graph node launch only if instrumentation is enabled
  // Otherwise, trace files would be empty (no data to write)
  if (has_any_instrumentation_enabled()) {
    const auto& meta = get_or_create_kernel_func_metadata(func, ctx);
    std::string base_filename = generate_kernel_log_basename(ctx, func, kernel_iter_map[func]++, meta.kernel_checksum);
    {
      std::unique_lock<std::shared_mutex> lock(ctx_state->writers_mutex);
      ctx_state->trace_writers[global_kernel_launch_id] = new TraceWriter(base_filename, trace_format);
    }
    ctx_state->trace_index_by_kernel[global_kernel_launch_id] = 0;

    // Write kernel_metadata for graph node launch trace files
    auto metadata = build_kernel_metadata_json(meta, dims, config.shmem_dynamic_nbytes, cpu_callstack, callstack_source,
                                               func, ctx_state);
    ctx_state->trace_writers[global_kernel_launch_id]->write_metadata(metadata);

    loprintf_v("Created TraceWriter for graph node launch_id %lu, file: %s\n", global_kernel_launch_id,
               base_filename.c_str());
  }

  // grid id can be changed here, since nvbit_set_at_launch() has copied its
  // value above.
  global_kernel_launch_id++;
  pthread_mutex_unlock(&mutex);
}

// Reference code from NVIDIA nvbit mem_trace tool with Meta modifications for
// env config
void nvbit_at_init() {
  // Initialize configuration from environment variables
  init_config_from_env();
  // Initialize delay JSON config for export (also loads replay config if specified)
  init_delay_json_config();
  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&mutex, &attr);

  pthread_mutex_init(&cuda_event_mutex, &attr);

  // Create kernel events writer if enabled
  if (kernel_events_mode != KernelEventsMode::DISABLED) {
    // Generate timestamped filename in output_dir
    char time_buf[32];
    time_t now = time(nullptr);
    struct tm tm_info;
    localtime_r(&now, &tm_info);
    strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", &tm_info);

    std::string events_basename;
    if (!output_dir.empty()) {
      events_basename = output_dir + (output_dir.back() != '/' ? "/" : "") + "cutracer_kernel_events_" + time_buf;
    } else {
      events_basename = std::string("cutracer_kernel_events_") + time_buf;
    }
    // Kernel events are structured JSON records — always use NDJSON format
    // regardless of CUTRACER_TRACE_FORMAT (text mode would silently drop them,
    // zstd adds overhead for small files).
    const int kernel_events_format = 2;  // NDJSON (uncompressed)
    if (trace_format == 0) {
      loprintf(
          "WARNING: CUTRACER_TRACE_FORMAT=text but CUTRACER_KERNEL_EVENTS is enabled; "
          "kernel events will use NDJSON format\n");
    }
    g_kernel_events_writer = new TraceWriter(events_basename, kernel_events_format);
    loprintf("Kernel events writer created: %s (mode: %s)\n", events_basename.c_str(),
             kernel_events_mode == KernelEventsMode::DEDUP     ? "dedup"
             : kernel_events_mode == KernelEventsMode::FULL    ? "full"
             : kernel_events_mode == KernelEventsMode::NOSTACK ? "nostack"
                                                               : "unknown");
  }
}
