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
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* contains definition of the reg_info_t and mem_access_t structure */
#include "common.h"
#include "cubin_identity.h"
#include "cuda_callback_guard.h"

/* analysis functionality */
#include "analysis.h"

/* precompiled flush_channel fatbin (bin2c byte array) */
#include "tool_func/flush_channel.c"

/* instrumentation functionality */
#include "instrument.h"
#include "instrument_metadata.h"
#include "tma_registers.h"

/* env config */
#include "env_config.h"
#include "python_callstack.h"
#include "trace_writer.h"

/* logging functionality */
#include "log.h"

/* delay configuration */
#include "delay_choice.h"
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

/* The kernel launch is identified by a global id */
std::atomic<uint64_t> global_kernel_launch_id{0};
std::mutex kernel_launch_state_mutex;

/* Kernel events writer for structured launch event recording */
static TraceWriter* g_kernel_events_writer = nullptr;

/* Callstack dedup table: FNV-1a hash → already emitted */
static std::unordered_map<uint64_t, bool> callstack_seen;

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

enum class ClusterDimSource {
  UNKNOWN,
  LAUNCH_ATTRIBUTE,
  REQUIRED_FUNCTION_ATTRIBUTE,
};

struct ClusterLaunchInfo {
  unsigned int x = 1;
  unsigned int y = 1;
  unsigned int z = 1;
  ClusterDimSource source = ClusterDimSource::UNKNOWN;

  unsigned int size() const {
    return x * y * z;
  }
};

static const char* cluster_dim_source_name(ClusterDimSource source) {
  switch (source) {
    case ClusterDimSource::LAUNCH_ATTRIBUTE:
      return "launch_attribute";
    case ClusterDimSource::REQUIRED_FUNCTION_ATTRIBUTE:
      return "required_function_attribute";
    case ClusterDimSource::UNKNOWN:
      return "unknown";
  }
  return "unknown";
}

static ClusterLaunchInfo resolve_cluster_launch_info(CUfunction func, bool has_dynamic_dim = false,
                                                     unsigned int dynamic_x = 0, unsigned int dynamic_y = 0,
                                                     unsigned int dynamic_z = 0) {
  if (has_dynamic_dim && dynamic_x > 0 && dynamic_y > 0 && dynamic_z > 0) {
    return {dynamic_x, dynamic_y, dynamic_z, ClusterDimSource::LAUNCH_ATTRIBUTE};
  }

  int required_x = 0, required_y = 0, required_z = 0;
  if (cuFuncGetAttribute(&required_x, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, func) != CUDA_SUCCESS) required_x = 0;
  if (cuFuncGetAttribute(&required_y, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, func) != CUDA_SUCCESS) required_y = 0;
  if (cuFuncGetAttribute(&required_z, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, func) != CUDA_SUCCESS) required_z = 0;
  if (required_x > 0 || required_y > 0 || required_z > 0) {
    return {required_x > 0 ? (unsigned int)required_x : 1, required_y > 0 ? (unsigned int)required_y : 1,
            required_z > 0 ? (unsigned int)required_z : 1, ClusterDimSource::REQUIRED_FUNCTION_ATTRIBUTE};
  }
  return {};
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
                                      const ClusterLaunchInfo& cluster, unsigned int dynamic_shmem, uint64_t stream_id,
                                      uint64_t launch_id, const std::vector<std::string>& callstack,
                                      const std::string& callstack_source) {
  if (!g_kernel_events_writer || kernel_events_mode == KernelEventsMode::DISABLED) {
    return;
  }

  nlohmann::json event;
  event["type"] = "kernel_launch";
  event["kernel_launch_id"] = launch_id;
  event["kernel_name"] = meta.unmangled_name;
  event["mangled_name"] = meta.mangled_name;
  event["kernel_checksum"] = meta.kernel_checksum;
  event["grid"] = {dims.gridDimX, dims.gridDimY, dims.gridDimZ};
  event["block"] = {dims.blockDimX, dims.blockDimY, dims.blockDimZ};
  event["cluster_dim"] = {cluster.x, cluster.y, cluster.z};
  event["cluster_size"] = cluster.size();
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
std::unordered_map<CUfunction, ClusterLaunchInfo> g_func_to_observed_cluster;
pthread_mutex_t g_cluster_obs_mutex = PTHREAD_MUTEX_INITIALIZER;

// Legacy trace names are kernel_<checksum>_iter<n>_pid<pid>_<name>. Number
// launches by checksum and name rather than by CUfunction, so other contexts and
// reloaded modules in this process cannot overwrite an earlier trace. Callers
// hold kernel_launch_state_mutex.
static std::map<std::pair<std::string, std::string>, uint32_t> kernel_iter_map;

static uint32_t next_kernel_iteration(const KernelFuncMetadata& meta, bool selected) {
  auto& next = kernel_iter_map[{meta.kernel_checksum, meta.mangled_name}];
  return selected ? next++ : next;
}

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

static bool should_instrument_instr_by_trace_category(Instr* instr) {
  if (!has_trace_category_filter_enabled()) return true;
  InstrCategory category = detect_instr_category(instr->getSass());
  return should_instrument_trace_category(category);
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

#include "runtime_hooks.h"

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
  if (!inserted && it->second.owner_context != ctx) {
    it->second = KernelFuncMetadata{};
    already_instrumented.erase(func);
    inserted = true;
  }
  if (inserted) {
    it->second.owner_context = ctx;
    // A missing module association remains unavailable to the identity path.
    const auto module_status = cuFuncGetModule(&it->second.owner_module, func);
    if (module_status != CUDA_SUCCESS) {
      it->second.owner_module = nullptr;
      if (runtime_identity.active()) {
        runtime_identity_error("cuFuncGetModule failed: " + std::to_string(static_cast<int>(module_status)));
      }
    }
    if (reused_function_handle(ctx, func)) {
      loprintf("WARNING: CUfunction %p was reused after a module unload in context %p; its launches are not traced\n",
               (void*)func, (void*)ctx);
    }
    it->second.unmangled_name = nvbit_get_func_name(ctx, func, false);
    it->second.func_addr = nvbit_get_func_addr(ctx, func);
    CUDA_SAFECALL(cuFuncGetAttribute(&it->second.nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));
    CUDA_SAFECALL(cuFuncGetAttribute(&it->second.shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));
    it->second.sm_family = nvbit_get_sm_family(ctx);
    ensure_kernel_checksum(it->second, ctx, func);
  }
  return it->second;
}

static void dump_kernel_cubin(CUcontext ctx, CUfunction func, KernelFuncMetadata& meta) {
  const std::string prefix = output_dir.empty() ? "" : output_dir + (output_dir.back() == '/' ? "" : "/");
  const std::string name = meta.mangled_name.substr(0, 150);
  std::string temporary_dir = prefix + ".cutracer-cubin-XXXXXX";
  if (mkdtemp(temporary_dir.data()) == nullptr) {
    loprintf("ERROR: could not create cubin snapshot for %s\n", meta.mangled_name.c_str());
    return;
  }
  // Reserve a private directory, then let NVBit create the file with the
  // caller's umask. mkstemp would leave the published cubin at mode 0600.
  const std::string temporary = temporary_dir + "/snapshot.cubin";
  // Hash the fresh snapshot before publishing it; a failed dump cannot reuse
  // bytes from an earlier capture.
  const bool dumped = nvbit_dump_cubin(ctx, func, temporary.c_str());
  const std::string digest = dumped ? cubin_file_sha256(temporary) : "";
  if (digest.empty()) {
    loprintf("ERROR: could not hash dumped cubin %s\n", temporary.c_str());
    unlink(temporary.c_str());
    rmdir(temporary_dir.c_str());
    return;
  }
  // The SASS checksum can stay identical when embedded PTX contracts differ.
  const std::string filename = "kernel_" + meta.kernel_checksum + "_" + digest + "_" + name + ".cubin";
  const std::string path = prefix + filename;
  if (rename(temporary.c_str(), path.c_str()) != 0) {
    loprintf("ERROR: could not preserve cubin snapshot %s\n", path.c_str());
    unlink(temporary.c_str());
    rmdir(temporary_dir.c_str());
    return;
  }
  rmdir(temporary_dir.c_str());
  // Artifacts are colocated with the trace and remain portable after upload.
  meta.cubin_path = filename;
  meta.cubin_sha256 = digest;
}

static bool matches_exact_kernel_selector(CUcontext ctx, CUfunction func) {
  const auto& selector = exact_kernel_selector();
  if (!selector.enabled()) return true;
  const auto& meta = get_or_create_kernel_func_metadata(func, ctx);
  return selector.matches(meta.mangled_name, meta.unmangled_name, meta.kernel_checksum);
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
  if (!has_any_instrumentation_enabled() || !matches_exact_kernel_selector(ctx, func)) {
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

  // Exact selection applies to the launch root, not its device callees.
  const bool exact_selection = exact_kernel_selector().enabled();
  bool should_instrument = exact_selection || kernel_filters.empty();
  bool delay_config_grew = false;
  /* iterate on function */
  for (auto f : related_functions) {
    const char* unmangled_name = nvbit_get_func_name(ctx, f, false);
    const char* mangled_name = nvbit_get_func_name(ctx, f, true);

    // Check if the current function matches any kernel filter
    bool kernel_filter_matched = exact_selection || kernel_filters.empty();
    if (!exact_selection) {
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
    }

    // Skip if filter not matched or already instrumented
    if (!kernel_filter_matched || !already_instrumented.insert(f).second) {
      continue;
    }

    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
    loprintf_v("Inspecting kernel %s at address 0x%lx\n", nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));

    // Get or create per-function metadata (checksum already computed by helper)
    auto& meta = get_or_create_kernel_func_metadata(f, ctx);
    meta.instruction_count = instrs.size();
    loprintf_v("Kernel checksum for %s: %s\n", meta.mangled_name.c_str(), meta.kernel_checksum.c_str());

    if (dump_cubin) {
      dump_kernel_cubin(ctx, f, meta);
    }

    // Create kernel delay config if dump path is set (for exporting to JSON)
    // Note: dump mode and replay mode are mutually exclusive
    KernelDelayInjectConfig* kernel_delay_config = nullptr;
    if (!delay_dump_path.empty() && is_instrument_type_enabled(InstrumentType::RANDOM_DELAY)) {
      assert(!is_delay_replay_mode() && "Dump and replay modes are mutually exclusive");
      // Reuse the kernel_checksum already computed above for trace writer
      kernel_delay_config = create_kernel_delay_config(mangled_name, meta.kernel_checksum);
      delay_config_grew = true;
    }

    // Get replay instrumentation points once for replay mode
    const std::map<uint64_t, DelayInstrumentationPoint>* replay_points = nullptr;
    if (is_delay_replay_mode() && is_instrument_type_enabled(InstrumentType::RANDOM_DELAY)) {
      assert(delay_dump_path.empty() && "Dump and replay modes are mutually exclusive");
      // Reuse the kernel_checksum already computed above for trace writer
      replay_points = get_replay_instrumentation_points(mangled_name, meta.kernel_checksum);
    }

    // Built here and published under instr_tables_mutex once complete.
    std::map<int, std::string> sass_by_opcode;
    std::map<int, std::vector<uint8_t>> binary_by_opcode;
    std::map<int, RegIndices> reg_indices_by_opcode;
    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
      const uint32_t instr_index = cnt++;
      if (instr_index < instr_begin_interval || instr_index >= instr_end_interval) {
        continue;
      }
      if (verbose) {
        instr->printDecoded();
      }

      OperandLists operands;
      OperandContext op_ctx;  // Context for implicit register collection
      int mref_idx = 0;
      int opcode_id = instr->getIdx();
      const bool trace_instruction_selected =
          should_instrument_instr_by_trace_category(instr) && should_instrument_trace_pc(instr->getOffset());
      sass_by_opcode[opcode_id] = std::string(instr->getSass());

      // Collect SASS binary encoding (nvbit 1.8+)
      {
        std::vector<uint8_t> binary;
        instr->getSassBinary([](uint8_t b, void* ud) { static_cast<std::vector<uint8_t>*>(ud)->push_back(b); },
                             &binary);
        if (!binary.empty()) {
          binary_by_opcode[opcode_id] = std::move(binary);
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
            ++meta.instrumentation_errors;
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
          if (trace_instruction_selected && is_instrument_type_enabled(InstrumentType::MEM_ADDR_TRACE)) {
            instrument_memory_addr_trace(instr, opcode_id, ctx_state, mref_idx);
          }
          // Memory value tracing (captures addresses + values at IPOINT_AFTER)
          if (trace_instruction_selected && is_instrument_type_enabled(InstrumentType::MEM_VALUE_TRACE)) {
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
          collect_tma_uniform_registers(*op, meta.sm_family, operands.ureg_nums);
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
        reg_indices_by_opcode[opcode_id] = std::move(reg_idx);
      }

      // Choose instrumentation based on enabled types (only if this instruction passes category filter)
      if (trace_instruction_selected && should_instrument_instr_by_category(instr)) {
        if (is_instrument_type_enabled(InstrumentType::OPCODE_ONLY)) {
          // Lightweight instrumentation for instruction histogram analysis.
          // This sends minimal data (opcode_id, warp_id) to the CPU, reducing
          // overhead.
          instrument_opcode_only(instr, opcode_id, ctx_state);
        } else if (is_instrument_type_enabled(InstrumentType::REG_TRACE)) {
          // Full register tracing.
          if (instrument_register_trace(instr, opcode_id, ctx_state, operands)) {
            ++meta.reg_trace_instruction_count;
          } else {
            ++meta.instrumentation_errors;
          }
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
          enabled = cutracer::delay_point_enabled(g_delay_seed, meta.kernel_checksum, instr->getOffset(),
                                                  g_delay_enable_prob);
          delay_ns = g_delay_ns;

          // Generate cluster seed only when cluster targeting is active —
          // it's the per-instrumentation-point selector for which CTA gets
          // delayed and is persisted to config for deterministic replay.
          if (g_delay_cta_target == 1) {
            cluster_seed = cutracer::delay_point_cluster_seed(g_delay_seed, meta.kernel_checksum, instr->getOffset());
          }

          // Register the point for config output if kernel_delay_config was created
          if (kernel_delay_config) {
            register_delay_instrumentation_point(kernel_delay_config, instr, delay_ns, enabled, cluster_seed);
          }
        }

        // Dispatcher: 5-branch combinatorial over orthogonal targeting axes.
        // Axes: distribution (g_delay_mode), CTA targeting (g_delay_cta_target),
        // warp targeting (g_delay_warp_mask / g_delay_warpgroup_id).
        const bool warp_active = (g_delay_warp_mask != 0) || (g_delay_warpgroup_id >= 0);
        const bool cluster_active = (g_delay_cta_target == 1);
        const uint32_t warp_mask = effective_delay_warp_mask(g_delay_warp_mask, g_delay_warpgroup_id);

        if (found && enabled) {
          if (cluster_active && warp_active) {
            // Cluster + warp targeting: stall only the selected warps of the
            // selected CTA of each cluster, with a fixed delay (warp lock-step).
            instrument_cluster_warp_delay_injection(instr, delay_ns, warp_mask, cluster_seed, g_cluster_cta_id);
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
            snprintf(warp_tag, sizeof(warp_tag), " [WARPGROUP mask=0x%x id=%d]", warp_mask, g_delay_warpgroup_id);
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
    std::unordered_set<int> clock_ids;
    std::unordered_set<int> exit_ids;
    for (std::map<int, std::string>::const_iterator it_sass = sass_by_opcode.begin(); it_sass != sass_by_opcode.end();
         ++it_sass) {
      const char* sass_cstr = it_sass->second.c_str();
      if (strstr(sass_cstr, "CS2R") && strstr(sass_cstr, "SR_CLOCKLO")) {
        clock_ids.insert(it_sass->first);
      }
      // Simple substring detection for EXIT mnemonic (predication handled by extract on host side)
      if (strstr(sass_cstr, "EXIT")) {
        exit_ids.insert(it_sass->first);
      }
      // Detect instruction category and log if matched
      InstrCategory category = detect_instr_category(sass_cstr);
      if (category != InstrCategory::NONE) {
        const char* desc = get_instr_pattern_description(sass_cstr);
        loprintf("%s detected: kernel=%s opcode_id=%d sass='%s' (%s)\n", get_instr_category_name(category),
                 unmangled_name, it_sass->first, sass_cstr, desc ? desc : "");
      }
    }
    {
      std::unique_lock<std::shared_mutex> tables(ctx_state->instr_tables_mutex);
      ctx_state->id_to_sass_map[f] = std::move(sass_by_opcode);
      ctx_state->id_to_sass_binary_map[f] = std::move(binary_by_opcode);
      ctx_state->id_to_reg_indices_map[f] = std::move(reg_indices_by_opcode);
      ctx_state->clock_opcode_ids[f] = std::move(clock_ids);
      ctx_state->exit_opcode_ids[f] = std::move(exit_ids);
    }

    /* ============================================ */
  }

  if (delay_config_grew) {
    save_delay_config();
  }
  return should_instrument;
}

// NVBit identifies tool threads by the hash of their std::thread::id, as
// ChannelHost::get_thread_id() does for the receiver.
static uint64_t nvbit_thread_key(std::thread::id id) {
  return std::hash<std::thread::id>{}(id);
}

// The drain worker issues the context's post-launch CUDA calls. It registers
// as an NVBit tool thread, so those calls trigger no callbacks; makes the
// context current; and switches to relaxed capture mode, so its event waits
// and private-stream syncs do not invalidate another thread's global-mode
// capture. Returns false if the worker cannot drain for this context.
static bool start_drain_thread(CUcontext ctx, CTXstate* ctx_state) {
  nvbit_set_tool_thread(nvbit_thread_key(std::this_thread::get_id()));
  cutracer::ScopedCudaCallbackSuppression suppress_callbacks;
  if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) return false;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  if (cudaThreadExchangeStreamCaptureMode(&mode) != cudaSuccess) return false;
  return cudaStreamCreateWithFlags(&ctx_state->drain_stream, cudaStreamNonBlocking) == cudaSuccess;
}

static void stop_drain_thread(CTXstate* ctx_state) {
  cutracer::ScopedCudaCallbackSuppression suppress_callbacks;
  if (ctx_state->drain_stream) {
    cudaStreamDestroy(ctx_state->drain_stream);
    ctx_state->drain_stream = nullptr;
  }
  nvbit_unset_tool_thread(nvbit_thread_key(std::this_thread::get_id()));
}

// Starts the context's drain worker and waits for it to come up. If it cannot,
// launches drain on the launching thread as before.
static void init_drain_worker(CUcontext ctx, CTXstate* ctx_state) {
  auto started = std::make_shared<std::promise<bool>>();
  auto started_result = started->get_future();
  auto worker = std::make_unique<cutracer::LaunchDrainWorker>(
      [ctx, ctx_state, started] { started->set_value(start_drain_thread(ctx, ctx_state)); },
      [ctx_state] { stop_drain_thread(ctx_state); });
  if (started_result.get()) {
    ctx_state->drain_worker = std::move(worker);
    return;
  }
  worker->stop();
  loprintf("WARNING: launch drain worker could not start for context %p; draining on the launching thread%s\n", ctx,
           nonblocking_launches ? ", so --nonblocking-launches has no effect in this context" : "");
}

// Reference code from NVIDIA nvbit mem_trace tool
// Reference code from NVIDIA nvbit mem_trace tool
void init_context_state(CUcontext ctx) {
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];
  ctx_state->hang_check = cutracer::HangCheckPolicy(
      {std::chrono::milliseconds(hang_check_interval_ms), hang_check_hits, std::chrono::seconds(hang_min_stall_s)});
  ctx_state->recv_thread_done = RecvThreadState::WORKING;
  cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
  ctx_state->channel_host.init((int)ctx_state_map.size() - 1, channel_buffer_size, ctx_state->channel_dev,
                               recv_thread_fun, ctx);
  nvbit_set_tool_thread(ctx_state->channel_host.get_thread_id());
  init_drain_worker(ctx, ctx_state);
  if (runtime_plan_requested.load()) apply_runtime_plan([&](auto& plan) { plan.initialize_context(ctx); });
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
                                                 unsigned int dynamic_shmem, const ClusterLaunchInfo& cluster,
                                                 const std::vector<std::string>& cpu_callstack = {},
                                                 const std::string& callstack_source = "", CUfunction func = nullptr,
                                                 CTXstate* ctx_state = nullptr, uint64_t launch_id = 0,
                                                 const char* launch_kind = "unknown") {
  auto md = meta.to_json();
  md["type"] = "kernel_metadata";
  md["capture_config"] = {{"version", 1},
                          {"grid_launch_id", launch_id},
                          {"launch_kind", launch_kind},
                          {"instr_begin", instr_begin_interval},
                          {"instr_end", instr_end_interval},
                          {"reg_trace_ipoint", reg_trace_ipoint() == IPOINT_BEFORE ? "before" : "after"},
                          {"root_function", meta.mangled_name},
                          {"pc_namespace", "root_function_without_executed_calls"},
                          {"root_instruction_count", meta.instruction_count},
                          {"reg_trace_instruction_count", meta.reg_trace_instruction_count},
                          {"instrumentation_errors", meta.instrumentation_errors}};
  if (runtime_plan_requested.load()) {
    md["capture_config"] = {{"version", 1},
                            {"grid_launch_id", launch_id},
                            {"launch_kind", launch_kind},
                            {"protocol", "cutracer.plan-runtime/1"},
                            {"applied_plan_required", true}};
  }
  md["grid"] = {dims.gridDimX, dims.gridDimY, dims.gridDimZ};
  md["block"] = {dims.blockDimX, dims.blockDimY, dims.blockDimZ};
  md["shmem_dynamic"] = dynamic_shmem;
  md["cluster_dim"] = {cluster.x, cluster.y, cluster.z};
  md["cluster_size"] = cluster.size();
  md["cluster_dim_source"] = cluster_dim_source_name(cluster.source);
  // Keep the metadata header tied to the records in this per-launch trace.
  // opcode_id is scoped to a CUfunction, so consumers need the actual launch
  // id (not the per-function iteration embedded in the filename) to form an
  // unambiguous join key across captured trace files.
  md["grid_launch_id"] = launch_id;
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

  // Effective enabled instrument modes (post init_analysis() if/else
  // precedence: e.g. proton_instr_histogram forces opcode_only,
  // deadlock_detection forces reg_trace). Order follows insertion order in
  // enabled_instrument_types_ordered for reproducibility. Detectors consume
  // this to verify the trace contract (e.g. ProxyFenceDetector requires
  // reg_trace + tma_trace) before running.
  nlohmann::json modes = nlohmann::json::array();
  for (InstrumentType t : enabled_instrument_types_ordered) {
    modes.push_back(instrument_type_to_name(t));
  }
  md["instrument_modes"] = modes;
  md["instrument_ipoints"] = instrument_ipoints_to_json();

  // Effective instruction-category filter (post init_instr_categories()).
  // null when CUTRACER_INSTR_CATEGORIES is unset -> no filter (all
  // categories instrumented); otherwise an array of lowercase category
  // tokens matching the env-var convention. Iterated in stable enum order
  // for reproducible JSON output (the underlying set is unordered).
  if (enabled_instr_categories.empty()) {
    md["instr_categories"] = nullptr;
  } else {
    nlohmann::json cats = nlohmann::json::array();
    if (enabled_instr_categories.count(InstrCategory::MMA)) cats.push_back("mma");
    if (enabled_instr_categories.count(InstrCategory::TMA)) cats.push_back("tma");
    if (enabled_instr_categories.count(InstrCategory::SYNC)) cats.push_back("sync");
    md["instr_categories"] = cats;
  }

  if (!has_trace_category_filter_enabled()) {
    md["trace_instr_categories"] = nullptr;
  } else {
    nlohmann::json cats = nlohmann::json::array();
    const auto& trace_categories = trace_instr_categories();
    if (trace_categories.count(InstrCategory::MMA)) cats.push_back("mma");
    if (trace_categories.count(InstrCategory::TMA)) cats.push_back("tma");
    if (trace_categories.count(InstrCategory::SYNC)) cats.push_back("sync");
    md["trace_instr_categories"] = cats;
  }

  const uint64_t selected_launch_id = trace_launch_id();
  md["trace_launch_id"] =
      selected_launch_id == UINT64_MAX ? nlohmann::json(nullptr) : nlohmann::json(selected_launch_id);
  const uint32_t selected_warp_mask = trace_warp_mask();
  md["trace_warp_mask"] = selected_warp_mask == 0 ? nlohmann::json(nullptr) : nlohmann::json(selected_warp_mask);
  const auto& selected_pc_offsets = trace_pc_offsets();
  if (selected_pc_offsets.empty()) {
    md["trace_pc_offsets"] = nullptr;
  } else {
    std::vector<uint64_t> pcs(selected_pc_offsets.begin(), selected_pc_offsets.end());
    std::sort(pcs.begin(), pcs.end());
    md["trace_pc_offsets"] = pcs;
  }
  md["trace_pc_filter_valid"] = trace_pc_filter_valid();

  return md;
}

// Caller holds writers_mutex through metadata publication and launch-specific setup.
static TraceWriter& create_trace_writer_locked(CTXstate* ctx_state, uint64_t launch_id,
                                               const std::string& base_filename, const nlohmann::json& metadata) {
  auto* writer = new TraceWriter(base_filename, trace_format);
  ctx_state->trace_writers[launch_id] = writer;
  // The receiver reads both maps under writers_mutex; publish them together.
  ctx_state->trace_index_by_kernel[launch_id] = 0;
  writer->write_metadata(metadata);
  return *writer;
}

// Reference code from NVIDIA nvbit mem_trace tool
// The functions whose code a launch of `func` selects: the root and the device
// functions it calls, which NVBit switches with it. Callers hold
// cuda_event_mutex (see CTXstate::launch_code_functions).
static const std::vector<const void*>& launch_code_functions(CUcontext ctx, CTXstate* ctx_state, CUfunction func) {
  auto cached = ctx_state->launch_code_functions.find(func);
  if (cached != ctx_state->launch_code_functions.end()) return cached->second;
  std::vector<const void*> functions{func};
  for (CUfunction callee : nvbit_get_related_functions(ctx, func)) functions.push_back(callee);
  return ctx_state->launch_code_functions.emplace(func, std::move(functions)).first->second;
}

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
 * @param launched_id The ID allocated for this call, absent during graph capture/build.
 * @param cbid The NVBit callback ID for the CUDA API call.
 * @param params A pointer to the parameters of the CUDA launch call.
 * @param stream_capture True if the launch is part of a CUDA stream capture.
 * @param build_graph True if the launch is part of a manual graph build.
 * @return `true` if the kernel was instrumented, `false` otherwise.
 */
static bool enter_kernel_launch(CUcontext ctx, CUfunction func, std::optional<uint64_t>& launched_id,
                                nvbit_api_cuda_t cbid, void* params, bool stream_capture = false,
                                bool build_graph = false) {
  launched_id.reset();
  if (runtime_plan_requested.load() &&
      (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync)) {
    runtime_plan_error("Plan capture does not support legacy launch geometry APIs");
    nvbit_set_at_launch(ctx, func, 0);
    nvbit_enable_instrumented(ctx, func, false);
    return false;
  }
  // Capture the CPU call stack early, before synchronization, so it reflects
  // the actual application call site that triggered this kernel launch.
  auto [cpu_callstack, callstack_source] = capture_cpu_callstack();

  CTXstate* ctx_state = ctx_state_map[ctx];

  // Graph nodes take their cluster geometry from graph construction. A function
  // captured with different cluster shapes leaves its graph launches unknown.
  unsigned int dynamic_cluster_x = 0, dynamic_cluster_y = 0, dynamic_cluster_z = 0;
  bool has_dynamic_cluster =
      extract_cluster_dim_from_launch_ex(cbid, params, dynamic_cluster_x, dynamic_cluster_y, dynamic_cluster_z);
  if (has_dynamic_cluster && (stream_capture || build_graph)) {
    const ClusterLaunchInfo observed =
        resolve_cluster_launch_info(func, true, dynamic_cluster_x, dynamic_cluster_y, dynamic_cluster_z);
    pthread_mutex_lock(&g_cluster_obs_mutex);
    const auto [known, inserted] = g_func_to_observed_cluster.emplace(func, observed);
    if (!inserted && (known->second.x != observed.x || known->second.y != observed.y || known->second.z != observed.z ||
                      known->second.source != observed.source)) {
      known->second = ClusterLaunchInfo{};
    }
    pthread_mutex_unlock(&g_cluster_obs_mutex);
  }
  ClusterLaunchInfo cluster =
      resolve_cluster_launch_info(func, has_dynamic_cluster, dynamic_cluster_x, dynamic_cluster_y, dynamic_cluster_z);

  // no need to sync during stream capture or manual graph build, since no
  // kernel is actually launched. Non-blocking launches sync below, and only if
  // this launch changes code an earlier launch may still be running.
  const bool launching = !stream_capture && !build_graph;
  if (launching && !nonblocking_launches && !ctx_state->channel_failed) {
    /* Make sure GPU is idle */
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();
  }

  // Get or create per-function metadata (basic fields auto-populated on first
  // access). Its NVBit queries must precede any NVBit control call below.
  auto& meta = get_or_create_kernel_func_metadata(func, ctx);
  identify_runtime_function(ctx, func, meta);
  // A reused handle runs its original code: NVBit's state for it is stale.
  const bool reused_handle = reused_function_handle(ctx, func);
  if (reused_handle) nvbit_enable_instrumented(ctx, func, false);
  const size_t instrumented_functions = already_instrumented.size();
  bool should_instrument =
      !ctx_state->channel_failed && !reused_handle &&
      (runtime_plan_requested.load() ? apply_runtime_plan([&](auto& plan) {
        return plan.prepare(ctx, func, meta.runtime_identity, reinterpret_cast<uint64_t>(ctx_state->channel_dev));
      })
                                     : instrument_function_if_needed(ctx, func));
  // This launch inserted instrumentation, into its root or a callee.
  const bool first_instrumentation = already_instrumented.size() != instrumented_functions;
  if (should_instrument)
    ctx_state->instrumented_launch_roots.insert(func);
  else
    ctx_state->instrumented_launch_roots.erase(func);

  const char* func_name = meta.unmangled_name.c_str();
  uint64_t pc = meta.func_addr;

  // Per-launch dynamic attributes, populated in the normal launch path below.
  // Declared here so they are accessible when writing kernel_metadata after TraceWriter creation.
  KernelDimensions dims{};
  unsigned int dynamic_shmem = 0;
  uint32_t current_iter = 0;
  nlohmann::json runtime_launch;
  nlohmann::json plan_applied;

  // during stream capture or manual graph build, no kernel is launched, so
  // do not set launch argument, do not print kernel info, do not increase
  // grid_launch_id. All these should be done at graph node launch time.
  if (!stream_capture && !build_graph) {
    /* set kernel launch id at launch time */
    const uint64_t kernel_launch_id = global_kernel_launch_id.fetch_add(1, std::memory_order_relaxed);
    launched_id = kernel_launch_id;
    nvbit_set_at_launch(ctx, func, runtime_plan_requested.load() ? 0 : kernel_launch_id);

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
      unsigned int dyn_cx = has_dynamic_cluster ? cluster.x : 0;
      unsigned int dyn_cy = has_dynamic_cluster ? cluster.y : 0;
      unsigned int dyn_cz = has_dynamic_cluster ? cluster.z : 0;
      log_cluster_info_once(ctx, func, func_name, dyn_cx, dyn_cy, dyn_cz);
    }

    // For histogram analysis, we need to map the kernel launch ID back to its
    // function and context to access the correct SASS and clock opcode maps.
    // We also track the iteration count for unique file naming.
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
    {
      std::lock_guard<std::mutex> lock(kernel_launch_state_mutex);
      current_iter = next_kernel_iteration(meta, should_instrument);
      kernel_launch_to_func_map[kernel_launch_id] = {ctx, func};
      kernel_launch_to_iter_map[kernel_launch_id] = current_iter;
      kernel_launch_to_dimensions_map[kernel_launch_id] = dims;
    }

    uint64_t runtime_stream = 0;
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
      runtime_stream = reinterpret_cast<uint64_t>(((cuLaunchKernelEx_params*)params)->config->hStream);
    } else if (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz ||
               cbid == API_CUDA_cuLaunchCooperativeKernel || cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
      runtime_stream = reinterpret_cast<uint64_t>(((cuLaunchKernel_params*)params)->hStream);
    }
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync) {
      runtime_event([&](auto& ledger) { ledger.gap("legacy_launch_geometry_unavailable"); });
    } else {
      runtime_launch = identify_runtime_launch(ctx, func, kernel_launch_id, dims, cluster, runtime_stream, "direct",
                                               should_instrument);
      if (!runtime_launch.is_null()) {
        std::lock_guard<std::mutex> lock(ctx_state->identified_launches_mutex);
        ctx_state->identified_launches.insert(kernel_launch_id);
      }
    }
    if (runtime_plan_requested.load() && should_instrument) {
      plan_applied = apply_runtime_plan([&](auto& plan) { return plan.launch(ctx, func, runtime_launch); });
      if (plan_applied.is_null()) {
        should_instrument = false;
        ctx_state->instrumented_launch_roots.erase(func);
        nvbit_set_at_launch(ctx, func, 0);
      }
    }

    // Write structured kernel launch event (applies to ALL kernels, not just instrumented)
    if (kernel_events_mode != KernelEventsMode::DISABLED) {
      uint64_t stream_id = 0;
      if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
        stream_id = (uint64_t)((cuLaunchKernelEx_params*)params)->config->hStream;
      } else {
        stream_id = (uint64_t)((cuLaunchKernel_params*)params)->hStream;
      }
      write_kernel_launch_event(meta, dims, cluster, dynamic_shmem, stream_id, kernel_launch_id, cpu_callstack,
                                callstack_source);
    }
  }

  if (should_instrument && !stream_capture && !build_graph) {
    // Record kernel start time at launch (before any GPU data arrives).
    // This enables no-data timeout detection even for immediate-silence hangs
    // where the kernel deadlocks from the very first instruction and the
    // recv thread never receives any trace data.
    // Create TraceWriter for each kernel launch (per-launch trace files)
    std::string base_filename = generate_kernel_log_basename(ctx, func, current_iter, meta.kernel_checksum);
    base_filename = runtime_trace_basename(base_filename, runtime_launch);
    const uint64_t current_launch_id = launched_id.value();
    auto metadata = build_kernel_metadata_json(meta, dims, dynamic_shmem, cluster, cpu_callstack, callstack_source,
                                               func, ctx_state, current_launch_id, "direct");
    if (!runtime_launch.is_null()) metadata["runtime_identity"] = runtime_launch;

    {
      std::unique_lock<std::shared_mutex> lock(ctx_state->writers_mutex);
      // A non-blocking launch can wait in its stream behind other work, so its
      // timers start at its first trace record instead.
      if (!nonblocking_launches) {
        ctx_state->kernel_start_time = time(nullptr);
        ctx_state->last_data_received_time = 0;
      }
      ctx_state->kernel_start_name = meta.unmangled_name;
      auto& writer = create_trace_writer_locked(ctx_state, current_launch_id, base_filename, metadata);
      if (!plan_applied.is_null()) writer.write_metadata(plan_applied);
    }

    loprintf_v("Created TraceWriter for launch_id %lu, mode %d, file: %s\n", current_launch_id, trace_format,
               base_filename.c_str());
  }

  // With nothing instrumented, no launch ever changes code.
  if (nonblocking_launches && has_any_instrumentation_enabled() &&
      ctx_state->launch_code.update(launch_code_functions(ctx, ctx_state, func), should_instrument,
                                    first_instrumentation) &&
      launching) {
    // Kernels still running must not see the code they run change.
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();
  }
  nvbit_enable_instrumented(ctx, func, should_instrument);
  if (runtime_plan_requested.load() && should_instrument && !stream_capture && !build_graph) {
    // Let NVBit finish preparing the instrumented launch: a launch on a
    // non-blocking stream can otherwise start without the plan's instrumentation.
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();
  }
  return should_instrument;
}

struct CaptureDrainResult {
  bool kernel_completed;
  bool channel_drained;
};

static bool flush_channel(CUcontext ctx, CTXstate* ctx_state, CUstream stream) {
  void* flush_args[] = {&ctx_state->channel_dev};
  return nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0, stream, flush_args, nullptr) == 0;
}

// Direct launches: the whole device must be idle before the flush, as before
// the drain moved off the launching thread.
static CaptureDrainResult drain_after_device_sync(CUcontext ctx, CTXstate* ctx_state, bool launch_succeeded) {
  const auto kernel_status = cudaDeviceSynchronize();
  const bool channel_drained =
      kernel_status == cudaSuccess && flush_channel(ctx, ctx_state, nullptr) && cudaDeviceSynchronize() == cudaSuccess;
  if (!channel_drained) {
    ctx_state->channel_failed = true;
  }
  return {launch_succeeded && kernel_status == cudaSuccess, channel_drained};
}

// Graph launches and non-blocking direct launches: wait for the event the
// launching thread recorded behind the launch on its stream, then flush on the
// worker's private stream. The kernels have finished, so the flush needs no
// ordering against application work.
static CaptureDrainResult drain_after_event(CUcontext ctx, CTXstate* ctx_state, cudaEvent_t launch_end,
                                            bool launch_succeeded) {
  const auto kernel_status = launch_end ? cudaEventSynchronize(launch_end) : cudaDeviceSynchronize();
  if (launch_end) cudaEventDestroy(launch_end);
  const auto flush_stream = ctx_state->drain_stream;
  const auto flushed = [&] { return flush_stream ? cudaStreamSynchronize(flush_stream) : cudaDeviceSynchronize(); };
  const bool channel_drained =
      kernel_status == cudaSuccess && flush_channel(ctx, ctx_state, flush_stream) && flushed() == cudaSuccess;
  if (!channel_drained) {
    ctx_state->channel_failed = true;
  }
  return {launch_succeeded && kernel_status == cudaSuccess, channel_drained};
}

// Records, on the launching thread, where a launch ends on its stream. A
// per-thread default stream names a different stream on any other thread, so
// the drain worker waits for this event instead of for the stream. Returns
// nullptr if the event cannot be recorded; the worker then waits for the device.
static cudaEvent_t record_launch_end(CUstream stream, int api) {
  const bool per_thread_api = api == API_CUDA_cuGraphLaunch_ptsz || api == API_CUDA_cuLaunchKernel_ptsz ||
                              api == API_CUDA_cuLaunchKernelEx_ptsz || api == API_CUDA_cuLaunchCooperativeKernel_ptsz;
  cudaStream_t record_stream = stream;
  if (record_stream == nullptr && per_thread_api) record_stream = cudaStreamPerThread;
  cudaEvent_t event = nullptr;
  if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) return nullptr;
  if (cudaEventRecord(event, record_stream) != cudaSuccess) {
    cudaEventDestroy(event);
    return nullptr;
  }
  return event;
}

// Queues a launch's drain on the context's worker (or runs it here if there is
// none) and returns its ticket. A drain that throws marks the channel failed and
// runs `on_failure`, which posts the launch's capture completion as failed:
// otherwise its trace would stay open and the no-data watchdog would later end
// the run as a hang.
template <typename Drain, typename OnFailure>
static uint64_t queue_drain(CTXstate* ctx_state, Drain drain, OnFailure on_failure) {
  auto job = [ctx_state, drain, on_failure]() mutable {
    cutracer::ScopedCudaCallbackSuppression suppress_callbacks;
    std::string failure;
    try {
      drain();
      return;
    } catch (const std::exception& error) {
      failure = error.what();
    } catch (...) {
      failure = "unknown exception";
    }
    ctx_state->channel_failed = true;
    loprintf("ERROR: launch drain failed: %s\n", failure.c_str());
    on_failure();
  };
  if (!ctx_state->drain_worker) {
    job();
    return 0;
  }
  return ctx_state->drain_worker->submit(std::move(job));
}

// Runs a launch's drain on the context's worker and waits for it, so the
// launching thread returns only after the capture completion is posted.
// Returns whether the drain saw the launch complete.
template <typename Drain, typename OnFailure>
static bool run_drain(CTXstate* ctx_state, Drain drain, OnFailure on_failure) {
  bool completed = false;
  const uint64_t ticket = queue_drain(ctx_state, [&completed, drain]() mutable { completed = drain(); }, on_failure);
  if (ctx_state->drain_worker) ctx_state->drain_worker->wait(ticket);
  return completed;
}

// Queues a non-blocking launch's drain. Each queued drain holds an event, and
// keeps its traces open (a file and a buffer each; a graph launch has one per
// instrumented node) until its launch finishes. A launching thread that gets
// too far ahead waits: for the oldest queued drain past kMaxQueuedDrains, and
// for every queued drain past kMaxOpenTraces open traces.
constexpr uint64_t kMaxQueuedDrains = 1024;
constexpr size_t kMaxOpenTraces = 256;

static size_t open_trace_count(CTXstate* ctx_state) {
  std::shared_lock<std::shared_mutex> lock(ctx_state->writers_mutex);
  return ctx_state->trace_writers.size();
}

template <typename Drain, typename OnFailure>
static void queue_nonblocking_drain(CTXstate* ctx_state, Drain drain, OnFailure on_failure) {
  const uint64_t ticket = queue_drain(ctx_state, std::move(drain), std::move(on_failure));
  if (!ctx_state->drain_worker) return;
  if (ticket > kMaxQueuedDrains) ctx_state->drain_worker->wait(ticket - kMaxQueuedDrains);
  if (open_trace_count(ctx_state) > kMaxOpenTraces) ctx_state->drain_worker->wait_idle();
}

// When the drain ran on the launching thread, a fault it saw also became that
// thread's last CUDA error, which a launch check such as PyTorch's reads right
// after the launch. Keep that: the device is idle by now, so this is cheap.
static void surface_launch_failure() {
  cudaDeviceSynchronize();
}

// `launch_end` is the event a non-blocking launch recorded behind itself; a
// blocking launch passes nullopt and the drain waits for the whole device.
static bool finish_kernel_launch(CUcontext ctx, CTXstate* ctx_state, std::optional<uint64_t> launch_id,
                                 bool launch_succeeded, std::optional<cudaEvent_t> launch_end) {
  const auto completion = launch_end ? drain_after_event(ctx, ctx_state, *launch_end, launch_succeeded)
                                     : drain_after_device_sync(ctx, ctx_state, launch_succeeded);
  const bool kernel_completed = completion.kernel_completed;
  const bool channel_drained = completion.channel_drained;
  if (launch_id) {
    bool identified = false;
    {
      std::lock_guard<std::mutex> lock(ctx_state->identified_launches_mutex);
      identified = ctx_state->identified_launches.erase(*launch_id) > 0;
    }
    if (identified) {
      runtime_event([&](auto& ledger) {
        ledger.complete(reinterpret_cast<uint64_t>(ctx), *launch_id, kernel_completed, channel_drained);
      });
    }
    if (runtime_plan_requested.load()) {
      apply_runtime_plan([&](auto& plan) { plan.device_complete(ctx, *launch_id, kernel_completed, channel_drained); });
    }
    const std::string error = kernel_completed && channel_drained ? "" : "cuda_launch_or_flush_failed";
    ctx_state->capture_completions.request({*launch_id, kernel_completed, channel_drained, error});
    if (!error.empty()) {
      loprintf("ERROR: incomplete capture for launch_id %lu: kernel_completed=%d channel_drained=%d\n", *launch_id,
               kernel_completed, channel_drained);
    }
  }
  return kernel_completed;
}

// Whether a launch has anything to drain: records in the channel (it has a
// trace writer) or a runtime identity record awaiting its completion.
static bool launch_needs_drain(CTXstate* ctx_state, uint64_t launch_id) {
  {
    std::shared_lock<std::shared_mutex> lock(ctx_state->writers_mutex);
    if (ctx_state->trace_writers.count(launch_id)) return true;
  }
  std::lock_guard<std::mutex> lock(ctx_state->identified_launches_mutex);
  return ctx_state->identified_launches.count(launch_id) > 0;
}

// the function is only called for non cuda graph launch cases.
static void leave_kernel_launch(CUcontext ctx, CTXstate* ctx_state, std::optional<uint64_t> launch_id,
                                bool launch_succeeded, CUstream stream, int api) {
  const auto fail = [=] {
    if (launch_id) ctx_state->capture_completions.request({*launch_id, false, false, "launch_drain_failed"});
  };
  if (!nonblocking_launches) {
    const auto drain = [=] { return finish_kernel_launch(ctx, ctx_state, launch_id, launch_succeeded, std::nullopt); };
    if (!run_drain(ctx_state, drain, fail)) surface_launch_failure();
    return;
  }
  // Nothing was instrumented or identified: there is nothing to wait for.
  if (!launch_id || !launch_needs_drain(ctx_state, *launch_id)) return;
  const cudaEvent_t launch_end = record_launch_end(stream, api);
  queue_nonblocking_drain(
      ctx_state, [=] { finish_kernel_launch(ctx, ctx_state, launch_id, launch_succeeded, launch_end); }, fail);
}

// The kernels a cuGraphLaunch registered and its runtime identity frame are
// taken at its exit, on the launching thread, and handed to its drain: with
// non-blocking launches a later cuGraphLaunch can run before this drain does.
struct GraphLaunchExit {
  std::vector<uint64_t> launch_ids;
  std::optional<cutracer::RuntimeGraphTracker::Exit> runtime;
};

static bool finish_graph_launch(CUcontext ctx, CTXstate* ctx_state, const GraphLaunchExit& exit, cudaEvent_t launch_end,
                                bool launch_succeeded) {
  const auto completion = drain_after_event(ctx, ctx_state, launch_end, launch_succeeded);
  const bool kernel_completed = completion.kernel_completed;
  const bool channel_drained = completion.channel_drained;
  const std::string error = kernel_completed && channel_drained ? "" : "cuda_graph_launch_or_flush_failed";
  complete_runtime_graph(ctx, ctx_state, exit.runtime, kernel_completed, channel_drained);
  if (!runtime_plan_requested.load())
    ctx_state->capture_completions.request_graph_completion(exit.launch_ids, kernel_completed, channel_drained, error);
  if (!error.empty()) {
    loprintf("ERROR: incomplete graph capture: kernel_completed=%d channel_drained=%d\n", kernel_completed,
             channel_drained);
  }
  return kernel_completed;
}

static void leave_graph_launch(CUcontext ctx, CTXstate* ctx_state, CUstream stream, int api, bool launch_succeeded) {
  const GraphLaunchExit exit{ctx_state->capture_completions.take_graph_launches(),
                             take_runtime_graph(ctx, stream, api, true)};
  // No instrumented node and no runtime identity frame: nothing to drain.
  if (nonblocking_launches && exit.launch_ids.empty() && !exit.runtime) return;
  const cudaEvent_t launch_end = record_launch_end(stream, api);
  const auto drain = [=] { return finish_graph_launch(ctx, ctx_state, exit, launch_end, launch_succeeded); };
  const auto fail = [=] {
    complete_runtime_graph(ctx, ctx_state, exit.runtime, false, false);
    if (!runtime_plan_requested.load())
      ctx_state->capture_completions.request_graph_completion(exit.launch_ids, false, false, "launch_drain_failed");
  };
  if (nonblocking_launches) {
    queue_nonblocking_drain(ctx_state, drain, fail);
    return;
  }
  if (!run_drain(ctx_state, drain, fail)) surface_launch_failure();
}

// CUDA 12+ launch APIs also accept a CUkernel, which the CUDA runtime passes
// for its registered kernels. NVBit reports graph nodes by the current
// context's CUfunction, so all per-function tool state uses that handle.
static CUfunction canonical_launch_function(CUfunction func) {
  CUmodule module = nullptr;
  if (!func || cuFuncGetModule(&module, func) != CUDA_ERROR_INVALID_HANDLE) return func;
  CUfunction resolved = nullptr;
  if (cuKernelGetFunction(&resolved, reinterpret_cast<CUkernel>(func)) != CUDA_SUCCESS || !resolved) return func;
  return resolved;
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char* name, void* params,
                         CUresult* pStatus) {
  // Check before locking: tool callbacks such as context teardown can issue
  // CUDA calls while holding mutex, while another thread owns cuda_event_mutex.
  if (cutracer::ScopedCudaCallbackSuppression::is_suppressed()) {
    return;
  }
  cutracer::ScopedCudaCallbackSuppression suppress_callbacks;
  // Driver calls may nest between their entry and exit callbacks. Preserve
  // each call's launch ID and lock ownership, including stream-capture calls.
  using LaunchCallback = cutracer::ScopedCudaLaunchCallback;
  static thread_local LaunchCallback::PendingLaunches pending_launches;
  // Keep launch arguments and channel flushing exclusive across the driver
  // call. The guard releases this callback's acquisition and any consumed
  // entry acquisition only after the completion request has been enqueued.
  LaunchCallback launch_callback(cuda_event_mutex, pending_launches);

  CTXstate* ctx_state = ctx_state_map[ctx];
  const auto take_pending_launch = [&]() -> std::optional<LaunchCallback::PendingLaunch> {
    const auto exit = launch_callback.take_pending_launch(ctx, cbid);
    if (!exit.in_order) {
      ctx_state->channel_failed = true;
      loprintf("WARNING: capture completion is unverified after %s launch exit %s\n",
               exit.launch ? "out-of-order" : "unmatched", name);
    }
    // Recover a non-top matching entry without consuming other live frames.
    // Its completion remains incomplete because channel_failed is sticky.
    return exit.launch;
  };

  observe_runtime_module(ctx, is_exit, cbid, name, params, pStatus);

  switch (cbid) {
    // Identify all the possible CUDA launch events without stream
    // parameters, they will not get involved with cuda graph
    case API_CUDA_cuLaunch:
    case API_CUDA_cuLaunchGrid: {
      cuLaunch_params* p = (cuLaunch_params*)params;
      CUfunction func = p->f;
      if (!is_exit) {
        ctx_state->need_sync = true;
        std::optional<uint64_t> launch_id;
        enter_kernel_launch(ctx, func, launch_id, cbid, params);
        launch_callback.enter({ctx, cbid, launch_id, true});
      } else if (const auto launch = take_pending_launch()) {
        leave_kernel_launch(ctx, ctx_state, launch->launch_id, pStatus && *pStatus == CUDA_SUCCESS, nullptr, cbid);
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
      func = canonical_launch_function(func);

      if (!is_exit) {
        cudaStreamCaptureStatus streamStatus;
        /* check if the stream is capturing, if yes, do not sync */
        CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
        bool stream_capture = (streamStatus == cudaStreamCaptureStatusActive);
        if (!stream_capture) {
          ctx_state->need_sync = true;
        }
        std::optional<uint64_t> launch_id;
        enter_kernel_launch(ctx, func, launch_id, cbid, params, stream_capture);
        launch_callback.enter({ctx, cbid, launch_id, !stream_capture});
      } else if (const auto launch = take_pending_launch()) {
        if (launch->retained_lock) {
          loprintf_vl(1, "kernel %s not captured by cuda graph\n", nvbit_get_func_name(ctx, func));
          leave_kernel_launch(ctx, ctx_state, launch->launch_id, pStatus && *pStatus == CUDA_SUCCESS, hStream, cbid);
        } else {
          loprintf_vl(1, "kernel %s captured by cuda graph\n", nvbit_get_func_name(ctx, func));
        }
      }
    } break;
    case API_CUDA_cuGraphAddKernelNode: {
      cuGraphAddKernelNode_params* p = (cuGraphAddKernelNode_params*)params;
      CUfunction func = canonical_launch_function(p->nodeParams->func);

      if (!is_exit) {
        // cuGraphAddKernelNode_params->nodeParams is the same as
        // cuLaunchKernel_params up to sharedMemBytes
        std::optional<uint64_t> captured_launch;
        enter_kernel_launch(ctx, func, captured_launch, cbid, (void*)p->nodeParams, false, true);
      }
    } break;
    case API_CUDA_cuGraphLaunch:
    case API_CUDA_cuGraphLaunch_ptsz: {
      // if we are exiting a cuda graph launch:
      // Wait until the graph is completed using
      // cudaStreamSynchronize()
      const CUstream stream = cbid == API_CUDA_cuGraphLaunch ? static_cast<cuGraphLaunch_params*>(params)->hStream
                                                             : static_cast<cuGraphLaunch_ptsz_params*>(params)->hStream;
      if (!is_exit) {
        launch_callback.enter({ctx, cbid, std::nullopt, true});
        if (runtime_identity.active()) {
          try {
            runtime_graph_tracker.begin(reinterpret_cast<uint64_t>(ctx), runtime_stream_key(stream), cbid);
          } catch (const std::exception& error) {
            runtime_identity_error(error.what());
            finish_runtime_graph(ctx, ctx_state, stream, cbid, false, false, false);
          }
        }
      } else if (take_pending_launch()) {
        leave_graph_launch(ctx, ctx_state, stream, cbid, pStatus && *pStatus == CUDA_SUCCESS);
      } else {
        // The exit cannot qualify any pending invocation. Retire its evidence
        // as incomplete without synchronizing an unpaired driver call.
        finish_runtime_graph(ctx, ctx_state, stream, cbid, false, false, false);
        if (!runtime_plan_requested.load())
          ctx_state->capture_completions.request_graph_completion(false, false, "unmatched_graph_exit");
      }

    } break;
    case API_CUDA_cuGraphLaunchWithArguments: {
      if (!is_exit) {
        if (runtime_plan_requested.load())
          runtime_plan_error("Plan capture does not support cuGraphLaunchWithArguments");
        launch_callback.enter({ctx, cbid, std::nullopt, true});
        // Graph node admission also checks this flag. Do not consume graph
        // completion IDs owned by an enclosing supported launch on this exit.
        ctx_state->channel_failed = true;
        loprintf("WARNING: capture completion is unverified after unsupported graph launch API %s\n", name);
      } else {
        take_pending_launch();
      }
    } break;
    default:
      break;
  };
}

// Reference NVIDIA record_reg_vals example
void nvbit_tool_init(CUcontext ctx) {
  cutracer::ScopedCudaCallbackSuppression suppress_callbacks;
  pthread_mutex_lock(&mutex);
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  init_context_state(ctx);
  pthread_mutex_unlock(&mutex);
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_ctx_init(CUcontext ctx) {
  cutracer::ScopedCudaCallbackSuppression suppress_callbacks;
  pthread_mutex_lock(&mutex);
  if (verbose) {
    printf("CUTracer: STARTING CONTEXT %p\n", ctx);
  }
  assert(ctx_state_map.find(ctx) == ctx_state_map.end());
  CTXstate* ctx_state = new CTXstate;
  ctx_state_map[ctx] = ctx_state;
  runtime_event([&](auto& ledger) { ledger.context_begin(reinterpret_cast<uint64_t>(ctx)); });

  nvbit_load_tool_module(ctx, (const void*)flush_channel_bin, &ctx_state->tool_module);
  assert(ctx_state->tool_module != nullptr);
  nvbit_find_function_by_name(ctx, ctx_state->tool_module, "flush_channel", &ctx_state->flush_channel_func);
  assert(ctx_state->flush_channel_func != nullptr);

  pthread_mutex_unlock(&mutex);
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_ctx_term(CUcontext ctx) {
  cutracer::ScopedCudaCallbackSuppression suppress_callbacks;
  pthread_mutex_lock(&mutex);
  loprintf_v("CUTracer: TERMINATING CONTEXT %p\n", ctx);
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  // Its last action destroys its stream and unregisters it from NVBit.
  if (ctx_state->drain_worker) {
    ctx_state->drain_worker->stop();
    ctx_state->drain_worker.reset();
  }

  /* Final flush: ensure last kernel's data is not lost */
  if (ctx_state->need_sync && !ctx_state->channel_failed) {
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
        if (runtime_plan_requested.load())
          apply_runtime_plan([&, id = launch_id, trace = writer](auto& plan) { plan.finish_trace(ctx, id, *trace); });
        writer->finish(launch_id, false, false);
        delete writer;
      }
    }
    ctx_state->trace_writers.clear();
    ctx_state->trace_index_by_kernel.clear();
  }

  // Clean up any remaining kernel mapping entries
  // (in case there were kernels launched but no data received)
  {
    std::lock_guard<std::mutex> lock(kernel_launch_state_mutex);
    for (auto it = kernel_launch_to_func_map.begin(); it != kernel_launch_to_func_map.end();) {
      if (it->second.first == ctx) {
        kernel_launch_to_iter_map.erase(it->first);
        kernel_launch_to_dimensions_map.erase(it->first);
        it = kernel_launch_to_func_map.erase(it);
      } else
        ++it;
    }
  }

  if (ctx_state->channel_dev != nullptr) {
    // Unregister before destroy() joins the receiver and resets its thread ID.
    nvbit_unset_tool_thread(ctx_state->channel_host.get_thread_id());
    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
  }
  ctx_state_map.erase(ctx);
  delete ctx_state;
  forget_runtime_module(ctx, nullptr);
  if (runtime_plan_requested.load()) apply_runtime_plan([&](auto& plan) { plan.destroy_context(ctx); });
  runtime_event([&](auto& ledger) { ledger.context_end(reinterpret_cast<uint64_t>(ctx)); });
  runtime_graph_tracker.clear(reinterpret_cast<uint64_t>(ctx));

  // Global cleanup: only when the last context is destroyed.
  // g_kernel_events_writer is shared across all contexts, so we must not
  // delete it while other contexts may still record events.
  if (ctx_state_map.empty()) {
    {
      std::lock_guard<std::mutex> lock(kernel_launch_state_mutex);
      kernel_launch_to_func_map.clear();
      kernel_launch_to_iter_map.clear();
      kernel_launch_to_dimensions_map.clear();
    }
    if (g_kernel_events_writer) {
      g_kernel_events_writer->flush();
      delete g_kernel_events_writer;
      g_kernel_events_writer = nullptr;
      callstack_seen.clear();
    }

    cleanup_log_handle();
    save_delay_config();
  }

  pthread_mutex_unlock(&mutex);
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_graph_node_launch(CUcontext ctx, CUfunction func, CUstream stream, uint64_t launch_handle) {
  cutracer::ScopedCudaCallbackSuppression suppress_callbacks;
  func_config_t config = {0};
  const char* func_name = nvbit_get_func_name(ctx, func);
  uint64_t pc = nvbit_get_func_addr(ctx, func);

  pthread_mutex_lock(&mutex);

  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  const uint64_t kernel_launch_id = global_kernel_launch_id.fetch_add(1, std::memory_order_relaxed);
  nvbit_set_at_launch(ctx, func, runtime_plan_requested.load() ? 0 : kernel_launch_id, stream, launch_handle);
  nvbit_get_func_config(ctx, func, &config);

  ClusterLaunchInfo cluster;
  bool has_observed_cluster = false;
  pthread_mutex_lock(&g_cluster_obs_mutex);
  auto cluster_it = g_func_to_observed_cluster.find(func);
  if (cluster_it != g_func_to_observed_cluster.end()) {
    cluster = cluster_it->second;
    has_observed_cluster = true;
  }
  pthread_mutex_unlock(&g_cluster_obs_mutex);
  if (!has_observed_cluster || cluster.source == ClusterDimSource::UNKNOWN) {
    cluster = resolve_cluster_launch_info(func);
  }

  loprintf(
      "CUTracer: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
      "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
      "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
      "id %ld\n",
      (uint64_t)ctx, pc, func_name, kernel_launch_id, config.gridDimX, config.gridDimY, config.gridDimZ,
      config.blockDimX, config.blockDimY, config.blockDimZ, config.num_registers,
      config.shmem_static_nbytes + config.shmem_dynamic_nbytes, (uint64_t)stream);

  // Cluster targeting: log cluster dims once per function. Look up the
  // runtime cluster dim observed at the most recent cuLaunchKernelEx for this
  // CUfunction (including the stream-capture launch that built this graph
  // node). Falls back to the kernel-declared __cluster_dims__ inside the
  // helper if neither source has dynamic info.
  if (g_delay_cta_target == 1) {
    unsigned int dyn_cx = cluster.source == ClusterDimSource::LAUNCH_ATTRIBUTE ? cluster.x : 0;
    unsigned int dyn_cy = cluster.source == ClusterDimSource::LAUNCH_ATTRIBUTE ? cluster.y : 0;
    unsigned int dyn_cz = cluster.source == ClusterDimSource::LAUNCH_ATTRIBUTE ? cluster.z : 0;
    log_cluster_info_once(ctx, func, func_name, dyn_cx, dyn_cy, dyn_cz);
  }

  // Store kernel dimensions
  KernelDimensions dims = {config.gridDimX,  config.gridDimY,  config.gridDimZ,
                           config.blockDimX, config.blockDimY, config.blockDimZ};
  bool capture_selected = ctx_state->instrumented_launch_roots.count(func) && !ctx_state->channel_failed;
  if (runtime_plan_requested.load()) {
    apply_runtime_plan([&](auto& plan) { plan.validate_graph_function(ctx, func); });
    auto plan = cutracer::current_runtime_plan();
    capture_selected = capture_selected && plan && !plan->disabled();
  }
  const bool instrument_enabled =
      (runtime_plan_requested.load() || has_any_instrumentation_enabled()) && capture_selected;
  const KernelFuncMetadata* node_meta = instrument_enabled ? &get_or_create_kernel_func_metadata(func, ctx) : nullptr;
  uint32_t current_iter = 0;
  {
    std::lock_guard<std::mutex> lock(kernel_launch_state_mutex);
    if (node_meta) current_iter = next_kernel_iteration(*node_meta, true);
    kernel_launch_to_func_map[kernel_launch_id] = {ctx, func};
    kernel_launch_to_iter_map[kernel_launch_id] = current_iter;
    kernel_launch_to_dimensions_map[kernel_launch_id] = dims;
  }

  std::optional<uint64_t> graph_invocation;
  if (runtime_identity.active()) {
    graph_invocation = runtime_graph_tracker.current(reinterpret_cast<uint64_t>(ctx), runtime_stream_key(stream));
  }
  const auto runtime_launch =
      identify_runtime_launch(ctx, func, kernel_launch_id, dims, cluster, reinterpret_cast<uint64_t>(stream), "graph",
                              capture_selected, launch_handle, graph_invocation);
  if (!runtime_launch.is_null()) {
    if (!graph_invocation) {
      runtime_identity_error("graph_node_without_matching_invocation");
    } else {
      runtime_event([&](auto&) {
        runtime_graph_tracker.attach(reinterpret_cast<uint64_t>(ctx), runtime_stream_key(stream), *graph_invocation,
                                     kernel_launch_id);
      });
    }
  }

  // Capture the CPU call stack for graph node launches as well
  auto [cpu_callstack, callstack_source] = capture_cpu_callstack();

  // Write structured kernel launch event for graph node launches
  if (kernel_events_mode != KernelEventsMode::DISABLED) {
    const auto& meta_ev = get_or_create_kernel_func_metadata(func, ctx);
    write_kernel_launch_event(meta_ev, dims, cluster, config.shmem_dynamic_nbytes, (uint64_t)stream, kernel_launch_id,
                              cpu_callstack, callstack_source);
  }

  nlohmann::json plan_applied;
  if (runtime_plan_requested.load() && capture_selected) {
    plan_applied = apply_runtime_plan([&](auto& plan) { return plan.launch(ctx, func, runtime_launch); });
    if (plan_applied.is_null()) {
      capture_selected = false;
      nvbit_set_at_launch(ctx, func, 0, stream, launch_handle);
    }
  }

  // Create TraceWriter for this graph node launch only if instrumentation is enabled
  // Otherwise, trace files would be empty (no data to write)
  if ((runtime_plan_requested.load() || has_any_instrumentation_enabled()) && capture_selected) {
    const auto& meta = get_or_create_kernel_func_metadata(func, ctx);
    std::string base_filename = generate_kernel_log_basename(ctx, func, current_iter, meta.kernel_checksum);
    base_filename = runtime_trace_basename(base_filename, runtime_launch);
    auto metadata = build_kernel_metadata_json(meta, dims, config.shmem_dynamic_nbytes, cluster, cpu_callstack,
                                               callstack_source, func, ctx_state, kernel_launch_id, "graph");
    if (!runtime_launch.is_null()) metadata["runtime_identity"] = runtime_launch;
    {
      std::unique_lock<std::shared_mutex> lock(ctx_state->writers_mutex);
      // A graph node launch is an actual kernel execution (unlike the direct
      // launches skipped during stream capture / graph build above), so it
      // starts the kernel timeout exactly as a direct launch does. Delay-only
      // instrumentation sends no trace records for the recv thread to learn
      // the launch from, so without this a hung graph kernel is bounded only
      // by the caller's outer timeout and never by CUTRACER_KERNEL_TIMEOUT_S.
      // The timeout state is a single global, cleared only once the whole
      // graph completes, so graph timing is scoped to the graph execution:
      // only the first node records the start and the name (marked as a
      // graph), and later nodes do not move the timer or rename it.
      if (!nonblocking_launches) {
        if (ctx_state->kernel_start_time == 0) {
          ctx_state->kernel_start_time = time(nullptr);
          ctx_state->kernel_start_name = meta.unmangled_name + " (graph)";
        }
        ctx_state->last_data_received_time = 0;
      }
      auto& writer = create_trace_writer_locked(ctx_state, kernel_launch_id, base_filename, metadata);
      if (runtime_plan_requested.load()) {
        writer.write_metadata(plan_applied);
      } else {
        writer.mark_capture_error("graph_launch_completion_unverified");
        ctx_state->capture_completions.register_graph_launch(kernel_launch_id);
      }
    }

    loprintf_v("Created TraceWriter for graph node launch_id %lu, file: %s\n", kernel_launch_id, base_filename.c_str());
  }

  pthread_mutex_unlock(&mutex);
}

void nvbit_at_term() {
  cutracer::publish_runtime_plan({});
  try {
    runtime_identity.stop();
  } catch (const std::exception& error) {
    runtime_identity_error(error.what());
  }
}

// Reference code from NVIDIA nvbit mem_trace tool with Meta modifications for
// env config
void nvbit_at_init() {
  // Initialize configuration from environment variables
  init_config_from_env();
  init_runtime_observation();
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
    events_basename += "_" + std::to_string(getpid());
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
