/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#pragma once

#include <stdint.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declaration to avoid circular dependency
enum class InstrumentType;
enum class InstrCategory;

/**
 * @brief Instrumentation point type for NVBit
 *
 * Controls whether instrumentation is inserted before or after the instruction.
 */
enum class IPointType {
  DEFAULT = -1,  // Not set - use the default_ipoint passed to get_ipoint_from_config
  BEFORE = 0,    // IPOINT_BEFORE - instrument before instruction execution
  AFTER = 1,     // IPOINT_AFTER - instrument after instruction execution
};

/**
 * @brief Defines the type of analysis to be performed on the collected trace
 * data.
 */
enum class AnalysisType {
  /**
   * @brief No analysis is performed.
   */
  ANALYSIS_NONE = 0,
  /**
   * @brief Enables instruction histogram analysis.
   *
   * This corresponds to the `proton_instr_histogram` setting and requires
   * `OPCODE_ONLY` instrumentation.
   */
  PROTON_INSTR_HISTOGRAM = 1,

  /**
   * @brief Enables deadlock detection analysis.
   */
  DEADLOCK_DETECTION = 2,

  /**
   * @brief Enables random delay injection for data race detection.
   */
  RANDOM_DELAY = 3,
};

// Configuration variables
extern uint32_t instr_begin_interval;
extern uint32_t instr_end_interval;
extern int verbose;
extern bool dump_cubin;

// Kernel name filters
extern std::vector<std::string> kernel_filters;

// Instrumentation configuration
// Uses instrument_type_to_index map for O(1) lookup (replaces the old unordered_set)
// Ordered list of enabled instrument types (preserves insertion order for IPOINT mapping)
extern std::vector<InstrumentType> enabled_instrument_types_ordered;
// Map from InstrumentType to its index in enabled_instrument_types_ordered (O(1) lookup)
extern std::unordered_map<InstrumentType, int> instrument_type_to_index;

// Analysis configuration
extern std::unordered_set<AnalysisType> enabled_analysis_types;

// Initialize configuration from environment variables
void init_config_from_env();

// Check if a specific instrumentation type is enabled
bool is_instrument_type_enabled(InstrumentType type);

// Check if any instrumentation type is enabled
bool has_any_instrumentation_enabled();

// Check if a specific analysis type is enabled
bool is_analysis_type_enabled(AnalysisType type);

// Initialize instrumentation configuration
void init_instrumentation(const std::string& instrument_str);

// Initialize analysis configuration
void init_analysis(const std::string& analysis_str);

// Trace format configuration
// 0 = text format
// 1 = NDJSON+Zstd (compressed JSON)
// 2 = NDJSON only (uncompressed JSON, default)
// 3 = CLP Archive
extern int trace_format;

// Zstd compression level (1-22, higher = better compression but slower)
// Default: 9 (balanced compression speed and ratio)
extern int zstd_compression_level;

// Delay value in nanoseconds for random delay instrumentation (max delay)
extern uint32_t g_delay_ns;

// Minimum delay value in nanoseconds (floor for random mode, default: 0)
// Setting min > 0 ensures every thread gets at least this much delay,
// which can help narrow the search space for race detection.
extern uint32_t g_delay_min_ns;

// Delay distribution mode (orthogonal to CTA targeting; see g_delay_cta_target):
// 0 = fixed: all delayed threads sleep `delay_ns` exactly (preserves relative
//     timing, often masks races)
// 1 = random (default): each delayed thread sleeps a random duration in
//     [min_delay_ns, delay_ns] using thread-local entropy (threadIdx, blockIdx,
//     clock) to create asymmetric timing that better exposes data races
extern int g_delay_mode;

// CTA targeting mode (orthogonal to delay distribution; see g_delay_mode):
// 0 = all (default): every CTA's threads receive the delay
// 1 = one_per_cluster: only one CTA per cluster (selected via cluster_seed or
//     g_cluster_cta_id) receives the delay; other CTAs in the cluster skip
//     entirely. Exposes missing inter-CTA synchronization. Requires sm_90+ and
//     a cluster-launched kernel; on non-cluster kernels the device function
//     short-circuits and a host-side WARNING is emitted once per CUfunction.
extern int g_delay_cta_target;

// One-CTA-per-cluster targeting override (only meaningful when
// g_delay_cta_target == 1).
// -1 (default) = random per instrumentation point (uses cluster_seed % cluster_size).
// >= 0        = force every instrumentation point to delay this CTA index in every cluster.
// Useful for deterministic A/B bisection of inter-CTA sync issues, or systematic
// sweeps over CTA indices instead of relying on random coverage.
//
// Precedence vs replay (delay_load_path): when this is set, it overrides the
// per-point cluster_seed loaded from the replay config — i.e. setting both
// makes the replay no longer bit-identical to the recording. A startup warning
// is emitted in that case (see init_config_from_env).
extern int g_cluster_cta_id;

// Warp-targeted delay: explicit bitmask of CTA-local warp ids that receive the
// delay (bit N == 1 means CTA-local warp N is delayed). 0 (default) means "no
// warp filtering" — the warpgroup dispatcher is not engaged and delay applies
// per the active CTA/distribution mode. Currently supports up to 32 warps per
// CTA (one bit per warp); warps >= 32 are silently skipped.
// Set via CUTRACER_DELAY_WARP_MASK environment variable.
extern uint32_t g_delay_warp_mask;

// Warp-targeted delay: warpgroup index (>= 0 selects warps [4N..4N+3]).
// -1 (default) means "no warpgroup targeting". When set, it is resolved to a
// mask on the host side and takes precedence over g_delay_warp_mask (a startup
// warning is emitted if both are set).
// Set via CUTRACER_DELAY_WARPGROUP_ID environment variable.
extern int g_delay_warpgroup_id;

// Probability per PC of enabling delay during recording (random mode only).
// Range [0.0, 1.0]. Default 0.5 preserves the historical 50/50 PC gate. Set
// to 1.0 for deterministic injection — strongly recommended with warp
// targeting, where the 0.5 default halves the active warpgroup PCs and
// usually just adds noise. No effect in replay mode (where per-point
// enabled is loaded from config).
// Set via CUTRACER_DELAY_ENABLE_PROB environment variable.
extern float g_delay_enable_prob;

// User-specified delay injection patterns (optional, overrides DELAY_INJECTION_PATTERNS)
// Comma-separated SASS substrings, e.g. "SYNCS.EXCH,BAR.SYNC"
extern std::vector<std::string> g_delay_patterns;

// Delay dump output path (optional)
// If set, instrumentation points will be written to this JSON file for later replay
extern std::string delay_dump_path;

// Delay load path (optional)
// If set, instrumentation points will be read from this JSON file for replay mode
extern std::string delay_load_path;

// Output directory for all CUTracer files (trace files and log files)
// When set, all output is written to this directory instead of the current directory
// Set via CUTRACER_OUTPUT_DIR environment variable
extern std::string output_dir;

// CPU call stack capture mode at kernel launch
// Set via CUTRACER_CPU_CALLSTACK environment variable
// Values: auto (default), auto_gil, pytorch, backtrace, 0 (disabled), 1 (=auto, backward compat)
enum class CpuCallstackMode {
  AUTO,       // Prefer PyTorch CapturedTraceback, fallback to backtrace()
  AUTO_GIL,   // Like AUTO, but acquires GIL if not held (for Triton launchers)
  PYTORCH,    // Force PyTorch only (empty if unavailable)
  BACKTRACE,  // Force backtrace() only (original behavior)
  DISABLED    // No call stack capture
};
extern CpuCallstackMode cpu_callstack_mode;

// Kernel events recording mode
// Set via CUTRACER_KERNEL_EVENTS environment variable
// Records structured kernel launch events (with optional Python callstack) to a
// separate file. Always uses NDJSON format regardless of CUTRACER_TRACE_FORMAT,
// since kernel events are structured JSON records.
// Callstacks can be deduplicated to minimize file size.
enum class KernelEventsMode {
  DISABLED,  // No kernel events file (default)
  DEDUP,     // Record events with deduplicated callstacks
  FULL,      // Record events with full callstack per launch
  NOSTACK,   // Record events without callstacks (metadata only)
};
extern KernelEventsMode kernel_events_mode;

// GPU channel buffer size for GPU→CPU communication (in bytes)
// Computed from CUTRACER_CHANNEL_RECORDS (number of records the buffer can hold)
// or defaults to 4MB if not set. Smaller values force more frequent flushes,
// which is useful for hang debugging (ensures trace data reaches CPU promptly).
// Set via CUTRACER_CHANNEL_RECORDS environment variable.
extern int channel_buffer_size;

// Instruction category filtering for conditional instrumentation
// If empty, all instructions are instrumented
// If set, only instructions in the specified categories are instrumented
extern std::unordered_set<InstrCategory> enabled_instr_categories;

// Kernel execution time limit in seconds (0 = disabled)
// When set, any kernel running longer than this value is automatically
// terminated with SIGTERM. Acts as a general safety valve independent of
// deadlock detection (does not require -a deadlock_detection).
// Set via CUTRACER_KERNEL_TIMEOUT_S environment variable
extern uint32_t kernel_timeout_s;

// No-data hang detection timeout in seconds (default: 15)
// Independent of deadlock detection (does not require -a deadlock_detection).
// When no trace data arrives for this duration, the kernel is considered hung.
// Catches "silent" hangs where all warps are blocked on synchronization
// primitives with zero trace output — whether the kernel went silent after
// producing some data, or never produced any data at all. Set to 0 to disable.
// Set via CUTRACER_NO_DATA_TIMEOUT_S environment variable
extern uint32_t no_data_timeout_s;

// Trace file size limit in MB (0 = disabled)
// When the trace file exceeds this limit, the process is automatically terminated.
// Set via CUTRACER_TRACE_SIZE_LIMIT_MB environment variable
extern uint32_t trace_size_limit_mb;

// Initialize instruction category filtering from environment variable
void init_instr_categories(const std::string& categories_str);

// Check if a specific instruction category should be instrumented
bool should_instrument_category(InstrCategory category);

// Check if category-based filtering is enabled
bool has_category_filter_enabled();

// IPOINT configuration for instrumentation
//
// Two environment variables control IPOINT behavior:
//
// 1. CUTRACER_INSTRUMENT_IPOINT_UNIFORM=a|b
//    Applies the same IPOINT to ALL enabled instruments.
//    Example: CUTRACER_INSTRUMENT_IPOINT_UNIFORM=b  (all use IPOINT_BEFORE)
//
// 2. CUTRACER_INSTRUMENT_IPOINT=a,b,a,...
//    Per-instrument IPOINT list. Count MUST match the final enabled instruments
//    (including analysis-added ones). Order follows enabled_instrument_types_ordered.
//    Example: CUTRACER_INSTRUMENT=reg_trace,mem_value_trace
//             CUTRACER_INSTRUMENT_IPOINT=b,a  (reg_trace=BEFORE, mem_value_trace=AFTER)
//
// Error if both are set simultaneously.
// Warning if mem_value_trace uses IPOINT_BEFORE (may not capture loaded values).
//
// Values: 'a'/'after' = IPOINT_AFTER, 'b'/'before' = IPOINT_BEFORE

// Uniform IPOINT for all instrumentation (set via CUTRACER_INSTRUMENT_IPOINT_UNIFORM)
extern IPointType uniform_ipoint;

// Per-instrument IPOINT overrides (set via CUTRACER_INSTRUMENT_IPOINT)
// Indexed by position in enabled_instrument_types_ordered
extern std::vector<IPointType> ipoint_overrides;

// Initialize IPOINT configuration from environment variables
void init_instrument_ipoint();
