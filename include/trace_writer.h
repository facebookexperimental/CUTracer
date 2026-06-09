/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <zstd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "common.h"
#include "nvbit.h"

enum class TraceMode : std::uint8_t {
  TEXT = 0,
  COMPRESSED_NDJSON = 1,
  UNCOMPRESSED_NDJSON = 2,
  CLP = 3,
};

/**
 * @brief Rich trace record combining GPU trace data with metadata.
 *
 * This structure acts as an intermediate layer between raw GPU communication
 * structs (reg_info_t, mem_addr_access_t, opcode_only_t) and the JSON output format.
 *
 * Benefits:
 * - GPU communication protocol remains unchanged
 * - Easy to add metadata fields without modifying GPU structs
 * - One record = one JSON line (atomic write operation)
 */
struct TraceRecord {
  // ========== Metadata (not in original GPU structs) ==========

  /**
   * @brief CUDA context pointer (for debugging/correlation).
   *
   * Source: recv_thread_fun() args
   * JSON field: "ctx" (hex string)
   */
  CUcontext context;

  /**
   * @brief SASS instruction string for human readability.
   *
   * Source: ctx_state->id_to_sass_map[func][opcode_id]
   * JSON field: "sass"
   */
  std::string sass_instruction;

  /**
   * @brief Per-kernel trace sequence number (monotonically increasing).
   *
   * Used to track trace ordering within a kernel for analysis.
   * Source: Maintained by analysis.cu per kernel_launch_id
   * JSON field: "trace_index"
   */
  uint64_t trace_index;

  /**
   * @brief Host-side timestamp when trace was received (nanoseconds).
   *
   * Source: Current time when trace is processed
   * JSON field: "timestamp"
   */
  uint64_t timestamp;

  // ========== Original trace data ==========

  /**
   * @brief Type of the trace record.
   */
  message_type_t type;

  /**
   * @brief Union containing pointer to original GPU data.
   *
   * Lifetime: Caller ensures the pointed-to struct remains valid
   * until write_trace() completes (typically local variable in analysis.cu).
   */
  union {
    const reg_info_t* reg_info;
    const mem_addr_access_t* mem_access;
    const mem_value_access_t* mem_value_access;
    const opcode_only_t* opcode_only;
    const tma_access_t* tma_access;
  } data;

  /**
   * @brief Register indices from CPU-side static mapping (optional).
   *
   * Source: ctx_state->id_to_reg_indices_map[func][opcode_id]
   * Lifetime: Caller ensures pointed-to RegIndices outlives write_trace() call.
   * nullptr when not available (e.g., lookup failure).
   */
  const RegIndices* reg_indices = nullptr;

  /**
   * @brief Parsed TMA transfer info from nvbit_parse_tma_transfer_info().
   *
   * Set only for MSG_TYPE_TMA_ACCESS records produced by the NVBit-1.8 path.
   * Lifetime: Caller ensures pointed-to TMATransferInfo_t outlives write_trace() call.
   */
  const TMATransferInfo_t* tma_info = nullptr;

  // ========== Constructors for convenience ==========

  /**
   * @brief Create a TraceRecord for reg_info_t.
   */
  static TraceRecord create_reg_trace(CUcontext ctx, const std::string& sass, uint64_t trace_idx, uint64_t ts,
                                      const reg_info_t* reg, const RegIndices* indices = nullptr) {
    TraceRecord record;
    record.context = ctx;
    record.sass_instruction = sass;
    record.trace_index = trace_idx;
    record.timestamp = ts;
    record.type = MSG_TYPE_REG_INFO;
    record.data.reg_info = reg;
    record.reg_indices = indices;
    return record;
  }

  /**
   * @brief Create a TraceRecord for mem_addr_access_t.
   */
  static TraceRecord create_mem_trace(CUcontext ctx, const std::string& sass, uint64_t trace_idx, uint64_t ts,
                                      const mem_addr_access_t* mem) {
    TraceRecord record;
    record.context = ctx;
    record.sass_instruction = sass;
    record.trace_index = trace_idx;
    record.timestamp = ts;
    record.type = MSG_TYPE_MEM_ADDR_ACCESS;
    record.data.mem_access = mem;
    return record;
  }

  /**
   * @brief Create a TraceRecord for mem_value_access_t.
   */
  static TraceRecord create_mem_value_trace(CUcontext ctx, const std::string& sass, uint64_t trace_idx, uint64_t ts,
                                            const mem_value_access_t* mem_value) {
    TraceRecord record;
    record.context = ctx;
    record.sass_instruction = sass;
    record.trace_index = trace_idx;
    record.timestamp = ts;
    record.type = MSG_TYPE_MEM_VALUE_ACCESS;
    record.data.mem_value_access = mem_value;
    return record;
  }

  /**
   * @brief Create a TraceRecord for opcode_only_t.
   */
  static TraceRecord create_opcode_trace(CUcontext ctx, const std::string& sass, uint64_t trace_idx, uint64_t ts,
                                         const opcode_only_t* opcode) {
    TraceRecord record;
    record.context = ctx;
    record.sass_instruction = sass;
    record.trace_index = trace_idx;
    record.timestamp = ts;
    record.type = MSG_TYPE_OPCODE_ONLY;
    record.data.opcode_only = opcode;
    return record;
  }

  /**
   * @brief Create a TraceRecord for tma_access_t.
   *
   * @param tma_info Optional parsed TMA transfer info (NVBit 1.8). When
   *                 non-null, structured tensor metadata is serialized into
   *                 the trace; when null, only raw fields are emitted.
   */
  static TraceRecord create_tma_trace(CUcontext ctx, const std::string& sass, uint64_t trace_idx, uint64_t ts,
                                      const tma_access_t* tma, const TMATransferInfo_t* tma_info = nullptr) {
    TraceRecord record;
    record.context = ctx;
    record.sass_instruction = sass;
    record.trace_index = trace_idx;
    record.timestamp = ts;
    record.type = MSG_TYPE_TMA_ACCESS;
    record.data.tma_access = tma;
    record.tma_info = tma_info;
    return record;
  }
};

/**
 * @brief TraceWriter for trace output in multiple formats.
 *
 * Unified writer supporting three output modes:
 * - Mode 0: Text format (legacy .log files)
 * - Mode 1: NDJSON + Zstd compression (.ndjson.zst)
 * - Mode 2: NDJSON uncompressed (.ndjson)
 *
 * Key features:
 * - Single unified API: write_trace(TraceRecord)
 * - Automatic format dispatch based on trace_mode
 * - Buffering for I/O efficiency
 * - Metadata (ctx, sass, trace_index, timestamp) automatically included
 */
class TraceWriter {
 private:
  std::string filename_;
  FILE* file_handle_;
  int fd_;  // File descriptor for POSIX write() (Mode 1/2 only)
  std::string json_buffer_;
  size_t buffer_threshold_;
  TraceMode trace_mode_;
  std::atomic<bool> enabled_;

  // Serializes every public mutator (write_trace, write_metadata, flush,
  // disable). TraceWriter is reachable concurrently from multiple threads:
  // each per-CTX recv_thread_fun in analysis.cu holds only a shared_lock on
  // the writers map and then calls write_trace() directly on the writer.
  // Without this mutex, concurrent append/+= on json_buffer_ corrupts the
  // std::string's internal heap allocation (libstdc++ _M_mutate double-free,
  // reproduced deterministically by tests/unit/test_trace_writer_stress).
  mutable std::mutex mu_;

  // ========== Mode 1 (Zstd compression) support ==========
  ZSTD_CCtx* zstd_ctx_;                  // Zstd compression context
  std::vector<char> compressed_buffer_;  // Pre-allocated compression output buffer
  int compression_level_;                // Zstd compression level (1-22, default 9)

 public:
  /**
   * @brief Construct TraceWriter.
   *
   * @param filename Base filename (extension added automatically based on mode)
   * @param trace_mode 0 for text, 1 for compressed JSON, 2 for uncompressed JSON
   * @param buffer_threshold Buffer flush threshold (default 1MB)
   */
  TraceWriter(const std::string& filename, int trace_mode, size_t buffer_threshold = 1024 * 1024);

  /**
   * @brief Destructor - flushes remaining data and closes file.
   */
  ~TraceWriter();

  // Disable copy
  TraceWriter(const TraceWriter&) = delete;
  TraceWriter& operator=(const TraceWriter&) = delete;

  /**
   * @brief Write a trace record to output.
   *
   * Mode 0: Formats as text and writes to .log file
   * Mode 1/2: Serializes to JSON and writes to .ndjson[.zst] file
   *
   * @param record Complete trace record with all information
   * @return true if successful, false on error
   */
  bool write_trace(const TraceRecord& record);

  /**
   * @brief Write a metadata JSON object as the first line(s) of the trace file.
   *
   * Only effective in NDJSON modes (mode 1/2/3). In text mode (mode 0),
   * this is a no-op since text mode does not use json_buffer_.
   *
   * The metadata is appended to json_buffer_ without triggering a flush,
   * so it will be written together with subsequent trace data.
   *
   * @param metadata A JSON object to write (e.g., kernel_metadata event)
   */
  void write_metadata(const nlohmann::json& metadata);

  /**
   * @brief Flush buffered data to disk.
   */
  void flush();

  /**
   * @brief Check if writer is functional.
   *
   * Lock-free atomic read; the value may go stale immediately after the call
   * if another thread invokes disable(). Callers should treat it as advisory.
   */
  bool is_enabled() const {
    return enabled_.load(std::memory_order_acquire);
  }

  /**
   * @brief Flush remaining data and disable further writes.
   *
   * Used by trace size limit to truncate a single kernel's trace without
   * killing the process. After calling disable(), all subsequent write_trace()
   * and write_metadata() calls become no-ops (return immediately).
   * Future kernel launches will create fresh TraceWriter instances unaffected
   * by this call.
   */
  void disable();

  /**
   * @brief Get the current file size in bytes.
   *
   * Uses fstat()/ftell() to query the current on-disk size of the trace file.
   * This includes data already flushed to disk but not buffered data.
   *
   * @return Current file size in bytes, or 0 if the file is not open.
   */
  size_t get_file_size_bytes() const;

 private:
  // ========== Output methods ==========

  /**
   * @brief Write record in text format (mode 0).
   */
  void write_text_format(const TraceRecord& record);

  /**
   * @brief Write record in JSON format (mode 1/2).
   */
  void write_json_format(const TraceRecord& record);

  /**
   * @brief Write CLP archive (mode 3).
   */
  void write_clp_archive();

  /**
   * @brief Write uncompressed buffer to file (mode 2).
   */
  void write_uncompressed();

  /**
   * @brief Compress and write buffer to file (mode 1).
   *
   * Compresses json_buffer_ using Zstd and writes to .ndjson.zst file.
   * Each flush creates an independent Zstd frame for incremental writing.
   */
  void write_compressed();

  /**
   * @brief Reliably write data to file with retry logic.
   *
   * Handles partial writes, EINTR interrupts, and detects zero-write deadlock.
   * On fatal error, sets enabled_ = false.
   *
   * @param data Pointer to data buffer
   * @param size Number of bytes to write
   * @param data_type Description for error messages (e.g., "bytes", "compressed bytes")
   * @return true if all data written successfully, false on error
   */
  bool write_data(const char* data, size_t size, const char* data_type);
};
