/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include "trace_writer.h"

#include <errno.h>
#include <fcntl.h>
#include <rapidjson/stringbuffer.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <nlohmann/json.hpp>
#include <sstream>

#include "env_config.h"
#include "trace_rapidjson_internal.h"

// ============================================================================
// PROTOTYPE: rapidjson streaming serializer (A/B vs nlohmann)
//
// Selected via env CUTRACER_JSON_ENGINE = nlohmann (default) | rapidjson | ab.
//   - rapidjson: serialize reg/mem_addr/mem_value/opcode into a reused
//     StringBuffer and append it straight into json_buffer_ (no per-record
//     std::string); tma falls back to nlohmann (deferred — see D3).
//   - ab: write the nlohmann line (canonical) AND SEMANTICALLY compare the
//     rapidjson output per record (parse both, compare as JSON values — key
//     order is irrelevant; no consumer depends on it), with per-type
//     match/mismatch/skipped counters.
// reg/mem/opcode/mem_value still emit keys in lexicographic order, so they are
// also byte-identical (a bonus the counters track); tma may use natural order.
//
// The serializer itself (rj_serialize_to_buffer) lives in trace_rapidjson.cpp
// so it can be unit-tested without libnvbit; this file keeps the engine switch,
// the A/B comparison, and the per-record dispatch.
// ============================================================================
namespace {

enum JsonEngine { ENG_NLOHMANN = 0, ENG_RAPIDJSON = 1, ENG_AB = 2 };

JsonEngine json_engine() {
  static const JsonEngine e = [] {
    const char* s = std::getenv("CUTRACER_JSON_ENGINE");
    if (s != nullptr) {
      if (std::strcmp(s, "rapidjson") == 0) {
        return ENG_RAPIDJSON;
      }
      if (std::strcmp(s, "ab") == 0) {
        return ENG_AB;
      }
    }
    return ENG_NLOHMANN;
  }();
  return e;
}

// A/B per-type counters (global across all TraceWriter instances), indexed by
// message_type_t (REG_INFO=0 .. TMA_ACCESS=4).
struct AbTypeCounters {
  std::atomic<uint64_t> match{0};       // semantically equal (key order ignored)
  std::atomic<uint64_t> byte_match{0};  // subset of `match` that is also byte-identical
  std::atomic<uint64_t> mismatch{0};
  std::atomic<uint64_t> skipped{0};  // rapidjson did not handle this type (e.g. tma fallback)
};

constexpr int kNumMsgTypes = 5;
AbTypeCounters g_ab[kNumMsgTypes];
constexpr const char* kMsgTypeName[kNumMsgTypes] = {"reg_trace", "mem_addr_trace", "opcode_only", "mem_value_trace",
                                                    "tma_trace"};

const char* ab_type_name(int type) {
  return (type >= 0 && type < kNumMsgTypes) ? kMsgTypeName[type] : "unknown";
}

// Semantic A/B compare: parse both NDJSON lines and compare as JSON values.
// nlohmann objects are std::map, so operator== ignores key order while staying
// value/type-sensitive — exactly the order-agnostic oracle we want. Also tracks
// the byte-identical subset as a bonus signal.
void ab_compare(const std::string& nl, const std::string& rj, int type) {
  AbTypeCounters& c = g_ab[(type >= 0 && type < kNumMsgTypes) ? type : 0];
  bool semantic_eq = false;
  try {
    semantic_eq = nlohmann::json::parse(nl) == nlohmann::json::parse(rj);
  } catch (const std::exception&) {
    semantic_eq = false;  // rapidjson emitted invalid JSON
  }
  if (semantic_eq) {
    c.match.fetch_add(1, std::memory_order_relaxed);
    if (nl == rj) {
      c.byte_match.fetch_add(1, std::memory_order_relaxed);
    }
    return;
  }
  const uint64_t n = c.mismatch.fetch_add(1, std::memory_order_relaxed);
  if (n < 10) {  // print the first few divergences only
    size_t i = 0;
    while (i < nl.size() && i < rj.size() && nl[i] == rj[i]) {
      i++;
    }
    fprintf(stderr, "[CUTracer AB MISMATCH type=%s at byte %zu]\n  nl=%.240s\n  rj=%.240s\n", ab_type_name(type), i,
            nl.c_str(), rj.c_str());
  }
}

void ab_record_skipped(int type) {
  g_ab[(type >= 0 && type < kNumMsgTypes) ? type : 0].skipped.fetch_add(1, std::memory_order_relaxed);
}

void ab_print_summary() {
  fprintf(stderr, "[CUTracer AB] semantic compare (rapidjson vs nlohmann), per record type:\n");
  for (int t = 0; t < kNumMsgTypes; t++) {
    const uint64_t m = g_ab[t].match.load();
    const uint64_t bm = g_ab[t].byte_match.load();
    const uint64_t x = g_ab[t].mismatch.load();
    const uint64_t s = g_ab[t].skipped.load();
    if (m == 0 && x == 0 && s == 0) {
      continue;  // type absent in this run
    }
    fprintf(stderr, "  %-16s match=%" PRIu64 " (byte-identical=%" PRIu64 ") mismatch=%" PRIu64 " skipped=%" PRIu64 "\n",
            ab_type_name(t), m, bm, x, s);
  }
}

}  // namespace

// ============================================================================
// Constructor & Destructor
// ============================================================================

TraceWriter::TraceWriter(const std::string& filename, int trace_mode, size_t buffer_threshold)
    : filename_(filename),
      file_handle_(nullptr),
      fd_(-1),
      buffer_threshold_(buffer_threshold),
      trace_mode_(static_cast<TraceMode>(trace_mode)),
      enabled_(true),
      zstd_ctx_(nullptr),
      compression_level_(zstd_compression_level) {  // Use configurable compression level from env_config

  // Validate trace mode
  if (trace_mode < 0 || trace_mode > 3) {
    fprintf(stderr, "TraceWriter: Invalid trace_mode %d (must be 0, 1, 2, or 3)\n", trace_mode);
    enabled_ = false;
    return;
  }

  // Determine filename based on trace mode
  std::string actual_filename;

  if (trace_mode_ == TraceMode::TEXT) {
    // Mode 0: Text format - use FILE* for fprintf compatibility
    actual_filename = filename + ".log";
    int text_fd = open(actual_filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (text_fd < 0) {
      fprintf(stderr, "TraceWriter: Failed to open %s (errno=%d: %s)\n", actual_filename.c_str(), errno,
              strerror(errno));
      enabled_ = false;
      return;
    }
    file_handle_ = fdopen(text_fd, "w");
    if (!file_handle_) {
      fprintf(stderr, "TraceWriter: Failed to fdopen %s (errno=%d: %s)\n", actual_filename.c_str(), errno,
              strerror(errno));
      ::close(text_fd);
      enabled_ = false;
      return;
    }

  } else if (trace_mode_ == TraceMode::COMPRESSED_NDJSON) {
    // Mode 1: NDJSON + Zstd compression - use POSIX write() for reliability
    actual_filename = filename + ".ndjson.zst";

    // Open with O_CREAT | O_WRONLY | O_TRUNC to overwrite across runs
    fd_ = open(actual_filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd_ < 0) {
      fprintf(stderr, "TraceWriter: Failed to open %s (errno=%d)\n", actual_filename.c_str(), errno);
      enabled_ = false;
      return;
    }

    // Initialize Zstd compression context
    zstd_ctx_ = ZSTD_createCCtx();
    if (!zstd_ctx_) {
      fprintf(stderr, "TraceWriter: Failed to initialize Zstd compression context\n");
      close(fd_);
      fd_ = -1;
      enabled_ = false;
      return;
    }

    // Pre-allocate compression buffer to avoid runtime allocation
    size_t max_compressed_size = ZSTD_compressBound(buffer_threshold);
    compressed_buffer_.resize(max_compressed_size);

  } else {  // trace_mode == UNCOMPRESSED_NDJSON || trace_mode == CLP
    // Mode 2/3: NDJSON uncompressed - use POSIX write() for reliability
    actual_filename = filename + ".ndjson";

    // Open with O_CREAT | O_WRONLY | O_TRUNC to overwrite across runs
    fd_ = open(actual_filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd_ < 0) {
      fprintf(stderr, "TraceWriter: Failed to open %s (errno=%d)\n", actual_filename.c_str(), errno);
      enabled_ = false;
      return;
    }
  }
}

TraceWriter::~TraceWriter() {
  // Flush any remaining data
  flush();

  // Close file handle (Mode 0)
  if (file_handle_) {
    fclose(file_handle_);
    file_handle_ = nullptr;
  }

  // Close file descriptor (Mode 1/2/3)
  if (fd_ >= 0) {
    // Ensure all buffered data is persisted to disk before closing.
    // This replaces the per-write fsync in write_data() which caused
    // excessive I/O pressure and potential partial writes under load.
    fsync(fd_);
    close(fd_);
    fd_ = -1;
  }

  // If in CLP mode, write to CLP archive file after fd is closed
  if (trace_mode_ == TraceMode::CLP) {
    write_clp_archive();
  }

  // Release Zstd compression context
  if (zstd_ctx_) {
    ZSTD_freeCCtx(zstd_ctx_);
    zstd_ctx_ = nullptr;
  }

  // PROTOTYPE A/B: report rapidjson-vs-nlohmann per-type comparison totals.
  if (json_engine() == ENG_AB) {
    ab_print_summary();
  }
}

// ============================================================================
// Public API
// ============================================================================

void TraceWriter::write_metadata(const nlohmann::json& metadata) {
  if (!enabled_) {
    return;
  }
  // Text mode (mode 0) does not use json_buffer_; skip.
  if (trace_mode_ == TraceMode::TEXT) {
    return;
  }

  json_buffer_ += metadata.dump() + "\n";
}

bool TraceWriter::write_trace(const TraceRecord& record) {
  if (!enabled_) {
    return false;
  }

  // Dispatch based on trace mode
  if (trace_mode_ == TraceMode::TEXT) {
    write_text_format(record);
  } else {
    write_json_format(record);
  }

  return true;
}

void TraceWriter::flush() {
  // Dispatch based on trace mode
  if (trace_mode_ == TraceMode::COMPRESSED_NDJSON) {
    write_compressed();
  } else if (trace_mode_ == TraceMode::UNCOMPRESSED_NDJSON || trace_mode_ == TraceMode::CLP) {
    write_uncompressed();
  }
  // Mode 0 (text) doesn't buffer, so no flush needed
}

void TraceWriter::disable() {
  flush();
  enabled_ = false;
}

size_t TraceWriter::get_file_size_bytes() const {
  if (fd_ >= 0) {
    struct stat st{};
    if (fstat(fd_, &st) == 0) {
      return static_cast<size_t>(st.st_size);
    }
  } else if (file_handle_) {
    struct stat st{};
    if (fstat(fileno(file_handle_), &st) == 0) {
      return static_cast<size_t>(st.st_size);
    }
  }
  return 0;
}

// ============================================================================
// Private Helpers
// ============================================================================

template <typename T>
void serialize_common_fields(nlohmann::json& j, const T* data) {
  j["grid_launch_id"] = data->kernel_launch_id;
  j["cta"] = {data->cta_id_x, data->cta_id_y, data->cta_id_z};
  j["warp"] = data->warp_id;
  j["opcode_id"] = data->opcode_id;

  std::stringstream pc_ss;
  pc_ss << "0x" << std::hex << data->pc;
  j["pc"] = pc_ss.str();
}

bool TraceWriter::write_data(const char* data, size_t size, const char* data_type) {
  if (fd_ < 0) {
    return false;
  }

  size_t total_written = 0;

  // Retry until all data is written or a fatal error occurs
  while (total_written < size) {
    ssize_t written = write(fd_, data + total_written, size - total_written);

    if (written < 0) {
      // Error occurred
      if (errno == EINTR) {
        // Interrupted by signal, retry
        continue;
      }
      // Fatal error
      fprintf(stderr, "TraceWriter: Fatal write error after %zu of %zu %s (errno=%d: %s)\n", total_written, size,
              data_type, errno, strerror(errno));
      enabled_ = false;
      return false;
    }

    // Check for write() returning 0 (no progress)
    if (written == 0) {
      fprintf(stderr, "TraceWriter: write() returned 0 after %zu of %zu %s (disk full or quota exceeded?)\n",
              total_written, size, data_type);
      enabled_ = false;
      return false;
    }

    total_written += written;
  }

  return true;
}

void TraceWriter::write_uncompressed() {
  if (json_buffer_.empty() || !enabled_) {
    return;
  }

  // CRITICAL FIX: Move json_buffer_ to temp to prevent data corruption
  //
  // Problem: Previously used json_buffer_.data() directly during write_data(),
  // which caused random NULL bytes to appear at line starts in output files.
  //
  // Root cause: If json_buffer_ internal pointer becomes invalid during write
  // (e.g., memory reallocation, or buffer state inconsistency), we'd be
  // writing from a stale pointer. This manifested as:
  //   - Random single NULL bytes replacing '{' at JSON line starts
  //   - Different error lines on each run (non-deterministic)
  //   - Mode 1 unaffected (uses separate compressed_buffer_)
  //
  // Solution: std::move() transfers ownership to temp_buffer BEFORE write,
  // ensuring json_buffer_ is immediately emptied and safe for new data,
  // regardless of write_data() success/failure.
  std::string temp_buffer = std::move(json_buffer_);

  // json_buffer_ is now empty (moved-from state)
  // Write from the temporary buffer
  write_data(temp_buffer.data(), temp_buffer.size(), "bytes");
}

void TraceWriter::write_clp_archive() {
  // Write to CLP archive file from uncompressed ndjson file
  std::string uncompressed_ndjson_file = filename_ + ".ndjson";
  std::string clp_archive_file = filename_ + ".clp";
  std::string clp_run_cmd = "clp-s c --single-file-archive " + clp_archive_file + " " + uncompressed_ndjson_file;
  // run the clp command line to compress and remove the uncompressed ndjson file
  int rc = std::system(clp_run_cmd.c_str());
  if (rc != 0) {
    fprintf(stderr, "TraceWriter: clp-s command line failed with error code %d\n", rc);
    return;
  }
  // remove the uncompressed ndjson file
  int ec = std::remove(uncompressed_ndjson_file.c_str());
  if (ec != 0) {
    fprintf(stderr, "TraceWriter: Failed to remove uncompressed ndjson file with error code %d\n", ec);
    return;
  }
}

void TraceWriter::write_compressed() {
  if (json_buffer_.empty() || !enabled_ || !zstd_ctx_) {
    return;
  }

  // CRITICAL FIX: Move json_buffer_ to temp to prevent data loss
  //
  // Problem: Previously compressed json_buffer_ directly, then cleared only on
  // write success. This caused Mode 1 to lose ~50% of records (e.g., 6,835 of
  // 13,008 records written, 6,173 lost).
  //
  // Root cause: If compression or write failed, json_buffer_ wasn't cleared,
  // causing data to accumulate beyond buffer_threshold_ (1MB). When buffer
  // exceeded the pre-allocated compressed_buffer_ capacity, subsequent
  // compressions failed silently, and all remaining data was lost.
  //
  // Solution: std::move() transfers ownership to temp_buffer BEFORE compression,
  // ensuring json_buffer_ is immediately emptied. This prevents buffer overflow
  // and ensures consistent behavior whether compression/write succeeds or fails.
  std::string temp_buffer = std::move(json_buffer_);

  // json_buffer_ is now empty (moved-from state)
  // Compress from the temporary buffer
  size_t compressed_size = ZSTD_compressCCtx(zstd_ctx_, compressed_buffer_.data(), compressed_buffer_.size(),
                                             temp_buffer.data(), temp_buffer.size(), compression_level_);

  // Check for compression errors
  if (ZSTD_isError(compressed_size)) {
    fprintf(stderr, "TraceWriter: Zstd compression error: %s\n", ZSTD_getErrorName(compressed_size));
    return;  // temp_buffer is automatically destroyed, json_buffer_ remains empty
  }

  // Write the compressed data
  write_data(compressed_buffer_.data(), compressed_size, "compressed bytes");
}

void TraceWriter::serialize_reg_info(nlohmann::json& j, const reg_info_t* reg, const RegIndices* indices) {
  if (!reg) {
    return;
  }

  using json = nlohmann::json;

  serialize_common_fields(j, reg);

  // CRITICAL: Transpose register array
  // C layout: reg_vals[thread][reg] → JSON: regs[reg][thread]
  // This ensures all values for the same register across all threads
  // are grouped together in the JSON output.
  json::array_t regs_array;
  for (int reg_idx = 0; reg_idx < reg->num_regs; reg_idx++) {
    json::array_t thread_vals;
    for (int thread = 0; thread < 32; thread++) {
      thread_vals.emplace_back(reg->reg_vals[thread][reg_idx]);
    }
    regs_array.emplace_back(thread_vals);
  }
  j["regs"] = regs_array;

  // Add register indices from CPU-side static mapping
  if (indices && !indices->reg_indices.empty()) {
    json::array_t regs_indices_array;
    for (auto idx : indices->reg_indices) {
      regs_indices_array.emplace_back(idx);
    }
    j["regs_indices"] = regs_indices_array;
  }

  // Add unified registers if present
  if (reg->num_uregs > 0) {
    json::array_t uregs_array;
    for (int i = 0; i < reg->num_uregs; i++) {
      uregs_array.emplace_back(reg->ureg_vals[i]);
    }
    j["uregs"] = uregs_array;

    // Add unified register indices from CPU-side static mapping
    if (indices && !indices->ureg_indices.empty()) {
      json::array_t uregs_indices_array;
      for (auto idx : indices->ureg_indices) {
        uregs_indices_array.emplace_back(idx);
      }
      j["uregs_indices"] = uregs_indices_array;
    }
  }
}

void TraceWriter::serialize_mem_access(nlohmann::json& j, const mem_addr_access_t* mem) {
  if (!mem) {
    return;
  }

  serialize_common_fields(j, mem);

  // Convert address array (32 addresses)
  std::vector<uint64_t> addrs(mem->addrs, mem->addrs + 32);
  j["addrs"] = addrs;
}

void TraceWriter::serialize_opcode_only(nlohmann::json& j, const opcode_only_t* opcode) {
  if (!opcode) {
    return;
  }

  serialize_common_fields(j, opcode);
}

void TraceWriter::serialize_mem_value_access(nlohmann::json& j, const mem_value_access_t* mem) {
  if (!mem) {
    return;
  }

  using json = nlohmann::json;

  serialize_common_fields(j, mem);

  // Memory access metadata
  j["mem_space"] = mem->mem_space;
  j["is_load"] = (mem->is_load == 1);
  j["access_size"] = mem->access_size;

  // Convert address array (32 addresses)
  std::vector<uint64_t> addrs(mem->addrs, mem->addrs + 32);
  j["addrs"] = addrs;

  // Convert values array (32 lanes x up to 4 registers based on access_size)
  // Only include registers needed for the access size
  int regs_needed = (mem->access_size + 3) / 4;
  if (regs_needed > 4) {
    regs_needed = 4;
  }

  json::array_t values_array;
  for (int lane = 0; lane < 32; lane++) {
    json::array_t lane_vals;
    for (int r = 0; r < regs_needed; r++) {
      lane_vals.emplace_back(mem->values[lane][r]);
    }
    values_array.emplace_back(lane_vals);
  }
  j["values"] = values_array;
}

// Helper: format a 64-bit unsigned int as a "0x..." hex string.
static std::string hex64(uint64_t v) {
  std::stringstream ss;
  ss << "0x" << std::hex << v;
  return ss.str();
}

// Helper: look up a string from an InstrType enum-string table. Falls back to
// the integer value if the enum is out of range.
template <size_t N>
static std::string enum_str(const char* const (&table)[N], int idx) {
  if (idx < 0 || static_cast<size_t>(idx) >= N) {
    return std::to_string(idx);
  }
  return table[idx];
}

// Serialize the parsed tiled-mode tensor map fields. Only call when
// info.is_tensor is true and info.tensor.mode == TILED.
static void serialize_tma_tiled_map(nlohmann::json& j, const TMATransferInfo_t& info) {
  const auto& tiled = info.tensor.map.tiled;
  j["data_type"] = enum_str(InstrType::CUtensorMapDataTypeStr, static_cast<int>(tiled.tensorDataType));
  j["data_type_id"] = static_cast<int>(tiled.tensorDataType);
  j["rank"] = tiled.tensorRank;
  j["global_address"] = hex64(reinterpret_cast<uint64_t>(tiled.globalAddress));
  j["global_dim"] = std::vector<uint64_t>(tiled.globalDim, tiled.globalDim + InstrType::MAX_TMA_DIM);
  j["global_strides"] = std::vector<uint64_t>(tiled.globalStrides, tiled.globalStrides + InstrType::MAX_TMA_DIM - 1);
  j["box_dim"] = std::vector<uint32_t>(tiled.boxDim, tiled.boxDim + InstrType::MAX_TMA_DIM);
  j["element_strides"] = std::vector<uint32_t>(tiled.elementStrides, tiled.elementStrides + InstrType::MAX_TMA_DIM);
  j["interleave"] = enum_str(InstrType::CUtensorMapInterleaveStr, static_cast<int>(tiled.interleave));
  j["interleave_id"] = static_cast<int>(tiled.interleave);
  j["swizzle"] = enum_str(InstrType::CUtensorMapSwizzleStr, static_cast<int>(tiled.swizzle));
  j["swizzle_id"] = static_cast<int>(tiled.swizzle);
  j["l2_promotion"] = enum_str(InstrType::CUtensorMapL2promotionStr, static_cast<int>(tiled.l2Promotion));
  j["oob_fill"] = enum_str(InstrType::CUtensorMapFloatOOBfillStr, static_cast<int>(tiled.oobFill));
}

static void serialize_tma_transfer_info(nlohmann::json& j, const TMATransferInfo_t& info) {
  using json = nlohmann::json;

  j["is_bulk"] = info.is_bulk;
  j["is_tensor"] = info.is_tensor;
  j["is_prefetch"] = info.is_prefetch;
  j["is_multicast"] = info.is_multicast;
  j["transfer_count"] = info.transfer_count;
  j["transfer_size"] = info.transfer_size;
  j["byte_count"] = info.byte_count;
  j["multicast_cta_mask"] = info.multicast_cta_mask;
  j["src_memspace"] = enum_str(InstrType::MemorySpaceStr, static_cast<int>(info.src_memspace));
  j["dst_memspace"] = enum_str(InstrType::MemorySpaceStr, static_cast<int>(info.dst_memspace));

  // Shared-memory destination addresses (UTMALDG: global → shared)
  if (info.dst_memspace == InstrType::MemorySpace::SHARED ||
      info.dst_memspace == InstrType::MemorySpace::DISTRIBUTED_SHARED) {
    json dst;
    dst["data_address"] = info.dst.shared.data_address;
    dst["data_address_offset"] = info.dst.shared.data_address_offset;
    dst["data_address_cluster_cta_id"] = info.dst.shared.data_address_cluster_cta_id;
    dst["is_mbar_valid"] = info.dst.shared.is_mbar_valid;
    if (info.dst.shared.is_mbar_valid) {
      dst["mbar_address"] = info.dst.shared.mbar_address;
      dst["mbar_address_offset"] = info.dst.shared.mbar_address_offset;
      dst["mbar_address_cluster_cta_id"] = info.dst.shared.mbar_address_cluster_cta_id;
    }
    j["dst"] = std::move(dst);
  }

  // Shared-memory source addresses (UTMASTG: shared → global)
  if (info.src_memspace == InstrType::MemorySpace::SHARED ||
      info.src_memspace == InstrType::MemorySpace::DISTRIBUTED_SHARED) {
    json src;
    src["data_address"] = info.src.shared.data_address;
    src["data_address_offset"] = info.src.shared.data_address_offset;
    src["data_address_cluster_cta_id"] = info.src.shared.data_address_cluster_cta_id;
    src["is_mbar_valid"] = info.src.shared.is_mbar_valid;
    if (info.src.shared.is_mbar_valid) {
      src["mbar_address"] = info.src.shared.mbar_address;
      src["mbar_address_offset"] = info.src.shared.mbar_address_offset;
      src["mbar_address_cluster_cta_id"] = info.src.shared.mbar_address_cluster_cta_id;
    }
    j["src"] = std::move(src);
  }

  if (!info.is_tensor) {
    return;
  }

  json tensor;
  tensor["dim"] = info.tensor.dim;
  tensor["mode"] = enum_str(InstrType::TMALoadModeStr, static_cast<int>(info.tensor.mode));
  tensor["coords"] = std::vector<int32_t>(info.tensor.coords, info.tensor.coords + InstrType::MAX_TMA_DIM);
  tensor["intended_transfer_count"] = info.tensor.intended_transfer_count;
  tensor["oob_transfer_count"] = info.tensor.oob_transfer_count;

  // Only TILED mode is fully serialized today — GEMM/Triton kernels use TILED.
  // IM2COL variants would emit different sub-fields; leave as TODO.
  if (info.tensor.mode == InstrType::TMALoadMode::TILED) {
    json tiled;
    serialize_tma_tiled_map(tiled, info);
    tensor["tiled"] = std::move(tiled);
  }

  j["tensor"] = std::move(tensor);
}

void TraceWriter::serialize_tma_access(nlohmann::json& j, const tma_access_t* tma, const TMATransferInfo_t* info) {
  if (!tma) {
    return;
  }

  serialize_common_fields(j, tma);

  // Always emit the captured handle size for debugging / re-parse.
  j["tma_param_size"] = tma->tma_param_size;

  if (info != nullptr) {
    nlohmann::json info_json;
    serialize_tma_transfer_info(info_json, *info);
    j["tma_transfer_info"] = std::move(info_json);

    // Stable per-tensor key for downstream analyzer dedup. Prefer the parsed
    // global tensor address; for non-tensor (bulk) transfers fall back to the
    // raw bulk source address.
    if (info->is_tensor && info->tensor.mode == InstrType::TMALoadMode::TILED) {
      j["desc_addr"] = hex64(reinterpret_cast<uint64_t>(info->tensor.map.tiled.globalAddress));
    } else if (info->is_bulk) {
      j["desc_addr"] = hex64(info->src.global.bulk_copy_address);
    }
  }
}

// ============================================================================
// Format-specific output methods
// ============================================================================

void TraceWriter::write_text_format(const TraceRecord& record) {
  if (!file_handle_) {
    return;
  }

  // Dispatch by trace type
  switch (record.type) {
    case MSG_TYPE_REG_INFO: {
      const reg_info_t* ri = record.data.reg_info;

      // Print header line
      fprintf(file_handle_, "CTX %p - CTA %d,%d,%d - warp %d - %s:\n", record.context, ri->cta_id_x, ri->cta_id_y,
              ri->cta_id_z, ri->warp_id, record.sass_instruction.c_str());

      // Print register values
      for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
        fprintf(file_handle_, "  * ");
        for (int i = 0; i < 32; i++) {
          fprintf(file_handle_, "Reg%d_T%02d: 0x%08x ", reg_idx, i, ri->reg_vals[i][reg_idx]);
        }
        fprintf(file_handle_, "\n");
      }

      // Print uniform register values (if present)
      if (ri->num_uregs > 0) {
        fprintf(file_handle_, "  * UR: ");
        for (int i = 0; i < ri->num_uregs; i++) {
          fprintf(file_handle_, "UR%d: 0x%08x ", i, ri->ureg_vals[i]);
        }
        fprintf(file_handle_, "\n");
      }

      fprintf(file_handle_, "\n");
      break;
    }

    case MSG_TYPE_MEM_ADDR_ACCESS: {
      const mem_addr_access_t* mem = record.data.mem_access;

      // Print header
      fprintf(file_handle_, "CTX %p - kernel_launch_id %ld - CTA %d,%d,%d - warp %d - PC %ld - %s:\n", record.context,
              mem->kernel_launch_id, mem->cta_id_x, mem->cta_id_y, mem->cta_id_z, mem->warp_id, mem->pc,
              record.sass_instruction.c_str());

      // Print memory addresses
      fprintf(file_handle_, "  Memory Addresses:\n  * ");
      int printed = 0;
      for (int i = 0; i < 32; i++) {
        if (mem->addrs[i] != 0) {
          fprintf(file_handle_, "T%02d: 0x%016lx ", i, mem->addrs[i]);
          printed++;
          if (printed % 4 == 0 && i < 31) {
            fprintf(file_handle_, "\n    ");
          }
        }
      }
      fprintf(file_handle_, "\n\n");
      break;
    }

    case MSG_TYPE_OPCODE_ONLY: {
      const opcode_only_t* oi = record.data.opcode_only;

      fprintf(file_handle_, "CTX %p - kernel_launch_id %ld - CTA %d,%d,%d - warp %d - PC %ld - %s\n\n", record.context,
              oi->kernel_launch_id, oi->cta_id_x, oi->cta_id_y, oi->cta_id_z, oi->warp_id, oi->pc,
              record.sass_instruction.c_str());
      break;
    }

    default:
      fprintf(stderr, "TraceWriter: Unknown message type %d in text mode\n", record.type);
      break;
  }

  fflush(file_handle_);
}

std::string TraceWriter::build_nlohmann_line(const TraceRecord& record) {
  try {
    using json = nlohmann::json;
    json j;

    // Type string
    switch (record.type) {
      case MSG_TYPE_REG_INFO:
        j["type"] = "reg_trace";
        break;
      case MSG_TYPE_MEM_ADDR_ACCESS:
        j["type"] = "mem_addr_trace";
        j["ipoint"] = "B";  // IPOINT_BEFORE
        break;
      case MSG_TYPE_MEM_VALUE_ACCESS:
        j["type"] = "mem_value_trace";
        j["ipoint"] = "A";  // IPOINT_AFTER
        break;
      case MSG_TYPE_OPCODE_ONLY:
        j["type"] = "opcode_only";
        break;
      case MSG_TYPE_TMA_ACCESS:
        j["type"] = "tma_trace";
        break;
      default:
        fprintf(stderr, "TraceWriter: Unknown message type %d\n", record.type);
        return std::string();
    }

    // Context pointer (as hex string)
    std::stringstream ss;
    ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(record.context);
    j["ctx"] = ss.str();

    // SASS instruction is no longer emitted per-record in JSON mode.
    // It is available in the kernel_metadata "instructions" table (keyed by opcode_id).
    // The Python TraceReader injects "sass" back into records on read for compatibility.

    j["trace_index"] = record.trace_index;
    j["timestamp"] = record.timestamp;

    // Serialize data (dispatch by type)
    switch (record.type) {
      case MSG_TYPE_REG_INFO:
        serialize_reg_info(j, record.data.reg_info, record.reg_indices);
        break;
      case MSG_TYPE_MEM_ADDR_ACCESS:
        serialize_mem_access(j, record.data.mem_access);
        break;
      case MSG_TYPE_MEM_VALUE_ACCESS:
        serialize_mem_value_access(j, record.data.mem_value_access);
        break;
      case MSG_TYPE_OPCODE_ONLY:
        serialize_opcode_only(j, record.data.opcode_only);
        break;
      case MSG_TYPE_TMA_ACCESS:
        serialize_tma_access(j, record.data.tma_access, record.tma_info);
        break;
    }

    return j.dump();
  } catch (const std::exception& e) {
    fprintf(stderr, "TraceWriter: JSON error in build_nlohmann_line: %s\n", e.what());
    return std::string();
  }
}

void TraceWriter::write_json_format(const TraceRecord& record) {
  // Emit one NDJSON line via the selected engine (PROTOTYPE). Default nlohmann;
  // rapidjson covers reg/mem/opcode/mem_value (tma falls back to nlohmann); ab
  // writes the nlohmann line and semantically compares rapidjson against it.
  const JsonEngine eng = json_engine();

  // Append `[p, p+n)` as one NDJSON line and flush if the buffer crossed the
  // threshold. Shared by all engine paths so the append/flush logic lives once.
  const auto append_line = [this](const char* p, size_t n) {
    json_buffer_.append(p, n);
    json_buffer_ += '\n';
    if (json_buffer_.size() >= buffer_threshold_) {
      if (trace_mode_ == TraceMode::COMPRESSED_NDJSON) {
        write_compressed();
      } else {
        write_uncompressed();
      }
    }
  };

  if (eng == ENG_NLOHMANN) {
    const std::string line = build_nlohmann_line(record);
    if (!line.empty()) {
      append_line(line.data(), line.size());
    }
    return;
  }

  // rapidjson / ab: serialize into a reused per-thread buffer (no per-record
  // std::string on the rapidjson hot path).
  thread_local rapidjson::StringBuffer sb;
  const bool rj_ok = cutracer::rj_serialize_to_buffer(record, sb);

  if (eng == ENG_RAPIDJSON) {
    if (rj_ok) {
      append_line(sb.GetString(), sb.GetSize());  // direct — no intermediate string
    } else {
      const std::string line = build_nlohmann_line(record);  // tma / unhandled fallback
      if (!line.empty()) {
        append_line(line.data(), line.size());
      }
    }
    return;
  }

  // ENG_AB: canonical output stays nlohmann; compare rapidjson semantically.
  const std::string nl = build_nlohmann_line(record);
  if (rj_ok) {
    ab_compare(nl, std::string(sb.GetString(), sb.GetSize()), static_cast<int>(record.type));
  } else {
    ab_record_skipped(static_cast<int>(record.type));
  }
  if (!nl.empty()) {
    append_line(nl.data(), nl.size());
  }
}
