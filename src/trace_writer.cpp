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

#include <mutex>
#include <nlohmann/json.hpp>

#include "env_config.h"
#include "trace_rapidjson_internal.h"

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
}

// ============================================================================
// Public API
// ============================================================================

void TraceWriter::write_metadata(const nlohmann::json& metadata) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  // Text mode (mode 0) does not use json_buffer_; skip.
  if (trace_mode_ == TraceMode::TEXT) {
    return;
  }

  json_buffer_ += metadata.dump() + "\n";
}

bool TraceWriter::write_trace(const TraceRecord& record) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!enabled_.load(std::memory_order_relaxed)) {
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
  std::lock_guard<std::mutex> lk(mu_);
  // Dispatch based on trace mode
  if (trace_mode_ == TraceMode::COMPRESSED_NDJSON) {
    write_compressed();
  } else if (trace_mode_ == TraceMode::UNCOMPRESSED_NDJSON || trace_mode_ == TraceMode::CLP) {
    write_uncompressed();
  }
  // Mode 0 (text) doesn't buffer, so no flush needed
}

void TraceWriter::disable() {
  std::lock_guard<std::mutex> lk(mu_);
  // Inlined flush() body — we already hold the lock, can't recursively re-take it.
  if (trace_mode_ == TraceMode::COMPRESSED_NDJSON) {
    write_compressed();
  } else if (trace_mode_ == TraceMode::UNCOMPRESSED_NDJSON || trace_mode_ == TraceMode::CLP) {
    write_uncompressed();
  }
  enabled_.store(false, std::memory_order_release);
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
  // Caller (write_trace / write_metadata / flush / disable) holds mu_, so
  // json_buffer_ is stable here. The earlier "NULL bytes at line starts" /
  // "Mode 1 unaffected" symptom was the std::string::append data race fixed
  // by mu_ in the public mutators — not anything about pointer staleness
  // during write_data(). The old std::move(json_buffer_) -> temp_buffer dance
  // did not actually address that race.
  if (json_buffer_.empty() || !enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  write_data(json_buffer_.data(), json_buffer_.size(), "bytes");
  json_buffer_.clear();
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
  // Caller holds mu_; json_buffer_ is stable. Clear it unconditionally after
  // attempting to compress + write so we never let buffered data grow past
  // buffer_threshold_ on a transient compression/write failure — the prior
  // bug of "Mode 1 lost ~50% of records" was caused by skipping the clear
  // on error and letting the next flush exceed compressed_buffer_ capacity.
  if (json_buffer_.empty() || !enabled_.load(std::memory_order_relaxed) || !zstd_ctx_) {
    return;
  }

  size_t compressed_size = ZSTD_compressCCtx(zstd_ctx_, compressed_buffer_.data(), compressed_buffer_.size(),
                                             json_buffer_.data(), json_buffer_.size(), compression_level_);

  // Always clear the source buffer, success or fail (see comment above).
  json_buffer_.clear();

  if (ZSTD_isError(compressed_size)) {
    fprintf(stderr, "TraceWriter: Zstd compression error: %s\n", ZSTD_getErrorName(compressed_size));
    return;
  }

  write_data(compressed_buffer_.data(), compressed_size, "compressed bytes");
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

void TraceWriter::write_json_format(const TraceRecord& record) {
  // Serialize one NDJSON line via the rapidjson streaming writer into a reused
  // per-thread buffer, then append it directly (no per-record std::string).
  thread_local rapidjson::StringBuffer sb;
  if (!cutracer::rj_serialize_to_buffer(record, sb)) {
    // rj_serialize_to_buffer only fails for an unknown type or null data (it
    // returns Writer::IsComplete() otherwise); skip rather than emit garbage.
    fprintf(stderr, "TraceWriter: skipping unserializable record (type %d)\n", static_cast<int>(record.type));
    return;
  }

  json_buffer_.append(sb.GetString(), sb.GetSize());
  json_buffer_ += '\n';
  if (json_buffer_.size() >= buffer_threshold_) {
    if (trace_mode_ == TraceMode::COMPRESSED_NDJSON) {
      write_compressed();
    } else {
      write_uncompressed();
    }
  }
}
