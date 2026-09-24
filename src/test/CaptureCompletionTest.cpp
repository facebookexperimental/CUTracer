/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "TemporaryDirectoryTest.h"
#include "capture_completion.h"
#include "trace_writer.h"

// TraceWriter references the extern configuration variable from env_config.h.
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
int zstd_compression_level = 9;

namespace {

class CaptureCompletionTest : public cutracer::test::TemporaryDirectoryTest {
 protected:
  std::vector<nlohmann::json> records(const std::string& name) const {
    std::ifstream file(path(name + ".ndjson"));
    std::vector<nlohmann::json> result;
    for (std::string line; std::getline(file, line);) {
      result.push_back(nlohmann::json::parse(line));
    }
    return result;
  }

  bool record(TraceWriter& writer, uint64_t launch_id, uint64_t index) {
    reg_info_t value{};
    value.header.type = MSG_TYPE_REG_INFO;
    value.kernel_launch_id = launch_id;
    value.opcode_id = 1;
    value.pc = 16;
    value.active_mask = 0xffffffff;
    auto trace = TraceRecord::create_reg_trace(nullptr, "EXIT;", index, 0, &value);
    trace.reg_ipoint = "before";
    return writer.write_trace(trace);
  }
};

TEST_F(CaptureCompletionTest, FinishedCaptureWritesCompletionAfterRecords) {
  TraceWriter writer(path("zero"), 2);
  writer.write_metadata({{"type", "kernel_metadata"}});
  writer.flush();
  ASSERT_EQ(records("zero").size(), 1);

  ASSERT_TRUE(record(writer, 0, 0));
  ASSERT_TRUE(record(writer, 0, 1));
  writer.finish(0, true, true);

  const auto rows = records("zero");
  ASSERT_EQ(rows.size(), 4);
  EXPECT_EQ(rows[1]["ipoint"], "before");
  const nlohmann::json expected = {{"type", "capture_completion"}, {"version", 1},
                                   {"grid_launch_id", 0},          {"kernel_completed", true},
                                   {"channel_drained", true},      {"trace_records_written", 2},
                                   {"dropped_records", 0},         {"errors", nlohmann::json::array()},
                                   {"status", "complete"}};
  EXPECT_EQ(rows.back(), expected);
}

TEST_F(CaptureCompletionTest, ClosingWithoutVerifiedCompletionHasNoCompleteFooter) {
  {
    TraceWriter writer(path("unfinished"), 2);
    ASSERT_TRUE(record(writer, 0, 0));
  }
  const auto rows = records("unfinished");
  ASSERT_EQ(rows.size(), 1);
  EXPECT_EQ(rows.back()["type"], "reg_trace");
}

TEST_F(CaptureCompletionTest, CompletionCountsFlushedAndBufferedRecordsOnce) {
  TraceWriter writer(path("batches"), 2, 1024);
  writer.write_metadata({{"type", "kernel_metadata"}});
  for (uint64_t index = 0; index < 10; ++index) {
    ASSERT_TRUE(record(writer, 4, index));
  }
  ASSERT_GT(writer.get_file_size_bytes(), 0);
  writer.finish(4, true, true);
  const auto rows = records("batches");
  ASSERT_EQ(rows.size(), 12);
  EXPECT_EQ(rows.back()["trace_records_written"], 10);
  EXPECT_EQ(rows.back()["dropped_records"], 0);
  EXPECT_EQ(rows.back()["status"], "complete");
}

TEST_F(CaptureCompletionTest, DisabledTraceRemainsIncompleteAfterSuccessfulKernelExit) {
  TraceWriter writer(path("disabled"), 2);
  ASSERT_TRUE(record(writer, 4, 0));
  writer.disable();
  EXPECT_FALSE(record(writer, 4, 1));
  writer.finish(4, true, true);
  const auto footer = records("disabled").back();
  EXPECT_EQ(footer["status"], "incomplete");
  EXPECT_EQ(footer["dropped_records"], 1);
  EXPECT_EQ(footer["errors"], nlohmann::json::array({"trace_disabled"}));
}

TEST_F(CaptureCompletionTest, MalformedAndUnserializableRecordsCannotBeOverwrittenBySuccess) {
  TraceWriter writer(path("malformed"), 2);
  writer.mark_capture_error("invalid_register_counts", true);
  TraceRecord bad{};
  bad.type = MSG_TYPE_REG_INFO;
  bad.sass_instruction = "unknown";
  EXPECT_FALSE(writer.write_trace(bad));
  ASSERT_TRUE(record(writer, 4, 0));
  writer.finish(4, true, true);
  const auto footer = records("malformed").back();
  EXPECT_EQ(footer["status"], "incomplete");
  EXPECT_EQ(footer["dropped_records"], 2);
  EXPECT_EQ(footer["trace_records_written"], 1);
}

TEST_F(CaptureCompletionTest, MissingKernelOrDrainConfirmationIsIncomplete) {
  for (bool drained : {false, true}) {
    const auto name = drained ? "no_kernel" : "no_drain";
    TraceWriter writer(path(name), 2);
    ASSERT_TRUE(record(writer, 2, 0));
    writer.finish(2, !drained, drained);
    EXPECT_EQ(records(name).back()["status"], "incomplete");
  }
}

TEST_F(CaptureCompletionTest, WriteFailureCountsTheWholeBatchWithoutCompletion) {
  if (!std::filesystem::exists("/dev/full")) {
    GTEST_SKIP() << "/dev/full unavailable";
  }
  for (const int mode : {1, 2}) {
    const auto name = "full_" + std::to_string(mode);
    const auto extension = mode == 1 ? ".ndjson.zst" : ".ndjson";
    std::filesystem::create_symlink("/dev/full", path(name + extension));
    TraceWriter writer(path(name), mode, 1024);
    uint64_t attempted = 0;
    bool accepted = true;
    while (accepted && attempted < 32) {
      accepted = record(writer, 1, attempted++);
    }
    ASSERT_FALSE(accepted);
    ASSERT_GT(attempted, 1);
    EXPECT_FALSE(writer.is_enabled());
    writer.flush();  // Retrying a disabled writer must not count the batch twice.
    EXPECT_FALSE(record(writer, 1, attempted++));
    testing::internal::CaptureStderr();
    writer.finish(1, true, true);
    const auto diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_NE(diagnostic.find("capture completion unavailable"), std::string::npos);
    EXPECT_NE(diagnostic.find("trace_records_written=0 dropped_records=" + std::to_string(attempted) + "\n"),
              std::string::npos);
  }
}

TEST_F(CaptureCompletionTest, FinalFlushFailureCountsPreviouslyAcceptedRecords) {
  if (!std::filesystem::exists("/dev/full")) {
    GTEST_SKIP() << "/dev/full unavailable";
  }
  std::filesystem::create_symlink("/dev/full", path("final_flush.ndjson"));
  TraceWriter writer(path("final_flush"), 2);
  for (uint64_t index = 0; index < 4; ++index) {
    ASSERT_TRUE(record(writer, 1, index));
  }
  testing::internal::CaptureStderr();
  writer.finish(1, true, true);
  const auto diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("trace_records_written=0 dropped_records=4\n"), std::string::npos);
}

TEST_F(CaptureCompletionTest, CompressionFailureRemainsSticky) {
  TraceWriter writer(path("compression"), 1, 1024);
  for (uint64_t index = 0; index < 3; ++index) {
    ASSERT_TRUE(record(writer, 1, index));
  }
  std::string noise;
  for (unsigned int i = 0; i < 4096; ++i) {
    noise.push_back(static_cast<char>(32 + (((i * 73) ^ (i >> 3)) % 94)));
  }
  writer.write_metadata({{"type", "kernel_metadata"}, {"noise", noise}});
  writer.flush();
  EXPECT_FALSE(writer.is_enabled());
  EXPECT_FALSE(record(writer, 1, 0));
  testing::internal::CaptureStderr();
  writer.finish(1, true, true);
  const auto diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("trace_records_written=0 dropped_records=4\n"), std::string::npos);
  EXPECT_EQ(std::filesystem::file_size(path("compression.ndjson.zst")), 0);
}

TEST_F(CaptureCompletionTest, SyncFailureReportsWhyCompletionIsUnavailable) {
  if (!std::filesystem::exists("/dev/null")) {
    GTEST_SKIP() << "/dev/null unavailable";
  }
  std::filesystem::create_symlink("/dev/null", path("sync.ndjson"));
  TraceWriter writer(path("sync"), 2);
  ASSERT_TRUE(record(writer, 1, 0));
  testing::internal::CaptureStderr();
  writer.finish(1, true, true);
  const auto diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(writer.is_enabled());
  EXPECT_NE(diagnostic.find("TraceWriter: failed to sync trace data"), std::string::npos);
}

TEST_F(CaptureCompletionTest, CopiedChunkMustBeParsedBeforeCompletion) {
  TraceWriter writer(path("zero"), 2);
  writer.write_metadata({{"type", "kernel_metadata"}});
  CaptureCompletionQueue queue;
  std::thread callback([&] { queue.request({0, true, true, ""}); });
  callback.join();
  writer.flush();
  ASSERT_EQ(records("zero").size(), 1);

  ASSERT_TRUE(record(writer, 0, 0));
  ASSERT_TRUE(record(writer, 0, 1));
  const auto ready = queue.take_at_chunk_boundary();
  ASSERT_EQ(ready.size(), 1);
  writer.finish(ready[0].launch_id, ready[0].kernel_completed, ready[0].channel_drained);

  const auto rows = records("zero");
  ASSERT_EQ(rows.size(), 4);
  EXPECT_EQ(rows[1]["ipoint"], "before");
  const nlohmann::json expected = {{"type", "capture_completion"}, {"version", 1},
                                   {"grid_launch_id", 0},          {"kernel_completed", true},
                                   {"channel_drained", true},      {"trace_records_written", 2},
                                   {"dropped_records", 0},         {"errors", nlohmann::json::array()},
                                   {"status", "complete"}};
  EXPECT_EQ(rows.back(), expected);
  EXPECT_TRUE(queue.take_at_chunk_boundary().empty());
}

TEST_F(CaptureCompletionTest, InterleavedLaunchesKeepTheirOwnWriters) {
  std::map<uint64_t, std::unique_ptr<TraceWriter>> writers;
  for (uint64_t id : {3, 9}) {
    writers[id] = std::make_unique<TraceWriter>(path(std::to_string(id)), 2);
  }
  ASSERT_TRUE(record(*writers[3], 3, 0));
  ASSERT_TRUE(record(*writers[9], 9, 0));
  CaptureCompletionQueue queue;
  queue.request({9, true, true, ""});
  queue.request({3, true, true, ""});
  ASSERT_TRUE(record(*writers[3], 3, 1));
  for (const auto& ready : queue.take_at_chunk_boundary()) {
    writers.at(ready.launch_id)->finish(ready.launch_id, ready.kernel_completed, ready.channel_drained);
  }
  EXPECT_EQ(records("3").back()["trace_records_written"], 2);
  EXPECT_EQ(records("9").back()["trace_records_written"], 1);
  EXPECT_EQ(records("3").back()["grid_launch_id"], 3);
  EXPECT_EQ(records("9").back()["grid_launch_id"], 9);
}

TEST_F(CaptureCompletionTest, GraphDrainIncludesEmptyLaunchAndRetiresTheBatch) {
  TraceWriter active(path("graph_active"), 2);
  TraceWriter empty(path("graph_empty"), 2);
  active.mark_capture_error("graph_launch_completion_unverified");
  empty.mark_capture_error("graph_launch_completion_unverified");
  CaptureCompletionQueue queue;
  queue.register_graph_launch(3);
  queue.register_graph_launch(9);
  ASSERT_TRUE(record(active, 3, 0));
  queue.request_graph_completion(true, true, "");
  ASSERT_TRUE(record(active, 3, 1));
  const auto ready = queue.take_at_chunk_boundary();
  ASSERT_EQ(ready.size(), 2);
  active.finish(ready[0].launch_id, ready[0].kernel_completed, ready[0].channel_drained);
  empty.finish(ready[1].launch_id, ready[1].kernel_completed, ready[1].channel_drained);
  const auto empty_rows = records("graph_empty");
  ASSERT_EQ(empty_rows.size(), 1);
  const nlohmann::json expected = {
      {"type", "capture_completion"}, {"version", 1},
      {"grid_launch_id", 9},          {"kernel_completed", true},
      {"channel_drained", true},      {"trace_records_written", 0},
      {"dropped_records", 0},         {"errors", nlohmann::json::array({"graph_launch_completion_unverified"})},
      {"status", "incomplete"}};
  EXPECT_EQ(empty_rows.back(), expected);
  const auto active_rows = records("graph_active");
  ASSERT_EQ(active_rows.size(), 3);
  EXPECT_EQ(active_rows.back()["trace_records_written"], 2);
  EXPECT_EQ(active_rows.back()["status"], "incomplete");

  queue.register_graph_launch(17);
  queue.request_graph_completion(true, true, "");
  const auto next = queue.take_at_chunk_boundary();
  ASSERT_EQ(next.size(), 1);
  EXPECT_EQ(next[0].launch_id, 17);
  queue.request_graph_completion(true, true, "");
  EXPECT_TRUE(queue.take_at_chunk_boundary().empty());
}

TEST_F(CaptureCompletionTest, GraphRequestPreservesLaunchFailureAfterSuccessfulDrain) {
  CaptureCompletionQueue queue;
  queue.register_graph_launch(4);
  queue.request_graph_completion(false, true, "cuda_graph_launch_or_flush_failed");
  const auto ready = queue.take_at_chunk_boundary();
  ASSERT_EQ(ready.size(), 1);
  EXPECT_EQ(ready[0].launch_id, 4);
  EXPECT_FALSE(ready[0].kernel_completed);
  EXPECT_TRUE(ready[0].channel_drained);
  EXPECT_EQ(ready[0].error, "cuda_graph_launch_or_flush_failed");
}

}  // namespace
