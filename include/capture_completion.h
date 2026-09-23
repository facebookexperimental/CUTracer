/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct CaptureCompletionRequest {
  uint64_t launch_id;
  bool kernel_completed;
  bool channel_drained;
  std::string error;
};

class CaptureCompletionQueue {
 public:
  void request(CaptureCompletionRequest request) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(std::move(request));
  }

  void register_graph_launch(uint64_t launch_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    graph_launches_.push_back(launch_id);
  }

  void request_graph_completion(bool kernel_completed, bool channel_drained, const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto launch_id : graph_launches_) {
      pending_.push_back({launch_id, kernel_completed, channel_drained, error});
    }
    graph_launches_.clear();
  }

  // ChannelHost ACKs after copying, before the receiver parses the chunk.
  // Only the receiver may consume this queue, between complete chunks.
  std::vector<CaptureCompletionRequest> take_at_chunk_boundary() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<CaptureCompletionRequest> ready;
    ready.swap(pending_);
    return ready;
  }

 private:
  std::mutex mutex_;
  std::vector<CaptureCompletionRequest> pending_;
  std::vector<uint64_t> graph_launches_;
};
