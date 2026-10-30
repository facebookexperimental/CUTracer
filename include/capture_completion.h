/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <chrono>
#include <condition_variable>
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
    taken_barrier_ = barrier_;
    std::vector<CaptureCompletionRequest> ready;
    ready.swap(pending_);
    return ready;
  }

  // The receiver calls this after finishing the requests it took.
  void finished_chunk_boundary() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_barrier_ = taken_barrier_;
    }
    boundary_.notify_all();
  }

  // Waits until the receiver has parsed every chunk it copied before this call
  // and finished the requests enqueued before it. False after the timeout.
  bool wait_for_receiver(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto barrier = ++barrier_;
    return boundary_.wait_for(lock, timeout, [&] { return finished_barrier_ >= barrier; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable boundary_;
  std::vector<CaptureCompletionRequest> pending_;
  std::vector<uint64_t> graph_launches_;
  uint64_t barrier_ = 0;
  uint64_t taken_barrier_ = 0;
  uint64_t finished_barrier_ = 0;
};
