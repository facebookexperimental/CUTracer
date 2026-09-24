/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <pthread.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

namespace cutracer {

// Suppress only CUDA callbacks caused by this thread's tool code. In particular,
// context teardown must not suppress another thread's matching launch exit.
class ScopedCudaCallbackSuppression {
 public:
  ScopedCudaCallbackSuppression() : previous_(suppressed_) {
    suppressed_ = true;
  }

  ~ScopedCudaCallbackSuppression() {
    suppressed_ = previous_;
  }

  ScopedCudaCallbackSuppression(const ScopedCudaCallbackSuppression&) = delete;
  ScopedCudaCallbackSuppression& operator=(const ScopedCudaCallbackSuppression&) = delete;
  ScopedCudaCallbackSuppression(ScopedCudaCallbackSuppression&&) = delete;
  ScopedCudaCallbackSuppression& operator=(ScopedCudaCallbackSuppression&&) = delete;

  static bool is_suppressed() {
    return suppressed_;
  }

 private:
  inline static thread_local bool suppressed_ = false;
  bool previous_;
};

// A callback owns one acquisition of the recursive event mutex. Launch entries
// may retain it across the driver call; an exit releases only its own entry's
// acquisition, even when that entry is no longer at the top of the thread's stack.
class ScopedCudaLaunchCallback {
 public:
  struct PendingLaunch {
    const void* context;
    int api;
    std::optional<uint64_t> launch_id;
    bool retained_lock;
  };

  using PendingLaunches = std::vector<PendingLaunch>;

  struct LaunchExit {
    std::optional<PendingLaunch> launch;
    bool in_order = false;
  };

  ScopedCudaLaunchCallback(pthread_mutex_t& mutex, PendingLaunches& pending_launches)
      : mutex_(mutex), pending_launches_(pending_launches) {
    pthread_mutex_lock(&mutex_);
  }

  ~ScopedCudaLaunchCallback() {
    if (release_entry_lock_) {
      pthread_mutex_unlock(&mutex_);
    }
    if (!retain_entry_lock_) {
      pthread_mutex_unlock(&mutex_);
    }
  }

  ScopedCudaLaunchCallback(const ScopedCudaLaunchCallback&) = delete;
  ScopedCudaLaunchCallback& operator=(const ScopedCudaLaunchCallback&) = delete;
  ScopedCudaLaunchCallback(ScopedCudaLaunchCallback&&) = delete;
  ScopedCudaLaunchCallback& operator=(ScopedCudaLaunchCallback&&) = delete;

  void enter(PendingLaunch launch) {
    pending_launches_.push_back(launch);
    retain_entry_lock_ = launch.retained_lock;
  }

  LaunchExit take_pending_launch(const void* context, int api) {
    const auto entry = std::find_if(
        pending_launches_.rbegin(), pending_launches_.rend(),
        [context, api](const PendingLaunch& launch) { return launch.context == context && launch.api == api; });
    if (entry == pending_launches_.rend()) {
      // An unrelated exit cannot prove that any existing entry has ended.
      return {};
    }
    const bool in_order = entry == pending_launches_.rbegin();
    const auto launch = *entry;
    pending_launches_.erase(std::next(entry).base());
    release_entry_lock_ = launch.retained_lock;
    return {launch, in_order};
  }

 private:
  pthread_mutex_t& mutex_;
  PendingLaunches& pending_launches_;
  bool retain_entry_lock_ = false;
  bool release_entry_lock_ = false;
};

}  // namespace cutracer
