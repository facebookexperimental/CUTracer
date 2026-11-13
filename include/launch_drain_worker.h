/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace cutracer {

// Runs one CUDA context's post-launch work (waiting for a launch to finish,
// flushing the trace channel, posting the capture completion) on a thread of
// its own, one job at a time in submission order. The launching thread decides
// whether to wait for its job or to return to the application.
//
// Jobs must not throw, and must not wait for a ticket themselves.
class LaunchDrainWorker {
 public:
  using Job = std::function<void()>;

  // `on_start` runs on the worker thread before its first job, for example to
  // make a CUDA context current there; `on_stop` runs there after the last
  // one. Neither may call wait() or stop().
  explicit LaunchDrainWorker(std::function<void()> on_start = {}, std::function<void()> on_stop = {})
      : thread_([this, on_start = std::move(on_start), on_stop = std::move(on_stop)] {
          if (on_start) {
            on_start();
          }
          run();
          if (on_stop) {
            on_stop();
          }
        }),
        worker_id_(thread_.get_id()) {
  }

  ~LaunchDrainWorker() {
    stop();
  }

  LaunchDrainWorker(const LaunchDrainWorker&) = delete;
  LaunchDrainWorker& operator=(const LaunchDrainWorker&) = delete;
  LaunchDrainWorker(LaunchDrainWorker&&) = delete;
  LaunchDrainWorker& operator=(LaunchDrainWorker&&) = delete;

  // Queues `job` and returns its ticket. Once the worker is stopping, runs the
  // job on the calling thread instead and returns ticket 0, which is always
  // complete.
  uint64_t submit(Job job) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopping_) {
      lock.unlock();
      job();
      return 0;
    }
    jobs_.push_back(std::move(job));
    const uint64_t ticket = ++submitted_;
    lock.unlock();
    work_ready_.notify_one();
    return ticket;
  }

  // Blocks until the job with `ticket`, and every job queued before it, has
  // run. Returns at once on the worker thread, which cannot wait for itself.
  void wait(uint64_t ticket) {
    if (std::this_thread::get_id() == worker_id_) {
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    work_done_.wait(lock, [&] { return completed_ >= ticket; });
  }

  // Blocks until every job queued so far has run.
  void wait_idle() {
    uint64_t last = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      last = submitted_;
    }
    wait(last);
  }

  // Runs the jobs still queued, then ends the worker thread. Later submissions
  // run on their caller. Safe to call more than once, from one thread at a time.
  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    work_ready_.notify_one();
    if (thread_.joinable() && std::this_thread::get_id() != worker_id_) {
      thread_.join();
    }
  }

  std::thread::id thread_id() const {
    return worker_id_;
  }

 private:
  void run() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      work_ready_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
      if (jobs_.empty()) {
        return;
      }
      Job job = std::move(jobs_.front());
      jobs_.pop_front();
      lock.unlock();
      job();
      lock.lock();
      ++completed_;
      work_done_.notify_all();
    }
  }

  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  std::deque<Job> jobs_;
  uint64_t submitted_ = 0;
  uint64_t completed_ = 0;
  bool stopping_ = false;
  // Declared after the members the thread uses, so it starts only once they
  // exist. worker_id_ is set right after the thread starts; only wait() and
  // stop() read it, and on_start and on_stop may not call them.
  std::thread thread_;
  std::thread::id worker_id_;
};

}  // namespace cutracer
