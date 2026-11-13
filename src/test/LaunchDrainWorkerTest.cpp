/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "launch_drain_worker.h"

namespace {

using cutracer::LaunchDrainWorker;

TEST(LaunchDrainWorker, RunsJobsInSubmissionOrderOffTheCallingThread) {
  std::vector<int> order;
  std::vector<std::thread::id> threads;
  std::mutex record;
  LaunchDrainWorker worker;
  uint64_t last = 0;
  for (int i = 0; i < 5; ++i) {
    last = worker.submit([&, i] {
      std::lock_guard<std::mutex> lock(record);
      order.push_back(i);
      threads.push_back(std::this_thread::get_id());
    });
  }
  worker.wait(last);

  const std::vector<int> expected{0, 1, 2, 3, 4};
  EXPECT_EQ(order, expected);
  for (const auto& id : threads) {
    EXPECT_NE(id, std::this_thread::get_id());
    EXPECT_EQ(id, threads.front());
  }
}

TEST(LaunchDrainWorker, WaitReturnsOnlyAfterThatJobRan) {
  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();
  std::atomic<bool> second_ran{false};
  LaunchDrainWorker worker;
  const uint64_t first = worker.submit([released] { released.wait(); });
  const uint64_t second = worker.submit([&] { second_ran = true; });

  auto waiter = std::async(std::launch::async, [&] { worker.wait(second); });
  EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  release.set_value();
  waiter.get();
  EXPECT_TRUE(second_ran);
  worker.wait(first);
}

TEST(LaunchDrainWorker, OnStartRunsOnTheWorkerBeforeTheFirstJob) {
  std::thread::id started_on;
  std::thread::id first_job_on;
  bool started_before_job = false;
  LaunchDrainWorker worker([&] { started_on = std::this_thread::get_id(); });
  worker.wait(worker.submit([&] {
    first_job_on = std::this_thread::get_id();
    started_before_job = started_on == first_job_on;
  }));
  EXPECT_TRUE(started_before_job);
}

TEST(LaunchDrainWorker, OnStopRunsOnTheWorkerAfterTheLastJob) {
  std::thread::id job_on;
  std::thread::id stopped_on;
  bool job_ran_first = false;
  bool job_ran = false;
  {
    LaunchDrainWorker worker({}, [&] {
      stopped_on = std::this_thread::get_id();
      job_ran_first = job_ran;
    });
    worker.submit([&] {
      job_on = std::this_thread::get_id();
      job_ran = true;
    });
  }
  EXPECT_TRUE(job_ran_first);
  EXPECT_EQ(stopped_on, job_on);
}

TEST(LaunchDrainWorker, StopRunsQueuedJobsAndLaterSubmissionsRunOnTheCaller) {
  std::atomic<int> ran{0};
  LaunchDrainWorker worker;
  for (int i = 0; i < 3; ++i) {
    worker.submit([&] { ++ran; });
  }
  worker.stop();
  EXPECT_EQ(ran, 3);

  std::thread::id ran_on;
  EXPECT_EQ(worker.submit([&] { ran_on = std::this_thread::get_id(); }), 0u);
  EXPECT_EQ(ran_on, std::this_thread::get_id());
  worker.wait(0);
  worker.stop();
}

TEST(LaunchDrainWorker, WaitIdleCoversEveryQueuedJob) {
  std::atomic<int> ran{0};
  LaunchDrainWorker worker;
  for (int i = 0; i < 4; ++i) {
    worker.submit([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      ++ran;
    });
  }
  worker.wait_idle();
  EXPECT_EQ(ran, 4);
}

TEST(LaunchDrainWorker, AJobWaitingForItselfDoesNotBlock) {
  // A job that blocks never returns: leak the worker instead of joining it, so
  // the test fails instead of hanging.
  auto* worker = new LaunchDrainWorker();
  auto own_ticket = std::make_shared<std::atomic<uint64_t>>(0);
  auto returned = std::make_shared<std::promise<void>>();
  std::future<void> job_returned = returned->get_future();
  std::promise<void> ticket_known;
  std::shared_future<void> known = ticket_known.get_future().share();
  *own_ticket = worker->submit([worker, own_ticket, returned, known] {
    known.wait();
    worker->wait(*own_ticket);
    returned->set_value();
  });
  ticket_known.set_value();
  ASSERT_EQ(job_returned.wait_for(std::chrono::seconds(30)), std::future_status::ready);
  delete worker;
}

}  // namespace
