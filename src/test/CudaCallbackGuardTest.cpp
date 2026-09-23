/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <cerrno>
#include <future>
#include <thread>

#include "cuda_callback_guard.h"

namespace cutracer {
namespace {

TEST(CudaCallbackGuardTest, NestedToolCallsRestoreThePreviousSuppression) {
  EXPECT_FALSE(ScopedCudaCallbackSuppression::is_suppressed());
  {
    ScopedCudaCallbackSuppression outer;
    EXPECT_TRUE(ScopedCudaCallbackSuppression::is_suppressed());
    {
      ScopedCudaCallbackSuppression inner;
      EXPECT_TRUE(ScopedCudaCallbackSuppression::is_suppressed());
    }
    EXPECT_TRUE(ScopedCudaCallbackSuppression::is_suppressed());
  }
  EXPECT_FALSE(ScopedCudaCallbackSuppression::is_suppressed());
}

TEST(CudaCallbackGuardTest, ContextTeardownDoesNotSuppressAnotherThreadsLaunchExit) {
  std::promise<void> teardown_entered;
  auto entered = teardown_entered.get_future();
  std::promise<void> exit_processed;
  auto processed = exit_processed.get_future();
  std::thread teardown([&] {
    ScopedCudaCallbackSuppression suppress_callbacks;
    teardown_entered.set_value();
    processed.wait();
    EXPECT_TRUE(ScopedCudaCallbackSuppression::is_suppressed());
  });
  entered.wait();
  EXPECT_FALSE(ScopedCudaCallbackSuppression::is_suppressed());
  {
    ScopedCudaCallbackSuppression launch_exit;
    EXPECT_TRUE(ScopedCudaCallbackSuppression::is_suppressed());
  }
  EXPECT_FALSE(ScopedCudaCallbackSuppression::is_suppressed());
  exit_processed.set_value();
  teardown.join();
}

class CudaLaunchCallbackTest : public ::testing::Test {
 protected:
  using Callback = ScopedCudaLaunchCallback;

  void SetUp() override {
    pthread_mutexattr_t attributes;
    ASSERT_EQ(pthread_mutexattr_init(&attributes), 0);
    ASSERT_EQ(pthread_mutexattr_settype(&attributes, PTHREAD_MUTEX_RECURSIVE), 0);
    ASSERT_EQ(pthread_mutex_init(&mutex_, &attributes), 0);
    ASSERT_EQ(pthread_mutexattr_destroy(&attributes), 0);
  }

  void TearDown() override {
    EXPECT_TRUE(pending_.empty());
    // Clean up even if a regression left a recursive acquisition behind.
    // trylock in the tests never blocks, so such a failure cannot hang CI.
    int remaining_acquisitions = 0;
    while (pthread_mutex_unlock(&mutex_) == 0) {
      ++remaining_acquisitions;
    }
    EXPECT_EQ(remaining_acquisitions, 0);
    EXPECT_EQ(pthread_mutex_destroy(&mutex_), 0);
  }

  void enter(int api, std::optional<uint64_t> launch_id, bool retain_lock = true, const void* context = nullptr) {
    Callback callback(mutex_, pending_);
    callback.enter({context, api, launch_id, retain_lock});
  }

  Callback::LaunchExit exit(int api, const void* context = nullptr) {
    Callback callback(mutex_, pending_);
    return callback.take_pending_launch(context, api);
  }

  int other_thread_lock_result() {
    int result = -1;
    std::thread other([&] {
      result = pthread_mutex_trylock(&mutex_);
      if (result == 0) {
        EXPECT_EQ(pthread_mutex_unlock(&mutex_), 0);
      }
    });
    other.join();
    return result;
  }

  pthread_mutex_t mutex_;
  Callback::PendingLaunches pending_;
};

TEST_F(CudaLaunchCallbackTest, MatchingExitReleasesTheEntryAndCallbackAcquisitions) {
  enter(1, 17);
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  const auto result = exit(1);
  ASSERT_TRUE(result.launch);
  EXPECT_EQ(result.launch->launch_id, 17);
  EXPECT_TRUE(result.in_order);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, NestedExitPreservesTheOuterLaunchLock) {
  enter(1, 17);
  enter(2, 23);
  const auto inner = exit(2);
  ASSERT_TRUE(inner.launch);
  EXPECT_EQ(inner.launch->launch_id, 23);
  EXPECT_TRUE(inner.in_order);
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  const auto outer = exit(1);
  ASSERT_TRUE(outer.launch);
  EXPECT_EQ(outer.launch->launch_id, 17);
  EXPECT_TRUE(outer.in_order);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, UnmatchedExitWithNoEntryReleasesTheCallbackAcquisition) {
  const auto result = exit(1);
  EXPECT_FALSE(result.launch);
  EXPECT_FALSE(result.in_order);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, UnmatchedExitPreservesAnUnrelatedEntry) {
  enter(1, 17);
  const auto unmatched = exit(2);
  EXPECT_FALSE(unmatched.launch);
  EXPECT_FALSE(unmatched.in_order);
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  const auto matched = exit(1);
  ASSERT_TRUE(matched.launch);
  EXPECT_EQ(matched.launch->launch_id, 17);
  EXPECT_TRUE(matched.in_order);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, OutOfOrderExitReleasesOnlyItsMatchingEntry) {
  enter(1, 17);
  enter(2, 23);
  const auto outer = exit(1);
  ASSERT_TRUE(outer.launch);
  EXPECT_EQ(outer.launch->launch_id, 17);
  EXPECT_FALSE(outer.in_order);
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  const auto inner = exit(2);
  ASSERT_TRUE(inner.launch);
  EXPECT_EQ(inner.launch->launch_id, 23);
  EXPECT_TRUE(inner.in_order);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, OutOfOrderExitPreservesBothOuterAndInnerFrames) {
  enter(1, 17);
  enter(2, 23);
  enter(3, 31);
  const auto middle = exit(2);
  ASSERT_TRUE(middle.launch);
  EXPECT_EQ(middle.launch->launch_id, 23);
  EXPECT_FALSE(middle.in_order);
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  const auto inner = exit(3);
  ASSERT_TRUE(inner.launch);
  EXPECT_EQ(inner.launch->launch_id, 31);
  EXPECT_TRUE(inner.in_order);
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  const auto outer = exit(1);
  ASSERT_TRUE(outer.launch);
  EXPECT_EQ(outer.launch->launch_id, 17);
  EXPECT_TRUE(outer.in_order);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, SameApiInDifferentContextsMatchesTheCorrectEntry) {
  int first_context;
  int second_context;
  enter(1, 17, true, &first_context);
  enter(1, 23, true, &second_context);
  const auto first = exit(1, &first_context);
  ASSERT_TRUE(first.launch);
  EXPECT_EQ(first.launch->launch_id, 17);
  EXPECT_FALSE(first.in_order);
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  const auto second = exit(1, &second_context);
  ASSERT_TRUE(second.launch);
  EXPECT_EQ(second.launch->launch_id, 23);
  EXPECT_TRUE(second.in_order);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, MissingContextDoesNotConsumeTheSameApiInAnotherContext) {
  int context;
  enter(1, 17, true, &context);
  EXPECT_FALSE(exit(1).launch);
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  EXPECT_TRUE(exit(1, &context).launch);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, RecursiveSameApiExitsUseTheNewestEntry) {
  enter(1, 17);
  enter(1, 23);
  const auto inner = exit(1);
  ASSERT_TRUE(inner.launch);
  EXPECT_EQ(inner.launch->launch_id, 23);
  EXPECT_TRUE(inner.in_order);
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  const auto outer = exit(1);
  ASSERT_TRUE(outer.launch);
  EXPECT_EQ(outer.launch->launch_id, 17);
  EXPECT_TRUE(outer.in_order);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, StreamCaptureDoesNotRetainAnEntryAcquisition) {
  enter(1, std::nullopt, false);
  EXPECT_EQ(other_thread_lock_result(), 0);
  const auto result = exit(1);
  ASSERT_TRUE(result.launch);
  EXPECT_FALSE(result.launch->retained_lock);
  EXPECT_TRUE(result.in_order);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, OutOfOrderStreamCaptureExitPreservesTheRetainedLaunch) {
  enter(1, std::nullopt, false);
  enter(2, 23);
  const auto captured = exit(1);
  ASSERT_TRUE(captured.launch);
  EXPECT_FALSE(captured.launch->retained_lock);
  EXPECT_FALSE(captured.in_order);
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  EXPECT_TRUE(exit(2).launch);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, GraphEntryWithoutALaunchIdStillOwnsItsLock) {
  enter(1, std::nullopt);
  enter(2, std::nullopt, false);
  const auto graph = exit(1);
  ASSERT_TRUE(graph.launch);
  EXPECT_FALSE(graph.launch->launch_id);
  EXPECT_TRUE(graph.launch->retained_lock);
  EXPECT_FALSE(graph.in_order);
  EXPECT_EQ(other_thread_lock_result(), 0);
  EXPECT_TRUE(exit(2).launch);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

TEST_F(CudaLaunchCallbackTest, NonLaunchCallbacksReleaseOnlyTheirOwnAcquisition) {
  enter(1, 17);
  {
    Callback callback(mutex_, pending_);
  }
  EXPECT_EQ(other_thread_lock_result(), EBUSY);
  EXPECT_TRUE(exit(1).launch);
  EXPECT_EQ(other_thread_lock_result(), 0);
}

}  // namespace
}  // namespace cutracer
