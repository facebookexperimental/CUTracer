/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>

#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "runtime_identity.h"

namespace {
using Json = nlohmann::json;
using cutracer::RuntimeIdentityLedger;

Json function_attributes(char binary = 'a') {
  return {{"entry", "same_name"},
          {"cubin_sha256", std::string(64, binary)},
          {"pc_namespace", "function_relative_bytes"},
          {"instructions", Json::array()}};
}

class RuntimeIdentityTest : public ::testing::Test {
 protected:
  std::vector<Json> events;
  RuntimeIdentityLedger ledger{Json{{"pid", 123}, {"nonce", "fixture"}},
                               [&](const auto& record) { events.push_back(record); }};
};

TEST_F(RuntimeIdentityTest, ContextAndModuleHandleReuseGetsNewLifetime) {
  const auto c1 = ledger.context_begin(1);
  const auto m1 = ledger.module_load(1, 2, "cuModuleLoadData");
  const auto f1 = ledger.function(1, 2, 3, function_attributes());
  ledger.module_unload(1, 2);
  const auto m2 = ledger.module_load(1, 2, "cuModuleLoadData");
  const auto f2 = ledger.function(1, 2, 3, function_attributes('b'));
  EXPECT_NE(m1, m2);
  EXPECT_NE(f1["function"], f2["function"]);
  ledger.context_end(1);
  const auto c2 = ledger.context_begin(1);
  EXPECT_NE(c1, c2);
}

TEST_F(RuntimeIdentityTest, SameNameDifferentModulesRetainBothBinaries) {
  ledger.context_begin(1);
  auto a = ledger.function(1, 2, 3, function_attributes());
  auto b = ledger.function(1, 4, 5, function_attributes('b'));
  EXPECT_EQ(a["entry"], b["entry"]);
  EXPECT_NE(a["cubin_sha256"], b["cubin_sha256"]);
  EXPECT_NE(a["module"], b["module"]);
  EXPECT_FALSE(a["module_load_observed"]);
  EXPECT_THROW(ledger.function(1, 2, 3, function_attributes('c')), std::logic_error);
}

TEST_F(RuntimeIdentityTest, FailedCaptureIsNotRetriedOnLaterLaunches) {
  ledger.context_begin(1);
  int attempts = 0;
  auto capture = [&]() -> Json {
    ++attempts;
    throw std::runtime_error("instruction bytes unavailable");
  };
  EXPECT_THROW(ledger.capture_function(1, 2, 3, capture), std::runtime_error);
  for (int launch = 0; launch < 1000; ++launch) {
    EXPECT_TRUE(ledger.capture_function(1, 2, 3, capture).is_null());
  }
  EXPECT_EQ(attempts, 1);
  EXPECT_THROW(ledger.launch(1, 3, 0, Json::object()), std::logic_error);
  ledger.finish();
  EXPECT_TRUE(events.back()["data"]["has_gaps"]);
}

TEST_F(RuntimeIdentityTest, CaptureFailuresRetireWithTheirModuleAndContext) {
  ledger.context_begin(1);
  ledger.context_begin(9);
  int attempts = 0;
  bool fail = true;
  auto capture = [&]() -> Json {
    ++attempts;
    if (fail) throw std::runtime_error("cubin unavailable");
    return function_attributes();
  };
  EXPECT_THROW(ledger.capture_function(1, 2, 3, capture), std::runtime_error);
  EXPECT_THROW(ledger.capture_function(9, 2, 3, capture), std::runtime_error);
  EXPECT_EQ(attempts, 2);
  ledger.module_unload(1, 2);
  ledger.module_load(1, 2, "cuModuleLoadData");
  fail = false;
  const auto identity = ledger.capture_function(1, 2, 3, capture);
  EXPECT_FALSE(identity.is_null());
  EXPECT_EQ(ledger.capture_function(1, 2, 3, capture), identity);
  EXPECT_TRUE(ledger.capture_function(9, 2, 3, capture).is_null());
  EXPECT_EQ(attempts, 3);
  ledger.context_end(9);
  ledger.context_begin(9);
  EXPECT_FALSE(ledger.capture_function(9, 2, 3, capture).is_null());
  EXPECT_EQ(attempts, 4);
}

TEST(RuntimeFunctionCaptureTest, FailedPublicationDoesNotYieldACachedIdentity) {
  bool fail = true;
  RuntimeIdentityLedger ledger(Json{{"pid", 123}}, [&](const auto& record) {
    if (fail && record.at("type") == "function") throw std::runtime_error("sink failed");
  });
  ledger.context_begin(1);
  int attempts = 0;
  auto capture = [&] {
    ++attempts;
    return function_attributes();
  };
  EXPECT_THROW(ledger.capture_function(1, 2, 3, capture), std::runtime_error);
  fail = false;
  EXPECT_TRUE(ledger.capture_function(1, 2, 3, capture).is_null());
  EXPECT_EQ(attempts, 1);
  EXPECT_THROW(ledger.launch(1, 3, 0, Json::object()), std::logic_error);
}

TEST(RuntimeFunctionCaptureTest, FailedDirectPublicationLeavesTheFunctionUnavailable) {
  bool fail = true;
  RuntimeIdentityLedger ledger(Json{{"pid", 123}}, [&](const auto& record) {
    if (fail && record.at("type") == "function") throw std::runtime_error("sink failed");
  });
  ledger.context_begin(1);
  EXPECT_THROW(ledger.function(1, 2, 3, function_attributes()), std::runtime_error);
  fail = false;
  int attempts = 0;
  auto capture = [&] {
    ++attempts;
    return function_attributes();
  };
  EXPECT_TRUE(ledger.capture_function(1, 2, 3, capture).is_null());
  EXPECT_EQ(attempts, 0);
  EXPECT_THROW(ledger.function(1, 2, 3, function_attributes()), std::logic_error);
  EXPECT_THROW(ledger.launch(1, 3, 0, Json::object()), std::logic_error);
}

TEST_F(RuntimeIdentityTest, DirectIdentityCannotReplaceAFailedCapture) {
  ledger.context_begin(1);
  int attempts = 0;
  auto capture = [&]() -> Json {
    ++attempts;
    throw std::runtime_error("instruction bytes unavailable");
  };
  EXPECT_THROW(ledger.capture_function(1, 2, 3, capture), std::runtime_error);
  EXPECT_THROW(ledger.function(1, 2, 3, function_attributes()), std::logic_error);
  EXPECT_THROW(ledger.launch(1, 3, 0, Json::object()), std::logic_error);
  EXPECT_EQ(attempts, 1);
}

TEST(RuntimeIdentityRetirementTest, FailedEndRecordsStillRetireTheirLifetimes) {
  for (const std::string failed : {"module_end", "launch_incomplete", "context_end"}) {
    SCOPED_TRACE(failed);
    bool fail = false;
    RuntimeIdentityLedger ledger(Json{{"pid", 123}}, [&](const Json& record) {
      if (fail && record.at("type") == failed) {
        fail = false;
        throw std::runtime_error("one-shot sink failure");
      }
    });
    const auto old_context = ledger.context_begin(1);
    ledger.module_load(1, 2, "cuModuleLoadData");
    ledger.capture_function(1, 2, 3, [] { return function_attributes('a'); });
    ledger.launch(1, 3, 0, Json::object());
    if (failed != "launch_incomplete") ledger.complete(1, 0, true, true);
    fail = true;
    if (failed == "module_end") {
      EXPECT_THROW(ledger.module_unload(1, 2), std::runtime_error);
    } else {
      EXPECT_THROW(ledger.context_end(1), std::runtime_error);
      EXPECT_NE(ledger.context_begin(1), old_context);
    }
    ledger.module_load(1, 2, "cuModuleLoadData");
    int attempts = 0;
    const auto identity = ledger.capture_function(1, 2, 3, [&] {
      ++attempts;
      return function_attributes('b');
    });
    EXPECT_EQ(attempts, 1);
    EXPECT_EQ(identity["cubin_sha256"], std::string(64, 'b'));
  }
}

TEST(RuntimeIdentityRetirementTest, UnpublishedLaunchIsNeverPending) {
  bool fail = false;
  std::vector<Json> events;
  RuntimeIdentityLedger ledger(Json{{"pid", 123}}, [&](const Json& record) {
    if (fail && record.at("type") == "launch") throw std::runtime_error("sink failed");
    events.push_back(record);
  });
  ledger.context_begin(1);
  ledger.function(1, 2, 3, function_attributes());
  fail = true;
  EXPECT_THROW(ledger.launch(1, 3, 0, Json::object()), std::runtime_error);
  fail = false;
  EXPECT_THROW(ledger.complete(1, 0, true, true), std::logic_error);
  EXPECT_THROW(ledger.launch(1, 3, 0, Json::object()), std::logic_error);
  ledger.context_end(1);
  EXPECT_EQ(events.back()["data"]["incomplete_launches"], 0);
  ledger.finish();
  EXPECT_TRUE(events.back()["data"]["has_gaps"]);
}

TEST(RuntimeIdentityLimitsTest, CachedIdentitiesStayReadableAtTheLimits) {
  cutracer::RuntimeIdentityLimits limits;
  limits.modules = 2;
  limits.functions = 2;
  RuntimeIdentityLedger ledger(Json{{"pid", 123}}, [](const Json&) {}, limits);
  ledger.context_begin(1);
  const auto identity = ledger.capture_function(1, 2, 3, [] { return function_attributes(); });
  int failures = 0;
  auto failure = [&]() -> Json {
    ++failures;
    throw std::runtime_error("cubin unavailable");
  };
  EXPECT_THROW(ledger.capture_function(1, 4, 5, failure), std::runtime_error);
  EXPECT_THROW(ledger.module_load(1, 6, "cuModuleLoadData"), std::logic_error);
  EXPECT_THROW(ledger.capture_function(1, 2, 7, [] { return function_attributes(); }), std::logic_error);
  EXPECT_EQ(ledger.capture_function(1, 2, 3, [] { return function_attributes('z'); }), identity);
  EXPECT_TRUE(ledger.capture_function(1, 4, 5, failure).is_null());
  EXPECT_EQ(ledger.function(1, 2, 3, function_attributes()), identity);
  EXPECT_EQ(failures, 1);
}

TEST(RuntimeIdentityLimitsTest, LaunchBudgetCountsOnlyInFlightLaunches) {
  cutracer::RuntimeIdentityLimits limits;
  limits.pending_launches = 2;
  limits.retired_launch_ids = 2;
  RuntimeIdentityLedger ledger(Json{{"pid", 123}}, [](const Json&) {}, limits);
  ledger.context_begin(1);
  ledger.function(1, 2, 3, function_attributes());
  for (uint64_t id = 0; id < 10; ++id) {
    ledger.launch(1, 3, id, Json::object());
    ledger.complete(1, id, true, true);
  }
  EXPECT_THROW(ledger.launch(1, 3, 9, Json::object()), std::logic_error);
  ledger.launch(1, 3, 10, Json::object());
  ledger.launch(1, 3, 11, Json::object());
  EXPECT_THROW(ledger.launch(1, 3, 12, Json::object()), std::logic_error);
  ledger.complete(1, 10, true, true);
  EXPECT_EQ(ledger.launch(1, 3, 12, Json::object())["kernel_ordinal"], 12);
}

TEST_F(RuntimeIdentityTest, OrdinalIsPerContextBinaryAndEntryAcrossReloads) {
  ledger.context_begin(1);
  ledger.context_begin(9);
  ledger.function(1, 2, 3, function_attributes());
  ledger.function(1, 4, 5, function_attributes('b'));
  ledger.function(9, 2, 3, function_attributes());
  EXPECT_EQ(ledger.launch(1, 3, 0, Json::object())["kernel_ordinal"], 0);
  EXPECT_EQ(ledger.launch(1, 3, 1, Json::object())["kernel_ordinal"], 1);
  EXPECT_EQ(ledger.launch(1, 5, 2, Json::object())["kernel_ordinal"], 0);
  EXPECT_EQ(ledger.launch(9, 3, 3, Json::object())["kernel_ordinal"], 0);
  ledger.module_unload(1, 2);
  ledger.function(1, 2, 3, function_attributes());
  EXPECT_EQ(ledger.launch(1, 3, 4, Json::object())["kernel_ordinal"], 2);
}

TEST_F(RuntimeIdentityTest, CompletionCannotBeSplicedFromAnotherContextOrLaunch) {
  ledger.context_begin(1);
  ledger.context_begin(9);
  ledger.function(1, 2, 3, function_attributes());
  const auto launch = ledger.launch(1, 3, 7, Json::object());
  EXPECT_THROW(ledger.complete(9, 7, true, true), std::logic_error);
  EXPECT_THROW(ledger.complete(1, 8, true, true), std::logic_error);
  ledger.complete(1, 7, true, true);
  EXPECT_EQ(events.back()["data"]["launch"], launch["launch"]);
  EXPECT_TRUE(events.back()["data"]["trace_footer_required"]);
  EXPECT_THROW(ledger.complete(1, 7, true, true), std::logic_error);
  EXPECT_THROW(ledger.launch(1, 3, 7, Json::object()), std::logic_error);
}

TEST_F(RuntimeIdentityTest, MissingCompletionAndFailureRemainIncomplete) {
  ledger.context_begin(1);
  ledger.function(1, 2, 3, function_attributes());
  ledger.launch(1, 3, 0, Json::object());
  ledger.launch(1, 3, 1, Json::object());
  ledger.complete(1, 1, false, false);
  EXPECT_FALSE(events.back()["data"]["kernel_completed"]);
  ledger.context_end(1);
  EXPECT_EQ(events[events.size() - 2]["type"], "launch_incomplete");
  EXPECT_EQ(events.back()["data"]["incomplete_launches"], 1);
  ledger.finish();
  EXPECT_TRUE(events.back()["data"]["has_gaps"]);
}

TEST_F(RuntimeIdentityTest, OldFunctionCannotBeUsedAfterUnload) {
  ledger.context_begin(1);
  ledger.function(1, 2, 3, function_attributes());
  ledger.module_unload(1, 2);
  EXPECT_THROW(ledger.launch(1, 3, 0, Json::object()), std::logic_error);
  ledger.context_end(1);
  EXPECT_THROW(ledger.function(1, 2, 3, function_attributes()), std::logic_error);
}

TEST_F(RuntimeIdentityTest, UnselectedAndGraphLaunchesStillHaveSeparateIdentity) {
  ledger.context_begin(1);
  ledger.function(1, 2, 3, function_attributes());
  const auto a = ledger.launch(1, 3, 1, {{"launch_kind", "direct"}, {"capture_selected", false}});
  const auto b =
      ledger.launch(1, 3, 2, {{"launch_kind", "graph"}, {"graph_node_handle", 99}, {"capture_selected", true}});
  const auto c =
      ledger.launch(1, 3, 3, {{"launch_kind", "graph"}, {"graph_node_handle", 99}, {"capture_selected", true}});
  EXPECT_FALSE(a["capture_selected"]);
  EXPECT_NE(a["launch"], b["launch"]);
  EXPECT_NE(b["launch"], c["launch"]);
  EXPECT_EQ(c["kernel_ordinal"], 2);
  ledger.complete(1, 1, true, true);
  EXPECT_FALSE(events.back()["data"]["trace_footer_required"]);
  ledger.complete(1, 2, true, true);
  EXPECT_TRUE(events.back()["data"]["trace_footer_required"]);
}

TEST_F(RuntimeIdentityTest, CanonicalKeysAndSequenceCoverEveryEvent) {
  ledger.context_begin(1);
  ledger.function(1, 2, 3, function_attributes());
  auto launch = ledger.launch(1, 3, 0, Json::object());
  const auto key = launch["launch"];
  launch.erase("launch");
  EXPECT_EQ(key, cutracer::runtime_digest(launch));
  ledger.complete(1, 0, true, true);
  ledger.context_end(1);
  ledger.finish();
  const auto size = events.size();
  ledger.finish();
  EXPECT_EQ(events.size(), size);
  for (size_t i = 0; i < events.size(); ++i) {
    EXPECT_EQ(events[i]["sequence"], i);
    EXPECT_EQ(events[i]["process"], ledger.process_key());
  }
  EXPECT_THROW(ledger.context_begin(2), std::logic_error);
}

TEST_F(RuntimeIdentityTest, ConcurrentCallbacksDoNotReuseSequenceOrLifetimes) {
  std::vector<std::thread> threads;
  for (uint64_t i = 1; i <= 8; ++i) {
    threads.emplace_back([&, i] {
      ledger.context_begin(i);
      ledger.function(i, 2, 3, function_attributes());
      ledger.launch(i, 3, i, Json::object());
      ledger.complete(i, i, true, true);
      ledger.context_end(i);
    });
  }
  for (auto& thread : threads) thread.join();
  std::set<std::string> contexts;
  for (size_t i = 0; i < events.size(); ++i) {
    EXPECT_EQ(events[i]["sequence"], i);
    if (events[i]["type"] == "context_begin") contexts.insert(events[i]["data"]["context"].get<std::string>());
  }
  EXPECT_EQ(contexts.size(), 8);
}

TEST_F(RuntimeIdentityTest, FailedSinkCannotProduceSuccessFooter) {
  int calls = 0;
  RuntimeIdentityLedger broken(Json::object(), [&](const Json&) {
    if (++calls > 1) throw std::runtime_error("write failed");
  });
  EXPECT_THROW(broken.context_begin(1), std::runtime_error);
  EXPECT_THROW(broken.finish(), std::runtime_error);
}

TEST_F(RuntimeIdentityTest, TransientCompletionWriteFailureRemainsVisible) {
  bool fail = false;
  RuntimeIdentityLedger transient(Json::object(), [&](const Json& record) {
    if (fail) throw std::runtime_error("transient sink failure");
    events.push_back(record);
  });
  transient.context_begin(1);
  transient.function(1, 2, 3, function_attributes());
  transient.launch(1, 3, 0, Json::object());
  fail = true;
  EXPECT_THROW(transient.complete(1, 0, true, true), std::runtime_error);
  fail = false;
  transient.context_end(1);
  EXPECT_EQ(events.back()["data"]["incomplete_launches"], 1);
  transient.finish();
  EXPECT_TRUE(events.back()["data"]["has_gaps"]);
}

TEST(RuntimeGraphTrackerTest, NestedInvocationsAndContextsHaveIndependentOwnership) {
  cutracer::RuntimeGraphTracker tracker;
  const auto outer = tracker.begin(1, 10, 1);
  tracker.attach(1, 10, outer, 100);
  const auto inner = tracker.begin(1, 20, 2);
  tracker.attach(1, 20, inner, 200);
  const auto other = tracker.begin(2, 10, 1);
  tracker.attach(2, 10, other, 300);
  EXPECT_NE(outer, other);
  EXPECT_FALSE(tracker.current(1, 10));
  const auto inner_exit = tracker.take(1, 20, 2, true);
  EXPECT_TRUE(inner_exit.matched);
  EXPECT_EQ(inner_exit.launches, std::vector<uint64_t>({200}));
  EXPECT_EQ(tracker.current(1, 10), outer);
  EXPECT_TRUE(tracker.take(2, 10, 1, true).matched);
  EXPECT_TRUE(tracker.take(1, 10, 1, true).matched);
  EXPECT_FALSE(tracker.current(1, 10));
}

TEST(RuntimeGraphTrackerTest, MismatchedStreamOrApiRetiresAllUncertainFramesOnlyInThatContext) {
  for (const auto api : {1, 9}) {
    cutracer::RuntimeGraphTracker tracker;
    const auto outer = tracker.begin(1, 10, 1);
    tracker.attach(1, 10, outer, 100);
    const auto inner = tracker.begin(1, 20, 2);
    tracker.attach(1, 20, inner, 200);
    const auto other = tracker.begin(2, 10, 1);
    const auto exit = tracker.take(1, 10, api, true);
    EXPECT_FALSE(exit.matched);
    EXPECT_EQ(exit.launches, std::vector<uint64_t>({100, 200}));
    EXPECT_FALSE(tracker.current(1, 20));
    EXPECT_EQ(tracker.current(2, 10), other);
    const auto replay = tracker.begin(1, 10, 1);
    EXPECT_GT(replay, inner);
    EXPECT_TRUE(tracker.take(1, 10, 1, true).matched);
  }
}

TEST(RuntimeGraphTrackerTest, UnmatchedExitAndExcessiveMissingExitsCannotGrowWithoutBound) {
  cutracer::RuntimeGraphTracker tracker;
  for (size_t depth = 0; depth < cutracer::RuntimeGraphTracker::kMaxDepth; ++depth) tracker.begin(1, 10, 1);
  EXPECT_THROW(tracker.begin(1, 10, 1), std::logic_error);
  const auto exit = tracker.take(1, 10, 1, false);
  EXPECT_FALSE(exit.matched);
  EXPECT_FALSE(tracker.current(1, 10));
  const auto replay = tracker.begin(1, 10, 1);
  tracker.clear(1);
  EXPECT_THROW(tracker.attach(1, 10, replay, 0), std::logic_error);
  EXPECT_FALSE(tracker.take(1, 10, 1, true).matched);
}

TEST(RuntimeIdentitySessionTest, TerminationWaitsForCallbacksAndRetainsSinkLifetime) {
  cutracer::RuntimeIdentitySession session;
  std::atomic<bool> closed{false};
  auto resource = std::shared_ptr<int>(new int(0), [&](int* pointer) {
    closed.store(true);
    delete pointer;
  });
  std::vector<Json> events;
  session.start({{"pid", 123}, {"nonce", "owner"}},
                [resource, &events](const Json& event) { events.push_back(event); });
  resource.reset();
  std::promise<void> entered, release, stopping;
  auto resumed = release.get_future();
  std::thread callback([&] {
    session.event([&](auto& ledger) {
      entered.set_value();
      resumed.wait();
      EXPECT_FALSE(closed.load());
      ledger.gap("controlled callback");
    });
  });
  entered.get_future().wait();
  std::thread shutdown([&] {
    stopping.set_value();
    session.stop();
  });
  stopping.get_future().wait();
  EXPECT_FALSE(closed.load());
  release.set_value();
  callback.join();
  shutdown.join();
  EXPECT_TRUE(closed.load());
  EXPECT_FALSE(session.active());
  EXPECT_FALSE(session.event([](auto&) { FAIL() << "Callback after shutdown"; }));
  ASSERT_FALSE(events.empty());
  EXPECT_EQ(events.back().at("type"), "process_end");
}

TEST(RuntimeIdentitySessionTest, FailedFinalWriteStillClosesTheSession) {
  cutracer::RuntimeIdentitySession session;
  bool fail = false;
  session.start({{"pid", 123}, {"nonce", "owner"}}, [&](const Json&) {
    if (fail) throw std::runtime_error("write failed");
  });
  fail = true;
  EXPECT_THROW(session.stop(), std::runtime_error);
  EXPECT_FALSE(session.active());
}

}  // namespace
