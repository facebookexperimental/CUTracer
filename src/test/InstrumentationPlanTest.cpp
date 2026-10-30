/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>

#include "TemporaryDirectoryTest.h"
#include "instrumentation_plan.h"

namespace {
using nlohmann::json;
using namespace cutracer::plan;

json example() {
  // A text fixture avoids expanding a deeply nested JSON initializer in dev builds.
  auto document = json::parse(R"json({
    "schema": null,
    "plan_id": "shared-observation",
    "policy": {
      "unmatched_kernel": "skip",
      "unsupported_request": "report_gap",
      "identity_mismatch": "error",
      "conflicts": "error"
    },
    "kernels": [{
      "match": {
        "entry": "kernel",
        "target": "sm_100a",
        "identity": {"kind": "cubin_sha256", "value": null}
      },
      "pc_namespace": "function_relative_bytes",
      "configs": [{
        "config_id": "value",
        "requests": ["numeric/0", "race/0"],
        "launches": {"mode": "per_context_kernel_ordinal", "indices": [3, 1]},
        "capture": {"ctas": [[1, 0, 0], [0, 0, 0]], "warps": [2, 0], "lanes": [31, 0]},
        "budget": {"max_records": 100, "max_bytes": 4096, "on_limit": "mark_incomplete"},
        "points": [{
          "function": "kernel",
          "pc": "0x0010",
          "expected_bytes": null,
          "phase": "before",
          "predicate_policy": "reached_lanes",
          "observations": [
            {"kind": "reg_trace", "registers": ["UR3", "R2", "UP1", "P0"]},
            {"kind": "mem_addr_trace", "mref": 0}
          ]
        }]
      }]
    }]
  })json");
  document["schema"] = kSchema;
  document["kernels"][0]["match"]["identity"]["value"] = std::string(64, 'a');
  document["kernels"][0]["configs"][0]["points"][0]["expected_bytes"] = std::string(32, '0');
  return document;
}

json& config(json& j) {
  return j["kernels"][0]["configs"][0];
}

json& point(json& j) {
  return config(j)["points"][0];
}

std::string record(const json& doc, size_t index = 0) {
  return json{{"schema", "cutracer.kernel_plan/v1"},
              {"plan_id", doc["plan_id"]},
              {"policy", doc["policy"]},
              {"kernel", doc["kernels"][index]}}
      .dump();
}

json memory_address_observations(uint64_t count) {
  auto observations = json::array();
  for (uint64_t mref = 0; mref < count; ++mref) {
    observations.push_back({{"kind", "mem_addr_trace"}, {"mref", mref}});
  }
  return observations;
}

class InstrumentationPlanTest : public cutracer::test::TemporaryDirectoryTest {};

TEST_F(InstrumentationPlanTest, TypedSelectorsAndIdentity) {
  const auto plan = PlanLoader::parse(example().dump());
  EXPECT_EQ(plan.id(), "shared-observation");
  EXPECT_TRUE(plan.report_unsupported_as_gap());
  ASSERT_EQ(plan.kernels().size(), 1);
  const auto& kernel = plan.kernels()[0];
  EXPECT_EQ(kernel.cubin_sha256, std::string(64, 'a'));
  const auto& c = kernel.configs[0];
  EXPECT_FALSE(c.launches.selects(0));
  EXPECT_TRUE(c.launches.selects(1));
  EXPECT_TRUE(c.launches.selects(3));
  EXPECT_EQ(c.capture.ctas.front(), (std::array<uint64_t, 3>{0, 0, 0}));
  EXPECT_EQ(c.capture.lanes, (std::vector<uint64_t>{0, 31}));
  EXPECT_EQ(c.budget.max_bytes, 4096);
  EXPECT_EQ(c.points[0].pc, 16);
  EXPECT_EQ(c.points[0].predicate, PredicatePolicy::Reached);
  EXPECT_EQ(plan.canonical()["kernels"][0]["configs"][0]["points"][0]["pc"], "0x10");
}

TEST_F(InstrumentationPlanTest, JsonNdjsonAndReorderingHaveSameCanonicalIdentity) {
  auto j = example();
  auto second = j["kernels"][0];
  second["match"]["identity"]["value"] = std::string(64, 'b');
  second["configs"][0]["config_id"] = "access";
  j["kernels"].push_back(second);
  const auto first = PlanLoader::parse(j.dump(2));
  const auto ndjson = PlanLoader::parse(record(j, 1) + "\n\n" + record(j) + "\r\n");
  EXPECT_EQ(first.canonical(), ndjson.canonical());
  EXPECT_EQ(first.canonical_sha256(), ndjson.canonical_sha256());
  EXPECT_NE(first.source_sha256(), ndjson.source_sha256());
  EXPECT_EQ(first.canonical_sha256(), PlanLoader::parse(first.canonical().dump()).canonical_sha256());
}

TEST_F(InstrumentationPlanTest, CompatibleDuplicatesUnionRegistersAndKeepConsumers) {
  auto j = example();
  auto p = point(j);
  p["observations"] = json::array({{{"kind", "reg_trace"}, {"registers", {"R1", "R2"}}}});
  config(j)["points"].push_back(p);
  const auto& original = config(j);
  j["kernels"][0]["configs"].push_back(original);
  const auto plan = PlanLoader::parse(j.dump());
  ASSERT_EQ(plan.kernels()[0].configs.size(), 1);
  const auto& c = plan.kernels()[0].configs[0];
  ASSERT_EQ(c.points.size(), 1);
  EXPECT_EQ(c.requests, (std::vector<std::string>{"numeric/0", "race/0"}));
  const auto serialized = to_json(c)["points"][0]["observations"];
  ASSERT_EQ(serialized.size(), 2);
  EXPECT_EQ(serialized[1]["registers"], json({"P0", "R1", "R2", "UP1", "UR3"}));
}

TEST_F(InstrumentationPlanTest, SeparateConfigurationsPreserveBudgetsAndRequests) {
  auto j = example();
  auto c = config(j);
  c["config_id"] = "event";
  c["requests"] = {"race/events"};
  c["budget"]["max_records"] = 7;
  c["launches"] = {{"mode", "all"}};
  j["kernels"][0]["configs"].push_back(c);
  const auto plan = PlanLoader::parse(j.dump());
  const auto& configs = plan.kernels()[0].configs;
  ASSERT_EQ(configs.size(), 2);
  EXPECT_EQ(configs[0].id, "event");
  EXPECT_EQ(configs[0].budget.max_records, 7);
  EXPECT_TRUE(configs[0].launches.selects(999));
  EXPECT_EQ(configs[1].budget.max_records, 100);
}

TEST_F(InstrumentationPlanTest, SameNameDifferentBinaryIsNotMerged) {
  auto j = example();
  auto k = j["kernels"][0];
  k["match"]["identity"]["value"] = std::string(64, 'b');
  k["configs"][0]["points"][0]["expected_bytes"] = std::string(32, 'f');
  j["kernels"].push_back(k);
  EXPECT_EQ(PlanLoader::parse(j.dump()).kernels().size(), 2);
}

TEST_F(InstrumentationPlanTest, SplitKernelRecordsMergeDeterministically) {
  auto j = example();
  auto other = j;
  config(other)["config_id"] = "another";
  const auto a = PlanLoader::parse(record(j) + "\n" + record(other));
  const auto b = PlanLoader::parse(record(other) + "\n" + record(j));
  EXPECT_EQ(a.canonical_sha256(), b.canonical_sha256());
  EXPECT_EQ(a.kernels()[0].configs.size(), 2);
}

TEST_F(InstrumentationPlanTest, DuplicateKeysRejectedAtEveryDepth) {
  for (const auto& name : {"schema", "match", "capture", "max_bytes", "pc", "kind"}) {
    auto bytes = example().dump();
    const std::string key = "\"" + std::string(name) + "\":";
    const auto pos = bytes.find(key);
    ASSERT_NE(pos, std::string::npos);
    bytes.insert(pos, key + "null,");
    EXPECT_THROW(PlanLoader::parse(bytes), PlanError) << name;
  }
}

TEST_F(InstrumentationPlanTest, UnknownFieldsAndVersionsRejected) {
  for (const auto& pointer : {"", "/policy", "/kernels/0", "/kernels/0/match", "/kernels/0/configs/0",
                              "/kernels/0/configs/0/points/0", "/kernels/0/configs/0/points/0/observations/0"}) {
    auto j = example();
    j[json::json_pointer(pointer)]["unknown"] = true;
    EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError) << pointer;
  }
  auto j = example();
  j["schema"] = "cutracer.instrumentation_plan/v2";
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
}

TEST_F(InstrumentationPlanTest, InvalidNumericKindsAndRangesRejected) {
  for (const auto& value : json::array({true, false, 1.0, 1.25, -1, 0, 9007199254740992ULL, "1", nullptr})) {
    auto j = example();
    config(j)["budget"]["max_records"] = value;
    EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError) << value;
  }
  auto j = example();
  config(j)["budget"]["max_records"] = kExactIntegerMax;
  config(j)["budget"]["max_bytes"] = kExactIntegerMax;
  EXPECT_NO_THROW(PlanLoader::parse(j.dump()));
}

TEST_F(InstrumentationPlanTest, SelectorsRejectEmptyDuplicatesAndOutOfRange) {
  for (const auto& field : {"lanes", "warps", "ctas"}) {
    auto j = example();
    config(j)["capture"][field] = json::array();
    EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
    j = example();
    auto& a = config(j)["capture"][field];
    a.push_back(a[0]);
    EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  }
  auto j = example();
  config(j)["capture"]["lanes"] = {32};
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  config(j)["capture"]["lanes"] = "all";
  config(j)["launches"]["indices"] = {1, 1};
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  config(j)["launches"] = {{"mode", "graph_replay"}, {"indices", {1}}};
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
}

TEST_F(InstrumentationPlanTest, ConflictingConfigAndPointAreRejected) {
  auto j = example();
  auto c = config(j);
  c["budget"]["max_bytes"] = 1;
  j["kernels"][0]["configs"].push_back(c);
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  j = example();
  auto p = point(j);
  p["predicate_policy"] = "executed_lanes";
  config(j)["points"].push_back(p);
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
}

TEST_F(InstrumentationPlanTest, BytesConflictAcrossEntriesAndConfigurations) {
  auto j = example();
  auto k = j["kernels"][0];
  k["match"]["entry"] = "other_entry";
  k["configs"][0]["points"][0]["expected_bytes"] = std::string(32, 'f');
  j["kernels"].push_back(k);
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
}

TEST_F(InstrumentationPlanTest, BeforeAndAfterAreDifferentPoints) {
  auto j = example();
  auto p = point(j);
  p["phase"] = "after";
  p["predicate_policy"] = "executed_lanes";
  config(j)["points"].push_back(p);
  const auto plan = PlanLoader::parse(j.dump());
  EXPECT_EQ(plan.kernels()[0].configs[0].points.size(), 2);
}

TEST_F(InstrumentationPlanTest, BadInstructionIdentityRejected) {
  for (const auto& value : {"0x", "0X10", "0xAB", "0x10000000000000000", "16"}) {
    auto j = example();
    point(j)["pc"] = value;
    EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  }
  for (const auto& value : {"", "ff", "FFFFFFFFFFFFFFFF", "000000000000000g", "00000000000000000"}) {
    auto j = example();
    point(j)["expected_bytes"] = value;
    EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  }
  auto j = example();
  point(j)["pc"] = "0xffffffffffffffff";
  EXPECT_EQ(PlanLoader::parse(j.dump()).kernels()[0].configs[0].points[0].pc, UINT64_MAX);
}

TEST_F(InstrumentationPlanTest, TargetsAndRegisterNamesAreStrict) {
  for (const auto& target : {"s", "sm_", "sm_a", "sm_100aa", "sm100", "sm_100aX"}) {
    auto j = example();
    j["kernels"][0]["match"]["target"] = target;
    EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  }
  for (const auto& name : {"RZ", "URZ", "PT", "UPT", "R-1", "R+1", "R4294967296"}) {
    auto j = example();
    point(j)["observations"][0]["registers"] = {name};
    EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  }
  auto j = example();
  point(j)["observations"][0]["registers"] = {"R1", "R01"};
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
}

TEST_F(InstrumentationPlanTest, MaximumObservationBatchHasStableCanonicalIdentity) {
  auto j = example();
  auto& observations = point(j)["observations"];
  observations = json::array();
  for (uint64_t i = 0; i < 512; ++i) {
    observations.push_back({{"kind", "mem_value_trace"}, {"mref", i}});
  }
  const auto first = PlanLoader::parse(j.dump());
  std::reverse(observations.begin(), observations.end());
  const auto second = PlanLoader::parse(j.dump());
  EXPECT_EQ(first.canonical_sha256(), second.canonical_sha256());
  EXPECT_EQ(first.kernels()[0].configs[0].points[0].observations.size(), 512);
}

TEST_F(InstrumentationPlanTest, MergedObservationAtLimitHasStableCanonicalIdentity) {
  auto j = example();
  point(j)["observations"] = memory_address_observations(512);
  auto duplicate = point(j);
  duplicate["observations"] = json::array({{{"kind", "mem_addr_trace"}, {"mref", 511}}});
  auto& points = config(j)["points"];
  points.push_back(duplicate);
  const auto first = PlanLoader::parse(j.dump());
  ASSERT_EQ(first.kernels()[0].configs[0].points.size(), 1);
  EXPECT_EQ(first.kernels()[0].configs[0].points[0].observations.size(), 512);
  std::reverse(points.begin(), points.end());
  EXPECT_EQ(first.canonical_sha256(), PlanLoader::parse(j.dump()).canonical_sha256());
  const auto round_trip = PlanLoader::parse(first.canonical().dump());
  EXPECT_EQ(first.canonical(), round_trip.canonical());
  EXPECT_EQ(first.canonical_sha256(), round_trip.canonical_sha256());
}

TEST_F(InstrumentationPlanTest, MergedObservationBeyondLimitRejected) {
  auto j = example();
  point(j)["observations"] = memory_address_observations(512);
  auto extra = point(j);
  extra["observations"] = json::array({{{"kind", "mem_addr_trace"}, {"mref", 512}}});
  auto& points = config(j)["points"];
  points.push_back(extra);
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  std::reverse(points.begin(), points.end());
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
}

TEST_F(InstrumentationPlanTest, AllObservationKindsAndDelayConflicts) {
  auto j = example();
  point(j)["observations"] = json::array({{{"kind", "opcode_only"}},
                                          {{"kind", "tma_trace"}},
                                          {{"kind", "mem_value_trace"}, {"mref", 1}},
                                          {{"kind", "random_delay"}, {"delay_ns", 5}}});
  EXPECT_EQ(PlanLoader::parse(j.dump()).kernels()[0].configs[0].points[0].observations.size(), 4);
  point(j)["observations"].push_back({{"kind", "random_delay"}, {"delay_ns", 6}});
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  point(j)["observations"] = json::array({{{"kind", "sync_event"}}});
  // report_gap cannot admit an unknown schema kind.
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
}

TEST_F(InstrumentationPlanTest, NdjsonTailAndHeadersAreTransactional) {
  const auto good = record(example());
  EXPECT_THROW(PlanLoader::parse(good + "\n{"), PlanError);
  auto other = example();
  other["plan_id"] = "other";
  EXPECT_THROW(PlanLoader::parse(good + "\n" + record(other)), PlanError);
  other = example();
  other["policy"]["unsupported_request"] = "error";
  EXPECT_THROW(PlanLoader::parse(good + "\n" + record(other)), PlanError);
  EXPECT_THROW(PlanLoader::parse(good + "\n" + example().dump()), PlanError);
}

TEST_F(InstrumentationPlanTest, RawNulJsonTailRejected) {
  const auto document = example().dump();
  for (const auto* tail : {"", "\n{"}) {
    const auto bytes = document + std::string(1, '\0') + tail;
    EXPECT_THROW(PlanLoader::parse(bytes), PlanError);
    {
      std::ofstream out(path("nul_plan.json"), std::ios::binary);
      out << bytes;
    }
    EXPECT_THROW(PlanLoader::load(path("nul_plan.json")), PlanError);
  }
}

TEST_F(InstrumentationPlanTest, RawNulCannotHideNdjsonRecords) {
  const auto first = record(example());
  auto other = example();
  other["kernels"][0]["match"]["identity"]["value"] = std::string(64, 'b');
  EXPECT_EQ(PlanLoader::parse(first + "\n" + record(other)).kernels().size(), 2);
  for (const auto* plan_id : {"shared-observation", "other-plan"}) {
    other["plan_id"] = plan_id;
    const auto bytes = first + std::string(1, '\0') + "\n" + record(other);
    EXPECT_THROW(PlanLoader::parse(bytes), PlanError);
    {
      std::ofstream out(path("nul_plan.ndjson"), std::ios::binary);
      out << bytes;
    }
    EXPECT_THROW(PlanLoader::load(path("nul_plan.ndjson")), PlanError);
  }
}

TEST_F(InstrumentationPlanTest, MalformedAndDeepInputsFailWithinBudget) {
  for (const auto& value : {"", "null", "[]", "{}", "NaN", "[Infinity]", "{} trailing"}) {
    EXPECT_THROW(PlanLoader::parse(value), PlanError);
  }
  EXPECT_THROW(PlanLoader::parse(std::string(70, '[') + "0" + std::string(70, ']')), PlanError);
  EXPECT_THROW(PlanLoader::parse(std::string(kFileBytesMax + 1, ' ')), PlanError);
}

TEST_F(InstrumentationPlanTest, EmptyAndMissingRequiredCollectionsRejected) {
  for (const auto& pointer : {"/kernels", "/kernels/0/configs", "/kernels/0/configs/0/points",
                              "/kernels/0/configs/0/requests", "/kernels/0/configs/0/points/0/observations"}) {
    auto j = example();
    j[json::json_pointer(pointer)] = json::array();
    EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
  }
  auto j = example();
  config(j).erase("requests");
  EXPECT_THROW(PlanLoader::parse(j.dump()), PlanError);
}

TEST_F(InstrumentationPlanTest, FileReadAndCanonicalRoundTrip) {
  const auto source = example().dump(2);
  {
    std::ofstream out(path("plan"), std::ios::binary);
    out << source;
  }
  const auto plan = PlanLoader::load(path("plan"));
  EXPECT_EQ(plan.source_sha256(), PlanLoader::parse(source).source_sha256());
  EXPECT_EQ(plan.canonical(), PlanLoader::parse(plan.canonical().dump()).canonical());
  EXPECT_THROW(PlanLoader::load(path("missing")), PlanError);
  EXPECT_THROW(PlanLoader::load(path("")), PlanError);
}

}  // namespace
