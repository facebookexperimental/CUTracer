/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <cuda.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>
#include <vector>

#include "launch_attributes.h"

namespace {

using cutracer::describe_function_attributes;
using cutracer::describe_launch_attributes;

CUlaunchAttribute attribute(CUlaunchAttributeID id) {
  CUlaunchAttribute attr{};
  attr.id = id;
  return attr;
}

CUevent fake_event(uintptr_t handle) {
  return reinterpret_cast<CUevent>(handle);
}

TEST(LaunchAttributes, PlainLaunchHasNone) {
  EXPECT_EQ(describe_launch_attributes(nullptr, 0, false), nlohmann::json::object());
}

template <typename Set>
CUlaunchAttribute attribute(CUlaunchAttributeID id, Set set) {
  CUlaunchAttribute attr = attribute(id);
  set(attr.value);
  return attr;
}

struct MappingCase {
  const char* name;
  std::vector<CUlaunchAttribute> attrs;
  nlohmann::json expected;
};

// Each recognized attribute, and what the launch record says about it.
std::vector<MappingCase> mapping_cases() {
  std::vector<MappingCase> cases{
      {"pdl",
       {attribute(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                  [](auto& v) { v.programmaticStreamSerializationAllowed = 1; }),
        attribute(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT,
                  [](auto& v) {
                    v.programmaticEvent.event = fake_event(0x1234);
                    v.programmaticEvent.flags = 0;
                    v.programmaticEvent.triggerAtBlockStart = 1;
                  })},
       {{"programmatic_stream_serialization", true},
        {"programmatic_event", {{"event", 0x1234}, {"flags", 0}, {"trigger_at_block_start", true}}}}},
      {"serialization set to zero",
       {attribute(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                  [](auto& v) { v.programmaticStreamSerializationAllowed = 0; })},
       {{"programmatic_stream_serialization", false}}},
      {"cluster and scheduling",
       {attribute(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, [](auto& v) { v.clusterDim = {2, 1, 1}; }),
        attribute(CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
                  [](auto& v) { v.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD; }),
        attribute(CU_LAUNCH_ATTRIBUTE_PRIORITY, [](auto& v) { v.priority = -1; })},
       {{"cluster_dim", {2, 1, 1}}, {"cluster_scheduling_policy", "spread"}, {"priority", -1}}},
      {"cooperative attribute of a cuLaunchKernelEx call",
       {attribute(CU_LAUNCH_ATTRIBUTE_COOPERATIVE, [](auto& v) { v.cooperative = 1; })},
       {{"cooperative", true}}},
      {"memory synchronization domains",
       {attribute(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN,
                  [](auto& v) { v.memSyncDomain = CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE; }),
        attribute(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP,
                  [](auto& v) {
                    v.memSyncDomainMap.default_ = 0;
                    v.memSyncDomainMap.remote = 1;
                  })},
       {{"mem_sync_domain", "remote"}, {"mem_sync_domain_map", {{"default", 0}, {"remote", 1}}}}},
  };
#if CUDA_VERSION >= 12080
  cases.push_back(
      {"preferred carveout and cluster dimension",
       {attribute(CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, [](auto& v) { v.sharedMemCarveout = 50; }),
        attribute(CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, [](auto& v) { v.preferredClusterDim = {4, 1, 1}; })},
       {{"preferred_shared_memory_carveout", 50}, {"preferred_cluster_dim", {4, 1, 1}}}});
#endif
#if CUDA_VERSION >= 13020
  cases.push_back(
      {"portable cluster size and shared memory modes",
       {attribute(CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE,
                  [](auto& v) { v.portableClusterSizeMode = CU_LAUNCH_PORTABLE_CLUSTER_MODE_ALLOW_NON_PORTABLE; }),
        attribute(CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE,
                  [](auto& v) { v.sharedMemoryMode = CU_SHARED_MEMORY_MODE_REQUIRE_PORTABLE; })},
       {{"portable_cluster_size_mode", "allow_non_portable"}, {"shared_memory_mode", "require_portable"}}});
#endif
  return cases;
}

TEST(LaunchAttributes, EachRecognizedAttributeIsRecorded) {
  for (const auto& c : mapping_cases()) {
    SCOPED_TRACE(c.name);
    EXPECT_EQ(describe_launch_attributes(c.attrs.data(), c.attrs.size(), false), c.expected);
  }
}

TEST(LaunchAttributes, CooperativeApiImpliesCooperative) {
  EXPECT_EQ(describe_launch_attributes(nullptr, 0, true), (nlohmann::json{{"cooperative", true}}));
}

TEST(LaunchAttributes, TheFirstEntryOfARepeatedIdWins) {
  std::vector<CUlaunchAttribute> attrs{
      attribute(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, [](auto& v) { v.clusterDim = {2, 1, 1}; }),
      attribute(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, [](auto& v) { v.clusterDim = {4, 1, 1}; })};
  EXPECT_EQ(describe_launch_attributes(attrs.data(), attrs.size(), false),
            (nlohmann::json{{"cluster_dim", {2, 1, 1}}}));
}

TEST(LaunchAttributes, UnknownIdsAreListedNotDropped) {
  // 15 is unassigned in every CUDA header through 13.4.
  std::vector<CUlaunchAttribute> attrs{attribute(static_cast<CUlaunchAttributeID>(15))};
  EXPECT_EQ(describe_launch_attributes(attrs.data(), attrs.size(), false), (nlohmann::json{{"unrecognized", {15}}}));
}

TEST(LaunchAttributes, IgnoredEntriesAreSkipped) {
  std::vector<CUlaunchAttribute> attrs{attribute(CU_LAUNCH_ATTRIBUTE_IGNORE)};
  EXPECT_EQ(describe_launch_attributes(attrs.data(), attrs.size(), false), nlohmann::json::object());
}

// A driver that reports `values` and fails every other attribute.
auto driver_reporting(const std::map<CUfunction_attribute, int>& values) {
  return [values](CUfunction_attribute attribute, int& value) {
    const auto it = values.find(attribute);
    if (it == values.end()) return false;
    value = it->second;
    return true;
  };
}

TEST(FunctionAttributes, ReportsWhatTheDriverReports) {
  const auto query = driver_reporting(
      {{CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 49152},
       {CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, -1},
       {CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, 0},
       {CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, 2},
       {CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, 1},
       {CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, 1},
       {CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1},
       {CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING}});
  const nlohmann::json expected = {
      {"max_dynamic_shared_size_bytes", 49152},    {"preferred_shared_memory_carveout", -1},
      {"cluster_size_must_be_set", false},         {"required_cluster_dim", {2, 1, 1}},
      {"non_portable_cluster_size_allowed", true}, {"cluster_scheduling_policy", "load_balancing"}};
  EXPECT_EQ(describe_function_attributes(query), expected);
}

TEST(FunctionAttributes, ARequiredClusterDimIsReportedOnlyWhole) {
  const auto query =
      driver_reporting({{CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, 2}, {CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, 1}});
  EXPECT_EQ(describe_function_attributes(query), nlohmann::json::object());
}

}  // namespace
