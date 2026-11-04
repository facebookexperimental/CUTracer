/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "TemporaryDirectoryTest.h"
#include "delay_choice.h"
#include "delay_config_file.h"
#include "nlohmann/json.hpp"

namespace {

using json = nlohmann::json;

class DelayConfigFileTest : public cutracer::test::TemporaryDirectoryTest {};

json point(uint64_t pc, bool on) {
  return {{"pc", pc}, {"sass", "SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [UR4], RZ ;"}, {"delay", 100}, {"on", on}};
}

json kernel(const std::string& name, bool on = true) {
  return {{"kernel_name", name},
          {"kernel_checksum", name + "_sum"},
          {"timestamp", "2026-11-04T00:00:00.000"},
          {"instrumentation_points", {{"16", point(16, on)}}}};
}

json config(uint64_t seed, const std::vector<std::string>& kernels, bool on = true) {
  json entries = json::object();
  for (const auto& name : kernels) {
    entries[name + "_" + name + "_sum"] = kernel(name, on);
  }
  return {{"version", "1.0"}, {"delay_ns", 100}, {"seed", seed}, {"kernels", entries}};
}

json read(const std::string& path) {
  std::ifstream file(path);
  return json::parse(file);
}

TEST_F(DelayConfigFileTest, WritesAConfigWhenNoneExists) {
  const auto dump = path("delay.json");
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, config(7, {"a"})), "");
  EXPECT_EQ(read(dump), config(7, {"a"}));
}

TEST_F(DelayConfigFileTest, KeepsKernelsOfTheSameRun) {
  const auto dump = path("delay.json");
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, config(7, {"a"})), "");
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, config(7, {"b"})), "");
  EXPECT_EQ(read(dump), config(7, {"a", "b"}));
}

TEST_F(DelayConfigFileTest, OwnKernelReplacesTheSameKey) {
  const auto dump = path("delay.json");
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, config(7, {"a"}, /*on=*/false)), "");
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, config(7, {"a"}, /*on=*/true)), "");
  EXPECT_EQ(read(dump), config(7, {"a"}, /*on=*/true));
}

TEST_F(DelayConfigFileTest, ReplacesTheFileOfAnotherRun) {
  const auto dump = path("delay.json");
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, config(7, {"a"})), "");
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, config(8, {"b"})), "");
  EXPECT_EQ(read(dump), config(8, {"b"}));

  json other_delay = config(8, {"c"});
  other_delay["delay_ns"] = 200;
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, other_delay), "");
  EXPECT_EQ(read(dump), other_delay);
}

TEST_F(DelayConfigFileTest, ConfigWithoutKernelsLeavesAnExistingFile) {
  const auto dump = path("delay.json");
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, config(7, {"a"})), "");
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, config(9, {})), "");
  EXPECT_EQ(read(dump), config(7, {"a"}));

  const auto fresh = path("fresh.json");
  ASSERT_EQ(cutracer::merge_delay_config_file(fresh, config(9, {})), "");
  EXPECT_EQ(read(fresh), config(9, {}));
}

TEST_F(DelayConfigFileTest, ReplacesAnUnreadableFile) {
  const auto dump = path("delay.json");
  std::ofstream(dump) << "{\"version\": \"1.0\", \"kern";
  ASSERT_EQ(cutracer::merge_delay_config_file(dump, config(7, {"a"})), "");
  EXPECT_EQ(read(dump), config(7, {"a"}));
}

TEST_F(DelayConfigFileTest, ReportsAnUnwritableDirectory) {
  EXPECT_NE(cutracer::merge_delay_config_file(path("missing/delay.json"), config(7, {"a"})), "");
}

TEST_F(DelayConfigFileTest, ConcurrentProcessesKeepEveryKernel) {
  const auto dump = path("delay.json");
  constexpr int kProcesses = 8;
  std::vector<pid_t> children;
  for (int i = 0; i < kProcesses; ++i) {
    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      bool ok = true;
      for (int round = 0; round < 20; ++round) {
        ok = ok && cutracer::merge_delay_config_file(dump, config(7, {"k" + std::to_string(i)})).empty();
      }
      _exit(ok ? 0 : 1);
    }
    children.push_back(pid);
  }
  for (const pid_t pid : children) {
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  std::vector<std::string> names;
  for (int i = 0; i < kProcesses; ++i) {
    names.push_back("k" + std::to_string(i));
  }
  EXPECT_EQ(read(dump), config(7, names));
}

TEST(DelayChoiceTest, ChoiceDependsOnlyOnSeedKernelAndPc) {
  EXPECT_EQ(cutracer::delay_point_hash(7, "abc", 16, 1), cutracer::delay_point_hash(7, "abc", 16, 1));
  EXPECT_NE(cutracer::delay_point_hash(7, "abc", 16, 1), cutracer::delay_point_hash(8, "abc", 16, 1));
  EXPECT_NE(cutracer::delay_point_hash(7, "abc", 16, 1), cutracer::delay_point_hash(7, "abd", 16, 1));
  EXPECT_NE(cutracer::delay_point_hash(7, "abc", 16, 1), cutracer::delay_point_hash(7, "abc", 32, 1));
  EXPECT_NE(cutracer::delay_point_hash(7, "abc", 16, 1), cutracer::delay_point_hash(7, "abc", 16, 2));
}

TEST(DelayChoiceTest, EndpointProbabilitiesAreExact) {
  for (uint64_t pc = 0; pc < 4096; pc += 16) {
    EXPECT_TRUE(cutracer::delay_point_enabled(7, "abc", pc, 1.0f));
    EXPECT_FALSE(cutracer::delay_point_enabled(7, "abc", pc, 0.0f));
  }
}

TEST(DelayChoiceTest, EnabledFractionFollowsTheProbability) {
  int enabled = 0;
  constexpr int kPoints = 20000;
  for (int i = 0; i < kPoints; ++i) {
    enabled += cutracer::delay_point_enabled(7, "abc", 16 * static_cast<uint64_t>(i), 0.3f) ? 1 : 0;
  }
  EXPECT_NEAR(static_cast<double>(enabled) / kPoints, 0.3, 0.02);
}

}  // namespace
