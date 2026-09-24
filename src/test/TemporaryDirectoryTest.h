/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace cutracer::test {

class TemporaryDirectoryTest : public testing::Test {
 protected:
  void SetUp() override {
    char directory[] = "/tmp/cutracer-test-XXXXXX";
    const auto* created = mkdtemp(directory);
    ASSERT_NE(created, nullptr);
    directory_ = created;
  }

  void TearDown() override {
    std::filesystem::remove_all(directory_);
  }

  std::string path(const std::string& name) const {
    return directory_ + "/" + name;
  }

 private:
  std::string directory_;
};

}  // namespace cutracer::test
