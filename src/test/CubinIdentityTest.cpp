/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "TemporaryDirectoryTest.h"
#include "cubin_identity.h"

namespace {

class CubinIdentityTest : public cutracer::test::TemporaryDirectoryTest {};

// ELF64 little-endian header of an EM_CUDA executable with the sm_100a flag word.
std::string sm100a_cubin(uint32_t flags = 0x0a006402) {
  std::string bytes(128, '\0');
  bytes.replace(0, 7,
                std::string("\x7f"
                            "ELF\x02\x01\x01",
                            7));
  bytes[7] = 65;
  bytes[8] = 8;
  const auto put = [&](size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
      bytes[offset + i] = static_cast<char>(value >> (8 * i));
    }
  };
  put(16, 0x00be0002);
  put(48, flags);
  return bytes;
}

void write_file(const std::string& path, const std::string& bytes) {
  std::ofstream file(path, std::ios::binary);
  file << bytes;
}

TEST_F(CubinIdentityTest, CubinIdentityUsesFileBytes) {
  {
    std::ofstream file(path("cubin"), std::ios::binary);
    file << "abc";
  }
  EXPECT_EQ(cubin_file_sha256(path("cubin")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_TRUE(cubin_file_sha256(path("missing")).empty());
  {
    std::ofstream empty(path("empty"), std::ios::binary);
  }
  EXPECT_TRUE(cubin_file_sha256(path("empty")).empty());
}

TEST_F(CubinIdentityTest, ReadFailureDoesNotProvideIdentity) {
  // Opening a directory can succeed even though reading its bytes fails.
  EXPECT_TRUE(cubin_file_sha256(path("")).empty());
}

TEST_F(CubinIdentityTest, BinaryPaddingAndChunkBoundaries) {
  struct TestVector {
    size_t size;
    const char* sha256;
  };

  // Independent hashlib.sha256 goldens for bytes(i % 256 for i in range(size)).
  // Cover both SHA-256 padding boundaries and the helper's 64 KiB input chunks.
  const TestVector vectors[] = {
      {55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"},
      {56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
      {63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"},
      {64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"},
      {65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"},
      {119, "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6"},
      {120, "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c"},
      {127, "92ca0fa6651ee2f97b884b7246a562fa71250fedefe5ebf270d31c546bfea976"},
      {128, "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"},
      {129, "5099c6a56203f9687f7d33f4bfdf576d31dc91f6b695ecea38b2770c87631135"},
      {65535, "5f1bf999bcba5e05d4c34a13710d2e4bff005877874dcce49ac87af61076231e"},
      {65536, "7daca2095d0438260fa849183dfc67faa459fdf4936e1bc91eec6b281b27e4c2"},
      {65537, "2deb0bd2129a9d3aed91e3cff58b3993752be549642890a3e853ec1065f9b617"},
  };
  for (const auto& vector : vectors) {
    SCOPED_TRACE(vector.size);
    std::string bytes(vector.size, '\0');
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<char>(i % 256);
    }
    {
      std::ofstream file(path("cubin"), std::ios::binary);
      file.write(bytes.data(), bytes.size());
      ASSERT_TRUE(file.good());
    }
    EXPECT_EQ(cubin_file_sha256(path("cubin")), vector.sha256);
  }
}

TEST_F(CubinIdentityTest, MillionByteStandardVector) {
  {
    std::ofstream file(path("cubin"), std::ios::binary);
    file << std::string(1000000, 'a');
    ASSERT_TRUE(file.good());
  }
  EXPECT_EQ(cubin_file_sha256(path("cubin")), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_F(CubinIdentityTest, DumpedCubinNamesResolveAgainstTheOutputDirectory) {
  EXPECT_EQ(cubin_artifact_path("", "kernel.cubin"), "kernel.cubin");
  EXPECT_EQ(cubin_artifact_path("traces", "kernel.cubin"), "traces/kernel.cubin");
  EXPECT_EQ(cubin_artifact_path("traces/", "kernel.cubin"), "traces/kernel.cubin");
  EXPECT_EQ(cubin_artifact_path("traces", "/data/kernel.cubin"), "/data/kernel.cubin");
}

TEST_F(CubinIdentityTest, PlanTargetReadsTheCubinUnderTheOutputDirectory) {
  ASSERT_TRUE(std::filesystem::create_directory(path("traces")));
  const std::string name = "plan-target-test.cubin";
  write_file(path("traces/" + name), sm100a_cubin());
  const auto sha256 = cubin_file_sha256(path("traces/" + name));
  EXPECT_EQ(plan_cubin_target(cubin_artifact_path(path("traces"), name), sha256, 100), "sm_100a");
  // The recorded name alone is relative to the trace directory, not the working directory.
  EXPECT_THROW(plan_cubin_target(name, sha256, 100), std::runtime_error);
}

TEST_F(CubinIdentityTest, PlanTargetRejectsChangedBytesAndUnqualifiedTargets) {
  write_file(path("kernel.cubin"), sm100a_cubin());
  const auto sha256 = cubin_file_sha256(path("kernel.cubin"));
  EXPECT_THROW(plan_cubin_target(path("kernel.cubin"), std::string(64, '0'), 100), std::runtime_error);
  EXPECT_THROW(plan_cubin_target(path("kernel.cubin"), sha256, 90), std::runtime_error);
  write_file(path("other.cubin"), sm100a_cubin(0x005a0502));
  EXPECT_THROW(plan_cubin_target(path("other.cubin"), cubin_file_sha256(path("other.cubin")), 100), std::runtime_error);
  write_file(path("short.cubin"), sm100a_cubin().substr(0, 63));
  EXPECT_THROW(plan_cubin_target(path("short.cubin"), cubin_file_sha256(path("short.cubin")), 100), std::runtime_error);
}

}  // namespace
