/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include "cubin_identity.h"

#include <picosha2.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

std::string cubin_file_sha256(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  picosha2::hash256_one_by_one hasher;
  // Bound both file reads and PicoSHA2's additional input buffer.
  std::array<char, 64 * 1024> buffer{};
  bool has_data = false;
  while (input) {
    input.read(buffer.data(), buffer.size());
    const auto size = input.gcount();
    if (size > 0) {
      hasher.process(buffer.begin(), buffer.begin() + size);
      has_data = true;
    }
  }
  if (input.bad() || !input.eof() || !has_data) {
    return {};
  }
  hasher.finish();
  std::array<unsigned char, picosha2::k_digest_size> digest{};
  hasher.get_hash_bytes(digest.begin(), digest.end());

  constexpr char hex[] = "0123456789abcdef";
  std::string result(2 * digest.size(), '0');
  for (size_t i = 0; i < digest.size(); ++i) {
    result[2 * i] = hex[digest[i] >> 4];
    result[2 * i + 1] = hex[digest[i] & 15];
  }
  return result;
}

std::string cubin_artifact_path(const std::string& output_dir, const std::string& name) {
  if (output_dir.empty() || (!name.empty() && name.front() == '/')) {
    return name;
  }
  return output_dir + (output_dir.back() == '/' ? "" : "/") + name;
}

std::string plan_cubin_target(const std::string& path, const std::string& sha256, int sm_family) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto length = input.tellg();
  if (!input || length < 64 || length > std::streamoff{256} * 1024 * 1024) {
    throw std::runtime_error("CUBIN size or read failure: " + path);
  }
  std::string bytes(static_cast<size_t>(length), '\0');
  input.seekg(0);
  if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size())) ||
      picosha2::hash256_hex_string(bytes) != sha256) {
    throw std::runtime_error("CUBIN bytes changed before plan application");
  }
  const auto u32 = [&](size_t offset) {
    uint32_t result = 0;
    for (size_t i = 0; i < 4; ++i) {
      result |= static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + i])) << (8 * i);
    }
    return result;
  };
  // Same full flag words qualified by the native SM100 importer. These are
  // target identities, not a suffix inferred from the executing GPU family.
  const auto flags = u32(48);
  if (bytes.compare(0, 7,
                    std::string("\x7f"
                                "ELF\x02\x01\x01",
                                7)) != 0 ||
      static_cast<uint8_t>(bytes[7]) != 65 || static_cast<uint8_t>(bytes[8]) != 8 || u32(16) != 0x00be0002 ||
      (flags != 0x0a006402 && flags != 0x09006402 && flags != 0x06006402) || sm_family != 100) {
    throw std::runtime_error("Actual CUBIN target is not qualified for plan capture");
  }
  return "sm_100a";
}
