/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include "cubin_identity.h"

#include <picosha2.h>

#include <array>
#include <fstream>
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
