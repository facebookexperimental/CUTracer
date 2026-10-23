/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string>
#include <string_view>

struct ExactKernelSelector {
  std::string name;
  std::string checksum;

  bool enabled() const {
    return !name.empty() || !checksum.empty();
  }

  bool matches(std::string_view mangled_name, std::string_view unmangled_name, std::string_view kernel_checksum) const {
    return !name.empty() && !checksum.empty() && checksum == kernel_checksum &&
           (name == mangled_name || name == unmangled_name);
  }
};
