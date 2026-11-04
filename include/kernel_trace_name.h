/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <sys/types.h>

#include <cstdint>
#include <string>

namespace cutracer {

// Path, without extension, of the files written for one kernel launch:
// "<output_dir>/kernel_<checksum>_iter<iteration>_pid<pid>_<mangled_name>",
// with the name cut to 150 characters. Every process of a run counts its own
// launches in the shared output directory, so two processes that launch the
// same kernel are told apart only by the pid.
inline std::string kernel_trace_basename(const std::string& output_dir, const std::string& kernel_checksum,
                                         uint32_t iteration, pid_t pid, const std::string& mangled_name) {
  std::string path = output_dir;
  if (!path.empty() && path.back() != '/') {
    path += '/';
  }
  return path + "kernel_" + kernel_checksum + "_iter" + std::to_string(iteration) + "_pid" + std::to_string(pid) + "_" +
         mangled_name.substr(0, 150);
}

}  // namespace cutracer
