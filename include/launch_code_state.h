/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <unordered_map>
#include <vector>

namespace cutracer {

// Tracks which code each launched function of a context runs, original or
// instrumented, so a non-blocking launch can tell whether it must first wait
// for running kernels.
//
// By default CUTracer waits for the device to go idle before every direct
// launch. With non-blocking launches it waits only when a launch changes the
// code of a function that an earlier launch may still be running: NVBit swaps
// code in place, for the launch root and every device function it calls, when
// it switches between original and instrumented code or first instruments a
// function. A function no launch has used yet cannot be running, so its first
// launch never waits, instrumented or not; nor do repeated launches with
// unchanged code, or any launch while nothing is instrumented.
//
// Not synchronized: the caller serializes every call.
class LaunchCodeState {
 public:
  // Records that `functions` (a launch root and the device functions it calls)
  // run their instrumented or original code from this launch on. Returns
  // whether that changes the code of a function some earlier launch used.
  // `first_instrumentation` means instrumentation was just inserted, which
  // changes the instrumented code itself.
  bool update(const std::vector<const void*>& functions, bool instrumented, bool first_instrumentation) {
    bool changes_used_code = false;
    for (const void* func : functions) {
      const auto used = instrumented_.find(func);
      if (used != instrumented_.end() && (used->second != instrumented || (instrumented && first_instrumentation))) {
        changes_used_code = true;
      }
      instrumented_[func] = instrumented;
    }
    return changes_used_code;
  }

  // A module unload retired `func`; a reused handle starts over.
  void forget(const void* func) {
    instrumented_.erase(func);
  }

 private:
  // Only functions some launch has used are present.
  std::unordered_map<const void*, bool> instrumented_;
};

}  // namespace cutracer
