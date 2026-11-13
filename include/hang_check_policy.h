/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace cutracer {

struct HangCheckConfig {
  std::chrono::milliseconds interval{1000};
  uint32_t required_hits{3};
  // The stall must also have lasted this long, measured from the first check
  // that saw no progress. Set it above the longest wait a correct kernel can
  // spend spinning, for example on another rank. Zero leaves it to the hits.
  std::chrono::milliseconds min_stall{0};
};

// When the online deadlock check (-a deadlock_detection) ends a kernel.
//
// The receiver asks `due()` as records arrive. A due check classifies every
// active warp and reports how many there are and how many are still making
// progress. A run of `required_hits` consecutive checks with active warps but
// none progressing terminates the kernel once the run has lasted `min_stall`;
// a check with no active warp starts the count over. Checks are at least
// `interval` apart on a monotonic clock, so a terminating run spans at least
// (required_hits - 1) intervals and at least `min_stall`.
class HangCheckPolicy {
 public:
  using Clock = std::chrono::steady_clock;

  enum class Outcome {
    Idle,       // no active warp to judge; the count starts over
    Progress,   // some active warp is still making progress
    Stall,      // no warp made progress; too few hits, or shorter than `min_stall`
    Terminate,  // `required_hits` checks without progress over `min_stall`
  };

  explicit HangCheckPolicy(HangCheckConfig config = {}) : config_{config} {
  }

  // Starts over at `now`: a new launch, or no active warp left to judge. The
  // next check is due one interval later, and earlier hits no longer count.
  void restart(Clock::time_point now) {
    last_check_ = now;
    hits_ = 0;
  }

  bool due(Clock::time_point now) const {
    return !last_check_ || now - *last_check_ >= config_.interval;
  }

  Outcome record(Clock::time_point now, size_t active_warps, size_t progressing_warps) {
    if (active_warps == 0) {
      restart(now);
      return Outcome::Idle;
    }
    last_check_ = now;
    if (progressing_warps > 0) {
      hits_ = 0;
      return Outcome::Progress;
    }
    if (hits_ == 0) {
      stall_start_ = now;
    }
    ++hits_;
    const bool long_enough = now - stall_start_ >= config_.min_stall;
    return (hits_ >= config_.required_hits && long_enough) ? Outcome::Terminate : Outcome::Stall;
  }

  uint32_t hits() const {
    return hits_;
  }

  // How long the current run of checks without progress has lasted.
  std::chrono::milliseconds stalled_for(Clock::time_point now) const {
    if (hits_ == 0) {
      return std::chrono::milliseconds{0};
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - stall_start_);
  }

  const HangCheckConfig& config() const {
    return config_;
  }

 private:
  HangCheckConfig config_;
  std::optional<Clock::time_point> last_check_;
  Clock::time_point stall_start_{};
  uint32_t hits_ = 0;
};

}  // namespace cutracer
