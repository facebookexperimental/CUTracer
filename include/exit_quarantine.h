/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#ifndef EXIT_QUARANTINE_H
#define EXIT_QUARANTINE_H

#include <stdint.h>

/*
 * The last exiting cohort of a CTA poisons its shared memory, waits, and
 * reports changed 8-byte words. Supported only for direct, blocking launches.
 * A late mbarrier arrive is not seen: on SM100 an arrive on a poisoned barrier
 * word changes nothing. State and counters use device memory: managed memory
 * migrated on first access and delayed the first quarantines by tens of us.
 */
constexpr uint32_t kExitQuarantineMaxReports = 16;

struct ExitQuarantineReport {
  uint32_t ctaid[3];
  uint32_t offset;  // shared-window byte offset of the changed word
  uint64_t expected;
  uint64_t observed;
  uint64_t mask;
};

struct ExitQuarantineState {
  // Parameters and counters are reset by the host at an idle launch boundary.
  uint32_t* exited_threads;
  uint64_t capacity;
  uint64_t launch_id;
  uint32_t enabled;
  uint32_t launch_smem_bytes;
  // cudaDevAttrReservedSharedMemoryPerBlock, below the kernel's shared window.
  uint32_t smem_base;
  uint32_t ctas_quarantined;
  unsigned long long changed_words;
  uint32_t exits_skipped;
  ExitQuarantineReport reports[kExitQuarantineMaxReports];
};

#if defined(__CUDACC__)
#define EXIT_QUARANTINE_HOST_DEVICE __host__ __device__
#else
#define EXIT_QUARANTINE_HOST_DEVICE
#endif

EXIT_QUARANTINE_HOST_DEVICE inline bool exit_quarantine_launch_matches(const ExitQuarantineState& state,
                                                                       uint64_t launch_id) {
  return state.enabled != 0u && state.launch_id == launch_id;
}

EXIT_QUARANTINE_HOST_DEVICE inline uint64_t exit_quarantine_word_mask(uint32_t bytes) {
  return bytes >= 8u ? UINT64_MAX : (uint64_t{1} << (bytes * 8u)) - 1u;
}

/* Distinct for every CTA at a fixed offset, and every offset in a fixed CTA. */
EXIT_QUARANTINE_HOST_DEVICE inline uint64_t exit_quarantine_poison(uint32_t offset, uint64_t cta) {
  return 0xC5C5C5C500000000ull ^ (cta * 0x9E3779B97F4A7C15ull) ^ offset;
}

#endif /* EXIT_QUARANTINE_H */
