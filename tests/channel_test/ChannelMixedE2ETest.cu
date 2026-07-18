// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "utils/channel.hpp"

namespace {

constexpr uint32_t kLargeType = 0x4c415247;  // "LARG"
constexpr uint32_t kSmallType = 0x534d414c;  // "SMAL"
constexpr uint32_t kLargeReserved = 0xc01dface;
constexpr uint32_t kSmallReserved = 0xc01dcafe;
constexpr int kBlocks = 128;
constexpr int kThreads = 256;
constexpr int kRecordsPerWarp = 200;
constexpr int kChannelBytes = 4 * 1024 * 1024;

struct alignas(8) LargeRecord {
  uint32_t type;
  uint32_t reserved;
  uint64_t sequence;
  uint32_t payload[536];
  uint64_t canary;
};

struct alignas(8) SmallRecord {
  uint32_t type;
  uint32_t reserved;
  uint64_t sequence;
  uint32_t payload[76];
  uint64_t canary;
};

static_assert(sizeof(LargeRecord) == 2168, "must match reg_info_t size");
static_assert(sizeof(SmallRecord) == 328, "must match tma_access_t size");

__host__ __device__ uint32_t payloadValue(uint64_t sequence, uint32_t index, uint32_t type) {
  uint64_t value = sequence * 0x9e3779b97f4a7c15ULL;
  value ^= static_cast<uint64_t>(index) * 0xbf58476d1ce4e5b9ULL;
  value ^= type;
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  return static_cast<uint32_t>(value ^ (value >> 32));
}

__host__ __device__ uint64_t recordCanary(uint64_t sequence, uint32_t type) {
  return sequence ^ (static_cast<uint64_t>(type) << 32) ^ 0xa55a5aa5deadbeefULL;
}

// Mirrors the shape and warp shuffles of instrument_reg_val while eliminating
// uninitialized data as a variable: every lane owns a zeroed large local
// record, live register columns are populated, and the first lane pushes it.
__device__ __noinline__ void pushRegLikeRecord(ChannelDev* channel, uint64_t sequence) {
  const uint32_t active = __ballot_sync(0xffffffffu, 1);
  const auto lane = threadIdx.x & 31;
  const int firstLane = __ffs(active) - 1;

  LargeRecord record{};
  record.type = kLargeType;
  record.reserved = kLargeReserved;
  record.sequence = sequence;
  for (uint32_t reg = 0; reg < 3; ++reg) {
    const uint32_t own = payloadValue(sequence, reg * 32 + lane, kLargeType);
    for (int sourceLane = 0; sourceLane < 32; ++sourceLane) {
      record.payload[sourceLane * 16 + reg] = __shfl_sync(active, own, sourceLane);
    }
  }
  // Same byte offset as reg_info_t::num_uregs, whose corruption caused the
  // original serializer out-of-bounds read.
  record.payload[520] = 0;
  record.canary = recordCanary(sequence, kLargeType);

  if (lane == firstLane) {
    channel->push(&record, sizeof(record));
  }
}

__device__ __noinline__ void pushTmaLikeRecord(ChannelDev* channel, uint64_t sequence) {
  const uint32_t active = __ballot_sync(0xffffffffu, 1);
  const auto lane = threadIdx.x & 31;
  if (lane != __ffs(active) - 1) {
    return;
  }

  SmallRecord record;
  record.type = kSmallType;
  record.reserved = kSmallReserved;
  record.sequence = sequence;
  for (uint32_t index = 0; index < 76; ++index) {
    record.payload[index] = payloadValue(sequence, index, kSmallType);
  }
  record.canary = recordCanary(sequence, kSmallType);
  channel->push(&record, sizeof(record));
}

__global__ void pushMixedRecords(ChannelDev* channel, int recordsPerWarp) {
  const uint64_t warp = static_cast<uint64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  for (int index = 0; index < recordsPerWarp; ++index) {
    const uint64_t sequence = warp * static_cast<uint64_t>(recordsPerWarp) + index;
    __syncwarp();
    pushRegLikeRecord(channel, sequence);
    if ((sequence & 127) == 0) {
      pushTmaLikeRecord(channel, sequence);
    }
  }
}

__global__ void flushChannel(ChannelDev* channel) {
  channel->flush();
}

struct Stats {
  uint64_t records{0};
  uint64_t corrupt{0};
  uint64_t unknown{0};
  uint64_t truncated{0};
};

bool validateLargeRecord(const LargeRecord& record) {
  if (record.type != kLargeType || record.reserved != kLargeReserved ||
      record.canary != recordCanary(record.sequence, kLargeType) || record.payload[520] != 0) {
    return false;
  }
  for (uint32_t reg = 0; reg < 3; ++reg) {
    for (uint32_t lane = 0; lane < 32; ++lane) {
      if (record.payload[lane * 16 + reg] != payloadValue(record.sequence, reg * 32 + lane, kLargeType)) {
        return false;
      }
    }
  }
  return true;
}

bool validateSmallRecord(const SmallRecord& record) {
  if (record.type != kSmallType || record.reserved != kSmallReserved ||
      record.canary != recordCanary(record.sequence, kSmallType)) {
    return false;
  }
  for (uint32_t index = 0; index < 76; ++index) {
    if (record.payload[index] != payloadValue(record.sequence, index, kSmallType)) {
      return false;
    }
  }
  return true;
}

void receiveRecords(ChannelHost* channel, std::atomic<bool>* producerDone, Stats* stats) {
  std::vector<uint8_t> buffer(kChannelBytes);
  int emptyAfterDone = 0;
  while (!producerDone->load(std::memory_order_acquire) || emptyAfterDone < 2) {
    const uint32_t received = channel->recv(buffer.data(), buffer.size());
    if (received == 0) {
      if (producerDone->load(std::memory_order_acquire)) {
        ++emptyAfterDone;
      }
      std::this_thread::yield();
      continue;
    }

    emptyAfterDone = 0;
    uint32_t offset = 0;
    while (offset < received) {
      if (received - offset < sizeof(uint32_t)) {
        ++stats->truncated;
        break;
      }

      uint32_t type = 0;
      std::memcpy(&type, buffer.data() + offset, sizeof(type));
      size_t recordSize = 0;
      bool valid = false;
      if (type == kLargeType) {
        recordSize = sizeof(LargeRecord);
        if (received - offset >= recordSize) {
          LargeRecord record;
          std::memcpy(&record, buffer.data() + offset, sizeof(record));
          valid = validateLargeRecord(record);
        }
      } else if (type == kSmallType) {
        recordSize = sizeof(SmallRecord);
        if (received - offset >= recordSize) {
          SmallRecord record;
          std::memcpy(&record, buffer.data() + offset, sizeof(record));
          valid = validateSmallRecord(record);
        }
      } else {
        ++stats->unknown;
        break;
      }

      if (recordSize == 0 || received - offset < recordSize) {
        ++stats->truncated;
        break;
      }
      ++stats->records;
      stats->corrupt += !valid;
      offset += recordSize;
    }
  }
}

TEST(ChannelE2E, PublishesMixedRecordsBeforeTail) {
  ChannelDev* channelDev = nullptr;
  ASSERT_EQ(cudaMallocManaged(&channelDev, sizeof(ChannelDev)), cudaSuccess);

  ChannelHost channelHost;
  channelHost.init(0, kChannelBytes, channelDev, nullptr);

  std::atomic<bool> producerDone{false};
  Stats stats;
  std::thread receiver(receiveRecords, &channelHost, &producerDone, &stats);

  pushMixedRecords<<<kBlocks, kThreads>>>(channelDev, kRecordsPerWarp);
  flushChannel<<<1, 1>>>(channelDev);
  const cudaError_t kernelStatus = cudaDeviceSynchronize();
  producerDone.store(true, std::memory_order_release);
  receiver.join();

  channelHost.destroy(true);
  const cudaError_t freeStatus = cudaFree(channelDev);

  constexpr uint64_t kWarps = static_cast<uint64_t>(kBlocks) * (kThreads / 32);
  constexpr uint64_t kLargeRecords = kWarps * kRecordsPerWarp;
  constexpr uint64_t kSmallRecords = (kLargeRecords + 127) / 128;
  constexpr uint64_t kExpectedRecords = kLargeRecords + kSmallRecords;

  EXPECT_EQ(kernelStatus, cudaSuccess) << cudaGetErrorString(kernelStatus);
  EXPECT_EQ(freeStatus, cudaSuccess) << cudaGetErrorString(freeStatus);
  EXPECT_EQ(stats.records, kExpectedRecords);
  EXPECT_EQ(stats.corrupt, 0);
  EXPECT_EQ(stats.unknown, 0);
  EXPECT_EQ(stats.truncated, 0);
}

}  // namespace
