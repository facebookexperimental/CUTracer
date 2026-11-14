/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cuda.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

namespace cutracer {

inline const char* cluster_scheduling_policy_name(CUclusterSchedulingPolicy policy) {
  switch (policy) {
    case CU_CLUSTER_SCHEDULING_POLICY_DEFAULT:
      return "default";
    case CU_CLUSTER_SCHEDULING_POLICY_SPREAD:
      return "spread";
    case CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING:
      return "load_balancing";
    default:
      return "unknown";
  }
}

inline const char* mem_sync_domain_name(CUlaunchMemSyncDomain domain) {
  switch (domain) {
    case CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT:
      return "default";
    case CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE:
      return "remote";
    default:
      return "unknown";
  }
}

#if CUDA_VERSION >= 13020
// Both enums share these values: 0 default, 1 require portable, 2 allow non-portable.
inline const char* portability_mode_name(int mode) {
  switch (mode) {
    case 0:
      return "default";
    case 1:
      return "require_portable";
    case 2:
      return "allow_non_portable";
    default:
      return "unknown";
  }
}
#endif

// What a direct launch asked of the driver beyond its geometry: the attributes
// of a cuLaunchKernelEx call, plus `cooperative` for cuLaunchCooperativeKernel.
// Only attributes the launch set appear; when an id repeats, the first entry
// wins, as for the cluster dims CUTracer reads elsewhere. A launch that may
// start before the previous kernel in its stream completes (PDL) has
// `programmatic_stream_serialization: true`; one that a kernel on another
// stream may start early behind has `programmatic_event`, whose `event` is the
// CUevent handle that consumer waits on.
inline nlohmann::json describe_launch_attributes(const CUlaunchAttribute* attrs, unsigned int count,
                                                 bool cooperative_api) {
  nlohmann::json described = nlohmann::json::object();
  if (cooperative_api) described["cooperative"] = true;
  nlohmann::json unrecognized = nlohmann::json::array();
  const auto set = [&described](const char* key, nlohmann::json value) {
    if (!described.contains(key)) described[key] = std::move(value);
  };
  const auto handle = [](CUevent event) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(event)); };
  for (unsigned int i = 0; attrs && i < count; ++i) {
    const CUlaunchAttributeValue& value = attrs[i].value;
    switch (attrs[i].id) {
      case CU_LAUNCH_ATTRIBUTE_IGNORE:
        break;
      case CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW:
        set("access_policy_window",
            {{"num_bytes", value.accessPolicyWindow.num_bytes}, {"hit_ratio", value.accessPolicyWindow.hitRatio}});
        break;
      case CU_LAUNCH_ATTRIBUTE_COOPERATIVE:
        set("cooperative", value.cooperative != 0);
        break;
      case CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION:
        set("cluster_dim", {value.clusterDim.x, value.clusterDim.y, value.clusterDim.z});
        break;
      case CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
        set("cluster_scheduling_policy", cluster_scheduling_policy_name(value.clusterSchedulingPolicyPreference));
        break;
      case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION:
        set("programmatic_stream_serialization", value.programmaticStreamSerializationAllowed != 0);
        break;
      case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT:
        set("programmatic_event", {{"event", handle(value.programmaticEvent.event)},
                                   {"flags", value.programmaticEvent.flags},
                                   {"trigger_at_block_start", value.programmaticEvent.triggerAtBlockStart != 0}});
        break;
      case CU_LAUNCH_ATTRIBUTE_PRIORITY:
        set("priority", value.priority);
        break;
      case CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP:
        set("mem_sync_domain_map",
            {{"default", value.memSyncDomainMap.default_}, {"remote", value.memSyncDomainMap.remote}});
        break;
      case CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN:
        set("mem_sync_domain", mem_sync_domain_name(value.memSyncDomain));
        break;
#if CUDA_VERSION >= 12040
      case CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT:
        set("launch_completion_event",
            {{"event", handle(value.launchCompletionEvent.event)}, {"flags", value.launchCompletionEvent.flags}});
        break;
      case CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE:
        set("device_updatable_kernel_node", value.deviceUpdatableKernelNode.deviceUpdatable != 0);
        break;
#endif
#if CUDA_VERSION >= 12080
      case CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION:
        set("preferred_cluster_dim",
            {value.preferredClusterDim.x, value.preferredClusterDim.y, value.preferredClusterDim.z});
        break;
      case CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
        set("preferred_shared_memory_carveout", value.sharedMemCarveout);
        break;
#endif
#if CUDA_VERSION >= 13000
      case CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING:
        set("nvlink_util_centric_scheduling", value.nvlinkUtilCentricScheduling != 0);
        break;
#endif
#if CUDA_VERSION >= 13020
      case CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE:
        set("portable_cluster_size_mode", portability_mode_name(value.portableClusterSizeMode));
        break;
      case CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE:
        set("shared_memory_mode", portability_mode_name(value.sharedMemoryMode));
        break;
#endif
      default:
        unrecognized.push_back(static_cast<int>(attrs[i].id));
        break;
    }
  }
  if (!unrecognized.empty()) described["unrecognized"] = unrecognized;
  return described;
}

// A function's launch-relevant attributes at one launch (an application can
// change them between launches with cuFuncSetAttribute). `query` reads one
// attribute the way cuFuncGetAttribute does and returns false when the driver
// cannot report it; such attributes are left out.
template <typename Query>
nlohmann::json describe_function_attributes(Query query) {
  nlohmann::json described = nlohmann::json::object();
  int value = 0;
  if (query(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, value)) described["max_dynamic_shared_size_bytes"] = value;
  // -1 means no preference.
  if (query(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, value))
    described["preferred_shared_memory_carveout"] = value;
  if (query(CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, value)) described["cluster_size_must_be_set"] = value != 0;
  int width = 0, height = 0, depth = 0;
  if (query(CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, width) &&
      query(CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, height) &&
      query(CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, depth))
    described["required_cluster_dim"] = {width, height, depth};
  if (query(CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, value))
    described["non_portable_cluster_size_allowed"] = value != 0;
  if (query(CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, value))
    described["cluster_scheduling_policy"] =
        cluster_scheduling_policy_name(static_cast<CUclusterSchedulingPolicy>(value));
  return described;
}

}  // namespace cutracer
