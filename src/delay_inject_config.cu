/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "delay_config_file.h"
#include "delay_inject_config.h"
#include "env_config.h"
#include "log.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// Global delay injection configuration
DelayInjectConfig g_delay_inject_config;

// Replay mode configuration (loaded from file)
static DelayInjectConfig g_replay_config;
static bool g_delay_replay_mode_active = false;

bool load_delay_config(const std::string& filepath);

// Helper function to get current timestamp in ISO 8601 format
static std::string get_current_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

  std::tm tm_now;
  localtime_r(&time_t_now, &tm_now);

  std::ostringstream oss;
  oss << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%S");
  oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
  return oss.str();
}

static json delay_config_json(const DelayInjectConfig& config) {
  json config_json;
  config_json["version"] = config.version;
  config_json["delay_ns"] = config.delay_ns;
  config_json["seed"] = g_delay_seed;

  json kernels_json = json::object();
  for (const auto& [kernel_name, kdc] : config.kernels) {
    json kernel_json;
    kernel_json["kernel_name"] = kdc.kernel_name;
    kernel_json["kernel_checksum"] = kdc.kernel_checksum;
    kernel_json["timestamp"] = kdc.timestamp;

    nlohmann::ordered_json points_json = nlohmann::ordered_json::object();
    for (const auto& [pc_offset, ip] : kdc.instrumentation_points) {
      nlohmann::ordered_json point_json;
      point_json["pc"] = ip.pc_offset;
      point_json["sass"] = ip.sass;
      point_json["delay"] = ip.delay_ns;
      point_json["on"] = ip.enabled;
      if (ip.cluster_seed != 0) {
        point_json["cluster_seed"] = ip.cluster_seed;
      }
      points_json[std::to_string(pc_offset)] = point_json;
    }
    kernel_json["instrumentation_points"] = points_json;
    kernels_json[kernel_name] = kernel_json;
  }
  config_json["kernels"] = kernels_json;
  return config_json;
}

void init_delay_json_config() {
  g_delay_inject_config.version = "1.0";
  g_delay_inject_config.delay_ns = g_delay_ns;
  g_delay_inject_config.kernels.clear();

  // Load replay config if specified
  if (!delay_load_path.empty()) {
    if (!load_delay_config(delay_load_path)) {
      fprintf(stderr, "FATAL: Failed to load replay config from %s\n", delay_load_path.c_str());
      exit(1);
    }
  }
}

KernelDelayInjectConfig* create_kernel_delay_config(const std::string& kernel_name,
                                                    const std::string& kernel_checksum) {
  // Create new entry with kernel_checksum-based key for robust identification
  std::string timestamp = get_current_timestamp();
  // Use kernel_checksum as key for robust kernel identification across recompilations
  // Same kernel with different SASS will get different entries
  std::string key = kernel_name + "_" + kernel_checksum;

  KernelDelayInjectConfig kdc;
  kdc.kernel_name = kernel_name;
  kdc.kernel_checksum = kernel_checksum;
  kdc.timestamp = timestamp;
  g_delay_inject_config.kernels[key] = kdc;

  loprintf_v("Created delay config for kernel: %s (checksum: %s, key: %s)\n", kernel_name.c_str(),
             kernel_checksum.c_str(), key.c_str());
  return &g_delay_inject_config.kernels[key];
}

void register_delay_instrumentation_point(KernelDelayInjectConfig* kdc, Instr* instr, uint32_t delay_ns, bool enabled,
                                          uint32_t cluster_seed) {
  if (!kdc || !instr) {
    return;
  }

  uint64_t pc_offset = instr->getOffset();
  auto ret = kdc->instrumentation_points.emplace(pc_offset, DelayInstrumentationPoint());
  if (!ret.second) {
    return;  // Already registered
  }

  auto& ip = ret.first->second;
  ip.pc_offset = pc_offset;
  ip.sass = std::string(instr->getSass());
  ip.delay_ns = delay_ns;
  ip.enabled = enabled;
  ip.cluster_seed = cluster_seed;
}

void save_delay_config() {
  if (delay_dump_path.empty()) {
    return;
  }

  loprintf_v("Saving delay config to %s (%zu kernels)\n", delay_dump_path.c_str(),
             g_delay_inject_config.kernels.size());

  const std::string error =
      cutracer::merge_delay_config_file(delay_dump_path, delay_config_json(g_delay_inject_config));
  if (!error.empty()) {
    fprintf(stderr, "ERROR: Failed to save delay config to %s: %s\n", delay_dump_path.c_str(), error.c_str());
  }
}

bool load_delay_config(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    fprintf(stderr, "ERROR: Failed to open delay config file: %s\n", filepath.c_str());
    return false;
  }

  try {
    json config_json;
    file >> config_json;

    g_replay_config.version = config_json.value("version", "1.0");
    g_replay_config.delay_ns = config_json.value("delay_ns", 0u);
    g_replay_config.kernels.clear();

    if (config_json.contains("kernels")) {
      for (const auto& [key, kernel_json] : config_json["kernels"].items()) {
        KernelDelayInjectConfig kdc;
        kdc.kernel_name = kernel_json.value("kernel_name", "");
        kdc.kernel_checksum = kernel_json.value("kernel_checksum", "");
        kdc.timestamp = kernel_json.value("timestamp", "");

        if (kernel_json.contains("instrumentation_points")) {
          for (const auto& [pc_str, point_json] : kernel_json["instrumentation_points"].items()) {
            DelayInstrumentationPoint ip;
            ip.pc_offset = std::stoull(pc_str);
            ip.sass = point_json.value("sass", "");
            ip.delay_ns = point_json.value("delay", 0u);
            ip.enabled = point_json.value("on", false);
            ip.cluster_seed = point_json.value("cluster_seed", 0u);
            kdc.instrumentation_points[ip.pc_offset] = ip;
          }
        }

        g_replay_config.kernels[key] = kdc;
        loprintf_v("Loaded kernel config: %s with %zu instrumentation points\n", kdc.kernel_name.c_str(),
                   kdc.instrumentation_points.size());
      }
    }

    g_delay_replay_mode_active = true;
    loprintf_v("Loaded delay config from %s (replay mode active)\n", filepath.c_str());
    loprintf_v("  Version: %s, Delay: %u ns, Kernels: %zu\n", g_replay_config.version.c_str(), g_replay_config.delay_ns,
               g_replay_config.kernels.size());

    return true;

  } catch (const json::exception& e) {
    fprintf(stderr, "ERROR: Failed to parse delay config JSON: %s\n", e.what());
    return false;
  }
}

bool is_delay_replay_mode() {
  return g_delay_replay_mode_active;
}

const std::map<uint64_t, DelayInstrumentationPoint>* get_replay_instrumentation_points(
    const std::string& kernel_name, const std::string& kernel_checksum) {
  if (!g_delay_replay_mode_active) {
    return nullptr;
  }

  // Search through all kernel configs to find a matching kernel by kernel_checksum
  // Since kernel_name is encoded in the checksum, matching checksum implies matching name
  for (const auto& [key, kdc] : g_replay_config.kernels) {
    if (kdc.kernel_checksum == kernel_checksum) {
      // Sanity check: if checksum matches, name must also match (since name is part of checksum)
      assert(kdc.kernel_name == kernel_name && "Checksum matched but kernel name differs - hash collision?");
      loprintf_v("Replay: Found config for kernel %s (checksum: %s) with %zu instrumentation points\n",
                 kernel_name.c_str(), kernel_checksum.c_str(), kdc.instrumentation_points.size());
      return &kdc.instrumentation_points;
    }
  }

  loprintf_v("Replay: No config found for kernel %s (checksum: %s)\n", kernel_name.c_str(), kernel_checksum.c_str());
  return nullptr;
}

bool lookup_replay_config(const std::map<uint64_t, DelayInstrumentationPoint>* replay_points, uint64_t pc_offset,
                          bool& enabled, uint32_t& delay_ns, uint32_t& cluster_seed) {
  if (!replay_points) {
    return false;
  }

  auto it = replay_points->find(pc_offset);
  if (it != replay_points->end()) {
    enabled = it->second.enabled;
    delay_ns = it->second.delay_ns;
    cluster_seed = it->second.cluster_seed;
    return true;
  }

  return false;
}
