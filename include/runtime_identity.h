/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cutracer {

// Invocation ownership is evidence bookkeeping, separate from CUDA's callback
// lock stack. An unpaired exit invalidates this context's outstanding evidence;
// it never assigns another invocation's launches to the exiting stream.
class RuntimeGraphTracker {
 public:
  static constexpr size_t kMaxDepth = 64;

  struct Exit {
    bool matched;
    std::vector<uint64_t> launches;
  };

  uint64_t begin(uint64_t context, uint64_t stream, int api);
  std::optional<uint64_t> current(uint64_t context, uint64_t stream);
  void attach(uint64_t context, uint64_t stream, uint64_t ordinal, uint64_t launch);
  Exit take(uint64_t context, uint64_t stream, int api, bool entry_matched);
  void clear(uint64_t context);

 private:
  struct Frame {
    uint64_t ordinal, stream;
    int api;
    std::vector<uint64_t> launches;
  };

  std::mutex mutex_;
  uint64_t next_ = 0;
  std::map<uint64_t, std::vector<Frame>> frames_;
};

// Budgets bound live state only. A completed launch leaves the in-flight
// budget; its ID stays rejected for a bounded window of recent retirements.
struct RuntimeIdentityLimits {
  size_t contexts = 1024;
  size_t modules = 65536;
  size_t functions = 100000;
  size_t pending_launches = 1000000;
  size_t retired_launch_ids = 65536;
};

// Host-only ledger. Its inputs come from actual CUDA/NVBit callbacks; the
// ledger itself does not turn caller-supplied attributes into hardware facts.
// State is retired before its end record is published, so a failed write
// leaves a gap but never keeps an ended lifetime usable.
class RuntimeIdentityLedger {
 public:
  using Json = nlohmann::json;
  using Sink = std::function<void(const Json&)>;
  RuntimeIdentityLedger(Json process, Sink sink, RuntimeIdentityLimits limits = {});
  std::string context_begin(uint64_t handle);
  void context_end(uint64_t handle);
  std::string module_load(uint64_t context, uint64_t module, const std::string& api);
  void module_unload(uint64_t context, uint64_t module);
  // An identity is served only after its record is published. A failed
  // capture or publication leaves the function unavailable until its module
  // or context ends. capture must not call back into this ledger.
  Json function(uint64_t context, uint64_t module, uint64_t function, Json attributes);
  Json capture_function(uint64_t context, uint64_t module, uint64_t function, const std::function<Json()>& capture);
  Json launch(uint64_t context, uint64_t function, uint64_t launch_id, Json attributes);
  void complete(uint64_t context, uint64_t launch_id, bool kernel_completed, bool channel_drained);
  void gap(const std::string& reason);
  void finish();

  const std::string& process_key() const {
    return process_key_;
  }

 private:
  struct Context {
    std::string key;
    std::map<std::pair<std::string, std::string>, uint64_t> ordinals;
  };

  struct Module {
    std::string key;
    bool load_observed;
  };

  struct Launch {
    uint64_t context;
    std::string key;
    bool capture_selected;
  };

  using HandleKey = std::pair<uint64_t, uint64_t>;
  std::string module_locked(uint64_t context, uint64_t module, const std::string& api, bool observed);
  Json function_locked(uint64_t context, uint64_t module, uint64_t function, Json attributes);
  void retire_launch_id(uint64_t launch_id);
  void emit(const std::string& type, Json data);
  void live() const;
  Context& context(uint64_t handle);
  std::mutex mutex_;
  Sink sink_;
  RuntimeIdentityLimits limits_;
  std::string process_key_;
  uint64_t sequence_ = 0, context_counter_ = 0, module_counter_ = 0;
  bool finished_ = false, gap_ = false;
  std::map<uint64_t, Context> contexts_;
  std::map<HandleKey, Module> modules_;
  std::map<HandleKey, Json> functions_;
  std::map<HandleKey, std::string> failed_functions_;
  std::map<uint64_t, Launch> pending_;
  std::set<uint64_t> retired_launch_ids_;
  std::deque<uint64_t> retired_launch_order_;
};

// Holds the ledger and its sink resources across callback/termination races.
// The sink must not call back into this owner while an event is in progress.
class RuntimeIdentitySession {
 public:
  void start(RuntimeIdentityLedger::Json process, RuntimeIdentityLedger::Sink sink);
  bool active();
  bool event(const std::function<void(RuntimeIdentityLedger&)>& callback);
  void stop();

 private:
  std::mutex mutex_;
  std::unique_ptr<RuntimeIdentityLedger> ledger_;
};

std::string runtime_digest(const nlohmann::json& value);
nlohmann::json runtime_process_identity(const std::string& tool_sha256);

}  // namespace cutracer
