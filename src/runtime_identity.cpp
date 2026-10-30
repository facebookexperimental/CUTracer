/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#include "runtime_identity.h"

#include <picosha2.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cutracer {
void RuntimeIdentitySession::start(RuntimeIdentityLedger::Json process, RuntimeIdentityLedger::Sink sink) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (ledger_) throw std::logic_error("Runtime identity session is already active");
  ledger_ = std::make_unique<RuntimeIdentityLedger>(std::move(process), std::move(sink));
}

bool RuntimeIdentitySession::active() {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<bool>(ledger_);
}

bool RuntimeIdentitySession::event(const std::function<void(RuntimeIdentityLedger&)>& callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!ledger_) return false;
  callback(*ledger_);
  return true;
}

void RuntimeIdentitySession::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto ledger = std::move(ledger_);
  if (ledger) ledger->finish();
}

namespace {
using Json = nlohmann::json;

std::string read_line(const char* path) {
  std::ifstream input(path);
  std::string line;
  if (!std::getline(input, line) || line.empty() || line.size() > 16384) {
    throw std::runtime_error(std::string("Cannot read process identity: ") + path);
  }
  return line;
}

std::string hint(const char* name) {
  const char* value = std::getenv(name);
  if (!value) return {};
  std::string result(value);
  if (result.size() > 1024) throw std::runtime_error("Oversized runtime identity hint");
  return result;
}

template <class Map, class Predicate>
void erase_where(Map& map, Predicate predicate) {
  for (auto it = map.begin(); it != map.end();) {
    if (predicate(*it))
      it = map.erase(it);
    else
      ++it;
  }
}
}  // namespace

uint64_t RuntimeGraphTracker::begin(uint64_t context, uint64_t stream, int api) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!context || (frames_.count(context) == 0 && frames_.size() >= 1024)) {
    throw std::logic_error("Invalid or excessive graph contexts");
  }
  auto& frames = frames_[context];
  if (frames.size() >= kMaxDepth) throw std::logic_error("Graph invocation depth budget exhausted");
  const auto ordinal = next_++;
  frames.push_back({ordinal, stream, api, {}});
  return ordinal;
}

std::optional<uint64_t> RuntimeGraphTracker::current(uint64_t context, uint64_t stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = frames_.find(context);
  if (it == frames_.end() || it->second.empty() || it->second.back().stream != stream) return std::nullopt;
  return it->second.back().ordinal;
}

void RuntimeGraphTracker::attach(uint64_t context, uint64_t stream, uint64_t ordinal, uint64_t launch) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = frames_.find(context);
  if (it == frames_.end() || it->second.empty()) throw std::logic_error("Graph node has no invocation");
  auto& frame = it->second.back();
  if (frame.ordinal != ordinal || frame.stream != stream || frame.launches.size() >= 1000000) {
    throw std::logic_error("Graph node invocation changed or exceeded its launch budget");
  }
  frame.launches.push_back(launch);
}

RuntimeGraphTracker::Exit RuntimeGraphTracker::take(uint64_t context, uint64_t stream, int api, bool entry_matched) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = frames_.find(context);
  if (it == frames_.end() || it->second.empty()) return {false, {}};
  auto& frames = it->second;
  if (entry_matched && frames.back().stream == stream && frames.back().api == api) {
    auto launches = std::move(frames.back().launches);
    frames.pop_back();
    if (frames.empty()) frames_.erase(it);
    return {true, std::move(launches)};
  }
  std::vector<uint64_t> incomplete;
  for (auto& frame : frames) incomplete.insert(incomplete.end(), frame.launches.begin(), frame.launches.end());
  frames_.erase(it);
  return {false, std::move(incomplete)};
}

void RuntimeGraphTracker::clear(uint64_t context) {
  std::lock_guard<std::mutex> lock(mutex_);
  frames_.erase(context);
}

std::string runtime_digest(const Json& value) {
  return picosha2::hash256_hex_string(value.dump(-1, ' ', true));
}

Json runtime_process_identity(const std::string& tool_sha256) {
  const auto stat = read_line("/proc/self/stat");
  // comm may contain spaces or ')'. The numeric fields start after its final ')'.
  const auto end = stat.rfind(')');
  if (end == std::string::npos) throw std::runtime_error("Malformed process stat");
  std::istringstream fields(stat.substr(end + 1));
  std::string token, start_ticks;
  for (int field = 3; field <= 22; ++field) {
    if (!(fields >> token)) throw std::runtime_error("Incomplete process stat");
    if (field == 22) start_ticks = token;
  }
  return {{"pid", getpid()},
          {"parent_pid", getppid()},
          {"start_ticks", start_ticks},
          {"boot_id", read_line("/proc/sys/kernel/random/boot_id")},
          {"nonce", read_line("/proc/sys/kernel/random/uuid")},
          {"tool_sha256", tool_sha256},
          {"run_hint", hint("CUTRACER_RUN_ID")},
          {"worker_hint", hint("CUTRACER_WORKER_ID")},
          {"rank_hint", hint("RANK")},
          {"world_size_hint", hint("WORLD_SIZE")},
          {"hints_authenticated", false}};
}

RuntimeIdentityLedger::RuntimeIdentityLedger(Json process, Sink sink, RuntimeIdentityLimits limits)
    : sink_(std::move(sink)), limits_(limits), process_key_(runtime_digest(process)) {
  if (!process.is_object() || !sink_) throw std::invalid_argument("Process object and runtime event sink required");
  emit("process_start", std::move(process));
}

void RuntimeIdentityLedger::live() const {
  if (finished_) throw std::logic_error("Runtime identity ledger is closed");
}

void RuntimeIdentityLedger::emit(const std::string& type, Json data) {
  live();
  try {
    sink_({{"schema", "cutracer.runtime/v1"},
           {"process", process_key_},
           {"sequence", sequence_++},
           {"type", type},
           {"data", std::move(data)}});
  } catch (...) {
    // A transient sink failure must remain visible even if later writes work.
    // A completion not published successfully remains pending for teardown.
    gap_ = true;
    throw;
  }
}

RuntimeIdentityLedger::Context& RuntimeIdentityLedger::context(uint64_t handle) {
  const auto it = contexts_.find(handle);
  if (it == contexts_.end()) throw std::logic_error("Unknown context lifetime");
  return it->second;
}

void RuntimeIdentityLedger::retire_launch_id(uint64_t launch_id) {
  if (!limits_.retired_launch_ids || !retired_launch_ids_.insert(launch_id).second) return;
  retired_launch_order_.push_back(launch_id);
  if (retired_launch_order_.size() > limits_.retired_launch_ids) {
    retired_launch_ids_.erase(retired_launch_order_.front());
    retired_launch_order_.pop_front();
  }
}

std::string RuntimeIdentityLedger::context_begin(uint64_t handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  live();
  if (!handle || contexts_.count(handle) || contexts_.size() >= limits_.contexts) {
    throw std::logic_error("Duplicate invalid or excessive live context");
  }
  const auto key =
      runtime_digest(Json{{"process", process_key_}, {"context_ordinal", context_counter_++}, {"handle", handle}});
  contexts_.emplace(handle, Context{key, {}});
  emit("context_begin", {{"context", key}, {"handle", handle}});
  return key;
}

void RuntimeIdentityLedger::context_end(uint64_t handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  live();
  const auto key = context(handle).key;
  std::vector<std::string> incomplete;
  for (auto it = pending_.begin(); it != pending_.end();) {
    if (it->second.context == handle) {
      incomplete.push_back(it->second.key);
      retire_launch_id(it->first);
      it = pending_.erase(it);
    } else
      ++it;
  }
  const auto in_context = [handle](const auto& entry) { return entry.first.first == handle; };
  erase_where(modules_, in_context);
  erase_where(functions_, in_context);
  erase_where(failed_functions_, in_context);
  contexts_.erase(handle);
  if (!incomplete.empty()) gap_ = true;
  for (const auto& launch : incomplete) {
    emit("launch_incomplete", {{"launch", launch}, {"reason", "context_ended_before_completion"}});
  }
  emit("context_end", {{"context", key}, {"incomplete_launches", incomplete.size()}});
}

std::string RuntimeIdentityLedger::module_locked(uint64_t ctx, uint64_t module, const std::string& api, bool observed) {
  const auto context_key = context(ctx).key;
  if (!module) throw std::logic_error("Invalid module lifetime");
  if (const auto it = modules_.find({ctx, module}); it != modules_.end()) {
    if (observed) throw std::logic_error("Module loaded without ending its previous lifetime");
    return it->second.key;
  }
  if (modules_.size() >= limits_.modules) throw std::logic_error("Excessive module lifetimes");
  const auto key =
      runtime_digest(Json{{"context", context_key}, {"module_ordinal", module_counter_++}, {"handle", module}});
  modules_.emplace(HandleKey{ctx, module}, Module{key, observed});
  emit("module_begin",
       {{"context", context_key}, {"module", key}, {"handle", module}, {"load_observed", observed}, {"api", api}});
  return key;
}

std::string RuntimeIdentityLedger::module_load(uint64_t ctx, uint64_t module, const std::string& api) {
  std::lock_guard<std::mutex> lock(mutex_);
  return module_locked(ctx, module, api, true);
}

void RuntimeIdentityLedger::module_unload(uint64_t ctx, uint64_t module) {
  std::lock_guard<std::mutex> lock(mutex_);
  live();
  const auto context_key = context(ctx).key;
  const auto it = modules_.find({ctx, module});
  if (it == modules_.end()) {
    gap_ = true;
    emit("gap", {{"reason", "unload_without_module_begin"}, {"context", context_key}, {"handle", module}});
    return;
  }
  const auto key = it->second.key;
  modules_.erase(it);
  erase_where(functions_,
              [&](const auto& entry) { return entry.first.first == ctx && entry.second.at("module") == key; });
  erase_where(failed_functions_, [&](const auto& entry) { return entry.first.first == ctx && entry.second == key; });
  emit("module_end", {{"context", context_key}, {"module", key}});
}

Json RuntimeIdentityLedger::function(uint64_t ctx, uint64_t module, uint64_t function, Json attributes) {
  std::lock_guard<std::mutex> lock(mutex_);
  return function_locked(ctx, module, function, std::move(attributes));
}

Json RuntimeIdentityLedger::capture_function(uint64_t ctx, uint64_t module, uint64_t function,
                                             const std::function<Json()>& capture) {
  std::lock_guard<std::mutex> lock(mutex_);
  live();
  if (!function || !capture) throw std::invalid_argument("Function handle and capture callback required");
  const auto module_key = module_locked(ctx, module, "first_seen_function", false);
  const auto key = HandleKey{ctx, function};
  if (const auto it = failed_functions_.find(key); it != failed_functions_.end()) {
    if (it->second != module_key) throw std::logic_error("Live function changed module without unload");
    return nullptr;
  }
  if (const auto it = functions_.find(key); it != functions_.end()) {
    if (it->second.at("module") != module_key) throw std::logic_error("Live function changed module without unload");
    return it->second;
  }
  if (functions_.size() + failed_functions_.size() >= limits_.functions) {
    throw std::logic_error("Excessive function inventory");
  }
  try {
    return function_locked(ctx, module, function, capture());
  } catch (...) {
    failed_functions_.emplace(key, module_key);
    gap_ = true;
    throw;
  }
}

Json RuntimeIdentityLedger::function_locked(uint64_t ctx, uint64_t module, uint64_t function, Json attributes) {
  live();
  if (!function || !attributes.is_object() || !attributes.contains("entry") || !attributes.contains("cubin_sha256")) {
    throw std::invalid_argument("Incomplete function identity");
  }
  const auto module_key = module_locked(ctx, module, "first_seen_function", false);
  const auto key = HandleKey{ctx, function};
  if (failed_functions_.count(key)) throw std::logic_error("Function identity unavailable until its module ends");
  attributes["process"] = process_key_;
  attributes["context"] = context(ctx).key;
  attributes["module"] = module_key;
  attributes["module_load_observed"] = modules_.at({ctx, module}).load_observed;
  attributes["function_handle"] = function;
  attributes["function"] = runtime_digest(attributes);
  if (const auto it = functions_.find(key); it != functions_.end()) {
    if (it->second != attributes) throw std::logic_error("Live function changed identity without unload");
    return it->second;
  }
  if (functions_.size() + failed_functions_.size() >= limits_.functions) {
    throw std::logic_error("Excessive function inventory");
  }
  try {
    emit("function", attributes);
  } catch (...) {
    failed_functions_.emplace(key, module_key);
    throw;
  }
  functions_.emplace(key, attributes);
  return attributes;
}

Json RuntimeIdentityLedger::launch(uint64_t ctx, uint64_t function, uint64_t id, Json attributes) {
  std::lock_guard<std::mutex> lock(mutex_);
  live();
  auto& current = context(ctx);
  const auto fit = functions_.find({ctx, function});
  if (fit == functions_.end() || !attributes.is_object()) throw std::logic_error("Launch lacks function identity");
  if (pending_.count(id) || retired_launch_ids_.count(id)) throw std::logic_error("Duplicate launch ID");
  if (pending_.size() >= limits_.pending_launches) throw std::logic_error("Excessive in-flight launches");
  const bool capture_selected = attributes.value("capture_selected", true);
  const auto& binding = fit->second;
  const auto ordinal_key =
      std::make_pair(binding.at("cubin_sha256").get<std::string>(), binding.at("entry").get<std::string>());
  attributes["process"] = process_key_;
  attributes["context"] = current.key;
  attributes["module"] = binding.at("module");
  attributes["function"] = binding.at("function");
  attributes["entry"] = binding.at("entry");
  attributes["cubin_sha256"] = binding.at("cubin_sha256");
  attributes["launch_id"] = id;
  attributes["kernel_ordinal"] = current.ordinals[ordinal_key]++;
  attributes["launch"] = runtime_digest(attributes);
  try {
    emit("launch", attributes);
  } catch (...) {
    retire_launch_id(id);
    throw;
  }
  pending_.emplace(id, Launch{ctx, attributes.at("launch").get<std::string>(), capture_selected});
  return attributes;
}

void RuntimeIdentityLedger::complete(uint64_t ctx, uint64_t id, bool completed, bool drained) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = pending_.find(id);
  if (it == pending_.end() || it->second.context != ctx) {
    throw std::logic_error("Completion does not identify a pending launch in this context");
  }
  emit("driver_completion", {{"launch", it->second.key},
                             {"launch_id", id},
                             {"context", context(ctx).key},
                             {"kernel_completed", completed},
                             {"channel_drained", drained},
                             {"trace_footer_required", it->second.capture_selected}});
  if (!completed || !drained) gap_ = true;
  retire_launch_id(id);
  pending_.erase(it);
}

void RuntimeIdentityLedger::gap(const std::string& reason) {
  std::lock_guard<std::mutex> lock(mutex_);
  gap_ = true;
  emit("gap", {{"reason", reason}});
}

void RuntimeIdentityLedger::plan_event(const std::string& type, Json data) {
  static const std::set<std::string> types = {"plan_loaded", "plan_applied", "plan_capture_completion", "plan_gap",
                                              "plan_error"};
  if (!types.count(type)) {
    throw std::invalid_argument("Unknown plan runtime event");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  live();
  if (type == "plan_gap" || type == "plan_error") {
    gap_ = true;
  }
  emit(type, std::move(data));
}

void RuntimeIdentityLedger::finish() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_) return;
  emit("process_end", {{"pending_launches", pending_.size()},
                       {"live_contexts", contexts_.size()},
                       {"has_gaps", gap_ || !pending_.empty()},
                       {"trace_footers_required", true}});
  finished_ = true;
}

}  // namespace cutracer
