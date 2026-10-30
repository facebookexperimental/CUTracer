/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#pragma once

// NVBit callback implementation details. Included once after the shared tool
// state and launch geometry definitions in cutracer.cu.
#include "plan_runtime.h"
#include "runtime_identity.h"

static KernelFuncMetadata& get_or_create_kernel_func_metadata(CUfunction func, CUcontext ctx);
static void dump_kernel_cubin(CUcontext ctx, CUfunction func, KernelFuncMetadata& meta);

static cutracer::RuntimeIdentitySession runtime_identity;
static std::atomic<bool> runtime_plan_requested{false};
static cutracer::RuntimeGraphTracker runtime_graph_tracker;

static void runtime_identity_error(const std::string& reason) {
  loprintf("WARNING: runtime identity unavailable: %s\n", reason.c_str());
  try {
    runtime_identity.event([&](auto& ledger) { ledger.gap(reason); });
  } catch (const std::exception& error) {
    loprintf("ERROR: could not preserve runtime gap: %s\n", error.what());
  }
}

template <class F>
static void runtime_event(F&& callback) {
  try {
    runtime_identity.event(std::forward<F>(callback));
  } catch (const std::exception& error) {
    runtime_identity_error(error.what());
  }
}

// NVBit caches instructions, CUBIN bytes and instrumentation per CUfunction and
// does not retire them on module unload, so a handle value reused by a later
// module in the same context cannot be inspected or instrumented reliably.
static bool reused_function_handle(CUcontext ctx, CUfunction func) {
  const auto state = ctx_state_map.find(ctx);
  return state != ctx_state_map.end() && state->second && state->second->retired_function_handles.count(func);
}

static void runtime_plan_error(const std::string& reason) {
  if (auto plan = cutracer::current_runtime_plan()) {
    try {
      plan->disable(reason);
    } catch (const std::exception& error) {
      loprintf("ERROR: could not finalize plan failure: %s\n", error.what());
    }
  }
  runtime_event([&](auto& ledger) { ledger.plan_event("plan_error", {{"reason", reason}}); });
  fprintf(stderr, "CUTracer: instrumentation plan disabled: %s\n", reason.c_str());
  fflush(stderr);
}

template <class F>
static auto apply_runtime_plan(F&& callback) -> decltype(callback(std::declval<cutracer::PlanRuntime&>())) {
  using Result = decltype(callback(std::declval<cutracer::PlanRuntime&>()));
  if (auto plan = cutracer::current_runtime_plan()) {
    try {
      return callback(*plan);
    } catch (const std::exception& error) {
      runtime_plan_error(error.what());
    }
  }
  if constexpr (!std::is_void_v<Result>) return Result{};
}

static void identify_runtime_function(CUcontext ctx, CUfunction func, KernelFuncMetadata& meta) {
  // A failed attempt already recorded its gap for this function lifetime.
  if (!runtime_identity.active() || meta.runtime_identity_attempted) return;
  meta.runtime_identity_attempted = true;
  runtime_event([&](auto& ledger) {
    meta.runtime_identity = ledger.capture_function(
        reinterpret_cast<uint64_t>(ctx), reinterpret_cast<uint64_t>(meta.owner_module),
        reinterpret_cast<uint64_t>(func), [&]() -> nlohmann::json {
          if (reused_function_handle(ctx, func)) {
            throw std::runtime_error("CUfunction handle reused after module unload; NVBit function state may be stale");
          }
          CUcontext current = nullptr;
          CUdevice device = 0;
          CUuuid uuid{};
          if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != ctx || cuCtxGetDevice(&device) != CUDA_SUCCESS ||
              cuDeviceGetUuid(&uuid, device) != CUDA_SUCCESS || !meta.owner_module) {
            throw std::runtime_error("CUDA context device or module identity unavailable");
          }
          if (meta.cubin_sha256.empty()) dump_kernel_cubin(ctx, func, meta);
          if (meta.cubin_sha256.empty()) throw std::runtime_error("NVBit cubin snapshot unavailable");
          std::string device_uuid;
          constexpr char hex[] = "0123456789abcdef";
          for (const auto byte : uuid.bytes) {
            const auto value = static_cast<unsigned char>(byte);
            device_uuid += hex[value >> 4];
            device_uuid += hex[value & 15];
          }
          nlohmann::json instructions = nlohmann::json::array();
          for (auto* instruction : nvbit_get_instrs(ctx, func)) {
            std::vector<uint8_t> bytes;
            instruction->getSassBinary(
                [](uint8_t b, void* data) { static_cast<std::vector<uint8_t>*>(data)->push_back(b); }, &bytes);
            if (bytes.empty()) throw std::runtime_error("NVBit instruction bytes unavailable");
            std::string binary;
            for (const auto byte : bytes) {
              binary += hex[byte >> 4];
              binary += hex[byte & 15];
            }
            instructions.push_back({{"pc", instruction->getOffset()},
                                    {"bytes", binary},
                                    {"opcode_index", instruction->getIdx()},
                                    {"sass", instruction->getSass()}});
          }
          int binary_version = 0, driver_version = 0;
          if (cuFuncGetAttribute(&binary_version, CU_FUNC_ATTRIBUTE_BINARY_VERSION, func) != CUDA_SUCCESS ||
              cuDriverGetVersion(&driver_version) != CUDA_SUCCESS) {
            throw std::runtime_error("CUDA binary or driver version unavailable");
          }
          return {{"entry", meta.mangled_name},
                  {"cubin_sha256", meta.cubin_sha256},
                  {"cubin_path", meta.cubin_path},
                  {"binary_identity_source", "nvbit_dump_cubin"},
                  {"pc_namespace", "function_relative_bytes"},
                  {"instructions", instructions},
                  {"device_ordinal", device},
                  {"device_uuid", device_uuid},
                  {"sm_family", meta.sm_family},
                  {"cuda_binary_version", binary_version},
                  {"cuda_driver_version", driver_version},
                  {"target_suffix_inferred", false}};
        });
  });
}

static std::string runtime_trace_basename(const std::string& legacy, const nlohmann::json& identity) {
  if (identity.is_null()) return legacy;
  const std::string prefix = output_dir.empty() ? "" : output_dir + (output_dir.back() == '/' ? "" : "/");
  return prefix + "capture_" + identity.at("launch").get<std::string>();
}

static nlohmann::json identify_runtime_launch(CUcontext ctx, CUfunction func, uint64_t id, const KernelDimensions& dims,
                                              const ClusterLaunchInfo& cluster, uint64_t stream, const char* kind,
                                              bool selected, uint64_t graph_handle = 0,
                                              std::optional<uint64_t> graph_invocation = std::nullopt) {
  nlohmann::json result;
  if (!runtime_identity.active()) return result;
  auto& meta = get_or_create_kernel_func_metadata(func, ctx);
  identify_runtime_function(ctx, func, meta);
  if (meta.runtime_identity.is_null()) return result;
  runtime_event([&](auto& ledger) {
    result = ledger.launch(
        reinterpret_cast<uint64_t>(ctx), reinterpret_cast<uint64_t>(func), id,
        {{"grid", {dims.gridDimX, dims.gridDimY, dims.gridDimZ}},
         {"block", {dims.blockDimX, dims.blockDimY, dims.blockDimZ}},
         {"cluster_dim", {cluster.x, cluster.y, cluster.z}},
         {"cluster_dim_source", cluster_dim_source_name(cluster.source)},
         {"stream_handle", stream},
         {"launch_kind", kind},
         {"graph_node_handle", graph_handle},
         {"graph_invocation", graph_invocation ? nlohmann::json(*graph_invocation) : nlohmann::json(nullptr)},
         {"capture_selected", selected},
         {"instrumentation_protocol", runtime_plan_requested.load() ? "cutracer.plan-runtime/1" : "legacy/v1"}});
  });
  return result;
}

// The receiver may still be parsing a chunk it acknowledged before the driver
// call returned. Retire the tables it reads only at its next chunk boundary.
static void wait_for_receiver_boundary(CTXstate* state) {
  if (!state || state->recv_thread_done != RecvThreadState::WORKING) return;
  if (!state->capture_completions.wait_for_receiver(std::chrono::seconds(120))) {
    state->channel_failed = true;
    loprintf("ERROR: trace receiver did not reach a chunk boundary before module unload\n");
  }
}

static void forget_runtime_module(CUcontext ctx, CUmodule module) {
  const auto owned = [&](const KernelFuncMetadata& meta) {
    return meta.owner_context == ctx && (!module || meta.owner_module == module);
  };
  const bool traced = std::any_of(kernel_metadata_by_func.begin(), kernel_metadata_by_func.end(),
                                  [&](const auto& entry) { return owned(entry.second); });
  // Context teardown passes no module and has already stopped the receiver.
  const auto state_it = ctx_state_map.find(ctx);
  CTXstate* const state = state_it == ctx_state_map.end() ? nullptr : state_it->second;
  if (traced) wait_for_receiver_boundary(state);
  if (runtime_plan_requested.load()) apply_runtime_plan([&](auto& plan) { plan.forget_module(ctx, module); });
  if (!traced) return;
  // Driver unload has completed. Retire cached metadata along with the real
  // module lifetime so a reused CUfunction cannot inherit the previous binary.
  for (auto it = kernel_metadata_by_func.begin(); it != kernel_metadata_by_func.end();) {
    const auto f = it->first;
    if (owned(it->second)) {
      already_instrumented.erase(f);
      pthread_mutex_lock(&g_cluster_obs_mutex);
      g_func_to_observed_cluster.erase(f);
      pthread_mutex_unlock(&g_cluster_obs_mutex);
      if (state) {
        if (module) state->retired_function_handles.insert(f);
        state->id_to_sass_map.erase(f);
        state->id_to_sass_binary_map.erase(f);
        state->id_to_reg_indices_map.erase(f);
        state->clock_opcode_ids.erase(f);
        state->exit_opcode_ids.erase(f);
        state->instrumented_launch_roots.erase(f);
      }
      it = kernel_metadata_by_func.erase(it);
    } else
      ++it;
  }
}

// NVBit reports graph nodes launched into either default stream as stream 0.
static uint64_t runtime_stream_key(CUstream stream) {
  return stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD ? 0 : reinterpret_cast<uint64_t>(stream);
}

static void finish_runtime_graph(CUcontext ctx, CTXstate* ctx_state, CUstream stream, int api, bool entry_matched,
                                 bool kernel_completed, bool channel_drained) {
  if (!runtime_identity.active()) return;
  const auto exit =
      runtime_graph_tracker.take(reinterpret_cast<uint64_t>(ctx), runtime_stream_key(stream), api, entry_matched);
  if (!exit.matched) {
    ctx_state->channel_failed = true;
    runtime_identity_error("graph_completion_without_matching_invocation");
  }
  for (const auto id : exit.launches) {
    runtime_event([&](auto& ledger) {
      ledger.complete(reinterpret_cast<uint64_t>(ctx), id,
                      kernel_completed && exit.matched && !ctx_state->channel_failed,
                      channel_drained && exit.matched && !ctx_state->channel_failed);
    });
    if (runtime_plan_requested.load()) {
      const bool done = kernel_completed && exit.matched && !ctx_state->channel_failed;
      const bool drained = channel_drained && exit.matched && !ctx_state->channel_failed;
      apply_runtime_plan([&](auto& plan) { plan.device_complete(ctx, id, done, drained); });
      ctx_state->capture_completions.request({id, done, drained, done && drained ? "" : "graph_completion_unverified"});
    }
  }
}

static void observe_runtime_module(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char* name, void* params,
                                   CUresult* pStatus) {
  switch (cbid) {
    case API_CUDA_cuModuleLoad:
    case API_CUDA_cuModuleLoadData:
    case API_CUDA_cuModuleLoadDataEx:
    case API_CUDA_cuModuleLoadFatBinary: {
      if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
        CUmodule module = nullptr;
        if (cbid == API_CUDA_cuModuleLoad)
          module = *static_cast<cuModuleLoad_params*>(params)->module;
        else if (cbid == API_CUDA_cuModuleLoadData)
          module = *static_cast<cuModuleLoadData_params*>(params)->module;
        else if (cbid == API_CUDA_cuModuleLoadDataEx)
          module = *static_cast<cuModuleLoadDataEx_params*>(params)->module;
        else
          module = *static_cast<cuModuleLoadFatBinary_params*>(params)->module;
        runtime_event([&](auto& ledger) {
          ledger.module_load(reinterpret_cast<uint64_t>(ctx), reinterpret_cast<uint64_t>(module), name);
        });
      }
    } break;
    case API_CUDA_cuModuleUnload: {
      if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
        const auto module = static_cast<cuModuleUnload_params*>(params)->hmod;
        runtime_event([&](auto& ledger) {
          ledger.module_unload(reinterpret_cast<uint64_t>(ctx), reinterpret_cast<uint64_t>(module));
        });
        forget_runtime_module(ctx, module);
      }
    } break;
    default:
      break;
  }
}

static void start_runtime_identity() {
  const char* enabled = std::getenv("CUTRACER_RUNTIME_IDENTITY");
  // A plan binds every capture to its launch record, so it implies the manifest.
  if ((!enabled || std::string(enabled) != "1") && !std::getenv("CUTRACER_INSTRUMENTATION_PLAN")) return;
  try {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&nvbit_at_init), &info) || !info.dli_fname) {
      throw std::runtime_error("Loaded tool path unavailable");
    }
    const auto tool_sha256 = cubin_file_sha256(info.dli_fname);
    if (tool_sha256.empty()) throw std::runtime_error("Loaded tool bytes unavailable");
    const auto process = cutracer::runtime_process_identity(tool_sha256);
    const std::string prefix = output_dir.empty() ? "" : output_dir + (output_dir.back() == '/' ? "" : "/");
    const auto path = prefix + "cutracer_runtime_" + std::to_string(getpid()) + "_" +
                      process.at("nonce").get<std::string>() + ".ndjson";
    const int descriptor = open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0644);
    if (descriptor < 0) throw std::runtime_error("Cannot create exclusive runtime manifest");
    auto manifest = std::shared_ptr<int>(new int(descriptor), [](int* fd) {
      close(*fd);
      delete fd;
    });
    runtime_identity.start(process, [manifest](const nlohmann::json& record) {
      const auto bytes = record.dump(-1, ' ', true) + "\n";
      size_t offset = 0;
      while (offset < bytes.size()) {
        const auto count = write(*manifest, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Runtime manifest write failed");
        offset += static_cast<size_t>(count);
      }
    });
    loprintf("Runtime identity manifest: %s\n", path.c_str());
  } catch (const std::exception& error) {
    runtime_identity_error(error.what());
  }
}

static void init_runtime_observation() {
  start_runtime_identity();
  if (const auto* path = std::getenv("CUTRACER_INSTRUMENTATION_PLAN")) {
    runtime_plan_requested.store(true);
    try {
      if (!runtime_identity.active()) {
        throw std::runtime_error("Plan capture requires runtime identity");
      }
      for (const char* option :
           {"CUTRACER_INSTRUMENT", "CUTRACER_ANALYSIS", "KERNEL_FILTERS", "INSTR_BEGIN", "INSTR_END",
            "CUTRACER_INSTR_CATEGORIES", "CUTRACER_INSTRUMENT_IPOINT", "CUTRACER_INSTRUMENT_IPOINT_UNIFORM"}) {
        const auto* value = std::getenv(option);
        if (value && *value) {
          throw std::runtime_error(std::string("Plan capture cannot combine legacy selection or analysis: ") + option);
        }
      }
      if (trace_format != 1 && trace_format != 2) {
        throw std::runtime_error("Plan capture requires NDJSON trace format 1 or 2");
      }
      cutracer::publish_runtime_plan(std::make_shared<cutracer::PlanRuntime>(
          cutracer::plan::PlanLoader::load(path), output_dir, [](const std::string& type, nlohmann::json data) {
            if (!runtime_identity.event([&](auto& ledger) { ledger.plan_event(type, std::move(data)); })) {
              throw cutracer::plan::PlanError("Runtime identity session ended");
            }
          }));
    } catch (const std::exception& error) {
      runtime_plan_error(error.what());
    }
  }
}
