/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include "delay_config_file.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace cutracer {
namespace {

using json = nlohmann::json;

class FileLock {
 public:
  explicit FileLock(const std::string& path) : fd_{open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644)} {
    if (fd_ >= 0 && flock(fd_, LOCK_EX) != 0) {
      close(fd_);
      fd_ = -1;
    }
  }

  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;

  ~FileLock() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  bool locked() const {
    return fd_ >= 0;
  }

 private:
  int fd_;
};

std::string errno_message(const std::string& what, const std::string& path) {
  return what + " " + path + ": " + std::strerror(errno);
}

bool same_run(const json& existing, const json& config) {
  return existing.is_object() && existing.value("version", json()) == config.value("version", json()) &&
         existing.value("seed", json()) == config.value("seed", json()) &&
         existing.value("delay_ns", json()) == config.value("delay_ns", json());
}

json same_run_kernels(const std::string& path, const json& config) {
  std::ifstream file(path);
  if (!file) {
    return json::object();
  }
  const json existing = json::parse(file, nullptr, /*allow_exceptions=*/false);
  if (!same_run(existing, config)) {
    return json::object();
  }
  const auto kernels = existing.find("kernels");
  return kernels != existing.end() && kernels->is_object() ? *kernels : json::object();
}

}  // namespace

std::string merge_delay_config_file(const std::string& path, const json& config) {
  const std::string lock_path = path + ".lock";
  const FileLock lock{lock_path};
  if (!lock.locked()) {
    return errno_message("cannot lock", lock_path);
  }

  const json own = config.value("kernels", json::object());
  if (own.empty() && access(path.c_str(), F_OK) == 0) {
    return "";
  }
  json kernels = same_run_kernels(path, config);
  for (const auto& [key, kernel] : own.items()) {
    kernels[key] = kernel;
  }
  json merged = config;
  merged["kernels"] = std::move(kernels);

  const std::string tmp_path = path + ".tmp." + std::to_string(getpid());
  {
    std::ofstream out(tmp_path, std::ios::trunc);
    if (!out) {
      return errno_message("cannot write", tmp_path);
    }
    out << merged.dump(2);
    out.close();
    if (!out) {
      std::remove(tmp_path.c_str());
      return "failed to write " + tmp_path;
    }
  }
  if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
    const std::string error = errno_message("cannot replace", path);
    std::remove(tmp_path.c_str());
    return error;
  }
  return "";
}

}  // namespace cutracer
