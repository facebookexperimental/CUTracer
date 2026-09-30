#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Inventory preinstalled CUDA tools without installing or updating packages."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def run(*args, timeout=60):
    print(f"\n$ {shlex.join(args)}")
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, errors="replace", timeout=timeout
        )
        record = {
            "command": list(args),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        record = {"command": list(args), "returncode": None, "error": str(exc)}
    for key in ("stdout", "stderr", "error"):
        if record.get(key):
            print(record[key].rstrip())
    print(f"[exit: {record['returncode']}]")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "phase": args.phase,
        "python": sys.executable,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CONDA_ENV", "VIRTUAL_ENV", "CUDA_HOME", "CUDA_PATH", "CUDACXX",
                "PATH", "LD_LIBRARY_PATH",
            )
        },
    }
    print(json.dumps(report, indent=2))
    print("\nnvidia-smi reports driver compatibility, not the installed toolkit version.")
    report["nvidia_smi"] = run("nvidia-smi")
    report["system_packages"] = run(
        "dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${db:Status-Status}\n",
        "cuda-*", "libcudart*", "libnvrtc*", "nvidia-cuda*",
    )
    print("\nAPT policy uses the existing cache only; availability is not installation.")
    report["apt_policy"] = run(
        "apt-cache", "policy", "cuda-toolkit-13-3", "cuda-nvcc-13-3", "cuda-cudart-13-3"
    )

    # Bounded search of common system/runner install locations. Do not follow
    # arbitrary directory symlinks or scan the whole machine. Supplement this
    # with resolved PATH entries and active Python package file inventories.
    roots = [path for path in ("/usr/local", "/opt", "/workspace") if Path(path).is_dir()]
    report["search_roots"] = roots
    search = run(
        "find", *roots, "-maxdepth", "12",
        "(", "-type", "d", "(", "-name", ".git", "-o", "-name", ".cache",
        "-o", "-name", "__pycache__", "-o", "-name", "node_modules", ")", "-prune", ")",
        "-o", "(", "(", "-type", "f", "-o", "-type", "l", ")",
        "(", "-name", "nvcc", "-o", "-name", "ptxas", "-o",
        "-path", "*/cuda*/version.json", "-o", "-path", "*/cuda*/version.txt", ")",
        "-print", ")",
        timeout=120,
    ) if roots else {"returncode": None, "error": "No search roots exist"}
    report["filesystem_search"] = search
    candidates = {Path(path) for path in search.get("stdout", "").splitlines()}
    for tool in ("nvcc", "ptxas"):
        if path := shutil.which(tool):
            candidates.add(Path(path))
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), sys.prefix, "/usr/lib/cuda"):
        if root:
            candidates.update(Path(root) / name for name in ("bin/nvcc", "bin/ptxas", "version.json", "version.txt"))
    if os.environ.get("CUDACXX"):
        candidates.add(Path(os.environ["CUDACXX"]))

    packages = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "")
        if name.lower().startswith(("nvidia", "cuda", "torch", "triton", "pytorch-triton")):
            packages.append({"name": name, "version": dist.version, "location": str(dist.locate_file(""))})
            for file in dist.files or ():
                if file.name in ("nvcc", "ptxas"):
                    candidates.add(Path(dist.locate_file(file)))
    report["python_packages"] = sorted(packages, key=lambda item: item["name"].lower())
    print("\nInstalled CUDA/PyTorch/Triton Python packages:")
    print(json.dumps(report["python_packages"], indent=2))
    report["torch"] = run(
        sys.executable, "-c",
        "import json, torch; print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda, 'path': torch.__file__}))",
    )

    report["tools"] = []
    report["version_files"] = []
    seen = set()
    for candidate in sorted(candidates):
        if not candidate.is_file():
            continue
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if candidate.name in ("version.json", "version.txt"):
            try:
                contents = path.read_text(errors="replace")
                print(f"\n{candidate} -> {path}:\n{contents}")
                report["version_files"].append({"path": str(path), "contents": contents})
            except OSError as exc:
                report["version_files"].append({"path": str(path), "error": str(exc)})
        else:
            result = run(str(path), "--version", timeout=15)
            output = result.get("stdout", "") + result.get("stderr", "")
            match = re.search(r"\brelease\s+(\d+\.\d+)\b", output)
            report["tools"].append({
                "tool": candidate.name, "path": str(path),
                "release": match.group(1) if match and result["returncode"] == 0 else None,
                **result,
            })

    report["cuda_13_3_nvcc_detected"] = any(
        tool["tool"] == "nvcc" and tool["release"] == "13.3" for tool in report["tools"]
    )
    report["cuda_13_3_ptxas_detected"] = any(
        tool["tool"] == "ptxas" and tool["release"] == "13.3" for tool in report["tools"]
    )
    lane = os.environ.get("CONDA_ENV", "unknown")
    lines = [
        f"### CUDA probe: {lane} / {args.phase}", "",
        f"- CUDA 13.3 nvcc detected: **{report['cuda_13_3_nvcc_detected']}**",
        f"- CUDA 13.3 ptxas detected: **{report['cuda_13_3_ptxas_detected']}**",
        f"- Filesystem search exit code: `{search['returncode']}` (nonzero means the inventory may be incomplete).",
        f"- Python: `{sys.executable}`", "",
        "| Tool | CUDA release | Resolved path |",
        "| --- | --- | --- |",
    ]
    for tool in report["tools"]:
        lines.append(f"| {tool['tool']} | {tool['release'] or 'unknown'} | `{tool['path']}` |")
    if not report["tools"]:
        lines.append("| No nvcc or ptxas detected | | |")
    lines.extend([
        "",
        "A bundled ptxas or Python runtime package does not establish that a full toolkit is installed. The CUDA version in nvidia-smi is driver compatibility. This probe does not run setup-h100.sh, which currently installs CUDA 13.0.",
        "",
        "See the JSON and logs in the uploaded artifact for package versions, toolkit version files, cached APT policy, PyTorch's build version, search scope, and errors.",
        "",
    ])
    summary = "\n".join(lines)
    print("\n" + summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a") as stream:
            stream.write(summary)


if __name__ == "__main__":
    main()
