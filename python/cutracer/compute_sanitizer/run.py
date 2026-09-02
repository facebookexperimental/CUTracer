# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Programmatic Compute Sanitizer execution used by CLI and service callers."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import click

RunTarget = Callable[..., "subprocess.CompletedProcess[str]"]
SupportsTensorOps = Callable[[str], bool]


@dataclass(frozen=True)
class ComputeSanitizerConfig:
    argv: Sequence[str]
    tool: str = "racecheck"
    output_dir: Optional[str] = None
    compute_sanitizer: Optional[str] = None
    cwd: Optional[str] = None
    timeout: Optional[int] = None
    base_env: Optional[Mapping[str, str]] = None
    env: Optional[Mapping[str, str]] = None
    capture_output: bool = True


@dataclass(frozen=True)
class ComputeSanitizerRunResult:
    process: "subprocess.CompletedProcess[str]"
    sanitizer_bin: str
    log_path: Optional[str]
    tensor_ops_enabled: bool


def resolve_compute_sanitizer(override: Optional[str]) -> str:
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise click.ClickException(
                f"--compute-sanitizer path is not a file: {override}"
            )
        if not os.access(path, os.X_OK):
            raise click.ClickException(
                f"--compute-sanitizer path is not executable: {override}"
            )
        return override
    found = shutil.which("compute-sanitizer")
    if found is None:
        raise click.ClickException(
            "compute-sanitizer not found on PATH. Install the CUDA toolkit's "
            "Compute Sanitizer and put `compute-sanitizer` on PATH, or pass "
            "--compute-sanitizer <path>."
        )
    return found


def compute_sanitizer_supports_tensor_ops(sanitizer_bin: str) -> bool:
    try:
        proc = subprocess.run(
            [sanitizer_bin, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "--check-tensor-ops" in ((proc.stdout or "") + (proc.stderr or ""))


def build_compute_sanitizer_argv(
    sanitizer_bin: str,
    tool: str,
    cmd: Sequence[str],
    log_file: Optional[str] = None,
    *,
    check_tensor_ops: bool = False,
) -> list[str]:
    argv = [
        sanitizer_bin,
        "--tool",
        tool,
        "--target-processes",
        "all",
    ]
    if check_tensor_ops:
        argv += ["--check-tensor-ops", "yes"]
    # --report-api-errors is a memcheck option; passing it to the other tools
    # makes some Compute Sanitizer versions reject the invocation.
    if tool == "memcheck":
        argv += ["--report-api-errors", "explicit"]
    if log_file is not None:
        argv += ["--log-file", log_file]
    return argv + list(cmd)


def run_compute_sanitizer(
    config: ComputeSanitizerConfig,
    *,
    runner: Optional[RunTarget] = None,
    supports_tensor_ops: Optional[SupportsTensorOps] = None,
) -> ComputeSanitizerRunResult:
    """Run one target under Compute Sanitizer without invoking the cutracer CLI."""
    if not config.argv:
        raise ValueError("sanitizer target argv must not be empty")

    sanitizer_bin = resolve_compute_sanitizer(config.compute_sanitizer)
    probe = supports_tensor_ops or compute_sanitizer_supports_tensor_ops
    tensor_ops = probe(sanitizer_bin)

    log_path = None
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)
        log_path = os.path.join(config.output_dir, f"{config.tool}.log")
        try:
            os.unlink(log_path)
        except FileNotFoundError:
            pass

    env = dict(os.environ if config.base_env is None else config.base_env)
    if config.env is not None:
        env.update(config.env)
    # Compute Sanitizer and NVBit compete for this injection slot. Never leak a
    # parent CUTracer run into the independent sanitizer branch.
    env.pop("CUDA_INJECTION64_PATH", None)

    run = runner or subprocess.run
    process = run(
        build_compute_sanitizer_argv(
            sanitizer_bin,
            config.tool,
            config.argv,
            log_file=log_path,
            check_tensor_ops=tensor_ops,
        ),
        env=env,
        cwd=config.cwd,
        timeout=config.timeout,
        capture_output=config.capture_output,
        text=True,
        check=False,
    )
    return ComputeSanitizerRunResult(
        process=process,
        sanitizer_bin=sanitizer_bin,
        log_path=log_path,
        tensor_ops_enabled=tensor_ops,
    )
