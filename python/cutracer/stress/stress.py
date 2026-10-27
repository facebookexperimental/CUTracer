# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Core random-delay stress search for ``cutracer stress``.

Runs a correctness oracle repeatedly under random delay injection to discover a
delay configuration that triggers a (data-race) bug, then keeps that config for
deterministic replay / reduction. This is the *discovery* counterpart to
``cutracer reduce`` (which minimizes an already-known triggering config); both
own their oracle-driven delay loops inside CUTracer.

The loop reuses CUTracer's public instrumented-target runner, so a single
attempt is exactly what ``cutracer trace -i random_delay`` would run -- no
self-subprocessing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import click
from cutracer.runner import InstrumentationConfig, run_instrumented_target, RunTarget


@dataclass
class StressConfig:
    """Inputs for a random-delay stress search."""

    oracle_argv: List[str]
    delay_ladder_ns: List[int]
    enable_prob: float = 1.0
    warpgroup_ids: List[int] = field(default_factory=list)
    attempts_per_delay: int = 3
    stop_on_first: bool = True
    # Non-zero oracle exit codes that mean "ran, but did not reproduce". Other
    # non-zero codes are treated as infra errors. Empty => any non-zero code is
    # a clean "did not reproduce" (the standalone default; the service pins [1]).
    not_interesting_exit_codes: List[int] = field(default_factory=list)
    kernel_filters: Optional[str] = None
    # Control runs performed BEFORE the search, with injection instrumented but
    # every point disabled. Without them a reproduction cannot be attributed to
    # the delay on any target that also fails on its own.
    #
    # Defaults to 0 (off) here but to 3 on the `cutracer stress` CLI: the CLI is
    # where a human reads a verdict, while programmatic callers (the service)
    # own their own budget and should opt in explicitly.
    baseline_runs: int = 0
    output_dir: str = "."
    timeout: int = 1800
    cwd: Optional[str] = None
    base_env: Optional[Mapping[str, str]] = None
    resume_result: Optional["StressResult"] = None
    progress_callback: Optional[Callable[["StressResult"], None]] = None
    use_bundled_cuda_tools: bool = True
    expected_kernel_name: Optional[str] = None
    expected_kernel_checksum: Optional[str] = None
    exact_kernel_name: Optional[str] = None
    exact_kernel_checksum: Optional[str] = None

    def __post_init__(self) -> None:
        # The `cutracer stress` CLI rejects an empty ladder with a UsageError,
        # but programmatic callers (the service, which builds this straight from
        # a spec) reach the search loop directly. An empty ladder means the
        # search would sweep nothing, which is a caller mistake, not a clean
        # zero-trial answer. ValueError, not IndexError-from-somewhere-inside:
        # the service classifies ValueError as an INFRA_ERROR.
        if not self.delay_ladder_ns:
            raise ValueError("delay_ladder_ns must contain at least one delay")
        if self.baseline_runs < 0:
            raise ValueError(f"baseline_runs must be >= 0, got {self.baseline_runs}")


@dataclass
class TriggeringConfig:
    """The delay config that reproduced the bug (pins the winning attempt)."""

    config_path: str
    delay_ns: int
    warpgroup_id: Optional[int]
    enable_prob: float
    attempt_index: int
    enabled_point_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_path": self.config_path,
            "delay_ns": self.delay_ns,
            "warpgroup_id": self.warpgroup_id,
            "enable_prob": self.enable_prob,
            "attempt_index": self.attempt_index,
            "enabled_point_count": self.enabled_point_count,
        }


# How a reproduction relates to the target's own failure rate.
ATTRIBUTION_NOT_REPRODUCED = "not_reproduced"
# No control runs were REQUESTED, so the natural rate is unknown and attribution
# is unproven. This is the caller opting out, not a failure.
ATTRIBUTION_NO_BASELINE = "no_baseline"
# A control arm WAS requested but did not produce the samples asked for (runs
# timed out or died with an unexpected exit code). Distinct from `no_baseline`
# on purpose: "the control crashed" must never read as "the user declined a
# control", and unlike `no_baseline` it corroborates nothing.
ATTRIBUTION_BASELINE_INCOMPLETE = "baseline_incomplete"
# The target reproduced with injection disabled: a stress hit may just be that.
ATTRIBUTION_NATURAL = "natural"
# The control never reproduced, so the injected delay is the distinguishing
# variable. Still only as strong as `baseline_completed` samples.
ATTRIBUTION_ATTRIBUTED = "attributed"


@dataclass
class StressResult:
    reproduced: bool
    completed_trials: int
    reproductions: int
    infra_errors: int
    reproduction_rate: float
    triggering: Optional[TriggeringConfig]
    log_path: Optional[str] = None
    # Attempts whose repro command hung (timed out). Counted separately from
    # ``infra_errors`` so a caller can tell "the workload hangs" apart from a
    # generic launch/tooling failure.
    timed_out: int = 0
    # Attempts where the oracle reported the bug but the delay config it ran
    # under enabled NO injection point. The failure is real, but nothing this
    # search did can have caused it -- see `_enabled_point_count`.
    unattributed_reproductions: int = 0
    # Control arm: runs with injection instrumented but every point disabled.
    #
    # `requested` and `completed` are tracked separately because they answer
    # different questions. `requested` is what the caller asked for;
    # `completed` is how many runs produced a usable sample. Collapsing them
    # would let three control runs that all timed out report as
    # `attribution=no_baseline`, i.e. "no control was asked for".
    baseline_requested: int = 0
    baseline_completed: int = 0
    baseline_reproductions: int = 0
    # Control runs that hung / failed to run, counted apart from each other for
    # the same reason the search arm keeps `timed_out` and `infra_errors` apart.
    baseline_timed_out: int = 0
    baseline_infra_errors: int = 0
    attribution: str = ATTRIBUTION_NO_BASELINE

    @property
    def baseline_rate(self) -> float:
        """Reproductions per *usable* control sample.

        Divides by `baseline_completed`, not `baseline_requested`: a run that
        timed out measured nothing, and counting it as a clean control run
        would understate the target's natural failure rate.
        """
        return (
            (self.baseline_reproductions / self.baseline_completed)
            if self.baseline_completed
            else 0.0
        )

    # Exact durable prefix represented by ``log_path``. ``None`` supports old
    # in-memory callers that did not record a boundary; zero is an explicit
    # empty durable prefix. This lets resume discard an uncheckpointed tail.
    log_size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reproduced": self.reproduced,
            "completed_trials": self.completed_trials,
            "reproductions": self.reproductions,
            "unattributed_reproductions": self.unattributed_reproductions,
            "infra_errors": self.infra_errors,
            "timed_out": self.timed_out,
            "log_size_bytes": self.log_size_bytes,
            "reproduction_rate": self.reproduction_rate,
            "baseline_requested": self.baseline_requested,
            "baseline_completed": self.baseline_completed,
            "baseline_reproductions": self.baseline_reproductions,
            "baseline_timed_out": self.baseline_timed_out,
            "baseline_infra_errors": self.baseline_infra_errors,
            "baseline_rate": self.baseline_rate,
            "attribution": self.attribution,
            "triggering_config": (
                None if self.triggering is None else self.triggering.to_dict()
            ),
        }


@dataclass(frozen=True)
class _AttemptResult:
    completed_trials: int
    reproductions: int
    infra_errors: int
    triggering: Optional[TriggeringConfig]
    log: str
    timed_out: int = 0
    unattributed: int = 0


def _enabled_point_count(config_path: str) -> Optional[int]:
    """How many injection points the dumped delay config actually turned on.

    A delay config records every candidate point with an ``on`` flag, and only
    the enabled ones stall. With a low ``--enable-prob`` and a large candidate
    set, a whole attempt can enable nothing: on a target that also fails on its
    own, the oracle then reports the bug during a run that injected no delay at
    all, and attributing that to the config would be wrong.

    ``None`` means the config could not be inspected. That is infrastructure
    failure rather than proof that no delay was injected.
    """
    try:
        with open(config_path) as source:
            payload = json.load(source)
        kernels = payload.get("kernels", {})
        return sum(
            raw_point.get("on") is True
            for raw_kernel in kernels.values()
            if isinstance(raw_kernel, dict)
            for raw_point in raw_kernel.get("instrumentation_points", {}).values()
            if isinstance(raw_point, dict)
        )
    except (AttributeError, OSError, UnicodeError, json.JSONDecodeError):
        return None


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _attempt_log(
    *,
    delay_ns: int,
    warpgroup: Optional[int],
    attempt: int,
    returncode: Optional[int] = None,
    stdout: str = "",
    stderr: str = "",
    error: str = "",
) -> str:
    target = "all" if warpgroup is None else str(warpgroup)
    sections = [
        f"=== delay_ns={delay_ns} warpgroup={target} attempt={attempt} ===",
    ]
    if returncode is not None:
        sections.append(f"returncode={returncode}")
    if stdout:
        sections.extend(("--- stdout ---", stdout))
    if stderr:
        sections.extend(("--- stderr ---", stderr))
    if error:
        sections.extend(("--- error ---", error))
    return "\n".join(sections)


def _validate_delay_config(  # noqa: C901
    path: str,
    *,
    delay_ns: int,
    expected_kernel_name: Optional[str],
    expected_kernel_checksum: Optional[str],
) -> tuple[Optional[str], Optional[tuple[str, str]], Optional[int]]:
    """Return validated kernel identity and enabled-PC count for a config."""
    if not os.path.isfile(path):
        return "CUTracer wrote no delay config", None, None
    try:
        with open(path) as source:
            payload = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"delay config is unreadable: {exc}", None, None
    if not isinstance(payload, dict) or payload.get("version") != "1.0":
        return "delay config has an unsupported or missing version", None, None
    configured_delay = payload.get("delay_ns")
    if type(configured_delay) is not int or configured_delay != delay_ns:
        return (
            "delay config does not match the attempted delay "
            f"({configured_delay!r} != {delay_ns})",
            None,
            None,
        )
    kernels = payload.get("kernels")
    if not isinstance(kernels, dict) or not kernels:
        return "delay config contains no instrumented kernels", None, None

    matched_identity: Optional[tuple[str, str]] = None
    for raw_kernel in kernels.values():
        if not isinstance(raw_kernel, dict):
            continue
        kernel_name = raw_kernel.get("kernel_name")
        kernel_checksum = raw_kernel.get("kernel_checksum")
        if not isinstance(kernel_name, str) or not kernel_name:
            continue
        if not isinstance(kernel_checksum, str) or not kernel_checksum:
            continue
        if expected_kernel_checksum is not None:
            if kernel_checksum != expected_kernel_checksum:
                continue
        elif expected_kernel_name is not None and kernel_name != expected_kernel_name:
            continue
        matched_identity = (kernel_name, kernel_checksum)
        points = raw_kernel.get("instrumentation_points")
        if not isinstance(points, dict):
            continue
        enabled_pcs: set[int] = set()
        for raw_point in points.values():
            if not isinstance(raw_point, dict) or raw_point.get("on") is not True:
                continue
            point_delay = raw_point.get("delay")
            point_pc = raw_point.get("pc")
            point_sass = raw_point.get("sass")
            if (
                type(point_delay) is int
                and point_delay == delay_ns
                and type(point_pc) is int
                and point_pc >= 0
                and isinstance(point_sass, str)
                and point_sass
            ):
                enabled_pcs.add(point_pc)
        if enabled_pcs:
            return None, (kernel_name, kernel_checksum), len(enabled_pcs)

    if matched_identity is None and (
        expected_kernel_name is not None or expected_kernel_checksum is not None
    ):
        return (
            "delay config does not contain the expected kernel identity",
            None,
            None,
        )
    return (
        "delay config contains no enabled point for the expected delay",
        None,
        None,
    )


def _validate_warpgroup_launch(  # noqa: C901
    output_dir: str,
    *,
    warpgroup: int,
    kernel_identity: tuple[str, str],
) -> Optional[str]:
    """Verify that a targeted warpgroup has at least one live warp at launch."""
    _kernel_name, kernel_checksum = kernel_identity
    matched_launch = False
    try:
        names = os.listdir(output_dir)
    except OSError as exc:
        return f"kernel launch evidence is unreadable: {exc}"
    read_errors: list[str] = []
    for name in names:
        if not name.startswith("cutracer_kernel_events_") or not name.endswith(
            ".ndjson"
        ):
            continue
        path = os.path.join(output_dir, name)
        try:
            with open(path) as source:
                for line in source:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        not isinstance(event, dict)
                        or event.get("type") != "kernel_launch"
                        or event.get("kernel_checksum") != kernel_checksum
                    ):
                        continue
                    matched_launch = True
                    block = event.get("block")
                    if (
                        not isinstance(block, list)
                        or len(block) != 3
                        or any(
                            type(dimension) is not int or dimension <= 0
                            for dimension in block
                        )
                    ):
                        continue
                    block_threads = block[0] * block[1] * block[2]
                    block_warps = (block_threads + 31) // 32
                    if 4 * warpgroup < block_warps:
                        return None
        except (OSError, UnicodeError) as exc:
            read_errors.append(f"{name}: {exc}")
    if read_errors:
        return "kernel launch evidence is unreadable: " + "; ".join(read_errors)
    if matched_launch:
        return f"warpgroup {warpgroup} has no live warp in the matching kernel launch"
    return "no matching kernel launch evidence was recorded for warpgroup targeting"


def _run_attempt_in_output_dir(
    config: StressConfig,
    *,
    delay_ns: int,
    warpgroup: Optional[int],
    attempt: int,
    cutracer_so: Optional[str],
    runner: Optional[RunTarget],
    instrumentation_output_dir: str,
) -> _AttemptResult:
    config_path = os.path.join(
        config.output_dir,
        f"delay-{delay_ns}-wg-{warpgroup}-attempt-{attempt}.json",
    )
    try:
        os.unlink(config_path)
    except FileNotFoundError:
        pass
    try:
        proc = run_instrumented_target(
            config.oracle_argv,
            InstrumentationConfig(
                cutracer_so=cutracer_so,
                instrument="random_delay",
                analysis="random_delay",
                kernel_filters=config.kernel_filters,
                exact_kernel_name=config.exact_kernel_name,
                exact_kernel_checksum=config.exact_kernel_checksum,
                output_dir=instrumentation_output_dir,
                delay_ns=delay_ns,
                delay_enable_prob=config.enable_prob,
                delay_mode="random",
                delay_warpgroup_id=warpgroup,
                delay_dump_path=config_path,
                cwd=config.cwd,
                timeout=config.timeout,
                base_env=config.base_env,
                use_bundled_cuda_tools=config.use_bundled_cuda_tools,
                kernel_events="nostack" if warpgroup is not None else None,
            ),
            runner=runner,
        )
    except subprocess.TimeoutExpired as exc:
        _unlink(config_path)
        return _AttemptResult(
            0,
            0,
            0,
            None,
            _attempt_log(
                delay_ns=delay_ns,
                warpgroup=warpgroup,
                attempt=attempt,
                error=f"timed out: {exc}",
            ),
            timed_out=1,
        )
    except (click.ClickException, OSError, ValueError) as exc:
        _unlink(config_path)
        return _AttemptResult(
            0,
            0,
            1,
            None,
            _attempt_log(
                delay_ns=delay_ns,
                warpgroup=warpgroup,
                attempt=attempt,
                error=f"launch failed: {type(exc).__name__}: {exc}",
            ),
        )

    log = _attempt_log(
        delay_ns=delay_ns,
        warpgroup=warpgroup,
        attempt=attempt,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
    is_interesting = proc.returncode == 0
    is_clean = proc.returncode != 0 and (
        not config.not_interesting_exit_codes
        or proc.returncode in config.not_interesting_exit_codes
    )
    if not is_interesting and not is_clean:
        _unlink(config_path)
        return _AttemptResult(
            0,
            0,
            1,
            None,
            log + f"\n--- error ---\nunexpected oracle returncode {proc.returncode}",
        )

    if (
        is_interesting
        and os.path.isfile(config_path)
        and _enabled_point_count(config_path) == 0
    ):
        # Real failure, but this attempt injected nothing (or its config cannot
        # prove otherwise), so it is evidence about the target rather than an
        # attributable triggering configuration.
        _unlink(config_path)
        return _AttemptResult(
            1,
            0,
            0,
            None,
            _attempt_log(
                delay_ns=delay_ns,
                warpgroup=warpgroup,
                attempt=attempt,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                error=(
                    "oracle reported the bug but the delay config enabled 0 "
                    "verifiable injection points: the target reproduces without "
                    "attributable injected delay"
                ),
            ),
            unattributed=1,
        )

    config_error, kernel_identity, enabled_point_count = _validate_delay_config(
        config_path,
        delay_ns=delay_ns,
        expected_kernel_name=config.expected_kernel_name,
        expected_kernel_checksum=config.expected_kernel_checksum,
    )
    if config_error is not None:
        _unlink(config_path)
        return _AttemptResult(
            0,
            0,
            1,
            None,
            log + f"\n--- error ---\n{config_error}",
        )
    assert kernel_identity is not None
    assert enabled_point_count is not None
    if warpgroup is not None:
        targeting_error = _validate_warpgroup_launch(
            instrumentation_output_dir,
            warpgroup=warpgroup,
            kernel_identity=kernel_identity,
        )
        if targeting_error is not None:
            _unlink(config_path)
            return _AttemptResult(
                0,
                0,
                1,
                None,
                log + f"\n--- error ---\n{targeting_error}",
            )

    if is_interesting:
        return _AttemptResult(
            1,
            1,
            0,
            TriggeringConfig(
                config_path=config_path,
                delay_ns=delay_ns,
                warpgroup_id=warpgroup,
                enable_prob=config.enable_prob,
                attempt_index=attempt,
                enabled_point_count=enabled_point_count,
            ),
            log,
        )
    _unlink(config_path)
    return _AttemptResult(1, 0, 0, None, log)


def _run_attempt(
    config: StressConfig,
    *,
    delay_ns: int,
    warpgroup: Optional[int],
    attempt: int,
    cutracer_so: Optional[str],
    runner: Optional[RunTarget],
) -> _AttemptResult:
    event_dir = (
        tempfile.mkdtemp(prefix=".stress_events_", dir=config.output_dir)
        if warpgroup is not None
        else None
    )
    try:
        return _run_attempt_in_output_dir(
            config,
            delay_ns=delay_ns,
            warpgroup=warpgroup,
            attempt=attempt,
            cutracer_so=cutracer_so,
            runner=runner,
            instrumentation_output_dir=event_dir or config.output_dir,
        )
    finally:
        if event_dir is not None:
            shutil.rmtree(event_dir, ignore_errors=True)


def _fsync_directory(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_stress_log(
    output_dir: str,
    log_part: str,
    *,
    append: bool,
) -> tuple[str, int]:
    """Durably append one trial, rolling back a failed partial write."""
    log_path = os.path.join(output_dir, "stress.log")
    previous_size = 0
    try:
        if append:
            try:
                previous_size = os.path.getsize(log_path)
            except FileNotFoundError as exc:
                raise OSError(
                    "durable stress log disappeared before the next append"
                ) from exc
        payload = (("\n" if previous_size else "") + log_part).encode("utf-8")
        with open(log_path, "ab" if append else "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if not append:
            _fsync_directory(output_dir)
        return log_path, previous_size + len(payload)
    except OSError:
        if append:
            try:
                with open(log_path, "r+b") as output:
                    output.truncate(previous_size)
                    output.flush()
                    os.fsync(output.fileno())
            except OSError:
                pass
        else:
            _unlink(log_path)
        raise


def _prepare_stress_log(
    output_dir: str,
    result: Optional[StressResult],
) -> tuple[Optional[str], int]:
    """Start a live append-only log from the last durable checkpoint prefix."""
    log_path = os.path.join(output_dir, "stress.log")
    if result is None:
        _unlink(log_path)
        return None, 0
    attempted = result.completed_trials + result.infra_errors + result.timed_out
    if attempted == 0:
        _unlink(log_path)
        return None, 0
    if result.log_path is None:
        raise ValueError("resumed stress progress requires a persisted log")
    source_size = os.path.getsize(result.log_path)
    expected_size = (
        source_size if result.log_size_bytes is None else result.log_size_bytes
    )
    if expected_size < 0 or source_size < expected_size:
        raise ValueError("resumed stress log is shorter than its durable prefix")

    if os.path.abspath(result.log_path) == os.path.abspath(log_path):
        if source_size > expected_size:
            with open(log_path, "r+b") as output:
                output.truncate(expected_size)
                output.flush()
                os.fsync(output.fileno())
        return log_path, expected_size

    descriptor, temporary = tempfile.mkstemp(prefix=".stress.log.", dir=output_dir)
    try:
        with (
            open(result.log_path, "rb") as source,
            os.fdopen(descriptor, "wb") as output,
        ):
            descriptor = -1
            remaining = expected_size
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(
                        "resumed stress log ended before its durable prefix"
                    )
                output.write(chunk)
                remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, log_path)
        temporary = ""
        _fsync_directory(output_dir)
        return log_path, expected_size
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            _unlink(temporary)


@dataclass(frozen=True)
class _BaselineResult:
    """Outcome of the control arm.

    ``requested`` and ``completed`` differ whenever a control run timed out or
    exited with an unexpected code; see ``StressResult`` for why that gap must
    stay visible.
    """

    requested: int
    completed: int
    reproductions: int
    timed_out: int
    infra_errors: int
    logs: List[str]


def _run_baseline(
    config: StressConfig,
    *,
    cutracer_so: Optional[str],
    runner: Optional[RunTarget],
) -> _BaselineResult:
    """Run the control arm: instrumented, but with every delay point disabled.

    A *matched* control on purpose. Running the oracle bare would also measure a
    natural rate, but it would differ from the search arm in two variables at
    once (instrumentation overhead and injected stalls), so it could not tell
    them apart. `--delay-enable-prob 0` keeps the instrumentation and removes
    only the stalls, which is the variable the search manipulates.

    No dump path is passed: an all-disabled config is not worth keeping, and not
    writing one keeps the baseline out of the search's config namespace.
    """
    completed = 0
    reproduced = 0
    timed_out = 0
    infra_errors = 0
    logs: List[str] = []
    if config.baseline_runs <= 0:
        # A caller that opted out of the control arm must not pay for anything
        # here -- including reading `delay_ladder_ns[0]`.
        return _BaselineResult(0, 0, 0, 0, 0, logs)
    delay_ns = config.delay_ladder_ns[0]
    for attempt in range(config.baseline_runs):
        try:
            proc = run_instrumented_target(
                config.oracle_argv,
                InstrumentationConfig(
                    cutracer_so=cutracer_so,
                    instrument="random_delay",
                    analysis="random_delay",
                    kernel_filters=config.kernel_filters,
                    exact_kernel_name=config.exact_kernel_name,
                    exact_kernel_checksum=config.exact_kernel_checksum,
                    output_dir=config.output_dir,
                    delay_ns=delay_ns,
                    delay_enable_prob=0.0,
                    delay_mode="random",
                    cwd=config.cwd,
                    timeout=config.timeout,
                    base_env=config.base_env,
                    use_bundled_cuda_tools=config.use_bundled_cuda_tools,
                ),
                runner=runner,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out += 1
            logs.append(
                _attempt_log(
                    delay_ns=delay_ns,
                    warpgroup=None,
                    attempt=attempt,
                    error=f"baseline timed out: {exc}",
                )
            )
            continue
        logs.append(
            _attempt_log(
                delay_ns=delay_ns,
                warpgroup=None,
                attempt=attempt,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                error="baseline (injection disabled)",
            )
        )
        if proc.returncode == 0:
            completed += 1
            reproduced += 1
        elif (
            not config.not_interesting_exit_codes
            or proc.returncode in config.not_interesting_exit_codes
        ):
            completed += 1
        else:
            # Same rule as the search arm: an exit code the oracle contract does
            # not cover means the control run did not measure anything.
            infra_errors += 1
    return _BaselineResult(
        requested=config.baseline_runs,
        completed=completed,
        reproductions=reproduced,
        timed_out=timed_out,
        infra_errors=infra_errors,
        logs=logs,
    )


def _attribution(
    *,
    baseline_requested: int,
    baseline_completed: int,
    baseline_reproductions: int,
    triggering: object,
) -> str:
    """Classify what a reproduction can be credited to."""
    if baseline_requested == 0:
        return ATTRIBUTION_NO_BASELINE
    if baseline_reproductions > 0:
        # The target fails with injection disabled, so a search hit is not
        # distinguishable from that. Reported even when the search found a
        # config, because the config may merely have coincided with a failure.
        # Checked before the completeness gate on purpose: a control run that
        # DID reproduce is positive evidence, and stays the stronger answer even
        # if its siblings never finished.
        return ATTRIBUTION_NATURAL
    if baseline_completed < baseline_requested:
        # A control arm was asked for and did not deliver. "0 of 3 control runs
        # produced a sample" is not the same claim as "no control was wanted",
        # and it must not be reported as one.
        return ATTRIBUTION_BASELINE_INCOMPLETE
    if triggering is None:
        return ATTRIBUTION_NOT_REPRODUCED
    return ATTRIBUTION_ATTRIBUTED


def run_stress(  # noqa: C901
    config: StressConfig,
    *,
    cutracer_so: Optional[str] = None,
    runner: Optional[RunTarget] = None,
) -> StressResult:
    """Sweep the delay ladder under random injection until the oracle reproduces.

    A "reproduction" requires an interesting oracle exit (0), a dumped delay
    config (otherwise there is nothing to replay/reduce, so the attempt is
    counted as an infra error), AND at least one enabled injection point in that
    config (otherwise the attempt injected nothing and cannot have caused the
    failure -- counted as an unattributed reproduction).
    """
    os.makedirs(config.output_dir, exist_ok=True)
    resume = config.resume_result
    completed = 0 if resume is None else resume.completed_trials
    reproductions = 0 if resume is None else resume.reproductions
    infra_errors = 0 if resume is None else resume.infra_errors
    timed_out = 0 if resume is None else resume.timed_out
    unattributed = 0 if resume is None else resume.unattributed_reproductions
    baseline = (
        _run_baseline(config, cutracer_so=cutracer_so, runner=runner)
        if resume is None
        else _BaselineResult(
            requested=resume.baseline_requested,
            completed=resume.baseline_completed,
            reproductions=resume.baseline_reproductions,
            timed_out=resume.baseline_timed_out,
            infra_errors=resume.baseline_infra_errors,
            logs=[],
        )
    )
    triggering = None if resume is None else resume.triggering
    stop = False
    warp_targets: List[Optional[int]] = (
        [int(x) for x in config.warpgroup_ids] if config.warpgroup_ids else [None]
    )
    planned_trials = (
        len(config.delay_ladder_ns) * len(warp_targets) * config.attempts_per_delay
    )
    attempted_trials = completed + infra_errors + timed_out
    if min(completed, reproductions, unattributed, infra_errors, timed_out) < 0:
        raise ValueError("resumed stress progress contains negative counts")
    if reproductions + unattributed > completed or attempted_trials > planned_trials:
        raise ValueError("resumed stress progress is outside the campaign plan")
    if triggering is not None and reproductions == 0:
        raise ValueError("resumed triggering config has no reproduced trial")
    if triggering is not None and not os.path.isfile(triggering.config_path):
        raise ValueError("resumed triggering config is unavailable")
    if reproductions > 0 and triggering is None:
        raise ValueError("resumed reproduction is missing its triggering config")
    log_path, log_size_bytes = _prepare_stress_log(config.output_dir, resume)
    if baseline.logs:
        log_path, log_size_bytes = _write_stress_log(
            config.output_dir,
            "\n".join(baseline.logs),
            append=log_path is not None,
        )
    if triggering is not None and config.stop_on_first:
        stop = True

    def progress_result(log_path: Optional[str]) -> StressResult:
        return StressResult(
            reproduced=triggering is not None,
            completed_trials=completed,
            reproductions=reproductions,
            infra_errors=infra_errors,
            reproduction_rate=reproductions / completed if completed else 0.0,
            triggering=triggering,
            log_path=log_path,
            timed_out=timed_out,
            unattributed_reproductions=unattributed,
            baseline_requested=baseline.requested,
            baseline_completed=baseline.completed,
            baseline_reproductions=baseline.reproductions,
            baseline_timed_out=baseline.timed_out,
            baseline_infra_errors=baseline.infra_errors,
            attribution=_attribution(
                baseline_requested=baseline.requested,
                baseline_completed=baseline.completed,
                baseline_reproductions=baseline.reproductions,
                triggering=triggering,
            ),
            log_size_bytes=log_size_bytes if log_path is not None else None,
        )

    trial_ordinal = 0
    for delay_ns in config.delay_ladder_ns:
        for warpgroup in warp_targets:
            for attempt in range(config.attempts_per_delay):
                if trial_ordinal < attempted_trials:
                    trial_ordinal += 1
                    continue
                if stop:
                    break
                outcome = _run_attempt(
                    config,
                    delay_ns=delay_ns,
                    warpgroup=warpgroup,
                    attempt=attempt,
                    cutracer_so=cutracer_so,
                    runner=runner,
                )
                completed += outcome.completed_trials
                reproductions += outcome.reproductions
                infra_errors += outcome.infra_errors
                timed_out += outcome.timed_out
                unattributed += outcome.unattributed
                trial_ordinal += 1
                if outcome.triggering is not None:
                    if triggering is None:
                        triggering = outcome.triggering
                    if config.stop_on_first:
                        stop = True

                try:
                    log_path, log_size_bytes = _write_stress_log(
                        config.output_dir,
                        outcome.log,
                        append=log_path is not None,
                    )
                except OSError:
                    # Surface the just-completed attempt to an adapter before
                    # propagating persistence failure. The adapter can retain
                    # exact counts even though no durable log is available.
                    if config.progress_callback is not None:
                        config.progress_callback(progress_result(None))
                    raise
                if config.progress_callback is not None:
                    config.progress_callback(progress_result(log_path))
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    rate = (reproductions / completed) if completed else 0.0
    return StressResult(
        reproduced=triggering is not None,
        completed_trials=completed,
        reproductions=reproductions,
        infra_errors=infra_errors,
        reproduction_rate=rate,
        triggering=triggering,
        log_path=log_path,
        timed_out=timed_out,
        unattributed_reproductions=unattributed,
        baseline_requested=baseline.requested,
        baseline_completed=baseline.completed,
        baseline_reproductions=baseline.reproductions,
        baseline_timed_out=baseline.timed_out,
        baseline_infra_errors=baseline.infra_errors,
        attribution=_attribution(
            baseline_requested=baseline.requested,
            baseline_completed=baseline.completed,
            baseline_reproductions=baseline.reproductions,
            triggering=triggering,
        ),
        log_size_bytes=log_size_bytes if log_path is not None else None,
    )


def save_report(result: StressResult, path: str) -> None:
    """Atomically write the JSON stress report."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    os.replace(tmp, path)
