# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Core reduction algorithm for finding minimal race-triggering delay configurations.

Implements two strategies:
- Linear: tests each point one by one (O(N) test runs)
- Bisect: ddmin-style bisection (O(N log N) worst case, often much faster)
"""

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Optional, Sequence

from click import ClickException
from cutracer.cutracer_logger import get_logger
from cutracer.reduce.config_mutator import DelayConfigMutator, DelayPoint
from cutracer.runner import InstrumentationConfig, run_instrumented_target, RunTarget

logger = get_logger("reduce")


class ReplayOutcome(str, Enum):
    """Classification of one deterministic delay-config replay."""

    INTERESTING = "interesting"
    NOT_INTERESTING = "not_interesting"
    INFRA_ERROR = "infra_error"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class ReplayConfig:
    """Inputs needed to replay a candidate with this CUTracer build."""

    oracle_argv: Sequence[str]
    cutracer_so: Optional[str] = None
    kernel_filters: Optional[str] = None
    delay_mode: Optional[str] = None
    delay_warpgroup_id: Optional[int] = None
    delay_warp_mask: Optional[str] = None
    output_dir: Optional[str] = None
    timeout: int = 300
    cwd: Optional[str] = None
    base_env: Optional[Mapping[str, str]] = None
    capture_output: bool = True
    # Exact non-zero oracle codes that mean a valid non-reproduction. An empty
    # tuple makes every non-zero code unexpected rather than accepting all.
    not_interesting_exit_codes: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("replay timeout must be positive")


@dataclass(frozen=True)
class ReplayResult:
    """Structured result of one deterministic delay-config replay."""

    outcome: ReplayOutcome
    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None

    @property
    def interesting(self) -> bool:
        return self.outcome == ReplayOutcome.INTERESTING


class ReplayExecutionError(RuntimeError):
    """Raised when a reducer candidate could not be evaluated reliably."""


class ConfigDoesNotTriggerError(ValueError):
    """Raised when the initial config does not reproduce the race.

    Subclasses ``ValueError`` so callers that catch ``ValueError`` keep working,
    while in-process callers (e.g. the service) can match this stale-config case
    with ``except ConfigDoesNotTriggerError`` instead of string-matching the
    human-readable message.
    """


@dataclass
class ReduceResult:
    """Result of a reduction run."""

    total_points: int
    essential_points: list[DelayPoint]
    iterations: int
    strategy: str = "linear"
    minimal_config_path: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether reduction found essential points."""
        return len(self.essential_points) > 0

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            "Reduction complete!",
            f"  Total points tested: {self.total_points}",
            f"  Essential points found: {len(self.essential_points)}",
            f"  Iterations: {self.iterations}",
        ]
        if self.essential_points:
            lines.append("\nEssential delay points:")
            for i, point in enumerate(self.essential_points, 1):
                lines.append(f"  {i}. {point.sass}")
                lines.append(f"     PC: {point.pc_offset}, Kernel: {point.kernel_name}")
        return "\n".join(lines)


@dataclass
class ReduceConfig:
    """Configuration for reduction."""

    config_path: str
    test_script: Optional[str] = None
    output_path: Optional[str] = None
    verbose: bool = False
    validate_schema: bool = True
    replay: Optional[ReplayConfig] = None
    replay_runner: Optional[RunTarget] = None


def _replay_delay_ns(config_path: str) -> int:
    mutator = DelayConfigMutator(config_path, validate=False)
    delay_ns = int(mutator.config.get("delay_ns", 0))
    if delay_ns <= 0:
        delay_ns = max((point.delay_ns for point in mutator.delay_points), default=0)
    if delay_ns <= 0:
        raise ValueError(f"delay config has no positive delay value: {config_path}")
    return delay_ns


def run_replay(
    config_path: str,
    config: ReplayConfig,
    *,
    runner: Optional[RunTarget] = None,
) -> ReplayResult:
    """Replay one candidate using the CUTracer library from this Python package."""
    if not config.oracle_argv:
        return ReplayResult(
            outcome=ReplayOutcome.INFRA_ERROR,
            returncode=None,
            error="replay oracle argv must not be empty",
        )

    try:
        proc = run_instrumented_target(
            config.oracle_argv,
            InstrumentationConfig(
                cutracer_so=config.cutracer_so,
                instrument="random_delay",
                analysis="random_delay",
                kernel_filters=config.kernel_filters,
                output_dir=config.output_dir,
                delay_ns=_replay_delay_ns(config_path),
                delay_mode=config.delay_mode,
                delay_warpgroup_id=config.delay_warpgroup_id,
                delay_warp_mask=config.delay_warp_mask,
                delay_load_path=config_path,
                cwd=config.cwd,
                timeout=config.timeout,
                base_env=config.base_env,
                capture_output=config.capture_output,
            ),
            runner=runner,
        )
    except subprocess.TimeoutExpired as exc:
        return ReplayResult(
            outcome=ReplayOutcome.TIMED_OUT,
            returncode=None,
            error=str(exc),
        )
    except (ClickException, OSError, TypeError, ValueError) as exc:
        return ReplayResult(
            outcome=ReplayOutcome.INFRA_ERROR,
            returncode=None,
            error=str(exc),
        )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode == 0:
        outcome = ReplayOutcome.INTERESTING
        error = None
    elif proc.returncode in config.not_interesting_exit_codes:
        outcome = ReplayOutcome.NOT_INTERESTING
        error = None
    else:
        outcome = ReplayOutcome.INFRA_ERROR
        error = f"oracle exited with unexpected code {proc.returncode}"
    return ReplayResult(
        outcome=outcome,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        error=error,
    )


def run_test(
    test_script: str,
    config_path: str,
    verbose: bool = False,
) -> bool:
    """
    Run the test script with a given delay config.

    Args:
        test_script: Path to the test script.
        config_path: Path to the delay config JSON file.
        verbose: Whether to print test output.

    Returns:
        True if data race occurred (exit code 0), False otherwise.
    """
    env = os.environ.copy()
    env["CUTRACER_DELAY_LOAD_PATH"] = config_path

    try:
        result = subprocess.run(
            test_script,
            shell=True,
            env=env,
            capture_output=not verbose,
            timeout=300,  # 5 minute timeout
        )
        return result.returncode == 0  # 0 = bad (race), non-zero = good
    except subprocess.TimeoutExpired:
        logger.debug("Test timed out (treating as no race)")
        return False
    except Exception as e:
        logger.debug(f"Test error: {e}")
        return False


def _candidate_is_interesting(config: ReduceConfig, config_path: str) -> bool:
    if config.replay is not None:
        result = run_replay(
            config_path,
            config.replay,
            runner=config.replay_runner,
        )
        if result.outcome in (ReplayOutcome.INFRA_ERROR, ReplayOutcome.TIMED_OUT):
            detail = result.error or result.stderr or result.outcome.value
            raise ReplayExecutionError(f"could not evaluate replay candidate: {detail}")
        return result.interesting
    if config.test_script is not None:
        return run_test(config.test_script, config_path, config.verbose)
    raise ValueError("reduce requires either replay config or test_script")


def _cleanup_temp_file(path: str) -> None:
    """Safely remove a temporary file."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _find_point_in_mutator(
    mutator: DelayConfigMutator, kernel_key: str, pc_offset: str
) -> Optional[DelayPoint]:
    """Find a matching point in a mutator by kernel_key and pc_offset."""
    for p in mutator.enabled_points:
        if p.kernel_key == kernel_key and p.pc_offset == pc_offset:
            return p
    return None


def _test_point_essentiality(
    point: DelayPoint,
    mutator: DelayConfigMutator,
    config: ReduceConfig,
) -> tuple[bool, str]:
    """
    Test if a point is essential by disabling it and running the test.

    Returns:
        Tuple of (is_essential, temp_config_path).
    """
    test_mutator = mutator.clone()
    cloned_point = _find_point_in_mutator(
        test_mutator, point.kernel_key, point.pc_offset
    )
    if cloned_point:
        test_mutator.set_point_enabled(cloned_point, False)

    test_config_path = test_mutator.save()
    try:
        race_occurs = _candidate_is_interesting(config, test_config_path)
    except Exception:
        _cleanup_temp_file(test_config_path)
        raise
    is_essential = not race_occurs

    return is_essential, test_config_path


def _validate_initial_config(
    mutator: DelayConfigMutator,
    config: ReduceConfig,
) -> str:
    """
    Validate that the initial config triggers a race.

    Returns:
        Path to the saved initial config.

    Raises:
        ValueError: If there are no enabled delay points.
        ConfigDoesNotTriggerError: If the initial config does not trigger the
            race (a ``ValueError`` subclass).
    """
    enabled_points = mutator.enabled_points

    if not enabled_points:
        raise ValueError("No enabled delay points in config")

    logger.debug(f"Loaded {len(enabled_points)} enabled delay points")

    initial_config_path = mutator.save()
    try:
        initial_is_interesting = _candidate_is_interesting(config, initial_config_path)
    except Exception:
        _cleanup_temp_file(initial_config_path)
        raise
    if not initial_is_interesting:
        _cleanup_temp_file(initial_config_path)
        raise ConfigDoesNotTriggerError(
            "Initial config does not trigger the race! "
            "Oracle must return 0 (race) with the original config."
        )

    logger.debug("Initial config triggers the race")

    return initial_config_path


def reduce_delay_points(
    config: ReduceConfig,
    progress_callback: Optional[Callable[[int, int, DelayPoint, bool], None]] = None,
) -> ReduceResult:
    """
    Find minimal set of delay points that trigger a data race.

    Uses delta debugging algorithm:
    1. Start with all enabled points
    2. For each point, disable it and test
    3. If race still occurs, the point is not essential
    4. If race disappears, the point is essential

    Args:
        config: Reduction configuration.
        progress_callback: Optional callback for progress updates.
            Called with (current_iteration, total, point, is_essential).

    Returns:
        ReduceResult with essential points and statistics.
    """
    mutator = DelayConfigMutator(config.config_path, validate=config.validate_schema)
    initial_config_path = _validate_initial_config(mutator, config)

    points_to_test = list(mutator.enabled_points)
    essential_points: list[DelayPoint] = []
    iteration = 0

    for point in points_to_test:
        iteration += 1

        logger.debug(f"[{iteration}/{len(points_to_test)}] Testing: {point.sass}")

        try:
            is_essential, test_config_path = _test_point_essentiality(
                point, mutator, config
            )
        except Exception:
            _cleanup_temp_file(initial_config_path)
            raise

        if is_essential:
            logger.debug("   -> Race disappears (point IS essential)")
            essential_points.append(point)
        else:
            logger.debug("   -> Race still occurs (point NOT essential)")
            # Update mutator to keep this point disabled
            original_point = _find_point_in_mutator(
                mutator, point.kernel_key, point.pc_offset
            )
            if original_point:
                mutator.set_point_enabled(original_point, False)

        if progress_callback:
            progress_callback(iteration, len(points_to_test), point, is_essential)

        _cleanup_temp_file(test_config_path)

    # Save minimal config if output path specified
    minimal_config_path = None
    if config.output_path:
        minimal_config_path = mutator.save(config.output_path)
        logger.debug(f"Minimal config saved to: {minimal_config_path}")

    _cleanup_temp_file(initial_config_path)

    return ReduceResult(
        total_points=len(points_to_test),
        essential_points=essential_points,
        iterations=iteration,
        strategy="linear",
        minimal_config_path=minimal_config_path,
    )


def _run_test_with_confidence(
    config: ReduceConfig,
    config_path: str,
    confidence_runs: int,
) -> bool:
    """
    Run test multiple times and use majority voting for probabilistic races.

    Args:
        config: Reduction configuration containing the replay oracle.
        config_path: Path to the delay config JSON file.
        confidence_runs: Number of times to run the test (must be odd).
    Returns:
        True if data race occurred in majority of runs.
    """
    if confidence_runs <= 1:
        return _candidate_is_interesting(config, config_path)

    race_count = 0
    threshold = confidence_runs // 2 + 1

    for i in range(confidence_runs):
        if _candidate_is_interesting(config, config_path):
            race_count += 1
            if race_count >= threshold:
                return True
        else:
            no_race_count = (i + 1) - race_count
            if no_race_count >= threshold:
                return False

    return race_count >= threshold


def _make_config_with_points(
    base_mutator: DelayConfigMutator,
    points_to_enable: list[DelayPoint],
) -> DelayConfigMutator:
    """
    Create a config with only the specified points enabled.

    Args:
        base_mutator: The base mutator to clone from.
        points_to_enable: The points that should be enabled.

    Returns:
        A new mutator with only the specified points enabled.
    """
    mutator = base_mutator.clone()
    mutator.set_all_enabled(False)

    enable_set = {(p.kernel_key, p.pc_offset) for p in points_to_enable}

    for point in mutator.delay_points:
        if (point.kernel_key, point.pc_offset) in enable_set:
            mutator.set_point_enabled(point, True)

    return mutator


def _test_subset(
    base_mutator: DelayConfigMutator,
    points: list[DelayPoint],
    config: ReduceConfig,
    confidence_runs: int,
) -> bool:
    """
    Test if a subset of points triggers the race.

    Returns:
        True if the race occurs with only these points enabled.
    """
    mutator = _make_config_with_points(base_mutator, points)
    config_path = mutator.save()
    try:
        return _run_test_with_confidence(config, config_path, confidence_runs)
    finally:
        _cleanup_temp_file(config_path)


def reduce_bisect(
    config: ReduceConfig,
    confidence_runs: int = 1,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> ReduceResult:
    """
    Find minimal set of delay points using bisection (ddmin algorithm).

    Much faster than linear reduction for large configs. Instead of testing
    each point one by one (O(N) tests), this splits points in half and
    recursively narrows down (typically O(N log N) worst case, often faster).

    The algorithm (adapted from Zeller's ddmin):
    1. Split current points into n chunks
    2. Test each chunk alone - if it triggers, recurse on that chunk
    3. Test complement of each chunk - if it triggers, the chunk is not needed
    4. If neither works, increase granularity (more chunks) and retry
    5. Stop when granularity equals number of points (fully reduced)

    For probabilistic races (e.g., random delay mode), use confidence_runs > 1
    to run each test multiple times with majority voting.

    Args:
        config: Reduction configuration.
        confidence_runs: Number of test runs for majority voting (default: 1).
            Use odd numbers (3, 5) for probabilistic races.
        progress_callback: Optional callback for progress updates.
            Called with (message, current_size, original_size).

    Returns:
        ReduceResult with the minimal set of points.
    """
    base_mutator = DelayConfigMutator(
        config.config_path, validate=config.validate_schema
    )

    initial_config_path = _validate_initial_config(base_mutator, config)
    _cleanup_temp_file(initial_config_path)

    points = list(base_mutator.enabled_points)
    total_points = len(points)
    iteration = 0

    def _ddmin(current_points: list[DelayPoint], n: int) -> list[DelayPoint]:
        nonlocal iteration

        size = len(current_points)

        if size <= 1:
            return current_points

        if n > size:
            n = size

        chunk_size = max(1, size // n)
        chunks = []
        for i in range(0, size, chunk_size):
            chunk = current_points[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)

        logger.debug(
            f"Bisecting {size} points into {len(chunks)} chunks "
            f"(chunk_size ~{chunk_size})"
        )

        # Phase 1: Test each chunk alone
        for i, chunk in enumerate(chunks):
            iteration += 1
            logger.debug(
                f"[{iteration}] Testing chunk {i + 1}/{len(chunks)} "
                f"({len(chunk)} points alone)..."
            )

            if _test_subset(base_mutator, chunk, config, confidence_runs):
                logger.debug(
                    f"-> Chunk {i + 1} alone triggers race! "
                    f"Recursing on {len(chunk)} points"
                )
                if progress_callback:
                    progress_callback(
                        f"Chunk {i + 1} triggers alone", len(chunk), total_points
                    )
                return _ddmin(chunk, 2)

        # Phase 2: Test complement of each chunk
        for i, _chunk in enumerate(chunks):
            iteration += 1
            complement = []
            for j, other_chunk in enumerate(chunks):
                if j != i:
                    complement.extend(other_chunk)

            logger.debug(
                f"[{iteration}] Testing complement of chunk {i + 1} "
                f"({len(complement)} points without chunk)..."
            )

            if _test_subset(
                base_mutator,
                complement,
                config,
                confidence_runs,
            ):
                logger.debug(
                    f"-> Complement triggers race! "
                    f"Chunk {i + 1} not needed, recursing on {len(complement)}"
                )
                if progress_callback:
                    progress_callback(
                        f"Chunk {i + 1} not needed", len(complement), total_points
                    )
                return _ddmin(complement, max(n - 1, 2))

        # Phase 3: Neither worked, increase granularity
        if n < size:
            new_n = min(2 * n, size)
            logger.debug(f"-> No reduction at granularity {n}, increasing to {new_n}")
            return _ddmin(current_points, new_n)

        logger.debug(f"-> Fully reduced to {size} points (cannot split further)")
        return current_points

    logger.debug(f"Starting bisection reduction on {total_points} enabled points")
    logger.debug(f"Confidence runs per test: {confidence_runs}")

    minimal_points = _ddmin(points, 2)

    minimal_config_path = None
    if config.output_path:
        mutator = _make_config_with_points(base_mutator, minimal_points)
        minimal_config_path = mutator.save(config.output_path)
        logger.debug(f"Minimal config saved to: {minimal_config_path}")

    return ReduceResult(
        total_points=total_points,
        essential_points=minimal_points,
        iterations=iteration,
        strategy="bisect",
        minimal_config_path=minimal_config_path,
    )
