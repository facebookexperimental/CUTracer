# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Confirm a ``cutracer stress`` finding by reducing and replaying it.

A search hit is one run. ``confirm_finding`` replays the triggering config
under the same targeting (``cutracer reduce``'s initial check), reduces it to
the delay points that matter with majority voting, and replays the minimal
config once more. Only a finding that survives all three stays a finding.

A confirmed finding then gets two checks of what its delay needs, each a
majority over replays of the minimal config: whether it still reproduces
with the target's warps left undelayed and every other warp delayed (the
target check), and whether it reproduces with every delay set to 0 ns, its
points still instrumented (the zero-delay check). Read together, they say
whether the target's delay, any delay, or only the instrumentation is what
the finding needs. They qualify what the finding says, not the verdict.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from cutracer.reduce.reduce import (
    ConfigDoesNotTriggerError,
    reduce_bisect,
    ReduceConfig,
    ReplayConfig,
    ReplayExecutionError,
    ReplayOutcome,
    ReplayResult,
    run_replay,
)
from cutracer.runner import RunTarget
from cutracer.stress.modes import get_mode
from cutracer.stress.stress import (
    _delay_patterns,
    _launch_options,
    _MAX_WARPS,
    _remaining_s,
    _run_timeout,
    _target_fields,
    _triggering_target,
    ATTRIBUTION_ATTRIBUTED,
    ATTRIBUTION_NATURAL,
    DelayTarget,
    MIN_RUN_BUDGET_S,
    StressConfig,
    StressResult,
    SYMPTOM_HANG,
    TriggeringConfig,
    Verdict,
    VERDICT_FINDING,
    VERDICT_INCOMPLETE,
    VERDICT_SUSPECT,
)

CONFIRMED = "confirmed"
# A replay of the triggering or the minimal config did not reproduce.
NOT_REPRODUCED = "not_reproduced"
# A replay could not be evaluated, or the time budget ran out.
UNCONFIRMED = "unconfirmed"

# How a target check came out: the minimal config reproduced with the
# target's warps undelayed, or did not; a check that could not decide; a
# target that names no warps to leave out.
TARGET_NOT_NEEDED = "not_needed"
TARGET_NEEDED = "not_reproduced_without_target"
CHECK_INCONCLUSIVE = "inconclusive"
TARGET_NOT_APPLICABLE = "not_applicable"
# How the zero-delay check came out.
ZERO_DELAY_REPRODUCED = "reproduced"
ZERO_DELAY_NOT_REPRODUCED = "not_reproduced"

_ALL_WARPS = (1 << _MAX_WARPS) - 1


@dataclass(frozen=True)
class ReplayCheck:
    """A majority vote over replays of the minimal config with one thing
    changed: the target's warps left out of the delay, or every delay set
    to 0 ns."""

    status: str
    reason: str
    replays: int = 0
    reproductions: int = 0
    # The search did not show that the bug stays away without a delay (its
    # control runs reproduced, or it had none), so a reproduction here may be
    # the bug's own rate.
    weak: bool = False
    # What the target check left out, and the warp mask it delayed instead.
    excluded: Optional[str] = None
    delay_warp_mask: Optional[str] = None
    # The delay config the replays loaded, and the `cutracer trace` options
    # that replay it (add `--delay-load-path <config> -- <oracle>`).
    config_path: Optional[str] = None
    replay_args: List[str] = field(default_factory=list)
    # Environment variables the replays ran without, which a pasted replay
    # command must unset too.
    unset_env: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Confirmation:
    status: str
    reason: str
    replays: int = 0
    minimal_config_path: Optional[str] = None
    # Each point of the minimal config: kernel, PC and SASS.
    essential_points: List[Dict[str, str]] = field(default_factory=list)
    # Only for a confirmed finding (see the module docstring).
    target_check: Optional[ReplayCheck] = None
    zero_delay_check: Optional[ReplayCheck] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "replays": self.replays,
            "minimal_config_path": self.minimal_config_path,
            "essential_points": list(self.essential_points),
            "target_check": None
            if self.target_check is None
            else self.target_check.to_dict(),
            "zero_delay_check": None
            if self.zero_delay_check is None
            else self.zero_delay_check.to_dict(),
        }


class _BudgetExhausted(Exception):
    pass


def replay_config(
    config: StressConfig,
    triggering: TriggeringConfig,
    *,
    cutracer_so: Optional[str],
    output_dir: Optional[str],
) -> ReplayConfig:
    """The replay of the triggering config with the targeting, kernel timeout,
    hang rule and launch handling the search ran it under (the in-process form
    of the `Reduce:` line `cutracer stress` prints)."""
    target = _triggering_target(triggering)
    mode_delay = get_mode(config.strategy).delay_mode
    fields: Dict[str, Any] = (
        {
            "delay_mode": None if mode_delay == "random" else mode_delay,
            "delay_patterns": _delay_patterns(config),
        }
        if target is None
        else _target_fields(config, target)
    )
    return ReplayConfig(
        oracle_argv=list(config.oracle_argv),
        cutracer_so=cutracer_so,
        kernel_filters=config.kernel_filters,
        exact_kernel_name=config.exact_kernel_name,
        exact_kernel_checksum=config.exact_kernel_checksum,
        delay_warpgroup_id=triggering.warpgroup_id if target is None else None,
        **fields,
        kernel_timeout_s=config.kernel_timeout_s,
        hang_is_bug=triggering.symptom == SYMPTOM_HANG,
        exit_quarantine_ns=config.exit_quarantine_ns,
        output_dir=output_dir,
        timeout=config.timeout,
        cwd=config.cwd,
        base_env=config.base_env,
        # A search without an explicit list counts any non-zero exit as "not
        # reproduced"; a replay needs the list, and 1 is the oracle default.
        not_interesting_exit_codes=tuple(config.not_interesting_exit_codes or (1,)),
        # The search's runs record kernel events, and a replay must do the
        # same host work per launch (see `_records_launches`).
        kernel_events="nostack" if _launch_options(config) else None,
        **_launch_options(config),
    )


_Evaluate = Callable[[str, ReplayConfig], ReplayResult]


def _vote(
    evaluate: _Evaluate, path: str, replay: ReplayConfig, runs: int
) -> Tuple[int, int, Optional[str]]:
    """Replay `path` until a majority of `runs` agrees: (reproductions,
    non-reproductions, why the vote stopped without a majority)."""
    needed = runs // 2 + 1
    yes = no = 0
    while yes < needed and no < needed and yes + no < runs:
        try:
            outcome = evaluate(path, replay)
        except _BudgetExhausted:
            return yes, no, "the time budget ran out"
        if outcome.outcome == ReplayOutcome.INTERESTING:
            yes += 1
        elif outcome.outcome == ReplayOutcome.NOT_INTERESTING:
            no += 1
        else:
            return yes, no, f"a replay failed: {outcome.error or outcome.outcome.value}"
    if yes < needed and no < needed:
        return yes, no, "the replays split without a majority"
    return yes, no, None


def _stopped(stopped: str, yes: int, no: int) -> str:
    if yes + no == 0:
        return f"{stopped} before any replay"
    return f"{stopped} after {yes} of {yes + no} replays reproduced"


def _excluded_warps(
    triggering: TriggeringConfig, target: Optional[DelayTarget]
) -> Tuple[Optional[str], int, str]:
    """(label, CTA-local warp mask) of the warps the triggering config
    delayed, or (None, 0, why the target check does not apply)."""
    if target is not None and target.barrier is not None:
        return None, 0, "a named-barrier target delays every warp at its barrier"
    if target is not None:
        label, mask = target.label(), int(target.warp_mask or "0", 16)
    elif triggering.target is not None:
        return None, 0, "the triggering config names a target that cannot be read"
    elif triggering.warpgroup_id is not None:
        label = f"warpgroup {triggering.warpgroup_id}"
        mask = (0xF << (4 * triggering.warpgroup_id)) & _ALL_WARPS
    else:
        return None, 0, "the search delayed every warp"
    if not mask:
        return None, 0, f"{label} names no warp a 32-bit warp mask covers"
    if not _ALL_WARPS & ~mask:
        return None, 0, f"{label} leaves no other warp to delay"
    return label, mask, ""


def _with_warp_mask(args: Sequence[str], mask: str) -> List[str]:
    """`cutracer trace` options `args` with their warp targeting replaced by
    `mask`."""
    kept: List[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg in ("--delay-warp-mask", "--delay-warpgroup-id"):
            skip = True
        elif not arg.startswith(("--delay-warp-mask=", "--delay-warpgroup-id=")):
            kept.append(arg)
    return [*kept, "--delay-warp-mask", mask]


def weak_reason(result: StressResult) -> str:
    """Why a check's reproduction may be the bug's own rate; empty when the
    search's control runs showed that it stays away without a delay."""
    if result.attribution == ATTRIBUTION_ATTRIBUTED:
        return ""
    if result.attribution == ATTRIBUTION_NATURAL:
        return "the search's control runs reproduced without any delay"
    return "no control run of the search showed that the bug stays away without a delay"


def _with_weak(reason: str, weak: str) -> str:
    """`reason` of a reproduction, with why it may be the bug's own rate."""
    return f"{reason}; weak: {weak}" if weak else reason


def check_target(
    evaluate: _Evaluate,
    replay: ReplayConfig,
    triggering: TriggeringConfig,
    minimal: str,
    *,
    runs: int,
    weak: str,
) -> ReplayCheck:
    """Replay the minimal config with the target's warps left undelayed and
    every other warp of the CTA delayed (warps a launch does not have are
    never delayed, so with no other warp nobody is): a reproduction says the
    target's own delay is not what the finding needs. A non-reproduction
    changes two things at once, so alone it does not show that the target's
    delay is needed (read it with `check_zero_delay`)."""
    label, mask, why = _excluded_warps(triggering, _triggering_target(triggering))
    if label is None:
        return ReplayCheck(
            TARGET_NOT_APPLICABLE, f"not checked: {why}", weak=bool(weak)
        )
    rest = hex(_ALL_WARPS & ~mask)
    excluded = dataclasses.replace(
        replay, delay_warp_mask=rest, delay_warpgroup_id=None
    )
    yes, no, stopped = _vote(evaluate, minimal, excluded, runs)
    what = f"excluding {label} from the delay (delaying warp mask {rest} instead)"
    if stopped is not None:
        status = CHECK_INCONCLUSIVE
        reason = f"{what}: {_stopped(stopped, yes, no)}"
    elif yes > no:
        status = TARGET_NOT_NEEDED
        hang = ""
        if triggering.symptom == SYMPTOM_HANG:
            hang = (
                "; for a hang, delaying the other warps can also run the kernel "
                "past its timeout by itself"
            )
        reason = _with_weak(
            f"{what} reproduced in {yes} of {yes + no} replays: the finding does "
            f"not need {label}'s delay{hang}",
            weak,
        )
    else:
        status = TARGET_NEEDED
        reason = (
            f"{what} did not reproduce in {no} of {yes + no} replays: {label}'s "
            "delay may be needed, or delaying the other warps may close the window"
        )
    return ReplayCheck(
        status,
        reason,
        replays=yes + no,
        reproductions=yes,
        weak=bool(weak),
        excluded=label,
        delay_warp_mask=rest,
        config_path=minimal,
        replay_args=_with_warp_mask(triggering.replay_args, rest),
    )


def _zero_delay_copy(minimal: str, path: str) -> None:
    """Write `minimal` with every point's delay set to 0 ns: CUTracer still
    instruments the enabled points but stalls no thread. The top-level
    `delay_ns` stays, since a replay needs a positive one."""
    with open(minimal) as source:
        config = json.load(source)
    for kernel in config.get("kernels", {}).values():
        for point in kernel.get("instrumentation_points", {}).values():
            point["delay"] = 0
    with open(path, "w") as sink:
        json.dump(config, sink, indent=2)


def check_zero_delay(
    evaluate: _Evaluate,
    replay: ReplayConfig,
    triggering: TriggeringConfig,
    minimal: str,
    *,
    runs: int,
    weak: str,
) -> ReplayCheck:
    """Replay the minimal config with every delay at 0 ns: a reproduction
    says the instrumentation's own overhead at its points is enough, so no
    warp's delay is what the finding needs."""
    path = os.path.join(os.path.dirname(minimal), "minimal_config_zero_delay.json")
    try:
        _zero_delay_copy(minimal, path)
    except (OSError, ValueError, AttributeError) as exc:
        return ReplayCheck(
            CHECK_INCONCLUSIVE, f"cannot write {path}: {exc}", weak=bool(weak)
        )
    # A random delay is drawn from [minimum, point delay]: a minimum inherited
    # from the environment would underflow against 0.
    unset = ["CUTRACER_DELAY_MIN_NS"]
    env = os.environ if replay.base_env is None else replay.base_env
    zero = dataclasses.replace(
        replay, base_env={k: v for k, v in env.items() if k not in unset}
    )
    yes, no, stopped = _vote(evaluate, path, zero, runs)
    what = (
        "with every delay of the minimal config at 0 ns (its points still instrumented)"
    )
    if stopped is not None:
        status = CHECK_INCONCLUSIVE
        reason = f"{what}: {_stopped(stopped, yes, no)}"
    elif yes > no:
        status = ZERO_DELAY_REPRODUCED
        reason = _with_weak(
            f"{what}, {yes} of {yes + no} replays reproduced: the instrumentation "
            "alone is enough, so no warp's delay is what the finding needs",
            weak,
        )
    else:
        status = ZERO_DELAY_NOT_REPRODUCED
        reason = (
            f"{what}, {no} of {yes + no} replays did not reproduce: the finding "
            "needs a delay"
        )
    return ReplayCheck(
        status,
        reason,
        replays=yes + no,
        reproductions=yes,
        weak=bool(weak),
        config_path=path,
        replay_args=list(triggering.replay_args),
        unset_env=unset,
    )


def confirm_finding(  # noqa: C901
    config: StressConfig,
    result: StressResult,
    *,
    cutracer_so: Optional[str],
    runner: Optional[RunTarget] = None,
    deadline: Optional[float] = None,
    confidence_runs: int = 3,
) -> Confirmation:
    """Reduce the triggering config of `result` (the search `config` ran),
    replay the minimal config once more and, when it reproduced, check what
    its delay needs. Writes `minimal_config.json` and the replays' scratch
    output under `config.output_dir`. No replay starts with less than
    `MIN_RUN_BUDGET_S` left before `deadline`."""
    triggering = result.triggering
    if triggering is None:
        raise ValueError("there is no triggering config to confirm")
    scratch = os.path.join(config.output_dir, "confirm")
    os.makedirs(scratch, exist_ok=True)
    replay = replay_config(
        config, triggering, cutracer_so=cutracer_so, output_dir=scratch
    )
    replays = 0

    def evaluate(path: str, candidate: ReplayConfig) -> ReplayResult:
        nonlocal replays
        remaining = _remaining_s(deadline)
        if remaining is not None and remaining < MIN_RUN_BUDGET_S:
            raise _BudgetExhausted
        replays += 1
        bounded = dataclasses.replace(
            candidate, timeout=_run_timeout(config, remaining)
        )
        return run_replay(path, bounded, runner=runner)

    minimal_path = os.path.join(config.output_dir, "minimal_config.json")
    try:
        reduced = reduce_bisect(
            ReduceConfig(
                config_path=triggering.config_path,
                output_path=minimal_path,
                replay=replay,
                replay_evaluator=evaluate,
            ),
            confidence_runs=confidence_runs,
        )
    except ConfigDoesNotTriggerError:
        return Confirmation(
            NOT_REPRODUCED,
            "the triggering config did not reproduce on replay",
            replays=replays,
        )
    except ReplayExecutionError as exc:
        return Confirmation(UNCONFIRMED, f"a replay failed: {exc}", replays=replays)
    except _BudgetExhausted:
        return Confirmation(
            UNCONFIRMED,
            "the time budget ran out while confirming",
            replays=replays,
        )
    minimal = reduced.minimal_config_path
    if not reduced.success or minimal is None:
        return Confirmation(
            NOT_REPRODUCED, "reduction kept no delay point", replays=replays
        )
    points = [
        {"kernel_name": p.kernel_name, "pc": str(p.pc_offset), "sass": p.sass}
        for p in reduced.essential_points
    ]

    def reduced_to(status: str, reason: str, **checks: Any) -> Confirmation:
        return Confirmation(
            status,
            reason,
            replays=replays,
            minimal_config_path=minimal,
            essential_points=points,
            **checks,
        )

    try:
        final = evaluate(minimal, replay)
    except _BudgetExhausted:
        return reduced_to(
            UNCONFIRMED,
            "the time budget ran out before the minimal config's confirmation replay",
        )
    if final.outcome == ReplayOutcome.NOT_INTERESTING:
        return reduced_to(
            NOT_REPRODUCED,
            "the minimal config did not reproduce on its confirmation replay",
        )
    if final.outcome != ReplayOutcome.INTERESTING:
        return reduced_to(
            UNCONFIRMED,
            f"the confirmation replay failed: {final.error or final.outcome.value}",
        )
    weak = weak_reason(result)
    target_check = check_target(
        evaluate, replay, triggering, minimal, runs=confidence_runs, weak=weak
    )
    zero_delay_check = check_zero_delay(
        evaluate, replay, triggering, minimal, runs=confidence_runs, weak=weak
    )
    return reduced_to(
        CONFIRMED,
        f"reduced to {len(points)} delay point(s); the minimal config reproduced again",
        target_check=target_check,
        zero_delay_check=zero_delay_check,
    )


def confirmed_verdict(
    verdict: Verdict, confirmation: Optional[Confirmation]
) -> Verdict:
    """A finding stays one only if its confirmation reproduced: a config that
    stops reproducing on replay is suspect, and a confirmation that could not
    finish leaves the search incomplete."""
    if confirmation is None or verdict.status != VERDICT_FINDING:
        return verdict
    reasons = [*verdict.reasons, confirmation.reason]
    if confirmation.status == CONFIRMED:
        return Verdict(VERDICT_FINDING, reasons)
    if confirmation.status == NOT_REPRODUCED:
        return Verdict(VERDICT_SUSPECT, reasons)
    return Verdict(VERDICT_INCOMPLETE, reasons)
