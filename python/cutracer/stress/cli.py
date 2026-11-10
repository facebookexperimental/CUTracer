# Copyright (c) Meta Platforms, Inc. and affiliates.

"""``cutracer stress`` — delay-injection bug-discovery search.

Discovery counterpart to ``cutracer reduce``: repeatedly runs a correctness
oracle under delay injection to find (and save) a delay config that triggers a
data race or a hang, which ``cutracer reduce`` can then minimize.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import time
from collections.abc import Callable
from typing import Any

import click
from cutracer.runner import resolve_cutracer_so
from cutracer.stress.confirm import (
    confirm_finding,
    Confirmation,
    CONFIRMED,
    confirmed_verdict,
)
from cutracer.stress.modes import get_mode, get_modes
from cutracer.stress.plan import MODE_RAN, PlanResult, run_stress_plan
from cutracer.stress.stress import (
    ATTRIBUTION_BASELINE_INCOMPLETE,
    ATTRIBUTION_NATURAL,
    ATTRIBUTION_NO_BASELINE,
    reduce_args,
    result_verdict,
    run_stress,
    save_report,
    STRATEGY_ISOLATE,
    STRATEGY_RANDOM,
    StressConfig,
    StressResult,
    SYMPTOM_HANG,
    Verdict,
    VERDICT_FINDING,
)

STRATEGY_ALL = "all"


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_int_ranges(text: str) -> list[int]:
    """Parse "0-3,8" into [0, 1, 2, 3, 8]."""
    values: list[int] = []
    for token in (part.strip() for part in text.split(",")):
        if not token:
            continue
        first, _, last = token.partition("-")
        start, end = int(first), int(last or first)
        if start > end:
            raise ValueError(f"warp range {token!r} runs backwards")
        values.extend(range(start, end + 1))
    return values


def _parse_bool(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected true or false, got {text!r}")


# `--mode-option MODE.KEY=VALUE`: the StressConfig fields one mode of a search
# may set, and how each value is parsed.
_MODE_OPTIONS: dict[str, Callable[[str], Any]] = {
    "delay_ladder_ns": _parse_int_list,
    "attempts_per_delay": int,
    "delay_patterns": str,
    "enable_prob": float,
    "warpgroup_ids": _parse_int_list,
    "isolate_ranks": _parse_int_list,
    "isolate_warps": _parse_int_ranges,
    "isolate_max_targets": int,
    "isolate_rotation": int,
    "kernel_timeout_s": int,
    "hang_is_bug": _parse_bool,
}


def _parse_modes(strategy: str) -> list[str]:
    """`all`, or comma-separated mode names, in the order they run."""
    known = list(get_modes())
    if strategy.strip() == STRATEGY_ALL:
        return known
    names = [name.strip() for name in strategy.split(",") if name.strip()]
    unknown = [name for name in names if name not in known]
    if not names or unknown:
        raise click.UsageError(
            f"--strategy {strategy!r}: expected {STRATEGY_ALL!r} or a "
            f"comma-separated list of {', '.join(known)}"
        )
    if len(set(names)) != len(names):
        raise click.UsageError(f"--strategy {strategy!r} names a mode twice")
    return names


def _parse_mode_options(
    values: tuple[str, ...], modes: list[str]
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for value in values:
        target, equals, text = value.partition("=")
        mode, dot, key = target.partition(".")
        if not equals or not dot:
            raise click.UsageError(f"--mode-option {value!r}: expected MODE.KEY=VALUE")
        if mode not in modes:
            raise click.UsageError(
                f"--mode-option {value!r}: {mode!r} is not a mode of this search "
                f"({', '.join(modes)})"
            )
        if key not in _MODE_OPTIONS:
            raise click.UsageError(
                f"--mode-option {value!r}: unknown key {key!r}; known: "
                f"{', '.join(_MODE_OPTIONS)}"
            )
        try:
            overrides.setdefault(mode, {})[key] = _MODE_OPTIONS[key](text)
        except ValueError as exc:
            raise click.UsageError(f"--mode-option {value!r}: {exc}") from exc
    return overrides


def _mode_config(
    name: str, shared: dict[str, Any], options: dict[str, Any]
) -> StressConfig:
    """One mode's search: its table defaults, then the shared and per-mode
    options."""
    mode = get_mode(name)
    fields: dict[str, Any] = {
        "strategy": mode.name,
        "delay_ladder_ns": list(mode.delay_ladder_ns),
        "attempts_per_delay": mode.attempts_per_delay,
        "kernel_timeout_s": mode.kernel_timeout_s,
        "hang_is_bug": mode.hang_is_bug,
        **shared,
        **options,
    }
    if not fields["delay_ladder_ns"]:
        raise click.UsageError(
            f"{name}: --delay-ladder-ns must contain at least one value"
        )
    try:
        return StressConfig(**fields)
    except ValueError as exc:
        raise click.UsageError(f"{name}: {exc}") from exc


@click.command(name="stress", context_settings={"ignore_unknown_options": True})
@click.option(
    "--strategy",
    default=STRATEGY_ALL,
    show_default=True,
    help="Delay modes to search with, in order: 'all', or a comma-separated list "
    "of 'random' (random per-thread delays at every candidate point, all warps "
    "at once or one warpgroup at a time) and 'isolate' (stall one (cluster rank, "
    "CTA-local warp) per attempt with a fixed delay before each mbarrier wait "
    "while every other warp runs at full speed; finds a warp the protocol does "
    "not wait for, such as a consumer that never releases its ring slot). With "
    "several modes, each writes its own report under <output-dir>/<mode>, modes "
    "with the same instrumentation share one control arm, and the search stops "
    "at the first mode that reproduces.",
)
@click.option(
    "--mode-option",
    "mode_options",
    multiple=True,
    metavar="MODE.KEY=VALUE",
    help="Set one option of one mode, e.g. 'random.delay_ladder_ns=10000,100000' "
    "or 'isolate.isolate_warps=0-7'. Keys: delay_ladder_ns, attempts_per_delay, "
    "delay_patterns, enable_prob, warpgroup_ids, isolate_ranks, isolate_warps, "
    "kernel_timeout_s, hang_is_bug. Repeatable.",
)
@click.option(
    "--confirm/--no-confirm",
    default=False,
    show_default=True,
    help="After a finding, reduce its triggering config to the delay points that "
    "matter (bisect, 3-run majority, with the search's targeting) and replay the "
    "minimal config once more. The verdict stays a finding only if both "
    "reproduce; the report gets a `confirmation` and minimal_config.json.",
)
@click.option(
    "--budget-s",
    type=click.IntRange(min=1),
    default=None,
    help="Wall-clock budget (s) for the whole search, control arms included. No "
    "run starts with less than 30 s left; the report then says the search is "
    "incomplete.",
)
@click.option(
    "--delay-ladder-ns",
    default=None,
    help="Comma-separated delay values (ns) to sweep, for a search with one mode "
    "(else use --mode-option). Default: 1000,5000,10000,50000,100000 (random), "
    "1000000 (isolate).",
)
@click.option(
    "--enable-prob",
    type=float,
    default=1.0,
    show_default=True,
    help="Per-point enable probability (random only; isolate enables every point).",
)
@click.option(
    "--warpgroup-ids",
    default="",
    help="Comma-separated warpgroup ids to target (random only; default: all "
    "warpgroups).",
)
@click.option(
    "--isolate-ranks",
    default="",
    help="Cluster ranks to isolate, e.g. '1' (isolate only; default: every rank "
    "of the instrumented launches, high to low).",
)
@click.option(
    "--isolate-warps",
    default="",
    help="CTA-local warps to isolate, e.g. '0-15' or '4,5' (isolate only; "
    "default: every warp of the instrumented launches).",
)
@click.option(
    "--delay-patterns",
    default=None,
    help="Comma-separated SASS substrings to delay at, for a search with one mode "
    "(else use --mode-option). Default: SYNCS.PHASECHK (mbarrier waits) for "
    "isolate, CUTracer's built-in patterns for random.",
)
@click.option(
    "--attempts-per-delay",
    type=click.IntRange(min=1),
    default=None,
    help="Attempts per delay (and per target), for a search with one mode (else "
    "use --mode-option). Default: 3 (random), 1 (isolate).",
)
@click.option(
    "--baseline-runs",
    type=int,
    default=3,
    show_default=True,
    help="Control runs before the search, with injection instrumented but every "
    "delay point disabled. If the target reproduces here it fails without any "
    "injected delay, so a search hit cannot be credited to the delay and the "
    "report says so. 0 skips the control and leaves attribution unproven.",
)
@click.option(
    "--stop-on-first/--no-stop-on-first",
    default=True,
    show_default=True,
    help="Stop at the first reproduced attempt whose delay config was dumped.",
)
@click.option(
    "--not-interesting-exit-codes",
    default="",
    help="Comma-separated oracle exit codes meaning 'ran, did not reproduce'. "
    "Other non-zero codes count as infra errors. Empty: any non-zero is a "
    "clean non-reproduction.",
)
@click.option("-k", "--kernel-filters", default=None, help="Kernel name filter.")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    default=".",
    show_default=True,
    help="Directory for per-attempt delay configs, the report, and stress.log "
    "(with several modes, one subdirectory per mode).",
)
@click.option(
    "-m",
    "--dump",
    "dump_path",
    type=click.Path(),
    default=None,
    help="Where to copy the triggering config (default: "
    "<output-dir>/triggering_config.json).",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(),
    default=None,
    help="Where to write the JSON report (default: <output-dir>/stress_report.json).",
)
@click.option(
    "--cutracer-so",
    default=None,
    help="Path to cutracer.so (default: bundled / auto-discovered).",
)
@click.option(
    "--timeout",
    type=int,
    default=1800,
    show_default=True,
    help="Per-attempt timeout (s).",
)
@click.option(
    "--kernel-timeout-s",
    type=click.IntRange(min=0),
    default=None,
    help="CUTracer's per-kernel timeout in every run (0 = off). With a timeout, "
    "the 15 s no-data watchdog is turned off, so a slow kernel is not mistaken "
    "for a hung one. Default: 0 (random), 60 (isolate).",
)
@click.option(
    "--hang-is-bug/--no-hang-is-bug",
    default=None,
    help="Count a run whose kernel CUTracer's watchdog ended (a hang) as a "
    "reproduction, whatever the oracle's exit code. Pair with --kernel-timeout-s. "
    "Default: off (random), on (isolate).",
)
@click.argument("oracle", nargs=-1, type=click.UNPROCESSED, required=True)
def stress_command(  # noqa: C901
    strategy: str,
    mode_options: tuple[str, ...],
    confirm: bool,
    budget_s: int | None,
    delay_ladder_ns: str | None,
    enable_prob: float,
    warpgroup_ids: str,
    isolate_ranks: str,
    isolate_warps: str,
    delay_patterns: str | None,
    attempts_per_delay: int | None,
    stop_on_first: bool,
    not_interesting_exit_codes: str,
    baseline_runs: int,
    kernel_filters: str | None,
    output_dir: str,
    dump_path: str | None,
    report_path: str | None,
    cutracer_so: str | None,
    timeout: int,
    kernel_timeout_s: int | None,
    hang_is_bug: bool | None,
    oracle: tuple[str, ...],
) -> None:
    """Search for a delay config that triggers a bug.

    Runs the ORACLE command repeatedly under delay injection. The oracle must
    exit 0 when the bug reproduced and non-zero otherwise (same convention as
    ``cutracer reduce -t``); with --hang-is-bug a hang that CUTracer's watchdog
    ends also counts. On the first reproduction whose delay config is captured,
    that config is saved for deterministic replay / ``cutracer reduce``. The
    report's `verdict` says whether the search found a bug the injected delay
    caused (finding), a failure it cannot credit to the delay (suspect), ran
    every planned attempt without one (clean), or neither (incomplete).

    \b
    Examples:
      cutracer stress -k my_kernel -o /tmp/stress -- ./oracle.sh
      cutracer stress --strategy isolate -k my_kernel -o /tmp/iso -- ./oracle.sh
    """
    if oracle and oracle[0].startswith("-"):
        raise click.UsageError(
            f"First token of the oracle command looks like a flag: {oracle[0]!r}. "
            "Put stress options before `--` and the oracle command after it."
        )
    modes = _parse_modes(strategy)
    single = modes[0] if len(modes) == 1 else None
    if warpgroup_ids and STRATEGY_RANDOM not in modes:
        raise click.UsageError("--warpgroup-ids applies to --strategy random only")
    if (isolate_ranks or isolate_warps) and STRATEGY_ISOLATE not in modes:
        raise click.UsageError(
            "--isolate-ranks and --isolate-warps need --strategy isolate"
        )
    per_mode_flags = {
        "--delay-ladder-ns": delay_ladder_ns,
        "--attempts-per-delay": attempts_per_delay,
        "--delay-patterns": delay_patterns,
    }
    if single is None:
        for flag, value in per_mode_flags.items():
            if value is not None:
                raise click.UsageError(
                    f"{flag} applies to one mode; with several modes use "
                    f"--mode-option MODE.{flag[2:].replace('-', '_')}=VALUE"
                )
    try:
        flags: dict[str, dict[str, Any]] = {name: {} for name in modes}
        if delay_ladder_ns is not None:
            flags[modes[0]]["delay_ladder_ns"] = _parse_int_list(delay_ladder_ns)
        if attempts_per_delay is not None:
            flags[modes[0]]["attempts_per_delay"] = attempts_per_delay
        if delay_patterns is not None:
            flags[modes[0]]["delay_patterns"] = delay_patterns
        if STRATEGY_RANDOM in modes:
            flags[STRATEGY_RANDOM]["warpgroup_ids"] = _parse_int_list(warpgroup_ids)
        if isolate_ranks:
            flags[STRATEGY_ISOLATE]["isolate_ranks"] = _parse_int_list(isolate_ranks)
        if isolate_warps:
            flags[STRATEGY_ISOLATE]["isolate_warps"] = _parse_int_ranges(isolate_warps)
        not_interesting = _parse_int_list(not_interesting_exit_codes)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    overrides = _parse_mode_options(mode_options, modes)
    shared: dict[str, Any] = {
        "oracle_argv": list(oracle),
        "enable_prob": enable_prob,
        "stop_on_first": stop_on_first,
        "not_interesting_exit_codes": not_interesting,
        "baseline_runs": baseline_runs,
        "kernel_filters": kernel_filters,
        "timeout": timeout,
    }
    if kernel_timeout_s is not None:
        shared["kernel_timeout_s"] = kernel_timeout_s
    if hang_is_bug is not None:
        shared["hang_is_bug"] = hang_is_bug
    configs = [
        _mode_config(
            name,
            {
                **shared,
                "output_dir": (
                    output_dir if single is not None else os.path.join(output_dir, name)
                ),
            },
            {**flags[name], **overrides.get(name, {})},
        )
        for name in modes
    ]
    sanitizer_so = resolve_cutracer_so(cutracer_so)
    if single is not None:
        _run_single(
            configs[0],
            sanitizer_so=sanitizer_so,
            confirm=confirm,
            budget_s=budget_s,
            output_dir=output_dir,
            dump_path=dump_path,
            report_path=report_path,
            oracle=oracle,
        )
    else:
        _run_plan(
            configs,
            sanitizer_so=sanitizer_so,
            confirm=confirm,
            budget_s=budget_s,
            stop_on_first=stop_on_first,
            output_dir=output_dir,
            dump_path=dump_path,
            report_path=report_path,
            oracle=oracle,
        )
    # Exit 0 whenever the search itself ran; callers read the report/config to
    # learn whether the bug reproduced (a non-zero exit means the search failed).
    sys.exit(0)


def _plan_line(config: StressConfig) -> str:
    if get_mode(config.strategy).isolates_targets:
        placements = "targets"
    else:
        placements = f"{len(config.warpgroup_ids) or 1} warpgroup(s)"
    return (
        f"{len(config.delay_ladder_ns)} delays x {placements} x "
        f"{config.attempts_per_delay} attempts"
    )


def _confirm(
    config: StressConfig,
    result: StressResult,
    verdict: Verdict,
    *,
    sanitizer_so: str,
    deadline: float | None,
) -> tuple[Verdict, Confirmation | None]:
    if verdict.status != VERDICT_FINDING:
        return verdict, None
    click.echo("Confirming: reduce, then replay the minimal config ...", err=True)
    confirmation = confirm_finding(
        config, result, cutracer_so=sanitizer_so, deadline=deadline
    )
    if confirmation.status == CONFIRMED:
        points = "; ".join(
            f"{p['sass']} (pc {p['pc']}, {p['kernel_name']})"
            for p in confirmation.essential_points
        )
        click.echo(
            f"Confirmed: {confirmation.reason}: {points} -> "
            f"{confirmation.minimal_config_path}"
        )
    else:
        click.echo(f"Not confirmed: {confirmation.reason}")
    return confirmed_verdict(verdict, confirmation), confirmation


def _report_extra(
    verdict: Verdict, confirmation: Confirmation | None, confirm: bool
) -> dict[str, Any] | None:
    if not confirm:
        return None
    return {
        "verdict": verdict.to_dict(),
        "confirmation": None if confirmation is None else confirmation.to_dict(),
    }


def _run_single(
    config: StressConfig,
    *,
    sanitizer_so: str,
    confirm: bool,
    budget_s: int | None,
    output_dir: str,
    dump_path: str | None,
    report_path: str | None,
    oracle: tuple[str, ...],
) -> None:
    click.echo(
        f"Running stress ({config.strategy}): {config.baseline_runs} baseline + "
        f"{_plan_line(config)} -- {' '.join(config.oracle_argv)}",
        err=True,
    )
    deadline = None if budget_s is None else time.monotonic() + budget_s
    try:
        result = run_stress(config, cutracer_so=sanitizer_so, deadline=deadline)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if get_mode(config.strategy).isolates_targets:
        click.echo(f"Isolated {len(result.targets)} target(s).", err=True)
    verdict, confirmation = result_verdict(result), None
    if confirm:
        verdict, confirmation = _confirm(
            config, result, verdict, sanitizer_so=sanitizer_so, deadline=deadline
        )

    report_out = report_path or f"{output_dir}/stress_report.json"
    save_report(result, report_out, _report_extra(verdict, confirmation, confirm))
    click.echo(f"Report: {report_out}", err=True)

    if result.triggering is not None and os.path.exists(result.triggering.config_path):
        dump_out = dump_path or f"{output_dir}/triggering_config.json"
        shutil.copyfile(result.triggering.config_path, dump_out)
        _echo_reproduction(config, result, dump_out, oracle)
    elif result.triggering is not None:
        click.echo(
            f"Reproduced (delay_ns={result.triggering.delay_ns}) but its delay "
            f"config {result.triggering.config_path} is missing -- nothing saved.",
            err=True,
        )
    else:
        click.echo(
            f"Not reproduced across {result.completed_trials} trial(s) "
            f"({result.infra_errors} infra error(s)). No triggering config saved."
        )
    _echo_notes(config, result)
    _echo_verdict(verdict.status, verdict.reasons)


def _echo_notes(config: StressConfig, result: StressResult) -> None:
    if result.hangs and not config.hang_is_bug:
        click.echo(
            f"NOTE: CUTracer's watchdog ended a hung kernel in {result.hangs} "
            f"attempt(s); they were classified by the oracle's exit code. Pass "
            f"--hang-is-bug to count hangs as reproductions.",
            err=True,
        )

    if result.attribution == ATTRIBUTION_NATURAL:
        click.echo(
            f"WARNING: the control arm reproduced {result.baseline_reproductions}/"
            f"{result.baseline_completed} times with injection DISABLED. This "
            f"target fails on its own, so nothing found here can be credited to "
            f"the injected delay. Any triggering config is a coincidence until "
            f"shown otherwise -- reduce it, or compare its failure magnitude "
            f"against the baseline.",
            err=True,
        )
    elif result.attribution == ATTRIBUTION_BASELINE_INCOMPLETE:
        click.echo(
            f"WARNING: only {result.baseline_completed} of "
            f"{result.baseline_requested} requested control run(s) produced a "
            f"usable sample ({result.baseline_timed_out} timed out, "
            f"{result.baseline_infra_errors} infra error(s)). The control arm was "
            f"asked for and did not deliver, so the target's natural failure rate "
            f"is still unmeasured and nothing found here is corroborated. Fix the "
            f"control runs (--timeout, --not-interesting-exit-codes) and re-run.",
            err=True,
        )
    elif result.attribution == ATTRIBUTION_NO_BASELINE:
        click.echo(
            "NOTE: no control runs were requested (--baseline-runs 0), so a "
            "reproduction here cannot be distinguished from the target's own "
            "failure rate.",
            err=True,
        )

    if result.unattributed_reproductions:
        click.echo(
            f"WARNING: {result.unattributed_reproductions} attempt(s) reported the "
            f"bug while injecting NO delay (the config enabled 0 points). This "
            f"target fails on its own, so any reproduction here may be its natural "
            f"rate rather than an effect of the injected delay. Establish the "
            f"baseline before trusting a triggering config.",
            err=True,
        )


def _echo_verdict(status: str, reasons: list[str]) -> None:
    click.echo(f"Verdict: {status}" + (f" ({'; '.join(reasons)})" if reasons else ""))


def _save_plan_report(
    plan: PlanResult, path: str, extra: dict[str, Any] | None = None
) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as output:
        json.dump({**plan.to_dict(), **(extra or {})}, output, indent=2)
    os.replace(tmp, path)


def _run_plan(
    configs: list[StressConfig],
    *,
    sanitizer_so: str,
    confirm: bool,
    budget_s: int | None,
    stop_on_first: bool,
    output_dir: str,
    dump_path: str | None,
    report_path: str | None,
    oracle: tuple[str, ...],
) -> None:
    click.echo(
        f"Running stress ({','.join(c.strategy for c in configs)}): "
        f"{configs[0].baseline_runs} baseline per instrumentation, then "
        + "; ".join(f"{c.strategy}: {_plan_line(c)}" for c in configs)
        + f" -- {' '.join(configs[0].oracle_argv)}",
        err=True,
    )
    deadline = None if budget_s is None else time.monotonic() + budget_s
    try:
        plan = run_stress_plan(
            configs,
            output_dir=output_dir,
            budget_s=budget_s,
            stop_on_first=stop_on_first,
            cutracer_so=sanitizer_so,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    verdict, confirmation = plan.verdict, None
    trigger = plan.triggering_run
    if confirm and trigger is not None and trigger.result is not None:
        verdict, confirmation = _confirm(
            trigger.config,
            trigger.result,
            verdict,
            sanitizer_so=sanitizer_so,
            deadline=deadline,
        )
    report_out = report_path or f"{output_dir}/stress_report.json"
    _save_plan_report(plan, report_out, _report_extra(verdict, confirmation, confirm))
    for run in plan.runs:
        if run.status != MODE_RAN or run.result is None:
            click.echo(f"{run.mode}: {run.status} ({run.reason})", err=True)
            continue
        mode_verdict = result_verdict(run.result)
        click.echo(
            f"{run.mode}: {mode_verdict.status} after "
            f"{run.result.completed_trials} trial(s) "
            f"({run.result.infra_errors} infra error(s)) -> {run.report_path}",
            err=True,
        )
    click.echo(f"Report: {report_out}", err=True)
    if trigger is not None and trigger.result is not None:
        result = trigger.result
        triggering = result.triggering
        assert triggering is not None
        if os.path.exists(triggering.config_path):
            dump_out = dump_path or f"{output_dir}/triggering_config.json"
            shutil.copyfile(triggering.config_path, dump_out)
            _echo_reproduction(trigger.config, result, dump_out, oracle)
        _echo_notes(trigger.config, result)
    _echo_verdict(verdict.status, verdict.reasons)


def _echo_reproduction(
    config: StressConfig,
    result: StressResult,
    dump_out: str,
    oracle: tuple[str, ...],
) -> None:
    triggering = result.triggering
    assert triggering is not None
    symptom = " (hang)" if triggering.symptom == SYMPTOM_HANG else ""
    target = triggering.target
    if target is None:
        placement = f"warpgroup={triggering.warpgroup_id}"
    elif target.get("cluster_rank") is None:
        placement = f"target=warp {target.get('warp')}"
    else:
        placement = (
            f"target=cluster rank {target.get('cluster_rank')}, "
            f"warp {target.get('warp')}"
        )
    click.echo(
        f"Reproduced{symptom}: delay_ns={triggering.delay_ns} {placement} "
        f"rate={result.reproduction_rate:.2f} -> {dump_out}"
    )
    if triggering.replay_args:
        replay = ["cutracer", "trace", *triggering.replay_args]
        replay += ["--delay-load-path", dump_out, "--", *oracle]
        click.echo(f"Replay: {shlex.join(replay)}")
        reduce = ["cutracer", "reduce", "-c", dump_out, "-t", shlex.join(oracle)]
        reduce += reduce_args(config, triggering)
        click.echo(f"Reduce: {shlex.join(reduce)}")
