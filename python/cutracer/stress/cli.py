# Copyright (c) Meta Platforms, Inc. and affiliates.

"""``cutracer stress`` — delay-injection bug-discovery search.

Discovery counterpart to ``cutracer reduce``: repeatedly runs a correctness
oracle under delay injection to find (and save) a delay config that triggers a
data race or a hang, which ``cutracer reduce`` can then minimize.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys

import click
from cutracer.runner import resolve_cutracer_so
from cutracer.stress.modes import get_mode
from cutracer.stress.stress import (
    ATTRIBUTION_BASELINE_INCOMPLETE,
    ATTRIBUTION_NATURAL,
    ATTRIBUTION_NO_BASELINE,
    reduce_args,
    run_stress,
    save_report,
    STRATEGY_ISOLATE,
    STRATEGY_RANDOM,
    StressConfig,
    StressResult,
    SYMPTOM_HANG,
)


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


@click.command(name="stress", context_settings={"ignore_unknown_options": True})
@click.option(
    "--strategy",
    type=click.Choice([STRATEGY_RANDOM, STRATEGY_ISOLATE]),
    default=STRATEGY_RANDOM,
    show_default=True,
    help="'random': random per-thread delays at every candidate point, all warps "
    "at once or one warpgroup at a time. 'isolate': stall one (cluster rank, "
    "CTA-local warp) per attempt with a fixed delay before each mbarrier wait "
    "while every other warp runs at full speed; finds a warp the protocol does "
    "not wait for, such as a consumer that never releases its ring slot.",
)
@click.option(
    "--delay-ladder-ns",
    default=None,
    help="Comma-separated delay values (ns) to sweep. Default: "
    "1000,5000,10000,50000,100000 (random), 1000000 (isolate).",
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
    help="Comma-separated SASS substrings to delay at. Default: SYNCS.PHASECHK "
    "(mbarrier waits) for isolate, CUTracer's built-in patterns for random.",
)
@click.option(
    "--attempts-per-delay",
    type=click.IntRange(min=1),
    default=None,
    help="Attempts per delay (and per target). Default: 3 (random), 1 (isolate).",
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
    help="Directory for per-attempt delay configs, the report, and stress.log.",
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
    that config is saved for deterministic replay / ``cutracer reduce``.

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

    defaults = get_mode(strategy)
    isolate = defaults.isolates_targets
    if isolate and warpgroup_ids:
        raise click.UsageError("--warpgroup-ids applies to --strategy random only")
    if not isolate and (isolate_ranks or isolate_warps):
        raise click.UsageError(
            "--isolate-ranks and --isolate-warps need --strategy isolate"
        )
    parsed_delays = (
        list(defaults.delay_ladder_ns)
        if delay_ladder_ns is None
        else _parse_int_list(delay_ladder_ns)
    )
    if not parsed_delays:
        raise click.UsageError("--delay-ladder-ns must contain at least one value")

    sanitizer_so = resolve_cutracer_so(cutracer_so)
    try:
        config = StressConfig(
            oracle_argv=list(oracle),
            delay_ladder_ns=parsed_delays,
            enable_prob=enable_prob,
            warpgroup_ids=_parse_int_list(warpgroup_ids),
            attempts_per_delay=(
                defaults.attempts_per_delay
                if attempts_per_delay is None
                else attempts_per_delay
            ),
            stop_on_first=stop_on_first,
            not_interesting_exit_codes=_parse_int_list(not_interesting_exit_codes),
            baseline_runs=baseline_runs,
            kernel_filters=kernel_filters,
            output_dir=output_dir,
            timeout=timeout,
            kernel_timeout_s=(
                defaults.kernel_timeout_s
                if kernel_timeout_s is None
                else kernel_timeout_s
            ),
            hang_is_bug=defaults.hang_is_bug if hang_is_bug is None else hang_is_bug,
            strategy=strategy,
            isolate_ranks=_parse_int_list(isolate_ranks) if isolate_ranks else None,
            isolate_warps=_parse_int_ranges(isolate_warps) if isolate_warps else None,
            delay_patterns=delay_patterns,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if isolate:
        placements = "targets"
    else:
        placements = f"{len(config.warpgroup_ids) or 1} warpgroup(s)"
    click.echo(
        f"Running stress ({strategy}): {config.baseline_runs} baseline + "
        f"{len(config.delay_ladder_ns)} delays x {placements} x "
        f"{config.attempts_per_delay} attempts -- {' '.join(config.oracle_argv)}",
        err=True,
    )

    try:
        result = run_stress(config, cutracer_so=sanitizer_so)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if isolate:
        click.echo(f"Isolated {len(result.targets)} target(s).", err=True)

    report_out = report_path or f"{output_dir}/stress_report.json"
    save_report(result, report_out)
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
    # Exit 0 whenever the search itself ran; callers read the report/config to
    # learn whether the bug reproduced (a non-zero exit means the search failed).
    sys.exit(0)


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
