# Copyright (c) Meta Platforms, Inc. and affiliates.

"""``cutracer stress`` — random-delay bug-discovery search.

Discovery counterpart to ``cutracer reduce``: repeatedly runs a correctness
oracle under random delay injection to find (and save) a delay config that
triggers a data race, which ``cutracer reduce`` can then minimize.
"""

from __future__ import annotations

import os
import shutil
import sys

import click
from cutracer.runner import resolve_cutracer_so
from cutracer.stress.stress import (
    ATTRIBUTION_BASELINE_INCOMPLETE,
    ATTRIBUTION_NATURAL,
    ATTRIBUTION_NO_BASELINE,
    run_stress,
    save_report,
    StressConfig,
)


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


@click.command(name="stress", context_settings={"ignore_unknown_options": True})
@click.option(
    "--delay-ladder-ns",
    default="1000,5000,10000,50000,100000",
    show_default=True,
    help="Comma-separated delay values (ns) to sweep.",
)
@click.option("--enable-prob", type=float, default=1.0, show_default=True)
@click.option(
    "--warpgroup-ids",
    default="",
    help="Comma-separated warpgroup ids to target (default: all warpgroups).",
)
@click.option("--attempts-per-delay", type=int, default=3, show_default=True)
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
@click.argument("oracle", nargs=-1, type=click.UNPROCESSED, required=True)
def stress_command(
    delay_ladder_ns: str,
    enable_prob: float,
    warpgroup_ids: str,
    attempts_per_delay: int,
    stop_on_first: bool,
    not_interesting_exit_codes: str,
    baseline_runs: int,
    kernel_filters: str | None,
    output_dir: str,
    dump_path: str | None,
    report_path: str | None,
    cutracer_so: str | None,
    timeout: int,
    oracle: tuple[str, ...],
) -> None:
    """Search for a random-delay config that triggers a bug.

    Runs the ORACLE command repeatedly under random delay injection. The oracle
    must exit 0 when the bug reproduced and non-zero otherwise (same convention
    as ``cutracer reduce -t``). On the first reproduction whose delay config is
    captured, that config is saved for deterministic replay / ``cutracer reduce``.

    \b
    Example:
      cutracer stress -k my_kernel -o /tmp/stress -- ./oracle.sh
    """
    if oracle and oracle[0].startswith("-"):
        raise click.UsageError(
            f"First token of the oracle command looks like a flag: {oracle[0]!r}. "
            "Put stress options before `--` and the oracle command after it."
        )

    parsed_delays = _parse_int_list(delay_ladder_ns)
    if not parsed_delays:
        raise click.UsageError("--delay-ladder-ns must contain at least one value")

    sanitizer_so = resolve_cutracer_so(cutracer_so)
    config = StressConfig(
        oracle_argv=list(oracle),
        delay_ladder_ns=parsed_delays,
        enable_prob=enable_prob,
        warpgroup_ids=_parse_int_list(warpgroup_ids),
        attempts_per_delay=attempts_per_delay,
        stop_on_first=stop_on_first,
        not_interesting_exit_codes=_parse_int_list(not_interesting_exit_codes),
        baseline_runs=baseline_runs,
        kernel_filters=kernel_filters,
        output_dir=output_dir,
        timeout=timeout,
    )
    click.echo(
        f"Running stress: {config.baseline_runs} baseline + "
        f"{len(config.delay_ladder_ns)} delays x "
        f"{len(config.warpgroup_ids) or 1} warpgroup(s) x "
        f"{config.attempts_per_delay} attempts -- {' '.join(config.oracle_argv)}",
        err=True,
    )

    result = run_stress(config, cutracer_so=sanitizer_so)

    report_out = report_path or f"{output_dir}/stress_report.json"
    save_report(result, report_out)
    click.echo(f"Report: {report_out}", err=True)

    if result.triggering is not None and os.path.exists(result.triggering.config_path):
        dump_out = dump_path or f"{output_dir}/triggering_config.json"
        shutil.copyfile(result.triggering.config_path, dump_out)
        click.echo(
            f"Reproduced: delay_ns={result.triggering.delay_ns} "
            f"warpgroup={result.triggering.warpgroup_id} "
            f"rate={result.reproduction_rate:.2f} -> {dump_out}"
        )
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
