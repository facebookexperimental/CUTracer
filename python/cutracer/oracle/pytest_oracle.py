#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""One pytest node as a ``cutracer stress`` / ``cutracer reduce`` oracle.

Exit 0 when the node fails its correctness check (an ``AssertionError`` raised
from one of the named assertion sites), 1 when it passes, and 2 for anything
else: a setup or collection error, a skip or xfail, another exception, a wrong
selection, a timeout, a signature that does not match an earlier reproduction,
or an injected run whose delay config never instrumented a launched kernel of
interest. A run that measured nothing must not read as a pass. Each run prints
its exit code and the reason on one stderr line.

The node runs in a child process of the project's interpreter (``--python``),
which inherits CUTracer's injection; the oracle itself never loads CUDA. The
child stays in the oracle's process group, so the runner that launched the
oracle can end the whole group. A multi-process test launched with
torchelastic sees a failed worker as ``ChildFailedError``; the oracle then
judges the first failed worker's own traceback, from the error file that
``@record`` on the worker's entry point writes.

Any command can be the oracle instead: ``cutracer oracle cmd`` maps its
exit codes to the same contract (``--fail-codes`` to 0, ``--pass-codes``
to 1, anything else to 2), optionally requires ``--fail-pattern`` in its
output, and pins the first reproduction's signature on the matched text.
Both modes share the coverage check, the dump/replay handling, the
evidence directory, the signature file and the stderr reason line.

With ``--numerics check --golden DIR``, the run is also compared tensor by
tensor with a golden that ``--numerics record`` wrote from plain runs: the
child imports ``cutracer_numerics.py`` (the capture), which records every
tensor the test compares. A large change counts as a reproduction even when
the test passes, and a run whose tensors cannot be lined up with the golden
measured nothing. ``--numerics record`` is not an oracle: it exits 0 for a
usable golden, 3 for a golden whose own plain runs already changed by a
large amount, and 2 when it wrote none.

This file needs only the standard library (and pytest in the child, for
pytest mode), so the project's interpreter can run it directly: ``cutracer
oracle export PATH`` writes it out, with the capture beside it, which saves
starting CUTracer for every run of a search. ``cutracer oracle pytest`` and
``cutracer oracle cmd`` run the same code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Union

REPRODUCED = 0
PASSED = 1
INCOMPLETE = 2
# `--numerics record` exits: a usable golden, or one whose plain runs changed
# by a large amount (INCOMPLETE when it wrote none).
GOLDEN_RECORDED = 0
GOLDEN_UNSTABLE = 3

NUMERICS_MODULE = "cutracer_numerics"
# The capture the process under test imports. `cutracer oracle` points this
# at its resource; an exported oracle finds it beside itself.
NUMERICS_CAPTURE = Path(__file__).resolve().with_name(f"{NUMERICS_MODULE}.py")
NUMERICS_ENV_PREFIX = "CUTRACER_NUMERICS_"
# Capture settings a golden records and every check of it replays.
_NUMERICS_SETTINGS = (
    "CUTRACER_NUMERICS_SEED",
    "CUTRACER_NUMERICS_PIN_HYPOTHESIS",
    "CUTRACER_NUMERICS_MAX_MB",
    "CUTRACER_NUMERICS_MAX_TOTAL_MB",
    "CUTRACER_NUMERICS_LARGE_REL_L2",
    "CUTRACER_NUMERICS_LARGE_ROW_REL_L2",
    "CUTRACER_NUMERICS_LARGE_ELEM_REL",
    "CUTRACER_NUMERICS_WRAP",
    "PYTHONHASHSEED",
)
MANIFEST = "manifest.json"
STABILITY = "stability.json"
_SEVERITY = {"same": 0, "small": 1, "unmeasured": 2, "large": 3}

KERNEL_EVENTS_PREFIX = "cutracer_kernel_events_"
# Disable plugins and project options that change how one node runs: an ini
# `addopts` can turn on xdist, and a plugin timeout would race the oracle's own.
DEFAULT_PYTEST_ARGS = ("-p", "no:timeout", "-o", "addopts=")


@dataclass
class NumericsOptions:
    """`--numerics`: compare the run's tensors with a golden."""

    mode: str  # "record" or "check"
    golden: Path
    runs: int = 3
    wrap: Sequence[str] = ()
    strict: bool = False
    capture: Optional[Path] = None


@dataclass
class OracleOptions:
    python: str
    rootdir: Path
    nodeid: str
    assertion_sites: Sequence[str] = ()
    kernels: Optional[str] = None
    output_dir: Optional[Path] = None
    expected_signature: Optional[str] = None
    signature_file: Optional[Path] = None
    timeout_s: Optional[float] = None
    plain: bool = False
    import_mode: str = "importlib"
    pytest_args: Sequence[str] = field(default_factory=lambda: DEFAULT_PYTEST_ARGS)
    # This file, which the child runs in its worker mode.
    script: Path = field(default_factory=lambda: Path(__file__).resolve())
    numerics: Optional[NumericsOptions] = None


@dataclass
class CmdOptions:
    command: Sequence[str]
    fail_codes: FrozenSet[int] = frozenset({1})
    pass_codes: FrozenSet[int] = frozenset({0})
    fail_pattern: Optional[str] = None
    kernels: Optional[str] = None
    output_dir: Optional[Path] = None
    signature_file: Optional[Path] = None
    timeout_s: Optional[float] = None
    plain: bool = False
    numerics: Optional[NumericsOptions] = None


def assertion_site(traceback: Any, rootdir: Path) -> Optional[List[Any]]:
    """The innermost frame under `rootdir`, outside installed packages:
    [path relative to rootdir, function, line, statement]."""
    root = rootdir.resolve()
    for entry in reversed(traceback):
        path = Path(str(entry.path)).resolve()
        if path.is_relative_to(root) and "site-packages" not in path.parts:
            statement = " ".join(str(entry.statement).split())
            return [
                path.relative_to(root).as_posix(),
                entry.name,
                entry.lineno + 1,
                statement,
            ]
    return None


_FRAME = re.compile(
    r'^  File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<name>\S+)\n'
    r"(?:    (?P<statement>[^\n]*))?",
    re.MULTILINE,
)
_CHAINED = re.compile(
    r"\n\n(?:During handling of the above exception|"
    r"The above exception was the direct cause)[^\n]*\n\n"
)


def text_failure(callstack: str, rootdir: Path) -> tuple[str, Optional[List[Any]]]:
    """The exception type and assertion site of a formatted traceback (the
    last one, if exceptions were chained)."""
    block = _CHAINED.split(callstack.strip())[-1]
    frames = list(_FRAME.finditer(block))
    # The exception line is the first unindented line after the last frame;
    # its message may go on for more lines (`torch.testing.assert_close`).
    tail = block[frames[-1].end() :] if frames else block
    header = next(
        (
            line
            for line in tail.splitlines()
            if line.strip()
            and not line[0].isspace()
            and not line.startswith("Traceback (most recent call last)")
        ),
        "",
    )
    exception = header.split(":", 1)[0].strip()
    root = rootdir.resolve()
    for frame in reversed(frames):
        path = Path(frame["path"]).resolve()
        if path.is_relative_to(root) and "site-packages" not in path.parts:
            statement = " ".join((frame["statement"] or "").split())
            site = [path.relative_to(root).as_posix(), frame["name"]]
            return exception, [*site, int(frame["line"]), statement]
    return exception, None


def child_failure(error: BaseException, rootdir: Path) -> Optional[tuple[str, Any]]:
    """For a torchelastic launch whose child failed (`ChildFailedError`), the
    first failed child's exception type and assertion site, from the
    traceback its `@record` error file holds; otherwise None."""
    first = getattr(error, "get_first_failure", None)
    if not callable(first):
        return None
    _, failure = first()
    message = getattr(failure, "message", None)
    callstack = (
        message.get("extraInfo", {}).get("py_callstack")
        if isinstance(message, dict)
        else None
    )
    if not isinstance(callstack, str):
        return "", None
    return text_failure(callstack, rootdir)


def signature(nodeid: str, site: Sequence[Any]) -> str:
    return hashlib.sha256(json.dumps([nodeid, *site]).encode()).hexdigest()


def _where(site: Any) -> str:
    if isinstance(site, list) and len(site) >= 3:
        return f"{site[0]}::{site[1]}:{site[2]}"
    return "an unknown site"


def _failure_reason(report: Dict[str, Any]) -> str:
    """Why a failed node is not a correctness failure."""
    site = report.get("assertion_site")
    if report.get("child_failure") and not report.get("exception"):
        return (
            "a worker process failed, and its error file has no traceback "
            "(decorate the worker's entry point with torchelastic's @record)"
        )
    exception = report.get("exception") or "an exception"
    where = f"at {_where(site)}" if site is not None else "outside the rootdir"
    return (
        f"the node failed with {exception} {where}, not an AssertionError at an "
        "assertion site"
    )


def classify(
    report: Dict[str, Any], nodeid: str, expected_signature: Optional[str] = None
) -> tuple[int, str]:
    """The oracle exit code for one worker report, and the reason for it."""
    selected = report.get("selected")
    if selected != [nodeid]:
        return INCOMPLETE, f"pytest selected {selected!r}, not exactly the node"
    if report.get("collection_errors"):
        return INCOMPLETE, "pytest could not collect the node"
    phases = report.get("phases", [])
    if [(p.get("nodeid"), p.get("when")) for p in phases] != [
        (nodeid, phase) for phase in ("setup", "call", "teardown")
    ]:
        return INCOMPLETE, "the node did not run its setup, call and teardown"
    for phase in (phases[0], phases[2]):
        if phase.get("outcome") != "passed":
            return INCOMPLETE, f"the node's {phase.get('when')} {phase.get('outcome')}"
    call = phases[1]
    outcome, pytest_exit = call.get("outcome"), report.get("pytest_exit")
    if call.get("xfail"):
        return INCOMPLETE, "the node is marked xfail"
    if outcome == "skipped":
        return INCOMPLETE, "the node was skipped"
    if outcome == "passed" and pytest_exit == 0:
        return PASSED, "the node passed"
    if outcome != "failed" or pytest_exit != 1:
        return INCOMPLETE, f"the node's call {outcome}, but pytest exited {pytest_exit}"
    if report.get("correctness_failure") is not True or not report.get("signature"):
        return INCOMPLETE, _failure_reason(report)
    where = _where(report.get("assertion_site"))
    if expected_signature is not None and report["signature"] != expected_signature:
        return (
            INCOMPLETE,
            f"the node failed its correctness check at {where}, which is not the "
            "pinned failure",
        )
    return REPRODUCED, f"the node failed its correctness check at {where}"


def cmd_signature(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def classify_cmd(
    returncode: int,
    output: str,
    fail_codes: FrozenSet[int],
    pass_codes: FrozenSet[int],
    fail_pattern: Optional[str] = None,
    expected_signature: Optional[str] = None,
) -> tuple[int, str, Optional[str]]:
    """The oracle exit code for one command run, the reason for it, and the
    failure's signature (None unless it reproduced)."""
    if returncode < 0:
        return INCOMPLETE, f"the command was killed by signal {-returncode}", None
    if returncode in fail_codes:
        if fail_pattern is not None:
            match = re.search(fail_pattern, output)
            if match is None:
                return (
                    INCOMPLETE,
                    f"the command exited {returncode}, but its output matched "
                    f"no {fail_pattern!r}",
                    None,
                )
            sig = cmd_signature(match.group(0))
            reason = (
                f"the command exited {returncode} with output matching {fail_pattern!r}"
            )
        else:
            sig = cmd_signature(f"exit:{returncode}")
            reason = f"the command exited {returncode}"
        if expected_signature is not None and sig != expected_signature:
            return INCOMPLETE, f"{reason}, which is not the pinned failure", sig
        return REPRODUCED, reason, sig
    if returncode in pass_codes:
        return PASSED, f"the command exited {returncode}", None
    return (
        INCOMPLETE,
        f"the command exited {returncode}, which is neither a fail code nor a "
        "pass code",
        None,
    )


def _matches(kernel_name: str, kernels: Optional[str]) -> bool:
    if kernels is None:
        return True
    return any(part and part in kernel_name for part in kernels.split(","))


def kernel_launches(paths: Sequence[Path]) -> set[tuple[str, str]]:
    """(name, checksum) of every `kernel_launch` event in the given files."""
    launches: set[tuple[str, str]] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "kernel_launch":
                launches.add(
                    (str(event.get("kernel_name")), str(event.get("kernel_checksum")))
                )
    return launches


def covered(
    config: Dict[str, Any], launches: set[tuple[str, str]], kernels: Optional[str]
) -> bool:
    """Whether a launched kernel of interest has delay points in the config.

    Both are needed: the config lists every instrumented kernel (launched or
    not), and launch events also list kernels the filter left out.
    """
    return any(
        isinstance(entry, dict)
        and _matches(str(entry.get("kernel_name", "")), kernels)
        and (str(entry.get("kernel_name")), str(entry.get("kernel_checksum")))
        in launches
        and bool(entry.get("instrumentation_points"))
        for entry in config.get("kernels", {}).values()
    )


def _event_files(directory: Path) -> set[Path]:
    try:
        return {
            path
            for path in directory.iterdir()
            if path.name.startswith(KERNEL_EVENTS_PREFIX) and path.suffix == ".ndjson"
        }
    except OSError:
        return set()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _pinned_signature(path: Optional[Path]) -> Optional[str]:
    """The signature an earlier reproduction pinned in `path`; None while
    there is none (no file, or an empty one)."""
    if path is None or not path.is_file():
        return None
    return path.read_text().strip() or None


def _write_atomically(path: Path, text: str) -> None:
    """Write `text` so that a reader sees either the old file or all of it."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _injection_error(plain: bool, env: Dict[str, str]) -> Optional[str]:
    """Why the injection's presence does not match `plain`; None when it does."""
    if plain == bool(env.get("CUDA_INJECTION64_PATH")):
        return (
            "a plain run found CUTracer's injection in its environment"
            if plain
            else "an injected run found no CUTracer injection"
        )
    return None


def _delay_evidence(
    env: Dict[str, str], attempt: Path
) -> tuple[Optional[str], Optional[str], Path, Path]:
    """The requested dump path, the replay path, the kernel-events directory,
    and the attempt's own delay-config evidence."""
    return (
        env.get("CUTRACER_DELAY_DUMP_PATH"),
        env.get("CUTRACER_DELAY_LOAD_PATH"),
        Path(env.get("CUTRACER_OUTPUT_DIR") or os.getcwd()),
        attempt / "delay.json",
    )


def _arm_injection(env: Dict[str, str], evidence: Path, replay: Optional[str]) -> None:
    """Point the child at the attempt's delay config (or keep the replay's
    candidate: CUTracer forbids dump and replay together, and the reducer
    removes its candidate file after the replay)."""
    env.setdefault("CUTRACER_KERNEL_EVENTS", "nostack")
    if replay:
        env.pop("CUTRACER_DELAY_DUMP_PATH", None)
        shutil.copyfile(replay, evidence)
    else:
        # Always dump (a control run may not ask for it) so coverage
        # can be checked; `_copy_dump_back` copies it to the requested path.
        env["CUTRACER_DELAY_DUMP_PATH"] = str(evidence)


def _copy_dump_back(evidence: Path, dump: Optional[str], replay: Optional[str]) -> None:
    """Give the caller the attempt's delay config at the path it asked CUTracer
    to dump to. Runs as soon as the child exits, before anything that can
    return early: a child that CUTracer's kernel timeout ended wrote no report,
    and the caller still needs the config to attribute the hang."""
    if dump and not replay and evidence.is_file():
        shutil.copyfile(evidence, dump)


def _check_coverage(
    evidence: Path,
    events_dir: Path,
    events_before: set[Path],
    kernels: Optional[str],
) -> tuple[bool, Optional[str]]:
    """Whether a launched kernel of interest has delay points in the attempt's
    delay config; the reason when it does not."""
    if not evidence.is_file():
        return False, "CUTracer wrote no delay config"
    config = json.loads(evidence.read_text())
    launches = kernel_launches(sorted(_event_files(events_dir) - events_before))
    if covered(config, launches, kernels):
        return True, None
    matching = f" matching {kernels!r}" if kernels else ""
    return (
        False,
        f"no launched kernel{matching} has delay points in CUTracer's delay config",
    )


def _finish_attempt(
    attempt: Path, outcome: Dict[str, Any], keep: bool, label: str = ""
) -> None:
    try:
        _write_json(attempt / "outcome.json", outcome)
    except OSError:
        pass
    if not keep:
        shutil.rmtree(attempt, ignore_errors=True)
    if "reason" in outcome:
        kept = f" (evidence: {attempt})" if keep else ""
        print(
            f"cutracer oracle: {label}exit {outcome['exit']}: {outcome['reason']}{kept}",
            file=sys.stderr,
        )


# Numerics ---------------------------------------------------------------------


@dataclass
class _NumericsRun:
    """How one child runs the capture."""

    mode: str  # "record" or "check"
    golden: Path
    # Capture settings: the user's when recording, the golden's when checking.
    settings: Dict[str, str]
    strict: bool
    capture: Path


def numerics_signature(label: Sequence[str]) -> str:
    """With `--numerics`, every reproduction of one test or command has this
    signature, whether its test failed or its numbers changed."""
    return hashlib.sha256(json.dumps(["numerics", *label]).encode()).hexdigest()


def _numerics_label(options: Union[OracleOptions, CmdOptions]) -> Dict[str, Any]:
    """What a golden was recorded for; a check of it must run the same."""
    if isinstance(options, OracleOptions):
        return {
            "oracle": "pytest",
            "nodeid": options.nodeid,
            "rootdir": str(options.rootdir),
        }
    return {"oracle": "cmd", "command": list(options.command), "cwd": os.getcwd()}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _record_settings(numerics: NumericsOptions, env: Dict[str, str]) -> Dict[str, str]:
    settings = {key: env[key] for key in _NUMERICS_SETTINGS if env.get(key)}
    # Hypothesis draws other examples in every run unless it is pinned.
    settings.setdefault("CUTRACER_NUMERICS_PIN_HYPOTHESIS", "1")
    settings.setdefault("PYTHONHASHSEED", "0")
    if numerics.wrap:
        settings["CUTRACER_NUMERICS_WRAP"] = ",".join(numerics.wrap)
    return settings


def _numerics_run(
    numerics: Optional[NumericsOptions],
    mode: Optional[str],
    golden: Optional[Path],
    label: Dict[str, Any],
    env: Dict[str, str],
) -> tuple[Optional[_NumericsRun], Optional[str]]:
    """The capture's setup for one child (None without `--numerics`), or why
    the run cannot be compared with the golden."""
    if numerics is None or mode is None or golden is None:
        return None, None
    capture = numerics.capture or NUMERICS_CAPTURE
    if not capture.is_file():
        return None, f"the numerics capture {capture} does not exist"
    if mode == "record":
        settings = _record_settings(numerics, env)
        return _NumericsRun(mode, golden, settings, False, capture), None
    manifest = _read_json(golden / MANIFEST)
    if manifest is None:
        return None, (
            f"{golden} holds no {MANIFEST}; record the golden with `--numerics record`"
        )
    for key, value in label.items():
        if manifest.get(key) != value:
            return None, (
                f"the golden was recorded for {key} {manifest.get(key)!r}, "
                f"not {value!r}"
            )
    if manifest.get("unstable"):
        return None, (
            "the golden's own plain runs reproduced without CUTracer (its record "
            f"exited {GOLDEN_UNSTABLE}): {manifest['unstable']}"
        )
    settings = dict(manifest.get("settings") or {})
    recorded_wrap = settings.get("CUTRACER_NUMERICS_WRAP", "")
    if numerics.wrap and sorted(numerics.wrap) != sorted(
        filter(None, recorded_wrap.split(","))
    ):
        return None, (
            f"the golden was recorded with --numerics-wrap {recorded_wrap!r}, not "
            f"{','.join(numerics.wrap)!r}"
        )
    if numerics.strict and not manifest.get("plain_runs"):
        return None, (
            "--numerics-strict needs plain-run records; record the golden with "
            "--numerics-runs 2 or more"
        )
    return _NumericsRun(mode, golden, settings, numerics.strict, capture), None


def _scrub_numerics(env: Dict[str, str]) -> None:
    """Drop inherited capture settings: only the oracle sets them."""
    for key in [key for key in env if key.startswith(NUMERICS_ENV_PREFIX)]:
        del env[key]


def _arm_numerics(
    env: Dict[str, str], attempt: Path, run: _NumericsRun, root: Path, cmd: bool
) -> Path:
    """Copy the capture into the attempt and point the child at it; return
    the copy. A pytest worker loads it by path; a command finds it through
    `PYTHONPATH` (a pytest command also through `PYTEST_PLUGINS`)."""
    lib = attempt / "numerics_lib"
    lib.mkdir()
    module = lib / f"{NUMERICS_MODULE}.py"
    shutil.copyfile(run.capture, module)
    env.update(run.settings)
    env.update(
        {
            "CUTRACER_NUMERICS_MODE": run.mode,
            "CUTRACER_NUMERICS_GOLDEN": str(run.golden.resolve()),
            "CUTRACER_NUMERICS_OUT": str(attempt / "numerics"),
            "CUTRACER_NUMERICS_ROOT": str(root),
        }
    )
    if cmd:
        env["CUTRACER_NUMERICS_LIB"] = str(lib)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(lib), env.get("PYTHONPATH")) if part
        )
        env["PYTEST_PLUGINS"] = ",".join(
            part for part in (env.get("PYTEST_PLUGINS"), NUMERICS_MODULE) if part
        )
    return module


def _numerics_results(out: Path) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    """The child's per-process results, and its notes (unlabeled processes,
    conflicting writers, write failures)."""
    results: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []
    if not out.is_dir():
        return results, notes
    for path in sorted(out.iterdir()):
        name = path.name
        if name.startswith("numerics-") and name.endswith(".json"):
            payload = _read_json(path) or {}
            summary = payload.get("numerics")
            if payload.get("schema") != 1 or not isinstance(summary, dict):
                summary = {
                    "errors": [f"{name} is not a capture result this oracle reads"]
                }
            results[name[len("numerics-") : -len(".json")]] = summary
        elif name.split("-", 1)[0] in ("unlabeled", "conflict", "error"):
            notes.append(name)
    return results, notes


def _golden_processes(golden: Path) -> List[str]:
    return sorted(
        path.name[len("calls-") : -len(".json")] for path in golden.glob("calls-*.json")
    )


# Keys a check result must have, and the verdicts it can give.
_CHECK_KEYS = (
    "verdict",
    "calls",
    "golden_calls",
    "environment",
    "structure",
    "first_structure_k",
    "first_large_k",
    "missing_calls",
    "first_large",
    "changed",
    "changed_stable_leaves",
    "stable_changed_head",
    "errors",
)
_CHECK_VERDICTS = ("same", "small", "large", "unmeasured")


def _format_problem(proc: str, summary: Dict[str, Any]) -> Optional[str]:
    """Why a check result cannot be read; None when it can."""
    if "verdict" not in summary and summary.get("errors"):
        return f"{proc}: {summary['errors'][0]}"
    missing = [key for key in _CHECK_KEYS if key not in summary]
    if missing:
        return (
            f"the capture's result for {proc} has no {missing[0]} (a capture from "
            "another CUTracer?)"
        )
    if summary["verdict"] not in _CHECK_VERDICTS:
        return f"the capture's result for {proc} has verdict {summary['verdict']!r}"
    return None


def _large_change(
    summary: Dict[str, Any], strict: bool
) -> Optional[tuple[str, Dict[str, Any]]]:
    """(rule, change) of a process's large change, if one came before any
    call that did not line up (a race can derail the rest of a test: a
    Hypothesis replay, a retry, an early exit)."""
    first_large = summary.get("first_large_k")
    first_off = summary.get("first_structure_k")
    if (
        summary.get("verdict") == "large"
        and first_large is not None
        and (first_off is None or first_large < first_off)
    ):
        return "large", summary.get("first_large") or {}
    if strict and summary.get("changed_stable_leaves") and first_off is None:
        return "strict", (summary.get("stable_changed_head") or [{}])[0]
    return None


def _describe_change(proc: str, change: Dict[str, Any], rule: str = "large") -> str:
    where = f"{change.get('call', '?')} {change.get('leaf', '?')}"
    facts = [
        f"{key}={change[key]:.3g}"
        if isinstance(change[key], float)
        else f"{key}={change[key]}"
        for key in ("rel_l2", "max_row_rel_l2", "n_outliers", "max_ulp", "n_diff")
        if change.get(key) is not None
    ]
    detail = f" ({', '.join(facts)})" if facts else ""
    if rule == "strict":
        what = "change of a tensor that was bitwise identical in every plain run"
    else:
        what = "large numerics change"
    return f"{what} in {proc}: {where}{detail}"


def _note_reason(note: str) -> str:
    kind = note.split("-", 1)[0]
    if kind == "unlabeled":
        return (
            f"a descendant process without a label compared tensors ({note}); "
            "label it with CUTRACER_NUMERICS_PROC or RANK"
        )
    if kind == "conflict":
        return f"two processes wrote the same numerics result ({note})"
    return f"the capture could not write its result ({note})"


def judge_numerics(  # noqa: C901
    results: Dict[str, Dict[str, Any]],
    notes: Sequence[str],
    golden_processes: Sequence[str],
    strict: bool = False,
    cmd: bool = False,
) -> tuple[Optional[int], str, str]:
    """Judge a comparable passing run by its numerics: (exit code, or None to
    keep the pass; the reason; the rule that decided)."""
    if not golden_processes:
        return INCOMPLETE, "the golden holds no recorded process", "golden"
    for proc in sorted(results):
        problem = _format_problem(proc, results[proc])
        if problem is not None:
            return INCOMPLETE, problem, "format"
        for problem in results[proc]["environment"]:
            return INCOMPLETE, f"{proc}: {problem}", "environment"
    # A second writer or a failed write: a result may not be this process's.
    for note in notes:
        if not note.startswith("unlabeled-"):
            return INCOMPLETE, _note_reason(note), "notes"
    for proc in sorted(results):
        found = _large_change(results[proc], strict)
        if found is not None:
            rule, change = found
            return REPRODUCED, _describe_change(proc, change, rule), rule
    if notes:
        return INCOMPLETE, _note_reason(notes[0]), "notes"
    missing = [proc for proc in golden_processes if proc not in results]
    if missing:
        if not results and cmd:
            reason = (
                f"the command never imported {NUMERICS_MODULE} (an interpreter "
                "that ignores PYTHONPATH?)"
            )
        else:
            reason = f"process {missing[0]} wrote no numerics result"
        return INCOMPLETE, reason, "missing_process"
    for proc in sorted(results):
        summary = results[proc]
        if proc not in golden_processes:
            return INCOMPLETE, f"process {proc} is not in the golden", "extra_process"
        if summary.get("errors"):
            return (
                INCOMPLETE,
                f"the capture failed in {proc}: {summary['errors'][0]}",
                "errors",
            )
        if summary.get("structure"):
            return INCOMPLETE, f"{proc}: {summary['structure'][0]}", "structure"
        if summary.get("missing_calls"):
            head = (summary.get("missing_head") or ["?"])[0]
            return (
                INCOMPLETE,
                f"{proc} made {summary.get('calls')} of the golden's "
                f"{summary.get('golden_calls')} calls; missing: {head}",
                "missing_calls",
            )
        if summary.get("verdict") == "unmeasured":
            change = next(
                (
                    c
                    for c in summary.get("changed") or []
                    if c.get("verdict") == "unmeasured"
                ),
                {},
            )
            return (
                INCOMPLETE,
                f"{proc}: a changed tensor could not be measured "
                f"({change.get('call', '?')} {change.get('leaf', '?')}: "
                f"{change.get('why', '?')})",
                "unmeasured",
            )
    verdict = max(
        (str(s.get("verdict")) for s in results.values()),
        key=lambda v: _SEVERITY.get(v, -1),
    )
    calls = sum(int(s.get("calls") or 0) for s in results.values())
    return None, f"numerics {verdict} over {calls} calls", "comparable"


def _numerics_evidence(
    run: _NumericsRun,
    results: Dict[str, Dict[str, Any]],
    notes: Sequence[str],
    rule: str,
) -> Dict[str, Any]:
    keys = (
        "verdict",
        "calls",
        "golden_calls",
        "missing_calls",
        "first_large_k",
        "first_structure_k",
        "changed_stable_leaves",
    )
    lists = ("environment", "structure", "errors", "changed")
    return {
        "mode": run.mode,
        "golden": str(run.golden),
        "rule": rule,
        "notes": list(notes),
        "processes": {
            proc: {
                **{key: summary.get(key) for key in keys if key in summary},
                **{
                    key: (summary.get(key) or [])[:4] for key in lists if key in summary
                },
            }
            for proc, summary in results.items()
        },
    }


def _apply_numerics(
    run: Optional[_NumericsRun],
    attempt: Path,
    code: int,
    reason: str,
    outcome: Dict[str, Any],
    cmd: bool,
) -> tuple[int, str]:
    """Fold a check run's numerics into the test's own verdict. Only a
    comparable run (it passed, or failed its correctness check) is judged;
    a reproduction stays one, and its numerics are evidence only."""
    if run is None or run.mode != "check" or code not in (PASSED, REPRODUCED):
        return code, reason
    results, notes = _numerics_results(attempt / "numerics")
    if code == REPRODUCED:
        outcome["numerics"] = _numerics_evidence(run, results, notes, "test")
        return code, reason
    numerics_code, numerics_reason, rule = judge_numerics(
        results, notes, _golden_processes(run.golden), run.strict, cmd
    )
    outcome["numerics"] = _numerics_evidence(run, results, notes, rule)
    if numerics_code is None:
        return code, f"{reason}; {numerics_reason}"
    return numerics_code, numerics_reason


def _record_problem(
    results: Dict[str, Dict[str, Any]], notes: Sequence[str], code: int, reason: str
) -> Optional[str]:
    """Why a record run wrote no usable golden; None when it did."""
    if code != PASSED:
        return f"the test did not pass while recording ({reason})"
    if notes:
        return _note_reason(notes[0])
    if not results:
        return (
            f"no process recorded anything (the test never imported {NUMERICS_MODULE}?)"
        )
    for proc, summary in sorted(results.items()):
        if summary.get("errors"):
            return f"the capture failed in {proc}: {summary['errors'][0]}"
        if summary.get("verdict") != "recorded":
            return f"{proc} did not record ({summary.get('verdict')})"
    if not sum(int(s.get("calls") or 0) for s in results.values()):
        return (
            "the test compared no tensors; name the functions whose results to "
            "compare with --numerics-wrap MODULE:FUNC, or call "
            f"{NUMERICS_MODULE}.record_tensor() in a driver (multi-process "
            "workers are not captured)"
        )
    return None


def _call_key(proc: str, entry: Dict[str, Any]) -> tuple:
    return (proc, entry["nodeid"], entry["fn"], entry["site"], entry["ordinal"])


def _plain_stats(outs: Sequence[Path]) -> Dict[tuple, Dict[str, Any]]:
    """Per tensor of the plain check runs: how many compared it, how many saw
    it change, and the largest change."""
    stats: Dict[tuple, Dict[str, Any]] = {}
    for path in sorted(p for out in outs for p in out.glob("calls-*.json")):
        proc = path.name[len("calls-") : -len(".json")]
        for entry in (_read_json(path) or {}).get("calls", []):
            for leaf in entry.get("leaves", []):
                cmp = leaf.get("cmp") or {}
                if cmp.get("verdict") not in ("same", "small", "large"):
                    continue
                record = stats.setdefault(
                    (*_call_key(proc, entry), leaf["name"]),
                    {"runs": 0, "changed": 0, "max_rel_l2": 0.0, "max_ulp": 0},
                )
                record["runs"] += 1
                record["changed"] += cmp["verdict"] != "same"
                rel = cmp.get("rel_l2")
                if isinstance(rel, (int, float)) and math.isfinite(rel):
                    record["max_rel_l2"] = max(record["max_rel_l2"], rel)
                elif rel is not None:
                    record["nonfinite_rel_l2"] = True
                record["max_ulp"] = max(record["max_ulp"], int(cmp.get("max_ulp") or 0))
    return stats


def _explained_by_plain_runs(out: Path, stats: Dict[tuple, Dict[str, Any]]) -> bool:
    """Whether every large change of a plain check run is an integer or bool
    tensor that changes between plain runs anyway (an argmax with ties): the
    run could not know that before the plain records were folded."""
    for path in sorted(out.glob("calls-*.json")):
        proc = path.name[len("calls-") : -len(".json")]
        for entry in (_read_json(path) or {}).get("calls", []):
            for leaf in entry.get("leaves", []):
                cmp = leaf.get("cmp") or {}
                if cmp.get("verdict") != "large":
                    continue
                record = stats.get((*_call_key(proc, entry), leaf.get("name")))
                tied = cmp.get("kind") == "exact" and not str(
                    cmp.get("dtype", "")
                ).startswith("complex")
                if not tied or not record or not record["changed"]:
                    return False
    return True


def _fold_plain_runs(golden: Path, outs: Sequence[Path]) -> Dict[tuple, Dict[str, Any]]:
    """Give every golden tensor the record of the plain check runs; return
    the records."""
    stats = _plain_stats(outs)
    for path in sorted(golden.glob("calls-*.json")):
        proc = path.name[len("calls-") : -len(".json")]
        payload = _read_json(path)
        if payload is None:
            continue
        for entry in payload.get("calls", []):
            for leaf in entry.get("leaves", []):
                record = stats.get((*_call_key(proc, entry), leaf.get("name")))
                if record is not None and leaf.get("kind") == "tensor":
                    leaf["plain"] = record
        _write_atomically(path, json.dumps(payload, indent=1) + "\n")
    return stats


# One run of the test for `record_golden`: (numerics mode, golden, where the
# attempt goes, label of its stderr line) -> (exit code, outcome, attempt).
_RunOnce = Callable[[str, Path, Path, str], "tuple[int, Dict[str, Any], Path]"]


def _worst_verdict(outcome: Dict[str, Any]) -> Optional[str]:
    processes = (outcome.get("numerics") or {}).get("processes") or {}
    verdicts = [str(p.get("verdict")) for p in processes.values()]
    return max(verdicts, key=lambda v: _SEVERITY.get(v, -1)) if verdicts else None


def record_golden(  # noqa: C901
    numerics: NumericsOptions,
    label: Dict[str, Any],
    output_dir: Optional[Path],
    run_once: _RunOnce,
) -> int:
    """`--numerics record`: run 1 records the golden, runs 2..N check it
    without CUTracer and record how much each tensor moves between plain
    runs. The golden appears only when every run is usable; without
    `output_dir`, the runs' evidence is kept only when it does not."""
    golden = numerics.golden.resolve()
    total = numerics.runs

    def done(code: int, reason: str) -> int:
        print(
            f"cutracer oracle: numerics record: exit {code}: {reason}", file=sys.stderr
        )
        return code

    if golden.exists() and (not golden.is_dir() or any(golden.iterdir())):
        return done(INCOMPLETE, f"{golden} is not an empty directory")
    staging = golden.with_name(f"{golden.name}.partial-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    attempts = output_dir or Path(tempfile.mkdtemp(prefix="cutracer-numerics-"))
    recorded = False
    try:
        code, outcome, attempt = run_once(
            "record", staging, attempts, f"numerics record run 1/{total} (golden): "
        )
        results, notes = _numerics_results(attempt / "numerics")
        problem = _record_problem(results, notes, code, str(outcome.get("reason")))
        if problem is not None:
            return done(INCOMPLETE, problem)
        calls = sum(int(s.get("calls") or 0) for s in results.values())
        manifest = {
            "schema": 1,
            **label,
            "settings": outcome.get("numerics_settings") or {},
            "processes": sorted(results),
            "calls": calls,
            "plain_runs": 0,
            "unstable": None,
            "recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _write_json(staging / MANIFEST, manifest)
        runs: List[Dict[str, Any]] = [{"run": 1, "mode": "record", "calls": calls}]
        checks: List[tuple[int, Path]] = []
        for number in range(2, total + 1):
            code, outcome, attempt = run_once(
                "check",
                staging,
                attempts,
                f"numerics record run {number}/{total} (plain check): ",
            )
            runs.append(
                {
                    "run": number,
                    "mode": "check",
                    "exit": code,
                    "reason": outcome.get("reason"),
                    "verdict": _worst_verdict(outcome),
                    "rule": (outcome.get("numerics") or {}).get("rule"),
                }
            )
            checks.append((code, attempt))
            if code == INCOMPLETE:
                return done(
                    INCOMPLETE,
                    f"plain run {number} could not be compared with run 1: "
                    f"{outcome.get('reason')}",
                )
        stats = _fold_plain_runs(
            staging, [attempt / "numerics" for _, attempt in checks]
        )
        for entry, (_, attempt) in zip(runs[1:], checks):
            if entry["exit"] == REPRODUCED and entry["rule"] == "large":
                entry["explained"] = _explained_by_plain_runs(
                    attempt / "numerics", stats
                )
        unstable = next(
            (r for r in runs[1:] if r["exit"] == REPRODUCED and not r.get("explained")),
            None,
        )
        final = GOLDEN_UNSTABLE if unstable is not None else GOLDEN_RECORDED
        manifest["plain_runs"] = len(checks)
        if unstable is not None:
            # Checks refuse this golden: its numbers change without CUTracer.
            manifest["unstable"] = f"plain run {unstable['run']}: {unstable['reason']}"
        _write_json(staging / MANIFEST, manifest)
        size = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
        _write_json(
            staging / STABILITY, {"runs": runs, "exit": final, "golden_bytes": size}
        )
        if golden.exists():
            golden.rmdir()
        staging.rename(golden)
        recorded = True
        if unstable is not None:
            return done(
                final,
                f"golden {golden} recorded, but plain run {unstable['run']} "
                f"already reproduced without CUTracer: {unstable['reason']}",
            )
        plain = ", ".join(
            "small (an integer tensor that varies anyway)"
            if r.get("explained")
            else str(r["verdict"])
            for r in runs[1:]
        )
        removed = "; the runs' evidence was removed" if output_dir is None else ""
        on_disk = (
            f"{size / 2**20:.1f} MiB" if size >= 2**20 else f"{size / 2**10:.0f} KiB"
        )
        return done(
            final,
            f"golden {golden} recorded: {calls} calls, {on_disk}; "
            f"plain runs: {plain or 'none'}{removed}",
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if output_dir is None and recorded:
            shutil.rmtree(attempts, ignore_errors=True)
        elif output_dir is None:
            print(
                f"cutracer oracle: numerics record: evidence in {attempts}",
                file=sys.stderr,
            )


def _worker_command(
    options: OracleOptions, report: Path, numerics_module: Optional[Path] = None
) -> List[str]:
    command = [
        options.python,
        str(options.script),
        "--worker-report",
        str(report),
        "--rootdir",
        str(options.rootdir),
        "--nodeid",
        options.nodeid,
        "--import-mode",
        options.import_mode,
    ]
    command += [f"--assertion-site={site}" for site in options.assertion_sites]
    if numerics_module is not None:
        command += ["--numerics-module", str(numerics_module)]
    return command + [f"--pytest-arg={arg}" for arg in options.pytest_args]


def _run_child(
    command: List[str],
    env: Dict[str, str],
    log_path: Path,
    timeout_s: Optional[float],
    what: str = "the node",
) -> tuple[Optional[int], List[str]]:
    """Run the worker in this process's group; end it on a timeout or signal."""
    errors: List[str] = []
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )

        def stop(signum: int, frame: object) -> None:
            raise InterruptedError(f"oracle interrupted by signal {signum}")

        previous = {
            sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
        }
        returncode: Optional[int] = None
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            errors.append(f"{what} ran past {timeout_s} s")
        except InterruptedError as error:
            errors.append(str(error))
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                errors.append(f"{what} did not exit after SIGKILL")
    return returncode, errors


def run_oracle(options: OracleOptions) -> int:
    """Run the node once and return the oracle exit code. Evidence (the worker
    report and log, the delay config, `outcome.json`) goes to a fresh
    `attempt-*` directory under `output_dir`, or to a temporary one. The exit
    code and its reason also go to stderr, as one line. With `--numerics
    record`, record a golden instead (`record_golden`)."""
    numerics = options.numerics
    if numerics is not None and numerics.mode == "record":
        plain = replace(
            options, plain=True, signature_file=None, expected_signature=None
        )
        return record_golden(
            numerics,
            _numerics_label(options),
            options.output_dir,
            lambda mode, golden, out, label: _run_pytest(
                replace(plain, output_dir=out), mode, golden, label
            ),
        )
    mode = numerics.mode if numerics is not None else None
    golden = numerics.golden if numerics is not None else None
    return _run_pytest(options, mode, golden)[0]


def _run_pytest(  # noqa: C901
    options: OracleOptions,
    mode: Optional[str],
    golden: Optional[Path],
    label: str = "",
) -> tuple[int, Dict[str, Any], Path]:
    """One run of the node: (exit code, outcome, attempt directory)."""
    keep = options.output_dir is not None
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=options.output_dir))
    outcome: Dict[str, Any] = {"exit": INCOMPLETE, "nodeid": options.nodeid}

    def finish(code: int, reason: str) -> tuple[int, Dict[str, Any], Path]:
        outcome.update(exit=code, reason=reason)
        return code, outcome, attempt

    try:
        env = dict(os.environ)
        run, error = _numerics_run(
            options.numerics, mode, golden, _numerics_label(options), env
        )
        if error is not None:
            return finish(INCOMPLETE, error)
        _scrub_numerics(env)
        error = _injection_error(options.plain, env)
        if error is not None:
            return finish(INCOMPLETE, error)
        dump, replay, events_dir, evidence = _delay_evidence(env, attempt)
        if not options.plain:
            _arm_injection(env, evidence, replay)
        module = None
        if run is not None:
            module = _arm_numerics(env, attempt, run, options.rootdir, cmd=False)
            if run.mode == "record":
                outcome["numerics_settings"] = run.settings
        events_before = _event_files(events_dir)
        report_path = attempt / "pytest.json"
        returncode, errors = _run_child(
            _worker_command(options, report_path, module),
            env,
            attempt / "pytest.log",
            options.timeout_s,
        )
        outcome.update(child_exit=returncode, errors=errors)
        if not options.plain:
            _copy_dump_back(evidence, dump, replay)
        if errors:
            return finish(INCOMPLETE, "; ".join(errors))
        report = json.loads(report_path.read_text())
        pinned = _pinned_signature(options.signature_file)
        expected = (
            options.expected_signature
            if options.expected_signature is not None
            else pinned
        )
        # With numerics, the signature is checked once the numbers are judged.
        code, reason = classify(
            report, options.nodeid, expected if run is None else None
        )
        outcome.update(
            signature=report.get("signature"),
            assertion_site=report.get("assertion_site"),
            failure=report.get("failure"),
        )
        if returncode != report.get("pytest_exit"):
            code, reason = (
                INCOMPLETE,
                f"the worker exited {returncode}, but its report says pytest "
                f"exited {report.get('pytest_exit')}",
            )
        if not options.plain:
            coverage, failure = _check_coverage(
                evidence, events_dir, events_before, options.kernels
            )
            outcome["coverage"] = coverage
            if failure is not None:
                return finish(INCOMPLETE, failure)
        sig = report.get("signature")
        if run is not None:
            code, reason = _apply_numerics(
                run, attempt, code, reason, outcome, cmd=False
            )
            sig = numerics_signature([options.nodeid]) if code == REPRODUCED else None
            outcome["signature"] = sig
            if code == REPRODUCED and expected is not None and sig != expected:
                code, reason = INCOMPLETE, f"{reason}, which is not the pinned failure"
        if code == REPRODUCED and options.signature_file is not None and not pinned:
            _write_atomically(options.signature_file, str(sig) + "\n")
        return finish(code, reason)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return finish(INCOMPLETE, f"{type(error).__name__}: {error}")
    finally:
        _finish_attempt(attempt, outcome, keep, label)


def _cmd_options_error(options: CmdOptions) -> Optional[str]:
    """Why the command options are unusable; None when they are fine."""
    if not options.fail_codes or not options.pass_codes:
        return "fail and pass codes must both be non-empty"
    overlap = sorted(set(options.fail_codes) & set(options.pass_codes))
    if overlap:
        return f"exit code(s) {overlap} are both fail and pass codes"
    if not options.command:
        return "expected a command after `--`"
    return None


def _pin_cmd_signature(
    code: int, signature_file: Optional[Path], pinned: Optional[str], sig: Optional[str]
) -> None:
    """Write the first reproduction's signature."""
    if sig is None:
        return
    if code == REPRODUCED and signature_file is not None and not pinned:
        _write_atomically(signature_file, sig + "\n")


def run_cmd_oracle(options: CmdOptions) -> int:
    """Run the command once and return the oracle exit code. Evidence (the
    log, the delay config, `outcome.json`) goes to a fresh `attempt-*`
    directory under `output_dir`, or to a temporary one. The exit code and
    its reason also go to stderr, as one line. With `--numerics record`,
    record a golden instead (`record_golden`)."""
    numerics = options.numerics
    if numerics is not None and numerics.mode == "record":
        error = _cmd_options_error(options)
        if error is not None:
            print(f"cutracer oracle: numerics record: exit 2: {error}", file=sys.stderr)
            return INCOMPLETE
        plain = replace(options, plain=True, signature_file=None)
        return record_golden(
            numerics,
            _numerics_label(options),
            options.output_dir,
            lambda mode, golden, out, label: _run_cmd(
                replace(plain, output_dir=out), mode, golden, label
            ),
        )
    mode = numerics.mode if numerics is not None else None
    golden = numerics.golden if numerics is not None else None
    return _run_cmd(options, mode, golden)[0]


def _run_cmd(  # noqa: C901
    options: CmdOptions,
    mode: Optional[str],
    golden: Optional[Path],
    label: str = "",
) -> tuple[int, Dict[str, Any], Path]:
    """One run of the command: (exit code, outcome, attempt directory)."""
    keep = options.output_dir is not None
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=options.output_dir))
    outcome: Dict[str, Any] = {"exit": INCOMPLETE, "command": list(options.command)}

    def finish(code: int, reason: str) -> tuple[int, Dict[str, Any], Path]:
        outcome.update(exit=code, reason=reason)
        return code, outcome, attempt

    try:
        env = dict(os.environ)
        error = _cmd_options_error(options)
        if error is not None:
            return finish(INCOMPLETE, error)
        run, error = _numerics_run(
            options.numerics, mode, golden, _numerics_label(options), env
        )
        if error is not None:
            return finish(INCOMPLETE, error)
        _scrub_numerics(env)
        error = _injection_error(options.plain, env)
        if error is not None:
            return finish(INCOMPLETE, error)
        dump, replay, events_dir, evidence = _delay_evidence(env, attempt)
        if not options.plain:
            _arm_injection(env, evidence, replay)
        if run is not None:
            _arm_numerics(env, attempt, run, Path(os.getcwd()), cmd=True)
            if run.mode == "record":
                outcome["numerics_settings"] = run.settings
        events_before = _event_files(events_dir)
        log_path = attempt / "cmd.log"
        returncode, errors = _run_child(
            list(options.command),
            env,
            log_path,
            options.timeout_s,
            what="the command",
        )
        outcome.update(child_exit=returncode, errors=errors)
        if not options.plain:
            _copy_dump_back(evidence, dump, replay)
        if errors or returncode is None:
            return finish(
                INCOMPLETE,
                "; ".join(errors) if errors else "the command reported no exit code",
            )
        output = log_path.read_text(errors="replace")
        pinned = _pinned_signature(options.signature_file)
        code, reason, sig = classify_cmd(
            returncode,
            output,
            frozenset(options.fail_codes),
            frozenset(options.pass_codes),
            options.fail_pattern,
            pinned if run is None else None,
        )
        outcome["signature"] = sig
        if not options.plain:
            coverage, failure = _check_coverage(
                evidence, events_dir, events_before, options.kernels
            )
            outcome["coverage"] = coverage
            if failure is not None:
                return finish(INCOMPLETE, failure)
        if run is not None:
            code, reason = _apply_numerics(
                run, attempt, code, reason, outcome, cmd=True
            )
            sig = (
                numerics_signature(list(options.command))
                if code == REPRODUCED
                else None
            )
            outcome["signature"] = sig
            if code == REPRODUCED and pinned is not None and sig != pinned:
                code, reason = INCOMPLETE, f"{reason}, which is not the pinned failure"
        _pin_cmd_signature(code, options.signature_file, pinned, sig)
        return finish(code, reason)
    except (OSError, ValueError, KeyError, TypeError, re.error) as error:
        return finish(INCOMPLETE, f"{type(error).__name__}: {error}")
    finally:
        _finish_attempt(attempt, outcome, keep, label)


def _load_numerics(path: Path) -> Any:
    """Import the capture from `path` as `cutracer_numerics`, before pytest
    imports any conftest; with its mode set, the import installs it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(NUMERICS_MODULE, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[NUMERICS_MODULE] = module
    spec.loader.exec_module(module)
    return module


def run_worker(args: argparse.Namespace) -> int:
    """The child: run the node in-process under pytest and write the report."""
    import pytest

    sites = {tuple(site.split("::", 1)) for site in args.assertion_site}
    state: Dict[str, Any] = {"selected": [], "phases": [], "collection_errors": []}

    class Outcomes:
        def pytest_collection_finish(self, session: Any) -> None:
            state["selected"] = [item.nodeid for item in session.items]

        def pytest_collectreport(self, report: Any) -> None:
            if report.failed:
                state["collection_errors"].append(str(report.longrepr))

        def pytest_runtest_makereport(self, item: Any, call: Any) -> None:
            if call.when != "call" or call.excinfo is None:
                return
            site = assertion_site(call.excinfo.traceback, args.rootdir)
            asserted = call.excinfo.type is AssertionError
            exception = call.excinfo.type.__name__
            # A multi-process test sees its child's failure as a launcher error.
            child = child_failure(call.excinfo.value, args.rootdir)
            if child is not None:
                exception, site = child
                asserted = exception == "AssertionError"
                state["child_failure"] = True
            state["exception"] = exception
            correctness = (
                asserted
                and site is not None
                and (not sites or (site[0], site[1]) in sites)
            )
            state["correctness_failure"] = correctness
            state["failure"] = str(call.excinfo.value)[:2000]
            state["assertion_site"] = site
            if correctness and site is not None:
                state["signature"] = signature(args.nodeid, site)

        def pytest_runtest_logreport(self, report: Any) -> None:
            state["phases"].append(
                {
                    "nodeid": report.nodeid,
                    "when": report.when,
                    "outcome": report.outcome,
                    "xfail": bool(getattr(report, "wasxfail", False)),
                }
            )

    plugins: List[Any] = [Outcomes()]
    if args.numerics_module is not None:
        plugins.append(_load_numerics(args.numerics_module))
    code = pytest.main(
        [
            "--rootdir",
            str(args.rootdir),
            f"--import-mode={args.import_mode}",
            "-q",
            "-s",
            "-p",
            "no:cacheprovider",
            *args.pytest_args,
            str(args.rootdir / args.nodeid),
        ],
        plugins=plugins,
    )
    state["pytest_exit"] = int(code)
    _write_json(args.worker_report, state)
    return int(code)


def _site(value: str) -> str:
    path, separator, function = value.partition("::")
    if not path or not separator or not function or "::" in function:
        raise argparse.ArgumentTypeError(f"expected FILE::FUNCTION, got {value!r}")
    return value


def _positive(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return number


def _codes(value: str) -> FrozenSet[int]:
    try:
        codes = frozenset(int(part) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated exit codes, got {value!r}"
        ) from error
    if not codes:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated exit codes, got {value!r}"
        )
    if any(code < 0 or code > 255 for code in codes):
        raise argparse.ArgumentTypeError(f"expected exit codes 0-255, got {value!r}")
    return codes


def _pattern(value: str) -> str:
    try:
        re.compile(value)
    except re.error as error:
        raise argparse.ArgumentTypeError(f"invalid regex {value!r}: {error}") from error
    return value


def _runs(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a count, got {value!r}") from error
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected at least 1, got {value!r}")
    return number


def _wrap_entry(value: str) -> str:
    module, separator, function = value.partition(":")
    if not module or not separator or not function or "," in value:
        raise argparse.ArgumentTypeError(f"expected MODULE:FUNCTION, got {value!r}")
    return value


def _add_numerics_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "numerics",
        "Compare every tensor the test compares (and the results of "
        "--numerics-wrap functions) with a golden recorded from plain runs.",
    )
    group.add_argument(
        "--numerics",
        choices=("record", "check"),
        default=None,
        help="`check`: a large change of a tensor counts as a reproduction even "
        "when the test passes, and a run that does not line up with the golden "
        "measured nothing (exit 2). `record`: write the golden from plain runs "
        "(not an oracle: exit 0 for a usable golden, 3 when its own plain runs "
        "change by a large amount, 2 for none).",
    )
    group.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="The golden directory (record: a new or empty one).",
    )
    group.add_argument(
        "--numerics-runs",
        type=_runs,
        default=None,
        metavar="N",
        help="record: run 1 records, runs 2..N check it without CUTracer and "
        "note how much each tensor moves between plain runs (default: 3).",
    )
    group.add_argument(
        "--numerics-wrap",
        type=_wrap_entry,
        action="append",
        default=[],
        metavar="MODULE:FUNCTION",
        help="Also compare this function's tensor results (`MODULE:*`: every "
        "public function of the module, `MODULE:**`: also its torch.library "
        "custom ops); repeatable. A check uses the golden's entries.",
    )
    group.add_argument(
        "--numerics-strict",
        action="store_true",
        help="check: also count any change of a tensor that was bitwise "
        "identical in every plain run of the golden (deterministic kernels).",
    )
    group.add_argument("--numerics-capture", type=Path, help=argparse.SUPPRESS)


def _numerics_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Optional[NumericsOptions]:
    """The numerics options, or a usage error for a combination that means
    nothing."""
    if args.numerics is None:
        given = [
            name
            for name, value in (
                ("--golden", args.golden),
                ("--numerics-runs", args.numerics_runs),
                ("--numerics-wrap", args.numerics_wrap),
                ("--numerics-strict", args.numerics_strict),
            )
            if value
        ]
        if given:
            parser.error(f"{given[0]} needs --numerics")
        return None
    if args.golden is None:
        parser.error("--numerics needs --golden DIR")
    if args.numerics == "check" and args.numerics_runs is not None:
        parser.error("--numerics-runs is for --numerics record")
    if args.numerics == "record" and args.numerics_strict:
        parser.error("--numerics-strict is for --numerics check")
    if args.numerics == "record" and (
        args.signature_file is not None
        or getattr(args, "expected_signature", None) is not None
    ):
        parser.error("--numerics record pins no signature: it is not an oracle")
    return NumericsOptions(
        mode=args.numerics,
        golden=args.golden.resolve(),
        runs=args.numerics_runs or 3,
        wrap=tuple(args.numerics_wrap),
        strict=args.numerics_strict,
        capture=args.numerics_capture,
    )


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run one pytest node as a stress/reduce oracle: exit 0 when "
        "it fails its correctness check, 1 when it passes, 2 otherwise.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="The project's interpreter, which has pytest and the code under "
        "test (default: this interpreter).",
    )
    parser.add_argument(
        "--rootdir",
        type=Path,
        required=True,
        help="pytest rootdir; the node id and assertion sites are relative to it.",
    )
    parser.add_argument("--nodeid", required=True, help="The one test node to run.")
    parser.add_argument(
        "--assertion-site",
        action="append",
        default=[],
        type=_site,
        metavar="FILE::FUNCTION",
        help="A function whose AssertionError is the correctness failure, by "
        "rootdir-relative file. The innermost frame under the rootdir "
        "(installed packages excluded) must be one of them. Repeatable; "
        "default: any AssertionError raised under the rootdir.",
    )
    parser.add_argument(
        "-k",
        "--kernels",
        default=None,
        help="Comma-separated kernel-name substrings. An injected run must have "
        "launched one of them with delay points in its config, or it measured "
        "nothing (exit 2). Default: any instrumented kernel.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Keep each run's evidence in a new attempt-* directory here (worker "
        "report and log, delay config, outcome.json). Default: discard it.",
    )
    parser.add_argument(
        "--expected-signature",
        default=None,
        help="Count a failure only with this signature (node + assertion site; "
        "with --numerics check, the numerics signature of the node).",
    )
    parser.add_argument(
        "--signature-file",
        type=Path,
        default=None,
        help="The first reproduction writes its signature here (an empty file "
        "holds none yet), and later runs count a failure only if it matches, so "
        "a search and its replays agree on which failure they reproduce. With "
        "--numerics check, every reproduction of the test has one signature.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive,
        default=None,
        help="End the node after this many seconds (exit 2).",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Run without CUTracer: the injection must be absent, and no delay "
        "config is checked (a preflight).",
    )
    parser.add_argument(
        "--import-mode", default="importlib", help="pytest --import-mode."
    )
    parser.add_argument(
        "--pytest-arg",
        dest="pytest_args",
        action="append",
        default=[],
        metavar="ARG",
        help="One more pytest argument, after the defaults `-p no:timeout -o "
        "addopts=`; repeatable. Write `--pytest-arg=-x` for one that starts "
        "with a dash.",
    )
    _add_numerics_arguments(parser)
    parser.add_argument("--worker-report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--numerics-module", type=Path, help=argparse.SUPPRESS)
    return parser


def build_cmd_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run any command as a stress/reduce oracle: exit 0 when it "
        "exits with a fail code (and its output matches --fail-pattern, if "
        "given), 1 when it exits with a pass code, 2 otherwise.",
    )
    parser.add_argument(
        "--fail-codes",
        type=_codes,
        default=frozenset({1}),
        metavar="N,...",
        help="Command exit codes that reproduce the bug (default: 1).",
    )
    parser.add_argument(
        "--pass-codes",
        type=_codes,
        default=frozenset({0}),
        metavar="N,...",
        help="Command exit codes that pass (default: 0). Any other code, a "
        "signal, a timeout or an interrupt measures nothing (exit 2).",
    )
    parser.add_argument(
        "--fail-pattern",
        type=_pattern,
        default=None,
        metavar="REGEX",
        help="Require this regex in the command's output (stdout and stderr) "
        "for a fail-code run to count; the signature is pinned on the matched "
        "text.",
    )
    parser.add_argument(
        "-k",
        "--kernels",
        default=None,
        help="Comma-separated kernel-name substrings. An injected run must have "
        "launched one of them with delay points in its config, or it measured "
        "nothing (exit 2). Default: any instrumented kernel.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Keep each run's evidence in a new attempt-* directory here (log, "
        "delay config, outcome.json). Default: discard it.",
    )
    parser.add_argument(
        "--signature-file",
        type=Path,
        default=None,
        help="The first reproduction writes its signature here (an empty file "
        "holds none yet), and later runs count a failure only if it matches, so "
        "a search and its replays agree on which failure they reproduce. With "
        "--numerics check, every reproduction of the test has one signature.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive,
        default=None,
        help="End the command after this many seconds (exit 2).",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Run without CUTracer: the injection must be absent, and no delay "
        "config is checked (a preflight).",
    )
    _add_numerics_arguments(parser)
    return parser


def main_cmd(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
    given = list(sys.argv[1:]) if argv is None else list(argv)
    command: List[str] = []
    if "--" in given:
        split = given.index("--")
        command, given = given[split + 1 :], given[:split]
    parser = build_cmd_parser(prog)
    args = parser.parse_args(given)
    if not command:
        parser.error("expected a command after `--`")
    if args.fail_codes & args.pass_codes:
        parser.error(
            f"exit code(s) {sorted(args.fail_codes & args.pass_codes)} are both "
            "fail and pass codes"
        )
    numerics = _numerics_options(parser, args)
    try:
        return run_cmd_oracle(
            CmdOptions(
                command=command,
                fail_codes=args.fail_codes,
                pass_codes=args.pass_codes,
                fail_pattern=args.fail_pattern,
                kernels=args.kernels,
                output_dir=args.output_dir,
                signature_file=args.signature_file,
                timeout_s=args.timeout,
                plain=args.plain,
                numerics=numerics,
            )
        )
    except Exception as error:  # noqa: B902
        # Python's own exit code for an uncaught exception is 1, which a
        # search reads as a clean pass; anything unexpected must be 2.
        print(f"oracle infrastructure error: {error!r}", file=sys.stderr)
        return INCOMPLETE


def main(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv[:1] == ["cmd"]:
        return main_cmd(argv[1:], prog=prog)
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    if args.worker_report is not None:
        return run_worker(args)
    numerics = _numerics_options(parser, args)
    try:
        return run_oracle(
            OracleOptions(
                python=args.python,
                rootdir=args.rootdir.resolve(),
                nodeid=args.nodeid,
                assertion_sites=args.assertion_site,
                kernels=args.kernels,
                output_dir=args.output_dir,
                expected_signature=args.expected_signature,
                signature_file=args.signature_file,
                timeout_s=args.timeout,
                plain=args.plain,
                import_mode=args.import_mode,
                pytest_args=(*DEFAULT_PYTEST_ARGS, *args.pytest_args),
                numerics=numerics,
            )
        )
    except Exception as error:  # noqa: B902
        # Python's own exit code for an uncaught exception is 1, which a
        # search reads as a clean pass; anything unexpected must be 2.
        print(f"oracle infrastructure error: {error!r}", file=sys.stderr)
        return INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
