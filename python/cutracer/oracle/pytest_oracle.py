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

This file needs only the standard library (and pytest in the child, for
pytest mode), so the project's interpreter can run it directly: ``cutracer
oracle export PATH`` writes it out, which saves starting CUTracer for
every run of a search. ``cutracer oracle pytest`` and ``cutracer oracle
cmd`` run the same code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

REPRODUCED = 0
PASSED = 1
INCOMPLETE = 2

KERNEL_EVENTS_PREFIX = "cutracer_kernel_events_"
# Disable plugins and project options that change how one node runs: an ini
# `addopts` can turn on xdist, and a plugin timeout would race the oracle's own.
DEFAULT_PYTEST_ARGS = ("-p", "no:timeout", "-o", "addopts=")


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
        # can be checked; copied back to the requested path below.
        env["CUTRACER_DELAY_DUMP_PATH"] = str(evidence)


def _check_coverage(
    evidence: Path,
    dump: Optional[str],
    replay: Optional[str],
    events_dir: Path,
    events_before: set[Path],
    kernels: Optional[str],
) -> tuple[bool, Optional[str]]:
    """Copy the dump back, then whether a launched kernel of interest has
    delay points in the config; the reason when it does not."""
    if not evidence.is_file():
        return False, "CUTracer wrote no delay config"
    if dump and not replay:
        shutil.copyfile(evidence, dump)
    config = json.loads(evidence.read_text())
    launches = kernel_launches(sorted(_event_files(events_dir) - events_before))
    if covered(config, launches, kernels):
        return True, None
    matching = f" matching {kernels!r}" if kernels else ""
    return (
        False,
        f"no launched kernel{matching} has delay points in CUTracer's delay config",
    )


def _finish_attempt(attempt: Path, outcome: Dict[str, Any], keep: bool) -> None:
    try:
        _write_json(attempt / "outcome.json", outcome)
    except OSError:
        pass
    if not keep:
        shutil.rmtree(attempt, ignore_errors=True)
    if "reason" in outcome:
        kept = f" (evidence: {attempt})" if keep else ""
        print(
            f"cutracer oracle: exit {outcome['exit']}: {outcome['reason']}{kept}",
            file=sys.stderr,
        )


def _worker_command(options: OracleOptions, report: Path) -> List[str]:
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


def run_oracle(options: OracleOptions) -> int:  # noqa: C901
    """Run the node once and return the oracle exit code. Evidence (the worker
    report and log, the delay config, `outcome.json`) goes to a fresh
    `attempt-*` directory under `output_dir`, or to a temporary one. The exit
    code and its reason also go to stderr, as one line."""
    keep = options.output_dir is not None
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=options.output_dir))
    outcome: Dict[str, Any] = {"exit": INCOMPLETE, "nodeid": options.nodeid}

    def finish(code: int, reason: str) -> int:
        outcome.update(exit=code, reason=reason)
        return code

    try:
        env = dict(os.environ)
        error = _injection_error(options.plain, env)
        if error is not None:
            return finish(INCOMPLETE, error)
        dump, replay, events_dir, evidence = _delay_evidence(env, attempt)
        if not options.plain:
            _arm_injection(env, evidence, replay)
        events_before = _event_files(events_dir)
        report_path = attempt / "pytest.json"
        returncode, errors = _run_child(
            _worker_command(options, report_path),
            env,
            attempt / "pytest.log",
            options.timeout_s,
        )
        outcome.update(child_exit=returncode, errors=errors)
        if errors:
            return finish(INCOMPLETE, "; ".join(errors))
        report = json.loads(report_path.read_text())
        pinned = _pinned_signature(options.signature_file)
        expected = (
            options.expected_signature
            if options.expected_signature is not None
            else pinned
        )
        code, reason = classify(report, options.nodeid, expected)
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
                evidence, dump, replay, events_dir, events_before, options.kernels
            )
            outcome["coverage"] = coverage
            if failure is not None:
                return finish(INCOMPLETE, failure)
        if code == REPRODUCED and options.signature_file is not None and not pinned:
            _write_atomically(options.signature_file, str(report["signature"]) + "\n")
        return finish(code, reason)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return finish(INCOMPLETE, f"{type(error).__name__}: {error}")
    finally:
        _finish_attempt(attempt, outcome, keep)


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
    its reason also go to stderr, as one line."""
    keep = options.output_dir is not None
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=options.output_dir))
    outcome: Dict[str, Any] = {"exit": INCOMPLETE, "command": list(options.command)}

    def finish(code: int, reason: str) -> int:
        outcome.update(exit=code, reason=reason)
        return code

    try:
        env = dict(os.environ)
        error = _cmd_options_error(options)
        if error is not None:
            return finish(INCOMPLETE, error)
        error = _injection_error(options.plain, env)
        if error is not None:
            return finish(INCOMPLETE, error)
        dump, replay, events_dir, evidence = _delay_evidence(env, attempt)
        if not options.plain:
            _arm_injection(env, evidence, replay)
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
            pinned,
        )
        outcome["signature"] = sig
        if not options.plain:
            coverage, failure = _check_coverage(
                evidence, dump, replay, events_dir, events_before, options.kernels
            )
            outcome["coverage"] = coverage
            if failure is not None:
                return finish(INCOMPLETE, failure)
        _pin_cmd_signature(code, options.signature_file, pinned, sig)
        return finish(code, reason)
    except (OSError, ValueError, KeyError, TypeError, re.error) as error:
        return finish(INCOMPLETE, f"{type(error).__name__}: {error}")
    finally:
        _finish_attempt(attempt, outcome, keep)


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
        plugins=[Outcomes()],
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
        help="Count a failure only with this signature (node + assertion site).",
    )
    parser.add_argument(
        "--signature-file",
        type=Path,
        default=None,
        help="The first reproduction writes its signature here (an empty file "
        "holds none yet), and later runs count a failure only if it matches, so "
        "a search and its replays agree on which failure they reproduce.",
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
    parser.add_argument("--worker-report", type=Path, help=argparse.SUPPRESS)
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
        "a search and its replays agree on which failure they reproduce.",
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
    args = build_parser(prog).parse_args(argv)
    if args.worker_report is not None:
        return run_worker(args)
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
            )
        )
    except Exception as error:  # noqa: B902
        # Python's own exit code for an uncaught exception is 1, which a
        # search reads as a clean pass; anything unexpected must be 2.
        print(f"oracle infrastructure error: {error!r}", file=sys.stderr)
        return INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
