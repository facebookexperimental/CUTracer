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

This file needs only the standard library (and pytest in the child), so the
project's interpreter can run it directly: ``cutracer oracle export PATH``
writes it out, which saves starting CUTracer for every run of a search.
``cutracer oracle pytest`` runs the same code.
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
from typing import Any, Dict, List, Optional, Sequence

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
    command: List[str], env: Dict[str, str], log_path: Path, timeout_s: Optional[float]
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
            errors.append(f"the node ran past {timeout_s} s")
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
                errors.append("the node did not exit after SIGKILL")
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
        injected = bool(env.get("CUDA_INJECTION64_PATH"))
        if options.plain == injected:
            return finish(
                INCOMPLETE,
                "a plain run found CUTracer's injection in its environment"
                if options.plain
                else "an injected run found no CUTracer injection",
            )
        dump = env.get("CUTRACER_DELAY_DUMP_PATH")
        replay = env.get("CUTRACER_DELAY_LOAD_PATH")
        events_dir = Path(env.get("CUTRACER_OUTPUT_DIR") or os.getcwd())
        evidence = attempt / "delay.json"
        if not options.plain:
            env.setdefault("CUTRACER_KERNEL_EVENTS", "nostack")
            if replay:
                # CUTracer forbids dump and replay together, and the reducer
                # removes its candidate file after the replay: keep a copy.
                env.pop("CUTRACER_DELAY_DUMP_PATH", None)
                shutil.copyfile(replay, evidence)
            else:
                # Always dump (a control run may not ask for it) so coverage
                # can be checked; copied back to the requested path below.
                env["CUTRACER_DELAY_DUMP_PATH"] = str(evidence)
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
            if not evidence.is_file():
                outcome["coverage"] = False
                return finish(INCOMPLETE, "CUTracer wrote no delay config")
            if dump and not replay:
                shutil.copyfile(evidence, dump)
            config = json.loads(evidence.read_text())
            launches = kernel_launches(sorted(_event_files(events_dir) - events_before))
            outcome["coverage"] = covered(config, launches, options.kernels)
            if not outcome["coverage"]:
                matching = f" matching {options.kernels!r}" if options.kernels else ""
                return finish(
                    INCOMPLETE,
                    f"no launched kernel{matching} has delay points in "
                    "CUTracer's delay config",
                )
        if code == REPRODUCED and options.signature_file is not None and not pinned:
            _write_atomically(options.signature_file, str(report["signature"]) + "\n")
        return finish(code, reason)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return finish(INCOMPLETE, f"{type(error).__name__}: {error}")
    finally:
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


def main(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
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
