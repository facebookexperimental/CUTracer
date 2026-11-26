# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pytest
from cutracer.oracle.cli import pytest_oracle
from cutracer.runner import ProcessTerminated, run_process_group
from cutracer.stress.modes import get_modes
from cutracer.validation.json_validator import validate_json_trace

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseSelection:
    root: Path
    inventory: dict
    cases: tuple[dict, ...]


def project_python(output: Path | None = None) -> str:
    runtime = os.environ.get("FB_PAR_RUNTIME_FILES")
    bootstrap = Path(runtime) / "_bootstrap.sh" if runtime else None
    if bootstrap and bootstrap.is_file():
        return str(bootstrap)
    if output is None:
        return sys.executable
    launcher = output / "project-python"
    python_path = os.pathsep.join(os.path.realpath(path) for path in sys.path if path)
    launcher.write_text(
        "#!/bin/sh\n"
        f'PYTHONPATH={shlex.quote(python_path)} exec {shlex.quote(sys.executable)} "$@"\n'
    )
    launcher.chmod(0o700)
    return str(launcher)


class _Collection:
    def __init__(self) -> None:
        self.cases: list[dict] = []
        self.paths: set[Path] = set()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        for item in session.items:
            self.cases.append(
                {
                    "case_id": hashlib.sha256(
                        f"pytest:{item.nodeid}".encode()
                    ).hexdigest()[:16],
                    "nodeid": item.nodeid,
                }
            )
            self.paths.add(Path(item.path))


def collect_inventory(
    root: Path,
    tests: Sequence[str],
    output: Path,
    *,
    pytest_args: Sequence[str] = (),
    import_mode: str = "prepend",
) -> None:
    collection = _Collection()
    code = pytest.main(
        [
            "--collect-only",
            "-q",
            "-p",
            "no:xdist",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--rootdir",
            str(root),
            f"--import-mode={import_mode}",
            *pytest_args,
            *tests,
        ],
        plugins=[collection],
    )
    if code != 0:
        raise RuntimeError(f"pytest collection failed: {code}")
    payload = {
        "namespace": "pytest",
        "collection_roots": list(tests),
        "source_hashes": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(collection.paths)
        },
        "cases": sorted(collection.cases, key=lambda case: case["nodeid"]),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def select_cases(
    root: Path, inventory: dict, args: argparse.Namespace
) -> CaseSelection:
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    if args.case_limit is not None and args.case_limit <= 0:
        raise ValueError("Case limit must be positive")
    cases = tuple(
        case
        for case in inventory["cases"]
        if args.node_filter in case["nodeid"]
        and int(case["case_id"], 16) % args.shard_count == args.shard_index
    )
    if args.case_limit is not None:
        cases = cases[: args.case_limit]
    return CaseSelection(root=root, inventory=inventory, cases=cases)


def _text(value: str | bytes | None) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def run_command(
    command: Sequence[str], log: Path, env: Mapping[str, str], cwd: Path, timeout: int
) -> int:
    child_env = dict(env)
    python_path = os.pathsep.join(path for path in sys.path if path)
    child_env["PYTHONPATH"] = python_path
    child_env["PAR_APPEND_PYTHONPATH"] = os.pathsep.join(
        path
        for path in (child_env.get("PAR_APPEND_PYTHONPATH", ""), python_path)
        if path
    )
    terminated = False
    try:
        result = run_process_group(
            command,
            cwd=str(cwd),
            env=child_env,
            timeout=timeout,
            capture_output=True,
            text=True,
            kill_grace_s=1.0,
            terminate_on_sigterm=True,
        )
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except ProcessTerminated as error:
        terminated = True
        code, stdout, stderr = 128 + signal.SIGTERM, error.output, error.stderr
    except subprocess.TimeoutExpired as error:
        code, stdout, stderr = 124, error.output, error.stderr
    log.write_text(_text(stdout) + _text(stderr))
    if terminated:
        raise SystemExit(code)
    return code


def prepare_selection(
    args: argparse.Namespace,
    root: Path,
    output: Path,
    env: Mapping[str, str],
    collection_command: Sequence[str],
) -> CaseSelection:
    inventory_path = output / "inventory.json"
    code = run_command(collection_command, output / "collection.log", env, root, 3600)
    if code != 0:
        raise RuntimeError(f"pytest collection failed: {code}; see collection.log")
    selection = select_cases(root, json.loads(inventory_path.read_text()), args)
    inventory = {
        **selection.inventory,
        "selected_case_ids": [case["case_id"] for case in selection.cases],
    }
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    return selection


def _pytest_arguments(
    args: argparse.Namespace, selection: CaseSelection, case: dict
) -> list[str]:
    return [
        "--rootdir",
        str(selection.root),
        "--nodeid",
        case["nodeid"],
        "--import-mode",
        args.import_mode,
        "--test-failures",
        *[f"--pytest-arg={arg}" for arg in args.pytest_arg],
    ]


def _worker_command(
    args: argparse.Namespace, selection: CaseSelection, case: dict, output: Path
) -> list[str]:
    return [
        project_python(output),
        str(output / "pytest-oracle.py"),
        "--worker-report",
        str(output / case["case_id"] / "pytest.json"),
        *_pytest_arguments(args, selection, case),
    ]


def _stress_options(args: argparse.Namespace) -> list[str]:
    modes = list(get_modes()) if args.strategy == "all" else args.strategy.split(",")
    if len(modes) == 1:
        return [
            "--delay-ladder-ns",
            args.delay_ladder,
            "--attempts-per-delay",
            str(args.attempts),
        ]
    return [
        value
        for mode in modes
        for value in (
            "--mode-option",
            f"{mode}.delay_ladder_ns={args.delay_ladder}",
            "--mode-option",
            f"{mode}.attempts_per_delay={args.attempts}",
        )
    ]


def _stress_command(
    args: argparse.Namespace,
    selection: CaseSelection,
    case: dict,
    output: Path,
    kernels: str,
) -> list[str]:
    return [
        project_python(output),
        "-m",
        "cutracer.cli",
        "stress",
        *args.cu_arg,
        "--strategy",
        args.strategy,
        "--baseline-runs",
        str(args.baseline_runs),
        "--budget-s",
        str(args.budget),
        "--timeout",
        str(args.timeout),
        *_stress_options(args),
        "--not-interesting-exit-codes",
        "1",
        "--kernel-filters",
        kernels,
        "--output-dir",
        str(output / case["case_id"]),
        "--",
        project_python(output),
        str(output / "pytest-oracle.py"),
        "--python",
        project_python(output),
        *_pytest_arguments(args, selection, case),
        "--kernels",
        kernels,
        "--output-dir",
        str(output / case["case_id"] / "oracle"),
    ]


def _trace_command(
    args: argparse.Namespace,
    selection: CaseSelection,
    case: dict,
    output: Path,
    kernels: str,
) -> list[str]:
    return [
        project_python(output),
        "-m",
        "cutracer.cli",
        "trace",
        "--instrument",
        "opcode_only",
        "--trace-format",
        "ndjson",
        "--kernel-timeout-s",
        str(args.timeout),
        "--no-data-timeout-s",
        "0",
        *args.cu_arg,
        "--kernel-filters",
        kernels,
        "--output-dir",
        str(output / case["case_id"] / "trace"),
        "--",
        *_worker_command(args, selection, case, output),
    ]


def inspect_trace(path: Path, kernels: str) -> dict | None:
    with path.open() as source:
        metadata = json.loads(source.readline())
        if metadata.get("type") != "kernel_metadata":
            return None
        names = (metadata.get("mangled_name", ""), metadata.get("unmangled_name", ""))
        if not any(
            part in name for part in kernels.split(",") if part for name in names
        ):
            return None
        modes = set(metadata.get("instrument_modes", []))
        launch = metadata.get("grid_launch_id")
        records, completion, wrong_launch = 0, None, False
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "capture_completion":
                completion = record
            elif record.get("type") in modes:
                records += 1
                wrong_launch |= record.get("grid_launch_id") != launch
    validation = validate_json_trace(path)
    complete = bool(
        validation["valid"]
        and records > 0
        and not wrong_launch
        and all(value > 0 for value in metadata.get("grid", []))
        and all(value > 0 for value in metadata.get("block", []))
        and completion
        and completion.get("grid_launch_id") == launch
        and completion.get("status") == "complete"
        and completion.get("kernel_completed") is True
        and completion.get("channel_drained") is True
        and completion.get("trace_records_written") == records
        and completion.get("dropped_records") == 0
        and not completion.get("errors")
    )
    return {
        "file": str(path),
        "kernel": metadata.get("unmangled_name"),
        "checksum": metadata.get("kernel_checksum"),
        "launch": launch,
        "instrument_modes": sorted(modes),
        "records": records,
        "complete": complete,
        "validation": validation,
    }


def _trace_result(
    args: argparse.Namespace,
    selection: CaseSelection,
    case: dict,
    directory: Path,
    code: int,
    kernels: str,
) -> dict:
    report = json.loads((directory / "pytest.json").read_text())
    with pytest_oracle() as oracle:
        test_code, reason = oracle.classify(report, case["nodeid"], test_failures=True)
    traces = [
        item
        for path in sorted((directory / "trace").glob("*.ndjson"))
        if (item := inspect_trace(path, kernels)) is not None
    ]
    for trace in traces:
        trace["file"] = str(Path(trace["file"]).relative_to(directory.parent))
    measured = bool(traces) and all(item["complete"] for item in traces)
    if not measured and test_code == 1:
        reason = "No complete target instruction capture; see trace evidence"
    if code != report.get("pytest_exit"):
        reason = (
            f"Trace command exited {code}, pytest reported {report.get('pytest_exit')}"
        )
    status = "incomplete"
    if code == report.get("pytest_exit") and measured:
        status = (
            "traced"
            if test_code == 1
            else "test_failed"
            if test_code == 0
            else "incomplete"
        )
    return {
        "status": status,
        "test_outcome": "passed"
        if test_code == 1
        else "failed"
        if test_code == 0
        else "incomplete",
        "reason": reason,
        "traces": traces,
        "capture_complete": measured,
    }


def _stress_result(directory: Path, code: int) -> dict:
    path = directory / "stress_report.json"
    payload = json.loads(path.read_text()) if path.is_file() else None
    outcomes = [
        json.loads(path.read_text())
        for path in sorted((directory / "oracle").glob("attempt-*/outcome.json"))
    ]
    return {
        "status": "incomplete"
        if payload is None or code != 0
        else payload["verdict"]["status"],
        "report": f"{directory.name}/stress_report.json" if payload else None,
        "oracle_outcomes": outcomes,
        "failure_scope": "pytest test failure, not an inferred numerical race",
    }


def _execute_action(
    args: argparse.Namespace,
    selection: CaseSelection,
    case: dict,
    output: Path,
    env: Mapping[str, str],
    kernels: str,
) -> int:
    directory = output / case["case_id"]
    if args.action == "trace":
        (directory / "trace").mkdir()
        command = _trace_command(args, selection, case, output, kernels)
        timeout = args.timeout
    else:
        command = _stress_command(args, selection, case, output, kernels)
        timeout = args.budget + args.timeout + 30
    return run_command(command, directory / "run.log", env, selection.root, timeout)


def execute_case(
    args: argparse.Namespace,
    selection: CaseSelection,
    case: dict,
    output: Path,
    env: Mapping[str, str],
) -> dict:
    directory = output / case["case_id"]
    directory.mkdir()
    started = time.monotonic()
    row = {**case, "action": args.action}
    try:
        filters = args.kernel_filters.split(",") if args.kernel_filters else []
        if not filters:
            return {**row, "status": "incomplete", "reason": "No target kernel filters"}
        kernels = ",".join(filters)
        row["kernel_filters"] = filters
        code = _execute_action(args, selection, case, output, env, kernels)
        row.update(
            returncode=code,
            elapsed_s=round(time.monotonic() - started, 1),
        )
        result = (
            _trace_result(args, selection, case, directory, code, kernels)
            if args.action == "trace"
            else _stress_result(directory, code)
        )
        return {**row, **result}
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        AttributeError,
    ) as error:
        return {
            **row,
            "status": "incomplete",
            "reason": f"{type(error).__name__}: {error}",
        }


def _save_results(
    output: Path, selection: CaseSelection, rows: list[dict], action: str
) -> None:
    complete_states = (
        ("clean", "finding") if action == "stress" else ("traced", "test_failed")
    )
    finished = {row["case_id"] for row in rows if row["status"] in complete_states}
    success = "clean" if action == "stress" else "traced"
    payload = {
        "action": action,
        "inventory_cases": len(selection.inventory["cases"]),
        "selected_cases": len(selection.cases),
        "result_cases": len(rows),
        "selection_covers_inventory": len(selection.cases)
        == len(selection.inventory["cases"]),
        "complete_for_selected_cases": bool(selection.cases)
        and len(finished) == len(selection.cases),
        "all_selected_successful": bool(selection.cases)
        and len(rows) == len(selection.cases)
        and all(row["status"] == success for row in rows),
        "required_missing": [
            case["case_id"]
            for case in selection.cases
            if case["case_id"] not in finished
        ],
        "scope": "Collected pytest cases with target instrumentation, not all library kernels or numerical oracle coverage",
    }
    (output / "coverage.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "case-results.json").write_text(json.dumps(rows, indent=2) + "\n")


def execute_selection(
    args: argparse.Namespace,
    selection: CaseSelection,
    output: Path,
    env: Mapping[str, str],
) -> int:
    if args.action == "stress" and args.baseline_runs < 1:
        raise ValueError("Stress requires at least one control run")
    oracle_source = resources.files("cutracer.oracle").joinpath("pytest_oracle.py")
    (output / "pytest-oracle.py").write_text(oracle_source.read_text())
    rows: list[dict] = []
    _save_results(output, selection, rows, args.action)
    for case in selection.cases:
        rows.append(execute_case(args, selection, case, output, env))
        _save_results(output, selection, rows, args.action)
        logger.info(
            "pytest action=%s case=%s status=%s",
            args.action,
            case["nodeid"],
            rows[-1]["status"],
        )
    complete_states = (
        ("clean", "finding") if args.action == "stress" else ("traced", "test_failed")
    )
    if not rows or any(row["status"] not in complete_states for row in rows):
        return 2
    return (
        1
        if (
            args.action == "trace"
            and any(row["status"] == "test_failed" for row in rows)
        )
        or (
            getattr(args, "fail_on_findings", False)
            and any(row["status"] == "finding" for row in rows)
        )
        else 0
    )


def run_workload(
    args: argparse.Namespace,
    root: Path,
    output: Path,
    collection_command: Sequence[str],
) -> int:
    if not args.select_only and args.action is None:
        raise ValueError("Specify --action stress or --action trace")
    output.mkdir(parents=True, exist_ok=True)
    selection = prepare_selection(args, root, output, os.environ, collection_command)
    run = {
        "source_revision": args.source_revision,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    if args.select_only:
        return 0
    return execute_selection(args, selection, output, os.environ)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action", choices=("stress", "trace"))
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Return exit 1 for measured stress findings without discarding complete results.",
    )
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=os.environ.get("WORKLOAD_OUTPUT_DIR")
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--node-filter", default="")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--kernel-filters")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=["-p", "no:timeout", "-p", "no:xdist", "-o", "addopts=", "--tb=short"],
    )
    parser.add_argument(
        "--import-mode", choices=("prepend", "append", "importlib"), default="prepend"
    )
    parser.add_argument("--cu-arg", action="append", default=[])
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--delay-ladder", default="1000,5000,10000,50000,100000")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--baseline-runs", type=int, default=3)
    parser.add_argument("--collect-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inventory", type=Path, help=argparse.SUPPRESS)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select pytest cases, then run CUTracer stress or trace."
    )
    add_arguments(parser)
    parser.add_argument(
        "--rootdir",
        type=Path,
        default=Path(os.environ.get("WORKLOAD_SOURCE_DIR", os.getcwd())),
    )
    parser.add_argument("--tests", nargs="+", required=True)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.collect_root is not None:
        if args.inventory is None:
            parser.error("--collect-root requires --inventory")
        collect_inventory(
            args.collect_root,
            args.tests,
            args.inventory,
            pytest_args=args.pytest_arg,
            import_mode=args.import_mode,
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir or WORKLOAD_OUTPUT_DIR is required")
    if not args.select_only and not args.kernel_filters:
        parser.error("--kernel-filters is required for execution")
    output = args.output_dir.resolve()
    root = args.rootdir.resolve()
    command = [
        project_python(),
        "-m",
        "cutracer.pytest_workload",
        "--collect-root",
        str(root),
        "--inventory",
        str(output / "inventory.json"),
        "--tests",
        *args.tests,
        "--import-mode",
        args.import_mode,
        *[f"--pytest-arg={arg}" for arg in args.pytest_arg],
    ]
    raise SystemExit(run_workload(args, root, output, command))


if __name__ == "__main__":
    main()
