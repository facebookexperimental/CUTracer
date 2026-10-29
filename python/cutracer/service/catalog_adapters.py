"""Expansion of catalog entries into portable diagnosis work units."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Protocol

from cutracer.service.catalog import (
    CatalogAdapter,
    CatalogSamplingPolicy,
    WorkloadCatalog,
    WorkloadCatalogEntry,
)
from cutracer.service.contracts import WorkUnit


@dataclasses.dataclass(frozen=True)
class CatalogWorkUnit:
    """A WorkUnit together with the scheduling policy that selected it."""

    catalog_revision: str
    entry_id: str
    selector_index: int
    kernel_filters: tuple[str, ...]
    sampling: CatalogSamplingPolicy
    cadence_days: int
    estimated_gpu_seconds: int
    unit: WorkUnit


class WorkloadCatalogAdapter(Protocol):
    def expand(
        self,
        entry: WorkloadCatalogEntry,
        *,
        catalog_revision: str,
    ) -> tuple[CatalogWorkUnit, ...]: ...


def _unit_id(
    entry: WorkloadCatalogEntry, argv: list[str], arch: str, selector_index: int
) -> str:
    identity = json.dumps(
        {
            "adapter": entry.adapter.value,
            "arch": arch,
            "argv": argv,
            "cwd": entry.cwd,
            "entry_id": entry.entry_id,
            "env": dict(entry.env),
            "kernel_filters": list(entry.kernel_filters),
            "oracle": None if entry.oracle is None else entry.oracle.to_dict(),
            "selector_index": selector_index,
            "source_revision": entry.source_revision,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"{entry.entry_id}-{hashlib.sha256(identity).hexdigest()[:16]}"


def _pytest_argv(
    entry: WorkloadCatalogEntry,
    selector: str,
    python_argv: tuple[str, ...],
) -> list[str]:
    return [
        *python_argv,
        "-p",
        "no:xdist",
        selector,
        *entry.extra_args,
    ]


@dataclasses.dataclass(frozen=True)
class PytestCatalogAdapter:
    adapter: CatalogAdapter
    python_argv: tuple[str, ...] = ("python", "-m", "pytest")

    def __post_init__(self) -> None:
        if self.adapter not in (
            CatalogAdapter.TRITON_PYTEST,
            CatalogAdapter.TLX_PYTEST,
        ):
            raise ValueError(
                f"unsupported pytest catalog adapter: {self.adapter.value}"
            )
        if not self.python_argv or any(not item.strip() for item in self.python_argv):
            raise ValueError("pytest adapter command must not be empty")

    def expand(
        self,
        entry: WorkloadCatalogEntry,
        *,
        catalog_revision: str,
    ) -> tuple[CatalogWorkUnit, ...]:
        if entry.adapter != self.adapter:
            raise ValueError(
                f"catalog entry {entry.entry_id!r} requires {entry.adapter.value}, "
                f"not {self.adapter.value}"
            )
        if not entry.selectors or not entry.architectures:
            raise ValueError(
                f"catalog entry {entry.entry_id!r} has no selectors or architectures"
            )
        targets: list[CatalogWorkUnit] = []
        for selector_index, selector in enumerate(entry.selectors):
            for arch in entry.architectures:
                argv = _pytest_argv(entry, selector, self.python_argv)
                unit = WorkUnit(
                    unit_id=_unit_id(entry, argv, arch, selector_index),
                    argv=argv,
                    workload=entry.adapter.value,
                    test_id=selector,
                    cwd=entry.cwd,
                    arch=arch,
                    source_revision=entry.source_revision,
                    oracle=entry.oracle,
                    env=dict(entry.env),
                )
                targets.append(
                    CatalogWorkUnit(
                        catalog_revision=catalog_revision,
                        entry_id=entry.entry_id,
                        selector_index=selector_index,
                        kernel_filters=entry.kernel_filters,
                        sampling=entry.sampling,
                        cadence_days=entry.cadence_days,
                        estimated_gpu_seconds=entry.estimated_gpu_seconds,
                        unit=unit,
                    )
                )
        return tuple(targets)


_TRITONBENCH_FORBIDDEN_OPTIONS = frozenset(
    {
        "--bypass-fail",
        "--metrics",
        "--num-inputs",
        "--op",
        "--rep",
        "--repcnt",
        "--test-only",
        "--warmup",
    }
)


def _option_name(argument: str) -> str:
    return argument.partition("=")[0]


def _tritonbench_argv(
    entry: WorkloadCatalogEntry,
    selector: str,
    target_argv: tuple[str, ...],
) -> list[str]:
    forbidden = [
        argument
        for argument in entry.extra_args
        if _option_name(argument) in _TRITONBENCH_FORBIDDEN_OPTIONS
    ]
    if forbidden:
        raise ValueError(
            "tritonbench catalog extra_args override guarded options: "
            f"{', '.join(forbidden)}"
        )
    return [
        *target_argv,
        "--",
        "--op",
        selector,
        "--metrics",
        "accuracy",
        "--num-inputs",
        "1",
        "--test-only",
        "--isolate",
        *entry.extra_args,
    ]


@dataclasses.dataclass(frozen=True)
class TritonbenchCatalogAdapter:
    target_argv: tuple[str, ...] = (
        "buck2",
        "--client-metadata",
        "id=cutracer_service",
        "run",
        "@mode/opt",
        "fbcode//pytorch/tritonbench:run",
    )

    def __post_init__(self) -> None:
        if not self.target_argv or any(not item.strip() for item in self.target_argv):
            raise ValueError("tritonbench adapter command must not be empty")

    def expand(
        self,
        entry: WorkloadCatalogEntry,
        *,
        catalog_revision: str,
    ) -> tuple[CatalogWorkUnit, ...]:
        if entry.adapter != CatalogAdapter.TRITONBENCH:
            raise ValueError(
                f"catalog entry {entry.entry_id!r} is not a tritonbench entry"
            )
        targets: list[CatalogWorkUnit] = []
        for selector_index, selector in enumerate(entry.selectors):
            for arch in entry.architectures:
                argv = _tritonbench_argv(entry, selector, self.target_argv)
                unit = WorkUnit(
                    unit_id=_unit_id(entry, argv, arch, selector_index),
                    argv=argv,
                    workload=entry.adapter.value,
                    test_id=selector,
                    cwd=entry.cwd,
                    arch=arch,
                    source_revision=entry.source_revision,
                    oracle=entry.oracle,
                    env=dict(entry.env),
                )
                targets.append(
                    CatalogWorkUnit(
                        catalog_revision=catalog_revision,
                        entry_id=entry.entry_id,
                        selector_index=selector_index,
                        kernel_filters=entry.kernel_filters,
                        sampling=entry.sampling,
                        cadence_days=entry.cadence_days,
                        estimated_gpu_seconds=entry.estimated_gpu_seconds,
                        unit=unit,
                    )
                )
        return tuple(targets)


def pytest_catalog_adapters() -> dict[CatalogAdapter, WorkloadCatalogAdapter]:
    return {
        kind: PytestCatalogAdapter(kind)
        for kind in (CatalogAdapter.TRITON_PYTEST, CatalogAdapter.TLX_PYTEST)
    }


def default_catalog_adapters() -> dict[CatalogAdapter, WorkloadCatalogAdapter]:
    return {
        **pytest_catalog_adapters(),
        CatalogAdapter.TRITONBENCH: TritonbenchCatalogAdapter(),
    }


def expand_workload_catalog(
    catalog: WorkloadCatalog,
    *,
    adapters: Mapping[CatalogAdapter, WorkloadCatalogAdapter] | None = None,
) -> tuple[CatalogWorkUnit, ...]:
    resolved_adapters = default_catalog_adapters() if adapters is None else adapters
    targets: list[CatalogWorkUnit] = []
    for entry in catalog.entries:
        if not entry.enabled:
            continue
        adapter = resolved_adapters.get(entry.adapter)
        if adapter is None:
            raise ValueError(f"no workload adapter for {entry.adapter.value}")
        targets.extend(adapter.expand(entry, catalog_revision=catalog.catalog_revision))
    return tuple(targets)
