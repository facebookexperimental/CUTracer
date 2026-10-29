# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Pure, deterministic identities for kernel-discovery children."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence

from cutracer.service.contracts import KernelId, WorkUnit


def _child_unit(parent: WorkUnit, kernel: KernelId) -> WorkUnit:
    identity = json.dumps(
        {
            "parent_unit_id": parent.unit_id,
            "kernel": {"name": kernel.name, "cubin_hash": kernel.cubin_hash},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    suffix = hashlib.sha256(identity).hexdigest()[:16]
    return dataclasses.replace(
        parent,
        unit_id=f"{parent.unit_id}-kernel-{suffix}",
        kernel=kernel,
        kernel_filters=[],
    )


def pin_discovered_work_units(
    target: WorkUnit,
    kernels: Sequence[KernelId],
) -> tuple[WorkUnit, ...]:
    """Create deterministic kernel-pinned children from one discovery target."""
    if target.kernel is not None:
        raise ValueError("cannot fan out an already pinned WorkUnit")
    return tuple(_child_unit(target, kernel) for kernel in kernels)


def discovery_child_session_id(parent_session_id: str, unit: WorkUnit) -> str:
    """Return a bounded stable session id for one discovered child."""
    unit_identity = unit.to_dict()
    serialized_kernel = unit_identity.get("kernel")
    if isinstance(serialized_kernel, dict):
        serialized_kernel.pop("mangled_name", None)
    identity = json.dumps(
        {"parent_session_id": parent_session_id, "unit": unit_identity},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    suffix = hashlib.sha256(identity).hexdigest()[:20]
    return f"{parent_session_id[:226]}-kernel-{suffix}"
