# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Pure decoder for supported AMDGCN LDS forms and workgroup barriers."""

from __future__ import annotations

import re
from dataclasses import dataclass


_RELEVANT_MNEMONIC_RE = re.compile(
    r"\b(?P<mnemonic>ds_[a-z0-9_]+|s_barrier(?:_[a-z0-9_]+)?)\b",
    re.IGNORECASE,
)
_VGPR_RE = re.compile(
    r"(?<![A-Za-z0-9_])v(?:\[(?P<range_start>\d+):\d+\]|(?P<single>\d+))",
    re.IGNORECASE,
)
_OFFSET_RE = re.compile(
    r"\boffset(?P<slot>[01]?):\s*(?P<value>0x[0-9a-f]+|\d+)",
    re.IGNORECASE,
)
_GDS_MODIFIER_RE = re.compile(r"\bgds\b", re.IGNORECASE)


@dataclass(frozen=True)
class AccessRange:
    """One byte range encoded by a supported DS instruction."""

    offset_bytes: int
    width_bytes: int


@dataclass(frozen=True)
class DecodedInstruction:
    """Static classification of one relevant AMDGPU instruction."""

    kind: str
    supported: bool
    address_vgpr: str | None = None
    accesses: tuple[AccessRange, ...] = ()


def _vgprs_in_order(operand_text: str) -> list[str]:
    registers: list[str] = []
    for match in _VGPR_RE.finditer(operand_text):
        number = match.group("range_start") or match.group("single")
        registers.append(f"v{int(number)}")
    return registers


def _encoded_offsets(operand_text: str) -> dict[str, int]:
    return {
        match.group("slot"): int(match.group("value"), 0)
        for match in _OFFSET_RE.finditer(operand_text)
    }


def _unsupported() -> DecodedInstruction:
    return DecodedInstruction(kind="unsupported", supported=False)


def decode_instruction(asm: str) -> DecodedInstruction | None:
    """Decode a relevant AMD LDS access or workgroup barrier.

    Returns ``None`` for instructions outside this decoder's LDS/barrier
    scope.  Returns a ``DecodedInstruction`` with ``supported=False`` for an
    in-scope DS or scalar-barrier mnemonic whose semantics are not implemented.
    Callers must surface those unsupported instructions in collection status.
    """

    mnemonic_match = _RELEVANT_MNEMONIC_RE.search(asm)
    if mnemonic_match is None:
        return None

    mnemonic = mnemonic_match.group("mnemonic").lower()
    if mnemonic == "s_barrier":
        return DecodedInstruction(
            kind="barrier",
            supported=True,
        )
    if mnemonic.startswith("s_barrier"):
        return _unsupported()

    operand_text = asm[mnemonic_match.end() :]
    if _GDS_MODIFIER_RE.search(operand_text):
        return _unsupported()

    if mnemonic == "ds_bpermute_b32":
        return DecodedInstruction(
            kind="ignored",
            supported=True,
        )

    if mnemonic == "ds_read_b128":
        kind, address_index = "read", 1
        width_bytes, offset_slots, offset_scale = 16, ("",), 1
    elif mnemonic == "ds_write2_b32":
        kind, address_index = "write", 0
        width_bytes, offset_slots, offset_scale = 4, ("0", "1"), 4
    else:
        return _unsupported()

    vgprs = _vgprs_in_order(operand_text)
    if len(vgprs) <= address_index:
        return _unsupported()

    offsets = _encoded_offsets(operand_text)
    if any(slot not in offset_slots for slot in offsets):
        return _unsupported()
    accesses = tuple(
        AccessRange(
            offset_bytes=offsets.get(slot, 0) * offset_scale,
            width_bytes=width_bytes,
        )
        for slot in offset_slots
    )

    return DecodedInstruction(
        kind=kind,
        supported=True,
        address_vgpr=vgprs[address_index],
        accesses=accesses,
    )
