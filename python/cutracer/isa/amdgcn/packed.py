# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Decode packed FP32 instructions and check observed results exactly.

Terms, for gfx950 (64 lanes per wave):
- VGPR: a vector register with one 32-bit word per lane, so `v18` is 64 words.
  A VGPR pair such as `v[18:19]` holds one packed value: `v18` is its low half
  and `v19` its high half. SGPRs (one word for the whole wave) are not modelled.
- DWORD: one 32-bit half of a pair.
- `op_sel[k]` / `op_sel_hi[k]`: which DWORD of source k (0 low, 1 high) feeds the
  low / high result.
- `neg_lo[k]` / `neg_hi[k]`: negate source k for the low / high result.
So Hongtao's `op_sel:[0,1] neg_lo:[0,1]` computes low = src0.lo - src1.hi.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

# Modelled packed FP32 opcodes, with the number of source VGPR pairs each reads.
PACKED_F32 = {"v_pk_add_f32": 2, "v_pk_mul_f32": 2, "v_pk_fma_f32": 3}

_PAIR = re.compile(r"v\[(\d+):(\d+)\]")
_MODIFIER = re.compile(r"\b(op_sel_hi|op_sel|neg_lo|neg_hi):\[([01](?:,[01])*)\]")


@dataclass(frozen=True)
class PackedInstruction:
    opcode: str
    destination: tuple[str, str]
    sources: tuple[tuple[str, str], ...]
    op_sel: tuple[int, ...]
    op_sel_hi: tuple[int, ...]
    neg_lo: tuple[int, ...]
    neg_hi: tuple[int, ...]

    @property
    def registers(self) -> tuple[str, ...]:
        """Every source DWORD, then the destination: what a snapshot must read.

        Example, for the subtraction in `decode_packed`'s example:
            ("v8", "v9", "v12", "v13", "v18", "v19")
        """
        pairs = (*self.sources, self.destination)
        return tuple(dict.fromkeys(name for pair in pairs for name in pair))

    def overwrites_input(self) -> bool:
        """True if the destination is also a source, so a later snapshot has lost it.

        Example:
            v_pk_add_f32 v[18:19], v[8:9], v[12:13] ...  -> False
            v_pk_fma_f32 v[0:1], v[0:1], v[2:3], v[4:5]  -> True
        """
        inputs = {name for pair in self.sources for name in pair}
        return bool(inputs & set(self.destination))


def decode_packed(asm: str) -> PackedInstruction | None:
    """Decode a packed FP32 instruction; return None for any other opcode.

    Only VGPR-pair operands and the four selection modifiers are modelled;
    other packed forms raise ValueError instead of being guessed.

    Example, the subtraction from Hongtao's gfx950 reproducer:
        decode_packed(
            "v_pk_add_f32 v[18:19], v[8:9], v[12:13] op_sel:[0,1] neg_lo:[0,1] neg_hi:[0,1]"
        )
        -> PackedInstruction(opcode="v_pk_add_f32", destination=("v18", "v19"),
               sources=(("v8", "v9"), ("v12", "v13")), op_sel=(0, 1),
               op_sel_hi=(1, 1), neg_lo=(0, 1), neg_hi=(0, 1))
        decode_packed("v_add_f32 v1, v2, v3")  -> None
        decode_packed("v_pk_add_f32 v[0:1], s[0:1], v[2:3]")  -> ValueError
    """
    opcode, _, rest = " ".join(asm.split()).partition(" ")
    count = PACKED_F32.get(opcode)
    if count is None:
        return None
    modifiers = {
        name: tuple(int(bit) for bit in bits.split(","))
        for name, bits in _MODIFIER.findall(rest)
    }
    fields = [field.strip() for field in _MODIFIER.sub("", rest).split(",")]
    pairs = [_PAIR.fullmatch(field) for field in fields]
    if len(fields) != count + 1 or any(
        pair is None or int(pair[2]) != int(pair[1]) + 1 for pair in pairs
    ):
        raise ValueError(f"unmodelled {opcode} operands: {rest}")
    if any(len(bits) != count for bits in modifiers.values()):
        raise ValueError(f"{opcode} modifiers must select {count} sources")
    destination, *sources = (
        (f"v{pair[1]}", f"v{pair[2]}") for pair in pairs if pair is not None
    )
    return PackedInstruction(
        opcode,
        destination,
        tuple(sources),
        modifiers.get("op_sel", (0,) * count),
        modifiers.get("op_sel_hi", (1,) * count),
        modifiers.get("neg_lo", (0,) * count),
        modifiers.get("neg_hi", (0,) * count),
    )


def fp32(word: int) -> float:
    """Reinterpret a 32-bit register word as FP32.

    Example: fp32(0x431E0000) -> 158.0; fp32(0xC3F70000) -> -494.0
    """
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _exact(word: int) -> Fraction | None:
    """Exact value of a normal or zero FP32 word, else None (subnormal, inf, NaN).

    Example: _exact(0x431E0000) -> Fraction(158, 1); _exact(0x00000001) -> None
    """
    magnitude = word & 0x7FFFFFFF
    if magnitude and not 0x00800000 <= magnitude < 0x7F800000:
        return None
    return Fraction(fp32(word))


def allowed_results(opcode: str, words: Sequence[int]) -> frozenset[int] | None:
    """Return the FP32 values bracketing the exact result of one packed half.

    Accepting both neighbours avoids assuming a rounding mode. Subnormal,
    infinite and NaN inputs or results return None; the sign of zero is not
    checked.

    Example, `words` are the selected inputs after negation; lane 48's low half
    of the subtraction adds 158.0 and -652.0:
        allowed_results("v_pk_add_f32", [0x431E0000, 0xC4230000])
        -> frozenset({0xC3F70000})  # exactly -494.0
        allowed_results("v_pk_add_f32", [0x3F800000, 0x33800000])
        -> frozenset({0x3F800000, 0x3F800001})  # 1 + 2**-24 lies between them
        allowed_results("v_pk_add_f32", [0x7FC00000, 0x3F800000])  -> None  # NaN
    """
    values = []
    for word in words:
        value = _exact(word)
        if value is None:
            return None
        values.append(value)
    if opcode == "v_pk_add_f32":
        exact = values[0] + values[1]
    elif opcode == "v_pk_mul_f32":
        exact = values[0] * values[1]
    else:
        exact = values[0] * values[1] + values[2]
    if exact == 0:
        return frozenset((0, 0x80000000))
    if not Fraction(2) ** -126 <= abs(exact) <= Fraction(fp32(0x7F7FFFFF)):
        return None
    nearest = struct.unpack("<I", struct.pack("<f", float(exact)))[0]
    rounded = Fraction(fp32(nearest))
    if rounded == exact:
        return frozenset((nearest,))
    toward = 1 if exact > rounded else -1
    return frozenset((nearest, nearest + (-toward if nearest >> 31 else toward)))


def selection_models(
    opcode: str,
    words: Sequence[int],
    unselected: Sequence[int],
    negations: Sequence[int],
    observed: int,
    allowed: frozenset[int],
) -> list[str]:
    """Name single-source substitutions that reproduce a wrong result exactly.

    `srcK_other_dword` uses the DWORD of source K that op_sel did not select;
    `srcK_zero` replaces source K with zero. Several models can coincide.

    Example, lane 48's low half: inputs 158.0 and -652.0, unselected DWORDs
    src0.hi = 301.0 and src1.lo = 0.0, neg_lo = (0, 1), and the lane got 158.0:
        selection_models("v_pk_add_f32", [0x431E0000, 0xC4230000],
                         [0x43968000, 0x00000000], [0, 1], 0x431E0000,
                         frozenset({0xC3F70000}))
        -> ["src1_other_dword", "src1_zero"]
    Both give 158.0 + -0.0 = 158.0, because src1.lo is 0.0.
    """
    models = []
    for index, other in enumerate(unselected):
        for name, word in ((f"src{index}_other_dword", other), (f"src{index}_zero", 0)):
            changed = list(words)
            changed[index] = word ^ (negations[index] << 31)
            bounds = allowed_results(opcode, changed)
            if bounds is not None and observed in bounds and not bounds & allowed:
                models.append(name)
    return models


def check_packed(
    instruction: PackedInstruction, snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    """Check both result halves in each active lane of a snapshot.

    The caller guarantees that the snapshot was taken directly after the
    instruction and that the instruction does not overwrite its inputs, so the
    recorded sources are the values it read.

    Example, the subtraction from `decode_packed` with a snapshot of the form
    {"registers": {"v8": [64 words], ...}, "exec": 1 << 48}, where lane 48 holds
    v8 = 158.0, v9 = 301.0, v12 = 0.0, v13 = 652.0, v18 = 158.0 (wrong) and
    v19 = -351.0 (right):
        -> {"checked": 2, "excluded": 0, "violations": [
               {"lane": 48, "half": 0, "register": "v18",
                "inputs": ["0x431e0000", "0xc4230000"], "observed": "0x431e0000",
                "allowed": ["0xc3f70000"], "models": ["src1_other_dword", "src1_zero"]}]}
    """
    registers, active = snapshot["registers"], snapshot["exec"]
    result: dict[str, Any] = {"checked": 0, "excluded": 0, "violations": []}
    for half, destination in enumerate(instruction.destination):
        selectors = instruction.op_sel_hi if half else instruction.op_sel
        negations = instruction.neg_hi if half else instruction.neg_lo
        for lane in range(64):
            if not active >> lane & 1:
                continue
            words = [
                registers[pair[selectors[index]]][lane] ^ (negations[index] << 31)
                for index, pair in enumerate(instruction.sources)
            ]
            allowed = allowed_results(instruction.opcode, words)
            if allowed is None:
                result["excluded"] += 1
                continue
            result["checked"] += 1
            observed = registers[destination][lane]
            if observed in allowed:
                continue
            unselected = [
                registers[pair[1 - selectors[index]]][lane]
                for index, pair in enumerate(instruction.sources)
            ]
            result["violations"].append(
                {
                    "lane": lane,
                    "half": half,
                    "register": destination,
                    "inputs": [f"0x{word:08x}" for word in words],
                    "observed": f"0x{observed:08x}",
                    "allowed": [f"0x{value:08x}" for value in sorted(allowed)],
                    "models": selection_models(
                        instruction.opcode,
                        words,
                        unselected,
                        negations,
                        observed,
                        allowed,
                    ),
                }
            )
    return result
