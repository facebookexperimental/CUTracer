# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Replay AMDGPU integer instructions with wrapping and with exact arithmetic.

Address arithmetic on AMD GPUs is often 32-bit: a 32-bit product is later
sign-extended into a 64-bit address (`v_ashrrev_i32 vhi, 31, vlo`). `Lane`
replays instructions for one lane with the hardware's wrapping arithmetic and,
alongside, with exact integers. When a computed 32-bit value is sign-extended
although its exact value does not fit 32 bits, the address the hardware
computes is not the intended one; `Lane.overflows` records each such widening
and the instruction that produced the value.

`MODELLED` lists the instructions replayed: those of the pre-fix gfx950
grouped GEMM's store address.

Terms:
- word: one 32-bit register, an SGPR (`s5`, the same in every lane) or one
  lane of a VGPR (`v5`). A pair such as `v[130:131]` is a 64-bit value, low
  word first.
- leaf: a word the replay did not compute; its bits come from the snapshot.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
_REGISTER = re.compile(r"([sv])(\d+)|([sv])\[(\d+):(\d+)\]")


def registers(token: str) -> list[str] | None:
    """Expand an SGPR or VGPR operand into 32-bit register names; None for others.

    Example: registers("v[130:131]") -> ["v130", "v131"];
    registers("s52") -> ["s52"]; registers("vcc") -> None; registers("31") -> None
    """
    match = _REGISTER.fullmatch(token)
    if match is None:
        return None
    if match[1]:
        return [f"{match[1]}{match[2]}"]
    return [f"{match[3]}{index}" for index in range(int(match[4]), int(match[5]) + 1)]


def _single(token: str) -> str:
    """The name of a 32-bit register operand; ValueError for anything else.

    Example: _single("v130") -> "v130"; _single("v[130:131]") raises ValueError
    """
    names = registers(token)
    if names is None or len(names) != 1:
        raise ValueError(f"unsupported 32-bit register {token!r}")
    return names[0]


def split(asm: str) -> tuple[str, list[str], list[str]]:
    """Split one instruction into opcode, operands and trailing modifiers.

    The `_e32`/`_e64` encoding suffix is dropped from the opcode.

    Example: split("global_load_ushort v191, v[130:131], off offset:16")
    -> ("global_load_ushort", ["v191", "v[130:131]", "off"], ["offset:16"]);
    split("v_ashrrev_i32_e32 v131, 31, v130") -> ("v_ashrrev_i32", ["v131", "31", "v130"], [])
    """
    opcode, _, rest = asm.strip().partition(" ")
    opcode = re.sub(r"_e(32|64)$", "", opcode)
    if not rest.strip():
        return opcode, [], []
    operands = [item.strip() for item in re.split(r",(?![^\[]*\])", rest)]
    last, *modifiers = operands[-1].split()
    return opcode, operands[:-1] + [last], modifiers


def immediate(token: str) -> int | None:
    """Parse an integer immediate; None for registers and float constants.

    Example: immediate("31") -> 31; immediate("0x43") -> 67; immediate("v2") -> None
    """
    try:
        return int(token, 0)
    except ValueError:
        return None


# Ops that write a second register destination (operand 1) besides operand 0:
# an add/subtract carry-out SGPR pair, or a 64-bit multiply-add's carry-out.
# They are not replayed, but the slice must know both are written.
_TWO_DESTINATIONS = (
    "v_add_co_u32",
    "v_sub_co_u32",
    "v_subrev_co_u32",
    "v_addc_co_u32",
    "v_subb_co_u32",
    "v_subbrev_co_u32",
    "v_add_co_ci_u32",
    "v_sub_co_ci_u32",
    "v_mad_u64_u32",
    "v_mad_i64_i32",
)


def effects(asm: str) -> tuple[set[str], set[str]]:
    """Return the SGPRs and VGPRs an instruction reads and writes.

    Operand 0 is the destination, with operand 1 a second destination for the
    carry-out ops in `_TWO_DESTINATIONS`. A store, an LDS-targeting load and an
    atomic with no return value write no register. `vcc` and `scc` are not
    tracked, so a carry held there is not listed.

    Example: effects("v_mul_lo_u32 v130, v139, s52") -> ({"v139", "s52"}, {"v130"});
    effects("v_mad_u64_u32 v[0:1], s[2:3], v4, v5, v[0:1]")
    -> ({"v4", "v5", "v0", "v1"}, {"v0", "v1", "s2", "s3"});
    effects("global_load_lds_dwordx4 v[2:3], off") -> ({"v2", "v3"}, set())
    """
    opcode, operands, modifiers = split(asm)
    names = [set(registers(token) or []) for token in operands]
    if not names:
        return set(), set()
    no_register_write = (
        "store" in opcode
        or opcode.startswith("ds_write")
        or "lds" in opcode
        or "lds" in modifiers
        or ("atomic" in opcode and not {"glc", "sc0"} & set(modifiers))
    )
    if no_register_write:
        return set().union(*names), set()
    destinations = 2 if opcode in _TWO_DESTINATIONS else 1
    writes = set().union(*names[:destinations])
    reads = set().union(*names[destinations:]) if names[destinations:] else set()
    return reads, writes


@dataclass
class Word:
    """One 32-bit value: the hardware's bits and its exact value.

    `kind` is "word" for an arithmetic result, whose `exact` may exceed 32 bits;
    "bits" for a bitwise result, which is always a valid 32-bit pattern;
    "pair_lo"/"pair_hi" for the halves of a 64-bit result, whose low word's
    `exact` holds the whole 64-bit value; or "sext" for a high word that
    sign-extends `ref`, whose `exact` is the sign word that `ref`'s exact value
    implies. `producer` is the index of the instruction that wrote it, None for
    a leaf.

    Example: `v_mul_lo_u32 v130, v139, s52` with v139 = 32768 and s52 = 65536
    writes Word(hw=0x80000000, exact=2147483648, producer=<its index>).
    """

    hw: int
    exact: int
    kind: str = "word"
    ref: Word | None = None
    producer: int | None = None

    def value(self, signed: bool = False) -> int:
        """The value a 32-bit operand read sees.

        An arithmetic result gives its exact value. Any other word gives its
        bits, as signed when `signed`: a leaf's exact value is unknown, and a
        bitwise result, a half of a 64-bit result and a sign word have no exact
        value beyond their 32-bit pattern.

        Example: Word(hw=0xffffffc0, exact=0xffffffc0).value(signed=True) -> -64
        """
        if self.producer is not None and self.kind == "word":
            return self.exact
        return self.hw - (1 << 32) if signed and self.hw >> 31 else self.hw


@dataclass
class Lane:
    """Replay state of one lane; `leaf(name)` reads a snapshot register or None.

    Example: Lane(leaf), then run(i, asm) for each instruction of an address
    computation, then pair("v[132:133]", at) -> (hardware address, exact address).
    """

    leaf: Callable[[str], int | None]
    words: dict[str, Word] = field(default_factory=dict)
    overflows: list[dict[str, Any]] = field(default_factory=list)

    def word(self, name: str) -> Word:
        """The register's current word, read from the snapshot on first use.

        Example: word("s52") -> Word(hw=65536, exact=65536) from the snapshot.
        """
        if name not in self.words:
            bits = self.leaf(name)
            if bits is None:
                raise LookupError(f"{name} is not in the snapshot")
            self.words[name] = Word(bits & MASK32, bits & MASK32)
        return self.words[name]

    def operand(self, token: str, signed: bool = False) -> tuple[int, int, Word | None]:
        """Read a 32-bit operand as (hardware bits, exact value, word).

        Example: operand("31") -> (31, 31, None); operand("0xffffffc0",
        signed=True) -> (0xffffffc0, -64, None); operand("v130") holding the
        product above -> (0x80000000, 2147483648, <Word>)
        """
        value = immediate(token)
        if value is not None:
            bits = value & MASK32
            return bits, bits - (1 << 32) if signed and bits >> 31 else bits, None
        current = self.word(_single(token))
        return current.hw, current.value(signed), current

    def pair(self, token: str, at: int) -> tuple[int, int]:
        """Read a 64-bit operand as (hardware bits, exact value).

        The exact value is known for a 64-bit result of the replay and for a
        word with its sign extension; for the latter, a computed word whose
        exact value does not fit 32 bits records an overflow at instruction
        `at`. Any other pair gives its bits, as a signed 64-bit value.

        Example: with v130 the product above and v131 its sign extension,
        pair("v[130:131]", at) -> (0xffffffff80000000, 2147483648) and records
        {"at": at, "producer": <v_mul_lo_u32 index>, "value": 2147483648}.
        """
        value = immediate(token)
        if value is not None:
            return value & MASK64, value
        names = registers(token)
        if names is None or len(names) != 2:
            raise ValueError(f"unsupported 64-bit operand {token!r}")
        low, high = self.word(names[0]), self.word(names[1])
        bits = low.hw | high.hw << 32
        if high.ref is low and high.kind == "pair_hi":
            return bits, low.exact
        if high.ref is low and high.kind == "sext":
            exact = low.value(signed=True)
            if not -(1 << 31) <= exact < 1 << 31:
                self.overflows.append(
                    {"at": at, "producer": low.producer, "value": exact}
                )
            return bits, exact
        return bits, bits - (1 << 64) if bits >> 63 else bits

    def set(
        self,
        token: str,
        hw: int,
        exact: int,
        index: int,
        kind: str = "word",
        ref: Word | None = None,
    ) -> None:
        """Write a 32-bit result to the register named by `token`."""
        self.words[_single(token)] = Word(hw & MASK32, exact, kind, ref, index)

    def set_pair(self, token: str, hw: int, exact: int, index: int) -> None:
        """Write a 64-bit result to the pair named by `token`."""
        names = registers(token)
        if names is None or len(names) != 2:
            raise ValueError(f"unsupported 64-bit destination {token!r}")
        low = Word(hw & MASK32, exact, "pair_lo", None, index)
        self.words[names[0]] = low
        self.words[names[1]] = Word(
            hw >> 32 & MASK32, exact >> 32, "pair_hi", low, index
        )

    def run(self, index: int, asm: str) -> None:
        """Replay one instruction; raise NotImplementedError if it is not modelled.

        Example: run(5, "v_mul_lo_u32 v130, v139, s52") with v139 = 32768 and
        s52 = 65536 makes v130 Word(hw=0x80000000, exact=2147483648, producer=5).
        """
        opcode, ops, _ = split(asm)
        handler = _HANDLERS.get(opcode)
        if handler is None:
            raise NotImplementedError(opcode)
        handler(self, index, ops)


def _bitwise(
    function: Callable[[int, int], int],
) -> Callable[[Lane, int, list[str]], None]:
    """`v_and_b32`/`v_or_b32 dst, a, b`: a 32-bit bit pattern, read signed on demand.

    The result is kind "bits", so a later signed read sign-extends its bits
    rather than reading them as a large unsigned value (which would look like an
    overflow at a sign extension, for example after masking a negative index).
    """

    def handle(lane: Lane, index: int, ops: list[str]) -> None:
        bits = function(lane.operand(ops[1])[0], lane.operand(ops[2])[0])
        lane.set(ops[0], bits, bits, index, "bits")

    return handle


def _mul_lo(lane: Lane, index: int, ops: list[str]) -> None:
    """`v_mul_lo_u32 dst, a, b`: the low 32 bits of a * b.

    The low bits do not depend on signedness; the exact product reads leaves
    and immediates as signed, as Triton's int32 index arithmetic is.
    """
    (a_bits, a, _), (b_bits, b, _) = (lane.operand(t, signed=True) for t in ops[1:3])
    lane.set(ops[0], a_bits * b_bits, a * b, index)


def _ashr(lane: Lane, index: int, ops: list[str]) -> None:
    """`v_ashrrev_i32 dst, amount, value`: value >> amount[4:0], arithmetic.

    A shift of a register by 31 writes its sign word, of kind "sext".
    """
    bits, exact, word = lane.operand(ops[2], signed=True)
    shift = lane.operand(ops[1])[0] & 31
    signed_bits = bits - (1 << 32) if bits >> 31 else bits
    if shift == 31 and word is not None:
        # The sign extension; an overflow of `word` is recorded where it widens.
        lane.set(ops[0], signed_bits >> 31, -1 if exact < 0 else 0, index, "sext", word)
    else:
        # A non-sign-extending shift drops bits, so an already-wrapped input
        # could be laundered back into range. Record that wrap here instead.
        if word is not None and word.producer is not None and signed_bits != exact:
            lane.overflows.append(
                {"at": index, "producer": word.producer, "value": exact}
            )
        lane.set(ops[0], signed_bits >> shift, exact >> shift, index)


def _lshl_add64(lane: Lane, index: int, ops: list[str]) -> None:
    """`v_lshl_add_u64 dst[2], a[2], shift, b[2]`: (a << shift) + b in 64 bits.

    LLVM emits it only for shifts of 0 to 4; others are not replayed.
    """
    a_bits, a = lane.pair(ops[1], index)
    shift = lane.operand(ops[2])[0]
    if shift > 4:
        raise ValueError(f"v_lshl_add_u64 shift {shift} is not 0 to 4")
    b_bits, b = lane.pair(ops[3], index)
    lane.set_pair(ops[0], (a_bits << shift) + b_bits, (a << shift) + b, index)


_HANDLERS: dict[str, Callable[[Lane, int, list[str]], None]] = {
    "v_and_b32": _bitwise(lambda a, b: a & b),
    "v_or_b32": _bitwise(lambda a, b: a | b),
    "v_mul_lo_u32": _mul_lo,
    "v_ashrrev_i32": _ashr,
    "v_lshl_add_u64": _lshl_add64,
}
MODELLED = frozenset(_HANDLERS)
