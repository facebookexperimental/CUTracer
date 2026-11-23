#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Record the tensors a test compares, and compare them with a golden run.

A delay injected by ``cutracer stress`` changes only the schedule, so a
kernel whose output changes under it computes something that depends on the
schedule. A test's own assertions are a coarse check of that: they pass
anything inside their tolerance and say nothing when the test only checks
shapes. This module records every tensor the test hands to a comparison
function in a plain run (the golden) and compares each of them in a later run,
normally one under CUTracer, bit for bit and then by magnitude.

``cutracer oracle --numerics`` installs it in the process under test as the
module ``cutracer_numerics``. Nothing here depends on the module's own name.
Importing it with ``CUTRACER_NUMERICS_MODE`` set installs the capture, which
imports torch (and numpy.testing, when numpy is installed); otherwise it
imports only the standard library.

Environment (``cutracer oracle`` sets it):

  CUTRACER_NUMERICS_MODE    ``record`` or ``check``; ``autotune`` records only the
                            Triton autotuner's choices; anything else: inert
  CUTRACER_NUMERICS_GOLDEN  golden directory (record writes it, check reads it)
  CUTRACER_NUMERICS_OUT     this run's output directory (``numerics-<proc>.json``,
                            ``calls-<proc>.json``); required to check
  CUTRACER_NUMERICS_ROOT    call sites are recorded relative to this directory
                            (default: the working directory)
  CUTRACER_NUMERICS_PROC    process label (default ``rank-$RANK``, else ``main``
                            for the process that installed first)
  CUTRACER_NUMERICS_SEED    seed for random, numpy and torch before each pytest
                            test (default 0; ``none`` keeps the test's seeding)
  CUTRACER_NUMERICS_MAX_MB  largest tensor saved, by its size in memory (default 512)
  CUTRACER_NUMERICS_MAX_TOTAL_MB  bytes each process writes to the golden (default
                            4096); a tensor is saved while its size in memory
                            fits in what is left
  CUTRACER_NUMERICS_WRAP    ``module:function`` entries whose tensor results are
                            recorded too (outermost calls; ``module:*`` is every
                            public function defined in the module, ``module:**``
                            also its torch.library custom ops). Recording copies
                            the result to the host, which waits for the device.
  CUTRACER_NUMERICS_PIN_HYPOTHESIS  ``1``: Hypothesis draws the same examples in
                            every run, with no deadline and no shrinking
  CUTRACER_NUMERICS_PIN_AUTOTUNE  ``0``: do not pin Triton autotuning (below)
  CUTRACER_NUMERICS_LARGE_REL_L2, _LARGE_ROW_REL_L2, _LARGE_ELEM_REL
                            thresholds of ``classify`` (0.02, 0.05, 0.25)

What is recorded, per process and in call order: both operands of
``torch.testing.assert_close`` (and the deprecated ``assert_allclose``),
``torch.allclose``, ``torch.equal``, ``Tensor.allclose``, ``Tensor.equal``,
``numpy.testing.assert_allclose``, ``assert_array_equal`` and
``assert_array_almost_equal``; the results of ``WRAP`` entries; and
``record_tensor(name, value)``. A call made inside another recorded call, or
whose caller is torch, numpy or Triton's compiler and runtime, is not
recorded (so ``torch._dynamo.utils.same``, PyTorch's ``TestCase.assertEqual``
and helpers under ``torch.*`` are not seen). A call is matched with the golden
by its pytest node, function, call site and ordinal for those three; calls
from more than one thread at one site cannot be matched and make the run
incomparable.

Only functions looked up after this module is imported are seen: a module
that bound ``assert_close`` before then keeps the original. ``-p
cutracer_numerics`` and ``PYTEST_PLUGINS`` import it before any conftest.

The golden keeps each distinct tensor once, in ``blobs/<xx>/<sha256>`` (the
sha256 of its bytes, which every process of the golden shares): its bytes in
4 MiB chunks, byte-shuffled by element size and compressed with zstd (Python
3.14's ``compression.zstd``, or ``zstandard``) or else zlib, or stored as they
are when a sample does not shrink by a tenth. Its dtype and shape are in the
calls file. A check reads a golden tensor only when the run's bytes differ,
and verifies it against its sha256.

Triton autotuning: an ``autotune`` run records the config that each
``triton.runtime.autotuner.Autotuner`` picks for each tuning key into
``GOLDEN/autotune-<proc>.jsonl``. ``record`` and ``check`` runs then give the
autotuner that config without benchmarking it (a config the autotuner makes
up while tuning is picked through fake timings), so the golden and every
checked run execute the same kernels. A tuning key that is not in the
record benchmarks as usual and counts as a miss, with the number of calls
recorded by then. An autotuner is told apart by its function, key names,
configs and its place among autotuners that share them. Subclasses that
override ``run``, other autotuners (Inductor's), and a reloaded autotuner
module are not pinned.

A check classifies each changed tensor (``classify``):

  large       an element changed between finite, +inf, -inf and nan; any
              change of an integer, bool or complex tensor or of a float
              format without a bit-level integer view; relative L2 over the
              finite positions above max(LARGE_REL_L2, eps) (any change of an
              all-zero golden); a row (last dimension) whose relative L2 is
              above max(LARGE_ROW_REL_L2, 2 eps); or an element that moved by
              more than max(LARGE_ELEM_REL, 4 eps) times the larger of its
              golden magnitude and its row's RMS
  small       any other bitwise change: the size of reduction-order noise
  unmeasured  changed, but the golden tensor was not saved (too big), or the
              tensor could not be compared
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import inspect
import json
import math
import os
import random
import socket
import struct
import sys
import tempfile
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

SCHEMA = 1

ENV_MODE = "CUTRACER_NUMERICS_MODE"
ENV_GOLDEN = "CUTRACER_NUMERICS_GOLDEN"
ENV_OUT = "CUTRACER_NUMERICS_OUT"
ENV_ROOT = "CUTRACER_NUMERICS_ROOT"
ENV_PROC = "CUTRACER_NUMERICS_PROC"
ENV_SEED = "CUTRACER_NUMERICS_SEED"
ENV_MAX_MB = "CUTRACER_NUMERICS_MAX_MB"
ENV_MAX_TOTAL_MB = "CUTRACER_NUMERICS_MAX_TOTAL_MB"
ENV_WRAP = "CUTRACER_NUMERICS_WRAP"
ENV_PIN_HYPOTHESIS = "CUTRACER_NUMERICS_PIN_HYPOTHESIS"
# Set by the first process of a tree that installs the capture: that process
# is `main`, and a descendant needs a label of its own.
ENV_ROOT_PID = "CUTRACER_NUMERICS_ROOT_PID"

# `autotune` records only the Triton autotuner's choices (a plain run before
# the golden's, which then replays them).
MODES = ("record", "check", "autotune")
SAME = "same"
SMALL = "small"
LARGE = "large"
UNMEASURED = "unmeasured"
_SEVERITY = {SAME: 0, SMALL: 1, UNMEASURED: 2, LARGE: 3}

_DEFAULT_REL_L2 = 0.02
_DEFAULT_ROW_REL_L2 = 0.05
_DEFAULT_ELEM_REL = 0.25
# Elements per slice when comparing, so a large tensor's float64 temporaries
# stay bounded.
_CHUNK_ELEMENTS = 1 << 22
_MAX_MEDIAN_SAMPLES = 1 << 20

# Callers whose frames are skipped when finding a call's site: the comparison
# functions' own helpers.
_PASS_THROUGH_MODULES = (
    "torch.testing",
    "numpy.testing",
    "triton.testing",
    "unittest",
    "functools",
    "contextlib",
)
# A comparison whose site is inside one of these is the library's own, not
# the test's (an Inductor guard, a Triton autotuner check). Only Triton's own
# subpackages: other code can live under the `triton` namespace.
_INTERNAL_MODULES = (
    "torch",
    "numpy",
    "triton.backends",
    "triton.compiler",
    "triton.language",
    "triton.runtime",
)

_COMPARISONS: Tuple[Tuple[str, str, str, Tuple[str, str]], ...] = (
    # (module, attribute path, label, operand names when no signature exists)
    (
        "torch.testing",
        "assert_close",
        "torch.testing.assert_close",
        ("actual", "expected"),
    ),
    (
        "torch.testing",
        "assert_allclose",
        "torch.testing.assert_allclose",
        ("actual", "expected"),
    ),
    ("torch", "allclose", "torch.allclose", ("input", "other")),
    ("torch", "equal", "torch.equal", ("input", "other")),
    ("torch", "Tensor.allclose", "Tensor.allclose", ("self", "other")),
    ("torch", "Tensor.equal", "Tensor.equal", ("self", "other")),
    (
        "numpy.testing",
        "assert_allclose",
        "numpy.testing.assert_allclose",
        ("actual", "desired"),
    ),
    (
        "numpy.testing",
        "assert_array_equal",
        "numpy.testing.assert_array_equal",
        ("x", "y"),
    ),
    (
        "numpy.testing",
        "assert_array_almost_equal",
        "numpy.testing.assert_array_almost_equal",
        ("x", "y"),
    ),
)

_tls = threading.local()


def _depth() -> int:
    return getattr(_tls, "depth", 0)


def _torch() -> Any:
    import torch

    return torch


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def thresholds() -> Dict[str, float]:
    """The thresholds `classify` uses, with environment overrides."""
    return {
        "rel_l2": _env_float("CUTRACER_NUMERICS_LARGE_REL_L2", _DEFAULT_REL_L2),
        "row_rel_l2": _env_float(
            "CUTRACER_NUMERICS_LARGE_ROW_REL_L2", _DEFAULT_ROW_REL_L2
        ),
        "elem_rel": _env_float("CUTRACER_NUMERICS_LARGE_ELEM_REL", _DEFAULT_ELEM_REL),
    }


# Dtypes ----------------------------------------------------------------------

_EPS = {
    "float64": 2.0**-52,
    "float32": 2.0**-23,
    "float16": 2.0**-10,
    "bfloat16": 2.0**-7,
    "float8_e4m3fn": 2.0**-3,
    "float8_e4m3fnuz": 2.0**-3,
    "float8_e5m2": 2.0**-2,
    "float8_e5m2fnuz": 2.0**-2,
    "float8_e8m0fnu": 1.0,
}


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _eps(dtype_name: str) -> Optional[float]:
    return _EPS.get(dtype_name)


def _int_view(dtype_name: str) -> Optional[Tuple[str, int, bool]]:
    """(integer dtype of the same width, bit width, has a sign bit) for the
    float formats compared value by value; None for every other dtype."""
    if dtype_name == "float64":
        return "int64", 64, True
    if dtype_name == "float32":
        return "int32", 32, True
    if dtype_name in ("float16", "bfloat16"):
        return "int16", 16, True
    if dtype_name.startswith("float8_e8m0"):
        return "uint8", 8, False
    if dtype_name in _EPS and dtype_name.startswith("float8"):
        return "uint8", 8, True
    return None


def _kind(dtype: Any) -> str:
    """``float`` (compared by value), ``opaque`` (a float format compared by
    its bytes) or ``exact`` (integer, bool, complex: any change counts)."""
    name = _dtype_name(dtype)
    if dtype.is_floating_point or name.startswith("float"):
        return "float" if _int_view(name) is not None else "opaque"
    return "exact"


# Comparing two tensors ---------------------------------------------------------


class _SumSq:
    """A sum of squares kept as ``scale**2 * sumsq`` so it cannot overflow."""

    def __init__(self) -> None:
        self.scale = 0.0
        self.sumsq = 0.0

    def add(self, values: Any) -> None:
        if values.numel() == 0:
            return
        amax = float(values.abs().max())
        if amax == 0.0:
            return
        if not math.isfinite(amax):
            self.scale, self.sumsq = math.inf, 1.0
            return
        part = float(((values / amax) ** 2).sum())
        if amax > self.scale:
            self.sumsq = part + (
                self.sumsq * (self.scale / amax) ** 2 if self.scale else 0.0
            )
            self.scale = amax
        else:
            self.sumsq += part * (amax / self.scale) ** 2

    def norm(self) -> float:
        return self.scale * math.sqrt(self.sumsq) if self.scale else 0.0

    def norm_over(self, divisor: float) -> float:
        """The norm divided by `divisor` (at least the largest value), which
        cannot overflow even when the norm itself would."""
        return (self.scale / divisor) * math.sqrt(self.sumsq) if self.scale else 0.0


def _blocks(cur: Any, gold: Any) -> Iterator[Tuple[int, int, Any, Any]]:
    """(row, first column, block of `cur`, block of `gold`), a bounded number
    of elements at a time. Rows run over the last dimension; a block holds
    whole rows, or one segment of a row wider than the bound."""
    width = gold.shape[-1] if gold.dim() >= 1 and gold.shape[-1] else 1
    cur_rows = cur.reshape(-1, width)
    gold_rows = gold.reshape(-1, width)
    if width <= _CHUNK_ELEMENTS:
        step = _CHUNK_ELEMENTS // width
        for row in range(0, gold_rows.shape[0], step):
            yield row, 0, cur_rows[row : row + step], gold_rows[row : row + step]
        return
    for row in range(gold_rows.shape[0]):
        for col in range(0, width, _CHUNK_ELEMENTS):
            cols = slice(col, col + _CHUNK_ELEMENTS)
            yield (
                row,
                col,
                cur_rows[row : row + 1, cols],
                gold_rows[row : row + 1, cols],
            )


def _ordered(x: Any, view: Tuple[str, int, bool]) -> Any:
    """Float bits as integers that are monotonic in the float value."""
    torch = _torch()
    int_dtype, width, signed = view
    bits = x.contiguous().view(getattr(torch, int_dtype)).to(torch.int64)
    if width < 64:
        bits = bits & ((1 << width) - 1)
    if not signed:
        return bits
    if width == 64:
        sign = bits < 0
        magnitude = bits & ((1 << 63) - 1)
    else:
        sign = ((bits >> (width - 1)) & 1) == 1
        magnitude = bits & ((1 << (width - 1)) - 1)
    return torch.where(sign, -magnitude, magnitude)


def _ulp_distance(cur: Any, gold: Any, view: Tuple[str, int, bool], bf16: bool) -> Any:
    """Per-element ULP distance (float64 for 64-bit formats, whose integer
    difference can overflow)."""
    torch = _torch()
    if bf16:
        cur = (cur.view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
        gold = (gold.view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
        view = ("int16", 16, True)
    a, b = _ordered(cur, view), _ordered(gold, view)
    if view[1] == 64:
        return (a.to(torch.float64) - b.to(torch.float64)).abs()
    return (a - b).abs()


def _value_class(x: Any) -> Any:
    """0 finite, 1 +inf, 2 -inf, 3 nan, per element of a float64 tensor."""
    torch = _torch()
    cls = torch.zeros(x.shape, dtype=torch.int8)
    cls = torch.where(torch.isposinf(x), torch.ones_like(cls), cls)
    cls = torch.where(torch.isneginf(x), torch.full_like(cls, 2), cls)
    return torch.where(torch.isnan(x), torch.full_like(cls, 3), cls)


def _rel_l2(a: Any, b: Any) -> Optional[float]:
    """||a - b|| / ||b|| over the positions finite in both (same shapes)."""
    torch = _torch()
    diff, ref = _SumSq(), _SumSq()
    seen = False
    for _, _, a_block, b_block in _blocks(a, b):
        x = a_block.to(torch.float64)
        y = b_block.to(torch.float64)
        both = torch.isfinite(x) & torch.isfinite(y)
        if not bool(both.any()):
            continue
        seen = True
        diff.add((x - y)[both])
        ref.add(y[both])
    if not seen:
        return None
    if ref.scale == 0.0:
        return 0.0 if diff.scale == 0.0 else math.inf
    divisor = max(diff.scale, ref.scale)
    return diff.norm_over(divisor) / ref.norm_over(divisor)


def _geometry(changed: Any) -> List[Dict[str, Any]]:
    """Per dimension: how many indices hold a change, and their range."""
    torch = _torch()
    out = []
    for dim in range(changed.dim()):
        others = [d for d in range(changed.dim()) if d != dim]
        hit = changed.any(dim=others) if others else changed
        idx = torch.nonzero(hit).flatten()
        row: Dict[str, Any] = {
            "dim": dim,
            "size": int(changed.shape[dim]),
            "count": int(idx.numel()),
        }
        if idx.numel():
            row["min"] = int(idx.min())
            row["max"] = int(idx.max())
            if idx.numel() <= 16:
                row["indices"] = [int(i) for i in idx]
        out.append(row)
    return out


class _FloatScale:
    """The first pass over a float pair."""

    def __init__(self) -> None:
        # The largest finite magnitude in either tensor.
        self.scale = 0.0
        # The golden over the positions finite in both: overall, and for
        # each row that is split into segments.
        self.total = _SumSq()
        self.wide_rows: Dict[int, _SumSq] = {}
        self.any_both = False
        # Both are float32 tensors that hold only bfloat16 values.
        self.bf16 = False


def _float_scale(cur: Any, gold: Any) -> _FloatScale:
    torch = _torch()
    out = _FloatScale()
    check_bf16 = cur.dtype == torch.float32
    low_bits = False
    width = gold.shape[-1] if gold.dim() >= 1 and gold.shape[-1] else 1
    for row, _, c_block, g_block in _blocks(cur, gold):
        c = c_block.to(torch.float64)
        g = g_block.to(torch.float64)
        both = torch.isfinite(c) & torch.isfinite(g)
        if bool(both.any()):
            out.any_both = True
            magnitude = max(float(c[both].abs().max()), float(g[both].abs().max()))
            out.scale = max(out.scale, magnitude)
            out.total.add(g[both])
            if width > _CHUNK_ELEMENTS:
                out.wide_rows.setdefault(row, _SumSq()).add(g[both])
        if check_bf16 and not low_bits:
            low_bits = bool(
                ((c_block.view(torch.int32) & 0xFFFF) != 0).any()
                or ((g_block.view(torch.int32) & 0xFFFF) != 0).any()
            )
    out.bf16 = check_bf16 and not low_bits
    return out


class _Limits:
    """The `large` limits for one tensor; floors are in scaled units."""

    def __init__(self, dtype: str, reference: float, n_rows: int, numel: int) -> None:
        eps = _eps(dtype) or 0.0
        limits = thresholds()
        floor = max(1e-3, 4 * eps) * reference
        self.row_floor = max(floor / math.sqrt(n_rows), 1e-300)
        self.elem_floor = floor / math.sqrt(max(numel, 1))
        self.row = max(limits["row_rel_l2"], 2 * eps)
        self.elem = max(limits["elem_rel"], 4 * eps)


class _FloatTally:
    """Running totals of the second pass over a float pair. Every norm here
    is of values divided by the first pass's scale, so none overflows."""

    def __init__(self) -> None:
        self.n_diff = 0
        self.nonfinite_changed = 0
        self.new_nonfinite = 0
        self.lost_nonfinite = 0
        self.max_delta = 0.0
        self.diff_sq = 0.0
        self.max_ulp = 0.0
        self.changed_ulps: List[Any] = []
        self.n_changed_ulps = 0
        self.max_row_rel = 0.0
        self.n_bad_rows = 0
        self.bad_rows: List[int] = []
        # Squared differences of the rows split into segments, by row.
        self.wide_row_diff_sq: Dict[int, float] = {}
        self.n_outliers = 0
        self.outliers_head: List[int] = []

    def rows(self, first_row: int, rel: Any, limit: float) -> None:
        """Count the rows of `rel` (relative errors from `first_row` on) above
        `limit`."""
        torch = _torch()
        self.max_row_rel = max(self.max_row_rel, float(rel.max()))
        bad = torch.nonzero(rel > limit).flatten()
        self.n_bad_rows += int(bad.numel())
        room = 16 - len(self.bad_rows)
        self.bad_rows.extend(first_row + i for i in bad[:room].tolist())


def _float_metrics(cur: Any, gold: Any, m: Dict[str, Any]) -> None:
    """Fill `m` for a float tensor compared value by value. Norms are taken
    on values divided by the largest finite magnitude, so float64 values near
    the format's limit do not overflow."""
    torch = _torch()
    view = _int_view(m["dtype"])
    assert view is not None
    int_dtype = getattr(torch, view[0])
    first = _float_scale(cur, gold)
    s = first.scale if first.scale > 0.0 else 1.0
    width = gold.shape[-1] if gold.dim() >= 1 and gold.shape[-1] else 1
    n_rows = max(gold.numel() // width, 1)
    reference = first.total.norm_over(s)
    limits = _Limits(m["dtype"], reference, n_rows, gold.numel())
    tally = _FloatTally()
    for row, col, c_block, g_block in _blocks(cur, gold):
        changed = c_block.view(int_dtype) != g_block.view(int_dtype)
        if not bool(changed.any()):
            continue
        row_ref = None
        if width > _CHUNK_ELEMENTS:
            wide = first.wide_rows.get(row)
            row_ref = wide.norm_over(s) if wide is not None else 0.0
        _tally_block(
            tally,
            (row, col, width),
            (c_block, g_block, changed),
            (view, first.bf16, s),
            limits,
            row_ref,
        )
    for row, diff_sq in sorted(tally.wide_row_diff_sq.items()):
        wide = first.wide_rows.get(row)
        row_ref = wide.norm_over(s) if wide is not None else 0.0
        rel = math.sqrt(diff_sq) / max(row_ref, limits.row_floor)
        tally.rows(row, torch.tensor([rel], dtype=torch.float64), limits.row)
    m["n_diff"] = tally.n_diff
    m["nonfinite_changed"] = tally.nonfinite_changed
    m["new_nonfinite"] = tally.new_nonfinite
    m["lost_nonfinite"] = tally.lost_nonfinite
    if not first.any_both:
        m["rel_l2"] = None
        m["max_abs"] = None
    else:
        diff = math.sqrt(tally.diff_sq)
        if reference == 0.0:
            m["rel_l2"] = 0.0 if diff == 0.0 else math.inf
        else:
            m["rel_l2"] = diff / reference
        m["max_abs"] = tally.max_delta * s
    if first.bf16:
        m["ulp_unit"] = "bfloat16"
    m["max_ulp"] = min(int(tally.max_ulp), 2**63 - 1)
    if tally.changed_ulps:
        m["median_ulp_of_changed"] = float(torch.cat(tally.changed_ulps).median())
    m["rows"] = n_rows
    m["max_row_rel_l2"] = tally.max_row_rel
    m["n_bad_rows"] = tally.n_bad_rows
    if tally.n_bad_rows:
        m["bad_rows_head"] = sorted(tally.bad_rows)
    m["n_outliers"] = tally.n_outliers
    if tally.n_outliers:
        m["outliers_head"] = tally.outliers_head


def _tally_block(
    tally: _FloatTally,
    where: Tuple[int, int, int],
    blocks: Tuple[Any, Any, Any],
    how: Tuple[Tuple[str, int, bool], bool, float],
    limits: _Limits,
    row_ref: Optional[float],
) -> None:
    """Add one block that holds a change to `tally`: whole rows, or (given
    `row_ref`, that row's scaled golden norm) one segment of a wide row."""
    torch = _torch()
    row, col, width = where
    c_block, g_block, changed = blocks
    view, bf16, s = how
    c = c_block.to(torch.float64)
    g = g_block.to(torch.float64)
    cls_c, cls_g = _value_class(c), _value_class(g)
    both = (cls_c == 0) & (cls_g == 0)
    tally.n_diff += int(changed.sum())
    tally.nonfinite_changed += int((cls_c != cls_g).sum())
    tally.new_nonfinite += int(((cls_c != 0) & (cls_g == 0)).sum())
    tally.lost_nonfinite += int(((cls_c == 0) & (cls_g != 0)).sum())
    zero = torch.zeros_like(c)
    gs = torch.where(both, g / s, zero)
    delta = (torch.where(both, c / s, zero) - gs).abs()
    tally.max_delta = max(tally.max_delta, float(delta.max()))
    tally.diff_sq += float((delta * delta).sum())
    ulp = torch.where(
        both,
        _ulp_distance(c_block, g_block, view, bf16),
        torch.zeros((), dtype=torch.int64),
    )
    tally.max_ulp = max(tally.max_ulp, float(ulp.max()))
    if tally.n_changed_ulps < _MAX_MEDIAN_SAMPLES:
        picked = ulp[changed & both].to(torch.float64)
        tally.changed_ulps.append(picked)
        tally.n_changed_ulps += int(picked.numel())
    if row_ref is None:
        ref = torch.linalg.vector_norm(gs, dim=1)
        rel = torch.linalg.vector_norm(delta, dim=1) / ref.clamp_min(limits.row_floor)
        tally.rows(row, rel, limits.row)
        row_rms = (ref / math.sqrt(width)).unsqueeze(1)
    else:
        so_far = tally.wide_row_diff_sq.get(row, 0.0)
        tally.wide_row_diff_sq[row] = so_far + float((delta * delta).sum())
        row_rms = torch.full((1, 1), row_ref / math.sqrt(width), dtype=torch.float64)
    elem_scale = torch.maximum(gs.abs(), row_rms).clamp_min(limits.elem_floor)
    hits = torch.nonzero(((delta > limits.elem * elem_scale) & both).flatten())
    tally.n_outliers += int(hits.numel())
    block_width = c_block.shape[1]
    for i in hits.flatten()[: 16 - len(tally.outliers_head)].tolist():
        tally.outliers_head.append(
            (row + i // block_width) * width + col + i % block_width
        )


def diff_metrics(
    cur: Any, gold: Any, plain: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """How `cur` differs from `gold` (CPU tensors of one shape and dtype).
    Only a change that `classify` with the golden leaf's `plain` record finds
    large gets its `geometry`: a whole-tensor mask is too costly for every bit
    of reduction-order noise."""
    torch = _torch()
    m: Dict[str, Any] = {
        "numel": int(gold.numel()),
        "dtype": _dtype_name(gold.dtype),
        "kind": _kind(gold.dtype),
    }
    if gold.numel() == 0:
        m["n_diff"] = 0
        return m
    if m["kind"] == "float":
        _float_metrics(cur, gold, m)
    elif m["kind"] == "opaque":
        m["n_diff"] = int(_changed(cur, gold, m).sum())
    else:
        changed = _changed(cur, gold, m)
        m["n_diff"] = int(changed.sum())
        if m["n_diff"] and not cur.dtype.is_complex and cur.dtype != torch.bool:
            moved = cur[changed].to(torch.float64) - gold[changed].to(torch.float64)
            m["max_abs"] = float(moved.abs().max())
    if m["n_diff"] and classify(m, plain) == LARGE:
        changed = _changed(cur, gold, m)
        m["geometry"] = _geometry(changed if changed.dim() else changed.reshape(1))
    return m


def _changed(cur: Any, gold: Any, m: Dict[str, Any]) -> Any:
    """The mask of elements whose bits differ."""
    torch = _torch()
    if m["kind"] == "opaque":
        return cur.reshape(-1).view(torch.uint8) != gold.reshape(-1).view(torch.uint8)
    if cur.dtype.is_complex:
        # By the bits of both parts: `!=` misses a signed zero.
        cur = torch.view_as_real(cur.resolve_conj())
        gold = torch.view_as_real(gold.resolve_conj())
        part = _int_view(_dtype_name(cur.dtype))
        if part is not None:
            int_dtype = getattr(torch, part[0])
            return (cur.view(int_dtype) != gold.view(int_dtype)).any(dim=-1)
        return (cur != gold).any(dim=-1)
    view = _int_view(m["dtype"]) if m["kind"] == "float" else None
    if view is None:
        return cur != gold
    int_dtype = getattr(torch, view[0])
    return cur.view(int_dtype) != gold.view(int_dtype)


def classify(m: Dict[str, Any], plain: Optional[Dict[str, Any]] = None) -> str:
    """`same`, `small`, `large` or `unmeasured` for one tensor's metrics.

    `plain` is the golden leaf's plain-run record, when the golden has one:
    an integer or bool tensor that already changed between plain runs (an
    argmax with ties) is only `small`. A complex tensor has no such excuse.
    """
    if m.get("unmeasured"):
        return UNMEASURED
    if not m.get("n_diff"):
        return SAME
    if m.get("kind") != "float":
        tied = m.get("kind") == "exact" and not m.get("dtype", "").startswith("complex")
        if tied and plain and plain.get("changed"):
            return SMALL
        return LARGE
    eps = _eps(m.get("dtype", "")) or 0.0
    limits = thresholds()
    rel = m.get("rel_l2")
    if (
        m.get("nonfinite_changed")
        or (rel is not None and (math.isnan(rel) or rel > max(limits["rel_l2"], eps)))
        or m.get("n_bad_rows")
        or m.get("n_outliers")
    ):
        return LARGE
    return SMALL


# Capturing operands -----------------------------------------------------------


def _to_cpu(t: Any) -> Any:
    torch = _torch()
    if type(t) is not torch.Tensor:
        t = t.as_subclass(torch.Tensor)
    t = t.detach()
    if t.is_conj():
        t = t.resolve_conj()
    if t.is_neg():
        t = t.resolve_neg()
    if t.layout != torch.strided:
        t = t.to_dense()
    return t.to("cpu", copy=True).contiguous()


def _buffer(u8: Any) -> Any:
    """A bytes-like view of a contiguous uint8 CPU tensor."""
    try:
        return u8.numpy().data
    except (RuntimeError, TypeError):  # torch without numpy
        return bytes(u8.tolist())


def _sha256(cpu: Any) -> str:
    return hashlib.sha256(_buffer(cpu.reshape(-1).view(_torch().uint8))).hexdigest()


def _numbers_tensor(values: Sequence[Any]) -> Any:
    torch = _torch()
    if all(isinstance(v, bool) for v in values):
        return torch.tensor(list(values), dtype=torch.bool)
    if any(isinstance(v, float) for v in values):
        return torch.tensor([float(v) for v in values], dtype=torch.float64)
    return torch.tensor([int(v) for v in values], dtype=torch.int64)


def _is_number(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def _leaves(value: Any, path: str = "") -> List[Tuple[str, Any]]:
    """Tensor leaves of a (possibly nested) operand. Numbers become tensors;
    another value is a leaf of its own (None) so the structure is kept."""
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return [(path, value)]
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(value, (numpy.ndarray, numpy.generic)):
        return [(path, torch.from_numpy(numpy.ascontiguousarray(value)))]
    if _is_number(value):
        return [(path, _numbers_tensor([value]).reshape(()))]
    if isinstance(value, (list, tuple)) and value and all(_is_number(v) for v in value):
        return [(path, _numbers_tensor(value))]
    if isinstance(value, (list, tuple)):
        out: List[Tuple[str, Any]] = []
        for i, item in enumerate(value):
            out.extend(_leaves(item, f"{path}[{i}]"))
        return out
    if isinstance(value, dict):
        out = []
        for key in sorted(value, key=str):
            out.extend(_leaves(value[key], f"{path}.{key}"))
        return out
    return [(path, None)]


def _module_matches(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _display_path(path: str) -> str:
    root = os.environ.get(ENV_ROOT) or os.getcwd()
    try:
        relative = os.path.relpath(path, root)
    except ValueError:
        return path
    return path if relative.startswith("..") else relative


def _call_site(skip: int = 2) -> Optional[str]:
    """``file:line`` of the code that made the call, or None when that code
    is torch, numpy or Triton's compiler and runtime."""
    frame: Any = sys._getframe(skip)
    own = __name__
    while frame is not None:
        name = str(frame.f_globals.get("__name__", ""))
        filename = frame.f_code.co_filename
        if (
            name == own
            or (filename.startswith("<") and name != "__main__")
            or _module_matches(name, _PASS_THROUGH_MODULES)
        ):
            frame = frame.f_back
            continue
        if _module_matches(name, _INTERNAL_MODULES):
            return None
        where = filename if filename.startswith("<") else _display_path(filename)
        return f"{where}:{frame.f_lineno}"
    return None


def _current_nodeid() -> str:
    """The pytest node running now (without its phase), or ''."""
    current = os.environ.get("PYTEST_CURRENT_TEST", "")
    return current.rsplit(" (", 1)[0] if current else ""


def _process_label() -> Optional[str]:
    """This process's label; None for a descendant with no label."""
    if os.environ.get(ENV_PROC):
        return os.environ[ENV_PROC]
    for rank in ("RANK", "LOCAL_RANK"):
        if os.environ.get(rank):
            return f"rank-{os.environ[rank]}"
    root = os.environ.get(ENV_ROOT_PID)
    if not root:
        os.environ[ENV_ROOT_PID] = str(os.getpid())
        return "main"
    return "main" if root == str(os.getpid()) else None


def _fingerprint() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version.split()[0],
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        env["torch"] = getattr(torch, "__version__", "?")
        try:
            if torch.cuda.is_initialized():
                env["gpu"] = torch.cuda.get_device_name()
        except Exception:  # noqa: B902
            pass
    triton = sys.modules.get("triton")
    if triton is not None:
        env["triton"] = getattr(triton, "__version__", "?")
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_INJECTION64_PATH",
        "CUTRACER_DELAY_DUMP_PATH",
        "CUTRACER_DELAY_LOAD_PATH",
        "PYTHONHASHSEED",
    ):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


# Environment facts that must match between the golden and a checked run.
_MATCHING_ENV = ("torch", "triton", "gpu")


def _write_new(path: Path, value: Any) -> bool:
    """Write `value` as JSON to `path`, which must not exist yet. Readers see
    the whole file or none of it, except on a file system without hard
    links, where the file is only created exclusively."""
    text = json.dumps(value, indent=1, default=str) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w") as sink:
            sink.write(text)
        try:
            os.link(tmp, str(path))
            return True
        except FileExistsError:
            return False
        except OSError:
            pass
    finally:
        os.unlink(tmp)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as sink:
        sink.write(text)
    return True


def _env_mb(name: str, default: float) -> int:
    return int(_env_float(name, default) * 2**20)


# Golden blobs -----------------------------------------------------------------

_BLOB_MAGIC = b"CTNUMB01"
# Magic, codec, shuffle width, reserved, size and chunk size in bytes; then
# each chunk as its stored length and its stored bytes.
_BLOB_HEADER = struct.Struct("<8sBBHQQ")
_BLOB_CHUNK_LENGTH = struct.Struct("<Q")
_BLOB_CODECS = ("raw", "zlib", "zstd")
_BLOB_CHUNK = 1 << 22
# Four slices of this many bytes decide whether a tensor larger than one chunk
# is compressed: one that shrinks by less than _BLOB_MIN_SAVING is stored raw.
_BLOB_SAMPLE = 1 << 18
_BLOB_MIN_SAVING = 0.1
_BLOB_THREADS = 8


def _zstd() -> Optional[Tuple[Callable[[Any], bytes], Callable[[bytes, int], bytes]]]:
    """zstd (compress, decompress) from Python 3.14's standard library or
    `zstandard`; None when neither is installed."""
    try:
        from compression import zstd
    except ImportError:
        try:
            import zstandard
        except ImportError:
            return None
        return (
            lambda data: zstandard.ZstdCompressor(level=3).compress(data),
            lambda data, size: zstandard.ZstdDecompressor().decompress(
                data, max_output_size=size
            ),
        )
    return (
        lambda data: zstd.compress(data, level=3),
        lambda data, size: zstd.decompress(data),
    )


def _codec(name: str) -> Tuple[Callable[[Any], Any], Callable[[bytes, int], bytes]]:
    """(compress, decompress) of a blob codec."""
    if name == "zlib":
        return (
            lambda data: zlib.compress(data, 1),
            lambda data, size: zlib.decompress(data),
        )
    if name == "zstd":
        functions = _zstd()
        if functions is None:
            raise ValueError(
                "the blob is zstd-compressed, and this interpreter has no zstd"
            )
        return functions
    return (lambda data: data, lambda data, size: data)


def _blob_path(golden: Path, sha256: str) -> Path:
    return golden / "blobs" / sha256[:2] / sha256


def _shuffled(u8: Any, width: int) -> Any:
    """`u8`'s bytes grouped by their place in an element, which compresses
    better than interleaved numbers."""
    return u8 if width == 1 else u8.view(-1, width).t().contiguous()


def _compressible(flat: Any, width: int, compress: Callable[[Any], Any]) -> bool:
    """Whether four evenly spaced slices of `flat` shrink by _BLOB_MIN_SAVING."""
    raw = stored = 0
    for i in range(4):
        start = flat.numel() * i // 4 // width * width
        piece = flat[start : start + _BLOB_SAMPLE]
        raw += piece.numel()
        stored += len(compress(_buffer(_shuffled(piece, width))))
    return stored <= (1 - _BLOB_MIN_SAVING) * raw


def _packed_chunks(
    flat: Any, width: int, compress: Callable[[Any], Any]
) -> Iterator[Any]:
    """`flat`'s chunks, shuffled and compressed, in order: several at a time
    (zlib and zstd release the GIL), with a bounded number held in memory."""

    def pack(start: int) -> Any:
        return compress(_buffer(_shuffled(flat[start : start + _BLOB_CHUNK], width)))

    workers = max(1, min(_BLOB_THREADS, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: List[Any] = []
        for start in range(0, flat.numel(), _BLOB_CHUNK):
            pending.append(pool.submit(pack, start))
            if len(pending) > workers:
                yield pending.pop(0).result()
        for future in pending:
            yield future.result()


def _write_blob(path: Path, cpu: Any) -> int:
    """Store `cpu`'s bytes at `path`, compressed unless that saves less than
    _BLOB_MIN_SAVING; the bytes written."""
    flat = cpu.reshape(-1).view(_torch().uint8)
    size = flat.numel()
    width = cpu.element_size()
    codec = "zstd" if _zstd() is not None else "zlib"
    compress = _codec(codec)[0]
    chunks: Iterator[Any]
    if size <= _BLOB_CHUNK:
        packed = compress(_buffer(_shuffled(flat, width)))
        chunks = iter([packed])
        if len(packed) > (1 - _BLOB_MIN_SAVING) * size:
            codec, width, chunks = "raw", 1, iter([_buffer(flat)])
    elif _compressible(flat, width, compress):
        chunks = _packed_chunks(flat, width, compress)
    else:
        codec, width = "raw", 1
        chunks = (
            _buffer(flat[i : i + _BLOB_CHUNK]) for i in range(0, size, _BLOB_CHUNK)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb") as sink:
            sink.write(
                _BLOB_HEADER.pack(
                    _BLOB_MAGIC, _BLOB_CODECS.index(codec), width, 0, size, _BLOB_CHUNK
                )
            )
            for chunk in chunks:
                sink.write(_BLOB_CHUNK_LENGTH.pack(len(chunk)))
                sink.write(chunk)
            written = sink.tell()
        # Another process storing the same bytes at once writes an equal blob.
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return written


def _read_blob(golden: Path, leaf: Dict[str, Any]) -> Any:
    """The golden tensor of `leaf`, verified against its sha256."""
    torch = _torch()
    path = _blob_path(golden, leaf["sha256"])
    size = int(leaf["nbytes"])
    out = torch.empty(size, dtype=torch.uint8)
    with path.open("rb") as source:
        magic, codec, width, _, stored, chunk = _BLOB_HEADER.unpack(
            source.read(_BLOB_HEADER.size)
        )
        if (
            magic != _BLOB_MAGIC
            or stored != size
            or codec >= len(_BLOB_CODECS)
            or not width
            or not chunk
            or chunk % width
        ):
            raise ValueError(f"{path.name} is not a golden blob of {size} bytes")
        decompress = _codec(_BLOB_CODECS[codec])[1]
        for start in range(0, size, chunk):
            expected = min(chunk, size - start)
            (length,) = _BLOB_CHUNK_LENGTH.unpack(source.read(_BLOB_CHUNK_LENGTH.size))
            piece = decompress(source.read(length), expected)
            if len(piece) != expected:
                raise ValueError(f"{path.name} is truncated")
            u8 = torch.frombuffer(bytearray(piece), dtype=torch.uint8)
            if width == 1:
                out[start : start + expected] = u8
            else:
                out[start : start + expected].view(-1, width).copy_(
                    u8.view(width, -1).t()
                )
    tensor = out.view(getattr(torch, leaf["dtype"])).reshape(leaf["shape"])
    if _sha256(tensor) != leaf["sha256"]:
        raise ValueError(f"{path.name} does not match its sha256")
    return tensor


_CALL_KEYS = ("k", "nodeid", "fn", "site", "ordinal", "leaves")


class _Recorder:
    """The calls of one process, recorded (record) or compared (check)."""

    def __init__(self, mode: str, golden: Path, out: Optional[Path], proc: str) -> None:
        self.mode = mode
        self.golden = golden
        self.out = out
        self.proc = proc
        self.pid = os.getpid()
        self.started = time.time()
        self.lock = threading.Lock()
        self.calls: List[Dict[str, Any]] = []
        self.ordinals: Dict[Tuple[str, str, str], int] = {}
        self.site_threads: Dict[Tuple[str, str, str], set] = {}
        self.errors: List[str] = []
        self.finished = False
        self.max_bytes = _env_mb(ENV_MAX_MB, 512)
        self.max_total = _env_mb(ENV_MAX_TOTAL_MB, 4096)
        # The blobs this process wrote, by their size in memory and on disk
        # (which max_total bounds), and the leaves whose blob was already there.
        self.saved_bytes = 0
        self.stored_bytes = 0
        self.reused = 0
        self.golden_env: Dict[str, Any] = {}
        self.golden_calls: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
        self.golden_site_threads: Dict[Tuple[str, str, str], set] = {}
        self.seen: set = set()
        # Why this process records nothing into the golden, if it does not.
        self.refused: Optional[str] = None
        if mode == "record":
            if (golden / f"calls-{proc}.json").exists():
                self.refused = (
                    f"the golden directory already holds calls-{proc}.json; "
                    "record into an empty directory"
                )
                self.errors.append(self.refused)
        elif mode == "check":
            self._load_golden()

    @property
    def recording(self) -> bool:
        return self.mode == "record" and self.refused is None

    def _load_golden(self) -> None:
        path = self.golden / f"calls-{self.proc}.json"
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
                raise ValueError(
                    f"schema {payload.get('schema') if isinstance(payload, dict) else '?'}, expected {SCHEMA}"
                )
            calls = payload["calls"]
            if not isinstance(calls, list) or not all(
                isinstance(c, dict) and all(k in c for k in _CALL_KEYS) for c in calls
            ):
                raise ValueError("malformed calls")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.errors.append(f"no usable golden for process {self.proc}: {error}")
            return
        self.golden_env = payload.get("env") or {}
        for entry in calls:
            site = (entry["nodeid"], entry["fn"], entry["site"])
            self.golden_calls[(*site, entry["ordinal"])] = entry
            self.golden_site_threads.setdefault(site, set()).add(entry.get("thread"))

    def capture_error(self, text: str) -> None:
        with self.lock:
            self._note_error(text)

    def _note_error(self, text: str) -> None:
        """Keep an error (the caller holds the lock, or runs before threads)."""
        if len(self.errors) < 64:
            self.errors.append(text[-800:])

    def observe(self, fn: str, operands: List[Tuple[str, Any]], site: str) -> None:
        with self.lock:
            nodeid = _current_nodeid()
            key = (nodeid, fn, site)
            ordinal = self.ordinals.get(key, 0)
            self.ordinals[key] = ordinal + 1
            thread = threading.current_thread().name
            self.site_threads.setdefault(key, set()).add(thread)
            entry: Dict[str, Any] = {
                "k": len(self.calls),
                "nodeid": nodeid,
                "fn": fn,
                "site": site,
                "ordinal": ordinal,
                "thread": thread,
                "leaves": [],
            }
            self.calls.append(entry)
            golden_entry = None
            if self.mode == "check":
                golden_entry = self.golden_calls.get((*key, ordinal))
                if golden_entry is None:
                    entry["structure"] = "extra call (not in the golden)"
                else:
                    self.seen.add((*key, ordinal))
            cpu_leaves: Dict[str, Any] = {}
            for side, operand in operands:
                self._observe_operand(entry, side, operand, golden_entry, cpu_leaves)
            if golden_entry is not None:
                self._mark_uncompared(entry, golden_entry)
                self._error_ratios(entry, golden_entry, cpu_leaves)

    def _observe_operand(
        self,
        entry: Dict[str, Any],
        side: str,
        operand: Any,
        golden_entry: Optional[Dict[str, Any]],
        cpu_leaves: Dict[str, Any],
    ) -> None:
        try:
            leaves = _leaves(operand)
        except Exception as error:  # noqa: B902
            entry["leaves"].append(
                {
                    "name": f"{side}0",
                    "path": "",
                    "kind": f"unsupported: {error!r}"[:200],
                }
            )
            return
        torch = _torch()
        for j, (path, value) in enumerate(leaves):
            leaf: Dict[str, Any] = {"name": f"{side}{j}", "path": path}
            entry["leaves"].append(leaf)
            if value is None or not isinstance(value, torch.Tensor):
                leaf["kind"] = "other"
                continue
            if type(value) is not torch.Tensor:
                leaf["class"] = type(value).__name__
            try:
                cpu = _to_cpu(value)
                self._observe_tensor(leaf, cpu, golden_entry)
                cpu_leaves[leaf["name"]] = cpu
            except Exception as error:  # noqa: B902
                # A golden with a tensor it could not copy can never be
                # checked, so record mode reports it too.
                leaf["kind"] = f"unsupported: {error!r}"[:200]
                self._note_error(f"{_describe(entry)} {leaf['name']}: {error!r}")

    def _observe_tensor(
        self,
        leaf: Dict[str, Any],
        cpu: Any,
        golden_entry: Optional[Dict[str, Any]],
    ) -> None:
        nbytes = cpu.numel() * cpu.element_size()
        leaf.update(
            kind="tensor",
            shape=list(cpu.shape),
            dtype=_dtype_name(cpu.dtype),
            nbytes=nbytes,
            sha256=_sha256(cpu),
        )
        if self.mode == "record":
            leaf["saved"] = self._save(leaf, cpu, nbytes)
        elif golden_entry is not None:
            leaf["cmp"] = self._compare(leaf, cpu, golden_entry)

    def _save(self, leaf: Dict[str, Any], cpu: Any, nbytes: int) -> bool:
        """Whether the golden holds this leaf's bytes, writing them if they
        are new and their size in memory fits in what is left of the budget."""
        if not self.recording or nbytes > self.max_bytes:
            return False
        path = _blob_path(self.golden, leaf["sha256"])
        if path.exists():
            self.reused += 1
            return True
        if self.stored_bytes + nbytes > self.max_total:
            return False
        self.stored_bytes += _write_blob(path, cpu)
        self.saved_bytes += nbytes
        return True

    def _compare(
        self, leaf: Dict[str, Any], cpu: Any, golden_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        name = leaf["name"]
        gold_leaf = next(
            (g for g in golden_entry["leaves"] if g.get("name") == name), None
        )
        if gold_leaf is None or gold_leaf.get("kind") != "tensor":
            return {"verdict": "structure", "why": "no golden tensor for this leaf"}
        if gold_leaf["shape"] != leaf["shape"] or gold_leaf["dtype"] != leaf["dtype"]:
            return {
                "verdict": "structure",
                "why": f"golden {gold_leaf['shape']} {gold_leaf['dtype']}",
            }
        if gold_leaf["sha256"] == leaf["sha256"]:
            return {"verdict": SAME}
        if not gold_leaf.get("saved"):
            m: Dict[str, Any] = {
                "unmeasured": True,
                "dtype": leaf["dtype"],
                "why": "changed; the golden tensor was too big to save",
            }
        else:
            try:
                m = diff_metrics(
                    cpu, _read_blob(self.golden, gold_leaf), gold_leaf.get("plain")
                )
            except Exception as error:  # noqa: B902
                self._note_error(f"{_describe(golden_entry)} {name}: {error!r}")
                m = {
                    "unmeasured": True,
                    "dtype": leaf["dtype"],
                    "why": f"changed; cannot compare: {error!r}"[:300],
                }
        plain = gold_leaf.get("plain")
        m["verdict"] = classify(m, plain)
        if plain and plain.get("runs") and not plain.get("changed"):
            m["plain_stable"] = True
        return m

    def _mark_uncompared(
        self, entry: Dict[str, Any], golden_entry: Dict[str, Any]
    ) -> None:
        """A golden tensor of this call that the run did not compare (the
        leaf is missing, or could not be copied) is unmeasured."""
        compared = {leaf["name"] for leaf in entry["leaves"] if "cmp" in leaf}
        by_name = {leaf["name"]: leaf for leaf in entry["leaves"]}
        for gold_leaf in golden_entry["leaves"]:
            name = gold_leaf.get("name")
            if gold_leaf.get("kind") != "tensor" or name in compared:
                continue
            why = "not compared in this run"
            leaf = by_name.get(name)
            if leaf is None:
                leaf = {
                    "name": name,
                    "path": gold_leaf.get("path", ""),
                    "kind": "missing",
                }
                entry["leaves"].append(leaf)
            else:
                why = f"not compared ({leaf.get('kind')})"
            leaf["cmp"] = {
                "verdict": UNMEASURED,
                "unmeasured": True,
                "dtype": gold_leaf.get("dtype"),
                "why": why[:300],
            }

    def _error_ratios(
        self,
        entry: Dict[str, Any],
        golden_entry: Dict[str, Any],
        cpu_leaves: Dict[str, Any],
    ) -> None:
        """For a call whose one side changed and the other did not, the
        changed side's relative L2 error against the other, now and in the
        golden."""
        by_name = {leaf["name"]: leaf for leaf in entry["leaves"]}
        for name, leaf in by_name.items():
            other = "b" + name[1:]
            if not name.startswith("a") or other not in by_name:
                continue
            verdicts = (
                leaf.get("cmp", {}).get("verdict"),
                by_name[other].get("cmp", {}).get("verdict"),
            )
            pairs = []
            if verdicts[0] in (SMALL, LARGE) and verdicts[1] == SAME:
                pairs.append((name, other))
            if verdicts[1] in (SMALL, LARGE) and verdicts[0] == SAME:
                pairs.append((other, name))
            for changed, ref in pairs:
                self._error_ratio(
                    golden_entry,
                    by_name[changed],
                    cpu_leaves.get(changed),
                    cpu_leaves.get(ref),
                )

    def _error_ratio(
        self, golden_entry: Dict[str, Any], leaf: Dict[str, Any], changed: Any, ref: Any
    ) -> None:
        if changed is None or ref is None or changed.shape != ref.shape:
            return
        if _kind(changed.dtype) != "float" or _kind(ref.dtype) != "float":
            return
        gold_leaf = next(
            (g for g in golden_entry["leaves"] if g.get("name") == leaf["name"]), {}
        )
        if not gold_leaf.get("saved"):
            return
        try:
            now = _rel_l2(changed, ref)
            then = _rel_l2(_read_blob(self.golden, gold_leaf), ref)
        except Exception as error:  # noqa: B902
            self._note_error(f"error ratio of {leaf['name']}: {error!r}")
            return
        cmp = leaf["cmp"]
        cmp["err_vs_other_now"] = now
        cmp["err_vs_other_golden"] = then
        if now is not None and then:
            cmp["err_ratio"] = now / then

    def _threaded_sites(self) -> List[str]:
        problems = []
        for site in sorted(set(self.site_threads) | set(self.golden_site_threads)):
            threads = (
                len(self.site_threads.get(site, ())),
                len(self.golden_site_threads.get(site, ())),
            )
            if max(threads) > 1:
                problems.append(
                    f"{_describe_key((*site, 0))}: calls from {max(threads)} threads; their order cannot be matched"
                )
        return problems

    def summary(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """The check verdict over every compared leaf."""
        verdict = SAME
        counts = {SAME: 0, SMALL: 0, LARGE: 0, UNMEASURED: 0, "unsupported": 0}
        environment = [
            f"{key} {env[key]} differs from the golden's {self.golden_env[key]}"
            for key in _MATCHING_ENV
            if key in self.golden_env
            and key in env
            and self.golden_env[key] != env[key]
        ]
        structure: List[str] = self._threaded_sites()
        first_structure: Optional[int] = 0 if structure else None
        first_large: Optional[int] = None
        first_large_change: Optional[Dict[str, Any]] = None
        changed: List[Dict[str, Any]] = []
        stable_changed = 0
        stable_head: List[Dict[str, Any]] = []
        for entry in self.calls:
            if "structure" in entry:
                structure.append(f"{_describe(entry)}: {entry['structure']}")
                first_structure = (
                    entry["k"] if first_structure is None else first_structure
                )
            for leaf in entry["leaves"]:
                if str(leaf.get("kind", "")).startswith("unsupported"):
                    counts["unsupported"] += 1
                cmp = leaf.get("cmp")
                if not cmp:
                    continue
                v = cmp["verdict"]
                if v == "structure":
                    structure.append(
                        f"{_describe(entry)} {leaf['name']}: {cmp.get('why')}"
                    )
                    first_structure = (
                        entry["k"] if first_structure is None else first_structure
                    )
                    continue
                counts[v] += 1
                if _SEVERITY[v] > _SEVERITY[verdict]:
                    verdict = v
                if v == SAME:
                    continue
                record = _change_record(entry, leaf, cmp)
                if v == LARGE and first_large is None:
                    first_large, first_large_change = entry["k"], record
                if v == SMALL and cmp.get("plain_stable"):
                    stable_changed += 1
                    if len(stable_head) < 8:
                        stable_head.append(record)
                changed.append(record)
        changed.sort(
            key=lambda c: (-_SEVERITY[c["verdict"]], -(c.get("rel_l2") or 0.0))
        )
        missing = [k for k in self.golden_calls if k not in self.seen]
        return {
            "verdict": verdict,
            "calls": len(self.calls),
            "golden_calls": len(self.golden_calls),
            "leaves": counts,
            "environment": environment,
            "structure": structure[:32],
            "first_structure_k": first_structure,
            "first_large_k": first_large,
            "first_large": first_large_change,
            "missing_calls": len(missing),
            "missing_head": [_describe_key(k) for k in missing[:8]],
            "changed": changed[:32],
            "changed_stable_leaves": stable_changed,
            "stable_changed_head": stable_head,
            "errors": self.errors[:16],
        }

    def write(self) -> None:
        if self.finished or os.getpid() != self.pid:
            return
        self.finished = True
        try:
            self._write()
        except Exception as error:  # noqa: B902
            print(
                f"cutracer_numerics: could not write the result of {self.proc}: {error!r}",
                file=sys.stderr,
            )
            if self.out is not None:
                try:
                    _write_new(
                        self.out / f"error-{self.proc}-{self.pid}.json",
                        {"proc": self.proc, "error": repr(error)},
                    )
                except Exception:  # noqa: B902
                    pass

    def _write(self) -> None:
        env = _fingerprint()
        if self.recording:
            self.golden.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": SCHEMA,
                "proc": self.proc,
                "pid": self.pid,
                "env": env,
                "calls": self.calls,
                "errors": self.errors,
            }
            if not _write_new(self.golden / f"calls-{self.proc}.json", payload):
                self.errors.append(
                    f"another process wrote calls-{self.proc}.json to the golden"
                )
                _write_new(
                    self.golden / f"conflict-{self.proc}-{self.pid}.json", payload
                )
        if self.out is None:
            return
        self.out.mkdir(parents=True, exist_ok=True)
        if self.mode == "check":
            numerics = self.summary(env)
        elif self.mode == "autotune":
            numerics = {
                "verdict": "autotune-recorded",
                "calls": 0,
                "errors": self.errors[:16],
            }
        else:
            numerics = {
                "verdict": "refused" if self.refused else "recorded",
                "calls": len(self.calls),
                "saved_bytes": self.saved_bytes,
                "stored_bytes": self.stored_bytes,
                "reused": self.reused,
                "errors": self.errors[:16],
            }
        numerics["autotune"] = _autotune_summary()
        result = {
            "schema": SCHEMA,
            "proc": self.proc,
            "pid": self.pid,
            "mode": self.mode,
            "env": env,
            "seconds": round(time.time() - self.started, 3),
            "numerics": numerics,
        }
        if _write_new(self.out / f"numerics-{self.proc}.json", result):
            _write_new(
                self.out / f"calls-{self.proc}.json",
                {"proc": self.proc, "calls": self.calls},
            )
        else:
            _write_new(self.out / f"conflict-{self.proc}-{self.pid}.json", result)


class _Unlabeled:
    """A descendant process with no label: records nothing, and leaves a note
    on its first comparison so the run does not read as fully compared."""

    def __init__(self, out: Optional[Path]) -> None:
        self.out = out
        self.pid = os.getpid()
        self.noted = False

    def observe(self, fn: str, operands: List[Tuple[str, Any]], site: str) -> None:
        if self.noted or self.out is None:
            return
        self.noted = True
        try:
            self.out.mkdir(parents=True, exist_ok=True)
            _write_new(
                self.out / f"unlabeled-{self.pid}.json",
                {
                    "pid": self.pid,
                    "fn": fn,
                    "site": site,
                    "root": os.environ.get(ENV_ROOT_PID),
                },
            )
        except OSError:
            pass

    def capture_error(self, text: str) -> None:
        pass

    def write(self) -> None:
        pass


def _describe(entry: Dict[str, Any]) -> str:
    return _describe_key(
        (entry["nodeid"], entry["fn"], entry["site"], entry["ordinal"])
    )


def _describe_key(key: Tuple[str, str, str, int]) -> str:
    nodeid, fn, site, ordinal = key
    where = f"{nodeid} " if nodeid else ""
    return f"{where}{fn} at {site} #{ordinal}"


def _change_record(
    entry: Dict[str, Any], leaf: Dict[str, Any], cmp: Dict[str, Any]
) -> Dict[str, Any]:
    keys = (
        "verdict",
        "dtype",
        "numel",
        "n_diff",
        "max_ulp",
        "median_ulp_of_changed",
        "ulp_unit",
        "rel_l2",
        "max_row_rel_l2",
        "n_bad_rows",
        "bad_rows_head",
        "n_outliers",
        "outliers_head",
        "nonfinite_changed",
        "new_nonfinite",
        "lost_nonfinite",
        "max_abs",
        "err_ratio",
        "err_vs_other_now",
        "err_vs_other_golden",
        "plain_stable",
        "why",
    )
    return {
        "call": _describe(entry),
        "k": entry["k"],
        "site": entry["site"],
        "leaf": leaf["name"],
        "path": leaf.get("path", ""),
        **{key: cmp[key] for key in keys if key in cmp},
    }


# Hooks -------------------------------------------------------------------------

_RECORDER: Any = None
# (owner, attribute, original, wrapper, owned) of every function this module
# hooked; `owned` is whether the attribute was in the owner's own namespace
# (a method inherited by `torch.Tensor` is not).
_HOOKS: List[Tuple[Any, str, Any, Any, bool]] = []
# Wrappers registered as TorchScript builtins, unregistered by `uninstall()`.
_SCRIPT_BUILTINS: List[Any] = []
_ORIGINAL = "_cutracer_numerics_original"
_previous_hypothesis_profile: Optional[str] = None
_fork_hook_registered = False
# Whether `install()` already said why it captures nothing.
_refused = False


def _operand_names(original: Any, fallback: Tuple[str, str]) -> Tuple[str, str]:
    """The names of a comparison's first two parameters (numpy 2 renamed
    some); `fallback` when the function has no signature (a builtin)."""
    try:
        params = list(inspect.signature(original).parameters.values())
    except (TypeError, ValueError):
        return fallback
    positional = [
        p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return (positional[0], positional[1]) if len(positional) >= 2 else fallback


def _operands(
    names: Tuple[str, str], args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> List[Tuple[str, Any]]:
    out = []
    for side, (i, key) in zip("ab", enumerate(names)):
        if i < len(args):
            out.append((side, args[i]))
        elif key in kwargs:
            out.append((side, kwargs[key]))
    return out


def _owned(owner: Any, attr: str) -> bool:
    """Whether `attr` belongs in `owner`'s own namespace, rather than being
    inherited from a base class (as `torch.Tensor.allclose` is)."""
    if not isinstance(owner, type):
        return True
    value = vars(owner).get(attr)
    if value is not None and getattr(value, _ORIGINAL, None) is None:
        return True
    return not any(attr in vars(base) for base in owner.__mro__[1:])


def _hook(owner: Any, attr: str, make_wrapper: Callable[[Any], Any]) -> None:
    current = getattr(owner, attr, None)
    if current is None:
        return
    original = getattr(current, _ORIGINAL, current)
    owned = _owned(owner, attr)
    wrapper = make_wrapper(original)
    setattr(wrapper, _ORIGINAL, original)
    setattr(owner, attr, wrapper)
    _HOOKS.append((owner, attr, original, wrapper, owned))
    _register_script_builtin(original, wrapper)


def _register_script_builtin(original: Any, wrapper: Any) -> None:
    """Keep `torch.jit.script` resolving a hooked torch builtin (it maps
    functions to aten ops by identity)."""
    try:
        from torch.jit import _builtins

        op = _builtins._find_builtin(original)
        if op is not None:
            _builtins._register_builtin(wrapper, op)
            _SCRIPT_BUILTINS.append(wrapper)
    except Exception:  # noqa: B902
        pass


def _unregister_script_builtins() -> None:
    try:
        from torch.jit import _builtins

        table = _builtins._get_builtin_table()
        for wrapper in _SCRIPT_BUILTINS:
            table.pop(id(wrapper), None)
    except Exception:  # noqa: B902
        pass
    _SCRIPT_BUILTINS.clear()


def _comparison_wrapper(label: str, fallback: Tuple[str, str]) -> Callable[[Any], Any]:
    def make(original: Any) -> Any:
        names = _operand_names(original, fallback)

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            recorder = _RECORDER
            if recorder is None or _depth():
                return original(*args, **kwargs)
            _tls.depth = _depth() + 1
            try:
                try:
                    site = _call_site()
                    if site is not None:
                        recorder.observe(label, _operands(names, args, kwargs), site)
                except Exception:  # noqa: B902
                    recorder.capture_error(traceback.format_exc(limit=4))
                return original(*args, **kwargs)
            finally:
                _tls.depth -= 1

        return wrapper

    return make


def _result_wrapper(label: str) -> Callable[[Any], Any]:
    def make(original: Any) -> Any:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            recorder = _RECORDER
            if recorder is None or _depth():
                return original(*args, **kwargs)
            _tls.depth = _depth() + 1
            try:
                result = original(*args, **kwargs)
                try:
                    site = _call_site()
                    if site is not None:
                        recorder.observe(label, [("a", result)], site)
                except Exception:  # noqa: B902
                    recorder.capture_error(traceback.format_exc(limit=4))
                return result
            finally:
                _tls.depth -= 1

        return wrapper

    return make


def _resolve(module_name: str, attr_path: str) -> Tuple[Any, str]:
    import importlib

    owner: Any = importlib.import_module(module_name)
    parts = attr_path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


def _hook_comparisons(recorder: Any) -> None:
    for module_name, attr_path, label, names in _COMPARISONS:
        try:
            owner, attr = _resolve(module_name, attr_path)
        except (ImportError, AttributeError):
            continue
        except Exception as error:  # noqa: B902
            recorder.capture_error(f"cannot hook {label}: {error!r}")
            continue
        _hook(owner, attr, _comparison_wrapper(label, names))


def _wrap_entries(recorder: Any) -> None:
    import importlib

    for entry in filter(
        None, (e.strip() for e in os.environ.get(ENV_WRAP, "").split(","))
    ):
        module_name, _, attr = entry.partition(":")
        try:
            module = importlib.import_module(module_name)
            names = (
                [attr]
                if attr not in ("*", "**")
                else _public_entries(module, module_name, attr == "**")
            )
            for name in names:
                getattr(module, name)  # an unknown name fails here
                label = f"result:{module_name.rsplit('.', 1)[-1]}.{name}"
                _hook(module, name, _result_wrapper(label))
        except Exception as error:  # noqa: B902
            recorder.capture_error(f"cannot wrap {entry}: {error!r}")


def _public_entries(module: Any, module_name: str, custom_ops: bool) -> List[str]:
    names = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        value = getattr(module, name, None)
        if inspect.isfunction(value) and value.__module__ == module_name:
            names.append(name)
        elif (
            custom_ops
            and type(value).__name__ == "CustomOpDef"
            and getattr(getattr(value, "_init_fn", None), "__module__", None)
            == module_name
        ):
            names.append(name)
    return names


# Triton autotuning ----------------------------------------------------------------

ENV_PIN_AUTOTUNE = "CUTRACER_NUMERICS_PIN_AUTOTUNE"
_AUTOTUNER_MODULE = "triton.runtime.autotuner"
# Cached on each autotuner: its identity in this process.
_IDENTITY = "_cutracer_numerics_identity"
# Marks import hooks of any copy of this module, so none consults another.
_HOOK_MARK = "_cutracer_numerics_import_hook"
# Per thread: the recorded config that `_bench` picks for the autotuner
# being tuned now, when that config is not in its config list.
_fallback = threading.local()
_import_guard = threading.local()


class _AutotunePins:
    """Triton autotuner choices: recorded from a plain run (`autotune` mode),
    then forced in every run that records or checks numerics, so the golden
    and the checked runs execute the same kernels."""

    def __init__(self, mode: str, golden: Path, proc: str) -> None:
        self.mode = mode  # "record" or "pin"
        self.path = golden / f"autotune-{proc}.jsonl"
        self.pid = os.getpid()
        self.lock = threading.Lock()
        # (autotuner, key) -> config; None when the record is ambiguous.
        self.table: Dict[Tuple[str, str], Optional[str]] = {}
        self.identities: Dict[str, int] = {}
        # (autotuner instance, key) pairs this process has seen.
        self.seen: Set[Tuple[int, str]] = set()
        self.state = "triton not imported"
        # Whether `_bench` is hooked, which a config the autotuner makes up
        # while tuning needs.
        self.bench_hooked = False
        self.recorded = 0
        self.pinned = 0
        self.n_misses = 0
        # The number of calls the capture had recorded at the first miss.
        self.first_miss_k: Optional[int] = None
        self.misses: List[str] = []
        self.notes: List[str] = []
        if mode == "record":
            golden.mkdir(parents=True, exist_ok=True)
            try:
                self.path.open("x").close()
            except FileExistsError:
                self.state = f"unavailable: another process records {self.path.name}"
            return
        try:
            for line in filter(None, self.path.read_text().splitlines()):
                entry = json.loads(line)
                slot = (entry["autotuner"], entry["key"])
                same = self.table.get(slot, entry["config"]) == entry["config"]
                self.table[slot] = entry["config"] if same else None
        except FileNotFoundError:
            self.state = "golden has no record"
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.state = f"unavailable: {error!r}"[:300]

    @property
    def waiting(self) -> bool:
        return self.state == "triton not imported"

    def miss(self, text: str) -> None:
        with self.lock:
            self.n_misses += 1
            if self.first_miss_k is None:
                self.first_miss_k = len(getattr(_RECORDER, "calls", ()))
            if len(self.misses) < 16:
                self.misses.append(text[:300])

    def note(self, text: str) -> None:
        with self.lock:
            if len(self.notes) < 16:
                self.notes.append(text[:300])

    def first_sight(self, tuner: Any, key: Any) -> bool:
        """Whether this is the first time the capture sees `key` of `tuner`
        (an autotuner can have chosen it before the capture was installed)."""
        slot = (id(tuner), repr(key))
        with self.lock:
            if slot in self.seen:
                return False
            self.seen.add(slot)
            return True

    def identity(self, tuner: Any) -> str:
        """The autotuned function, its key names and configs, and the
        autotuner's place among those that share them, in first-use order
        (one function can be tuned by several autotuners).

        First-use order is stable across runs only for sequential use:
        autotuners of one function first used from several threads can swap
        ordinals, which shows as misses (a record then fails). Record mode
        names an autotuner after its `run` and pin mode before it, which is
        the same order while no tuning happens inside another's `run`. The
        name is cached on the instance, also across `uninstall()`."""
        cached = vars(tuner).get(_IDENTITY)
        if cached is not None:
            return cached
        fn = getattr(tuner, "base_fn", None) or getattr(tuner, "fn", None)
        while fn is not None and not inspect.isfunction(fn) and hasattr(fn, "fn"):
            fn = fn.fn
        name = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}"
        configs = "|".join(sorted(_config_identity(c) for c in tuner.configs))
        digest = hashlib.sha256(configs.encode()).hexdigest()[:12]
        base = f"{name}[{','.join(map(str, tuner.keys))}]#{digest}"
        with self.lock:
            ordinal = self.identities.get(base, 0)
            self.identities[base] = ordinal + 1
        identity = f"{base}#{ordinal}"
        vars(tuner)[_IDENTITY] = identity
        return identity

    def record(self, tuner: Any, key: Any) -> None:
        if os.getpid() != self.pid or self.state != "on":
            return
        entry = {
            "autotuner": self.identity(tuner),
            "key": repr(key),
            "config": _config_identity(tuner.cache[key]),
        }
        with self.lock, self.path.open("a") as sink:
            sink.write(json.dumps(entry) + "\n")
            self.recorded += 1

    def summary(self) -> Dict[str, Any]:
        state = self.state
        if self.waiting and _AUTOTUNER_MODULE in sys.modules:
            state = f"missed: {_AUTOTUNER_MODULE} was imported but not patched"
        return {
            "pinning": state,
            "recorded": self.recorded,
            "pinned": self.pinned,
            "misses": self.n_misses,
            "first_miss_k": self.first_miss_k,
            "miss_head": self.misses[:8],
            "notes": self.notes[:8],
        }


def _config_identity(config: Any) -> str:
    return json.dumps(config.all_kwargs(), sort_keys=True, default=repr)


def _tuning_key(tuner: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """The key Triton's `Autotuner.run` keeps a choice under: the values of
    its key arguments, then the dtype of every argument that has one."""
    names = tuner.arg_names
    given = {**dict(zip(names, args)), **kwargs}
    picked = {name: value for name, value in given.items() if name in names}
    key = [picked[name] for name in tuner.keys if name in picked]
    key += [str(value.dtype) for value in picked.values() if hasattr(value, "dtype")]
    return tuple(key)


def _pin(
    pins: _AutotunePins, tuner: Any, key: Any, args: Any, kwargs: Any
) -> Optional[str]:
    """Make `tuner` choose the recorded config for `key` without
    benchmarking; that config's identity, or None after a miss."""
    recorded = _recorded(pins, tuner, key)
    if recorded is None:
        return None
    identity = pins.identity(tuner)
    matches = [c for c in tuner.configs if _config_identity(c) == recorded]
    if len(matches) > 1:
        pins.miss(f"{identity} {key!r}: {len(matches)} configs match the record")
        return None
    if matches:
        tuner.cache[key] = matches[0]
        # What Triton does after benchmarking, before the launch.
        nargs = dict(zip(tuner.arg_names, args))
        tuner.pre_hook({**nargs, **kwargs, **matches[0].all_kwargs()}, reset_only=True)
    elif pins.bench_hooked:
        # A config the autotuner makes up while tuning (NPOT variants, a
        # `prune_configs` override): `_bench` picks it.
        _fallback_pins()[id(tuner)] = recorded
    else:
        pins.miss(
            f"{identity} {key!r}: the recorded config is not one the autotuner "
            "declares, and this Triton's Autotuner has no _bench to pick it"
        )
        return None
    with pins.lock:
        pins.pinned += 1
    return recorded


def _recorded(pins: _AutotunePins, tuner: Any, key: Any) -> Optional[str]:
    """The recorded config for `key`, or None after a miss."""
    identity = pins.identity(tuner)
    slot = (identity, repr(key))
    recorded = pins.table.get(slot)
    if recorded is None:
        why = (
            "recorded with different configs"
            if slot in pins.table
            else "not in the golden's record"
        )
        pins.miss(f"{identity} {key!r}: {why}")
    return recorded


def _fallback_pins() -> Dict[int, str]:
    if not hasattr(_fallback, "pins"):
        _fallback.pins = {}
    return _fallback.pins


def _recording_run(original: Any) -> Any:
    @functools.wraps(original)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        pins = _AUTOTUNE
        cache = getattr(self, "cache", None)
        if pins is None or not isinstance(cache, dict):
            return original(self, *args, **kwargs)
        before = set(cache)
        try:
            key = (
                _tuning_key(self, args, kwargs)
                if len(getattr(self, "configs", ())) > 1
                else None
            )
        except Exception:  # noqa: B902
            key = None
        try:
            return original(self, *args, **kwargs)
        finally:
            chosen = [k for k in list(cache) if k not in before]
            for new in chosen:
                pins.first_sight(self, new)
            # A choice made before the capture was installed.
            if key is not None and key in before and pins.first_sight(self, key):
                chosen.append(key)
            for new in chosen:
                try:
                    pins.record(self, new)
                except Exception as error:  # noqa: B902
                    pins.note(f"cannot record an autotuner choice: {error!r}")

    return run


def _pinned_run(original: Any) -> Any:
    @functools.wraps(original)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        pins = _AUTOTUNE
        if pins is None or pins.state != "on" or len(getattr(self, "configs", ())) < 2:
            return original(self, *args, **kwargs)
        key, expected = None, None
        try:
            key = _tuning_key(self, args, kwargs)
            first = pins.first_sight(self, key)
            if key not in self.cache:
                expected = _pin(pins, self, key, args, kwargs)
            elif first:
                # Chosen before the capture was installed: checked before the
                # launch, which can record a tensor.
                recorded = _recorded(pins, self, key)
                chosen = self.cache[key]
                if recorded is not None and _config_identity(chosen) != recorded:
                    pins.miss(f"{pins.identity(self)} {key!r}: chose {chosen}")
        except Exception as error:  # noqa: B902
            pins.miss(f"{getattr(self, 'fn', '?')}: cannot pin: {error!r}")
        try:
            return original(self, *args, **kwargs)
        finally:
            _fallback_pins().pop(id(self), None)
            if expected is not None:
                chosen = self.cache.get(key)
                if chosen is None or _config_identity(chosen) != expected:
                    pins.miss(f"{pins.identity(self)} {key!r}: chose {chosen}")

    return run


def _pinned_bench(original: Any) -> Any:
    @functools.wraps(original)
    def _bench(self: Any, *args: Any, config: Any, **meta: Any) -> Any:
        recorded = _fallback_pins().get(id(self))
        if recorded is None:
            return original(self, *args, config=config, **meta)
        self._last_compiled_kernel = None
        fastest = _config_identity(config) == recorded
        return [0.0, 0.0, 0.0] if fastest else [float("inf")] * 3

    return _bench


def _pinned_disk_cache(original: Any) -> Any:
    @functools.wraps(original)
    def check_disk_cache(
        self: Any, tuning_key: Any, configs: Any, bench_fn: Any
    ) -> bool:
        # A pinned run neither reads timings measured elsewhere nor writes
        # its own (made up, or slowed down by CUTracer).
        bench_fn()
        return False

    return check_disk_cache


def _patch_autotuner(module: Any) -> None:
    pins = _AUTOTUNE
    if pins is None or not pins.waiting:
        return
    cls = getattr(module, "Autotuner", None)
    if cls is None or not callable(getattr(cls, "run", None)):
        pins.state = "unavailable: this Triton's Autotuner has no run"
        return
    if pins.mode == "record":
        _hook(cls, "run", _recording_run)
    else:
        _hook(cls, "run", _pinned_run)
        if callable(getattr(cls, "_bench", None)):
            _hook(cls, "_bench", _pinned_bench)
            pins.bench_hooked = True
        if callable(getattr(cls, "check_disk_cache", None)):
            _hook(cls, "check_disk_cache", _pinned_disk_cache)
    pins.state = "on"


class _PatchingLoader:
    """Runs a module's loader, then patches the autotuner it defined."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader

    def create_module(self, spec: Any) -> Any:
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: Any) -> None:
        self.loader.exec_module(module)
        _remove_import_hook()
        try:
            _patch_autotuner(module)
        except Exception as error:  # noqa: B902
            if _AUTOTUNE is not None:
                _AUTOTUNE.state = f"unavailable: {error!r}"[:300]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


class _AutotunerImportHook:
    """Patches Triton's autotuner as soon as it is imported: the capture is
    imported before the test, and must not import Triton itself. Hooks of
    other copies of this module are skipped (no recursion); such a copy then
    reports `missed`, which fails a record rather than passing silently."""

    _cutracer_numerics_import_hook = True

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != _AUTOTUNER_MODULE or getattr(_import_guard, "active", False):
            return None
        _import_guard.active = True
        try:
            for finder in list(sys.meta_path):
                if getattr(finder, _HOOK_MARK, False) or not hasattr(
                    finder, "find_spec"
                ):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is None:
                    continue
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _PatchingLoader(spec.loader)
                    return spec
                return None
        except Exception:  # noqa: B902
            return None
        finally:
            _import_guard.active = False
        return None


_AUTOTUNE: Optional[_AutotunePins] = None
_IMPORT_HOOK: Optional[_AutotunerImportHook] = None


def _remove_import_hook() -> None:
    global _IMPORT_HOOK
    if _IMPORT_HOOK is not None and _IMPORT_HOOK in sys.meta_path:
        sys.meta_path.remove(_IMPORT_HOOK)
    _IMPORT_HOOK = None


def _arm_autotune(mode: str, golden: Path, proc: str) -> None:
    """Record (`autotune` mode) or pin the autotuner's choices, now if Triton
    is imported, else right after it is."""
    global _AUTOTUNE, _IMPORT_HOOK
    _AUTOTUNE = _AutotunePins("record" if mode == "autotune" else "pin", golden, proc)
    if not _AUTOTUNE.waiting:
        return
    module = sys.modules.get(_AUTOTUNER_MODULE)
    if module is not None:
        _patch_autotuner(module)
        return
    _IMPORT_HOOK = _AutotunerImportHook()
    sys.meta_path.insert(0, _IMPORT_HOOK)


def _autotune_summary() -> Dict[str, Any]:
    return _AUTOTUNE.summary() if _AUTOTUNE is not None else {"pinning": "off"}


def _pin_hypothesis() -> None:
    """Same examples in every run, no deadline (CUTracer slows kernels), and
    no shrinking (it adds calls after a failure)."""
    global _previous_hypothesis_profile
    try:
        from hypothesis import HealthCheck, Phase, settings
    except ImportError:
        return
    _previous_hypothesis_profile = (
        getattr(settings, "_current_profile", None) or "default"
    )
    settings.register_profile(
        "cutracer_numerics",
        derandomize=True,
        database=None,
        deadline=None,
        suppress_health_check=list(HealthCheck),
        phases=[Phase.explicit, Phase.reuse, Phase.generate],
    )
    settings.load_profile("cutracer_numerics")


def _restore_hypothesis() -> None:
    global _previous_hypothesis_profile
    if _previous_hypothesis_profile is None:
        return
    try:
        from hypothesis import settings

        settings.load_profile(_previous_hypothesis_profile)
    except Exception:  # noqa: B902
        pass
    _previous_hypothesis_profile = None


def _after_fork_in_child() -> None:
    """A forked child does not write its parent's record."""
    global _RECORDER
    if _RECORDER is not None and _RECORDER.pid != os.getpid():
        _RECORDER = _Unlabeled(getattr(_RECORDER, "out", None))


def install() -> Any:
    """Hook the comparison functions and start recording, once; None when
    the environment asks for no capture."""
    global _RECORDER, _fork_hook_registered, _refused
    if _RECORDER is not None:
        return _RECORDER
    mode = os.environ.get(ENV_MODE, "")
    golden = os.environ.get(ENV_GOLDEN, "")
    if mode not in MODES or not golden:
        return None
    out = Path(os.environ[ENV_OUT]) if os.environ.get(ENV_OUT) else None
    if mode == "check" and out is None:
        if not _refused:
            print(
                f"cutracer_numerics: {ENV_OUT} is not set; nothing is checked",
                file=sys.stderr,
            )
            _refused = True
        return None
    proc = _process_label()
    recorder: Any = (
        _Unlabeled(out) if proc is None else _Recorder(mode, Path(golden), out, proc)
    )
    _RECORDER = recorder
    if mode != "autotune":
        _hook_comparisons(recorder)
        _wrap_entries(recorder)
    if proc is not None and (
        mode == "autotune" or os.environ.get(ENV_PIN_AUTOTUNE) != "0"
    ):
        try:
            _arm_autotune(mode, Path(golden), proc)
        except Exception as error:  # noqa: B902
            recorder.capture_error(f"cannot pin Triton autotuning: {error!r}")
    if os.environ.get(ENV_PIN_HYPOTHESIS) == "1":
        _pin_hypothesis()
    if not _fork_hook_registered and hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_after_fork_in_child)
        _fork_hook_registered = True
    atexit.register(finish)
    return recorder


def uninstall() -> None:
    """Restore every hooked function and stop recording (without writing)."""
    global _RECORDER, _refused, _AUTOTUNE
    while _HOOKS:
        owner, attr, original, wrapper, owned = _HOOKS.pop()
        if vars(owner).get(attr) is not wrapper:
            continue
        if owned:
            setattr(owner, attr, original)
        else:
            delattr(owner, attr)
    _unregister_script_builtins()
    _restore_hypothesis()
    _remove_import_hook()
    atexit.unregister(finish)
    _RECORDER = None
    _AUTOTUNE = None
    _refused = False


def record_tensor(name: str, value: Any) -> None:
    """Record one value explicitly (for a driver that compares nothing)."""
    recorder = install()
    if recorder is None:
        return
    try:
        recorder.observe(f"record:{name}", [("a", value)], _call_site() or "?")
    except Exception:  # noqa: B902
        recorder.capture_error(traceback.format_exc(limit=4))


def seed_everything() -> None:
    seed = os.environ.get(ENV_SEED, "0")
    if seed.lower() == "none":
        return
    value = int(seed)
    random.seed(value)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(value)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(value)


def finish() -> None:
    """Write this process's result, once."""
    recorder = _RECORDER
    if recorder is not None:
        recorder.write()


# pytest plugin hooks -------------------------------------------------------------


def pytest_configure(config: Any) -> None:
    install()


def pytest_runtest_setup(item: Any) -> None:
    if _RECORDER is not None:
        seed_everything()
    # In case Triton was imported in a way the import hook did not see.
    if _AUTOTUNE is not None and _AUTOTUNE.waiting and _AUTOTUNER_MODULE in sys.modules:
        _patch_autotuner(sys.modules[_AUTOTUNER_MODULE])


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    finish()


if os.environ.get(ENV_MODE) in MODES:
    try:
        install()
    except Exception as _error:  # noqa: B902
        print(
            f"cutracer_numerics: could not install the capture: {_error!r}",
            file=sys.stderr,
        )
