# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import unittest

from cutracer.isa.amdgcn.lds import decode_instruction


class AMDLDSDecoderTest(unittest.TestCase):
    def test_access_shapes(self) -> None:
        cases = (
            (
                "ds_read_b128 v[208:211], v184 offset:32",
                ("read", "v184", [(32, 16)]),
            ),
            (
                "ds_write2_b32 v160, v204, v205 offset0:8 offset1:12",
                ("write", "v160", [(32, 4), (48, 4)]),
            ),
        )
        for asm, expected in cases:
            with self.subTest(asm=asm):
                decoded = decode_instruction(asm)
                self.assertIsNotNone(decoded)
                assert decoded is not None
                ranges = [
                    (access.offset_bytes, access.width_bytes)
                    for access in decoded.accesses
                ]
                self.assertEqual(
                    expected,
                    (decoded.kind, decoded.address_vgpr, ranges),
                )

    def test_barrier_shuffle_and_unsupported_forms(self) -> None:
        cases = (
            ("s_barrier", ("barrier", True)),
            ("s_barrier_signal 1", ("unsupported", False)),
            ("ds_read_b128 v[0:3], v1 gds", ("unsupported", False)),
            ("ds_bpermute_b32 v0, v1, v2", ("ignored", True)),
            ("ds_swizzle_b32 v0, v1 offset:1", ("unsupported", False)),
            ("ds_read_b128 v[0:3], v1 offset0:1", ("unsupported", False)),
            ("ds_write2_b32 v0, v1, v2 offset:1", ("unsupported", False)),
        )
        for asm, expected in cases:
            with self.subTest(asm=asm):
                decoded = decode_instruction(asm)
                self.assertIsNotNone(decoded)
                assert decoded is not None
                self.assertEqual(expected, (decoded.kind, decoded.supported))

        self.assertIsNone(decode_instruction("v_add_u32 v0, v1, v2"))
