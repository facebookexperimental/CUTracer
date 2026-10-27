# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the OSS Compute Sanitizer parser."""

import unittest

from cutracer.compute_sanitizer.parser import (
    CheckerTool,
    parse_checker_log,
    parse_racecheck_log,
    RacecheckParseDiagnostic,
    RacecheckSeverity,
)
from tests.test_base import RACECHECK_CLEAN, RACECHECK_HAZARD


class ComputeSanitizerParserTest(unittest.TestCase):
    def test_clean_log_has_no_findings(self) -> None:
        result = parse_racecheck_log(RACECHECK_CLEAN.read_text())
        self.assertEqual(result.findings, [])
        self.assertIsNotNone(result.summary)
        assert result.summary is not None
        self.assertEqual(result.summary.total_hazards, 0)
        self.assertEqual(result.summary.errors, 0)
        self.assertEqual(result.summary.warnings, 0)
        self.assertTrue(result.is_complete)

    def test_hazard_log_parses_neutral_records(self) -> None:
        result = parse_racecheck_log(RACECHECK_HAZARD.read_text())
        self.assertEqual(len(result.findings), 2)

        error, warning = result.findings
        self.assertEqual(error.severity, RacecheckSeverity.ERROR)
        self.assertEqual(error.max_hazards, 5)
        self.assertEqual(error.access_kinds, ["Write", "Read"])
        self.assertEqual(
            error.locations, ["kernels.py:42", "kernels.py:57", "kernels.py:58"]
        )
        self.assertEqual(len(error.accesses), 3)
        self.assertEqual(error.accesses[0].role, "primary")
        self.assertEqual(error.accesses[0].function, "kernel_a")
        self.assertEqual(error.accesses[1].hazards, 3)
        self.assertIn("Race reported between", error.raw_block)
        self.assertIn("child process output", RACECHECK_HAZARD.read_text())

        self.assertEqual(warning.severity, RacecheckSeverity.WARNING)
        self.assertEqual(warning.max_hazards, 2)
        self.assertEqual(warning.access_kinds, ["Read", "Write"])

        self.assertIsNotNone(result.summary)
        assert result.summary is not None
        self.assertEqual(result.summary.total_hazards, 7)
        self.assertEqual(result.summary.errors, 6)
        self.assertEqual(result.summary.warnings, 1)
        self.assertTrue(result.is_complete)

    def test_empty_and_malformed_input(self) -> None:
        for text in ("", "garbage\n", "========= COMPUTE-SANITIZER\n"):
            result = parse_racecheck_log(text)
            self.assertEqual(result.findings, [])
            self.assertIsNone(result.summary)
            self.assertIn(RacecheckParseDiagnostic.MISSING_SUMMARY, result.diagnostics)

    def test_internal_tool_error_makes_racecheck_incomplete(self) -> None:
        text = (
            "========= Internal Sanitizer Error: The Sanitizer failed to handle "
            "a hardware exception.\n"
            "========= RACECHECK SUMMARY: 1 hazard displayed "
            "(1 error, 0 warnings)\n"
        )

        result = parse_racecheck_log(text)

        self.assertIn(
            RacecheckParseDiagnostic.INTERNAL_TOOL_ERROR,
            result.diagnostics,
        )
        self.assertFalse(result.is_complete)

    def test_internal_tool_error_prevents_zero_summary_from_claiming_clean(
        self,
    ) -> None:
        result = parse_checker_log(
            CheckerTool.INITCHECK,
            "========= Internal Sanitizer Error: failed to initialize.\n"
            "========= ERROR SUMMARY: 0 errors\n",
        )

        self.assertIn(
            RacecheckParseDiagnostic.INTERNAL_TOOL_ERROR,
            result.diagnostics,
        )
        self.assertFalse(result.is_complete)

    def test_summary_and_conflict_singular_grammar(self) -> None:
        text = (
            "========= Error: Race reported between Read access at "
            "kernel+0x100 in test.py:7\n"
            "========= and Write access at kernel+0x200 in test.py:9 "
            "[1 hazard]\n"
            "========= RACECHECK SUMMARY: 1 hazard displayed "
            "(1 error, 0 warnings)\n"
        )
        result = parse_racecheck_log(text)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].max_hazards, 1)
        self.assertIsNotNone(result.summary)
        assert result.summary is not None
        self.assertEqual(result.summary.total_hazards, 1)

    def test_warning_only_summary_is_complete(self) -> None:
        text = (
            "========= Warning: Race reported between Read access at "
            "kernel+0x100 in test.py:7\n"
            "========= and Write access at kernel+0x200 in test.py:9 "
            "[2 hazards]\n"
            "========= RACECHECK SUMMARY: 2 hazards displayed "
            "(0 errors, 2 warnings)\n"
        )
        result = parse_racecheck_log(text)
        self.assertTrue(result.is_complete)
        self.assertEqual(result.findings[0].severity, RacecheckSeverity.WARNING)

    def test_summary_counts_are_not_finding_block_cardinality(self) -> None:
        text = (
            "========= Error: Race reported between Read access at "
            "kernel+0x100 in test.py:7\n"
            "========= and Write access at kernel+0x200 in test.py:9 "
            "[100 hazards]\n"
            "========= RACECHECK SUMMARY: 100 hazards displayed "
            "(316 errors, 1 warning)\n"
        )
        result = parse_racecheck_log(text)
        self.assertTrue(result.is_complete)
        self.assertEqual(len(result.findings), 1)

    def test_incomplete_and_inconsistent_logs_are_diagnosed(self) -> None:
        complete_block = (
            "========= Error: Race reported between Read access at "
            "kernel+0x100 in test.py:7\n"
            "========= and Write access at kernel+0x200 in test.py:9 "
            "[1 hazard]\n"
        )
        cases = (
            (
                complete_block,
                RacecheckParseDiagnostic.MISSING_SUMMARY,
            ),
            (
                "========= RACECHECK SUMMARY: 0 issues found\n",
                RacecheckParseDiagnostic.MALFORMED_SUMMARY,
            ),
            (
                "========= RACECHECK SUMMARY: 0 hazards displayed "
                "(0 errors, 0 warnings)\n"
                "========= RACECHECK SUMMARY: 0 hazards displayed "
                "(0 errors, 0 warnings)\n",
                RacecheckParseDiagnostic.MULTIPLE_SUMMARIES,
            ),
            (
                "========= Error: Race reported between Read access at "
                "kernel+0x100 in test.py:7\n"
                "========= RACECHECK SUMMARY: 1 hazard displayed "
                "(1 error, 0 warnings)\n",
                RacecheckParseDiagnostic.INCOMPLETE_FINDING,
            ),
            (
                "========= Error: Race reported between an unknown format\n"
                "========= RACECHECK SUMMARY: 1 hazard displayed "
                "(1 error, 0 warnings)\n",
                RacecheckParseDiagnostic.UNPARSED_FINDING,
            ),
            (
                "========= RACECHECK SUMMARY: 1 hazard displayed "
                "(1 error, 0 warnings)\n",
                RacecheckParseDiagnostic.SUMMARY_FINDING_MISMATCH,
            ),
            (
                complete_block + "========= RACECHECK SUMMARY: 0 hazards displayed "
                "(0 errors, 0 warnings)\n",
                RacecheckParseDiagnostic.SUMMARY_FINDING_MISMATCH,
            ),
            (
                "========= RACECHECK SUMMARY: 0 hazards displayed "
                "(1 error, 0 warnings)\n",
                RacecheckParseDiagnostic.SUMMARY_COUNT_MISMATCH,
            ),
        )
        for text, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic.value):
                result = parse_racecheck_log(text)
                self.assertFalse(result.is_complete)
                self.assertIn(diagnostic, result.diagnostics)

    def test_invalid_access_site_uses_full_lookahead_budget(self) -> None:
        text = (
            "========= Potential invalid __shared__ read of size 8 bytes\n"
            "=========     by thread (0,0,0) in block (0,0,0)\n"
            "=========     Address 0x10 is out of bounds\n"
            "=========     device frame follows\n"
            "=========     at k+0x20 in t.py:3\n"
            "========= RACECHECK SUMMARY: 1 hazard displayed "
            "(1 error, 0 warnings)\n"
        )

        result = parse_racecheck_log(text)

        self.assertEqual(len(result.invalid_accesses), 1)
        access = result.invalid_accesses[0].access
        self.assertIsNotNone(access)
        assert access is not None
        self.assertEqual(access.function, "k")

    def test_invalid_access_site_accepts_a_source_path_with_spaces(self) -> None:
        text = (
            "========= Potential invalid __shared__ read of size 8 bytes\n"
            "=========     at k+0x20 in /tmp/source tree/t.py:3\n"
            "========= RACECHECK SUMMARY: 1 hazard displayed "
            "(1 error, 0 warnings)\n"
        )

        result = parse_racecheck_log(text)

        access = result.invalid_accesses[0].access
        self.assertIsNotNone(access)
        assert access is not None
        self.assertEqual(
            (access.function, access.offset, access.file, access.line),
            ("k", "0x20", "/tmp/source tree/t.py", 3),
        )

    def test_truncation_without_a_report_is_incomplete(self) -> None:
        text = (
            "========= Warning: Maximum number of hazards reached, "
            "output will be truncated\n"
            "========= RACECHECK SUMMARY: 100 hazards displayed "
            "(316 errors, 1 warning)\n"
        )

        result = parse_racecheck_log(text)

        self.assertTrue(result.output_truncated)
        self.assertIn(
            RacecheckParseDiagnostic.SUMMARY_FINDING_MISMATCH,
            result.diagnostics,
        )
        self.assertFalse(result.is_complete)

    def test_first_diagnostic_glued_to_child_output_is_parsed(self) -> None:
        # The child process and the sanitizer share a stream, so the opening
        # diagnostic regularly lands on the tail of a pytest progress line.
        text = (
            "test_foo.py ========= Error: Race reported between Read access at "
            "kernel+0x100 in test.py:7\n"
            "========= and Write access at kernel+0x200 in test.py:9 "
            "[1 hazard]\n"
            "========= RACECHECK SUMMARY: 1 hazard displayed "
            "(1 error, 0 warnings)\n"
        )
        result = parse_racecheck_log(text)
        self.assertTrue(result.is_complete)
        self.assertEqual(len(result.findings), 1)

    def test_hazard_display_cap_is_truncation_not_a_finding(self) -> None:
        text = (
            "========= Error: Race reported between Read access at "
            "kernel+0x100 in test.py:7\n"
            "========= and Write access at kernel+0x200 in test.py:9 "
            "[1 hazard]\n"
            "========= Warning: Maximum number of hazards reached, "
            "output will be truncated\n"
            "========= RACECHECK SUMMARY: 100 hazards displayed "
            "(316 errors, 1 warning)\n"
        )
        result = parse_racecheck_log(text)
        self.assertTrue(result.output_truncated)
        # A display cap does not make the summary counts unknown.
        self.assertTrue(result.is_complete)
        self.assertEqual(len(result.findings), 1)

    def test_potential_invalid_shared_access_is_kept_out_of_findings(self) -> None:
        text = (
            "========= Potential invalid __shared__ write of size 4 bytes\n"
            "=========     at remote_shmem_store_kernel+0x150 "
            "in test_tlx_cluster.py:88\n"
            "========= RACECHECK SUMMARY: 1 hazard displayed "
            "(1 error, 0 warnings)\n"
        )
        result = parse_racecheck_log(text)
        # It carries one access, not a pair, so paired-access consumers must
        # not see it -- but it still proves the run reported something.
        self.assertEqual(result.findings, [])
        self.assertEqual(len(result.invalid_accesses), 1)
        self.assertTrue(result.has_positive_signal)
        self.assertTrue(result.is_complete)
        invalid = result.invalid_accesses[0]
        self.assertEqual(invalid.access_type, "write")
        self.assertEqual(invalid.size_bytes, 4)
        assert invalid.access is not None
        self.assertEqual(invalid.access.function, "remote_shmem_store_kernel")
        self.assertEqual(invalid.access.line, 88)

    def test_invalid_access_site_survives_an_intervening_line(self) -> None:
        text = (
            "========= Potential invalid __shared__ read of size 8 bytes\n"
            "=========     by thread (0,0,0) in block (0,0,0)\n"
            "=========     at k+0x20 in t.py:3\n"
            "========= RACECHECK SUMMARY: 1 hazard displayed "
            "(1 error, 0 warnings)\n"
        )
        result = parse_racecheck_log(text)
        self.assertEqual(len(result.invalid_accesses), 1)
        access = result.invalid_accesses[0].access
        assert access is not None
        self.assertEqual(access.function, "k")

    def test_invalid_access_without_site_does_not_consume_next_finding(self) -> None:
        text = (
            "========= Potential invalid __shared__ read of size 8 bytes\n"
            "========= Error: Race reported between Read access at "
            "kernel+0x100 in test.py:7\n"
            "========= and Write access at kernel+0x200 in test.py:9 "
            "[1 hazard]\n"
            "========= RACECHECK SUMMARY: 2 hazards displayed "
            "(2 errors, 0 warnings)\n"
        )
        result = parse_racecheck_log(text)
        self.assertEqual(len(result.invalid_accesses), 1)
        self.assertIsNone(result.invalid_accesses[0].access)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].accesses[0].function, "kernel")


class CheckerLayerTest(unittest.TestCase):
    """The tool-agnostic projection used by evidence conversion."""

    def test_initcheck_issue_and_site(self) -> None:
        text = (
            "test_addmm.py ========= Uninitialized __global__ memory read "
            "of size 16 bytes\n"
            "=========     at void at::native::vectorized_elementwise_kernel"
            "<(int)8, std::array<char *, (unsigned long)3>>(int, T2, T3)+0x110\n"
            "=========     by thread (64,0,0) in block (6,0,0)\n"
            "========= ERROR SUMMARY: 393216 errors\n"
        )
        result = parse_checker_log(CheckerTool.INITCHECK, text)
        self.assertTrue(result.is_complete)
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertEqual(
            issue.message, "Uninitialized __global__ memory read of size 16 bytes"
        )
        site = issue.primary_site
        assert site is not None
        # The demangled symbol contains spaces, angle brackets, and commas; the
        # offset must still be split off cleanly from the right.
        self.assertTrue(site.symbol.startswith("void at::native::"))
        self.assertEqual(site.offset, "0x110")
        self.assertIsNone(site.location)
        assert result.summary is not None
        self.assertEqual(result.summary.errors, 393216)

    def test_checker_issue_preserves_multiple_sites(self) -> None:
        text = (
            "========= Uninitialized __global__ memory read of size 4 bytes\n"
            "=========     at leaf+0x10 in leaf.py:3\n"
            "=========     at caller+0x20 in caller.py:7\n"
            "========= ERROR SUMMARY: 1 error\n"
        )

        result = parse_checker_log(CheckerTool.INITCHECK, text)

        self.assertEqual(
            [site.symbol for site in result.issues[0].sites],
            ["leaf", "caller"],
        )

    def test_checker_site_requires_a_trailing_hex_offset(self) -> None:
        text = (
            "========= Uninitialized __global__ memory read of size 4 bytes\n"
            "=========     at symbol_with<literal+0x10>\n"
            "========= ERROR SUMMARY: 1 error\n"
        )

        result = parse_checker_log(CheckerTool.INITCHECK, text)

        site = result.issues[0].primary_site
        self.assertIsNotNone(site)
        assert site is not None
        self.assertEqual(site.symbol, "symbol_with<literal+0x10>")
        self.assertIsNone(site.offset)

    def test_checker_site_source_path_may_contain_spaces(self) -> None:
        text = (
            "========= Invalid __global__ read of size 4 bytes at 0x7f00\n"
            "=========     at foo_kernel+0x40 in /tmp/source tree/foo.py:12\n"
            "========= ERROR SUMMARY: 1 error\n"
        )

        result = parse_checker_log(CheckerTool.MEMCHECK, text)

        site = result.issues[0].primary_site
        self.assertIsNotNone(site)
        assert site is not None
        self.assertEqual(
            (site.symbol, site.offset, site.location),
            ("foo_kernel", "0x40", "/tmp/source tree/foo.py:12"),
        )

    def test_synccheck_issue_without_source_location(self) -> None:
        text = (
            "========= Barrier error detected. Missing init.\n"
            "=========     at tmem_load_store_kernel+0xffffffffffffff20\n"
            "========= ERROR SUMMARY: 1 error\n"
        )
        result = parse_checker_log(CheckerTool.SYNCCHECK, text)
        self.assertTrue(result.is_complete)
        self.assertEqual(
            result.issues[0].message, "Barrier error detected. Missing init"
        )
        site = result.issues[0].primary_site
        assert site is not None
        self.assertEqual(site.symbol, "tmem_load_store_kernel")
        self.assertIsNone(site.location)

    def test_synccheck_message_only_removes_its_final_delimiter(self) -> None:
        text = (
            "========= Barrier error detected. Waiting...\n"
            "=========     at tmem_load_store_kernel+0x20\n"
            "========= ERROR SUMMARY: 1 error\n"
        )

        result = parse_checker_log(CheckerTool.SYNCCHECK, text)

        self.assertEqual(
            result.issues[0].message,
            "Barrier error detected. Waiting..",
        )

    def test_memcheck_message_drops_the_per_run_address(self) -> None:
        text = (
            "========= Invalid __global__ read of size 4 bytes at 0x7f06e0203400\n"
            "=========     at foo_kernel+0x40 in foo.py:12\n"
            "========= ERROR SUMMARY: 1 error\n"
        )
        result = parse_checker_log(CheckerTool.MEMCHECK, text)
        self.assertEqual(
            result.issues[0].message, "Invalid __global__ read of size 4 bytes"
        )
        site = result.issues[0].primary_site
        assert site is not None
        self.assertEqual(site.location, "foo.py:12")

    def test_memcheck_message_preserves_non_address_at_phrase(self) -> None:
        text = (
            "========= Invalid access at synchronization boundary at 0x7f00\n"
            "=========     at foo_kernel+0x40 in foo.py:12\n"
            "========= ERROR SUMMARY: 1 error\n"
        )
        result = parse_checker_log(CheckerTool.MEMCHECK, text)

        self.assertEqual(
            result.issues[0].message,
            "Invalid access at synchronization boundary",
        )

    def test_memcheck_api_error_echo_keeps_the_device_error_complete(self) -> None:
        text = (
            "========= Invalid __global__ write of size 4 bytes\n"
            "=========     at kernel+0x70 in k.py:6\n"
            "=========     by thread (0,0,0) in block (0,0,0)\n"
            "=========     Address 0x7f671ac00010 is out of bounds\n"
            "=========     and is 1 bytes after the nearest allocation at "
            "0x7f671ac00000 of size 16 bytes\n"
            "=========     Saved host backtrace up to driver entry point at "
            "kernel launch time\n"
            "=========     Host Frame: [0x2774ec] in "
            "/lib/x86_64-linux-gnu/libcuda.so.1\n"
            "=========\n"
            "========= Program hit cudaErrorLaunchFailure (error 719) due to "
            '"unspecified launch failure" on CUDA API call to '
            "cudaDeviceSynchronize.\n"
            "=========     Saved host backtrace up to driver entry point at error\n"
            "=========     Host Frame: [0x4772b5] in "
            "/lib/x86_64-linux-gnu/libcuda.so.1\n"
            "=========\n"
            "========= ERROR SUMMARY: 2 errors\n"
        )

        result = parse_checker_log(CheckerTool.MEMCHECK, text)

        self.assertTrue(result.is_complete)
        self.assertEqual(result.diagnostics, [])
        device_error, api_error = result.issues
        assert device_error.primary_site is not None
        self.assertEqual(device_error.primary_site.location, "k.py:6")
        self.assertTrue(
            api_error.message.startswith("Program hit cudaErrorLaunchFailure")
        )
        self.assertEqual(api_error.sites, [])

    def test_only_device_side_issues_require_a_site(self) -> None:
        cases = (
            ("Leaked 1024 bytes at 0x7f0d6e000000", True),
            (
                "Program hit cudaErrorInvalidValue (error 1) due to "
                '"invalid argument" on CUDA API call to cudaFuncSetAttribute.',
                True,
            ),
            ("Invalid __global__ read of size 4 bytes at 0x7f0d6e000000", False),
        )
        for message, complete in cases:
            with self.subTest(message=message):
                result = parse_checker_log(
                    CheckerTool.MEMCHECK,
                    f"========= {message}\n"
                    "=========     Saved host backtrace up to driver entry point\n"
                    "========= ERROR SUMMARY: 1 error\n",
                )

                self.assertEqual(result.is_complete, complete)
                self.assertEqual(
                    RacecheckParseDiagnostic.INCOMPLETE_FINDING in result.diagnostics,
                    not complete,
                )

    def test_checker_raw_block_keeps_every_lookahead_line(self) -> None:
        text = (
            "========= Invalid __global__ write of size 4 bytes\n"
            "=========     at kernel+0x70 in k.py:6\n"
            "=========     by thread (0,0,0) in block (0,0,0)\n"
            "=========     Address 0x7f671ac00010 is out of bounds\n"
            "=========     and is 1 bytes after the nearest allocation at "
            "0x7f671ac00000 of size 16 bytes\n"
            "========= ERROR SUMMARY: 1 error\n"
        )

        result = parse_checker_log(CheckerTool.MEMCHECK, text)

        self.assertEqual(
            result.issues[0].raw_block.splitlines(),
            [
                "Invalid __global__ write of size 4 bytes",
                "at kernel+0x70 in k.py:6",
                "by thread (0,0,0) in block (0,0,0)",
                "Address 0x7f671ac00010 is out of bounds",
            ],
        )

    def test_checker_site_lookahead_ignores_non_sanitizer_output(self) -> None:
        text = "\n".join(
            [
                "========= Invalid __global__ read of size 4 bytes at 0x7f00",
                *(f"application output {index}" for index in range(20)),
                "=========     at foo_kernel+0x40 in foo.py:12",
                "========= ERROR SUMMARY: 1 error",
            ]
        )
        result = parse_checker_log(CheckerTool.MEMCHECK, text)

        site = result.issues[0].primary_site
        self.assertIsNotNone(site)
        assert site is not None
        self.assertEqual(site.location, "foo.py:12")

    def test_not_printed_note_is_truncation_not_a_second_summary(self) -> None:
        text = (
            "========= Uninitialized __global__ memory read of size 4 bytes\n"
            "=========     at k+0x10\n"
            "========= ERROR SUMMARY: 393216 errors\n"
            "========= ERROR SUMMARY: 393116 errors were not printed. "
            "Use --print-limit option to adjust the number of printed errors\n"
        )
        result = parse_checker_log(CheckerTool.INITCHECK, text)
        self.assertNotIn(
            RacecheckParseDiagnostic.MULTIPLE_SUMMARIES, result.diagnostics
        )
        self.assertTrue(result.output_truncated)
        self.assertTrue(result.is_complete)
        assert result.summary is not None
        self.assertEqual(result.summary.errors, 393216)
        self.assertEqual(result.summary.not_printed, 393116)

    def test_issue_lookahead_stops_at_the_next_issue(self) -> None:
        text = (
            "========= Uninitialized __global__ memory read of size 4 bytes\n"
            "========= Uninitialized __shared__ memory read of size 8 bytes\n"
            "=========     at second_kernel+0x20 in test.py:9\n"
            "========= ERROR SUMMARY: 2 errors\n"
        )
        result = parse_checker_log(CheckerTool.INITCHECK, text)
        self.assertEqual(len(result.issues), 2)
        self.assertEqual(result.issues[0].sites, [])
        site = result.issues[1].primary_site
        assert site is not None
        self.assertEqual(site.location, "test.py:9")

    def test_missing_summary_is_unknown_not_truncated(self) -> None:
        # A run killed before it printed a summary is incomplete; that is a
        # different fact from a run that capped how much it printed.
        text = (
            "========= Uninitialized __global__ memory read of size 4 bytes\n"
            "=========     at k+0x10\n"
        )
        result = parse_checker_log(CheckerTool.INITCHECK, text)
        self.assertFalse(result.output_truncated)
        self.assertFalse(result.is_complete)
        self.assertIn(RacecheckParseDiagnostic.MISSING_SUMMARY, result.diagnostics)

    def test_clean_run_reports_zero_and_stays_complete(self) -> None:
        for tool in (
            CheckerTool.INITCHECK,
            CheckerTool.MEMCHECK,
            CheckerTool.SYNCCHECK,
        ):
            with self.subTest(tool=tool.value):
                result = parse_checker_log(tool, "========= ERROR SUMMARY: 0 errors\n")
                self.assertTrue(result.is_complete)
                self.assertEqual(result.issues, [])
                assert result.summary is not None
                self.assertTrue(result.summary.is_zero)

    def test_positive_summary_without_blocks_is_inconsistent(self) -> None:
        result = parse_checker_log(
            CheckerTool.SYNCCHECK, "========= ERROR SUMMARY: 3 errors\n"
        )
        self.assertIn(
            RacecheckParseDiagnostic.SUMMARY_FINDING_MISMATCH, result.diagnostics
        )
        self.assertFalse(result.is_complete)

    def test_racecheck_projects_both_sides_of_the_conflict(self) -> None:
        text = (
            "========= Error: Race reported between Read access at "
            "kernel+0x100 in test.py:7\n"
            "========= and Write access at kernel+0x200 in test.py:9 "
            "[4 hazards]\n"
            "========= Potential invalid __shared__ read of size 8 bytes\n"
            "=========     at kernel+0x300 in test.py:11\n"
            "========= RACECHECK SUMMARY: 5 hazards displayed "
            "(5 errors, 0 warnings)\n"
        )
        result = parse_checker_log(CheckerTool.RACECHECK, text)
        self.assertEqual(result.tool, CheckerTool.RACECHECK)
        self.assertEqual(len(result.issues), 2)
        race = result.issues[0]
        self.assertEqual(race.message, "Race reported between Read access")
        self.assertEqual(race.severity, RacecheckSeverity.ERROR)
        self.assertEqual(race.hazards, 4)
        self.assertEqual(
            [site.location for site in race.sites], ["test.py:7", "test.py:9"]
        )
        self.assertEqual(
            result.issues[1].message,
            "Potential invalid __shared__ read of size 8 bytes",
        )
        # Racecheck bills these against the error count, not the warning count:
        # a corpus log with two such blocks and no races summarizes as
        # "2 hazards displayed (2 errors, 0 warnings)". Downgrading them to
        # warnings would misreport what the tool decided.
        self.assertEqual(result.issues[1].severity, RacecheckSeverity.ERROR)
        assert result.summary is not None
        self.assertEqual(result.summary.total_hazards, 5)

    def test_unrelated_child_output_is_ignored(self) -> None:
        text = (
            "============================= test session starts "
            "==============================\n"
            "collected 1 item\n"
            "========= ERROR SUMMARY: 0 errors\n"
        )
        result = parse_checker_log(CheckerTool.INITCHECK, text)
        self.assertEqual(result.issues, [])
        self.assertTrue(result.is_complete)


if __name__ == "__main__":
    unittest.main()
