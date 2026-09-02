# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the OSS Compute Sanitizer racecheck parser."""

import unittest

from cutracer.compute_sanitizer.parser import (
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


if __name__ == "__main__":
    unittest.main()
