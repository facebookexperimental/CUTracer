# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Guard that requirements.txt mirrors the dependencies in pyproject.toml.

pyproject.toml is the canonical source; requirements.txt flattens
[project.dependencies] and [project.optional-dependencies] into a single
file for `pip install -r requirements.txt`. The two have drifted before
(e.g. the Python 3.10 `importlib_resources` backport existed only in
requirements.txt), so compare them section by section here. A dependency
moved between sections must fail even though the union is unchanged.
"""

from __future__ import annotations

import re
import types
import unittest
from pathlib import Path


def _toml_parser() -> types.ModuleType | None:
    """Return the stdlib TOML parser, or None on Python < 3.11."""
    try:
        import tomllib
    except ModuleNotFoundError:
        return None
    return tomllib


# Section headers in requirements.txt -> key in
# [project.optional-dependencies] ("core" means [project.dependencies]).
_SECTION_HEADERS = {
    "# Core dependencies": "core",
    "# CLP support": "clp",
    "# Development dependencies": "dev",
}

_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _find_manifest_dir() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (
            (parent / "pyproject.toml").is_file()
            and (parent / "requirements.txt").is_file()
            and (parent / "cutracer").is_dir()
        ):
            return parent
    return None


def _normalize(requirement: str) -> tuple[str, str]:
    """Return (normalized package name, normalized requirement string)."""
    match = _NAME_RE.match(requirement.strip())
    if match is None:
        raise ValueError(f"Cannot parse requirement: {requirement!r}")
    name = re.sub(r"[-_.]+", "-", match.group(1).lower())
    rest = re.sub(r"\s+", "", requirement.strip()[match.end() :])
    return name, name + rest.replace("'", '"')


def _to_dict(requirements: list[str], source: str) -> tuple[dict[str, str], list[str]]:
    result: dict[str, str] = {}
    duplicates: list[str] = []
    for requirement in requirements:
        name, normalized = _normalize(requirement)
        if name in result:
            duplicates.append(f"duplicate {name!r} in {source}")
        result[name] = normalized
    return result, duplicates


def _parse_requirements_txt(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {key: [] for key in _SECTION_HEADERS.values()}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            for header, key in _SECTION_HEADERS.items():
                if line.startswith(header):
                    current = key
                    break
            continue
        if current is None:
            raise ValueError(f"Requirement before any section header: {line!r}")
        sections[current].append(line)
    return sections


class DependencySyncTest(unittest.TestCase):
    """requirements.txt sections must equal the pyproject.toml sections."""

    @classmethod
    def setUpClass(cls) -> None:
        toml_parser = _toml_parser()
        if toml_parser is None:
            raise unittest.SkipTest(
                "tomllib requires Python >= 3.11; dependency sync is "
                "enforced by the dependency-sync GitHub workflow"
            )
        manifest_dir = _find_manifest_dir()
        if manifest_dir is None:
            raise unittest.SkipTest(
                "pyproject.toml/requirements.txt not found next to the "
                "source tree (e.g. packaged test sandbox); dependency "
                "sync is enforced by the dependency-sync GitHub workflow"
            )
        with open(manifest_dir / "pyproject.toml", "rb") as f:
            pyproject = toml_parser.load(f)
        project = pyproject["project"]
        optional = project.get("optional-dependencies", {})
        cls.expected = {
            "core": project.get("dependencies", []),
            "clp": optional.get("clp", []),
            "dev": optional.get("dev", []),
        }
        cls.actual = _parse_requirements_txt(manifest_dir / "requirements.txt")

    def _assert_section_in_sync(self, section: str) -> None:
        expected, dup_expected = _to_dict(
            self.expected[section], f"pyproject.toml [{section}]"
        )
        actual, dup_actual = _to_dict(
            self.actual[section], f"requirements.txt [{section}]"
        )
        problems = dup_expected + dup_actual
        for name in sorted(expected.keys() - actual.keys()):
            problems.append(f"{name}: only in pyproject.toml ({expected[name]})")
        for name in sorted(actual.keys() - expected.keys()):
            problems.append(f"{name}: only in requirements.txt ({actual[name]})")
        for name in sorted(expected.keys() & actual.keys()):
            if expected[name] != actual[name]:
                problems.append(
                    f"{name}: pyproject.toml has {expected[name]!r}, "
                    f"requirements.txt has {actual[name]!r}"
                )
        self.assertEqual(
            problems,
            [],
            f"Dependency section [{section}] is out of sync; update "
            "pyproject.toml and requirements.txt together:\n" + "\n".join(problems),
        )

    def test_core_dependencies_in_sync(self) -> None:
        self._assert_section_in_sync("core")

    def test_clp_dependencies_in_sync(self) -> None:
        self._assert_section_in_sync("clp")

    def test_dev_dependencies_in_sync(self) -> None:
        self._assert_section_in_sync("dev")


if __name__ == "__main__":
    unittest.main()
