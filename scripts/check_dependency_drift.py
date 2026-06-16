#!/usr/bin/env python
"""Report newer releases for direct pinned project dependencies."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
SPECIFIER_OPERATORS = ("==", ">=", "<=", "~=", "!=", ">", "<")
NORMALIZED_NAME_SEPARATOR_RE = re.compile(r"[-_.]+")


def normalize_package_name(name: str) -> str:
    return NORMALIZED_NAME_SEPARATOR_RE.sub("-", name).lower()


def requirement_name(requirement: str) -> str:
    for operator in SPECIFIER_OPERATORS:
        if operator in requirement:
            return requirement.split(operator, maxsplit=1)[0].strip()
    return requirement.strip()


def direct_dependency_names(pyproject_path: Path = PYPROJECT) -> set[str]:
    pyproject = tomllib.loads(pyproject_path.read_text())
    requirements = list(pyproject["project"]["dependencies"])
    for optional_requirements in pyproject["project"].get("optional-dependencies", {}).values():
        requirements.extend(optional_requirements)
    return {requirement_name(requirement) for requirement in requirements}


def direct_outdated_packages(outdated_packages: list[dict[str, str]], direct_names: set[str]) -> list[dict[str, str]]:
    normalized_direct_names = {normalize_package_name(name) for name in direct_names}
    return sorted(
        (package for package in outdated_packages if normalize_package_name(package["name"]) in normalized_direct_names),
        key=lambda package: normalize_package_name(package["name"]),
    )


def load_outdated_packages() -> list[dict[str, str]]:
    result = subprocess.run(
        (
            "uv",
            "pip",
            "list",
            "--outdated",
            "--format",
            "json",
            "--exclude-editable",
        ),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def format_report(packages: list[dict[str, str]]) -> str:
    if not packages:
        return "All direct pinned dependencies are current."

    rows = [("Package", "Current", "Latest")]
    rows.extend((package["name"], package["version"], package["latest_version"]) for package in packages)
    widths = tuple(max(len(row[idx]) for row in rows) for idx in range(3))
    return "\n".join(f"{name:<{widths[0]}}  {current:<{widths[1]}}  {latest:<{widths[2]}}" for name, current, latest in rows)


def main() -> int:
    direct_names = direct_dependency_names()
    outdated_packages = load_outdated_packages()
    print(format_report(direct_outdated_packages(outdated_packages, direct_names)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
