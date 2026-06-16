#!/usr/bin/env python
"""Run tests against candidate JAX pins in a temporary project copy."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_ROOT = PROJECT_ROOT / "tmp" / "dependency-upgrade-probe"
PINNED_REQUIREMENT_RE = re.compile(r'(?P<prefix>\s*")(?P<name>{name})==(?P<version>[^"]+)(?P<suffix>",?\s*)')


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-version", required=True, help="Candidate jax version, for example 0.10.2.")
    parser.add_argument(
        "--jaxlib-version",
        help="Candidate jaxlib version. Defaults to --jax-version because JAX/JAXLIB pins usually move together.",
    )
    parser.add_argument("--libtpu-version", help="Optional candidate libtpu version for TPU upgrade probes.")
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=DEFAULT_PROBE_ROOT,
        help="Temporary directory for the patched project copy. It is deleted and recreated.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run inside the probe copy. Prefix with '--', for example: -- make verify-critical-cpu.",
    )
    return parser.parse_args(argv)


def _copy_project(source: Path, destination: Path):
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "tmp"),
    )


def _replace_exact_pin(text: str, package_name: str, version: str) -> str:
    pattern = re.compile(PINNED_REQUIREMENT_RE.pattern.format(name=re.escape(package_name)))
    text, count = pattern.subn(rf"\g<prefix>{package_name}=={version}\g<suffix>", text)
    if count == 0:
        raise ValueError(f"Could not find exact pin for {package_name!r} in pyproject.toml.")
    return text


def patch_dependency_pins(pyproject_path: Path, pins: dict[str, str]):
    original = pyproject_path.read_text()
    patched = original
    for package_name, version in pins.items():
        patched = _replace_exact_pin(patched, package_name, version)
    pyproject_path.write_text(patched)
    tomllib.loads(patched)


def _format_pin_summary(pins: dict[str, str]) -> str:
    return ", ".join(f"{name}=={version}" for name, version in pins.items())


def _probe_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("VIRTUAL_ENV", "UV_PROJECT", "UV_WORKING_DIR", "UV_NO_PROJECT"):
        env.pop(name, None)
    env.setdefault("JAX_PLATFORMS", "cpu")
    return env


def _run(command: list[str], cwd: Path, env: dict[str, str] | None = None):
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=env)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("A probe command is required after '--'.")

    pins = {
        "jax": args.jax_version,
        "jaxlib": args.jaxlib_version or args.jax_version,
    }
    if args.libtpu_version is not None:
        pins["libtpu"] = args.libtpu_version

    probe_root = args.probe_root.resolve()
    env = _probe_environment()
    print(f"Creating dependency upgrade probe in {probe_root}", flush=True)
    print(f"Candidate pins: {_format_pin_summary(pins)}", flush=True)
    _copy_project(PROJECT_ROOT, probe_root)
    patch_dependency_pins(probe_root / "pyproject.toml", pins)
    _run(["uv", "lock"], cwd=probe_root, env=env)
    _run(command, cwd=probe_root, env=env)
    print("Dependency upgrade probe completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
