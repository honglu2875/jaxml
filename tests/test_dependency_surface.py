import importlib.metadata as metadata
import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPECIFIER_OPERATORS = ("==", ">=", "<=", "~=", "!=", ">", "<")
_SUPPORTED_CI_PYTHON_VERSIONS = ("3.11", "3.12")
_CPU_CADENCE_MARKERS = ("critical", "milestone")

pytestmark = pytest.mark.critical


def _project_config():
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def _lock_config():
    return tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())


def _makefile_targets():
    targets = {}
    for line in (PROJECT_ROOT / "Makefile").read_text().splitlines():
        if not line or line.startswith(("\t", ".", "#")) or ":" not in line:
            continue
        target, dependencies = line.split(":", maxsplit=1)
        targets[target] = tuple(dependencies.split())
    return targets


def _makefile_recipes():
    recipes = {}
    current_target = None
    for line in (PROJECT_ROOT / "Makefile").read_text().splitlines():
        if line and not line.startswith(("\t", ".", "#")) and ":" in line:
            current_target = line.split(":", maxsplit=1)[0]
            recipes[current_target] = []
            continue
        if current_target is not None and line.startswith("\t"):
            recipes[current_target].append(line.strip())
    return recipes


def _workflow_config():
    return (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()


def _workflow_python_versions(job_name: str) -> tuple[str, ...]:
    workflow = _workflow_config()
    job_match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:|\Z)", workflow, re.MULTILINE | re.DOTALL
    )
    if job_match is None:
        raise AssertionError(f"CI workflow does not define job {job_name!r}.")
    matrix_match = re.search(r"python-version:\n(?P<versions>(?:\s+- \"[^\"]+\"\n)+)", job_match.group("body"))
    if matrix_match is None:
        raise AssertionError(f"CI job {job_name!r} does not define a python-version matrix.")
    return tuple(re.findall(r'- "([^"]+)"', matrix_match.group("versions")))


def _dependency_drift_module():
    path = PROJECT_ROOT / "scripts" / "check_dependency_drift.py"
    spec = importlib.util.spec_from_file_location("check_dependency_drift", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_modules():
    return sorted((PROJECT_ROOT / "tests").glob("test_*.py"))


def _test_module_cadence_markers(path: Path):
    text = path.read_text()
    return tuple(marker for marker in _CPU_CADENCE_MARKERS if f"pytest.mark.{marker}" in text)


def _exact_direct_pins():
    pyproject = _project_config()
    dependencies = list(pyproject["project"]["dependencies"])
    dependencies.extend(pyproject["project"]["optional-dependencies"]["dev"])
    for requirement in dependencies:
        if "==" not in requirement:
            continue
        name, version = requirement.split("==", maxsplit=1)
        yield pytest.param(name, version, id=name)


def _project_runtime_requirements():
    pyproject = _project_config()
    yield from pyproject["project"]["dependencies"]
    for requirements in pyproject["project"]["optional-dependencies"].values():
        yield from requirements


def _requirement_entry(requirement: str, marker: str | None = None):
    for operator in _SPECIFIER_OPERATORS:
        if operator not in requirement:
            continue
        name, version = requirement.split(operator, maxsplit=1)
        return name, operator + version, marker
    return requirement, None, marker


def _exact_pin_map(requirements):
    pins = {}
    for requirement in requirements:
        if "==" not in requirement:
            continue
        name, version = requirement.split("==", maxsplit=1)
        pins[name] = version
    return pins


def _jaxml_lock_package():
    lock = _lock_config()
    for package in lock["package"]:
        if package["name"] == "jaxml":
            return package
    raise AssertionError("uv.lock does not contain the editable jaxml package.")


@pytest.mark.parametrize(("package_name", "expected_version"), list(_exact_direct_pins()))
def test_installed_direct_dependency_matches_project_pin(package_name, expected_version):
    assert metadata.version(package_name) == expected_version


def test_lock_metadata_matches_project_dependencies():
    pyproject = _project_config()
    expected = {_requirement_entry(requirement) for requirement in pyproject["project"]["dependencies"]}
    for extra_name, requirements in pyproject["project"]["optional-dependencies"].items():
        marker = f"extra == '{extra_name}'"
        expected.update(_requirement_entry(requirement, marker) for requirement in requirements)

    lock_requires_dist = _jaxml_lock_package()["metadata"]["requires-dist"]
    actual = {
        (
            requirement["name"],
            requirement.get("specifier"),
            requirement.get("marker"),
        )
        for requirement in lock_requires_dist
    }

    assert actual == expected


def test_static_project_version_avoids_scm_build_versioning_dependency():
    pyproject = _project_config()

    assert "version" in pyproject["project"]
    assert all("setuptools_scm" not in requirement for requirement in pyproject["build-system"]["requires"])


def test_project_runtime_requirements_are_exactly_pinned():
    unpinned = [requirement for requirement in _project_runtime_requirements() if "==" not in requirement]

    assert unpinned == []


def test_tpu_extra_keeps_jax_runtime_pins_aligned_with_base_dependencies():
    pyproject = _project_config()
    base_pins = _exact_pin_map(pyproject["project"]["dependencies"])
    tpu_pins = _exact_pin_map(pyproject["project"]["optional-dependencies"]["tpu"])

    assert tpu_pins["jax"] == base_pins["jax"]
    assert tpu_pins["jaxlib"] == base_pins["jaxlib"]


def test_tpu_extra_keeps_libtpu_explicitly_pinned():
    pyproject = _project_config()
    tpu_pins = _exact_pin_map(pyproject["project"]["optional-dependencies"]["tpu"])

    assert "libtpu" in tpu_pins


def test_tpu_verification_checks_lockfile_and_installed_dependencies_before_tests():
    targets = _makefile_targets()

    assert targets["verify-tpu"] == ("lock-check", "dependency-check", "pytest-tpu")


def test_critical_cpu_verification_runs_push_cadence_checks_before_tests():
    targets = _makefile_targets()

    assert targets["verify-critical-cpu"] == (
        "lock-check",
        "dependency-check",
        "lint",
        "format-check",
        "build-check",
        "pytest-critical-cpu",
    )


def test_build_check_builds_source_and_wheel_artifacts_in_throwaway_path():
    recipes = _makefile_recipes()

    assert recipes["build-check"] == ["uv build --clear --out-dir tmp/build-check"]


def test_milestone_cpu_verification_keeps_full_cpu_suite_available():
    targets = _makefile_targets()

    assert targets["verify-cpu"] == ("verify-milestone-cpu",)
    assert targets["verify-milestone-cpu"] == ("lock-check", "dependency-check", "lint", "format-check", "pytest-cpu")


def test_cpu_test_targets_use_expected_pytest_markers():
    recipes = _makefile_recipes()

    assert recipes["pytest-critical-cpu"] == ["${CRITICAL_CPU_TESTS}"]
    assert recipes["pytest-cpu"] == ["${CPU_TESTS}"]


def test_cpu_test_modules_are_assigned_to_exactly_one_cadence_marker():
    incorrectly_marked = {}
    for path in _test_modules():
        if path.name == "test_tpu.py":
            continue
        markers = _test_module_cadence_markers(path)
        if len(markers) != 1:
            incorrectly_marked[path.name] = markers

    assert incorrectly_marked == {}


def test_ci_runs_critical_cpu_suite_on_push_and_full_suite_on_milestone_events():
    workflow = _workflow_config()

    assert "uv run --frozen --extra dev make verify-critical-cpu" in workflow
    assert "uv run --frozen --extra dev make verify-milestone-cpu" in workflow
    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow


@pytest.mark.parametrize("job_name", ["cpu", "milestone-cpu"])
def test_ci_cpu_jobs_cover_supported_python_versions(job_name):
    assert _workflow_python_versions(job_name) == _SUPPORTED_CI_PYTHON_VERSIONS


def test_jax_runtime_surface_executes_jitted_cpu_work():
    import jax
    import jax.numpy as jnp

    @jax.jit
    def add_one(x):
        return x + 1

    assert jax.default_backend() == "cpu"
    result = add_one(jnp.arange(4, dtype=jnp.float32))
    assert result.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_flax_module_surface_initializes_and_applies():
    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    class TinyModule(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(features=2, use_bias=False)(x)

    module = TinyModule()
    variables = module.init(jax.random.PRNGKey(0), jnp.ones((1, 3), dtype=jnp.float32))
    output = module.apply(variables, jnp.ones((1, 3), dtype=jnp.float32))

    assert output.shape == (1, 2)


def test_hf_config_surface_exposes_supported_model_configs():
    from transformers import Gemma3TextConfig, GPTNeoXConfig, LlamaConfig

    assert LlamaConfig(model_type="llama").model_type == "llama"
    assert GPTNeoXConfig(model_type="gpt_neox").model_type == "gpt_neox"
    assert Gemma3TextConfig(model_type="gemma3").model_type == "gemma3"


def test_torch_tensor_surface_interops_with_numpy():
    import numpy as np
    import torch

    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    array = tensor.detach().numpy()

    assert np.asarray(array).shape == (2, 2)
    assert array.tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_dependency_drift_target_audits_direct_project_pins():
    recipes = _makefile_recipes()

    assert recipes["dependency-drift"] == ["uv run --frozen --extra dev python scripts/check_dependency_drift.py"]


def test_dependency_drift_helper_filters_to_direct_project_pins():
    drift = _dependency_drift_module()
    direct_names = {"jax", "jaxlib", "flax", "transformers", "torch"}
    outdated_packages = [
        {"name": "jax", "version": "0.1.0", "latest_version": "0.2.0"},
        {"name": "tokenizers", "version": "0.1.0", "latest_version": "0.2.0"},
        {"name": "torch", "version": "1.0.0", "latest_version": "2.0.0"},
    ]

    direct_outdated = drift.direct_outdated_packages(outdated_packages, direct_names)

    assert [package["name"] for package in direct_outdated] == ["jax", "torch"]


def test_dependency_drift_helper_reports_current_direct_pins_cleanly():
    drift = _dependency_drift_module()

    assert drift.format_report([]) == "All direct pinned dependencies are current."
