import importlib.metadata as metadata
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _exact_direct_pins():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    dependencies = list(pyproject["project"]["dependencies"])
    dependencies.extend(pyproject["project"]["optional-dependencies"]["dev"])
    for requirement in dependencies:
        if "==" not in requirement:
            continue
        name, version = requirement.split("==", maxsplit=1)
        yield pytest.param(name, version, id=name)


def _exact_pin_map(requirements):
    pins = {}
    for requirement in requirements:
        if "==" not in requirement:
            continue
        name, version = requirement.split("==", maxsplit=1)
        pins[name] = version
    return pins


@pytest.mark.parametrize(("package_name", "expected_version"), list(_exact_direct_pins()))
def test_installed_direct_dependency_matches_project_pin(package_name, expected_version):
    assert metadata.version(package_name) == expected_version


def test_tpu_extra_keeps_jax_runtime_pins_aligned_with_base_dependencies():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    base_pins = _exact_pin_map(pyproject["project"]["dependencies"])
    tpu_pins = _exact_pin_map(pyproject["project"]["optional-dependencies"]["tpu"])

    assert tpu_pins["jax"] == base_pins["jax"]
    assert tpu_pins["jaxlib"] == base_pins["jaxlib"]


def test_tpu_extra_keeps_libtpu_explicitly_pinned():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    tpu_pins = _exact_pin_map(pyproject["project"]["optional-dependencies"]["tpu"])

    assert "libtpu" in tpu_pins


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
