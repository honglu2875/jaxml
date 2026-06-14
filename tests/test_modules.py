#  Copyright 2024 Honglu Fan
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jaxml.config import ModelConfig
from jaxml.models.gemma3 import GemmaMLP, GemmaRMSNorm
from jaxml.models.gpt_neox import GPTNeoXMLP
from jaxml.models.llama import LlamaMLP
from jaxml.utils import torch_to_jax_states


@pytest.mark.parametrize("mlp_cls", [LlamaMLP, GPTNeoXMLP, GemmaMLP])
def test_mlp_rejects_hidden_size_mismatch(mlp_cls):
    config = ModelConfig(
        hidden_size=48,
        head_dim=8,
        num_heads=6,
        num_layers=1,
        intermediate_ratio=(2, 1),
        max_position_embeddings=16,
        vocab_size=128,
        attn_scale=8**-0.5,
    )
    mlp = mlp_cls(config=config, dtype=jnp.float32)
    x = jnp.ones((1, 2, 47), dtype=jnp.float32)

    with pytest.raises(ValueError, match="hidden dimension"):
        mlp.init(jax.random.PRNGKey(0), x)


def test_gemma_rms_norm_rejects_disabled_upcast():
    norm = GemmaRMSNorm(hidden_size=4, upcast=False)
    x = jnp.ones((1, 2, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="upcast=True"):
        norm.init(jax.random.PRNGKey(0), x)


@pytest.mark.parametrize("name", ["dense", "rms_norm", "layer_norm"])
def test_modules(jax_component_factory, torch_component_factory, name):
    jax_comp = jax_component_factory(name)
    torch_comp = torch_component_factory(name)

    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (4, 10, 48), dtype=jnp.float32)
        params = torch_to_jax_states(torch_comp, dtype=torch.float32)
        y = jax_comp.apply(params, x)
        with torch.no_grad():
            y2 = torch_comp(torch.tensor(np.array(x))).numpy()

        assert np.allclose(y, y2, atol=1e-5)


@pytest.mark.parametrize("dtype", ["bad", torch.int32])
def test_torch_to_jax_states_rejects_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="Unsupported dtype"):
        torch_to_jax_states({"weight": torch.ones(1)}, dtype=dtype)


def test_torch_to_jax_states_rejects_invalid_dtype_type():
    with pytest.raises(TypeError, match="Expected dtype"):
        torch_to_jax_states({"weight": torch.ones(1)}, dtype=np.float32)


def test_torch_to_jax_states_rejects_non_tensor_state_values():
    with pytest.raises(TypeError, match="State value for key 'weight'"):
        torch_to_jax_states({"weight": np.ones(1)}, dtype=torch.float32)


def test_torch_to_jax_states_normalizes_repeated_numeric_key_segments():
    params = torch_to_jax_states(
        {"encoder.0.block.1.weight": torch.ones((2, 3))},
        dtype=torch.float32,
    )

    assert params["params"]["encoder_0"]["block_1"]["kernel"].shape == (3, 2)


def test_torch_to_jax_states_normalizes_adjacent_numeric_key_segments():
    params = torch_to_jax_states(
        {"stack.0.1.weight": torch.ones((2, 3))},
        dtype=torch.float32,
    )

    assert params["params"]["stack_0_1"]["kernel"].shape == (3, 2)


def test_torch_to_jax_states_rejects_duplicate_normalized_destinations():
    state = {
        "stack.0.weight": torch.ones((2, 3)),
        "stack_0.weight": torch.ones((2, 3)),
    }

    with pytest.raises(ValueError, match="Multiple state keys map"):
        torch_to_jax_states(state, dtype=torch.float32)


def test_torch_to_jax_states_rejects_leaf_subtree_conflicts():
    state = {
        "stack.bias": torch.ones((2, 3)),
        "stack.bias.extra": torch.ones((2, 3)),
    }

    with pytest.raises(ValueError, match="conflicts with existing leaf"):
        torch_to_jax_states(state, dtype=torch.float32)
