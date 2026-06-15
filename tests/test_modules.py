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
from jaxml.nn.embedding import Embed
from jaxml.nn.linear import DenseGeneral
from jaxml.nn.norms import LayerNorm, RMSNorm
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


@pytest.mark.parametrize("norm_cls", [RMSNorm, LayerNorm])
def test_norms_reject_hidden_size_mismatch(norm_cls):
    norm = norm_cls(hidden_size=4)
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="hidden dimension mismatch"):
        norm.init(jax.random.PRNGKey(0), x)


@pytest.mark.parametrize("norm_cls", [RMSNorm, LayerNorm])
def test_norms_reject_scalar_inputs(norm_cls):
    norm = norm_cls(hidden_size=4)
    x = jnp.array(1.0, dtype=jnp.float32)

    with pytest.raises(ValueError, match="at least one dimension"):
        norm.init(jax.random.PRNGKey(0), x)


@pytest.mark.parametrize("norm_cls", [RMSNorm, LayerNorm, GemmaRMSNorm])
def test_norms_reject_non_floating_inputs(norm_cls):
    norm = norm_cls(hidden_size=4)
    x = jnp.ones((1, 2, 4), dtype=jnp.int32)

    with pytest.raises(TypeError, match="floating point"):
        norm.init(jax.random.PRNGKey(0), x)


@pytest.mark.parametrize("norm_cls", [RMSNorm, LayerNorm])
@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"hidden_size": True}, TypeError, "hidden_size must be an integer"),
        ({"hidden_size": np.bool_(True)}, TypeError, "hidden_size must be an integer"),
        ({"hidden_size": 1.5}, TypeError, "hidden_size must be an integer"),
        ({"hidden_size": 0}, ValueError, "hidden_size must be positive"),
        ({"eps": True}, TypeError, "eps must be a real number"),
        ({"eps": "1e-6"}, TypeError, "eps must be a real number"),
        ({"eps": float("nan")}, ValueError, "eps must be finite"),
        ({"eps": 0.0}, ValueError, "eps must be positive"),
        ({"upcast": 1}, TypeError, "upcast must be a boolean"),
    ],
)
def test_norms_reject_invalid_parameters(norm_cls, kwargs, exception, match):
    norm = norm_cls(**({"hidden_size": 4} | kwargs))
    x = jnp.ones((1, 2, 4), dtype=jnp.float32)

    with pytest.raises(exception, match=match):
        norm.init(jax.random.PRNGKey(0), x)


def test_layer_norm_rejects_non_boolean_use_bias():
    norm = LayerNorm(hidden_size=4, use_bias=1)
    x = jnp.ones((1, 2, 4), dtype=jnp.float32)

    with pytest.raises(TypeError, match="use_bias must be a boolean"):
        norm.init(jax.random.PRNGKey(0), x)


@pytest.mark.parametrize("norm_cls", [RMSNorm, LayerNorm])
def test_norms_accept_numpy_scalar_parameters(norm_cls):
    norm = norm_cls(hidden_size=np.int64(4), eps=np.float64(1e-6), upcast=np.bool_(True))
    x = jnp.ones((1, 2, 4), dtype=jnp.float32)

    params = norm.init(jax.random.PRNGKey(0), x)
    y = norm.apply(params, x)

    assert y.shape == x.shape


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"num_embeddings": True, "features": 4}, TypeError, "num_embeddings must be an integer"),
        ({"num_embeddings": np.bool_(True), "features": 4}, TypeError, "num_embeddings must be an integer"),
        ({"num_embeddings": 1.5, "features": 4}, TypeError, "num_embeddings must be an integer"),
        ({"num_embeddings": 0, "features": 4}, ValueError, "num_embeddings must be positive"),
        ({"num_embeddings": 8, "features": True}, TypeError, "features must be an integer"),
        ({"num_embeddings": 8, "features": np.bool_(True)}, TypeError, "features must be an integer"),
        ({"num_embeddings": 8, "features": 1.5}, TypeError, "features must be an integer"),
        ({"num_embeddings": 8, "features": 0}, ValueError, "features must be positive"),
    ],
)
def test_embed_rejects_invalid_shape_parameters(kwargs, exception, match):
    embed = Embed(**kwargs)
    x = jnp.array([0, 1], dtype=jnp.int32)

    with pytest.raises(exception, match=match):
        embed.init(jax.random.PRNGKey(0), x)


@pytest.mark.parametrize("one_hot", [False, True])
def test_embed_accepts_numpy_integer_shape_parameters(one_hot):
    embed = Embed(num_embeddings=np.int64(8), features=np.int64(4), one_hot=one_hot)
    x = jnp.array([0, 1], dtype=jnp.int32)

    params = embed.init(jax.random.PRNGKey(0), x)
    y = embed.apply(params, x)

    assert y.shape == (2, 4)


def test_embed_rejects_non_boolean_one_hot():
    embed = Embed(num_embeddings=8, features=4, one_hot=1)
    x = jnp.array([0, 1], dtype=jnp.int32)

    with pytest.raises(TypeError, match="one_hot must be a boolean"):
        embed.init(jax.random.PRNGKey(0), x)


def test_embed_accepts_numpy_boolean_one_hot():
    embed = Embed(num_embeddings=8, features=4, one_hot=np.bool_(True))
    x = jnp.array([0, 1], dtype=jnp.int32)

    params = embed.init(jax.random.PRNGKey(0), x)
    y = embed.apply(params, x)

    assert y.shape == (2, 4)


@pytest.mark.parametrize("one_hot", [False, True])
@pytest.mark.parametrize(
    "x,match",
    [
        (jnp.array([0, -1], dtype=jnp.int32), "non-negative"),
        (jnp.array([0, 8], dtype=jnp.int32), "less than num_embeddings"),
    ],
)
def test_embed_rejects_out_of_range_input_ids(one_hot, x, match):
    embed = Embed(num_embeddings=8, features=4, one_hot=one_hot)
    valid_x = jnp.array([0, 1], dtype=jnp.int32)
    params = embed.init(jax.random.PRNGKey(0), valid_x)

    with pytest.raises(ValueError, match=match):
        embed.apply(params, x)


@pytest.mark.parametrize("one_hot", [False, True])
def test_embed_accepts_traced_input_ids(one_hot):
    embed = Embed(num_embeddings=8, features=4, one_hot=one_hot)
    x = jnp.array([0, 1], dtype=jnp.int32)
    params = embed.init(jax.random.PRNGKey(0), x)

    @jax.jit
    def apply(inputs):
        return embed.apply(params, inputs)

    y = apply(x)

    assert y.shape == (2, 4)


@pytest.mark.parametrize(
    "axis,exception,match",
    [
        ((-1, -1), ValueError, "unique"),
        (3, ValueError, "out of bounds"),
        (1.5, TypeError, "integers"),
        (True, TypeError, "integers"),
        (np.bool_(True), TypeError, "integers"),
    ],
)
def test_dense_general_rejects_invalid_axes(axis, exception, match):
    dense = DenseGeneral(
        features=4,
        axis=axis,
        kernel_axes=("embed", "features"),
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    with pytest.raises(exception, match=match):
        dense.init(jax.random.PRNGKey(0), x)


def test_dense_general_accepts_numpy_integer_axis():
    dense = DenseGeneral(
        features=4,
        axis=np.int64(-1),
        kernel_axes=("embed", "features"),
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    params = dense.init(jax.random.PRNGKey(0), x)
    y = dense.apply(params, x)

    assert y.shape == (1, 2, 4)


def test_dense_general_rejects_non_floating_inputs():
    dense = DenseGeneral(
        features=4,
        axis=-1,
        kernel_axes=("embed", "features"),
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.int32)

    with pytest.raises(TypeError, match="floating point"):
        dense.init(jax.random.PRNGKey(0), x)


def test_dense_general_rejects_non_boolean_with_logical_partitioning():
    dense = DenseGeneral(
        features=4,
        axis=-1,
        kernel_axes=("embed", "features"),
        with_logical_partitioning=1,
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    with pytest.raises(TypeError, match="with_logical_partitioning must be a boolean"):
        dense.init(jax.random.PRNGKey(0), x)


def test_dense_general_rejects_missing_kernel_axes_with_logical_partitioning():
    dense = DenseGeneral(
        features=4,
        axis=-1,
        kernel_axes=(),
        with_logical_partitioning=True,
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Kernel axes must be specified"):
        dense.init(jax.random.PRNGKey(0), x)


def test_dense_general_rejects_kernel_axes_rank_mismatch():
    dense = DenseGeneral(
        features=4,
        axis=-1,
        kernel_axes=("embed",),
        with_logical_partitioning=True,
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="one axis name per kernel dimension"):
        dense.init(jax.random.PRNGKey(0), x)


def test_dense_general_rejects_string_kernel_axes():
    dense = DenseGeneral(
        features=4,
        axis=-1,
        kernel_axes="embed",
        with_logical_partitioning=True,
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    with pytest.raises(TypeError, match="kernel_axes must be a tuple"):
        dense.init(jax.random.PRNGKey(0), x)


def test_dense_general_rejects_non_string_kernel_axis_entries():
    dense = DenseGeneral(
        features=4,
        axis=-1,
        kernel_axes=("embed", 1),
        with_logical_partitioning=True,
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    with pytest.raises(TypeError, match="kernel_axes entries"):
        dense.init(jax.random.PRNGKey(0), x)


def test_dense_general_accepts_numpy_boolean_with_logical_partitioning():
    dense = DenseGeneral(
        features=4,
        axis=-1,
        with_logical_partitioning=np.bool_(False),
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    params = dense.init(jax.random.PRNGKey(0), x)
    y = dense.apply(params, x)

    assert y.shape == (1, 2, 4)


@pytest.mark.parametrize(
    "features,exception,match",
    [
        (True, TypeError, "features values must be integers"),
        (np.bool_(True), TypeError, "features values must be integers"),
        (1.5, TypeError, "features values must be integers"),
        (0, ValueError, "features values must be positive"),
        ((4, True), TypeError, "features values must be integers"),
        ((4, np.bool_(True)), TypeError, "features values must be integers"),
        ((4, 1.5), TypeError, "features values must be integers"),
        ((4, 0), ValueError, "features values must be positive"),
    ],
)
def test_dense_general_rejects_invalid_features(features, exception, match):
    dense = DenseGeneral(
        features=features,
        axis=-1,
        kernel_axes=("embed", "features"),
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    with pytest.raises(exception, match=match):
        dense.init(jax.random.PRNGKey(0), x)


def test_dense_general_accepts_numpy_integer_features():
    dense = DenseGeneral(
        features=(np.int64(4), np.int64(2)),
        axis=-1,
        kernel_axes=("embed", "features_a", "features_b"),
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    params = dense.init(jax.random.PRNGKey(0), x)
    y = dense.apply(params, x)

    assert y.shape == (1, 2, 4, 2)


def test_dense_general_rejects_non_boolean_use_bias():
    dense = DenseGeneral(
        features=4,
        axis=-1,
        kernel_axes=("embed", "features"),
        use_bias=1,
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    with pytest.raises(TypeError, match="use_bias must be a boolean"):
        dense.init(jax.random.PRNGKey(0), x)


def test_dense_general_accepts_numpy_boolean_use_bias():
    dense = DenseGeneral(
        features=4,
        axis=-1,
        kernel_axes=("embed", "features"),
        use_bias=np.bool_(True),
    )
    x = jnp.ones((1, 2, 3), dtype=jnp.float32)

    params = dense.init(jax.random.PRNGKey(0), x)
    y = dense.apply(params, x)

    assert y.shape == (1, 2, 4)


def test_layer_norm_without_bias_matches_zero_bias_torch_layer_norm():
    jax_norm = LayerNorm(hidden_size=4, use_bias=False, dtype=jnp.float32)
    torch_norm = torch.nn.LayerNorm(4, eps=1e-6, elementwise_affine=True)
    torch_norm.bias.data.zero_()
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    params = {"params": {"weight": np.array(torch_norm.weight.detach().numpy(), dtype=np.float32)}}

    y = jax_norm.apply(params, x)
    with torch.no_grad():
        y2 = torch_norm(torch.tensor(np.array(x))).numpy()

    assert np.allclose(y, y2, atol=1e-6)


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


@pytest.mark.parametrize("head_dim", [True, np.bool_(True), 1.5])
def test_torch_to_jax_states_rejects_non_integer_head_dim(head_dim):
    with pytest.raises(TypeError, match="head_dim must be an integer"):
        torch_to_jax_states({"weight": torch.ones(1)}, dtype=torch.float32, head_dim=head_dim)


@pytest.mark.parametrize("head_dim", [0, -1])
def test_torch_to_jax_states_rejects_non_positive_head_dim(head_dim):
    with pytest.raises(ValueError, match="head_dim must be positive"):
        torch_to_jax_states({"weight": torch.ones(1)}, dtype=torch.float32, head_dim=head_dim)


def test_torch_to_jax_states_accepts_numpy_integer_head_dim():
    params = torch_to_jax_states(
        {"q_proj.weight": torch.ones((4, 8))},
        dtype=torch.float32,
        head_dim=np.int64(4),
    )

    assert params["params"]["q_proj"]["kernel"].shape == (8, 1, 4)


@pytest.mark.parametrize(
    "state,match",
    [
        ({"q_proj.weight": torch.ones((5, 8))}, "Q/K/V projection output dimension"),
        ({"k_proj.bias": torch.ones(5)}, "Q/K/V projection bias length"),
        ({"qkv_proj.weight": torch.ones((10, 8))}, "Fused QKV projection output dimension"),
        ({"qkv_proj.bias": torch.ones(10)}, "Fused QKV projection bias length"),
    ],
)
def test_torch_to_jax_states_rejects_projection_shapes_incompatible_with_head_dim(state, match):
    with pytest.raises(ValueError, match=match):
        torch_to_jax_states(state, dtype=torch.float32, head_dim=4)


def test_torch_to_jax_states_rejects_non_tensor_state_values():
    with pytest.raises(TypeError, match="State value for key 'weight'"):
        torch_to_jax_states({"weight": np.ones(1)}, dtype=torch.float32)


def test_torch_to_jax_states_detaches_values_requiring_grad():
    params = torch_to_jax_states(
        {"bias": torch.tensor([1.0, 2.0], requires_grad=True)},
        dtype=torch.float32,
    )

    assert np.array_equal(params["params"]["bias"], np.array([1.0, 2.0], dtype=np.float32))


def test_torch_to_jax_states_converts_bfloat16_source_values():
    params = torch_to_jax_states(
        {"bias": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)},
        dtype="bfloat16",
    )

    assert params["params"]["bias"].dtype == jnp.bfloat16
    assert np.array_equal(params["params"]["bias"], np.array([1.0, 2.0], dtype=jnp.bfloat16))


def test_torch_to_jax_states_rejects_non_string_state_keys():
    with pytest.raises(TypeError, match="State key must be a string"):
        torch_to_jax_states({1: torch.ones(1)}, dtype=torch.float32)


@pytest.mark.parametrize("key", ["", ".weight", "stack..weight", "stack.weight."])
def test_torch_to_jax_states_rejects_empty_state_key_segments(key):
    with pytest.raises(ValueError, match="State key"):
        torch_to_jax_states({key: torch.ones(1)}, dtype=torch.float32)


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
