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
from jaxml.hf_utils import to_llama_jax_params, to_neox_jax_params
from jaxml.nn.attention import Attention, AttentionWithRoPE
from jaxml.test_utils.torch_utils import DummyPosEmb

pytestmark = pytest.mark.milestone


def _identity_cos_sin(config: ModelConfig, seq_len: int, dtype=jnp.float32):
    rotary_dim = int(config.head_dim * config.rotary_pct)
    return (
        jnp.ones((seq_len, rotary_dim), dtype=dtype),
        jnp.zeros((seq_len, rotary_dim), dtype=dtype),
    )


def _param_value(value):
    return getattr(value, "value", value)


def test_fused_qkv_rejects_mismatched_kv_heads():
    config = ModelConfig(
        head_dim=4,
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=4**-0.5,
        use_rope=False,
    )
    attn = Attention(config, fused_qkv=True)
    hidden_states = jnp.zeros((1, 2, config.hidden_size), dtype=jnp.float32)

    with pytest.raises(ValueError, match="fused_qkv"):
        attn.init(jax.random.PRNGKey(0), hidden_states)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"dtype": None}, "dtype must be a valid JAX dtype"),
        ({"dtype": "not-a-dtype"}, "dtype must be a valid JAX dtype"),
        ({"weight_dtype": None}, "weight_dtype must be a valid JAX dtype"),
        ({"weight_dtype": "not-a-dtype"}, "weight_dtype must be a valid JAX dtype"),
        ({"with_logical_partitioning": 1}, "with_logical_partitioning must be a boolean"),
    ],
)
def test_attention_rejects_invalid_projection_arguments(kwargs, match):
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config, **kwargs)
    hidden_states = jnp.zeros((1, 2, 1), dtype=jnp.float32)

    with pytest.raises(TypeError, match=match):
        attn.init(jax.random.PRNGKey(0), hidden_states)


def test_attention_applies_weight_dtype_to_all_projections():
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config, dtype=np.float32, weight_dtype=jnp.bfloat16, with_logical_partitioning=np.bool_(False))
    hidden_states = jnp.zeros((1, 2, 1), dtype=jnp.float32)

    params = attn.init(jax.random.PRNGKey(0), hidden_states)

    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert _param_value(params["params"][name]["kernel"]).dtype == jnp.bfloat16


def test_sliding_window_decode_masks_old_keys():
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    query_states = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    key_states = jnp.zeros((1, 5, 1, 1), dtype=jnp.float32)
    value_states = jnp.arange(5, dtype=jnp.float32).reshape(1, 5, 1, 1) * 10

    output, weights = attn.apply(
        {},
        query_states,
        key_states,
        value_states,
        sliding_window=2,
        output_attentions=True,
        method=Attention.mha,
    )

    assert np.allclose(output, np.array([[[[35.0]]]], dtype=np.float32))
    assert np.allclose(weights, np.array([[[[0.0, 0.0, 0.0, 0.5, 0.5]]]], dtype=np.float32))


def test_sliding_window_decode_respects_padded_cache_mask():
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    query_states = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    key_states = jnp.zeros((1, 8, 1, 1), dtype=jnp.float32)
    value_states = jnp.arange(8, dtype=jnp.float32).reshape(1, 8, 1, 1) * 10
    attention_mask = jnp.array([[True, True, True, True, False, False, False, False]])

    output, weights = attn.apply(
        {},
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        sliding_window=2,
        output_attentions=True,
        method=Attention.mha,
    )

    assert np.all(np.isfinite(output))
    assert np.allclose(output, np.array([[[[25.0]]]], dtype=np.float32))
    assert np.allclose(weights, np.array([[[[0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32))


def test_attention_all_masked_rows_remain_finite():
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    query_states = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    key_states = jnp.zeros((1, 2, 1, 1), dtype=jnp.float32)
    value_states = jnp.array([[[[2.0]], [[4.0]]]], dtype=jnp.float32)
    attention_mask = jnp.array([[False, False]])

    output, weights = attn.apply(
        {},
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        output_attentions=True,
        method=Attention.mha,
    )

    assert np.all(np.isfinite(output))
    assert np.all(np.isfinite(weights))
    assert np.allclose(output, np.array([[[[3.0]]]], dtype=np.float32))
    assert np.allclose(weights, np.array([[[[0.5, 0.5]]]], dtype=np.float32))


def test_attention_mha_canonicalizes_integer_attention_mask():
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    query_states = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    key_states = jnp.zeros((1, 3, 1, 1), dtype=jnp.float32)
    value_states = jnp.arange(3, dtype=jnp.float32).reshape(1, 3, 1, 1)
    integer_mask = jnp.array([[1, 1, 0]], dtype=jnp.int32)
    bool_mask = integer_mask.astype(bool)

    output, weights = attn.apply(
        {},
        query_states,
        key_states,
        value_states,
        attention_mask=integer_mask,
        output_attentions=True,
        method=Attention.mha,
    )
    expected, expected_weights = attn.apply(
        {},
        query_states,
        key_states,
        value_states,
        attention_mask=bool_mask,
        output_attentions=True,
        method=Attention.mha,
    )

    assert np.allclose(output, expected)
    assert np.allclose(weights, expected_weights)


def test_attention_repeat_kv_repeats_grouped_key_values():
    config = ModelConfig(
        head_dim=2,
        hidden_size=4,
        num_heads=2,
        num_kv_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=2**-0.5,
        use_rope=False,
    )
    attn = Attention(config)
    key_states = jnp.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=jnp.float32)
    value_states = key_states + 10

    keys, values = attn.apply({}, key_states, value_states, method=Attention.repeat_kv)

    assert keys.shape == (1, 2, 2, 2)
    assert values.shape == (1, 2, 2, 2)
    assert np.allclose(keys[:, :, 0], key_states[:, :, 0])
    assert np.allclose(keys[:, :, 1], key_states[:, :, 0])
    assert np.allclose(values[:, :, 0], value_states[:, :, 0])
    assert np.allclose(values[:, :, 1], value_states[:, :, 0])


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"key_states": jnp.zeros((1, 2, 1), dtype=jnp.float32)}, ValueError, "key_states must be a 4D array"),
        ({"key_states": jnp.zeros((0, 2, 1, 2), dtype=jnp.float32)}, ValueError, "batch axis must be non-empty"),
        ({"key_states": jnp.zeros((1, 0, 1, 2), dtype=jnp.float32)}, ValueError, "sequence axis must be non-empty"),
        ({"key_states": jnp.zeros((1, 2, 0, 2), dtype=jnp.float32)}, ValueError, "head axis must be non-empty"),
        ({"key_states": jnp.zeros((1, 2, 1, 0), dtype=jnp.float32)}, ValueError, "head_dim axis must be non-empty"),
        ({"value_states": jnp.zeros((1, 2, 1, 2), dtype=jnp.int32)}, TypeError, "value_states must contain floating"),
        ({"value_states": jnp.zeros((1, 3, 1, 2), dtype=jnp.float32)}, ValueError, "key_states and value_states"),
        (
            {
                "key_states": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
                "value_states": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
            },
            ValueError,
            "num_key_value_heads",
        ),
        (
            {
                "key_states": jnp.zeros((1, 2, 1, 3), dtype=jnp.float32),
                "value_states": jnp.zeros((1, 2, 1, 3), dtype=jnp.float32),
            },
            ValueError,
            "config.head_dim",
        ),
    ],
)
def test_attention_repeat_kv_rejects_invalid_states(kwargs, exception, match):
    config = ModelConfig(
        head_dim=2,
        hidden_size=4,
        num_heads=2,
        num_kv_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=2**-0.5,
        use_rope=False,
    )
    attn = Attention(config)
    defaults = {
        "key_states": jnp.zeros((1, 2, 1, 2), dtype=jnp.float32),
        "value_states": jnp.zeros((1, 2, 1, 2), dtype=jnp.float32),
    }

    with pytest.raises(exception, match=match):
        attn.apply({}, **(defaults | kwargs), method=Attention.repeat_kv)


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, -jnp.inf])
@pytest.mark.parametrize("field_name", ["key_states", "value_states"])
def test_attention_repeat_kv_rejects_non_finite_states(field_name, value):
    config = ModelConfig(
        head_dim=2,
        hidden_size=4,
        num_heads=2,
        num_kv_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=2**-0.5,
        use_rope=False,
    )
    attn = Attention(config)
    kwargs = {
        "key_states": jnp.zeros((1, 2, 1, 2), dtype=jnp.float32),
        "value_states": jnp.zeros((1, 2, 1, 2), dtype=jnp.float32),
    }
    kwargs[field_name] = kwargs[field_name].at[0, 0, 0, 0].set(value)

    with pytest.raises(ValueError, match=f"{field_name} must contain only finite values"):
        attn.apply({}, **kwargs, method=Attention.repeat_kv)


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"query_states": jnp.zeros((1, 1, 1), dtype=jnp.float32)}, ValueError, "query_states must be a 4D array"),
        ({"key_states": jnp.zeros((1, 3, 1), dtype=jnp.float32)}, ValueError, "key_states must be a 4D array"),
        ({"value_states": jnp.zeros((1, 3, 1), dtype=jnp.float32)}, ValueError, "value_states must be a 4D array"),
        ({"query_states": jnp.zeros((0, 1, 1, 1), dtype=jnp.float32)}, ValueError, "batch axis must be non-empty"),
        ({"query_states": jnp.zeros((1, 0, 1, 1), dtype=jnp.float32)}, ValueError, "sequence axis must be non-empty"),
        ({"query_states": jnp.zeros((1, 1, 0, 1), dtype=jnp.float32)}, ValueError, "head axis must be non-empty"),
        ({"query_states": jnp.zeros((1, 1, 1, 0), dtype=jnp.float32)}, ValueError, "head_dim axis must be non-empty"),
        ({"key_states": jnp.zeros((1, 0, 1, 1), dtype=jnp.float32)}, ValueError, "sequence axis must be non-empty"),
        ({"value_states": jnp.zeros((1, 0, 1, 1), dtype=jnp.float32)}, ValueError, "sequence axis must be non-empty"),
        ({"query_states": jnp.zeros((1, 1, 1, 1), dtype=jnp.int32)}, TypeError, "query_states must contain floating"),
        ({"key_states": jnp.zeros((1, 3, 1, 1), dtype=jnp.int32)}, TypeError, "key_states must contain floating"),
        ({"value_states": jnp.zeros((1, 3, 1, 1), dtype=jnp.int32)}, TypeError, "value_states must contain floating"),
        ({"value_states": jnp.zeros((1, 2, 1, 1), dtype=jnp.float32)}, ValueError, "key_states and value_states"),
        ({"query_states": jnp.zeros((2, 1, 1, 1), dtype=jnp.float32)}, ValueError, "matching batch"),
        ({"query_states": jnp.zeros((1, 1, 2, 1), dtype=jnp.float32)}, ValueError, "matching batch"),
        ({"query_states": jnp.zeros((1, 1, 1, 2), dtype=jnp.float32)}, ValueError, "matching batch"),
    ],
)
def test_attention_mha_rejects_invalid_attention_states(kwargs, exception, match):
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    defaults = {
        "query_states": jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
        "key_states": jnp.zeros((1, 3, 1, 1), dtype=jnp.float32),
        "value_states": jnp.zeros((1, 3, 1, 1), dtype=jnp.float32),
    }

    with pytest.raises(exception, match=match):
        attn.apply(
            {},
            **(defaults | kwargs),
            method=Attention.mha,
        )


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, -jnp.inf])
@pytest.mark.parametrize("field_name", ["query_states", "key_states", "value_states"])
def test_attention_mha_rejects_non_finite_attention_states(field_name, value):
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    kwargs = {
        "query_states": jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
        "key_states": jnp.zeros((1, 3, 1, 1), dtype=jnp.float32),
        "value_states": jnp.zeros((1, 3, 1, 1), dtype=jnp.float32),
    }
    kwargs[field_name] = kwargs[field_name].at[0, 0, 0, 0].set(value)

    with pytest.raises(ValueError, match=f"{field_name} must contain only finite values"):
        attn.apply(
            {},
            **kwargs,
            method=Attention.mha,
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"causal": 1}, "causal must be a boolean"),
        ({"softmax_fp32": 1}, "softmax_fp32 must be a boolean"),
        ({"output_attentions": 1}, "output_attentions must be a boolean"),
    ],
)
def test_attention_mha_rejects_non_boolean_flags(kwargs, match):
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    query_states = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    key_states = jnp.zeros((1, 3, 1, 1), dtype=jnp.float32)
    value_states = jnp.zeros((1, 3, 1, 1), dtype=jnp.float32)

    with pytest.raises(TypeError, match=match):
        attn.apply(
            {},
            query_states,
            key_states,
            value_states,
            **kwargs,
            method=Attention.mha,
        )


@pytest.mark.parametrize(
    "attn_cls,use_rope",
    [
        (Attention, False),
        (AttentionWithRoPE, True),
    ],
)
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"output_attentions": 1}, "output_attentions must be a boolean"),
        ({"use_flash": 1}, "use_flash must be a boolean"),
    ],
)
def test_attention_call_rejects_non_boolean_flags(attn_cls, use_rope, kwargs, match):
    config = ModelConfig(
        head_dim=2 if use_rope else 1,
        hidden_size=2 if use_rope else 1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=use_rope,
    )
    attn = attn_cls(config)
    hidden_states = jnp.zeros((1, 2, config.hidden_size), dtype=jnp.float32)
    call_kwargs = {"cos_sin": _identity_cos_sin(config, hidden_states.shape[1])} if use_rope else {}
    params = attn.init(jax.random.PRNGKey(0), hidden_states, **call_kwargs)

    with pytest.raises(TypeError, match=match):
        attn.apply(params, hidden_states, **call_kwargs, **kwargs)


@pytest.mark.parametrize(
    "attn_cls,use_rope,apply_kwargs",
    [
        (Attention, False, {"kv_cache": object()}),
        (AttentionWithRoPE, True, {}),
    ],
)
def test_attention_call_rejects_flash_before_deeper_validation(attn_cls, use_rope, apply_kwargs):
    config = ModelConfig(
        head_dim=2 if use_rope else 1,
        hidden_size=2 if use_rope else 1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=use_rope,
    )
    attn = attn_cls(config)
    hidden_states = jnp.zeros((1, 2, config.hidden_size), dtype=jnp.float32)
    init_kwargs = {"cos_sin": _identity_cos_sin(config, hidden_states.shape[1])} if use_rope else {}
    params = attn.init(jax.random.PRNGKey(0), hidden_states, **init_kwargs)

    with pytest.raises(NotImplementedError, match="flash attention is not enabled"):
        attn.apply(params, hidden_states, use_flash=True, **apply_kwargs)


@pytest.mark.parametrize(
    "attn_cls,use_rope",
    [
        (Attention, False),
        (AttentionWithRoPE, True),
    ],
)
@pytest.mark.parametrize(
    "hidden_states,exception,match",
    [
        (jnp.zeros((1, 2), dtype=jnp.float32), ValueError, "hidden_states must be a 3D array"),
        (jnp.zeros((1, 2, 1, 1), dtype=jnp.float32), ValueError, "hidden_states must be a 3D array"),
        (jnp.zeros((0, 2, 1), dtype=jnp.float32), ValueError, "empty axes"),
        (jnp.zeros((1, 0, 1), dtype=jnp.float32), ValueError, "empty axes"),
        (jnp.zeros((1, 2, 0), dtype=jnp.float32), ValueError, "empty axes"),
        (jnp.zeros((1, 2, 1), dtype=jnp.int32), TypeError, "hidden_states must contain floating"),
    ],
)
def test_attention_call_rejects_invalid_hidden_states(attn_cls, use_rope, hidden_states, exception, match):
    config = ModelConfig(
        head_dim=2 if use_rope else 1,
        hidden_size=2 if use_rope else 1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=use_rope,
    )
    attn = attn_cls(config)
    valid_hidden_states = jnp.zeros((1, 2, config.hidden_size), dtype=jnp.float32)
    call_kwargs = {"cos_sin": _identity_cos_sin(config, valid_hidden_states.shape[1])} if use_rope else {}
    params = attn.init(jax.random.PRNGKey(0), valid_hidden_states, **call_kwargs)

    with pytest.raises(exception, match=match):
        attn.apply(params, hidden_states, **call_kwargs)


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, -jnp.inf])
@pytest.mark.parametrize(
    "attn_cls,use_rope",
    [
        (Attention, False),
        (AttentionWithRoPE, True),
    ],
)
def test_attention_call_rejects_non_finite_hidden_states(attn_cls, use_rope, value):
    config = ModelConfig(
        head_dim=2 if use_rope else 1,
        hidden_size=2 if use_rope else 1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=use_rope,
    )
    attn = attn_cls(config)
    valid_hidden_states = jnp.zeros((1, 2, config.hidden_size), dtype=jnp.float32)
    hidden_states = valid_hidden_states.at[0, 0, 0].set(value)
    call_kwargs = {"cos_sin": _identity_cos_sin(config, valid_hidden_states.shape[1])} if use_rope else {}
    params = attn.init(jax.random.PRNGKey(0), valid_hidden_states, **call_kwargs)

    with pytest.raises(ValueError, match="hidden_states must contain only finite values"):
        attn.apply(params, hidden_states, **call_kwargs)


@pytest.mark.parametrize(
    "attn_cls,use_rope",
    [
        (Attention, False),
        (AttentionWithRoPE, True),
    ],
)
def test_attention_call_rejects_non_kv_cache(attn_cls, use_rope):
    config = ModelConfig(
        head_dim=2 if use_rope else 1,
        hidden_size=2 if use_rope else 1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=use_rope,
    )
    attn = attn_cls(config)
    hidden_states = jnp.zeros((1, 2, config.hidden_size), dtype=jnp.float32)
    call_kwargs = {"cos_sin": _identity_cos_sin(config, hidden_states.shape[1])} if use_rope else {}
    params = attn.init(jax.random.PRNGKey(0), hidden_states, **call_kwargs)

    with pytest.raises(TypeError, match="kv_cache must be a KVCache"):
        attn.apply(params, hidden_states, **call_kwargs, kv_cache=object())


def test_attention_with_rope_requires_cos_sin():
    config = ModelConfig(
        head_dim=2,
        hidden_size=2,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=2**-0.5,
        use_rope=True,
    )
    attn = AttentionWithRoPE(config)
    hidden_states = jnp.zeros((1, 2, config.hidden_size), dtype=jnp.float32)
    cos_sin = _identity_cos_sin(config, hidden_states.shape[1])
    params = attn.init(jax.random.PRNGKey(0), hidden_states, cos_sin=cos_sin)

    with pytest.raises(ValueError, match="requires cos_sin"):
        attn.apply(params, hidden_states)


@pytest.mark.parametrize(
    "cos_sin,exception,match",
    [
        ((), ValueError, "exactly two arrays"),
        ((jnp.ones((2, 2), dtype=jnp.float32),), ValueError, "exactly two arrays"),
        (
            (
                jnp.ones((2, 2), dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.float32),
                jnp.ones((2, 2), dtype=jnp.float32),
            ),
            ValueError,
            "exactly two arrays",
        ),
        (jnp.ones((2, 2), dtype=jnp.float32), TypeError, "not a single array"),
        (np.ones((2, 2), dtype=np.float32), TypeError, "not a single array"),
        (object(), TypeError, "cos_sin must be a pair"),
    ],
)
def test_attention_with_rope_rejects_invalid_cos_sin_pair(cos_sin, exception, match):
    config = ModelConfig(
        head_dim=2,
        hidden_size=2,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=2**-0.5,
        use_rope=True,
    )
    attn = AttentionWithRoPE(config)
    hidden_states = jnp.zeros((1, 2, config.hidden_size), dtype=jnp.float32)
    params = attn.init(jax.random.PRNGKey(0), hidden_states, cos_sin=_identity_cos_sin(config, hidden_states.shape[1]))

    with pytest.raises(exception, match=match):
        attn.apply(params, hidden_states, cos_sin=cos_sin)


@pytest.mark.parametrize(
    "attention_mask,exception,match",
    [
        (jnp.ones((1, 2), dtype=bool), ValueError, "attention_mask shape must match"),
        (jnp.ones((1, 3), dtype=jnp.float32), TypeError, "attention_mask must be boolean or integer"),
        (jnp.array([[1, 2, 1]], dtype=jnp.int32), ValueError, "integer values must be 0 or 1"),
    ],
)
def test_attention_mha_rejects_invalid_attention_mask(attention_mask, exception, match):
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    query_states = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    key_states = jnp.zeros((1, 3, 1, 1), dtype=jnp.float32)
    value_states = jnp.arange(3, dtype=jnp.float32).reshape(1, 3, 1, 1)

    with pytest.raises(exception, match=match):
        attn.apply(
            {},
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            method=Attention.mha,
        )


@pytest.mark.parametrize(
    "sliding_window,exception,match",
    [
        (True, TypeError, "sliding_window must be an integer"),
        (np.bool_(True), TypeError, "sliding_window must be an integer"),
        (1.5, TypeError, "sliding_window must be an integer"),
        (0, ValueError, "sliding_window must be positive"),
    ],
)
def test_attention_mha_rejects_invalid_sliding_window(sliding_window, exception, match):
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    query_states = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    key_states = jnp.zeros((1, 3, 1, 1), dtype=jnp.float32)
    value_states = jnp.arange(3, dtype=jnp.float32).reshape(1, 3, 1, 1)

    with pytest.raises(exception, match=match):
        attn.apply(
            {},
            query_states,
            key_states,
            value_states,
            sliding_window=sliding_window,
            method=Attention.mha,
        )


@pytest.mark.parametrize(
    "alibi_slope,exception,match",
    [
        (jnp.ones((1, 1), dtype=jnp.float32), ValueError, "1D array"),
        (jnp.ones((2,), dtype=jnp.float32), ValueError, "contain 1 values"),
        (jnp.ones((1,), dtype=jnp.int32), TypeError, "floating point"),
    ],
)
def test_attention_mha_rejects_invalid_alibi_slope(alibi_slope, exception, match):
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    query_states = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    key_states = jnp.zeros((1, 3, 1, 1), dtype=jnp.float32)
    value_states = jnp.arange(3, dtype=jnp.float32).reshape(1, 3, 1, 1)

    with pytest.raises(exception, match=match):
        attn.apply(
            {},
            query_states,
            key_states,
            value_states,
            alibi_slope=alibi_slope,
            method=Attention.mha,
        )


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, -jnp.inf])
def test_attention_mha_rejects_non_finite_alibi_slope(value):
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=False,
    )
    attn = Attention(config)
    query_states = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    key_states = jnp.zeros((1, 3, 1, 1), dtype=jnp.float32)
    value_states = jnp.arange(3, dtype=jnp.float32).reshape(1, 3, 1, 1)

    with pytest.raises(ValueError, match="alibi_slope must contain only finite values"):
        attn.apply(
            {},
            query_states,
            key_states,
            value_states,
            alibi_slope=jnp.array([value], dtype=jnp.float32),
            method=Attention.mha,
        )


@pytest.mark.parametrize(
    "model_type,with_rope",
    [
        ("llama", False),
        ("llama", True),
        ("neox", False),
        ("neox", True),
    ],
)
def test_attention_with_rope(attention_factory, model_type, with_rope, cos_sin_factory):
    with jax.default_device(jax.devices("cpu")[0]):
        bs, seq_len = 4, 10
        cos_sin = cos_sin_factory(model_type)
        hf, (attn, init_param) = attention_factory(model_type=model_type, with_rope=with_rope)
        if not with_rope:
            hf.rotary_emb = DummyPosEmb()

        match model_type:
            case "llama":
                params = to_llama_jax_params(hf, dtype="float32")
            case "neox":
                params = to_neox_jax_params(hf, dtype="float32")
            case _:
                raise

        key = jax.random.PRNGKey(0)

        hidden_size = attn.config.hidden_size
        hidden = jax.random.uniform(key, (bs, seq_len, hidden_size), dtype=jnp.float32)

        # When RoPE is available, a "cache" field would exist in init_param which is needed for fwd pass
        output = attn.apply({**init_param, "params": params["params"]}, hidden, cos_sin=cos_sin, output_attentions=True)
        out = output.attention_output
        # out_flash = attn.apply({**init_param, "params": params["params"]}, hidden, use_flash=True)
        # out_flash = out_flash.attention_output
        kwargs = {
            "hidden_states": torch.tensor(np.array(hidden)),
            "position_ids": torch.arange(seq_len)[None],
            "attention_mask": torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    fill_value=float("-inf"),
                    dtype=torch.float32,
                ),
                diagonal=1,
            )[None, None].repeat(bs, 1, 1, 1),
            "output_attentions": True,
        }
        # if model_type == "neox":
        rotary_dim = int(attn.config.head_dim * attn.config.rotary_pct)
        if with_rope:
            kwargs["position_embeddings"] = tuple(
                map(lambda x: torch.tensor(np.array(x[None, :seq_len, :rotary_dim])), cos_sin)
            )
        else:
            kwargs["position_embeddings"] = (
                torch.ones((1, seq_len, rotary_dim), dtype=torch.float32),
                torch.zeros((1, seq_len, rotary_dim), dtype=torch.float32),
            )
        with torch.no_grad():
            output2 = hf(**kwargs)
            out2 = output2[0]

        assert out.shape == out2.shape
        # print(np.abs(out - out2.numpy()).max())
        assert np.allclose(out, out2.numpy(), atol=1e-5)


"""
@pytest.mark.parametrize("with_rope", [False, True])
@pytest.mark.parametrize("seq_len", [1, 60, 497])
def test_flash_attention(attention_with_rope_small, attention_small, with_rope, seq_len):
    bs = 4
    if with_rope:
        attn, init_param = attention_with_rope_small
    else:
        attn, init_param = attention_small

    params = init_param

    key = jax.random.PRNGKey(0)

    hidden_size = attn.config.hidden_size
    hidden = jax.random.uniform(key, (bs, seq_len, hidden_size), dtype=jnp.float32)

    out = attn.apply({**init_param, "params": params["params"]}, hidden)
    out = out.attention_output
    out_flash = attn.apply({**init_param, "params": params["params"]}, hidden, use_flash=True)
    out_flash = out_flash.attention_output

    # Eager and JAX FA somehow have a bit of discrepancy, though still tolerable for inference
    assert jnp.allclose(out, out_flash, atol=1e-2)
"""
