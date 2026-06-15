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


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"query_states": jnp.zeros((1, 1, 1), dtype=jnp.float32)}, ValueError, "query_states must be a 4D array"),
        ({"key_states": jnp.zeros((1, 3, 1), dtype=jnp.float32)}, ValueError, "key_states must be a 4D array"),
        ({"value_states": jnp.zeros((1, 3, 1), dtype=jnp.float32)}, ValueError, "value_states must be a 4D array"),
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
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=use_rope,
    )
    attn = attn_cls(config)
    hidden_states = jnp.zeros((1, 2, 1), dtype=jnp.float32)
    params = attn.init(jax.random.PRNGKey(0), hidden_states)

    with pytest.raises(TypeError, match=match):
        attn.apply(params, hidden_states, **kwargs)


@pytest.mark.parametrize(
    "attn_cls,use_rope",
    [
        (Attention, False),
        (AttentionWithRoPE, True),
    ],
)
def test_attention_call_rejects_non_kv_cache(attn_cls, use_rope):
    config = ModelConfig(
        head_dim=1,
        hidden_size=1,
        num_heads=1,
        num_layers=1,
        max_position_embeddings=8,
        vocab_size=8,
        attn_scale=1.0,
        use_rope=use_rope,
    )
    attn = attn_cls(config)
    hidden_states = jnp.zeros((1, 2, 1), dtype=jnp.float32)
    params = attn.init(jax.random.PRNGKey(0), hidden_states)

    with pytest.raises(TypeError, match="kv_cache must be a KVCache"):
        attn.apply(params, hidden_states, kv_cache=object())


@pytest.mark.parametrize(
    "attention_mask,exception,match",
    [
        (jnp.ones((1, 2), dtype=bool), ValueError, "attention_mask shape must match"),
        (jnp.ones((1, 3), dtype=jnp.float32), TypeError, "attention_mask must be boolean or integer"),
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
