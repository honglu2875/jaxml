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

from jaxml.cache import KVCache
from jaxml.config import ModelConfig
from jaxml.hf_utils import to_qwen35_jax_params
from jaxml.models.qwen3_5 import (
    Qwen35DeltaNetCache,
    Qwen35ModelWithHead,
    Qwen35MoE,
    gated_delta_recurrent_scan,
)

pytestmark = pytest.mark.milestone


def _tiny_qwen35_config(**overrides):
    from transformers import Qwen3_5MoeTextConfig

    values = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.5,
        },
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        num_experts=4,
        num_experts_per_tok=2,
        attention_dropout=0.0,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    values.update(overrides)
    return Qwen3_5MoeTextConfig(**values)


def _gated_delta_numpy_oracle(query, key, value, g, beta, initial_state, token_mask):
    query = np.asarray(query, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    token_mask = np.asarray(token_mask, dtype=bool)

    query = query / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
    query *= query.shape[-1] ** -0.5
    key = key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    state = np.asarray(initial_state, dtype=np.float32).copy()
    output = np.zeros(value.shape, dtype=np.float32)

    for token_idx in range(query.shape[1]):
        decayed = state * np.exp(g[:, token_idx])[..., None, None]
        prediction = np.einsum("bhd,bhdv->bhv", key[:, token_idx], decayed)
        delta = beta[:, token_idx, :, None] * (value[:, token_idx] - prediction)
        candidate = decayed + np.einsum("bhd,bhv->bhdv", key[:, token_idx], delta)
        candidate_output = np.einsum("bhd,bhdv->bhv", query[:, token_idx], candidate)

        valid = token_mask[:, token_idx]
        state = np.where(valid[:, None, None, None], candidate, state)
        output[:, token_idx] = np.where(valid[:, None, None], candidate_output, 0.0)

    return output, state


def test_gated_delta_recurrent_scan_matches_masked_numpy_oracle_with_initial_state():
    rng = np.random.default_rng(0)
    query = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    key = rng.normal(size=query.shape).astype(np.float32)
    value = rng.normal(size=(2, 5, 3, 2)).astype(np.float32)
    g = -rng.uniform(0.01, 0.4, size=(2, 5, 3)).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, size=(2, 5, 3)).astype(np.float32)
    initial_state = rng.normal(size=(2, 3, 4, 2)).astype(np.float32)
    token_mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool)

    output, final_state = gated_delta_recurrent_scan(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        token_mask=token_mask,
    )
    expected_output, expected_state = _gated_delta_numpy_oracle(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        token_mask,
    )

    assert output.dtype == jnp.float32
    assert final_state.dtype == jnp.float32
    assert np.allclose(output, expected_output, atol=2e-6, rtol=2e-6)
    assert np.allclose(final_state, expected_state, atol=2e-6, rtol=2e-6)


def test_qwen35_tiny_hybrid_logits_match_hugging_face():
    from transformers import Qwen3_5MoeForCausalLM

    torch.manual_seed(0)
    hf_config = _tiny_qwen35_config()
    hf_config._attn_implementation = "eager"
    hf_model = Qwen3_5MoeForCausalLM(hf_config).eval()

    config = ModelConfig.from_hf(hf_config)
    model = Qwen35ModelWithHead(config, dtype=jnp.float32, moe_implementation="dense")
    input_ids = np.array([[1, 2, 3, 4, 5], [7, 6, 5, 4, 3]], dtype=np.int32)

    with jax.default_device(jax.devices("cpu")[0]):
        params = to_qwen35_jax_params(hf_model, dtype="float32")
        init_variables = model.init(jax.random.PRNGKey(0), jnp.asarray(input_ids))
        variables = {**init_variables, "params": params["params"]}
        logits = model.apply(variables, jnp.asarray(input_ids), use_cache=False).logits
        with torch.no_grad():
            hf_logits = hf_model(input_ids=torch.from_numpy(input_ids), use_cache=False).logits.numpy()

    assert logits.shape == hf_logits.shape
    assert np.allclose(logits, hf_logits, atol=2e-4, rtol=2e-4)


def test_qwen35_hybrid_prefill_decode_cache_matches_full_sequence():
    from transformers import Qwen3_5MoeForCausalLM

    torch.manual_seed(1)
    hf_config = _tiny_qwen35_config()
    hf_config._attn_implementation = "eager"
    hf_model = Qwen3_5MoeForCausalLM(hf_config).eval()
    config = ModelConfig.from_hf(hf_config)
    model = Qwen35ModelWithHead(config, dtype=jnp.float32, moe_implementation="dense")
    input_ids = jnp.array([[1, 2, 3, 4, 5], [7, 6, 5, 4, 3]], dtype=jnp.int32)

    with jax.default_device(jax.devices("cpu")[0]):
        params = to_qwen35_jax_params(hf_model, dtype="float32")
        init_variables = model.init(jax.random.PRNGKey(0), input_ids)
        variables = {**init_variables, "params": params["params"]}
        full_logits = model.apply(variables, input_ids, use_cache=False).logits[:, -1]
        caches = model.init_cache(max_seq_len=8)
        prefill = model.apply(variables, input_ids[:, :4], kv_caches=caches, use_cache=True)
        decode = model.apply(
            variables,
            input_ids[:, 4:],
            attention_mask=prefill.kv_caches[0].mask,
            kv_caches=prefill.kv_caches,
            use_cache=True,
        )

    delta_cache, attention_cache = decode.kv_caches
    assert isinstance(delta_cache, Qwen35DeltaNetCache)
    assert delta_cache.validate_state() is delta_cache
    conv_dim = sum(
        (
            2 * config.linear_num_key_heads * config.linear_key_head_dim,
            config.linear_num_value_heads * config.linear_value_head_dim,
        )
    )
    assert delta_cache.conv_state.shape == (2, conv_dim, config.linear_conv_kernel_dim)
    assert delta_cache.conv_state.dtype == jnp.float32
    assert delta_cache.recurrent_state.shape == (
        2,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    )
    assert delta_cache.recurrent_state.dtype == jnp.float32
    assert delta_cache.mask.shape == (2, 8)
    assert delta_cache.mask.dtype == jnp.bool_
    assert delta_cache.pos_id.dtype == jnp.int32
    assert np.array_equal(delta_cache.pos_id, np.array([[4], [4]], dtype=np.int32))

    assert isinstance(attention_cache, KVCache)
    assert attention_cache.k.shape == (2, 8, config.num_key_value_heads, config.head_dim)
    assert attention_cache.v.shape == attention_cache.k.shape
    assert attention_cache.k.dtype == jnp.float32
    assert attention_cache.v.dtype == jnp.float32
    assert attention_cache.mask.dtype == jnp.bool_
    assert attention_cache.pos_id.dtype == jnp.int32
    assert np.array_equal(attention_cache.mask, delta_cache.mask)
    assert np.allclose(decode.logits[:, 0], full_logits, atol=2e-5, rtol=2e-5)


@pytest.mark.tpu
def test_qwen35_bfloat16_gmm_matches_dense_on_tpu_without_dtype_promotion():
    hf_config = _tiny_qwen35_config(
        vocab_size=128,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts=8,
    )
    config = ModelConfig.from_hf(hf_config)
    dense = Qwen35MoE(config, dtype=jnp.bfloat16, moe_implementation="dense")
    gmm = Qwen35MoE(config, dtype=jnp.bfloat16, moe_implementation="gmm")
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, config.hidden_size), dtype=jnp.bfloat16)
    params = dense.init(jax.random.PRNGKey(1), x)

    dense_output = dense.apply(params, x)
    gmm_output = gmm.apply(params, x)

    assert dense_output.dtype == jnp.bfloat16
    assert gmm_output.dtype == jnp.bfloat16
    assert np.allclose(
        np.asarray(gmm_output, dtype=np.float32),
        np.asarray(dense_output, dtype=np.float32),
        atol=5e-2,
        rtol=5e-2,
    )
