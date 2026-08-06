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

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jaxml.config import ModelConfig
from jaxml.hf_utils import (
    _normalize_hf_architecture,
    load_model_from_hf,
    to_kimi_linear_jax_params,
)
from jaxml.models.deepseek import DeepseekModel, MLAAttention
from jaxml.models.kimi_linear import KDACache, KimiLinearModelWithHead, kda_recurrent_scan

pytestmark = pytest.mark.milestone


def _tiny_hf_kimi_linear_config(**overrides):
    values = dict(
        model_type="kimi_linear",
        vocab_size=64,
        hidden_size=32,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        model_max_length=64,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        rope_scaling=None,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        mla_use_nope=True,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=16,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=2.446,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        first_k_dense_replace=1,
        moe_layer_freq=1,
        linear_attn_config={
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
            "num_heads": 4,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
        },
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _tiny_kimi_linear_state_dict(config, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.02

    hidden = config.hidden_size
    kda_heads = config.linear_attn_config["num_heads"]
    kda_dim = config.linear_attn_config["head_dim"]
    conv_width = config.linear_attn_config["short_conv_kernel_size"]
    mla_heads = config.num_attention_heads
    qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    states = {
        "model.embed_tokens.weight": randn(config.vocab_size, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": randn(config.vocab_size, hidden),
    }
    for idx in range(config.num_hidden_layers):
        layer = f"model.layers.{idx}."
        attention = layer + "self_attn."
        states[layer + "input_layernorm.weight"] = torch.ones(hidden)
        states[layer + "post_attention_layernorm.weight"] = torch.ones(hidden)
        if idx + 1 in config.linear_attn_config["kda_layers"]:
            for name in ("q_proj", "k_proj", "v_proj"):
                states[attention + name + ".weight"] = randn(kda_heads * kda_dim, hidden)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                states[attention + name + ".weight"] = randn(kda_heads * kda_dim, 1, conv_width)
            states[attention + "A_log"] = torch.log(torch.linspace(1.0, 2.0, kda_heads)).reshape(1, 1, -1, 1)
            states[attention + "dt_bias"] = randn(kda_heads * kda_dim)
            states[attention + "f_a_proj.weight"] = randn(kda_dim, hidden)
            states[attention + "f_b_proj.weight"] = randn(kda_heads * kda_dim, kda_dim)
            states[attention + "b_proj.weight"] = randn(kda_heads, hidden)
            states[attention + "g_a_proj.weight"] = randn(kda_dim, hidden)
            states[attention + "g_b_proj.weight"] = randn(kda_heads * kda_dim, kda_dim)
            states[attention + "o_norm.weight"] = torch.ones(kda_dim)
            states[attention + "o_proj.weight"] = randn(hidden, kda_heads * kda_dim)
        else:
            states[attention + "q_proj.weight"] = randn(mla_heads * qk_dim, hidden)
            states[attention + "kv_a_proj_with_mqa.weight"] = randn(
                config.kv_lora_rank + config.qk_rope_head_dim,
                hidden,
            )
            states[attention + "kv_a_layernorm.weight"] = torch.ones(config.kv_lora_rank)
            states[attention + "kv_b_proj.weight"] = randn(
                mla_heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            )
            states[attention + "o_proj.weight"] = randn(hidden, mla_heads * config.v_head_dim)

        if idx >= config.first_k_dense_replace and idx % config.moe_layer_freq == 0:
            moe = layer + "block_sparse_moe."
            states[moe + "gate.weight"] = randn(config.num_experts, hidden)
            states[moe + "gate.e_score_correction_bias"] = randn(config.num_experts)
            for expert_idx in range(config.num_experts):
                expert = moe + f"experts.{expert_idx}."
                states[expert + "w1.weight"] = randn(config.moe_intermediate_size, hidden)
                states[expert + "w2.weight"] = randn(hidden, config.moe_intermediate_size)
                states[expert + "w3.weight"] = randn(config.moe_intermediate_size, hidden)
            shared = config.moe_intermediate_size * config.num_shared_experts
            states[moe + "shared_experts.gate_proj.weight"] = randn(shared, hidden)
            states[moe + "shared_experts.up_proj.weight"] = randn(shared, hidden)
            states[moe + "shared_experts.down_proj.weight"] = randn(hidden, shared)
        else:
            mlp = layer + "mlp."
            states[mlp + "gate_proj.weight"] = randn(config.intermediate_size, hidden)
            states[mlp + "up_proj.weight"] = randn(config.intermediate_size, hidden)
            states[mlp + "down_proj.weight"] = randn(hidden, config.intermediate_size)
    return states


@pytest.mark.critical
def test_kimi_linear_config_maps_hybrid_attention_and_moe_schedule():
    config = ModelConfig.from_hf(_tiny_hf_kimi_linear_config())

    assert config.architecture == "kimi_linear"
    assert config.max_position_embeddings == 64
    assert config.mla_use_nope is True
    assert config.use_rope is False
    assert config.kda_layers == (0, 1, 2)
    assert config.kda_num_heads == 4
    assert config.kda_head_dim == 8
    assert config.kda_short_conv_kernel_size == 4
    assert [config.is_kda_layer(idx) for idx in range(4)] == [True, True, True, False]
    assert [config.is_moe_layer(idx) for idx in range(4)] == [False, True, True, True]
    assert config.n_routed_experts == 4
    assert config.num_experts_per_tok == 2
    assert config.n_group == 1
    assert config.norm_topk_prob is True
    assert _normalize_hf_architecture("kimi") == "kimi_linear"
    assert _normalize_hf_architecture("kimi-k2") == "kimi_k2"


@pytest.mark.milestone
def test_kimi_linear_nope_mla_runs_without_rotary_tables():
    config = ModelConfig.from_hf(_tiny_hf_kimi_linear_config())
    attention = MLAAttention(config, dtype=jnp.float32)
    hidden = jax.random.normal(jax.random.PRNGKey(0), (2, 5, config.hidden_size))

    with jax.default_device(jax.devices("cpu")[0]):
        params = attention.init(jax.random.PRNGKey(1), hidden, cos_sin=None)
        output = attention.apply(params, hidden, cos_sin=None).attention_output

    assert output.shape == hidden.shape
    assert np.all(np.isfinite(output))


@pytest.mark.milestone
def test_kimi_linear_config_is_rejected_by_deepseek_backbone():
    config = ModelConfig.from_hf(_tiny_hf_kimi_linear_config())

    with pytest.raises(ValueError, match="require KimiLinearModel"):
        DeepseekModel(config).init(jax.random.PRNGKey(0), jnp.array([[1]], dtype=jnp.int32))


@pytest.mark.milestone
def test_kda_recurrent_scan_matches_explicit_masked_oracle():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(2, 5, 2, 3)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    v = rng.normal(size=q.shape).astype(np.float32)
    log_decay = -rng.uniform(0.01, 0.4, size=q.shape).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, size=q.shape[:-1]).astype(np.float32)
    mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool)

    output, final_state = kda_recurrent_scan(q, k, v, log_decay, beta, token_mask=mask)

    q32 = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    q32 *= q.shape[-1] ** -0.5
    k32 = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    state = np.zeros((2, 2, 3, 3), dtype=np.float32)
    expected = np.zeros_like(v)
    for token_idx in range(q.shape[1]):
        decayed = state * np.exp(log_decay[:, token_idx])[..., None]
        prediction = np.einsum("bhd,bhdv->bhv", k32[:, token_idx], decayed)
        error = v[:, token_idx] - prediction
        candidate = decayed + np.einsum(
            "bhd,bhv->bhdv",
            beta[:, token_idx, :, None] * k32[:, token_idx],
            error,
        )
        candidate_output = np.einsum("bhd,bhdv->bhv", q32[:, token_idx], candidate)
        valid = mask[:, token_idx]
        state = np.where(valid[:, None, None, None], candidate, state)
        expected[:, token_idx] = np.where(valid[:, None, None], candidate_output, 0.0)

    assert np.allclose(output, expected, atol=2e-6, rtol=2e-6)
    assert np.allclose(final_state, state, atol=2e-6, rtol=2e-6)
    assert final_state.dtype == jnp.float32


@pytest.mark.milestone
def test_kimi_linear_hybrid_cache_matches_full_sequence():
    hf_config = _tiny_hf_kimi_linear_config(
        num_hidden_layers=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 4,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
        },
    )
    config = ModelConfig.from_hf(hf_config)
    model = KimiLinearModelWithHead(config, dtype=jnp.float32, moe_implementation="dense")
    input_ids = jnp.array([[1, 2, 3, 4, 5], [7, 6, 5, 4, 3]], dtype=jnp.int32)

    with jax.default_device(jax.devices("cpu")[0]):
        params = model.init(jax.random.PRNGKey(1), input_ids)
        full = model.apply(params, input_ids).logits[:, -1]
        caches = model.init_cache(max_seq_len=8)
        prefill = model.apply(params, input_ids[:, :4], kv_caches=caches, use_cache=True)
        decode = model.apply(
            params,
            input_ids[:, 4:],
            attention_mask=prefill.kv_caches[0].mask,
            kv_caches=prefill.kv_caches,
            use_cache=True,
        )

    assert isinstance(decode.kv_caches[0], KDACache)
    assert decode.kv_caches[0].recurrent_state.dtype == jnp.float32
    assert np.allclose(decode.logits[:, 0], full, atol=1e-5, rtol=1e-5)


@pytest.mark.milestone
def test_kimi_linear_converter_and_direct_safetensors_loader_build_runnable_hybrid_tree(tmp_path):
    hf_config = _tiny_hf_kimi_linear_config(
        num_hidden_layers=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 4,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
        },
    )
    hf_states = _tiny_kimi_linear_state_dict(hf_config)
    hf_model = SimpleNamespace(config=hf_config, state_dict=lambda: hf_states)
    config = ModelConfig.from_hf(hf_config)
    model = KimiLinearModelWithHead(config, dtype=jnp.float32, moe_implementation="dense")

    with jax.default_device(jax.devices("cpu")[0]):
        params = to_kimi_linear_jax_params(hf_model, dtype="float32")
        logits = model.apply(params, jnp.array([[1, 2, 3]], dtype=jnp.int32)).logits

    from safetensors.torch import save_file

    first_shard = {
        key: value
        for key, value in hf_states.items()
        if key.startswith("model.layers.0.") or key == "model.embed_tokens.weight"
    }
    second_shard = {key: value for key, value in hf_states.items() if key not in first_shard}
    save_file(first_shard, tmp_path / "model-00001-of-00002.safetensors")
    save_file(second_shard, tmp_path / "model-00002-of-00002.safetensors")
    weight_map = {
        key: "model-00001-of-00002.safetensors" if key in first_shard else "model-00002-of-00002.safetensors"
        for key in hf_states
    }
    (tmp_path / "config.json").write_text(json.dumps(vars(hf_config)))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    loaded_model, loaded_params = load_model_from_hf(tmp_path, dtype="float32")

    kda_params = params["params"]["model"]["layers_0"]["self_attn"]
    moe_params = params["params"]["model"]["layers_1"]["block_sparse_moe"]
    assert kda_params["q_conv1d"]["weight"].shape == (4, 8, 4)
    assert kda_params["A_log"].shape == (4,)
    assert kda_params["A_log"].dtype == np.float32
    assert kda_params["dt_bias"].dtype == np.float32
    assert moe_params["gate_up_kernel"].shape == (4, 32, 32)
    assert moe_params["down_kernel"].shape == (4, 16, 32)
    assert logits.shape == (1, 3, hf_config.vocab_size)
    assert np.all(np.isfinite(logits))
    assert isinstance(loaded_model, KimiLinearModelWithHead)
    assert jax.tree.structure(loaded_params) == jax.tree.structure(params)
    assert all(np.array_equal(left, right) for left, right in zip(jax.tree.leaves(loaded_params), jax.tree.leaves(params)))
