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
from flax import linen as nn

from jaxml.cache import KVCache
from jaxml.config import ModelConfig
from jaxml.models.deepseek import DeepseekMoE, MLAAttention

pytestmark = pytest.mark.milestone


def _tiny_hf_deepseek_config(**overrides):
    from transformers import DeepseekV3Config

    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=4,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=0,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=100,
        max_position_embeddings=64,
    )
    kwargs.update(overrides)
    return DeepseekV3Config(**kwargs)


def _np(tensor):
    return tensor.detach().cpu().float().numpy()


def _moe_params_from_hf(hf_moe):
    """Map an HF DeepseekV3MoE state dict to the jaxml DeepseekMoE param tree.

    HF stores routed experts as stacked 3D tensors with the torch ``Linear``
    convention (out, in); jaxml expects ``x @ kernel``, so each expert kernel is
    transposed on its last two axes.
    """
    sd = hf_moe.state_dict()
    params = {
        "gate_kernel": _np(sd["gate.weight"]).T,
        "e_score_correction_bias": _np(sd["gate.e_score_correction_bias"]),
        "gate_up_kernel": _np(sd["experts.gate_up_proj"]).transpose(0, 2, 1),
        "down_kernel": _np(sd["experts.down_proj"]).transpose(0, 2, 1),
        "shared_gate_proj": {"kernel": _np(sd["shared_experts.gate_proj.weight"]).T},
        "shared_up_proj": {"kernel": _np(sd["shared_experts.up_proj.weight"]).T},
        "shared_down_proj": {"kernel": _np(sd["shared_experts.down_proj.weight"]).T},
    }
    return {"params": jax.tree.map(jnp.asarray, params)}


def test_deepseek_moe_matches_hf():
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

    torch.manual_seed(0)
    hf_cfg = _tiny_hf_deepseek_config()
    hf_moe = DeepseekV3MoE(hf_cfg).eval()
    # DeepseekV3NaiveMoe allocates expert weights with torch.empty (uninitialized);
    # initialize every parameter + the correction-bias buffer to finite values.
    with torch.no_grad():
        for param in hf_moe.parameters():
            param.normal_(0.0, 0.02)
        hf_moe.gate.e_score_correction_bias.normal_(0.0, 0.02)

    config = ModelConfig.from_hf(hf_cfg)
    model = DeepseekMoE(config, dtype=jnp.float32)

    bs, seq_len = 2, 5
    with jax.default_device(jax.devices("cpu")[0]):
        x = jax.random.normal(jax.random.PRNGKey(0), (bs, seq_len, config.hidden_size), dtype=jnp.float32)
        params = _moe_params_from_hf(hf_moe)
        # Converted params must match a fresh init leaf-for-leaf in name and shape
        # (apply would also raise on a mismatch, but check explicitly for clarity).
        init_shapes = jax.tree.map(jnp.shape, nn.meta.unbox(model.init(jax.random.PRNGKey(1), x)))
        assert init_shapes == jax.tree.map(jnp.shape, params)

        y = model.apply(params, x)
        with torch.no_grad():
            y2 = hf_moe(torch.tensor(np.array(x))).numpy()

    assert np.allclose(y, y2, atol=1e-5)


def _mla_params_from_hf(hf_attn, config):
    """Map an HF DeepseekV3Attention state dict to the jaxml MLAAttention tree."""
    sd = hf_attn.state_dict()
    nh, qk = config.num_heads, config.qk_head_dim
    nope, v = config.qk_nope_head_dim, config.v_head_dim
    p = {}
    if "q_proj.weight" in sd:
        p["q_proj"] = {"kernel": _np(sd["q_proj.weight"]).T.reshape(-1, nh, qk)}
    else:
        p["q_a_proj"] = {"kernel": _np(sd["q_a_proj.weight"]).T}
        p["q_a_layernorm"] = {"weight": _np(sd["q_a_layernorm.weight"])}
        p["q_b_proj"] = {"kernel": _np(sd["q_b_proj.weight"]).T.reshape(-1, nh, qk)}
    p["kv_a_proj_with_mqa"] = {"kernel": _np(sd["kv_a_proj_with_mqa.weight"]).T}
    p["kv_a_layernorm"] = {"weight": _np(sd["kv_a_layernorm.weight"])}
    p["kv_b_proj"] = {"kernel": _np(sd["kv_b_proj.weight"]).T.reshape(-1, nh, nope + v)}
    p["o_proj"] = {"kernel": _np(sd["o_proj.weight"]).T}
    return {"params": jax.tree.map(jnp.asarray, p)}


def test_mla_attention_prefill_matches_hf():
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3Attention,
        DeepseekV3RotaryEmbedding,
    )

    torch.manual_seed(0)
    hf_cfg = _tiny_hf_deepseek_config()
    hf_cfg._attn_implementation = "eager"
    hf_attn = DeepseekV3Attention(hf_cfg, layer_idx=0).eval()
    rotary = DeepseekV3RotaryEmbedding(hf_cfg)

    config = ModelConfig.from_hf(hf_cfg)
    model = MLAAttention(config, dtype=jnp.float32)

    bs, seq_len = 2, 6
    with jax.default_device(jax.devices("cpu")[0]):
        x = jax.random.normal(jax.random.PRNGKey(0), (bs, seq_len, config.hidden_size), dtype=jnp.float32)
        xt = torch.tensor(np.array(x))
        position_ids = torch.arange(seq_len)[None].expand(bs, -1)
        cos, sin = rotary(xt, position_ids)
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)[None, None]
        with torch.no_grad():
            hf_out = hf_attn(xt, position_embeddings=(cos, sin), attention_mask=causal, past_key_values=None)[0].numpy()

        params = _mla_params_from_hf(hf_attn, config)
        cos_sin = (jnp.asarray(cos.numpy()), jnp.asarray(sin.numpy()))
        out = model.apply(params, x, cos_sin).attention_output

    assert np.allclose(out, hf_out, atol=1e-5)


def test_mla_attention_decode_cache_consistency():
    """Incremental decode through KVCache must match full-sequence prefill."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding

    hf_cfg = _tiny_hf_deepseek_config()
    rotary = DeepseekV3RotaryEmbedding(hf_cfg)
    config = ModelConfig.from_hf(hf_cfg)
    model = MLAAttention(config, dtype=jnp.float32)

    bs, seq_len = 2, 4
    with jax.default_device(jax.devices("cpu")[0]):
        x = jax.random.normal(jax.random.PRNGKey(0), (bs, seq_len, config.hidden_size), dtype=jnp.float32)
        position_ids = torch.arange(seq_len)[None].expand(bs, -1)
        cos, sin = rotary(torch.tensor(np.array(x)), position_ids)
        cos, sin = jnp.asarray(cos.numpy()), jnp.asarray(sin.numpy())

        params = model.init(jax.random.PRNGKey(1), x, (cos, sin))
        full = model.apply(params, x, (cos, sin)).attention_output

        cache = KVCache.init(max_seq_len=seq_len, dtype=jnp.float32)
        prefill = model.apply(params, x[:, : seq_len - 1], (cos[:, : seq_len - 1], sin[:, : seq_len - 1]), kv_cache=cache)
        decode = model.apply(
            params,
            x[:, seq_len - 1 :],
            (cos[:, seq_len - 1 :], sin[:, seq_len - 1 :]),
            kv_cache=prefill.kv_cache,
        )

    assert np.allclose(decode.attention_output[:, -1], full[:, -1], atol=1e-5)
