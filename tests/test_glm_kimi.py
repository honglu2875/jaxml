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

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jaxml.config import ModelConfig
from jaxml.hf_utils import to_deepseek_jax_params
from jaxml.models.deepseek import DeepseekModelWithHead, _deepseek_cos_sin

pytestmark = pytest.mark.milestone


def _tiny_glm_config():
    from transformers import Glm4MoeLiteConfig

    return Glm4MoeLiteConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        mlp_layer_types=["dense", "sparse"],
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        routed_scaling_factor=1.25,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=32,
    )


def _tiny_kimi_yarn_config():
    from transformers import DeepseekV3Config

    class KimiK2Config(DeepseekV3Config):
        model_type = "kimi_k2"

    return KimiK2Config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        vocab_size=64,
        max_position_embeddings=512,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 64,
            "beta_fast": 12.0,
            "beta_slow": 2.0,
            "mscale": 0.8,
            "mscale_all_dim": 0.4,
            "truncate": False,
        },
    )


@pytest.mark.critical
def test_glm4_moe_lite_config_maps_to_shared_deepseek_shape():
    config = ModelConfig.from_hf(_tiny_glm_config())

    assert config.hidden_size == 32
    assert config.intermediate_size == 64
    assert config.num_layers == 2
    assert config.num_heads == 4
    assert config.q_lora_rank == 16
    assert config.kv_lora_rank == 8
    assert config.qk_nope_head_dim == 8
    assert config.qk_rope_head_dim == 8
    assert config.qk_head_dim == 16
    assert config.v_head_dim == 8
    assert config.n_routed_experts == 4
    assert config.n_shared_experts == 1
    assert config.num_experts_per_tok == 2
    assert config.first_k_dense_replace == 1
    assert config.moe_layer_freq == 1
    assert config.routed_scaling_factor == 1.25
    assert config.rope_type == "default"


@pytest.mark.milestone
def test_glm4_moe_lite_logits_match_hf_through_deepseek_converter():
    from transformers import Glm4MoeLiteForCausalLM

    torch.manual_seed(0)
    hf_config = _tiny_glm_config()
    hf_config._attn_implementation = "eager"
    hf_model = Glm4MoeLiteForCausalLM(hf_config).eval()

    config = ModelConfig.from_hf(hf_config)
    model = DeepseekModelWithHead(config, dtype=jnp.float32)
    input_ids = np.array([[1, 2, 3, 4, 5], [7, 6, 5, 4, 3]], dtype=np.int32)

    with jax.default_device(jax.devices("cpu")[0]):
        params = to_deepseek_jax_params(hf_model, dtype="float32")
        logits = model.apply(params, jnp.asarray(input_ids), use_cache=False).logits
        with torch.no_grad():
            hf_logits = hf_model(input_ids=torch.from_numpy(input_ids), use_cache=False).logits.numpy()

    assert logits.shape == hf_logits.shape
    assert np.allclose(logits, hf_logits, atol=1e-4, rtol=1e-4)


@pytest.mark.milestone
def test_glm_converter_accepts_raw_per_expert_checkpoint_layout():
    from transformers import Glm4MoeLiteForCausalLM

    hf_model = Glm4MoeLiteForCausalLM(_tiny_glm_config()).eval()
    fused = to_deepseek_jax_params(hf_model, dtype="float32")
    raw_state = dict(hf_model.state_dict())

    prefix = "model.layers.1.mlp."
    gate_up = raw_state.pop(prefix + "experts.gate_up_proj")
    down = raw_state.pop(prefix + "experts.down_proj")
    intermediate = gate_up.shape[1] // 2
    for expert_idx in range(gate_up.shape[0]):
        expert_prefix = prefix + f"experts.{expert_idx}."
        raw_state[expert_prefix + "gate_proj.weight"] = gate_up[expert_idx, :intermediate]
        raw_state[expert_prefix + "up_proj.weight"] = gate_up[expert_idx, intermediate:]
        raw_state[expert_prefix + "down_proj.weight"] = down[expert_idx]

    raw_model = SimpleNamespace(config=hf_model.config, state_dict=lambda: raw_state)
    raw = to_deepseek_jax_params(raw_model, dtype="float32")

    assert jax.tree.structure(raw) == jax.tree.structure(fused)
    assert all(np.array_equal(left, right) for left, right in zip(jax.tree.leaves(raw), jax.tree.leaves(fused)))


@pytest.mark.milestone
def test_kimi_yarn_rotary_tables_match_hf():
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding

    hf_config = _tiny_kimi_yarn_config()
    config = ModelConfig.from_hf(hf_config)
    position_ids = np.array(
        [[0, 1, 63, 64, 127, 511], [3, 17, 65, 129, 255, 510]],
        dtype=np.int32,
    )

    hf_rotary = DeepseekV3RotaryEmbedding(hf_config)
    with torch.no_grad():
        hf_cos, hf_sin = hf_rotary(
            torch.zeros((*position_ids.shape, hf_config.hidden_size), dtype=torch.float32),
            torch.from_numpy(position_ids),
        )

    cos, sin = _deepseek_cos_sin(jnp.asarray(position_ids), config, jnp.float32)

    assert hf_config.model_type == "kimi_k2"
    assert config.rope_type == "yarn"
    assert config.rope_factor == 8.0
    assert config.rope_original_max_position_embeddings == 64
    assert config.rope_beta_fast == 12.0
    assert config.rope_beta_slow == 2.0
    assert config.rope_mscale == 0.8
    assert config.rope_mscale_all_dim == 0.4
    assert config.rope_truncate is False
    assert np.allclose(cos, hf_cos.numpy(), atol=2e-6, rtol=2e-6)
    assert np.allclose(sin, hf_sin.numpy(), atol=2e-6, rtol=2e-6)
