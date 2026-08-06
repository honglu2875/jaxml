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
#
# Multi-device tests fake a 4-device CPU mesh; run them with:
#   XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
#       uv run --frozen --extra dev pytest tests/test_sharding.py
# On a single-device run they skip.

import flax.linen as nn
import jax
import pytest
from jax.sharding import PartitionSpec

from jaxml.config import ModelConfig
from jaxml.inference_engine.engine import (
    SHARDING_RULES,
    Engine,
    InferenceConfig,
    _logical_axes_in_spec,
    _validate_rules_cover_axes,
)
from jaxml.models.gpt_neox import GPTNeoXModelWithHead
from jaxml.models.llama import LlamaModelWithHead

pytestmark = pytest.mark.milestone

needs_four_devices = pytest.mark.skipif(
    jax.device_count() < 4,
    reason="needs >=4 devices (run with XLA_FLAGS=--xla_force_host_platform_device_count=4)",
)


def _deepseek_config():
    from transformers import DeepseekV3Config

    hf = DeepseekV3Config(
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
        first_k_dense_replace=1,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=64,
        max_position_embeddings=64,
    )
    return ModelConfig.from_hf(hf)


def _llama_config():
    return ModelConfig(
        hidden_size=32,
        head_dim=8,
        num_heads=4,
        num_layers=2,
        intermediate_ratio=(2, 1),
        max_position_embeddings=64,
        vocab_size=64,
        num_kv_heads=2,
        attn_scale=8**-0.5,
    )


# --- Device-agnostic: notation + rule coverage ---------------------------------------


def test_inference_config_mesh_layout_and_num_devices():
    config = InferenceConfig(dp_size=2, tp_size=3, ep_size=4)
    assert config.mesh_layout == (2, 3, 4)
    assert config.num_devices == 24


def test_inference_config_rejects_non_positive_ep_size():
    with pytest.raises(ValueError, match="ep_size must be positive"):
        InferenceConfig(ep_size=0)


def test_sharding_rules_cover_llama_axes():
    model = LlamaModelWithHead(_llama_config())
    spec = nn.get_partition_spec(jax.eval_shape(model.init, jax.random.PRNGKey(0), jax.numpy.ones((1, 1), int)))
    _validate_rules_cover_axes(spec)  # must not raise


def test_sharding_rules_cover_gpt_neox_fused_qkv_axis():
    config = _llama_config().replace(num_kv_heads=4, use_bias=True)
    model = GPTNeoXModelWithHead(config)
    spec = nn.get_partition_spec(jax.eval_shape(model.init, jax.random.PRNGKey(0), jax.numpy.ones((1, 1), int)))

    assert "qkv" in _logical_axes_in_spec(spec)
    assert dict(SHARDING_RULES)["qkv"] is None
    _validate_rules_cover_axes(spec)  # must not raise


def test_sharding_rules_cover_deepseek_axes():
    from jaxml.models.deepseek import DeepseekModelWithHead

    model = DeepseekModelWithHead(_deepseek_config())
    spec = nn.get_partition_spec(jax.eval_shape(model.init, jax.random.PRNGKey(0), jax.numpy.ones((1, 1), int)))
    _validate_rules_cover_axes(spec)  # exercises expert / q_lora / kv_lora / kv_lora_rope


def test_validate_rules_cover_axes_rejects_unknown_axis():
    bad = {"params": {"w": PartitionSpec("embed", "not_a_real_axis")}}
    with pytest.raises(ValueError, match="not_a_real_axis"):
        _validate_rules_cover_axes(bad)


def test_expert_axis_has_a_rule():
    rules = dict(SHARDING_RULES)
    assert rules["expert"] == "expert"
    assert rules["batch"] == "data"


# --- Multi-device (4 fake CPU chips) -------------------------------------------------


@needs_four_devices
def test_engine_requires_exact_device_count():
    model = LlamaModelWithHead(_llama_config())
    # 2x2x1 == 4 is accepted.
    Engine(model, InferenceConfig(dp_size=2, tp_size=2), {"params": {}})
    # 2x1x1 == 2 != 4 is rejected.
    with pytest.raises(ValueError, match="requires exactly 2 devices"):
        Engine(model, InferenceConfig(dp_size=2), {"params": {}})


@needs_four_devices
def test_configured_mesh_has_named_axes():
    model = LlamaModelWithHead(_llama_config())
    engine = Engine(model, InferenceConfig(dp_size=2, tp_size=2, ep_size=1), {"params": {}})
    mesh = engine._create_configured_mesh()
    assert dict(mesh.shape) == {"data": 2, "model": 2, "expert": 1}
