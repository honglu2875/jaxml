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

from jaxml.config import ModelConfig
from jaxml.models.deepseek import DeepseekMoE

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
