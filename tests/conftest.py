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

from jaxml.nn.attention import Attention, AttentionWithRoPE
from jaxml.config import ModelConfig
import pytest
import jax
import jax.numpy as jnp


@pytest.fixture
def config_small():
    return ModelConfig(
        head_dim=8,
        num_heads=6,
        num_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_kv_heads=3
    )

@pytest.fixture
def hf_llama_config():
    from transformers import LlamaConfig
    return LlamaConfig(
        hidden_size=48,
        intermediate_size=144,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
        hidden_act="silu",
    )
    
@pytest.fixture
def hf_mistral_config():
    from transformers import MistralConfig
    return MistralConfig(
        hidden_size=48,
        intermediate_size=144,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
        sliding_window=16,
        hidden_act="silu",
    )
    
def get_layer_and_param(cls, config):
    layer = cls(config)
    x = jnp.zeros((2, 10, config.num_heads * config.head_dim), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    params = layer.init(key, x)
    return layer, params

@pytest.fixture
def attn_cls():
    return Attention

@pytest.fixture
def attn_with_rope_cls():
    return AttentionWithRoPE
    
@pytest.fixture
def attention_small(attn_cls, config_small):
    return get_layer_and_param(attn_cls, config_small)

@pytest.fixture
def attention_with_rope_small(attn_with_rope_cls, config_small):
    return get_layer_and_param(attn_with_rope_cls, config_small)

@pytest.fixture
def hf_attention_with_rope(hf_llama_config):
    from transformers.models.llama.modeling_llama import LlamaAttention
    return LlamaAttention(hf_llama_config, layer_idx=0)
    
@pytest.fixture
def hf_attention_mistral(hf_mistral_config):
    from transformers.models.mistral.modeling_mistral import MistralAttention
    return MistralAttention(hf_mistral_config, layer_idx=0)

@pytest.fixture
def torch_dense():
    import torch
    return torch.nn.Linear(48, 64)

@pytest.fixture
def jax_dense():
    from jaxml.nn.linear import DenseGeneral
    model = DenseGeneral(
        features=(64,),
        axis=-1,
        kernel_axes=("in", "out"),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        name="test",
        use_bias=True,
    )
    return model


