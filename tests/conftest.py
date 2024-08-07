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
import pytest

from jaxml.config import ModelConfig
from jaxml.models.llama import LlamaMLP as LlamaMLPJAX, LlamaDecoder, LlamaModel, LlamaForCausalLM as LlamaForCausalLMJAX
from jaxml.nn.attention import Attention, AttentionWithRoPE


@pytest.fixture
def config_small():
    return ModelConfig(
        head_dim=8,
        num_heads=6,
        num_layers=2,
        intermediate_ratio=(3, 1),
        max_position_embeddings=256,
        vocab_size=1024,
        num_kv_heads=3,
        norm_eps=1e-6,
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
        rms_norm_eps=1e-6,
        attn_implementation="eager",
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


def get_layer_and_param(cls, config, discrete=False):
    layer = cls(config=config, dtype=jnp.float32)
    if discrete:
        x = jnp.zeros((2, 10), dtype=jnp.int32)
    else:
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
def llama_mlp_cls():
    return LlamaMLPJAX


@pytest.fixture
def llama_decoder_cls():
    return LlamaDecoder

@pytest.fixture
def llama_model_cls():
    return LlamaModel

@pytest.fixture
def llama_causal_model_cls():
    return LlamaForCausalLMJAX


@pytest.fixture
def attention_small(attn_cls, config_small):
    return get_layer_and_param(attn_cls, config_small)


@pytest.fixture
def attention_with_rope_small(attn_with_rope_cls, config_small):
    return get_layer_and_param(attn_with_rope_cls, config_small)


@pytest.fixture
def llama_mlp(llama_mlp_cls, config_small):
    return get_layer_and_param(llama_mlp_cls, config_small)


@pytest.fixture
def llama_decoder(llama_decoder_cls, config_small):
    return get_layer_and_param(llama_decoder_cls, config_small)

@pytest.fixture
def llama_model(llama_model_cls, config_small):
    return get_layer_and_param(llama_model_cls, config_small, discrete=True)

@pytest.fixture
def llama_causal_model(llama_causal_model_cls, config_small):
    return get_layer_and_param(llama_causal_model_cls, config_small, discrete=True)

@pytest.fixture
def hf_attention_with_rope(hf_llama_config):
    from transformers.models.llama.modeling_llama import LlamaAttention

    return LlamaAttention(hf_llama_config, layer_idx=0)


@pytest.fixture
def hf_llama_mlp(hf_llama_config):
    from transformers.models.llama.modeling_llama import LlamaMLP

    return LlamaMLP(hf_llama_config)


@pytest.fixture
def hf_llama_decoder(hf_llama_config):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    return LlamaDecoderLayer(hf_llama_config, layer_idx=0)


@pytest.fixture
def hf_llama_model(hf_llama_config):
    from transformers.models.llama.modeling_llama import LlamaModel

    return LlamaModel(hf_llama_config)


@pytest.fixture
def hf_llama_causal_model(hf_llama_config):
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    return LlamaForCausalLM(hf_llama_config)


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
