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
from jaxml.models.gpt_neox import GPTNeoXModel, GPTNeoXModelWithHead
from jaxml.models.gemma3 import GemmaDecoder, GemmaModel, GemmaModelWithHead
from jaxml.models.llama import LlamaDecoder
from jaxml.models.llama import LlamaMLP as LlamaMLPJAX
from jaxml.models.llama import LlamaModel, LlamaModelWithHead
from jaxml.nn.attention import Attention, AttentionWithRoPE
from jaxml.nn.position import RotaryEmbedding


# ---------- Configs ---------- #
@pytest.fixture
def config_small():
    return ModelConfig(
        hidden_size=48,
        head_dim=8,
        num_heads=6,
        num_layers=2,
        intermediate_ratio=(3, 1),
        max_position_embeddings=512,
        vocab_size=1024,
        num_kv_heads=3,
        norm_eps=1e-6,
        attn_scale=8**-0.5,
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
def hf_neox_config():
    from transformers import GPTNeoXConfig

    return GPTNeoXConfig(
        hidden_size=48,
        intermediate_size=144,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        rotary_emb_base=10000.0,
        rotary_pct=0.5,
        use_parallel_residual=True,
        attn_implementation="eager",
        attention_bias=True,
    )


@pytest.fixture
def hf_gemma_config():
    from transformers import Gemma3TextConfig

    return Gemma3TextConfig(
        # hidden is larger than 6*8=48
        hidden_size=64,  
        head_dim=8,
        # rotary is 1M global/10k local, omitted
        intermediate_size=144,
        num_hidden_layers=4,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
        rope_scaling={
          "factor": 8.0,
          "rope_type": "linear"
        },
        sliding_window=32,
        sliding_window_pattern=2,
        use_parallel_residual=True,
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


def get_layer_and_param(cls, config, discrete=False, fused_qkv=False, use_hidden=False):
    if fused_qkv:
        layer = cls(config=config, fused_qkv=fused_qkv, dtype=jnp.float32)
    else:
        layer = cls(config=config, dtype=jnp.float32)
    if discrete:
        x = jnp.zeros((2, 10), dtype=jnp.int32)
    elif use_hidden:
        x = jnp.zeros((2, 10, config.hidden_size), dtype=jnp.float32)
    else:
        x = jnp.zeros((2, 10, config.num_heads * config.head_dim), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    params = layer.init(key, x)
    return layer, params


# ---------- Individual layers ---------- #
@pytest.fixture
def attention_factory(hf_llama_config, hf_neox_config, hf_gemma_config):
    def _fn(model_type: str, with_rope: bool):
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
        from transformers.models.llama.modeling_llama import LlamaAttention
        from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention

        match model_type:
            case "llama":
                hf = LlamaAttention(hf_llama_config, layer_idx=0)
            case "neox":
                hf = GPTNeoXAttention(hf_neox_config, layer_idx=0)
            case "gemma":
                hf = Gemma3Attention(hf_gemma_config, layer_idx=0)
            case _:
                raise

        config = ModelConfig.from_hf(hf.config)
        if with_rope:
            return hf, get_layer_and_param(AttentionWithRoPE, config, fused_qkv=model_type == "neox")
        else:
            return hf, get_layer_and_param(Attention, config, fused_qkv=model_type == "neox")

    return _fn


@pytest.fixture
def rope_factory(hf_llama_config, hf_neox_config, hf_gemma_config):
    def _fn(model_type):
        import torch
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding
        from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb as apply_neox
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as apply_llama
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

        match model_type:
            case "llama":
                hf = LlamaRotaryEmbedding(config=hf_llama_config)

                def apply_fn(query, key, cos, sin, rotary_ndims):
                    return apply_llama(query, key, cos, sin)

            case "neox":
                hf = GPTNeoXRotaryEmbedding(config=hf_neox_config)

                def apply_fn(query, key, cos, sin, rotary_ndims):
                    query_rot = query[..., :rotary_ndims]
                    query_pass = query[..., rotary_ndims:]
                    key_rot = key[..., :rotary_ndims]
                    key_pass = key[..., rotary_ndims:]
                    query, key = apply_neox(query_rot, key_rot, cos, sin)
                    query = torch.cat((query, query_pass), dim=-1)
                    key = torch.cat((key, key_pass), dim=-1)
                    return query, key

            case "gemma":
                hf = Gemma3RotaryEmbedding(config=hf_gemma_config)

                def apply_fn(query, key, cos, sin, rotary_ndims):
                    return apply_llama(query, key, cos, sin)


            case _:
                raise

        config = ModelConfig.from_hf(hf.config)
        return (
            hf,
            RotaryEmbedding(
                dim=config.head_dim,
                max_length=config.max_position_embeddings,
                base=config.rope_theta,
                rotary_pct=config.rotary_pct,
                rope_scale=config.rope_scale,
            ),
            apply_fn,
        )

    return _fn


@pytest.fixture
def cos_sin_factory(rope_factory):
    def _fn(model_type: str):
        hf, rope, _ = rope_factory(model_type)
        seq_len = 10
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (4, seq_len, hf.config.hidden_size), dtype=jnp.float32)
        p = rope.init(key, x, seq_len=seq_len)
        return rope.apply(p, x, seq_len=seq_len)
    return _fn


@pytest.fixture
def llama_mlp(config_small):
    return get_layer_and_param(LlamaMLPJAX, config_small)


@pytest.fixture
def llama_decoder(config_small):
    return get_layer_and_param(LlamaDecoder, config_small)


@pytest.fixture
def hf_llama_mlp(hf_llama_config):
    from transformers.models.llama.modeling_llama import LlamaMLP

    return LlamaMLP(hf_llama_config)


@pytest.fixture
def hf_attention_mistral(hf_mistral_config):
    from transformers.models.mistral.modeling_mistral import MistralAttention

    return MistralAttention(hf_mistral_config, layer_idx=0)


@pytest.fixture
def hf_llama_decoder(hf_llama_config):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    return LlamaDecoderLayer(hf_llama_config, layer_idx=0)


# ---------- Models ---------- #
@pytest.fixture
def llama_model(config_small):
    return get_layer_and_param(LlamaModel, config_small, discrete=True)


@pytest.fixture
def llama_model_with_head(config_small):
    return get_layer_and_param(LlamaModelWithHead, config_small, discrete=True)


@pytest.fixture
def neox_model(hf_neox_config):
    cfg = ModelConfig.from_hf(hf_neox_config)
    return get_layer_and_param(GPTNeoXModel, cfg, discrete=True)


@pytest.fixture
def neox_model_with_head(hf_neox_config):
    cfg = ModelConfig.from_hf(hf_neox_config)
    return get_layer_and_param(GPTNeoXModelWithHead, cfg, discrete=True)


@pytest.fixture
def hf_gemma_decoder(hf_gemma_config):
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

    return Gemma3DecoderLayer(hf_gemma_config, layer_idx=0)

@pytest.fixture
def hf_gemma_decoder_global(hf_gemma_config):
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

    return Gemma3DecoderLayer(hf_gemma_config, layer_idx=1)

@pytest.fixture
def gemma_decoder(hf_gemma_config):
    cfg = ModelConfig.from_hf(hf_gemma_config)
    # Gemma hidden size != num_head * head_dim
    return get_layer_and_param(GemmaDecoder, cfg, use_hidden=True)

@pytest.fixture
def gemma_model(hf_gemma_config):
    cfg = ModelConfig.from_hf(hf_gemma_config)
    return get_layer_and_param(GemmaModel, cfg, discrete=True)

@pytest.fixture
def gemma_model_with_head(hf_gemma_config):
    cfg = ModelConfig.from_hf(hf_gemma_config)
    return get_layer_and_param(GemmaModelWithHead, cfg, discrete=True)


@pytest.fixture
def hf_llama_model(hf_llama_config):
    from transformers import LlamaModel

    return LlamaModel(hf_llama_config)


@pytest.fixture
def hf_llama_causal_model(hf_llama_config):
    from transformers import LlamaForCausalLM

    return LlamaForCausalLM(hf_llama_config)


@pytest.fixture
def hf_neox_model(hf_neox_config):
    from transformers import GPTNeoXModel

    return GPTNeoXModel(hf_neox_config)


@pytest.fixture
def hf_neox_causal_model(hf_neox_config):
    from transformers import GPTNeoXForCausalLM

    return GPTNeoXForCausalLM(hf_neox_config)


@pytest.fixture
def hf_gemma_model(hf_gemma_config):
    from transformers import Gemma3TextModel

    return Gemma3TextModel(hf_gemma_config)


@pytest.fixture
def hf_gemma_causal_model(hf_gemma_config):
    from transformers import Gemma3ForCausalLM

    return Gemma3ForCausalLM(hf_gemma_config)

# ---------- Component unit test utilities ---------- #
def dummy_module_wrap(module, name: str):
    import torch

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.add_module(name, module)

        def forward(self, *args, **kwargs):
            return getattr(self, name)(*args, **kwargs)

    return Dummy()


def dummy_flax_module_wrap(module, name: str):
    import flax

    class Dummy(flax.linen.Module):
        def setup(self):
            setattr(self, name, module)

        def __call__(self, *args, **kwargs):
            return getattr(self, name)(*args, **kwargs)

    return Dummy()


@pytest.fixture
def torch_component_factory():
    def _fn(name: str):
        import torch

        match name:
            case "dense":
                return torch.nn.Linear(48, 64)
            case "rms_norm":
                return dummy_module_wrap(torch.nn.RMSNorm(48, eps=1e-5, dtype=torch.float32), name="norm")
            case "layer_norm":
                return dummy_module_wrap(torch.nn.LayerNorm(48, eps=1e-5, bias=True, dtype=torch.float32), name="norm")

    return _fn


@pytest.fixture
def jax_component_factory():
    def _fn(name: str):
        from jaxml.nn.linear import DenseGeneral
        from jaxml.nn.norms import LayerNorm, RMSNorm

        match name:
            case "dense":
                return DenseGeneral(
                    features=(64,),
                    axis=-1,
                    kernel_axes=("in", "out"),
                    dtype=jnp.float32,
                    weight_dtype=jnp.float32,
                    name="test",
                    use_bias=True,
                )
            case "rms_norm":
                return dummy_flax_module_wrap(
                    RMSNorm(
                        hidden_size=48,
                        eps=1e-5,
                        dtype=jnp.float32,
                    ),
                    "norm",
                )
            case "layer_norm":
                return dummy_flax_module_wrap(
                    LayerNorm(
                        hidden_size=48,
                        eps=1e-5,
                        use_bias=True,
                        dtype=jnp.float32,
                    ),
                    "norm",
                )

    return _fn
