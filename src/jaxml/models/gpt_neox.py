# coding=utf-8
# Copyright 2024 Honglu Fan (https://github.com/honglu2875).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.partitioning import with_sharding_constraint

from ..cache import KVCache
from ..nn.attention import Attention, AttentionWithRoPE
from ..nn.embedding import Embed
from ..nn.linear import DenseGeneral
from ..nn.module import Block
from ..nn.norms import LayerNorm
from ..nn.position import RotaryEmbedding
from ..outputs import AttentionOutput, BaseModelOutputWithCache, CausalLMOutputWithCache, DecoderOutput


class GPTNeoXMLP(Block):
    kernel_init: Any = nn.initializers.xavier_uniform
    act_fn: Any = functools.partial(jax.nn.gelu, approximate=False)

    def setup(self):
        if self.config is None:
            raise ValueError("Must provide a config for MLP.")
        # input dim supposed to be self.hidden_size
        self.up_proj = DenseGeneral(
            features=self.intermediate_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("embed", "intermediate"),
            name="up_proj",
            use_bias=self.use_bias,
            precision="high",
        )
        self.down_proj = DenseGeneral(
            features=self.hidden_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("intermediate", "embed"),
            name="down_proj",
            use_bias=self.use_bias,
            precision="high",
        )

    def __call__(self, x, **kwargs):
        assert (
            x.shape[-1] == self.hidden_size
        ), f"Input to MLP layers have different dimensions than the hidden dimension. Got {x.shape[-1]}"
        x = with_sharding_constraint(x, ("batch", "length", "embed"))
        x = self.act_fn(self.up_proj(x))
        x = self.down_proj(x)
        return x


class GPTNeoXDecoder(Block):
    def setup(self):
        if not self.config.use_rope:
            self.self_attn = Attention(self.config, fused_qkv=True, dtype=self.dtype, mm_precision="high")
        else:
            self.self_attn = AttentionWithRoPE(self.config, fused_qkv=True, dtype=self.dtype, mm_precision="high")
        self.mlp = GPTNeoXMLP(self.config, dtype=self.dtype)
        self.input_layernorm = LayerNorm(
            hidden_size=self.hidden_size,
            eps=self.norm_eps,
            dtype=self.dtype,
            upcast=True,
            use_bias=True,
        )
        self.post_attention_layernorm = LayerNorm(
            hidden_size=self.hidden_size,
            eps=self.norm_eps,
            dtype=self.dtype,
            upcast=True,
            use_bias=True,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos_sin: Optional[tuple[jnp.ndarray, jnp.ndarray]] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> DecoderOutput:
        attn_input = self.input_layernorm(hidden_states)
        attn_output: AttentionOutput = self.self_attn(
            hidden_states=attn_input,
            cos_sin=cos_sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            output_attentions=output_attentions,
        )

        if self.config.use_parallel_residual:
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output.attention_output + hidden_states
        else:
            attn_output_with_res = attn_output.attention_output + hidden_states
            mlp_input = self.post_attention_layernorm(attn_output_with_res)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output_with_res

        return DecoderOutput(
            hidden_states=hidden_states,
            kv_cache=attn_output.kv_cache,
            attention_weight=attn_output.attention_weight,
        )


class GPTNeoXModel(Block):
    def setup(self):
        self.embed_tokens = Embed(num_embeddings=self.config.vocab_size, features=self.hidden_size, dtype=self.dtype)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_length=self.config.max_position_embeddings,
            base=self.config.rope_theta,
            rotary_pct=self.config.rotary_pct,
            rope_scale=self.config.rope_scale,
        )
        self.layers = [GPTNeoXDecoder(self.config, dtype=self.dtype) for _ in range(self.num_layers)]
        self.norm = LayerNorm(self.hidden_size, eps=self.norm_eps, use_bias=True)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[tuple[KVCache, ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> BaseModelOutputWithCache:
        batch_size, seq_length = input_ids.shape

        if attention_mask is None:
            # need to apply a default value if kv_cache is either unused or empty
            if kv_caches is None or kv_caches[0].mask is None:
                # our convention is that negative ids (such as -100) is masked by default.
                attention_mask = input_ids >= 0

        inputs_embeds = self.embed_tokens(input_ids).astype(self.dtype)
        hidden_states = with_sharding_constraint(inputs_embeds, ("batch", "length", "embed"))

        all_hidden_states = []
        all_self_attns = []
        next_kv_caches = []

        k_len = None if kv_caches is None or kv_caches[0].k is None else kv_caches[0].k.shape[1]
        cos_sin = self.rotary_emb(hidden_states, seq_len=k_len)

        for idx, decoder_layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)

            kv_cache = None if kv_caches is None else kv_caches[idx]

            output = decoder_layer(
                hidden_states,
                cos_sin=cos_sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
            )
            hidden_states, kv_cache, attn_weight = (output.hidden_states, output.kv_cache, output.attention_weight)

            all_self_attns.append(attn_weight)
            next_kv_caches.append(kv_cache)

        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        return BaseModelOutputWithCache(
            last_hidden_state=hidden_states,
            kv_caches=tuple(next_kv_caches) if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attention_weights=tuple(all_self_attns) if output_attentions else None,
        )


class GPTNeoXModelWithHead(Block):

    lm_head_init: Any = nn.initializers.xavier_uniform
    lm_head_init_args: tuple = ()

    def setup(self):
        self.gpt_neox = GPTNeoXModel(self.config, dtype=self.dtype)
        self.lm_head = DenseGeneral(
            features=self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=self.lm_head_init,
            kernel_init_args=self.lm_head_init_args,
            with_logical_partitioning=True,
            kernel_axes=("embed", "vocab"),
            name="lm_head",
            use_bias=False,  # lm_head almost certain does not use bias
            precision="high",
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[tuple[KVCache, ...]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        keep_last_n_logits: int = 0,  # 0: keep all logits
    ) -> CausalLMOutputWithCache:
        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_caches=kv_caches,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # logits.shape: (bs, keep_last_n_logits, vocab_size)
        logits = self.lm_head(outputs.last_hidden_state[:, -keep_last_n_logits:])
        return CausalLMOutputWithCache(
            logits=logits,
            kv_caches=outputs.kv_caches,
            hidden_states=outputs.hidden_states,
            attention_weights=outputs.attention_weights,
        )
