#  Copyright 2024 Honglu Fan
#  This file is based on code by the authors denoted below and has been modified from its original version.
#
#  Copyright 2023 Google LLC
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
# This file has been modified from its original version
# Link: https://github.com/google/maxtext/blob/4f3a0d3cf8509d05ce040e35d88ea7bf57797945/MaxText/layers/attentions.py

from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..cache import KVCache
from ..outputs import AttentionOutput
from ..types import Array
from ..utils import get_default_pos_ids
from .linear import DenseGeneral
from .module import Block
from .position import RotaryEmbedding


class Attention(Block):
    """
    Flax base model of attention.
    """

    kernel_init: Any = nn.initializers.xavier_uniform
    kernel_init_args: tuple = ()
    with_logical_partitioning: bool = True
    weight_dtype: Any = jnp.float32
    fused_qkv: bool = False
    use_alibi: bool = False

    def setup(self):
        self.num_key_value_heads = self.config.num_key_value_heads

        if self.fused_qkv:
            assert self.num_heads == self.num_key_value_heads
            self.qkv_proj = DenseGeneral(
                features=(3, self.num_heads, self.head_dim),
                axis=-1,
                kernel_init=self.kernel_init,
                kernel_init_args=self.kernel_init_args,
                with_logical_partitioning=self.with_logical_partitioning,
                kernel_axes=("embed", "qkv", "heads", "joined_kv"),
                dtype=self.dtype,
                weight_dtype=self.weight_dtype,
                name="qkv_proj",
            )
        else:
            self.q_proj, self.k_proj, self.v_proj = map(
                lambda x: DenseGeneral(
                    features=(x[0], self.head_dim),
                    axis=-1,
                    kernel_init=self.kernel_init,
                    kernel_init_args=self.kernel_init_args,
                    with_logical_partitioning=self.with_logical_partitioning,
                    kernel_axes=("embed", "heads", "joined_kv"),
                    dtype=self.dtype,
                    weight_dtype=self.weight_dtype,
                    name=x[1],
                ),
                (
                    (self.num_heads, "q_proj"),
                    (self.num_key_value_heads, "k_proj"),
                    (self.num_key_value_heads, "v_proj"),
                ),
            )
        self.o_proj = DenseGeneral(
            features=self.head_dim * self.num_heads,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("joined_kv", "embed"),
            name="o_proj",
        )

    def qkv_proj(self, hidden: Array):
        if self.fused_qkv:
            out = self.qkv_proj(hidden)
            query, key, value = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        else:
            query, key, value = (
                self.q_proj(hidden),
                self.k_proj(hidden),
                self.v_proj(hidden),
            )

        return query, key, value

    def apply_kv_cache(self, key_states: Array, value_states: Array, kv_cache: Optional[KVCache]):
        if kv_cache is None:
            return key_states, value_states, None

        kv_cache = kv_cache.update(key_states, value_states)
        key_states, value_states = kv_cache.get_kv()
        return key_states, value_states, kv_cache

    def repeat_kv(self, key_states: Array, value_states: Array):
        batch, seq_len, num_key_value_heads, head_dim = key_states.shape
        n_rep = self.config.num_heads // self.config.num_key_value_heads

        def _repeat(hidden_states: Array):
            """
            This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
            seq_len, num_key_value_heads, head_dim) to (batch, seq_len, num_attention_heads, head_dim)
            """
            if n_rep == 1:
                return hidden_states
            hidden_states = jnp.broadcast_to(
                hidden_states[:, :, :, None, :],
                (batch, seq_len, num_key_value_heads, n_rep, head_dim),
            )
            return hidden_states.reshape(batch, seq_len, num_key_value_heads * n_rep, head_dim)

        return tuple(map(_repeat, (key_states, value_states)))

    @property
    def qk_scale(self):
        """The scale applied after qk in MHA. Feel free to override in cases such as muP."""
        return jnp.sqrt(self.head_dim)

    def mha(
        self,
        query_states: Array,
        key_states: Array,
        value_states: Array,
        attention_mask: Optional[Array] = None,
        causal: bool = True,
        alibi_slope=None,
        softmax_fp32: bool = True,
        output_attentions: bool = False,
    ):
        x = jnp.einsum("bshn,bthn->bhst", query_states, key_states) / self.qk_scale

        _, _, q_len, k_len = x.shape
        dtype = query_states.dtype

        if alibi_slope is not None:
            # bias: (head_dim, 1, k_len)
            bias = -(jnp.arange(k_len, dtype=dtype)[None, None] * alibi_slope[:, None, None])
            x += bias

        if causal:
            x += jnp.triu(jnp.full((q_len, k_len), float("-inf"), dtype=dtype), k=1)

        if attention_mask is not None:
            x += jnp.where(attention_mask[:, None, None, :], 0, float("-inf"))

        if softmax_fp32:
            attn_weight = jax.nn.softmax(x.astype(jnp.float32), axis=-1).astype(dtype)
        else:
            attn_weight = jax.nn.softmax(x, axis=-1)

        if output_attentions:
            out_weight = attn_weight
        else:
            out_weight = None

        attn_output = jnp.einsum("bhst,bthn->bshn", attn_weight, value_states)

        return attn_output, out_weight

    def post_mha(
        self,
        attn_output: jnp.ndarray,
    ) -> jnp.ndarray:
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))
        attn_output = self.o_proj(attn_output)
        return attn_output

    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Optional[Array] = None,
        position_ids: Optional[Array] = None,
        kv_cache: Optional[KVCache] = None,
        use_alibi: bool = False,
        output_attentions: bool = False,
        *kwargs,
    ) -> AttentionOutput:
        """The base class implements basic MHA **without** positional encoding such as RoPE."""
        if position_ids is not None:
            raise NotImplementedError("MHA with given position_ids is not implemented.")

        query_states, key_states, value_states = self.qkv_proj(hidden_states)
        key_states, value_states, kv_cache = self.apply_kv_cache(key_states, value_states, kv_cache)
        key_states, value_states = self.repeat_kv(key_states, value_states)

        if use_alibi:
            dtype = jnp.float32 if self.config.upcast_alibi else self.dtype
            alibi_slope = 2 ** (jnp.arange(1, self.num_heads + 1, dtype=dtype) * (-8 / self.num_heads))
        else:
            alibi_slope = None

        attn_output, attn_weight = self.mha(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            causal=True,
            alibi_slope=alibi_slope,
            softmax_fp32=True,
            output_attentions=output_attentions,
        )

        return AttentionOutput(
            attention_output=self.post_mha(attn_output),
            attention_weight=attn_weight,
            kv_cache=kv_cache,
        )


class AttentionWithRoPE(Attention):

    max_position_embeddings: int = 2048
    rope_theta: int = 10_000
    use_alibi: bool = False

    def setup(self):
        if self.use_alibi:
            raise NotImplementedError("Having ALiBi and RoPE together is not intended (though the math works).")
        super().setup()
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_length=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Optional[Array] = None,
        position_ids: Optional[Array] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> AttentionOutput:
        query_states, key_states, value_states = self.qkv_proj(hidden_states)
        key_states, value_states, kv_cache = self.apply_kv_cache(key_states, value_states, kv_cache)

        key_len = key_states.shape[1]
        bs = hidden_states.shape[0]
        if position_ids is None:
            # infer the position ids according to mask (ignore padding)
            position_ids = get_default_pos_ids((bs, key_len), mask=attention_mask)
        cos, sin = self.rotary_emb(key_states, seq_len=key_len)
        # Could save some FLOP at inference if moving it before kv cache (key_len=1 back then)
        # but dynamically reading the kv_cache length might let JAX complain about abstract shapes
        # especially when inside a scan loop.
        query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        key_states, value_states = self.repeat_kv(key_states, value_states)

        attn_output, attn_weight = self.mha(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            causal=True,
            alibi_slope=None,
            softmax_fp32=True,
            output_attentions=output_attentions,
        )

        return AttentionOutput(
            attention_output=self.post_mha(attn_output),
            attention_weight=attn_weight,
            kv_cache=kv_cache,
        )
