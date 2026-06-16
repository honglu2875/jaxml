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

import operator
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

# from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from flax import linen as nn

from ..cache import KVCache
from ..outputs import AttentionOutput
from .linear import DenseGeneral
from .module import Block
from .position import RotaryEmbedding


def _normalize_optional_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer when set, got {type(value)}.")
    try:
        value = operator.index(value)
    except TypeError as e:
        raise TypeError(f"{name} must be an integer when set, got {type(value)}.") from e
    if value <= 0:
        raise ValueError(f"{name} must be positive when set, got {value}.")
    return value


def _normalize_bool(name: str, value: bool) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a boolean, got {type(value)}.")


def _normalize_dtype(name: str, value: Any):
    if value is None:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.")
    try:
        return jnp.dtype(value)
    except TypeError as e:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.") from e


def _contains_tracer(x: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(x))


def _validate_finite_values(name: str, values: jnp.ndarray):
    try:
        has_only_finite_values = bool(jnp.all(jnp.isfinite(values)))
    except jax.errors.TracerBoolConversionError:
        return
    if not _contains_tracer(values) and not has_only_finite_values:
        raise ValueError(f"{name} must contain only finite values.")


def _validate_attention_states(query_states: jnp.ndarray, key_states: jnp.ndarray, value_states: jnp.ndarray):
    for name, states in (("query_states", query_states), ("key_states", key_states), ("value_states", value_states)):
        if states.ndim != 4:
            raise ValueError(f"{name} must be a 4D array, got shape {states.shape}.")
        axis_names = ("batch", "sequence", "head", "head_dim")
        for axis_name, axis_size in zip(axis_names, states.shape):
            if axis_size <= 0:
                raise ValueError(f"{name} {axis_name} axis must be non-empty, got shape {states.shape}.")
        if not jnp.issubdtype(states.dtype, jnp.floating):
            raise TypeError(f"{name} must contain floating point values, got dtype {states.dtype}.")
        _validate_finite_values(name, states)

    if key_states.shape != value_states.shape:
        raise ValueError(
            f"key_states and value_states must have the same shape, got {key_states.shape} and {value_states.shape}."
        )
    if query_states.shape[0] != key_states.shape[0] or query_states.shape[2:] != key_states.shape[2:]:
        raise ValueError(
            "query_states and key_states must have matching batch, head count, and head dimension, "
            f"got {query_states.shape} and {key_states.shape}."
        )


def _validate_key_value_states(
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
    *,
    num_key_value_heads: int,
    head_dim: int,
):
    for name, states in (("key_states", key_states), ("value_states", value_states)):
        if states.ndim != 4:
            raise ValueError(f"{name} must be a 4D array, got shape {states.shape}.")
        axis_names = ("batch", "sequence", "head", "head_dim")
        for axis_name, axis_size in zip(axis_names, states.shape):
            if axis_size <= 0:
                raise ValueError(f"{name} {axis_name} axis must be non-empty, got shape {states.shape}.")
        if not jnp.issubdtype(states.dtype, jnp.floating):
            raise TypeError(f"{name} must contain floating point values, got dtype {states.dtype}.")
        _validate_finite_values(name, states)

    if key_states.shape != value_states.shape:
        raise ValueError(
            f"key_states and value_states must have the same shape, got {key_states.shape} and {value_states.shape}."
        )
    if key_states.shape[2] != num_key_value_heads:
        raise ValueError(
            "key_states and value_states must match config.num_key_value_heads, "
            f"got {key_states.shape[2]} and expected {num_key_value_heads}."
        )
    if key_states.shape[-1] != head_dim:
        raise ValueError(
            "key_states and value_states must match config.head_dim, " f"got {key_states.shape[-1]} and expected {head_dim}."
        )


def _validate_alibi_slope(alibi_slope, num_heads: int):
    if alibi_slope is None:
        return None
    alibi_slope = jnp.asarray(alibi_slope)
    if alibi_slope.ndim != 1:
        raise ValueError(f"alibi_slope must be a 1D array with one value per attention head, got shape {alibi_slope.shape}.")
    if alibi_slope.shape[0] != num_heads:
        raise ValueError(f"alibi_slope must contain {num_heads} values, got {alibi_slope.shape[0]}.")
    if not jnp.issubdtype(alibi_slope.dtype, jnp.floating):
        raise TypeError(f"alibi_slope must contain floating point values, got dtype {alibi_slope.dtype}.")
    _validate_finite_values("alibi_slope", alibi_slope)
    return alibi_slope


def _validate_optional_kv_cache(kv_cache: Optional[KVCache]) -> Optional[KVCache]:
    if kv_cache is None:
        return None
    if not isinstance(kv_cache, KVCache):
        raise TypeError(f"kv_cache must be a KVCache instance or None, got {type(kv_cache)}.")
    return kv_cache


def _validate_hidden_states(hidden_states: jnp.ndarray) -> jnp.ndarray:
    hidden_states = jnp.asarray(hidden_states)
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be a 3D array, got shape {hidden_states.shape}.")
    if any(axis_size <= 0 for axis_size in hidden_states.shape):
        raise ValueError(f"hidden_states must not contain empty axes, got shape {hidden_states.shape}.")
    if not jnp.issubdtype(hidden_states.dtype, jnp.floating):
        raise TypeError(f"hidden_states must contain floating point values, got dtype {hidden_states.dtype}.")
    _validate_finite_values("hidden_states", hidden_states)
    return hidden_states


def _reject_flash_attention_if_requested(use_flash: bool):
    if use_flash:
        raise NotImplementedError("JAX flash attention is not enabled in jaxml yet.")


def _validate_cos_sin(cos_sin):
    if cos_sin is None:
        raise ValueError("AttentionWithRoPE requires cos_sin.")
    if isinstance(cos_sin, (jnp.ndarray, np.ndarray)):
        raise TypeError("cos_sin must be a pair of cosine and sine arrays, not a single array.")
    try:
        cos, sin = cos_sin
    except TypeError as e:
        raise TypeError(f"cos_sin must be a pair of cosine and sine arrays, got {type(cos_sin)}.") from e
    except ValueError as e:
        raise ValueError("cos_sin must contain exactly two arrays: cosine and sine.") from e
    return cos, sin


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

    # Gemma3 applies RMS norms after q and k projections
    qk_norm_cls_and_args: tuple[Callable, dict] | None = None

    mm_precision: str = "high"

    def setup(self):
        dtype = _normalize_dtype("dtype", self.dtype)
        weight_dtype = _normalize_dtype("weight_dtype", self.weight_dtype)
        with_logical_partitioning = _normalize_bool("with_logical_partitioning", self.with_logical_partitioning)
        self.num_key_value_heads = self.config.num_key_value_heads

        if self.config.use_alibi:
            alibi_dtype = jnp.float32 if self.config.upcast_alibi else dtype
            self.alibi_slope = 2 ** (jnp.arange(1, self.num_heads + 1, dtype=alibi_dtype) * (-8 / self.num_heads))
        else:
            self.alibi_slope = None

        if self.fused_qkv:
            if self.num_heads != self.num_key_value_heads:
                raise ValueError(
                    "fused_qkv requires num_heads to equal num_key_value_heads, "
                    f"got {self.num_heads} and {self.num_key_value_heads}."
                )
            self.qkv_proj = DenseGeneral(
                features=(3, self.num_heads, self.head_dim),
                axis=-1,
                kernel_init=self.kernel_init,
                kernel_init_args=self.kernel_init_args,
                with_logical_partitioning=with_logical_partitioning,
                kernel_axes=("embed", "qkv", "heads", "head_states"),
                dtype=dtype,
                weight_dtype=weight_dtype,
                name="qkv_proj",
                use_bias=self.use_bias,
                precision=self.mm_precision,
            )
        else:
            self.q_proj, self.k_proj, self.v_proj = map(
                lambda x: DenseGeneral(
                    features=(x[0], self.head_dim),
                    axis=-1,
                    kernel_init=self.kernel_init,
                    kernel_init_args=self.kernel_init_args,
                    with_logical_partitioning=with_logical_partitioning,
                    kernel_axes=("embed", "heads", "head_states"),
                    dtype=dtype,
                    weight_dtype=weight_dtype,
                    name=x[1],
                    use_bias=self.use_bias,
                    precision=self.mm_precision,
                ),
                (
                    (self.num_heads, "q_proj"),
                    (self.num_key_value_heads, "k_proj"),
                    (self.num_key_value_heads, "v_proj"),
                ),
            )
        if self.qk_norm_cls_and_args is not None:
            _cls, _args = self.qk_norm_cls_and_args
            self.q_norm, self.k_norm = _cls(**_args), _cls(**_args)

        self.o_proj = DenseGeneral(
            features=self.hidden_size,
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=with_logical_partitioning,
            kernel_axes=("heads_merged", "embed"),
            name="o_proj",
            use_bias=self.use_bias,
            precision=self.mm_precision,
        )

    def qkv(self, hidden: jnp.ndarray):
        if self.fused_qkv:
            out = self.qkv_proj(hidden)
            query, key, value = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        else:
            query, key, value = (
                self.q_proj(hidden),
                self.k_proj(hidden),
                self.v_proj(hidden),
            )

        if self.qk_norm_cls_and_args is not None:
            query, key = (self.q_norm(query), self.k_norm(key))

        return query, key, value

    def apply_kv_cache(
        self,
        key_states: jnp.ndarray,
        value_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        kv_cache: Optional[KVCache],
    ):
        kv_cache = _validate_optional_kv_cache(kv_cache)
        if kv_cache is None:
            return key_states, value_states, attention_mask, None

        kv_cache = kv_cache.update(key_states, value_states, attention_mask)
        return kv_cache.k, kv_cache.v, kv_cache.mask, kv_cache

    def repeat_kv(self, key_states: jnp.ndarray, value_states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        key_states = jnp.asarray(key_states)
        value_states = jnp.asarray(value_states)
        _validate_key_value_states(
            key_states,
            value_states,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
        )
        batch, seq_len, num_key_value_heads, head_dim = key_states.shape
        n_rep = self.config.num_heads // self.config.num_key_value_heads

        def _repeat(hidden_states: jnp.ndarray):
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

        return tuple(_repeat(x) for x in (key_states, value_states))

    def mha(
        self,
        query_states: jnp.ndarray,
        key_states: jnp.ndarray,
        value_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        causal: bool = True,
        sliding_window: int | None = None,
        alibi_slope=None,
        softmax_fp32: bool = True,
        output_attentions: bool = False,
    ):
        query_states = jnp.asarray(query_states)
        key_states = jnp.asarray(key_states)
        value_states = jnp.asarray(value_states)
        _validate_attention_states(query_states, key_states, value_states)
        sliding_window = _normalize_optional_count("sliding_window", sliding_window)
        causal = _normalize_bool("causal", causal)
        softmax_fp32 = _normalize_bool("softmax_fp32", softmax_fp32)
        output_attentions = _normalize_bool("output_attentions", output_attentions)
        alibi_slope = _validate_alibi_slope(alibi_slope, query_states.shape[2])
        x = jnp.einsum("bshn,bthn->bhst", query_states, key_states, precision=self.mm_precision) * self.attn_scale

        _, _, q_len, k_len = x.shape
        dtype = query_states.dtype

        if attention_mask is not None:
            attention_mask = jnp.asarray(attention_mask)
            if attention_mask.shape != (query_states.shape[0], k_len):
                raise ValueError(
                    "attention_mask shape must match attention batch and key length, "
                    f"got {attention_mask.shape} and {(query_states.shape[0], k_len)}."
                )
            if not (jnp.issubdtype(attention_mask.dtype, jnp.bool_) or jnp.issubdtype(attention_mask.dtype, jnp.integer)):
                raise TypeError(f"attention_mask must be boolean or integer, got dtype {attention_mask.dtype}.")
            if jnp.issubdtype(attention_mask.dtype, jnp.integer) and not _contains_tracer(attention_mask):
                if bool(jnp.any((attention_mask != 0) & (attention_mask != 1))):
                    raise ValueError("attention_mask integer values must be 0 or 1.")
            attention_mask = attention_mask.astype(bool)

        if alibi_slope is not None:
            # bias: (head_dim, 1, k_len)
            bias = -(jnp.arange(k_len, dtype=dtype)[None, None] * alibi_slope[:, None, None])
            x += bias

        if causal and q_len != 1:
            # q_len == 1 iff decoding => mask is redundant
            x += jnp.triu(jnp.full((q_len, k_len), float("-inf"), dtype=dtype), k=1)

        if sliding_window is not None:
            if q_len != 1:
                x += jnp.tril(jnp.full((q_len, k_len), float("-inf"), dtype=dtype), k=-sliding_window)
            elif attention_mask is not None:
                positions = jnp.arange(k_len)[None, :]
                current_pos = jnp.max(jnp.where(attention_mask, positions, -1), axis=-1, keepdims=True)
                window_mask = positions >= current_pos - sliding_window + 1
                x += jnp.where(window_mask[:, None, None, :], 0, float("-inf"))
            else:
                # Decode with an unpadded cache: keep only the last window.
                x = x.at[:, :, :, :-sliding_window].set(float("-inf"))

        if attention_mask is not None:
            x += jnp.where(attention_mask[:, None, None, :], 0, float("-inf"))

        all_masked = jnp.all(jnp.isneginf(x), axis=-1, keepdims=True)
        x = jnp.where(all_masked, 0, x)

        if softmax_fp32:
            attn_weight = jax.nn.softmax(x.astype(jnp.float32), axis=-1).astype(dtype)
        else:
            attn_weight = jax.nn.softmax(x, axis=-1)

        if output_attentions:
            out_weight = attn_weight
        else:
            out_weight = None

        attn_output = jnp.einsum("bhst,bthn->bshn", attn_weight, value_states, precision=self.mm_precision)

        return attn_output, out_weight

    def mha_flash(
        self,
        query_states: jnp.ndarray,
        key_states: jnp.ndarray,
        value_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        causal: bool = True,
        sliding_window: int | None = None,
        alibi_slope=None,
        softmax_fp32: bool = True,
        output_attentions: bool = False,
    ):
        """
        if output_attentions:
            raise NotImplementedError("Output attention scores for TPU attention is not implemented.")
        if alibi_slope is not None:
            raise NotImplementedError("The current JAX FlashAttention materializes the whole alibi-bias, defeating the purpose of using it for inference.")
        bs, q_len, num_heads, head_dim = query_states.shape
        kv_len = key_states.shape[1]
        if q_len < 128:
            # Pad to minimal block size
            query_states = jnp.concatenate([query_states, jnp.zeros((bs, 128 - q_len, num_heads, head_dim), dtype=query_states.dtype)], axis=1)
        if kv_len % 128 != 0:
            key_states = jnp.concatenate([key_states, jnp.zeros((bs, 128 - kv_len % 128, num_heads, head_dim), dtype=query_states.dtype)], axis=1)
            value_states = jnp.concatenate([value_states, jnp.zeros((bs, 128 - kv_len % 128, num_heads, head_dim), dtype=query_states.dtype)], axis=1)

        query_states, key_states, value_states = map(lambda x: x.transpose(0, 2, 1, 3), (query_states, key_states, value_states))
        output = flash_attention(query_states, key_states, value_states, causal=causal, sm_scale=1 / self.qk_scale)

        return output.transpose(0, 2, 1, 3)[:, :q_len], None
        """
        raise NotImplementedError("JAX flash attention has a bug... Waiting to be fixed.")

    def post_mha(
        self,
        attn_output: jnp.ndarray,
    ) -> jnp.ndarray:
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _get_default_mask(self, hidden_states, kv_cache):
        kv_cache = _validate_optional_kv_cache(kv_cache)
        if kv_cache is not None and kv_cache.mask is not None:
            return kv_cache.next_mask
        return jnp.ones(hidden_states.shape[:2], dtype=bool)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_flash: bool = False,
        sliding_window: int | None = None,
        **kwargs,
    ) -> AttentionOutput:
        """The base class implements basic MHA **without** positional encoding such as RoPE."""
        hidden_states = _validate_hidden_states(hidden_states)
        output_attentions = _normalize_bool("output_attentions", output_attentions)
        use_flash = _normalize_bool("use_flash", use_flash)
        _reject_flash_attention_if_requested(use_flash)
        if position_ids is not None:
            raise NotImplementedError("MHA with given position_ids is not implemented.")

        query_states, key_states, value_states = self.qkv(hidden_states)
        key_states, value_states, attention_mask, kv_cache = self.apply_kv_cache(
            key_states, value_states, attention_mask, kv_cache
        )
        key_states, value_states = self.repeat_kv(key_states, value_states)

        mha_fn = self.mha_flash if use_flash else self.mha
        attn_output, attn_weight = mha_fn(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            causal=True,
            sliding_window=sliding_window,
            alibi_slope=self.alibi_slope,
            softmax_fp32=True,
            output_attentions=output_attentions,
        )

        return AttentionOutput(
            attention_output=self.post_mha(attn_output),
            attention_weight=attn_weight,
            kv_cache=kv_cache,
        )


class AttentionWithRoPE(Attention):

    def setup(self):
        super().setup()
        if self.config.use_alibi:
            raise NotImplementedError("Having ALiBi and RoPE together is not intended (though the math works).")

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos_sin: Optional[tuple[jnp.ndarray, jnp.ndarray]] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_flash: bool = False,
        sliding_window: int | None = None,
    ) -> AttentionOutput:
        hidden_states = _validate_hidden_states(hidden_states)
        output_attentions = _normalize_bool("output_attentions", output_attentions)
        use_flash = _normalize_bool("use_flash", use_flash)
        _reject_flash_attention_if_requested(use_flash)
        cos, sin = _validate_cos_sin(cos_sin)
        query_states, key_states, value_states = self.qkv(hidden_states)

        if attention_mask is None:
            attention_mask = self._get_default_mask(hidden_states, kv_cache)

        if position_ids is None:
            if kv_cache is not None and kv_cache.pos_id is not None:
                position_ids = kv_cache.next_pos_id
            else:
                position_ids = jnp.repeat(jnp.arange(key_states.shape[1])[None], key_states.shape[0], axis=0)

        query_states, key_states = RotaryEmbedding.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )
        key_states, value_states, attention_mask, kv_cache = self.apply_kv_cache(
            key_states, value_states, attention_mask, kv_cache
        )
        key_states, value_states = self.repeat_kv(key_states, value_states)

        mha_fn = self.mha_flash if use_flash else self.mha
        attn_output, attn_weight = mha_fn(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            causal=True,
            sliding_window=sliding_window,
            alibi_slope=None,
            softmax_fp32=True,
            output_attentions=output_attentions,
        )

        return AttentionOutput(
            attention_output=self.post_mha(attn_output),
            attention_weight=attn_weight,
            kv_cache=kv_cache,
        )
