# coding=utf-8
# Copyright 2026 Honglu Fan (https://github.com/honglu2875).
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
"""Text-only Qwen3.5/Qwen3.6 MoE inference model.

The architecture alternates fixed-state Gated DeltaNet token mixers with
ordinary gated full-attention layers.  The recurrent reference path deliberately
uses :func:`jax.lax.scan`: it is exact and keeps decode state constant-sized,
while leaving a clean replacement point for a future chunkwise TPU prefill
kernel.
"""

import operator
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct
from flax.linen.partitioning import with_sharding_constraint

from .._validation import contains_tracer as _contains_tracer
from .._validation import validate_finite_values as _validate_finite_values
from ..cache import KVCache
from ..nn.embedding import Embed
from ..nn.linear import DenseGeneral
from ..nn.module import Block
from ..nn.position import RotaryEmbedding
from ..outputs import AttentionOutput, BaseModelOutputWithCache, CausalLMOutputWithCache, DecoderOutput
from ..utils import get_default_pos_ids
from ._utils import (
    normalize_model_output_flags,
    prepare_default_attention_mask,
    prepare_model_inputs,
    prepare_position_ids,
    slice_last_n_logits_hidden_states,
)

_GMM_TILE = 128


def _normalize_count(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {type(value)}.")
    try:
        return operator.index(value)
    except TypeError as e:
        raise TypeError(f"{name} must be an integer, got {type(value)}.") from e


def _normalize_dtype(name: str, value: Any):
    if value is None:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.")
    try:
        return jax.dtypes.canonicalize_dtype(jnp.dtype(value))
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.") from e


def _binary_mask(name: str, value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {value.shape}.")
    if not (jnp.issubdtype(value.dtype, jnp.bool_) or jnp.issubdtype(value.dtype, jnp.integer)):
        raise TypeError(f"{name} must be boolean or integer, got dtype {value.dtype}.")
    if jnp.issubdtype(value.dtype, jnp.integer) and not _contains_tracer(value):
        if bool(jnp.any((value != 0) & (value != 1))):
            raise ValueError(f"{name} integer values must be 0 or 1.")
    return value.astype(bool)


def _validate_right_padding(name: str, mask: jnp.ndarray):
    if not _contains_tracer(mask) and mask.shape[1] > 1 and bool(jnp.any(mask[:, :-1] < mask[:, 1:])):
        raise ValueError(f"{name} must be right padded.")


def _rotary_cos_sin(
    seq_len: int,
    *,
    head_dim: int,
    base: float,
    rotary_pct: float,
    rope_scale: float,
    dtype: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build the non-persistent text RoPE table used by Qwen full attention."""
    rotary_dim = head_dim if rotary_pct >= 1.0 else int(head_dim * rotary_pct)
    inv_freq = 1.0 / (base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32) / rope_scale
    frequencies = jnp.einsum("s,d->sd", positions, inv_freq, precision="high")
    embeddings = jnp.concatenate((frequencies, frequencies), axis=-1)
    return jnp.cos(embeddings).astype(dtype), jnp.sin(embeddings).astype(dtype)


class Qwen35DeltaNetCache(struct.PyTreeNode):
    """Per-layer convolution and recurrent state for Qwen Gated DeltaNet."""

    conv_state: jnp.ndarray | None = struct.field(pytree_node=True)
    recurrent_state: jnp.ndarray | None = struct.field(pytree_node=True)
    max_seq_len: int = struct.field(pytree_node=False)
    conv_dim: int = struct.field(pytree_node=False)
    num_heads: int = struct.field(pytree_node=False)
    key_head_dim: int = struct.field(pytree_node=False)
    value_head_dim: int = struct.field(pytree_node=False)
    conv_kernel_size: int = struct.field(pytree_node=False)
    mask: jnp.ndarray | None = struct.field(pytree_node=True)
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)
    state_dtype: Any = struct.field(pytree_node=False, default=jnp.float32)
    pos_id: jnp.ndarray | None = struct.field(default=None, pytree_node=True)

    @classmethod
    def init(
        cls,
        max_seq_len: int,
        *,
        conv_dim: int,
        num_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        dtype: Any = jnp.float32,
        state_dtype: Any = jnp.float32,
    ) -> "Qwen35DeltaNetCache":
        values = {
            name: _normalize_count(name, value)
            for name, value in (
                ("max_seq_len", max_seq_len),
                ("conv_dim", conv_dim),
                ("num_heads", num_heads),
                ("key_head_dim", key_head_dim),
                ("value_head_dim", value_head_dim),
                ("conv_kernel_size", conv_kernel_size),
            )
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"DeltaNet cache dimensions must be positive, got {values}.")
        dtype = _normalize_dtype("dtype", dtype)
        state_dtype = _normalize_dtype("state_dtype", state_dtype)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError(f"DeltaNet convolution cache dtype must be floating point, got {dtype}.")
        if state_dtype != jnp.dtype(jnp.float32):
            raise TypeError(f"DeltaNet recurrent state must use float32, got {state_dtype}.")
        return cls(
            conv_state=None,
            recurrent_state=None,
            mask=None,
            dtype=dtype,
            state_dtype=state_dtype,
            pos_id=None,
            **values,
        )

    @property
    def is_populated(self) -> bool:
        return self.recurrent_state is not None

    @property
    def batch_size(self) -> int | None:
        return None if self.recurrent_state is None else self.recurrent_state.shape[0]

    @property
    def next_pos_id(self) -> jnp.ndarray:
        if self.pos_id is None:
            raise ValueError("Cannot get next position ids from an empty DeltaNet cache.")
        return self.pos_id + 1

    @property
    def next_mask(self) -> jnp.ndarray | None:
        return self.mask

    def _validate_static_state(self):
        for name, value in (
            ("max_seq_len", self.max_seq_len),
            ("conv_dim", self.conv_dim),
            ("num_heads", self.num_heads),
            ("key_head_dim", self.key_head_dim),
            ("value_head_dim", self.value_head_dim),
            ("conv_kernel_size", self.conv_kernel_size),
        ):
            value = _normalize_count(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        dtype = _normalize_dtype("dtype", self.dtype)
        state_dtype = _normalize_dtype("state_dtype", self.state_dtype)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError(f"DeltaNet convolution cache dtype must be floating point, got {dtype}.")
        if state_dtype != jnp.dtype(jnp.float32):
            raise TypeError(f"DeltaNet recurrent state dtype must be float32, got {state_dtype}.")
        return dtype, state_dtype

    def validate_state(self) -> "Qwen35DeltaNetCache":
        dtype, state_dtype = self._validate_static_state()
        if self.conv_state is None and self.recurrent_state is None:
            if self.mask is not None or self.pos_id is not None:
                raise ValueError("Empty DeltaNet cache cannot contain position metadata.")
            return self
        if self.conv_state is None or self.recurrent_state is None or self.mask is None or self.pos_id is None:
            raise ValueError("DeltaNet cache has partial state.")
        batch = self.recurrent_state.shape[0]
        conv_shape = (batch, self.conv_dim, self.conv_kernel_size)
        recurrent_shape = (batch, self.num_heads, self.key_head_dim, self.value_head_dim)
        if self.conv_state.shape != conv_shape:
            raise ValueError(f"conv_state must have shape {conv_shape}, got {self.conv_state.shape}.")
        if self.recurrent_state.shape != recurrent_shape:
            raise ValueError(f"recurrent_state must have shape {recurrent_shape}, got {self.recurrent_state.shape}.")
        if self.conv_state.dtype != dtype:
            raise TypeError(f"conv_state dtype must match cache dtype {dtype}, got {self.conv_state.dtype}.")
        if self.recurrent_state.dtype != state_dtype:
            raise TypeError(f"recurrent_state dtype must match state_dtype {state_dtype}, got {self.recurrent_state.dtype}.")
        _validate_finite_values("conv_state", self.conv_state)
        _validate_finite_values("recurrent_state", self.recurrent_state)

        if self.mask.shape != (batch, self.max_seq_len):
            raise ValueError(f"DeltaNet cache mask must have shape {(batch, self.max_seq_len)}, got {self.mask.shape}.")
        if not jnp.issubdtype(self.mask.dtype, jnp.bool_):
            raise TypeError(f"DeltaNet cache mask must be boolean, got dtype {self.mask.dtype}.")
        if self.pos_id.shape != (batch, 1):
            raise ValueError(f"DeltaNet cache pos_id must have shape {(batch, 1)}, got {self.pos_id.shape}.")
        if not jnp.issubdtype(self.pos_id.dtype, jnp.integer):
            raise TypeError(f"DeltaNet cache pos_id must contain integers, got dtype {self.pos_id.dtype}.")
        _validate_right_padding("DeltaNet cache mask", self.mask)
        if not _contains_tracer((self.mask, self.pos_id)):
            if not bool(jnp.all(jnp.any(self.mask, axis=1))):
                raise ValueError("DeltaNet cache mask must contain at least one valid token per batch row.")
            if bool(jnp.any((self.pos_id < 0) | (self.pos_id >= self.max_seq_len))):
                raise ValueError(f"DeltaNet cache pos_id values must be within [0, {self.max_seq_len}).")
            if not bool(jnp.array_equal(self.pos_id, get_default_pos_ids(self.mask))):
                raise ValueError("DeltaNet cache pos_id must identify the last valid position in its mask.")
        return self

    def update(
        self,
        conv_state: jnp.ndarray,
        recurrent_state: jnp.ndarray,
        token_mask: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
    ) -> "Qwen35DeltaNetCache":
        self._validate_static_state()
        token_mask = _binary_mask("token_mask", token_mask)
        conv_state = jnp.asarray(conv_state, dtype=self.dtype)
        recurrent_state = jnp.asarray(recurrent_state, dtype=self.state_dtype)
        batch, seq_len = token_mask.shape
        if conv_state.shape != (batch, self.conv_dim, self.conv_kernel_size):
            raise ValueError("DeltaNet convolution state shape does not match this cache.")
        if recurrent_state.shape != (batch, self.num_heads, self.key_head_dim, self.value_head_dim):
            raise ValueError("DeltaNet recurrent state shape does not match this cache.")
        _validate_finite_values("conv_state", conv_state)
        _validate_finite_values("recurrent_state", recurrent_state)

        if not self.is_populated:
            if seq_len > self.max_seq_len:
                raise ValueError(f"Cannot cache {seq_len} tokens in max_seq_len={self.max_seq_len}.")
            if not _contains_tracer(token_mask) and not bool(jnp.all(jnp.any(token_mask, axis=1))):
                raise ValueError("token_mask must contain at least one valid token per batch row.")
            _validate_right_padding("token_mask", token_mask)
            mask = jnp.pad(token_mask, ((0, 0), (0, self.max_seq_len - seq_len)), constant_values=False)
            return self.replace(
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                mask=mask,
                pos_id=get_default_pos_ids(mask),
            )

        self.validate_state()
        if batch != self.batch_size or seq_len != 1:
            raise ValueError("Populated DeltaNet caches require one token with the cached batch size.")
        if attention_mask is not None:
            attention_mask = _binary_mask("attention_mask", attention_mask)
            if attention_mask.shape != self.mask.shape:
                raise ValueError("Decode attention_mask must match the cached mask shape.")
            if not _contains_tracer((attention_mask, self.mask)) and not bool(jnp.array_equal(attention_mask, self.mask)):
                raise ValueError("Decode attention_mask must describe the current cache before its update.")
        if not _contains_tracer(token_mask) and not bool(jnp.all(token_mask)):
            raise ValueError("Every decode row must contain a valid token.")
        if not _contains_tracer(self.next_pos_id) and bool(jnp.any(self.next_pos_id >= self.max_seq_len)):
            raise ValueError(f"Cannot decode past max_seq_len={self.max_seq_len}.")
        rows = jnp.arange(batch)[:, None]
        indices = jnp.concatenate((rows, self.next_pos_id), axis=1)
        return self.replace(
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            mask=self.mask.at[tuple(indices.T)].set(True),
            pos_id=self.next_pos_id,
        )

    def resize(self, new_size: int) -> "Qwen35DeltaNetCache":
        new_size = _normalize_count("new_size", new_size)
        if new_size <= 0:
            raise ValueError(f"new_size must be positive, got {new_size}.")
        self.validate_state()
        if not self.is_populated:
            return self.replace(max_seq_len=new_size)
        if not _contains_tracer(self.pos_id) and bool(jnp.any(self.pos_id >= new_size)):
            raise ValueError("Cannot resize below a populated cache position.")
        if new_size > self.max_seq_len:
            mask = jnp.pad(self.mask, ((0, 0), (0, new_size - self.max_seq_len)), constant_values=False)
        else:
            mask = self.mask[:, :new_size]
        return self.replace(max_seq_len=new_size, mask=mask)

    def reorder_cache(self, beam_idx: jnp.ndarray) -> "Qwen35DeltaNetCache":
        self.validate_state()
        if not self.is_populated:
            return self
        beam_idx = jnp.asarray(beam_idx)
        if beam_idx.ndim != 1 or not jnp.issubdtype(beam_idx.dtype, jnp.integer):
            raise TypeError("beam_idx must be a 1D integer array.")
        if beam_idx.size == 0:
            raise ValueError("beam_idx must contain at least one index.")
        if not _contains_tracer(beam_idx) and bool(jnp.any((beam_idx < 0) | (beam_idx >= self.batch_size))):
            raise ValueError(f"beam_idx values must be within [0, {self.batch_size}).")
        return self.replace(
            conv_state=self.conv_state[beam_idx],
            recurrent_state=self.recurrent_state[beam_idx],
            mask=self.mask[beam_idx],
            pos_id=self.pos_id[beam_idx],
        )

    def rollback(self, n: int):
        del n
        raise NotImplementedError("DeltaNet rollback requires checkpointed historical recurrent states.")


class _Qwen35RMSNorm(nn.Module):
    hidden_size: int
    eps: float
    dtype: Any = jnp.float32
    axis_name: str = "embed"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param(
            "weight",
            nn.with_logical_partitioning(nn.initializers.zeros, (self.axis_name,)),
            (self.hidden_size,),
            self.dtype,
        )
        input_dtype = x.dtype
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return (x * (1.0 + jnp.asarray(weight, jnp.float32))).astype(input_dtype)


class _Qwen35GatedRMSNorm(nn.Module):
    hidden_size: int
    eps: float
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, gate: jnp.ndarray) -> jnp.ndarray:
        weight = self.param(
            "weight",
            nn.with_logical_partitioning(nn.initializers.ones, ("head_states",)),
            (self.hidden_size,),
            self.dtype,
        )
        input_dtype = x.dtype
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        x = x * jnp.asarray(weight, jnp.float32) * jax.nn.silu(gate.astype(jnp.float32))
        return x.astype(input_dtype)


class _Qwen35CausalConv1D(nn.Module):
    channels: int
    kernel_size: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        state: jnp.ndarray | None,
        token_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        weight = self.param(
            "weight",
            nn.with_logical_partitioning(nn.initializers.normal(stddev=0.02), ("heads_merged", None)),
            (self.channels, self.kernel_size),
            self.dtype,
        )
        if state is None:
            state = jnp.zeros((x.shape[0], self.channels, self.kernel_size), dtype=x.dtype)
        else:
            state = jnp.asarray(state, dtype=x.dtype)
        expected = (x.shape[0], self.channels, self.kernel_size)
        if state.shape != expected:
            raise ValueError(f"Convolution state must have shape {expected}, got {state.shape}.")
        weight_f32 = jnp.asarray(weight, jnp.float32)

        def step(history, inputs):
            x_t, valid = inputs
            candidate = jnp.concatenate((history[..., 1:], x_t[:, :, None]), axis=-1)
            y = jnp.einsum("bck,ck->bc", candidate.astype(jnp.float32), weight_f32, precision="high")
            y = jax.nn.silu(y).astype(x.dtype)
            history = jnp.where(valid[:, None, None], candidate, history)
            y = jnp.where(valid[:, None], y, jnp.zeros((), y.dtype))
            return history, y

        final_state, output = jax.lax.scan(step, state, (jnp.swapaxes(x, 0, 1), jnp.swapaxes(token_mask, 0, 1)))
        return jnp.swapaxes(output, 0, 1), final_state


def gated_delta_recurrent_scan(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    initial_state: jnp.ndarray | None = None,
    token_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exact float32 recurrent Gated DeltaNet rule.

    ``query``/``key`` use ``[B, S, H, K]`` and ``value`` uses
    ``[B, S, H, V]``.  ``g`` and ``beta`` use ``[B, S, H]``.  The returned
    fixed-size state has shape ``[B, H, K, V]``.
    """
    query, key, value, g, beta = map(jnp.asarray, (query, key, value, g, beta))
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError(f"query and key must have the same 4D shape, got {query.shape} and {key.shape}.")
    batch, seq_len, num_heads, key_dim = query.shape
    if value.shape[:3] != (batch, seq_len, num_heads) or value.ndim != 4:
        raise ValueError("value must share query's batch, sequence, and head axes.")
    value_dim = value.shape[-1]
    if g.shape != (batch, seq_len, num_heads) or beta.shape != g.shape:
        raise ValueError(f"g and beta must have shape {(batch, seq_len, num_heads)}.")
    if token_mask is None:
        token_mask = jnp.ones((batch, seq_len), dtype=bool)
    else:
        token_mask = _binary_mask("token_mask", token_mask)
        if token_mask.shape != (batch, seq_len):
            raise ValueError(f"token_mask must have shape {(batch, seq_len)}, got {token_mask.shape}.")
    state_shape = (batch, num_heads, key_dim, value_dim)
    if initial_state is None:
        initial_state = jnp.zeros(state_shape, dtype=jnp.float32)
    else:
        initial_state = jnp.asarray(initial_state, dtype=jnp.float32)
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}, got {initial_state.shape}.")

    output_dtype = value.dtype
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    query = query * jax.lax.rsqrt(jnp.sum(jnp.square(query), axis=-1, keepdims=True) + 1e-6)
    key = key * jax.lax.rsqrt(jnp.sum(jnp.square(key), axis=-1, keepdims=True) + 1e-6)
    query = query * (key_dim**-0.5)

    def step(state, inputs):
        q_t, k_t, v_t, g_t, beta_t, valid = inputs
        decayed = state * jnp.exp(g_t)[..., None, None]
        prediction = jnp.einsum("bhk,bhkv->bhv", k_t, decayed, precision="high")
        delta = (v_t - prediction) * beta_t[..., None]
        candidate = decayed + jnp.einsum("bhk,bhv->bhkv", k_t, delta, precision="high")
        output = jnp.einsum("bhk,bhkv->bhv", q_t, candidate, precision="high")
        state = jnp.where(valid[:, None, None, None], candidate, state)
        output = jnp.where(valid[:, None, None], output, 0.0)
        return state, output

    inputs = tuple(jnp.swapaxes(x, 0, 1) for x in (query, key, value, g, beta, token_mask))
    final_state, output = jax.lax.scan(step, initial_state, inputs)
    return jnp.swapaxes(output, 0, 1).astype(output_dtype), final_state


class Qwen35GatedDeltaNet(Block):
    """Qwen's fused-projection Gated DeltaNet token mixer."""

    def setup(self):
        cfg = self.config
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.key_head_dim = cfg.linear_key_head_dim
        self.value_head_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.key_head_dim
        self.value_dim = self.num_v_heads * self.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads.")

        common = dict(dtype=self.dtype, weight_dtype=self.dtype, use_bias=False)
        self.in_proj_qkv = DenseGeneral(
            features=self.conv_dim,
            kernel_axes=("embed", "heads_merged"),
            name="in_proj_qkv",
            **common,
        )
        self.in_proj_z = DenseGeneral(
            features=(self.num_v_heads, self.value_head_dim),
            kernel_axes=("embed", "heads", "head_states"),
            name="in_proj_z",
            **common,
        )
        self.in_proj_b = DenseGeneral(
            features=self.num_v_heads,
            kernel_axes=("embed", "heads"),
            name="in_proj_b",
            **common,
        )
        self.in_proj_a = DenseGeneral(
            features=self.num_v_heads,
            kernel_axes=("embed", "heads"),
            name="in_proj_a",
            **common,
        )
        self.conv1d = _Qwen35CausalConv1D(
            channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            dtype=self.dtype,
            name="conv1d",
        )
        self.dt_bias = self.param(
            "dt_bias",
            nn.with_logical_partitioning(nn.initializers.ones, ("heads",)),
            (self.num_v_heads,),
            jnp.float32,
        )

        def init_a_log(key, shape, dtype):
            return jnp.log(jax.random.uniform(key, shape, dtype=dtype, minval=1.0, maxval=16.0))

        self.A_log = self.param(
            "A_log",
            nn.with_logical_partitioning(init_a_log, ("heads",)),
            (self.num_v_heads,),
            jnp.float32,
        )
        self.norm = _Qwen35GatedRMSNorm(
            self.value_head_dim,
            eps=cfg.norm_eps,
            dtype=self.dtype,
            name="norm",
        )
        self.out_proj = DenseGeneral(
            features=cfg.hidden_size,
            dtype=self.dtype,
            weight_dtype=self.dtype,
            kernel_axes=("heads_merged", "embed"),
            use_bias=False,
            name="out_proj",
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        kv_cache: Qwen35DeltaNetCache | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> AttentionOutput:
        del position_ids, output_attentions, kwargs
        hidden_states = jnp.asarray(hidden_states)
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states must have shape [batch, sequence, {self.hidden_size}].")
        if kv_cache is not None and not isinstance(kv_cache, Qwen35DeltaNetCache):
            raise TypeError(f"Linear-attention layers require Qwen35DeltaNetCache, got {type(kv_cache)}.")
        batch, seq_len, _ = hidden_states.shape
        populated = kv_cache is not None and kv_cache.is_populated
        if populated:
            kv_cache.validate_state()
            if seq_len != 1:
                raise ValueError("Populated DeltaNet cache calls require exactly one token.")
            token_mask = jnp.ones((batch, 1), dtype=bool)
            conv_state = kv_cache.conv_state
            recurrent_state = kv_cache.recurrent_state
        else:
            token_mask = (
                jnp.ones((batch, seq_len), dtype=bool)
                if attention_mask is None
                else _binary_mask("attention_mask", attention_mask)
            )
            if token_mask.shape != (batch, seq_len):
                raise ValueError(f"DeltaNet prefill mask must have shape {(batch, seq_len)}, got {token_mask.shape}.")
            conv_state = recurrent_state = None

        hidden_states = jnp.where(token_mask[..., None], hidden_states, jnp.zeros((), hidden_states.dtype))
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv, conv_state = self.conv1d(mixed_qkv, state=conv_state, token_mask=token_mask)
        query, key, value = jnp.split(mixed_qkv, (self.key_dim, 2 * self.key_dim), axis=-1)
        query = query.reshape(batch, seq_len, self.num_k_heads, self.key_head_dim)
        key = key.reshape(batch, seq_len, self.num_k_heads, self.key_head_dim)
        value = value.reshape(batch, seq_len, self.num_v_heads, self.value_head_dim)
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = jnp.repeat(query, repeat, axis=2)
            key = jnp.repeat(key, repeat, axis=2)

        beta = jax.nn.sigmoid(self.in_proj_b(hidden_states).astype(jnp.float32))
        a = self.in_proj_a(hidden_states).astype(jnp.float32)
        g = -jnp.exp(jnp.asarray(self.A_log, jnp.float32))[None, None, :] * jax.nn.softplus(
            a + jnp.asarray(self.dt_bias, jnp.float32)[None, None, :]
        )
        output, recurrent_state = gated_delta_recurrent_scan(
            query,
            key,
            value,
            g,
            beta,
            initial_state=recurrent_state,
            token_mask=token_mask,
        )
        z = self.in_proj_z(hidden_states)
        output = self.norm(output, z).reshape(batch, seq_len, self.value_dim)
        output = self.out_proj(output)

        if kv_cache is not None:
            kv_cache = kv_cache.update(
                conv_state,
                recurrent_state,
                token_mask,
                attention_mask=attention_mask if populated else None,
            )
        return AttentionOutput(attention_output=output, attention_weight=None, kv_cache=kv_cache)


class _Qwen35SharedExpert(Block):
    intermediate_size_override: int = 0

    def setup(self):
        intermediate = self.intermediate_size_override
        common = dict(dtype=self.dtype, weight_dtype=self.dtype, use_bias=False)
        self.gate_proj = DenseGeneral(
            features=intermediate,
            kernel_axes=("embed", "intermediate"),
            name="gate_proj",
            **common,
        )
        self.up_proj = DenseGeneral(
            features=intermediate,
            kernel_axes=("embed", "intermediate"),
            name="up_proj",
            **common,
        )
        self.down_proj = DenseGeneral(
            features=self.hidden_size,
            kernel_axes=("intermediate", "embed"),
            name="down_proj",
            **common,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35MoE(Block):
    """Top-k softmax routed Qwen MoE with a sigmoid-gated shared expert."""

    moe_implementation: str = "auto"

    def setup(self):
        cfg = self.config
        if not cfg.is_moe:
            raise ValueError("Qwen35MoE requires routed experts.")
        self.num_experts = cfg.n_routed_experts
        self.top_k = cfg.num_experts_per_tok
        self.moe_intermediate_size = cfg.moe_intermediate_size
        init = nn.initializers.normal(stddev=0.02)
        self.gate_kernel = self.param(
            "gate_kernel",
            nn.with_logical_partitioning(nn.initializers.zeros, ("embed", "expert")),
            (self.hidden_size, self.num_experts),
            jnp.float32,
        )
        self.gate_up_kernel = self.param(
            "gate_up_kernel",
            nn.with_logical_partitioning(init, ("expert", "embed", "intermediate")),
            (self.num_experts, self.hidden_size, 2 * self.moe_intermediate_size),
            self.dtype,
        )
        self.down_kernel = self.param(
            "down_kernel",
            nn.with_logical_partitioning(init, ("expert", "intermediate", "embed")),
            (self.num_experts, self.moe_intermediate_size, self.hidden_size),
            self.dtype,
        )
        self.shared_expert = _Qwen35SharedExpert(
            cfg,
            dtype=self.dtype,
            intermediate_size_override=cfg.shared_expert_intermediate_size,
            name="shared_expert",
        )
        self.shared_expert_gate = DenseGeneral(
            features=1,
            dtype=self.dtype,
            weight_dtype=self.dtype,
            kernel_axes=("embed", None),
            use_bias=False,
            name="shared_expert_gate",
        )

    def _route(self, hidden: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits = jnp.einsum("th,he->te", hidden.astype(jnp.float32), self.gate_kernel, precision="high")
        probs = jax.nn.softmax(logits, axis=-1)
        weights, indices = jax.lax.top_k(probs, self.top_k)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        return indices.astype(jnp.int32), weights.astype(jnp.float32)

    def _dense(self, hidden: jnp.ndarray, indices: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        rows = jnp.arange(hidden.shape[0])[:, None]
        routing = jnp.zeros((hidden.shape[0], self.num_experts), jnp.float32).at[rows, indices].set(weights)
        gate_up = jnp.einsum("th,ehf->tef", hidden.astype(self.dtype), self.gate_up_kernel, precision="high")
        gate, up = jnp.split(gate_up, 2, axis=-1)
        expert_hidden = jax.nn.silu(gate) * up
        expert_out = jnp.einsum("tei,eih->teh", expert_hidden, self.down_kernel, precision="high")
        return jnp.einsum("te,teh->th", routing.astype(self.dtype), expert_out, precision="high")

    def _gmm(self, hidden: jnp.ndarray, indices: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        from jax.experimental.pallas.ops.tpu import megablox

        token_count, hidden_size = hidden.shape
        assignments = token_count * self.top_k
        padded = ((assignments + _GMM_TILE - 1) // _GMM_TILE) * _GMM_TILE
        flat_expert = indices.reshape(-1)
        flat_weight = weights.reshape(-1)
        flat_token = jnp.repeat(jnp.arange(token_count, dtype=jnp.int32), self.top_k)
        order = jnp.argsort(flat_expert)
        sorted_token = flat_token[order]
        sorted_weight = flat_weight[order]
        group_sizes = jnp.bincount(flat_expert, length=self.num_experts).astype(jnp.int32)
        lhs = hidden[sorted_token].astype(self.dtype)
        lhs = jnp.pad(lhs, ((0, padded - assignments), (0, 0)))
        group_sizes = group_sizes.at[-1].add(padded - assignments)
        gate_up = megablox.gmm(
            lhs,
            self.gate_up_kernel,
            group_sizes,
            preferred_element_type=self.dtype,
        )
        gate, up = jnp.split(gate_up, 2, axis=-1)
        expert_hidden = jax.nn.silu(gate) * up
        expert_out = megablox.gmm(
            expert_hidden,
            self.down_kernel,
            group_sizes,
            preferred_element_type=self.dtype,
        )[:assignments]
        contributions = expert_out * sorted_weight[:, None].astype(expert_out.dtype)
        return jnp.zeros((token_count, hidden_size), expert_out.dtype).at[sorted_token].add(contributions)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.moe_implementation not in ("auto", "dense", "gmm"):
            raise ValueError("moe_implementation must be 'auto', 'dense', or 'gmm'.")
        x = with_sharding_constraint(x, ("batch", "length", "embed"))
        shape = x.shape
        hidden = x.reshape(-1, shape[-1])
        indices, weights = self._route(hidden)
        implementation = self.moe_implementation
        if implementation == "auto":
            implementation = "gmm" if jax.default_backend() == "tpu" else "dense"
        routed = self._gmm(hidden, indices, weights) if implementation == "gmm" else self._dense(hidden, indices, weights)
        shared = self.shared_expert(hidden) * jax.nn.sigmoid(self.shared_expert_gate(hidden))
        return (routed + shared).reshape(shape)


class _Qwen35Attention(Block):
    def setup(self):
        cfg = self.config
        self.num_kv_heads = cfg.num_key_value_heads
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_key_value_heads.")
        common = dict(dtype=self.dtype, weight_dtype=self.dtype, use_bias=False)
        self.q_proj = DenseGeneral(
            features=(self.num_heads, 2, self.head_dim),
            kernel_axes=("embed", "heads", None, "head_states"),
            name="q_proj",
            **common,
        )
        # Released Qwen checkpoints have only two KV heads.  Keep that axis
        # replicated instead of attempting to split it over four TPU shards.
        self.k_proj = DenseGeneral(
            features=(self.num_kv_heads, self.head_dim),
            kernel_axes=("embed", None, "head_states"),
            name="k_proj",
            **common,
        )
        self.v_proj = DenseGeneral(
            features=(self.num_kv_heads, self.head_dim),
            kernel_axes=("embed", None, "head_states"),
            name="v_proj",
            **common,
        )
        self.o_proj = DenseGeneral(
            features=self.hidden_size,
            dtype=self.dtype,
            weight_dtype=self.dtype,
            kernel_axes=("heads_merged", "embed"),
            use_bias=False,
            name="o_proj",
        )
        self.q_norm = _Qwen35RMSNorm(self.head_dim, self.norm_eps, self.dtype, axis_name="head_states", name="q_norm")
        self.k_norm = _Qwen35RMSNorm(self.head_dim, self.norm_eps, self.dtype, axis_name="head_states", name="k_norm")

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        *,
        cos_sin: tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        kv_cache: KVCache | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        batch, seq_len, _ = hidden_states.shape
        projected = self.q_proj(hidden_states)
        query, gate = projected[:, :, :, 0, :], projected[:, :, :, 1, :]
        key, value = self.k_proj(hidden_states), self.v_proj(hidden_states)
        query, key = self.q_norm(query), self.k_norm(key)
        if position_ids is None:
            if kv_cache is not None and kv_cache.pos_id is not None:
                position_ids = kv_cache.next_pos_id
            else:
                position_ids = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch, seq_len))
        query, key = RotaryEmbedding.apply_rotary_pos_emb(query, key, *cos_sin, position_ids)
        if kv_cache is not None:
            kv_cache = kv_cache.update(key, value, attention_mask)
            key, value, attention_mask = kv_cache.k, kv_cache.v, kv_cache.mask
        elif attention_mask is None:
            attention_mask = jnp.ones((batch, key.shape[1]), dtype=bool)

        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            key = jnp.repeat(key, repeats, axis=2)
            value = jnp.repeat(value, repeats, axis=2)
        scores = jnp.einsum("bshd,bthd->bhst", query, key, precision="high") * (self.head_dim**-0.5)
        key_len = key.shape[1]
        if seq_len != 1:
            scores += jnp.triu(jnp.full((seq_len, key_len), -jnp.inf, scores.dtype), k=1)
        if attention_mask is not None:
            attention_mask = _binary_mask("attention_mask", attention_mask)
            scores += jnp.where(attention_mask[:, None, None, :], 0.0, -jnp.inf)
        all_masked = jnp.all(jnp.isneginf(scores), axis=-1, keepdims=True)
        scores = jnp.where(all_masked, 0.0, scores)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
        output = jnp.einsum("bhst,bthd->bshd", weights, value, precision="high")
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(output * jax.nn.sigmoid(gate.reshape(batch, seq_len, -1)))
        return AttentionOutput(
            attention_output=output,
            attention_weight=weights if output_attentions else None,
            kv_cache=kv_cache,
        )


LayerCache = KVCache | Qwen35DeltaNetCache


class Qwen35Decoder(Block):
    layer_idx: int = 0
    moe_implementation: str = "auto"

    def setup(self):
        self.is_linear_attention = self.config.is_linear_attention_layer(self.layer_idx)
        if self.is_linear_attention:
            self.linear_attn = Qwen35GatedDeltaNet(self.config, dtype=self.dtype, name="linear_attn")
        else:
            self.self_attn = _Qwen35Attention(self.config, dtype=self.dtype, name="self_attn")
        self.mlp = Qwen35MoE(
            self.config,
            dtype=self.dtype,
            moe_implementation=self.moe_implementation,
            name="mlp",
        )
        self.input_layernorm = _Qwen35RMSNorm(
            self.hidden_size,
            self.norm_eps,
            self.dtype,
            name="input_layernorm",
        )
        self.post_attention_layernorm = _Qwen35RMSNorm(
            self.hidden_size,
            self.norm_eps,
            self.dtype,
            name="post_attention_layernorm",
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        *,
        cos_sin: tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        kv_cache: LayerCache | None = None,
        output_attentions: bool = False,
    ) -> DecoderOutput:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        if self.is_linear_attention:
            output = self.linear_attn(
                normalized,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
            )
        else:
            output = self.self_attn(
                normalized,
                cos_sin=cos_sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
            )
        hidden_states = residual + output.attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return DecoderOutput(
            hidden_states=hidden_states,
            kv_cache=output.kv_cache,
            attention_weight=output.attention_weight,
        )


def _init_qwen35_caches(config, max_seq_len: int, dtype: Any) -> tuple[LayerCache, ...]:
    max_seq_len = _normalize_count("max_seq_len", max_seq_len)
    dtype = _normalize_dtype("dtype", dtype)
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    return tuple(
        (
            Qwen35DeltaNetCache.init(
                max_seq_len,
                conv_dim=conv_dim,
                num_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                dtype=dtype,
            )
            if config.is_linear_attention_layer(idx)
            else KVCache.init(max_seq_len=max_seq_len, dtype=dtype)
        )
        for idx in range(config.num_layers)
    )


class Qwen35Model(Block):
    """Text-only Qwen3.5/Qwen3.6 hybrid MoE backbone."""

    moe_implementation: str = "auto"

    def setup(self):
        cfg = self.config
        if cfg.architecture != "qwen3_5_moe":
            raise ValueError(f"Qwen35Model requires architecture='qwen3_5_moe', got {cfg.architecture!r}.")
        self.embed_tokens = Embed(num_embeddings=cfg.vocab_size, features=self.hidden_size, dtype=self.dtype)
        self.layers = [
            Qwen35Decoder(
                cfg,
                dtype=self.dtype,
                layer_idx=idx,
                moe_implementation=self.moe_implementation,
            )
            for idx in range(self.num_layers)
        ]
        self.norm = _Qwen35RMSNorm(self.hidden_size, self.norm_eps, self.dtype, name="norm")

    @nn.nowrap
    def init_cache(self, max_seq_len: int, dtype: Any = None) -> tuple[LayerCache, ...]:
        return _init_qwen35_caches(self.config, max_seq_len, self.dtype if dtype is None else dtype)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[tuple[LayerCache, ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> BaseModelOutputWithCache:
        use_cache, output_attentions, output_hidden_states = normalize_model_output_flags(
            use_cache,
            output_attentions,
            output_hidden_states,
        )
        input_ids, attention_mask, kv_caches = prepare_model_inputs(
            input_ids,
            attention_mask,
            kv_caches,
            self.num_layers,
            vocab_size=self.config.vocab_size,
        )
        position_ids = prepare_position_ids(position_ids, input_ids, self.config.max_position_embeddings)
        if kv_caches is not None:
            for idx, cache in enumerate(kv_caches):
                expected = Qwen35DeltaNetCache if self.config.is_linear_attention_layer(idx) else KVCache
                if cache is not None and not isinstance(cache, expected):
                    raise TypeError(f"Layer {idx} requires {expected.__name__}, got {type(cache)}.")
        if attention_mask is None:
            attention_mask = prepare_default_attention_mask(input_ids, kv_caches)

        hidden_states = self.embed_tokens(jnp.where(input_ids < 0, 0, input_ids)).astype(self.dtype)
        hidden_states = with_sharding_constraint(hidden_states, ("batch", "length", "embed"))
        cache_lengths = () if kv_caches is None else tuple(cache.max_seq_len for cache in kv_caches if cache is not None)
        rotary_length = max((input_ids.shape[1], *cache_lengths))
        cos_sin = _rotary_cos_sin(
            rotary_length,
            head_dim=self.head_dim,
            base=self.config.rope_theta,
            rotary_pct=self.config.rotary_pct,
            rope_scale=self.config.rope_scale,
            dtype=hidden_states.dtype,
        )

        all_hidden_states = []
        all_attentions = []
        next_caches = []
        for idx, layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)
            output = layer(
                hidden_states,
                cos_sin=cos_sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=None if kv_caches is None else kv_caches[idx],
                output_attentions=output_attentions,
            )
            hidden_states = output.hidden_states
            all_attentions.append(output.attention_weight)
            next_caches.append(output.kv_cache)
        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)
        return BaseModelOutputWithCache(
            last_hidden_state=hidden_states,
            kv_caches=tuple(next_caches) if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attention_weights=tuple(all_attentions) if output_attentions else None,
        )


class Qwen35ModelWithHead(Block):
    lm_head_init: Any = nn.initializers.xavier_uniform
    lm_head_init_args: tuple = ()
    moe_implementation: str = "auto"

    def setup(self):
        self.model = Qwen35Model(
            self.config,
            dtype=self.dtype,
            moe_implementation=self.moe_implementation,
            name="model",
        )
        self.lm_head = DenseGeneral(
            features=self.config.vocab_size,
            dtype=self.dtype,
            weight_dtype=self.dtype,
            kernel_init=self.lm_head_init,
            kernel_init_args=self.lm_head_init_args,
            kernel_axes=("embed", "vocab"),
            name="lm_head",
        )

    @nn.nowrap
    def init_cache(self, max_seq_len: int, dtype: Any = None) -> tuple[LayerCache, ...]:
        return _init_qwen35_caches(self.config, max_seq_len, self.dtype if dtype is None else dtype)

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[tuple[LayerCache, ...]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        keep_last_n_logits: int = 0,
        logits_positions: Optional[jnp.ndarray] = None,
    ) -> CausalLMOutputWithCache:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_caches=kv_caches,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = slice_last_n_logits_hidden_states(
            outputs.last_hidden_state,
            keep_last_n_logits,
            logits_positions=logits_positions,
        )
        return CausalLMOutputWithCache(
            logits=self.lm_head(hidden_states),
            kv_caches=outputs.kv_caches,
            hidden_states=outputs.hidden_states,
            attention_weights=outputs.attention_weights,
        )


__all__ = [
    "LayerCache",
    "Qwen35Decoder",
    "Qwen35DeltaNetCache",
    "Qwen35GatedDeltaNet",
    "Qwen35Model",
    "Qwen35ModelWithHead",
    "Qwen35MoE",
    "gated_delta_recurrent_scan",
]
