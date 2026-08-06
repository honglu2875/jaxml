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
"""Reference Kimi Linear model implementation.

Kimi Linear alternates constant-state Kimi Delta Attention (KDA) layers with
global NoPE MLA layers.  This module intentionally favors a small, auditable
JAX reference over a specialized long-prefill kernel: KDA is evaluated with
``jax.lax.scan`` and keeps its recurrent matrix in float32.
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
from ..nn.norms import RMSNorm
from ..outputs import AttentionOutput, BaseModelOutputWithCache, CausalLMOutputWithCache, DecoderOutput
from ..utils import get_default_pos_ids
from ._utils import (
    normalize_model_output_flags,
    prepare_default_attention_mask,
    prepare_model_inputs,
    prepare_position_ids,
    slice_last_n_logits_hidden_states,
)
from .deepseek import DeepseekMoE, MLAAttention
from .llama import LlamaMLP


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


def _validate_binary_mask(name: str, mask: jnp.ndarray) -> jnp.ndarray:
    mask = jnp.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {mask.shape}.")
    if not (jnp.issubdtype(mask.dtype, jnp.bool_) or jnp.issubdtype(mask.dtype, jnp.integer)):
        raise TypeError(f"{name} must be boolean or integer, got dtype {mask.dtype}.")
    if jnp.issubdtype(mask.dtype, jnp.integer) and not _contains_tracer(mask):
        if bool(jnp.any((mask != 0) & (mask != 1))):
            raise ValueError(f"{name} integer values must be 0 or 1.")
    return mask.astype(bool)


def _validate_right_padded_mask(name: str, mask: jnp.ndarray):
    if _contains_tracer(mask) or mask.shape[1] <= 1:
        return
    if bool(jnp.any(mask[:, :-1] < mask[:, 1:])):
        raise ValueError(f"{name} must be right padded: valid tokens cannot appear after masked positions.")


class KDACache(struct.PyTreeNode):
    """Per-layer recurrent cache for Kimi Delta Attention.

    Convolution histories contain the full causal-convolution width, matching
    FLA's ``ShortConvolution`` cache.  The recurrent state is always float32;
    convolution histories use ``dtype``.
    """

    q_conv_state: jnp.ndarray | None = struct.field(pytree_node=True)
    k_conv_state: jnp.ndarray | None = struct.field(pytree_node=True)
    v_conv_state: jnp.ndarray | None = struct.field(pytree_node=True)
    recurrent_state: jnp.ndarray | None = struct.field(pytree_node=True)
    max_seq_len: int = struct.field(pytree_node=False)
    num_heads: int = struct.field(pytree_node=False)
    head_dim: int = struct.field(pytree_node=False)
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
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        dtype: Any = jnp.float32,
        state_dtype: Any = jnp.float32,
    ) -> "KDACache":
        max_seq_len = _normalize_count("max_seq_len", max_seq_len)
        num_heads = _normalize_count("num_heads", num_heads)
        head_dim = _normalize_count("head_dim", head_dim)
        conv_kernel_size = _normalize_count("conv_kernel_size", conv_kernel_size)
        dtype = _normalize_dtype("dtype", dtype)
        state_dtype = _normalize_dtype("state_dtype", state_dtype)
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}.")
        if conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {conv_kernel_size}.")
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError(f"KDA convolution cache dtype must be floating point, got {dtype}.")
        if state_dtype != jnp.dtype(jnp.float32):
            raise TypeError(f"KDA recurrent state dtype must be float32, got {state_dtype}.")
        return cls(
            q_conv_state=None,
            k_conv_state=None,
            v_conv_state=None,
            recurrent_state=None,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            conv_kernel_size=conv_kernel_size,
            mask=None,
            dtype=dtype,
            state_dtype=state_dtype,
            pos_id=None,
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
            raise ValueError("Cannot get next position ids before KDA cache initialization.")
        return self.pos_id + 1

    @property
    def next_mask(self) -> jnp.ndarray | None:
        return self.mask

    def _validate_static_state(self):
        for name, value in (
            ("max_seq_len", self.max_seq_len),
            ("num_heads", self.num_heads),
            ("head_dim", self.head_dim),
            ("conv_kernel_size", self.conv_kernel_size),
        ):
            value = _normalize_count(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        dtype = _normalize_dtype("dtype", self.dtype)
        state_dtype = _normalize_dtype("state_dtype", self.state_dtype)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError(f"KDA convolution cache dtype must be floating point, got {dtype}.")
        if state_dtype != jnp.dtype(jnp.float32):
            raise TypeError(f"KDA recurrent state dtype must be float32, got {state_dtype}.")
        return dtype, state_dtype

    def validate_state(self) -> "KDACache":
        dtype, state_dtype = self._validate_static_state()
        states = (self.q_conv_state, self.k_conv_state, self.v_conv_state, self.recurrent_state)
        if all(state is None for state in states):
            if self.mask is not None or self.pos_id is not None:
                raise ValueError("Empty KDA cache cannot contain mask or position metadata.")
            return self
        if any(state is None for state in states):
            raise ValueError("KDA cache has partial state; all convolution and recurrent states are required.")

        batch = self.recurrent_state.shape[0]
        conv_shape = (batch, self.num_heads, self.head_dim, self.conv_kernel_size)
        recurrent_shape = (batch, self.num_heads, self.head_dim, self.head_dim)
        for name, state in (
            ("q_conv_state", self.q_conv_state),
            ("k_conv_state", self.k_conv_state),
            ("v_conv_state", self.v_conv_state),
        ):
            if state.shape != conv_shape:
                raise ValueError(f"{name} must have shape {conv_shape}, got {state.shape}.")
            if state.dtype != dtype:
                raise TypeError(f"{name} dtype must match cache dtype {dtype}, got {state.dtype}.")
            _validate_finite_values(name, state)
        if self.recurrent_state.shape != recurrent_shape:
            raise ValueError(f"recurrent_state must have shape {recurrent_shape}, got {self.recurrent_state.shape}.")
        if self.recurrent_state.dtype != state_dtype:
            raise TypeError(f"recurrent_state dtype must match state_dtype {state_dtype}, got {self.recurrent_state.dtype}.")
        _validate_finite_values("recurrent_state", self.recurrent_state)

        if self.mask is None or self.pos_id is None:
            raise ValueError("Populated KDA cache requires mask and position metadata.")
        if self.mask.shape != (batch, self.max_seq_len):
            raise ValueError(f"KDA cache mask must have shape {(batch, self.max_seq_len)}, got {self.mask.shape}.")
        if not jnp.issubdtype(self.mask.dtype, jnp.bool_):
            raise TypeError(f"KDA cache mask must be boolean, got dtype {self.mask.dtype}.")
        if self.pos_id.shape != (batch, 1):
            raise ValueError(f"KDA cache pos_id must have shape {(batch, 1)}, got {self.pos_id.shape}.")
        if not jnp.issubdtype(self.pos_id.dtype, jnp.integer):
            raise TypeError(f"KDA cache pos_id must contain integers, got dtype {self.pos_id.dtype}.")
        _validate_right_padded_mask("KDA cache mask", self.mask)
        if not _contains_tracer((self.mask, self.pos_id)):
            if not bool(jnp.all(jnp.any(self.mask, axis=1))):
                raise ValueError("KDA cache mask must contain at least one valid token per batch row.")
            if bool(jnp.any((self.pos_id < 0) | (self.pos_id >= self.max_seq_len))):
                raise ValueError(f"KDA cache pos_id values must be within [0, {self.max_seq_len}).")
            if not bool(jnp.array_equal(self.pos_id, get_default_pos_ids(self.mask))):
                raise ValueError("KDA cache pos_id must identify the last valid position in its mask.")
        return self

    def update(
        self,
        q_conv_state: jnp.ndarray,
        k_conv_state: jnp.ndarray,
        v_conv_state: jnp.ndarray,
        recurrent_state: jnp.ndarray,
        token_mask: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
    ) -> "KDACache":
        """Store a prefill result or advance a populated cache by one token."""
        self._validate_static_state()
        token_mask = _validate_binary_mask("token_mask", token_mask)
        q_conv_state = jnp.asarray(q_conv_state, dtype=self.dtype)
        k_conv_state = jnp.asarray(k_conv_state, dtype=self.dtype)
        v_conv_state = jnp.asarray(v_conv_state, dtype=self.dtype)
        recurrent_state = jnp.asarray(recurrent_state, dtype=self.state_dtype)
        batch, seq_len = token_mask.shape

        conv_shape = (batch, self.num_heads, self.head_dim, self.conv_kernel_size)
        recurrent_shape = (batch, self.num_heads, self.head_dim, self.head_dim)
        for name, state in (
            ("q_conv_state", q_conv_state),
            ("k_conv_state", k_conv_state),
            ("v_conv_state", v_conv_state),
        ):
            if state.shape != conv_shape:
                raise ValueError(f"{name} must have shape {conv_shape}, got {state.shape}.")
        if recurrent_state.shape != recurrent_shape:
            raise ValueError(f"recurrent_state must have shape {recurrent_shape}, got {recurrent_state.shape}.")

        if not self.is_populated:
            if seq_len > self.max_seq_len:
                raise ValueError(f"Cannot cache {seq_len} tokens in max_seq_len={self.max_seq_len}.")
            if not _contains_tracer(token_mask) and not bool(jnp.all(jnp.any(token_mask, axis=1))):
                raise ValueError("token_mask must contain at least one valid token per batch row.")
            _validate_right_padded_mask("token_mask", token_mask)
            pad = self.max_seq_len - seq_len
            mask = jnp.pad(token_mask, ((0, 0), (0, pad)), constant_values=False)
            return self.replace(
                q_conv_state=q_conv_state,
                k_conv_state=k_conv_state,
                v_conv_state=v_conv_state,
                recurrent_state=recurrent_state,
                mask=mask,
                pos_id=get_default_pos_ids(mask),
            )

        self.validate_state()
        if batch != self.batch_size:
            raise ValueError(f"Decode batch size must match cached batch size, got {batch} and {self.batch_size}.")
        if seq_len != 1:
            raise ValueError(f"Populated KDA cache updates require exactly one token, got sequence length {seq_len}.")
        if attention_mask is not None:
            attention_mask = _validate_binary_mask("attention_mask", attention_mask)
            if attention_mask.shape != self.mask.shape:
                raise ValueError(
                    f"Decode attention_mask must match cached mask shape, got {attention_mask.shape} and {self.mask.shape}."
                )
            if not _contains_tracer((attention_mask, self.mask)) and not bool(jnp.array_equal(attention_mask, self.mask)):
                raise ValueError("Decode attention_mask must match the current KDA cache mask.")
        if not _contains_tracer(token_mask) and not bool(jnp.all(token_mask)):
            raise ValueError("Every populated KDA decode update must contain one valid token per batch row.")
        if not _contains_tracer(self.next_pos_id) and bool(jnp.any(self.next_pos_id >= self.max_seq_len)):
            raise ValueError(f"Cannot decode past max_seq_len={self.max_seq_len}.")

        batch_idx = jnp.arange(batch)[:, None]
        full_idx = jnp.concatenate([batch_idx, self.next_pos_id], axis=1)
        mask = self.mask.at[tuple(full_idx.T)].set(True)
        return self.replace(
            q_conv_state=q_conv_state,
            k_conv_state=k_conv_state,
            v_conv_state=v_conv_state,
            recurrent_state=recurrent_state,
            mask=mask,
            pos_id=self.next_pos_id,
        )

    def resize(self, new_size: int) -> "KDACache":
        new_size = _normalize_count("new_size", new_size)
        if new_size <= 0:
            raise ValueError(f"new_size must be positive, got {new_size}.")
        self.validate_state()
        if not self.is_populated:
            return self.replace(max_seq_len=new_size)
        if not _contains_tracer(self.pos_id) and bool(jnp.any(self.pos_id >= new_size)):
            raise ValueError("Cannot resize KDA cache below the highest cached position.")
        if new_size > self.max_seq_len:
            mask = jnp.pad(self.mask, ((0, 0), (0, new_size - self.max_seq_len)), constant_values=False)
        else:
            mask = self.mask[:, :new_size]
        return self.replace(mask=mask, max_seq_len=new_size)

    def reorder_cache(self, beam_idx: jnp.ndarray) -> "KDACache":
        """Return a cache reordered along its batch axis (for beam search)."""
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
            q_conv_state=self.q_conv_state[beam_idx],
            k_conv_state=self.k_conv_state[beam_idx],
            v_conv_state=self.v_conv_state[beam_idx],
            recurrent_state=self.recurrent_state[beam_idx],
            mask=self.mask[beam_idx],
            pos_id=self.pos_id[beam_idx],
        )

    def rollback(self, n: int):
        del n
        raise NotImplementedError("KDA rollback requires checkpointed historical recurrent states.")


LayerCache = KVCache | KDACache


class KimiShortConvolution(nn.Module):
    """Causal depthwise convolution used by KDA, including its SiLU."""

    num_heads: int
    head_dim: int
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
        x = jnp.asarray(x)
        if x.ndim != 4 or x.shape[2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                "KimiShortConvolution input must have shape "
                f"[batch, sequence, {self.num_heads}, {self.head_dim}], got {x.shape}."
            )
        token_mask = _validate_binary_mask("token_mask", token_mask)
        if token_mask.shape != x.shape[:2]:
            raise ValueError(f"token_mask shape must match convolution batch/sequence axes, got {token_mask.shape}.")

        weight_init = nn.with_logical_partitioning(
            nn.initializers.lecun_normal(),
            ("heads", "head_states", None),
        )
        weight = self.param(
            "weight",
            weight_init,
            (self.num_heads, self.head_dim, self.kernel_size),
            self.dtype,
        )
        if state is None:
            state = jnp.zeros(
                (x.shape[0], self.num_heads, self.head_dim, self.kernel_size),
                dtype=x.dtype,
            )
        else:
            state = jnp.asarray(state, dtype=x.dtype)
        expected_state_shape = (x.shape[0], self.num_heads, self.head_dim, self.kernel_size)
        if state.shape != expected_state_shape:
            raise ValueError(f"Convolution state must have shape {expected_state_shape}, got {state.shape}.")

        weight_f32 = jnp.asarray(weight, dtype=jnp.float32)

        def step(history, inputs):
            x_t, valid = inputs
            candidate = jnp.concatenate([history[..., 1:], x_t[..., None]], axis=-1)
            conv = jnp.einsum(
                "bhdw,hdw->bhd",
                candidate.astype(jnp.float32),
                weight_f32,
                precision="high",
            )
            conv = jax.nn.silu(conv).astype(x.dtype)
            valid_state = valid[:, None, None, None]
            valid_output = valid[:, None, None]
            history = jnp.where(valid_state, candidate, history)
            conv = jnp.where(valid_output, conv, jnp.zeros((), dtype=conv.dtype))
            return history, conv

        final_state, output = jax.lax.scan(
            step,
            state,
            (jnp.swapaxes(x, 0, 1), jnp.swapaxes(token_mask, 0, 1)),
        )
        return jnp.swapaxes(output, 0, 1), final_state


def kda_recurrent_scan(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    log_decay: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    initial_state: jnp.ndarray | None = None,
    token_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reference recurrent KDA equation.

    Inputs use ``[batch, sequence, heads, dimension]`` except ``beta``, which
    uses ``[batch, sequence, heads]``.  Outputs are cast back to ``v.dtype``;
    the returned recurrent state is float32.
    """
    q, k, v, log_decay, beta = map(jnp.asarray, (q, k, v, log_decay, beta))
    if not (q.shape == k.shape == v.shape == log_decay.shape):
        raise ValueError(
            "q, k, v, and log_decay must have identical shapes, got "
            f"{q.shape}, {k.shape}, {v.shape}, and {log_decay.shape}."
        )
    if q.ndim != 4 or any(axis <= 0 for axis in q.shape):
        raise ValueError(f"KDA q/k/v inputs must be non-empty 4D arrays, got shape {q.shape}.")
    batch, seq_len, num_heads, head_dim = q.shape
    if beta.shape != (batch, seq_len, num_heads):
        raise ValueError(f"beta must have shape {(batch, seq_len, num_heads)}, got {beta.shape}.")
    if token_mask is None:
        token_mask = jnp.ones((batch, seq_len), dtype=bool)
    else:
        token_mask = _validate_binary_mask("token_mask", token_mask)
        if token_mask.shape != (batch, seq_len):
            raise ValueError(f"token_mask must have shape {(batch, seq_len)}, got {token_mask.shape}.")
    state_shape = (batch, num_heads, head_dim, head_dim)
    if initial_state is None:
        initial_state = jnp.zeros(state_shape, dtype=jnp.float32)
    else:
        initial_state = jnp.asarray(initial_state, dtype=jnp.float32)
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}, got {initial_state.shape}.")

    output_dtype = v.dtype
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    log_decay = log_decay.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    q = q / jnp.sqrt(jnp.sum(jnp.square(q), axis=-1, keepdims=True) + 1e-6)
    k = k / jnp.sqrt(jnp.sum(jnp.square(k), axis=-1, keepdims=True) + 1e-6)
    q = q * (head_dim**-0.5)

    def step(state, inputs):
        q_t, k_t, v_t, decay_t, beta_t, valid = inputs
        decayed = state * jnp.exp(decay_t)[..., None]
        prediction = jnp.einsum("bhd,bhdv->bhv", k_t, decayed, precision="high")
        error = v_t - prediction
        candidate = decayed + jnp.einsum(
            "bhd,bhv->bhdv",
            beta_t[..., None] * k_t,
            error,
            precision="high",
        )
        output = jnp.einsum("bhd,bhdv->bhv", q_t, candidate, precision="high")
        state = jnp.where(valid[:, None, None, None], candidate, state)
        output = jnp.where(valid[:, None, None], output, 0.0)
        return state, output

    scan_inputs = tuple(jnp.swapaxes(x, 0, 1) for x in (q, k, v, log_decay, beta, token_mask))
    final_state, output = jax.lax.scan(step, initial_state, scan_inputs)
    return jnp.swapaxes(output, 0, 1).astype(output_dtype), final_state


class KimiGatedRMSNorm(nn.Module):
    """Head-wise RMSNorm followed by Kimi's sigmoid output gate."""

    head_dim: int
    eps: float
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, gate: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        gate = jnp.asarray(gate)
        if x.shape != gate.shape or x.shape[-1] != self.head_dim:
            raise ValueError(
                f"KimiGatedRMSNorm x and gate must share a trailing dimension of {self.head_dim}, "
                f"got {x.shape} and {gate.shape}."
            )
        weight_init = nn.with_logical_partitioning(nn.initializers.ones, ("head_states",))
        weight = self.param("weight", weight_init, (self.head_dim,), self.dtype)
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        normalized = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        output = normalized * jnp.asarray(weight, jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
        return output.astype(x_dtype)


class KimiDeltaAttention(Block):
    """Kimi Delta Attention with exact recurrent reference math."""

    def setup(self):
        cfg = self.config
        if cfg.kda_num_heads is None or cfg.kda_head_dim is None:
            raise ValueError("KimiDeltaAttention requires kda_num_heads and kda_head_dim.")
        self.kda_num_heads = cfg.kda_num_heads
        self.kda_head_dim = cfg.kda_head_dim
        self.projection_size = self.kda_num_heads * self.kda_head_dim

        projection_args = dict(
            features=(self.kda_num_heads, self.kda_head_dim),
            axis=-1,
            kernel_axes=("embed", "heads", "head_states"),
            dtype=self.dtype,
            weight_dtype=self.dtype,
            use_bias=False,
        )
        self.q_proj = DenseGeneral(**projection_args, name="q_proj")
        self.k_proj = DenseGeneral(**projection_args, name="k_proj")
        self.v_proj = DenseGeneral(**projection_args, name="v_proj")

        conv_args = dict(
            num_heads=self.kda_num_heads,
            head_dim=self.kda_head_dim,
            kernel_size=cfg.kda_short_conv_kernel_size,
            dtype=self.dtype,
        )
        self.q_conv1d = KimiShortConvolution(**conv_args, name="q_conv1d")
        self.k_conv1d = KimiShortConvolution(**conv_args, name="k_conv1d")
        self.v_conv1d = KimiShortConvolution(**conv_args, name="v_conv1d")

        rank_args = dict(
            features=self.kda_head_dim,
            axis=-1,
            kernel_axes=("embed", "head_states"),
            dtype=self.dtype,
            weight_dtype=self.dtype,
            use_bias=False,
        )
        expand_args = dict(
            features=(self.kda_num_heads, self.kda_head_dim),
            axis=-1,
            kernel_axes=("head_states", "heads", None),
            dtype=self.dtype,
            weight_dtype=self.dtype,
            use_bias=False,
        )
        self.f_a_proj = DenseGeneral(**rank_args, name="f_a_proj")
        self.f_b_proj = DenseGeneral(**expand_args, name="f_b_proj")
        self.b_proj = DenseGeneral(
            features=self.kda_num_heads,
            axis=-1,
            kernel_axes=("embed", "heads"),
            dtype=self.dtype,
            weight_dtype=self.dtype,
            use_bias=False,
            name="b_proj",
        )
        self.g_a_proj = DenseGeneral(**rank_args, name="g_a_proj")
        self.g_b_proj = DenseGeneral(**expand_args, name="g_b_proj")

        def init_a_log(key, shape, dtype):
            values = jax.random.uniform(key, shape, dtype=dtype, minval=1.0, maxval=16.0)
            return jnp.log(values)

        self.A_log = self.param(
            "A_log",
            nn.with_logical_partitioning(init_a_log, ("heads",)),
            (self.kda_num_heads,),
            jnp.float32,
        )
        self.dt_bias = self.param(
            "dt_bias",
            nn.with_logical_partitioning(nn.initializers.zeros, ("heads", "head_states")),
            (self.kda_num_heads, self.kda_head_dim),
            jnp.float32,
        )
        self.o_norm = KimiGatedRMSNorm(self.kda_head_dim, eps=cfg.norm_eps, dtype=self.dtype, name="o_norm")
        self.o_proj = DenseGeneral(
            features=cfg.hidden_size,
            axis=-1,
            kernel_axes=("heads_merged", "embed"),
            dtype=self.dtype,
            weight_dtype=self.dtype,
            use_bias=False,
            name="o_proj",
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        kv_cache: KDACache | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> AttentionOutput:
        del position_ids, output_attentions, kwargs
        hidden_states = jnp.asarray(hidden_states)
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"KimiDeltaAttention hidden_states must have shape [batch, sequence, {self.hidden_size}], "
                f"got {hidden_states.shape}."
            )
        if not jnp.issubdtype(hidden_states.dtype, jnp.floating):
            raise TypeError(f"hidden_states must contain floating point values, got {hidden_states.dtype}.")
        if kv_cache is not None and not isinstance(kv_cache, KDACache):
            raise TypeError(f"KDA layers require KDACache or None, got {type(kv_cache)}.")

        batch, seq_len, _ = hidden_states.shape
        populated = kv_cache is not None and kv_cache.is_populated
        if populated:
            kv_cache.validate_state()
            if seq_len != 1:
                raise ValueError(f"Populated KDA cache calls require one token, got sequence length {seq_len}.")
            token_mask = jnp.ones((batch, 1), dtype=bool)
            q_state, k_state, v_state = kv_cache.q_conv_state, kv_cache.k_conv_state, kv_cache.v_conv_state
            recurrent_state = kv_cache.recurrent_state
        else:
            if attention_mask is None:
                token_mask = jnp.ones((batch, seq_len), dtype=bool)
            else:
                token_mask = _validate_binary_mask("attention_mask", attention_mask)
                if token_mask.shape != (batch, seq_len):
                    raise ValueError(f"KDA prefill attention_mask must have shape {(batch, seq_len)}, got {token_mask.shape}.")
            q_state = k_state = v_state = recurrent_state = None

        q, q_state = self.q_conv1d(self.q_proj(hidden_states), state=q_state, token_mask=token_mask)
        k, k_state = self.k_conv1d(self.k_proj(hidden_states), state=k_state, token_mask=token_mask)
        v, v_state = self.v_conv1d(self.v_proj(hidden_states), state=v_state, token_mask=token_mask)

        fine_gate = self.f_b_proj(self.f_a_proj(hidden_states)).astype(jnp.float32)
        log_decay = -jnp.exp(jnp.asarray(self.A_log, jnp.float32))[None, None, :, None] * jax.nn.softplus(
            fine_gate + jnp.asarray(self.dt_bias, jnp.float32)[None, None, :, :]
        )
        beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))
        output, recurrent_state = kda_recurrent_scan(
            q,
            k,
            v,
            log_decay,
            beta,
            initial_state=recurrent_state,
            token_mask=token_mask,
        )

        output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        output = self.o_norm(output, output_gate)
        output = self.o_proj(output.reshape(batch, seq_len, self.projection_size))

        if kv_cache is not None:
            kv_cache = kv_cache.update(
                q_state,
                k_state,
                v_state,
                recurrent_state,
                token_mask,
                attention_mask=attention_mask if populated else None,
            )
        return AttentionOutput(attention_output=output, attention_weight=None, kv_cache=kv_cache)


class KimiLinearDecoder(Block):
    """One hybrid Kimi Linear decoder layer."""

    layer_idx: int = 0
    moe_implementation: str = "auto"

    def setup(self):
        cfg = self.config
        self.is_kda = cfg.is_kda_layer(self.layer_idx)
        self.use_moe = cfg.is_moe_layer(self.layer_idx)
        self.self_attn = (
            KimiDeltaAttention(cfg, dtype=self.dtype, name="self_attn")
            if self.is_kda
            else MLAAttention(cfg, dtype=self.dtype, name="self_attn")
        )
        if self.use_moe:
            self.block_sparse_moe = DeepseekMoE(
                cfg,
                dtype=self.dtype,
                moe_implementation=self.moe_implementation,
                name="block_sparse_moe",
            )
        else:
            self.mlp = LlamaMLP(cfg, dtype=self.dtype, name="mlp")
        self.input_layernorm = RMSNorm(self.hidden_size, eps=self.norm_eps, dtype=self.dtype, upcast=True)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.norm_eps, dtype=self.dtype, upcast=True)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        kv_cache: LayerCache | None = None,
        output_attentions: bool = False,
    ) -> DecoderOutput:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        output = (
            self.self_attn(
                hidden_states=hidden_states,
                cos_sin=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
            )
            if not self.is_kda
            else self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
            )
        )
        hidden_states = residual + output.attention_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_moe:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return DecoderOutput(
            hidden_states=hidden_states,
            kv_cache=output.kv_cache,
            attention_weight=output.attention_weight,
        )


def _init_kimi_caches(config, max_seq_len: int, dtype: Any) -> tuple[LayerCache, ...]:
    max_seq_len = _normalize_count("max_seq_len", max_seq_len)
    dtype = _normalize_dtype("dtype", dtype)
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
    return tuple(
        (
            KDACache.init(
                max_seq_len,
                num_heads=config.kda_num_heads,
                head_dim=config.kda_head_dim,
                conv_kernel_size=config.kda_short_conv_kernel_size,
                dtype=dtype,
            )
            if config.is_kda_layer(idx)
            else KVCache.init(max_seq_len=max_seq_len, dtype=dtype)
        )
        for idx in range(config.num_layers)
    )


class KimiLinearModel(Block):
    """Kimi Linear hybrid KDA/NoPE-MLA backbone."""

    moe_implementation: str = "auto"

    def setup(self):
        cfg = self.config
        if cfg.architecture != "kimi_linear":
            raise ValueError(f"KimiLinearModel requires architecture='kimi_linear', got {cfg.architecture!r}.")
        if not (cfg.is_mla and cfg.is_moe):
            raise ValueError("KimiLinearModel requires MLA and MoE configuration fields.")
        if not cfg.mla_use_nope:
            raise ValueError("KimiLinearModel requires mla_use_nope=True.")
        if cfg.q_lora_rank is not None:
            raise ValueError("Kimi Linear requires q_lora_rank=None.")
        if not cfg.kda_layers:
            raise ValueError("KimiLinearModel requires at least one KDA layer.")
        if len(cfg.kda_layers) == cfg.num_layers:
            raise ValueError("KimiLinearModel requires at least one global MLA layer.")

        self.embed_tokens = Embed(num_embeddings=cfg.vocab_size, features=self.hidden_size, dtype=self.dtype)
        self.layers = [
            KimiLinearDecoder(
                cfg,
                dtype=self.dtype,
                layer_idx=idx,
                moe_implementation=self.moe_implementation,
            )
            for idx in range(self.num_layers)
        ]
        self.norm = RMSNorm(self.hidden_size, eps=self.norm_eps, dtype=self.dtype, upcast=True)

    @nn.nowrap
    def init_cache(self, max_seq_len: int, dtype: Any = None) -> tuple[LayerCache, ...]:
        return _init_kimi_caches(self.config, max_seq_len, self.dtype if dtype is None else dtype)

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
        cfg = self.config
        use_cache, output_attentions, output_hidden_states = normalize_model_output_flags(
            use_cache, output_attentions, output_hidden_states
        )
        input_ids, attention_mask, kv_caches = prepare_model_inputs(
            input_ids,
            attention_mask,
            kv_caches,
            self.num_layers,
            vocab_size=cfg.vocab_size,
        )
        position_ids = prepare_position_ids(position_ids, input_ids, cfg.max_position_embeddings)
        if kv_caches is not None:
            for idx, cache in enumerate(kv_caches):
                expected = KDACache if cfg.is_kda_layer(idx) else KVCache
                if cache is not None and not isinstance(cache, expected):
                    raise TypeError(f"Layer {idx} requires {expected.__name__} or None, got {type(cache)}.")
        if attention_mask is None:
            attention_mask = prepare_default_attention_mask(input_ids, kv_caches)

        safe_input_ids = jnp.where(input_ids < 0, 0, input_ids)
        hidden_states = self.embed_tokens(safe_input_ids).astype(self.dtype)
        hidden_states = with_sharding_constraint(hidden_states, ("batch", "length", "embed"))

        all_hidden_states = []
        all_self_attns = []
        next_caches = []
        for idx, decoder_layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)
            layer_cache = None if kv_caches is None else kv_caches[idx]
            output = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=layer_cache,
                output_attentions=output_attentions,
            )
            hidden_states = output.hidden_states
            all_self_attns.append(output.attention_weight)
            next_caches.append(output.kv_cache)

        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)
        return BaseModelOutputWithCache(
            last_hidden_state=hidden_states,
            kv_caches=tuple(next_caches) if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attention_weights=tuple(all_self_attns) if output_attentions else None,
        )


class KimiLinearModelWithHead(Block):
    """Kimi Linear causal language model with an untied output head."""

    lm_head_init: Any = nn.initializers.xavier_uniform
    lm_head_init_args: tuple = ()
    moe_implementation: str = "auto"

    def setup(self):
        self.model = KimiLinearModel(
            self.config,
            dtype=self.dtype,
            moe_implementation=self.moe_implementation,
        )
        self.lm_head = DenseGeneral(
            features=self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=self.lm_head_init,
            kernel_init_args=self.lm_head_init_args,
            with_logical_partitioning=True,
            kernel_axes=("embed", "vocab"),
            name="lm_head",
        )

    @nn.nowrap
    def init_cache(self, max_seq_len: int, dtype: Any = None) -> tuple[LayerCache, ...]:
        return _init_kimi_caches(self.config, max_seq_len, self.dtype if dtype is None else dtype)

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
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithCache(
            logits=logits,
            kv_caches=outputs.kv_caches,
            hidden_states=outputs.hidden_states,
            attention_weights=outputs.attention_weights,
        )


__all__ = [
    "KDACache",
    "KimiDeltaAttention",
    "KimiGatedRMSNorm",
    "KimiLinearDecoder",
    "KimiLinearModel",
    "KimiLinearModelWithHead",
    "KimiShortConvolution",
    "LayerCache",
    "kda_recurrent_scan",
]
