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

import math
import operator
from numbers import Real
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def _normalize_count(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {type(value)}.")
    try:
        return operator.index(value)
    except TypeError as e:
        raise TypeError(f"{name} must be an integer, got {type(value)}.") from e


def _normalize_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value)}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
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


class RotaryEmbedding(nn.Module):
    dim: int
    # max_trained_length is the initial context window, and we may extend it at inference time.
    max_length: int = 2048
    base: float = 10000.0
    dtype: Any = jnp.float32
    disable_cache: bool = False
    rotary_pct: float = 1.0
    rope_scale: float = 1.0

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return jnp.concatenate((-x2, x1), axis=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        q = jnp.asarray(q)
        k = jnp.asarray(k)
        cos = jnp.asarray(cos)
        sin = jnp.asarray(sin)
        position_ids = jnp.asarray(position_ids)
        if q.ndim != 4:
            raise ValueError(f"q must be a 4D array, got shape {q.shape}.")
        if k.ndim != 4:
            raise ValueError(f"k must be a 4D array, got shape {k.shape}.")
        for name, value in (("q", q), ("k", k), ("cos", cos), ("sin", sin)):
            if not jnp.issubdtype(value.dtype, jnp.floating):
                raise TypeError(f"{name} must contain floating point values, got dtype {value.dtype}.")
        if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
            raise ValueError(f"q and k must have matching batch, sequence, and head dimensions, got {q.shape} and {k.shape}.")
        if cos.ndim != 2 or sin.ndim != 2:
            raise ValueError(f"cos and sin must be 2D arrays, got shapes {cos.shape} and {sin.shape}.")
        if cos.shape != sin.shape:
            raise ValueError(f"cos and sin must have the same shape, got {cos.shape} and {sin.shape}.")
        if cos.shape[0] <= 0:
            raise ValueError("cos and sin must contain at least one position.")
        if cos.shape[-1] <= 0 or cos.shape[-1] % 2:
            raise ValueError(f"rotary dimension must be positive and even, got {cos.shape[-1]}.")
        if cos.shape[-1] > q.shape[-1]:
            raise ValueError(f"rotary dimension cannot exceed q/k head dimension, got {cos.shape[-1]} and {q.shape[-1]}.")
        if position_ids.ndim != 2:
            raise ValueError(f"position_ids must be a 2D array, got shape {position_ids.shape}.")
        if position_ids.shape[1] != q.shape[1] or position_ids.shape[0] not in (1, q.shape[0]):
            raise ValueError(
                "position_ids shape must be broadcastable to q/k batch and sequence axes, "
                f"got {position_ids.shape} and {q.shape[:2]}."
            )
        if not jnp.issubdtype(position_ids.dtype, jnp.integer):
            raise TypeError(f"position_ids must contain integer positions, got dtype {position_ids.dtype}.")
        if not _contains_tracer(position_ids):
            if bool(jnp.any(position_ids < 0)):
                raise ValueError("position_ids must contain non-negative positions.")
            if bool(jnp.any(position_ids >= cos.shape[0])):
                raise ValueError(f"position_ids must be within rotary table length {cos.shape[0]}.")

        # [seq_len, dim] -> [batch_size, seq_len, 1, head_dim]
        rotary_dim = cos.shape[-1]
        cos = jnp.expand_dims(jnp.take(cos, position_ids, axis=0), axis=2)
        sin = jnp.expand_dims(jnp.take(sin, position_ids, axis=0), axis=2)
        qr, kr = q[..., :rotary_dim], k[..., :rotary_dim]
        q_embed = (qr * cos) + (RotaryEmbedding.rotate_half(qr) * sin)
        k_embed = (kr * cos) + (RotaryEmbedding.rotate_half(kr) * sin)
        if rotary_dim < q.shape[-1]:
            q_embed, k_embed = map(
                lambda x: jnp.concatenate([x[0], x[1][..., rotary_dim:]], axis=-1),
                ((q_embed, q), (k_embed, k)),
            )
        return q_embed, k_embed

    @property
    def full_rotate(self):
        return self.rotary_pct >= 1.0

    def setup(self):
        dim = _normalize_count("dim", self.dim)
        max_length = _normalize_count("max_length", self.max_length)
        base = _normalize_float("base", self.base)
        rope_scale = _normalize_float("rope_scale", self.rope_scale)
        rotary_pct = _normalize_float("rotary_pct", self.rotary_pct)
        dtype = _normalize_dtype("dtype", self.dtype)
        disable_cache = _normalize_bool("disable_cache", self.disable_cache)
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}.")
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}.")
        if rope_scale < 1.0:
            raise ValueError(
                f"Although rope scale in theory can be < 1.0, it is more likely a mistake (potential confusion of "
                f"whether dividing or multiplying it). I raise the error for awareness (got {rope_scale})."
            )
        if not 0 < rotary_pct <= 1:
            raise ValueError(f"rotary_pct must be in (0, 1], got {rotary_pct}.")
        if rotary_pct >= 1.0:
            self.rotary_dim = dim
        else:
            self.rotary_dim = int(dim * rotary_pct)
        if self.rotary_dim <= 0:
            raise ValueError(
                f"Rotary dimension cannot be less than or equal to 0. The `rotary_pct`({rotary_pct}) might be too "
                f"small relative to the head dim ({dim})."
            )
        elif self.rotary_dim % 2 == 1:
            raise ValueError(
                f"Rotary dimension cannot be an odd number. Please adjust the `rotary_pct`({rotary_pct}) "
                f"according to the head dim ({dim})."
            )

        self.inv_freq = self.variable(
            "cache",
            "inv_freq",
            lambda: 1.0 / (base ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)),
        )

        if not disable_cache:
            emb = self.get_emb(max_length)
            self.cos_cached = self.variable("cache", "cos_cached", lambda: jnp.cos(emb).astype(dtype))
            self.sin_cached = self.variable("cache", "sin_cached", lambda: jnp.sin(emb).astype(dtype))

    def get_emb(self, seq_len: int):
        t = jnp.arange(seq_len, dtype=jnp.float32) / self.rope_scale
        freqs = jnp.dot(t[:, None], self.inv_freq.value[None], precision="float32", preferred_element_type="float32")
        # freqs = jnp.einsum("i,j->ij", t, self.inv_freq.value)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return emb

    def __call__(self, x: jnp.ndarray, seq_len=None):
        # x supplies dtype plus batch/sequence axes; callers may pass hidden states or projected states.
        x = jnp.asarray(x)
        if x.ndim < 2:
            raise ValueError(f"x must have batch and sequence axes, got shape {x.shape}.")
        if not jnp.issubdtype(x.dtype, jnp.floating):
            raise TypeError(f"x must contain floating point values, got dtype {x.dtype}.")
        if seq_len is None:
            seq_len = x.shape[1]
        seq_len = _normalize_count("seq_len", seq_len)
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")

        if _normalize_bool("disable_cache", self.disable_cache):
            # Skip updating caches and directly go for the result.
            emb = self.get_emb(seq_len)
            return jnp.cos(emb).astype(x.dtype), jnp.sin(emb).astype(x.dtype)

        max_length = self.cos_cached.value.shape[0]
        if seq_len > max_length:
            raise ValueError(
                f"Cached rotary embeddings only cover max_length={max_length}, got seq_len={seq_len}. "
                "Set disable_cache=True to compute longer embeddings dynamically."
            )
        return (
            self.cos_cached.value.astype(x.dtype),
            self.sin_cached.value.astype(x.dtype),
        )
