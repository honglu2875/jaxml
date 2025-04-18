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

from typing import Any

import jax.numpy as jnp
from flax import linen as nn


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
        if self.rope_scale < 1.0:
            raise ValueError(
                f"Although rope scale in theory can be < 1.0, it is more likely a mistake (potential confusion of "
                f"whether dividing or multiplying it). I raise the error for awareness (got {self.rope_scale})."
            )
        if self.full_rotate:
            self.rotary_dim = self.dim
        else:
            self.rotary_dim = int(self.dim * self.rotary_pct)
        if self.rotary_dim <= 0:
            raise ValueError(
                f"Rotary dimension cannot be less than or equal to 0. The `rotary_pct`({self.rotary_pct}) might be too "
                f"small relative to the head dim ({self.dim})."
            )
        elif self.rotary_dim % 2 == 1:
            raise ValueError(
                f"Rotary dimension cannot be an odd number. Please adjust the `rotary_pct`({self.rotary_pct}) "
                f"according to the head dim ({self.dim})."
            )

        self.inv_freq = self.variable(
            "cache",
            "inv_freq",
            lambda: 1.0 / (self.base ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)),
        )

        if not self.disable_cache:
            emb = self.get_emb(self.max_length)
            self.cos_cached = self.variable("cache", "cos_cached", lambda: jnp.cos(emb).astype(self.dtype))
            self.sin_cached = self.variable("cache", "sin_cached", lambda: jnp.sin(emb).astype(self.dtype))

    def get_emb(self, seq_len: int):
        t = jnp.arange(seq_len, dtype=jnp.float32) / self.rope_scale
        freqs = jnp.dot(t[:, None], self.inv_freq.value[None], precision="float32", preferred_element_type="float32")
        # freqs = jnp.einsum("i,j->ij", t, self.inv_freq.value)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return emb

    def __call__(self, x: jnp.ndarray, seq_len=None):
        # x: [bs, seq_len, num_attention_heads, head_size]
        if seq_len is None:
            seq_len = x.shape[1]

        if self.disable_cache:
            # Skip updating caches and directly go for the result.
            emb = self.get_emb(seq_len)
            return jnp.cos(emb).astype(x.dtype), jnp.sin(emb).astype(x.dtype)

        return (
            self.cos_cached.value.astype(x.dtype),
            self.sin_cached.value.astype(x.dtype),
        )
