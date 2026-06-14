# coding=utf-8
# Copyright 2023 Honglu Fan (https://github.com/honglu2875).
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

from typing import Any, Optional

import jax.numpy as jnp
from flax import struct

from .utils import get_default_pos_ids


class KVCache(struct.PyTreeNode):
    """Simple pytree object for recording kv cache."""

    k: jnp.ndarray | None = struct.field(pytree_node=True)
    v: jnp.ndarray | None = struct.field(pytree_node=True)
    max_seq_len: int = struct.field(pytree_node=False)
    mask: jnp.ndarray | None = struct.field(pytree_node=True)
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)
    pos_id: Optional[jnp.ndarray] = struct.field(default=None, pytree_node=True)

    @classmethod
    def init(
        cls,
        max_seq_len: int,
        k: Optional[jnp.ndarray] = None,
        v: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        dtype: Any = jnp.float32,
    ):
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
        return cls(
            k=k,
            v=v,
            max_seq_len=max_seq_len,
            mask=mask,
            dtype=dtype,
        )

    @property
    def next_pos_id(self):
        assert self.pos_id is not None
        return self.pos_id + 1

    @property
    def next_mask(self):
        return self.mask

    def _pad(self, x):
        x, value, max_seq_len = x
        shape = (x.shape[0], max_seq_len - x.shape[1]) + x.shape[2:]
        return jnp.concatenate([x, jnp.full(shape, value, dtype=x.dtype)], axis=1)

    def _validate_kv_inputs(self, k: jnp.ndarray, v: jnp.ndarray):
        if k.shape != v.shape:
            raise ValueError(f"k and v must have the same shape, got {k.shape} and {v.shape}.")
        if k.ndim < 2:
            raise ValueError(f"k and v must have at least batch and sequence axes, got shape {k.shape}.")

    def update(self, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray]):
        self._validate_kv_inputs(k, v)
        if self.k is None:
            if self.v is not None or self.mask is not None:
                raise ValueError("KVCache has partial state: k is empty but v or mask is populated.")
            if k.shape[1] > self.max_seq_len:
                raise ValueError(f"Cannot cache {k.shape[1]} tokens in max_seq_len={self.max_seq_len}.")
            if mask is None:
                mask = jnp.ones(k.shape[:2], dtype=bool)
            if mask.shape != k.shape[:2]:
                raise ValueError(f"mask shape must match k/v batch and sequence axes, got {mask.shape} and {k.shape[:2]}.")
            if not bool(jnp.all(jnp.any(mask, axis=1))):
                raise ValueError("mask must contain at least one valid token per batch row.")
            pos_id = get_default_pos_ids(mask)
            max_seq_len = self.max_seq_len
            k, v, mask = map(
                self._pad,
                ((k, 0, max_seq_len), (v, 0, max_seq_len), (mask, False, max_seq_len)),
            )
            return self.replace(k=k, v=v, mask=mask, pos_id=pos_id)

        if self.v is None or self.mask is None or self.pos_id is None:
            raise ValueError("KVCache has partial state: populated k requires v, mask, and pos_id.")
        if k.shape[0] != self.k.shape[0]:
            raise ValueError(f"Batch size must match cached batch size, got {k.shape[0]} and {self.k.shape[0]}.")
        if k.shape[1] != 1:
            raise ValueError(f"Decode cache updates must contain exactly one token, got sequence length {k.shape[1]}.")
        if k.shape[2:] != self.k.shape[2:]:
            raise ValueError(f"k/v trailing shape must match cached shape, got {k.shape[2:]} and {self.k.shape[2:]}.")
        batch_idx = jnp.arange(k.shape[0])[:, None]
        full_idx = jnp.concatenate([batch_idx, self.next_pos_id], axis=1)
        new_k = self.k.at[tuple(full_idx.T)].set(k.squeeze(1))
        new_v = self.v.at[tuple(full_idx.T)].set(v.squeeze(1))
        new_mask = self.mask.at[tuple(full_idx.T)].set(True)

        return self.replace(k=new_k, v=new_v, mask=new_mask, pos_id=self.next_pos_id)

    def rollback(self, n: int):
        if n <= 0:
            raise ValueError("n must be greater than 0.")
        if self.k is None or self.v is None:
            return self
        if self.pos_id is None:
            raise ValueError("Cannot roll back a cache without position ids.")
        prev_pos = self.pos_id - n
        if bool(jnp.any(prev_pos < 0)):
            raise ValueError("Cannot roll back past the beginning of the KV cache.")
        filter_mask = jnp.arange(self.k.shape[1]) <= prev_pos
        new_k = jnp.where(filter_mask[..., None, None], self.k, 0)
        new_v = jnp.where(filter_mask[..., None, None], self.v, 0)

        return self.replace(k=new_k, v=new_v, mask=filter_mask, pos_id=prev_pos)

    def resize(self, new_size: int):
        if new_size <= 0:
            raise ValueError(f"new_size must be positive, got {new_size}.")
        if self.k is None:
            return self.replace(max_seq_len=new_size)
        if self.v is None or self.mask is None or self.pos_id is None:
            raise ValueError("KVCache has partial state: populated k requires v, mask, and pos_id.")
        if bool(jnp.any(self.pos_id >= new_size)):
            raise ValueError("Cannot resize KV cache below the highest cached position.")
        if new_size > self.max_seq_len:
            new_k, new_v, new_mask = map(
                self._pad,
                ((self.k, 0, new_size), (self.v, 0, new_size), (self.mask, False, new_size)),
            )
            return self.replace(k=new_k, v=new_v, mask=new_mask, max_seq_len=new_size)
        else:
            return self.replace(
                k=self.k[:, :new_size],
                v=self.v[:, :new_size],
                mask=self.mask[:, :new_size],
                max_seq_len=new_size,
            )
