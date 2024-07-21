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

import jax
import jax.numpy as jnp
from flax import struct

from .utils import get_default_pos_ids


class KVCache(struct.PyTreeNode):
    """Simple pytree object for recording kv cache."""

    k: jnp.ndarray = struct.field(pytree_node=True)
    v: jnp.ndarray = struct.field(pytree_node=True)
    max_seq_len: int = struct.field(pytree_node=False)
    mask: jnp.ndarray = struct.field(pytree_node=True)
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
        return cls(
            k=k,
            v=v,
            max_seq_len=max_seq_len,
            mask=mask,
            dtype=dtype,
        )

    @property
    def next_pos_id(self):
        return self.pos_id + 1

    @property
    def next_mask(self):
        return self.mask

    def _pad(self, x):
        x, value = x
        shape = (x.shape[0], self.max_seq_len - x.shape[1]) + x.shape[2:]
        return jnp.concatenate([x, jnp.full(shape, value, dtype=x.dtype)], axis=1)

    def update(self, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray]):
        if self.k is None:
            assert self.v is None
            assert self.mask is None
            pos_id = get_default_pos_ids(mask)
            k, v, mask = map(
                self._pad,
                ((k, 0), (v, 0), (mask, False))
            )
            return self.replace(k=k, v=v, mask=mask, pos_id=pos_id)

        assert k.shape[1] == v.shape[1] == 1
        batch_idx = jnp.arange(k.shape[0])[:, None]
        full_idx = jnp.concatenate([batch_idx, self.next_pos_id], axis=1)
        new_k = self.k.at[tuple(full_idx.T)].set(k.squeeze(1))
        new_v = self.v.at[tuple(full_idx.T)].set(v.squeeze(1))
        new_mask = self.mask.at[tuple(full_idx.T)].set(1)

        return self.replace(k=new_k, v=new_v, mask=new_mask, pos_id=self.next_pos_id)
    
    def rollback(self, n: int):
        if n <= 0:
            raise ValueError("n must be greater than 0.")
        if self.k is None or self.v is None:
            return self
        prev_pos = self.pos_id - n
        filter_mask = jnp.arange(self.k.shape[1]) <= prev_pos
        new_k = jnp.where(filter_mask[..., None, None], self.k, 0)
        new_v = jnp.where(filter_mask[..., None, None], self.v, 0)
        
        return self.replace(k=new_k, v=new_v, mask=filter_mask, pos_id=prev_pos)


