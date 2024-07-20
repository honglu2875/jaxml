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
    max_seq_len: int = struct.field(pytree_node=True)
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
        #return jnp.concatenate([self.pos_id, self.pos_id[:, -1:] + 1], axis=1)
        return self.pos_id + 1

    def update(self, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray]):
        if self.k is None:
            assert self.v is None
            assert self.mask is None
            pos_id = get_default_pos_ids(mask)[:, -1:]
            return self.replace(k=k, v=v, mask=mask, pos_id=pos_id)
        new_k = jnp.concatenate([self.k, k], axis=1)
        new_v = jnp.concatenate([self.v, v], axis=1)
        new_mask = jnp.concatenate(
            [
                self.mask, 
                jnp.ones(
                    (self.mask.shape[0], k.shape[1]), 
                    dtype=bool,
                ),
            ],
                axis=1,
        )

        return self.replace(k=new_k, v=new_v, mask=new_mask, pos_id=self.next_pos_id)
