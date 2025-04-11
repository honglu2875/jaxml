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

from typing import Optional

import flax.struct
import jax.numpy as jnp

from .cache import KVCache


@flax.struct.dataclass
class BaseModelOutputWithCache:
    last_hidden_state: jnp.ndarray
    kv_caches: Optional[tuple[KVCache, ...]] = None
    hidden_states: Optional[tuple[jnp.ndarray, ...]] = None
    attention_weights: Optional[tuple[jnp.ndarray, ...]] = None


@flax.struct.dataclass
class CausalLMOutputWithCache:
    logits: jnp.ndarray
    kv_caches: Optional[tuple[KVCache, ...]] = None
    hidden_states: Optional[tuple[jnp.ndarray, ...]] = None
    attention_weights: Optional[tuple[jnp.ndarray, ...]] = None


@flax.struct.dataclass
class DecoderOutput:
    hidden_states: jnp.ndarray
    kv_cache: Optional[KVCache] = None
    attention_weight: Optional[jnp.ndarray] = None


@flax.struct.dataclass
class AttentionOutput:
    attention_output: jnp.ndarray
    attention_weight: Optional[jnp.ndarray] = None
    kv_cache: Optional[KVCache] = None


@flax.struct.dataclass
class GenerationOutput:
    tokens: jnp.ndarray
    kv_caches: tuple[KVCache, ...]
