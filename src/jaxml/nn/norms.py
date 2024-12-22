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

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.partitioning import param_with_axes


def _init_with_ones(axis_name: str):
    return nn.with_logical_partitioning(lambda _, shape, dtype: jnp.ones(shape, dtype=dtype), (axis_name,))


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    axis_name: str = "embed"
    upcast: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        self.weight = param_with_axes(
            "weight",
            _init_with_ones(self.axis_name),
            (self.hidden_size,),
            self.dtype,
            axes=(self.axis_name,),
        )

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        if self.upcast:
            hidden_states = hidden_states.astype(jnp.float32)
        square_mean = jnp.square(hidden_states).mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(square_mean + self.eps)
        return self.weight * hidden_states.astype(input_dtype)


class LayerNorm(nn.Module):
    hidden_size: int  # LayerNorm normalized_shape param with singleton dim
    eps: float = 1e-6
    axis_name: str = "embed"
    upcast: bool = True
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32


    def setup(self):
        """
        Aim to be the same as torch LayerNorm
        """
        self.weight = param_with_axes(
            "weight",
            _init_with_ones(self.axis_name),
            (self.hidden_size,),
            self.dtype,
            axes=(self.axis_name,),
        )
        if self.use_bias:
            self.bias = param_with_axes(
                "bias",
                _init_with_ones(self.axis_name),
                (self.hidden_size,),
                self.dtype,
                axes=(self.axis_name,),
            )

    def  __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        if self.upcast:
            hidden_states = hidden_states.astype(jnp.float32)

        mean = hidden_states.mean(-1, keepdims=True)
        var = hidden_states.var(-1, keepdims=True)
        out = (hidden_states - mean) / jnp.sqrt(var + self.eps) * self.weight + self.bias
        return out.astype(input_dtype)
