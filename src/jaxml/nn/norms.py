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

import math
import operator
from numbers import Real

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.partitioning import param_with_axes


def _init_with_ones(axis_name: str):
    return nn.with_logical_partitioning(lambda _, shape, dtype: jnp.ones(shape, dtype=dtype), (axis_name,))


def _init_with_zeros(axis_name: str):
    return nn.with_logical_partitioning(lambda _, shape, dtype: jnp.zeros(shape, dtype=dtype), (axis_name,))


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


def _normalize_dtype(name: str, value):
    if value is None:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.")
    try:
        return jnp.dtype(value)
    except TypeError as e:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.") from e


def _validate_hidden_states_shape(module_name: str, hidden_states, hidden_size: int):
    hidden_states = jnp.asarray(hidden_states)
    if hidden_states.ndim == 0:
        raise ValueError(f"{module_name} input must have at least one dimension.")
    if not jnp.issubdtype(hidden_states.dtype, jnp.floating):
        raise TypeError(f"{module_name} input must contain floating point values, got dtype {hidden_states.dtype}.")
    if hidden_states.size == 0:
        raise ValueError(f"{module_name} input must not contain empty axes, got shape {hidden_states.shape}.")
    if hidden_states.shape[-1] != hidden_size:
        raise ValueError(f"{module_name} hidden dimension mismatch: got {hidden_states.shape[-1]} and expected {hidden_size}.")
    return hidden_states


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
        hidden_size = _normalize_count("hidden_size", self.hidden_size)
        eps = _normalize_float("eps", self.eps)
        _normalize_bool("upcast", self.upcast)
        dtype = _normalize_dtype("dtype", self.dtype)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.weight = param_with_axes(
            "weight",
            _init_with_ones(self.axis_name),
            (hidden_size,),
            dtype,
            axes=(self.axis_name,),
        )

    def __call__(self, hidden_states):
        hidden_states = _validate_hidden_states_shape("RMSNorm", hidden_states, self.hidden_size)
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
        hidden_size = _normalize_count("hidden_size", self.hidden_size)
        eps = _normalize_float("eps", self.eps)
        _normalize_bool("upcast", self.upcast)
        use_bias = _normalize_bool("use_bias", self.use_bias)
        dtype = _normalize_dtype("dtype", self.dtype)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.weight = param_with_axes(
            "weight",
            _init_with_ones(self.axis_name),
            (hidden_size,),
            dtype,
            axes=(self.axis_name,),
        )
        if use_bias:
            self.bias = param_with_axes(
                "bias",
                _init_with_zeros(self.axis_name),
                (hidden_size,),
                dtype,
                axes=(self.axis_name,),
            )

    def __call__(self, hidden_states):
        hidden_states = _validate_hidden_states_shape("LayerNorm", hidden_states, self.hidden_size)
        input_dtype = hidden_states.dtype
        if self.upcast:
            hidden_states = hidden_states.astype(jnp.float32)

        mean = hidden_states.mean(-1, keepdims=True)
        var = hidden_states.var(-1, keepdims=True)
        out = (hidden_states - mean) / jnp.sqrt(var + self.eps) * self.weight
        if self.use_bias:
            out += self.bias
        return out.astype(input_dtype)
