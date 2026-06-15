from typing import Any, Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from ..config import ModelConfig


def _normalize_bool(name: str, value: bool) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a boolean, got {type(value)}.")


def _normalize_kernel_axes(kernel_axes, shape: tuple) -> Tuple[str | None, ...]:
    if isinstance(kernel_axes, str):
        raise TypeError("kernel_axes must be a tuple of axis names, not a string.")
    try:
        kernel_axes = tuple(kernel_axes)
    except TypeError as e:
        raise TypeError(f"kernel_axes must be an iterable of axis names, got {type(kernel_axes)}.") from e
    if len(kernel_axes) != len(shape):
        raise ValueError(f"kernel_axes must contain one axis name per kernel dimension, got {kernel_axes} for shape {shape}.")
    for axis_name in kernel_axes:
        if axis_name is not None and not isinstance(axis_name, str):
            raise TypeError(f"kernel_axes entries must be strings or None, got {type(axis_name)}.")
    return kernel_axes


class Module(nn.Module):
    """A thin wrapper over nn.Module where init functions are automatically applied to logically
    partitioned axes."""

    kernel_init: Callable = nn.initializers.variance_scaling
    kernel_init_args: tuple = (1.0, "fan_in", "truncated_normal")
    kernel_axes: Tuple[str, ...] = ()
    with_logical_partitioning: bool = True

    def setup(self):
        with_logical_partitioning = _normalize_bool("with_logical_partitioning", self.with_logical_partitioning)

        # wrap over init function in order to receive in_axis and out_axis
        def init_fn(key: jnp.ndarray, shape: tuple, dtype: Any, in_axis: int, out_axis: int):
            fn = self.kernel_init(*self.kernel_init_args, in_axis=in_axis, out_axis=out_axis)
            if with_logical_partitioning:
                if not self.kernel_axes:
                    raise ValueError("with_logical_partitioning is True. Kernel axes must be specified.")
                fn = nn.with_logical_partitioning(fn, _normalize_kernel_axes(self.kernel_axes, shape))
            return fn(key, shape, dtype)

        self.wrapped_kernel_init = init_fn


class Block(nn.Module):
    """A template of building blocks for model implementations."""

    config: ModelConfig
    dtype: Any = jnp.float32

    @property
    def num_heads(self):
        return self.config.num_heads

    @property
    def num_layers(self):
        return self.config.num_layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def intermediate_size(self):
        return self.config.intermediate_size

    @property
    def norm_eps(self):
        return self.config.norm_eps

    @property
    def use_bias(self):
        return self.config.use_bias

    @property
    def attn_scale(self):
        return self.config.attn_scale
