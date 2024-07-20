from typing import Any, Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp

from ..config import ModelConfig


class Module(nn.Module):
    """A thin wrapper over nn.Module where init functions are automatically applied to logically
    partitioned axes."""

    kernel_init: Callable = nn.initializers.variance_scaling
    kernel_init_args: tuple = (1.0, "fan_in", "truncated_normal")
    kernel_axes: Tuple[str, ...] = ()
    with_logical_partitioning: bool = True

    def setup(self):
        # wrap over init function in order to receive in_axis and out_axis
        def init_fn(key: jnp.ndarray, shape: tuple, dtype: Any, in_axis: int, out_axis: int):
            fn = self.kernel_init(*self.kernel_init_args, in_axis=in_axis, out_axis=out_axis)
            if self.with_logical_partitioning:
                if not self.kernel_axes:
                    raise ValueError("with_logical_partitioning is True. Kernel axes must be specified.")
                fn = nn.with_logical_partitioning(fn, self.kernel_axes)
            return fn(key, shape, dtype)

        self.wrapped_kernel_init = init_fn


class Block(nn.Module):
    """A template of building blocks for model implementations."""

    config: ModelConfig
    dtype: Any = jnp.bfloat16

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
