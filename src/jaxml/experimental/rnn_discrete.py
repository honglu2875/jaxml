import operator
from dataclasses import field
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ..nn.embedding import Embed
from ..nn.linear import DenseGeneral
from ..nn.module import Module
from ..nn.norms import RMSNorm


def _normalize_count(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {type(value)}.")
    try:
        return operator.index(value)
    except TypeError as e:
        raise TypeError(f"{name} must be an integer, got {type(value)}.") from e


def _normalize_bool(name: str, value: bool) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a boolean, got {type(value)}.")


@struct.dataclass
class RNNDiscreteConfig:
    num_classes: int = struct.field(pytree_node=False)
    num_layers: int = struct.field(pytree_node=False)
    hidden_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)

    num_output_classes: Optional[int] = struct.field(default=None, pytree_node=False)
    use_bias: bool = struct.field(default=False, pytree_node=False)

    def __post_init__(self):
        for name in ("num_classes", "num_layers", "hidden_dim", "state_dim"):
            value = _normalize_count(name, getattr(self, name))
            object.__setattr__(self, name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.num_output_classes is not None:
            num_output_classes = _normalize_count("num_output_classes", self.num_output_classes)
            object.__setattr__(self, "num_output_classes", num_output_classes)
            if num_output_classes <= 0:
                raise ValueError(f"num_output_classes must be positive, got {num_output_classes}.")
        object.__setattr__(self, "use_bias", _normalize_bool("use_bias", self.use_bias))

    @property
    def output_classes(self) -> int:
        return self.num_classes if self.num_output_classes is None else self.num_output_classes


class RNNDiscrete(Module):
    """A Simple RNN where inputs are discrete classes and outputs are single-class
    probability distributions."""

    config: RNNDiscreteConfig = field(kw_only=True)
    dtype: Any = jnp.bfloat16
    act_fn: Any = jax.nn.silu

    def setup(self):
        self.embed = Embed(
            num_embeddings=self.config.num_classes,
            features=self.config.hidden_dim,
            dtype=self.dtype,
        )
        layers = []
        for layer_idx in range(self.config.num_layers):
            kernel_axes = ("embed", "state") if layer_idx == 0 else ("state", "state")
            layers.append(
                DenseGeneral(
                    features=self.config.state_dim,
                    axis=-1,
                    weight_dtype=self.dtype,
                    dtype=self.dtype,
                    kernel_axes=kernel_axes,
                    use_bias=self.config.use_bias,
                )
            )
        self.layers = tuple(layers)
        self.norm = RMSNorm(self.config.state_dim, dtype=self.dtype)
        self.output = DenseGeneral(
            features=self.config.output_classes,
            axis=-1,
            weight_dtype=self.dtype,
            dtype=self.dtype,
            kernel_axes=("state", "classes"),
            use_bias=self.config.use_bias,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
    ):
        out = self.embed(input_ids).astype(self.dtype)
        for layer in self.layers:
            out = layer(out)
            out = self.norm(out)
            out = self.act_fn(out)

        return self.output(out)
