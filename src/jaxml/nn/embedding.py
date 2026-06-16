import operator
from dataclasses import field
from typing import Any, Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jax import lax

from .._validation import contains_tracer as _contains_tracer
from .module import Module


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


def _normalize_dtype(name: str, value: Any):
    if value is None:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.")
    try:
        return jnp.dtype(value)
    except TypeError as e:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.") from e


class Embed(Module):
    """A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    kernel_axes: tuple with axes to apply the embedding on.
    one_hot: performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
    """

    # Required:
    num_embeddings: int = field(kw_only=True)
    features: int = field(kw_only=True)

    dtype: Any = jnp.float32

    # Embed: Space(vocab) -> Space(embed)
    kernel_axes: Tuple[str, ...] = ("vocab", "embed")
    one_hot: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        num_embeddings = _normalize_count("num_embeddings", self.num_embeddings)
        features = _normalize_count("features", self.features)
        one_hot = _normalize_bool("one_hot", self.one_hot)
        dtype = _normalize_dtype("dtype", self.dtype)
        if num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {num_embeddings}.")
        if features <= 0:
            raise ValueError(f"features must be positive, got {features}.")

        inputs = jnp.asarray(inputs)
        if inputs.size == 0:
            raise ValueError("Embed inputs must contain at least one token id.")

        embedding = self.param(
            "embedding",
            self.wrapped_kernel_init,
            (num_embeddings, features),
            dtype,
            0,
            1,
        )

        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        inputs_are_traced = _contains_tracer(inputs)
        valid_inputs = (inputs >= 0) & (inputs < num_embeddings)
        if not inputs_are_traced:
            if bool(jnp.any(inputs < 0)):
                raise ValueError("Input token ids must be non-negative.")
            if bool(jnp.any(inputs >= num_embeddings)):
                raise ValueError(f"Input token ids must be less than num_embeddings={num_embeddings}.")
        if one_hot:
            iota = lax.iota(jnp.int32, num_embeddings)
            one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=dtype)
            output = jnp.dot(one_hot, jnp.asarray(embedding, dtype))
        else:
            safe_inputs = jnp.where(valid_inputs, inputs, 0)
            output = jnp.asarray(embedding, dtype)[safe_inputs]
            if inputs_are_traced:
                output = jnp.where(valid_inputs[..., None], output, 0)
        return output
