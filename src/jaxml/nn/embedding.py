from dataclasses import field
from typing import Any, Tuple

import flax.linen as nn
import jax.numpy as jnp
from jax import lax

from .module import Module


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
        embedding = self.param(
            "embedding",
            self.wrapped_kernel_init,
            (self.num_embeddings, self.features),
            self.dtype,
            0,
            1,
        )

        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        if self.one_hot:
            iota = lax.iota(jnp.int32, self.num_embeddings)
            one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
            output = jnp.dot(one_hot, jnp.asarray(embedding, self.dtype))
        else:
            output = jnp.asarray(embedding, self.dtype)[inputs]
        return output
