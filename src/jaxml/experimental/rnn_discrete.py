import flax
import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp


from ..nn.linear import DenseGeneral
from ..nn.embedding import Embed
from .module import Module


class RNNDiscreteConfig:
    num_classes: int = struct.field(pytree_node=False)
    num_layers: int = struct.field(pytree_node=False)
    hidden_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)

    num_output_classes: Optional[int] = struct.field(default=None, pytree_node=False)
    use_bias: bool = struct.field(default=False, pytree_node=False)


class RNNDiscrete(Module):
    """A Simple RNN where inputs are discrete classes and outputs are single-class 
    probability distributions."""

    config: RNNDiscreteConfig
    dtype: Any = jnp.bfloat16
    act_fn: Any = jax.nn.silu

    def setup(self):
        self.embed = Embed(
            num_embedding=self.config.num_classes,
            features=self.config.hidden_dim,
            dtype=self.dtype, 
        )
        self.layers = [
            DenseGeneral(
                features=self.config.hidden_dim,
                axis=-1,
                weight_dtype=self.dtype,
                dtype=self.dtype,
                use_bias=self.config.use_bias,
            ) 
        ]
        self.norm = RMSNorm(self.config.hidden_size, eps=self.norm_eps)
        self.output = DenseGeneral(
            features=self.config.num_classes,
            axis=-1,
            weight_dtype=self.dtype,
            dtype=self.dtype,
            use_bias=self.config.use_bias, 
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
    ):
        out = self.embed_tokens(input_ids).astype(self.dtype)
        for layer in self.layers:
            out = layer(out)
            out = self.norm(out)
            out = self.act_fn(out)

        out = self.output(out)
