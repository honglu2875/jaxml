from typing import Optional

from flax import struct


@struct.dataclass
class ModelConfig:
    head_dim: int = struct.field(pytree_node=False)
    num_heads: int = struct.field(pytree_node=False)
    num_layers: int = struct.field(pytree_node=False)
    max_position_embeddings: int = struct.field(pytree_node=False)
    vocab_size: int = struct.field(pytree_node=False)

    # record numerator and denominator separately to avoid errors
    intermediate_ratio: tuple[int, int] = struct.field(default=(8, 3), pytree_node=False)
    norm_eps: float = struct.field(default=1e-6, pytree_node=False)

    num_kv_heads: Optional[int] = struct.field(default=None, pytree_node=False)
    sliding_window: Optional[int] = struct.field(default=None, pytree_node=False)

    use_bias: bool = struct.field(default=False, pytree_node=False)
    # Only effective when using ALiBi
    upcast_alibi: bool = struct.field(default=True, pytree_node=False)

    @property
    def num_key_value_heads(self):
        if self.num_kv_heads is None:
            return self.num_heads
        return self.num_kv_heads

    @property
    def hidden_size(self):
        return self.head_dim * self.num_heads

    @property
    def intermediate_size(self):
        return self.hidden_size * self.intermediate_ratio[0] // self.intermediate_ratio[1]
