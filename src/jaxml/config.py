import math
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

    use_alibi: bool = struct.field(default=False, pytree_node=False)
    use_rope: bool = struct.field(default=True, pytree_node=False)
    # Only effective when using RoPE
    rope_theta: int = struct.field(default=10_000, pytree_node=False)

    def __post_init__(self):
        if self.use_alibi and self.use_rope:
            raise ValueError("AliBi and RoPE cannot both be used.")

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

    @classmethod
    def from_hf(cls, config):
        """Construct a ModelConfig from LlamaConfig."""
        factor = math.gcd(config.intermediate_size, config.hidden_size)
        return cls(
            head_dim=config.hidden_size // config.num_attention_heads,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            vocab_size=config.vocab_size,
            intermediate_ratio=(config.intermediate_size // factor, config.hidden_size // factor),
            norm_eps=config.rms_norm_eps,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
        )
