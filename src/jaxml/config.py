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
    use_alibi: bool = struct.field(default=False, pytree_node=False)
    use_rope: bool = struct.field(default=True, pytree_node=False)

    ####### Only effective for ALiBi #######
    upcast_alibi: bool = struct.field(default=True, pytree_node=False)

    ####### Only effective for RoPE #######
    rope_theta: float = struct.field(default=10_000.0, pytree_node=False)
    # Limit the percentage of heads to apply RoPE (NeoX style)
    rotary_pct: float = struct.field(default=1.0, pytree_node=False)
    # Only effective in GPT-NeoX for the two arch variants
    use_parallel_residual: bool = struct.field(default=True, pytree_node=False)

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
        """Construct a ModelConfig from LlamaConfig or GPTNeoXConfig."""
        from transformers import GPTNeoXConfig, LlamaConfig

        factor = math.gcd(config.intermediate_size, config.hidden_size)

        ####### Shared params #######
        head_dim = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers
        max_position_embeddings = config.max_position_embeddings
        vocab_size = config.vocab_size
        intermediate_ratio = (config.intermediate_size // factor, config.hidden_size // factor)

        ####### Case-by-case #######
        if type(config) is LlamaConfig:
            norm_eps = config.rms_norm_eps
            num_kv_heads = config.num_key_value_heads
            rope_theta = config.rope_theta
            # no impact
            use_parallel_residual, rotary_pct = True, 1.0
            use_bias = False
        elif type(config) is GPTNeoXConfig:
            norm_eps = config.layer_norm_eps
            num_kv_heads = num_heads
            rope_theta = float(config.rotary_emb_base)
            use_parallel_residual = config.use_parallel_residual
            rotary_pct = config.rotary_pct
            use_bias = True
        else:
            raise ValueError(f"Unsupported config class {config.__class__}")

        return cls(
            head_dim=head_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            intermediate_ratio=intermediate_ratio,
            norm_eps=norm_eps,
            num_kv_heads=num_kv_heads,
            use_rope=True,
            rope_theta=rope_theta,
            use_parallel_residual=use_parallel_residual,
            rotary_pct=rotary_pct,
            use_bias=use_bias,
        )
