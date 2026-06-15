import math
import operator
from dataclasses import fields
from typing import Optional

import numpy as np
from flax import struct


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


def _infer_hf_head_dim(config) -> int:
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        head_dim = _normalize_count("head_dim", head_dim)
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}.")
        return head_dim

    hidden_size = _normalize_count("hidden_size", config.hidden_size)
    num_attention_heads = _normalize_count("num_attention_heads", config.num_attention_heads)
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
    if num_attention_heads <= 0:
        raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}.")
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads when head_dim is not set.")
    return hidden_size // num_attention_heads


@struct.dataclass
class ModelConfig:
    head_dim: int = struct.field(pytree_node=False)
    # in Gemma3, hidden_size is different than num_heads*head_dim!
    hidden_size: int = struct.field(pytree_node=False)
    num_heads: int = struct.field(pytree_node=False)
    num_layers: int = struct.field(pytree_node=False)
    max_position_embeddings: int = struct.field(pytree_node=False)
    vocab_size: int = struct.field(pytree_node=False)

    # record numerator and denominator separately to avoid errors
    intermediate_ratio: tuple[int, int] = struct.field(default=(8, 3), pytree_node=False)
    norm_eps: float = struct.field(default=1e-6, pytree_node=False)

    num_kv_heads: Optional[int] = struct.field(default=None, pytree_node=False)
    sliding_window: Optional[int] = struct.field(default=None, pytree_node=False)
    sliding_window_pattern: Optional[int] = struct.field(default=None, pytree_node=False)
    # most of the time it is just head_dim ** -0.5, but just in case
    attn_scale: float = struct.field(default=None, pytree_node=False)

    use_bias: bool = struct.field(default=False, pytree_node=False)
    use_alibi: bool = struct.field(default=False, pytree_node=False)
    use_rope: bool = struct.field(default=True, pytree_node=False)

    # Only effective for ALiBi.
    upcast_alibi: bool = struct.field(default=True, pytree_node=False)

    # Only effective for RoPE.
    rope_theta: float = struct.field(default=10_000.0, pytree_node=False)
    rope_scale: float = struct.field(default=1.0, pytree_node=False)
    # Limit the percentage of heads to apply RoPE (NeoX style)
    rotary_pct: float = struct.field(default=1.0, pytree_node=False)
    # Only effective in GPT-NeoX for the two arch variants
    use_parallel_residual: bool = struct.field(default=True, pytree_node=False)

    def __post_init__(self):
        for name in (
            "use_bias",
            "use_alibi",
            "use_rope",
            "upcast_alibi",
            "use_parallel_residual",
        ):
            object.__setattr__(self, name, _normalize_bool(name, getattr(self, name)))

        for name in (
            "head_dim",
            "hidden_size",
            "num_heads",
            "num_layers",
            "max_position_embeddings",
            "vocab_size",
        ):
            value = _normalize_count(name, getattr(self, name))
            object.__setattr__(self, name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if not hasattr(self.intermediate_ratio, "__len__") or len(self.intermediate_ratio) != 2:
            raise ValueError(f"intermediate_ratio must contain numerator and denominator, got {self.intermediate_ratio}.")
        intermediate_ratio = (
            _normalize_count("intermediate_ratio numerator", self.intermediate_ratio[0]),
            _normalize_count("intermediate_ratio denominator", self.intermediate_ratio[1]),
        )
        object.__setattr__(self, "intermediate_ratio", intermediate_ratio)
        if intermediate_ratio[0] <= 0 or intermediate_ratio[1] <= 0:
            raise ValueError(f"intermediate_ratio values must be positive, got {intermediate_ratio}.")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}.")

        if self.num_kv_heads is not None:
            object.__setattr__(self, "num_kv_heads", _normalize_count("num_kv_heads", self.num_kv_heads))
        if self.num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {self.num_key_value_heads}.")
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_key_value_heads, "
                f"got {self.num_heads} and {self.num_key_value_heads}."
            )

        if self.sliding_window is not None:
            object.__setattr__(self, "sliding_window", _normalize_count("sliding_window", self.sliding_window))
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive when set, got {self.sliding_window}.")
        if self.sliding_window_pattern is not None:
            object.__setattr__(
                self,
                "sliding_window_pattern",
                _normalize_count("sliding_window_pattern", self.sliding_window_pattern),
            )
        if self.sliding_window_pattern is not None and self.sliding_window_pattern <= 0:
            raise ValueError(f"sliding_window_pattern must be positive when set, got {self.sliding_window_pattern}.")
        if self.attn_scale is None:
            object.__setattr__(self, "attn_scale", self.head_dim**-0.5)
        elif self.attn_scale <= 0:
            raise ValueError(f"attn_scale must be positive, got {self.attn_scale}.")

        if self.use_alibi and self.use_rope:
            raise ValueError("AliBi and RoPE cannot both be used.")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}.")
        if self.rope_scale < 1.0:
            raise ValueError(f"rope_scale must be greater than or equal to 1.0, got {self.rope_scale}.")
        if not 0 < self.rotary_pct <= 1:
            raise ValueError(f"rotary_pct must be in (0, 1], got {self.rotary_pct}.")

    def replace(self, **kwargs):
        args_dict = {k.name: getattr(self, k.name) for k in fields(self)} | kwargs
        return self.__class__(**args_dict)

    @property
    def num_key_value_heads(self):
        if self.num_kv_heads is None:
            return self.num_heads
        return self.num_kv_heads

    @property
    def intermediate_size(self):
        return self.hidden_size * self.intermediate_ratio[0] // self.intermediate_ratio[1]

    @classmethod
    def from_hf(cls, config):
        """Construct a ModelConfig from a supported Hugging Face config."""
        from transformers import Gemma3TextConfig, GPTNeoXConfig, LlamaConfig

        if isinstance(config, LlamaConfig):
            config_type = "llama"
        elif isinstance(config, GPTNeoXConfig):
            config_type = "gpt_neox"
        elif isinstance(config, Gemma3TextConfig):
            config_type = "gemma3"
        else:
            raise ValueError(f"Unsupported config class {config.__class__}")

        factor = math.gcd(config.intermediate_size, config.hidden_size)

        # Shared params.
        # hidden_size is guaranteed to exist in HF config
        hidden_size = config.hidden_size
        # head_dim is usually hidden_size // num_attention_heads, but it can specify a different number
        head_dim = _infer_hf_head_dim(config)
        num_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers
        max_position_embeddings = config.max_position_embeddings
        vocab_size = config.vocab_size
        intermediate_ratio = (config.intermediate_size // factor, config.hidden_size // factor)
        attn_scale = head_dim**-0.5

        # Case-by-case.
        if config_type == "llama":
            norm_eps = config.rms_norm_eps
            num_kv_heads = config.num_key_value_heads
            rope_theta = config.rope_theta
            rope_scale = config.rope_scaling["factor"] if config.rope_scaling is not None else 1.0
            # no impact
            use_parallel_residual, rotary_pct = True, 1.0
            use_bias = False
            sliding_window = None
            sliding_window_pattern = None
        elif config_type == "gpt_neox":
            norm_eps = config.layer_norm_eps
            num_kv_heads = num_heads
            rope_theta = float(config.rotary_emb_base)
            rope_scale = 1.0  # NeoX was born before RoPE scaling
            use_parallel_residual = config.use_parallel_residual
            rotary_pct = config.rotary_pct
            use_bias = True
            sliding_window = None
            sliding_window_pattern = None
        elif config_type == "gemma3":
            attn_scale = config.query_pre_attn_scalar**-0.5
            norm_eps = config.rms_norm_eps
            num_kv_heads = config.num_key_value_heads
            rope_theta = config.rope_theta
            rope_scale = config.rope_scaling["factor"] if config.rope_scaling is not None else 1.0
            # no impact
            use_parallel_residual, rotary_pct = True, 1.0
            use_bias = False
            sliding_window = config.sliding_window
            sliding_window_pattern = int(config.sliding_window_pattern)

        return cls(
            hidden_size=hidden_size,
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
            sliding_window=sliding_window,
            sliding_window_pattern=sliding_window_pattern,
            rope_scale=rope_scale,
            attn_scale=attn_scale,
        )
