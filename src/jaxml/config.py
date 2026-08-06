import math
import operator
from collections.abc import Mapping
from dataclasses import fields
from typing import Optional

import numpy as np
from flax import struct

from ._validation import normalize_float as _normalize_float


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


def _infer_hf_rope_parameters(config, attention_kind: str | None = None) -> Mapping:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is None:
        return {}
    if not isinstance(rope_parameters, Mapping):
        raise TypeError(f"rope_parameters must be a mapping when set, got {type(rope_parameters)}.")
    if attention_kind is None:
        return rope_parameters
    attention_parameters = rope_parameters.get(attention_kind, {})
    if not isinstance(attention_parameters, Mapping):
        raise TypeError(f"rope_parameters[{attention_kind!r}] must be a mapping when set, got {type(attention_parameters)}.")
    return attention_parameters


def _infer_hf_rope_theta(config, *, fallback_attr: str = "rope_theta", attention_kind: str | None = None) -> float:
    rope_parameters = _infer_hf_rope_parameters(config, attention_kind=attention_kind)
    if "rope_theta" in rope_parameters:
        return _normalize_float("rope_theta", rope_parameters["rope_theta"])
    if not hasattr(config, fallback_attr):
        raise ValueError(f"{config.__class__.__name__} must expose rope_theta.")
    return _normalize_float("rope_theta", getattr(config, fallback_attr))


def _infer_hf_rope_scale(config, *, attention_kind: str | None = None) -> float:
    def _validate_supported_rope_type(parameters: Mapping, source: str):
        rope_type = parameters.get("rope_type", parameters.get("type"))
        if rope_type not in (None, "default", "linear"):
            raise NotImplementedError(
                f"{source} rope_type={rope_type!r} is not supported; jaxml currently supports only default and linear RoPE."
            )
        return rope_type

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None and not isinstance(rope_scaling, Mapping):
        raise TypeError(f"rope_scaling must be a mapping when set, got {type(rope_scaling)}.")
    if isinstance(rope_scaling, Mapping):
        rope_scaling_for_attention = rope_scaling
        if attention_kind is not None and attention_kind in rope_scaling:
            rope_scaling_for_attention = rope_scaling[attention_kind]
            if not isinstance(rope_scaling_for_attention, Mapping):
                raise TypeError(
                    f"rope_scaling[{attention_kind!r}] must be a mapping when set, got {type(rope_scaling_for_attention)}."
                )
        rope_type = _validate_supported_rope_type(rope_scaling_for_attention, "rope_scaling")
        if "factor" in rope_scaling_for_attention:
            return _normalize_float("rope_scaling factor", rope_scaling_for_attention["factor"])
        if rope_type == "linear":
            raise ValueError("rope_scaling must include a factor when set.")

    rope_parameters = _infer_hf_rope_parameters(config, attention_kind=attention_kind)
    rope_type = _validate_supported_rope_type(rope_parameters, "rope_parameters")
    if "factor" in rope_parameters:
        return _normalize_float("rope_parameters factor", rope_parameters["factor"])
    if rope_type == "linear":
        raise ValueError("rope_parameters must include a factor when set.")
    return 1.0


def _infer_hf_mla_rope_parameters(config) -> dict:
    """Normalize the default/YaRN metadata used by MLA model families.

    Recent Transformers configs expose ``rope_parameters`` while older remote
    model configs (notably Kimi-K2) expose the same data as ``rope_scaling``.
    Keep this adapter local to MLA: classic model families still map linear
    scaling to ``rope_scale`` through :func:`_infer_hf_rope_scale`.
    """
    parameters = getattr(config, "rope_parameters", None)
    if parameters is None:
        parameters = getattr(config, "rope_scaling", None)
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise TypeError(f"MLA rope parameters must be a mapping when set, got {type(parameters)}.")

    rope_type = parameters.get("rope_type", parameters.get("type", "default"))
    if not isinstance(rope_type, str):
        raise TypeError(f"rope_type must be a string, got {type(rope_type)}.")
    rope_type = rope_type.lower()
    if rope_type not in ("default", "yarn"):
        raise NotImplementedError(
            f"MLA rope_type={rope_type!r} is not supported; jaxml currently supports default and YaRN RoPE."
        )

    partial_rotary_factor = parameters.get("partial_rotary_factor", 1.0)
    partial_rotary_factor = _normalize_float("partial_rotary_factor", partial_rotary_factor)
    if partial_rotary_factor != 1.0:
        raise NotImplementedError(
            "MLA configs with partial_rotary_factor != 1.0 are not supported; "
            "qk_rope_head_dim already specifies the decoupled rotary width."
        )

    rope_theta = parameters.get("rope_theta", getattr(config, "rope_theta", None))
    if rope_theta is None:
        raise ValueError(f"{config.__class__.__name__} must expose rope_theta in its RoPE parameters or config.")

    factor = parameters.get("factor", 1.0)
    original_max = parameters.get(
        "original_max_position_embeddings",
        getattr(config, "original_max_position_embeddings", None),
    )
    if rope_type == "yarn" and original_max is None:
        raise ValueError("YaRN rope parameters must include original_max_position_embeddings.")
    if factor is None:
        factor = config.max_position_embeddings / original_max

    truncate = parameters.get("truncate", True)
    truncate = _normalize_bool("rope truncate", truncate)
    return {
        "rope_type": rope_type,
        "rope_theta": _normalize_float("rope_theta", rope_theta),
        "rope_factor": _normalize_float("rope factor", factor),
        "rope_original_max_position_embeddings": original_max,
        "rope_beta_fast": _normalize_float(
            "rope beta_fast",
            32.0 if parameters.get("beta_fast") is None else parameters["beta_fast"],
        ),
        "rope_beta_slow": _normalize_float(
            "rope beta_slow",
            1.0 if parameters.get("beta_slow") is None else parameters["beta_slow"],
        ),
        "rope_mscale": parameters.get("mscale"),
        "rope_mscale_all_dim": parameters.get("mscale_all_dim"),
        "rope_attention_factor": parameters.get("attention_factor"),
        "rope_truncate": truncate,
    }


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _infer_hf_mlp_schedule(config, num_layers: int) -> tuple[int, int]:
    """Return ``(dense_prefix, sparse_frequency)`` for a supported MoE schedule."""
    mlp_layer_types = getattr(config, "mlp_layer_types", None)
    if mlp_layer_types is None:
        return (
            _normalize_count("first_k_dense_replace", getattr(config, "first_k_dense_replace", 0)),
            _normalize_count("moe_layer_freq", getattr(config, "moe_layer_freq", 1)),
        )

    if isinstance(mlp_layer_types, str):
        raise TypeError("mlp_layer_types must be a sequence of 'dense'/'sparse' entries, not a string.")
    try:
        layer_types = tuple(str(value).lower() for value in mlp_layer_types)
    except TypeError as e:
        raise TypeError("mlp_layer_types must be an iterable of 'dense'/'sparse' entries.") from e
    if len(layer_types) != num_layers:
        raise ValueError(f"mlp_layer_types must contain one entry per layer, got {len(layer_types)} for {num_layers} layers.")
    if any(value not in ("dense", "sparse") for value in layer_types):
        raise ValueError(f"mlp_layer_types entries must be 'dense' or 'sparse', got {layer_types}.")

    first_sparse = next((idx for idx, value in enumerate(layer_types) if value == "sparse"), num_layers)
    expected = ("dense",) * first_sparse + ("sparse",) * (num_layers - first_sparse)
    if layer_types != expected:
        raise NotImplementedError(
            "Only a contiguous dense-prefix/sparse-suffix mlp_layer_types schedule is currently supported."
        )
    return first_sparse, 1


def _infer_hf_rotary_pct(config) -> float:
    rotary_pct = getattr(config, "rotary_pct", None)
    if rotary_pct is not None:
        return _normalize_float("rotary_pct", rotary_pct)
    rope_parameters = _infer_hf_rope_parameters(config)
    if "partial_rotary_factor" in rope_parameters:
        return _normalize_float("rotary_pct", rope_parameters["partial_rotary_factor"])
    return 1.0


def _validate_rope_rotary_dim(head_dim: int, rotary_pct: float) -> int:
    rotary_dim = head_dim if rotary_pct >= 1.0 else int(head_dim * rotary_pct)
    if rotary_dim <= 0:
        raise ValueError(
            f"rotary_pct must produce a positive rotary dimension, got rotary_pct={rotary_pct} and head_dim={head_dim}."
        )
    if rotary_dim % 2:
        raise ValueError(
            f"rotary_pct must produce an even rotary dimension, got rotary_dim={rotary_dim}, "
            f"rotary_pct={rotary_pct}, and head_dim={head_dim}."
        )
    return rotary_dim


@struct.dataclass
class ModelConfig:
    head_dim: int = struct.field(pytree_node=False)
    # in Gemma3, hidden_size is different than num_heads*head_dim!
    hidden_size: int = struct.field(pytree_node=False)
    num_heads: int = struct.field(pytree_node=False)
    num_layers: int = struct.field(pytree_node=False)
    max_position_embeddings: int = struct.field(pytree_node=False)
    vocab_size: int = struct.field(pytree_node=False)
    # Source architecture after adapting an external config. Generic hand-built
    # configs keep the default and remain fully backward compatible.
    architecture: str = struct.field(default="generic", pytree_node=False)

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
    # Advanced RoPE metadata used by MLA models such as Kimi-K2. ``rope_scale``
    # remains the simple linear-scaling factor used by classic RoPE models.
    rope_type: str = struct.field(default="default", pytree_node=False)
    rope_factor: float = struct.field(default=1.0, pytree_node=False)
    rope_original_max_position_embeddings: Optional[int] = struct.field(default=None, pytree_node=False)
    rope_beta_fast: float = struct.field(default=32.0, pytree_node=False)
    rope_beta_slow: float = struct.field(default=1.0, pytree_node=False)
    rope_mscale: Optional[float] = struct.field(default=None, pytree_node=False)
    rope_mscale_all_dim: Optional[float] = struct.field(default=None, pytree_node=False)
    rope_attention_factor: Optional[float] = struct.field(default=None, pytree_node=False)
    rope_truncate: bool = struct.field(default=True, pytree_node=False)
    # Limit the percentage of heads to apply RoPE (NeoX style)
    rotary_pct: float = struct.field(default=1.0, pytree_node=False)
    # Only effective in GPT-NeoX for the two arch variants
    use_parallel_residual: bool = struct.field(default=True, pytree_node=False)

    # --- DeepSeek-V3 MLA (Multi-head Latent Attention). All None for non-MLA models. ---
    # When q_lora_rank is None the query uses a single projection (no low-rank compression).
    q_lora_rank: Optional[int] = struct.field(default=None, pytree_node=False)
    kv_lora_rank: Optional[int] = struct.field(default=None, pytree_node=False)
    qk_nope_head_dim: Optional[int] = struct.field(default=None, pytree_node=False)
    qk_rope_head_dim: Optional[int] = struct.field(default=None, pytree_node=False)
    v_head_dim: Optional[int] = struct.field(default=None, pytree_node=False)
    # Kimi Linear keeps the nominal RoPE slice as an ordinary NoPE feature.
    mla_use_nope: bool = struct.field(default=False, pytree_node=False)

    # --- Kimi Delta Attention. Layer indices are normalized to zero-based. ---
    kda_layers: tuple[int, ...] = struct.field(default=(), pytree_node=False)
    kda_num_heads: Optional[int] = struct.field(default=None, pytree_node=False)
    kda_head_dim: Optional[int] = struct.field(default=None, pytree_node=False)
    kda_short_conv_kernel_size: int = struct.field(default=4, pytree_node=False)

    # --- DeepSeek-V3 fine-grained MoE. n_routed_experts is None for dense models. ---
    n_routed_experts: Optional[int] = struct.field(default=None, pytree_node=False)
    n_shared_experts: Optional[int] = struct.field(default=None, pytree_node=False)
    moe_intermediate_size: Optional[int] = struct.field(default=None, pytree_node=False)
    num_experts_per_tok: Optional[int] = struct.field(default=None, pytree_node=False)
    n_group: Optional[int] = struct.field(default=None, pytree_node=False)
    topk_group: Optional[int] = struct.field(default=None, pytree_node=False)
    routed_scaling_factor: float = struct.field(default=1.0, pytree_node=False)
    norm_topk_prob: bool = struct.field(default=True, pytree_node=False)
    # Layers with index < first_k_dense_replace use a dense MLP instead of MoE.
    first_k_dense_replace: int = struct.field(default=0, pytree_node=False)
    # After the dense prefix, one out of every ``moe_layer_freq`` layers is sparse.
    moe_layer_freq: int = struct.field(default=1, pytree_node=False)

    def __post_init__(self):
        for name in (
            "use_bias",
            "use_alibi",
            "use_rope",
            "upcast_alibi",
            "use_parallel_residual",
            "rope_truncate",
            "mla_use_nope",
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
        if self.hidden_size * intermediate_ratio[0] // intermediate_ratio[1] <= 0:
            raise ValueError(f"intermediate_ratio must produce a positive intermediate_size, got {intermediate_ratio}.")
        if not isinstance(self.rope_type, str):
            raise TypeError(f"rope_type must be a string, got {type(self.rope_type)}.")
        rope_type = self.rope_type.lower()
        object.__setattr__(self, "rope_type", rope_type)
        if rope_type not in ("default", "yarn"):
            raise NotImplementedError(f"rope_type={rope_type!r} is not supported for ModelConfig.")

        for name in ("norm_eps", "rope_theta", "rope_scale", "rotary_pct", "rope_factor", "rope_beta_fast", "rope_beta_slow"):
            object.__setattr__(self, name, _normalize_float(name, getattr(self, name)))
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}.")

        if self.num_kv_heads is not None:
            object.__setattr__(self, "num_kv_heads", _normalize_count("num_kv_heads", self.num_kv_heads))
        if self.num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {self.num_key_value_heads}.")
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_key_value_heads, " f"got {self.num_heads} and {self.num_key_value_heads}."
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
        if (self.sliding_window is None) != (self.sliding_window_pattern is None):
            raise ValueError("sliding_window and sliding_window_pattern must both be set or both be None.")
        if self.sliding_window_pattern is not None and self.sliding_window_pattern <= 0:
            raise ValueError(f"sliding_window_pattern must be positive when set, got {self.sliding_window_pattern}.")
        if self.attn_scale is None:
            object.__setattr__(self, "attn_scale", self.head_dim**-0.5)
        else:
            object.__setattr__(self, "attn_scale", _normalize_float("attn_scale", self.attn_scale))
            if self.attn_scale <= 0:
                raise ValueError(f"attn_scale must be positive, got {self.attn_scale}.")

        if self.use_alibi and self.use_rope:
            raise ValueError("AliBi and RoPE cannot both be used.")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}.")
        if self.rope_scale < 1.0:
            raise ValueError(f"rope_scale must be greater than or equal to 1.0, got {self.rope_scale}.")
        if self.rope_factor < 1.0:
            raise ValueError(f"rope_factor must be greater than or equal to 1.0, got {self.rope_factor}.")
        for name in ("rope_mscale", "rope_mscale_all_dim", "rope_attention_factor"):
            value = getattr(self, name)
            if value is not None:
                value = _normalize_float(name, value)
                object.__setattr__(self, name, value)
                if name == "rope_attention_factor" and value <= 0:
                    raise ValueError(f"{name} must be positive when set, got {value}.")
                if name != "rope_attention_factor" and value < 0:
                    raise ValueError(f"{name} must be non-negative when set, got {value}.")
        if self.rope_original_max_position_embeddings is not None:
            original_max = _normalize_count(
                "rope_original_max_position_embeddings", self.rope_original_max_position_embeddings
            )
            object.__setattr__(self, "rope_original_max_position_embeddings", original_max)
            if original_max <= 0:
                raise ValueError("rope_original_max_position_embeddings must be positive when set, " f"got {original_max}.")
        if self.rope_type == "yarn" and self.rope_original_max_position_embeddings is None:
            raise ValueError("rope_original_max_position_embeddings is required for YaRN.")
        if self.rope_type == "yarn" and self.rope_theta <= 1.0:
            raise ValueError(f"YaRN rope_theta must be greater than 1.0, got {self.rope_theta}.")
        if self.rope_type == "yarn" and (self.rope_beta_fast <= 0 or self.rope_beta_slow <= 0):
            raise ValueError(
                "YaRN rope_beta_fast and rope_beta_slow must be positive, "
                f"got {self.rope_beta_fast} and {self.rope_beta_slow}."
            )
        if not 0 < self.rotary_pct <= 1:
            raise ValueError(f"rotary_pct must be in (0, 1], got {self.rotary_pct}.")

        # DeepSeek-V3 MLA / MoE fields. Each is optional; validate only when set.
        for name in (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "n_routed_experts",
            "n_shared_experts",
            "moe_intermediate_size",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "kda_num_heads",
            "kda_head_dim",
        ):
            value = getattr(self, name)
            if value is not None:
                value = _normalize_count(name, value)
                object.__setattr__(self, name, value)
                if value <= 0:
                    raise ValueError(f"{name} must be positive when set, got {value}.")

        if not isinstance(self.architecture, str):
            raise TypeError(f"architecture must be a string, got {type(self.architecture)}.")
        architecture = self.architecture.lower()
        object.__setattr__(self, "architecture", architecture)
        supported_architectures = {
            "generic",
            "llama",
            "mistral",
            "deepseek",
            "gpt_neox",
            "gemma3",
            "glm4_moe_lite",
            "kimi_k2",
            "kimi_linear",
        }
        if architecture not in supported_architectures:
            raise ValueError(f"Unsupported architecture {architecture!r}.")

        try:
            kda_layers = tuple(_normalize_count("kda layer index", idx) for idx in self.kda_layers)
        except TypeError as e:
            raise TypeError("kda_layers must be an iterable of zero-based layer indices.") from e
        object.__setattr__(self, "kda_layers", kda_layers)
        if len(set(kda_layers)) != len(kda_layers):
            raise ValueError(f"kda_layers must not contain duplicates, got {kda_layers}.")
        if any(idx < 0 or idx >= self.num_layers for idx in kda_layers):
            raise ValueError(f"kda_layers entries must be in [0, {self.num_layers}), got {kda_layers}.")
        object.__setattr__(
            self,
            "kda_short_conv_kernel_size",
            _normalize_count("kda_short_conv_kernel_size", self.kda_short_conv_kernel_size),
        )
        if self.kda_short_conv_kernel_size <= 0:
            raise ValueError(f"kda_short_conv_kernel_size must be positive, got {self.kda_short_conv_kernel_size}.")
        if kda_layers and (self.kda_num_heads is None or self.kda_head_dim is None):
            raise ValueError("KDA configs must set kda_num_heads and kda_head_dim when kda_layers is non-empty.")

        object.__setattr__(self, "norm_topk_prob", _normalize_bool("norm_topk_prob", self.norm_topk_prob))
        object.__setattr__(
            self, "routed_scaling_factor", _normalize_float("routed_scaling_factor", self.routed_scaling_factor)
        )
        object.__setattr__(
            self, "first_k_dense_replace", _normalize_count("first_k_dense_replace", self.first_k_dense_replace)
        )
        object.__setattr__(self, "moe_layer_freq", _normalize_count("moe_layer_freq", self.moe_layer_freq))
        if self.first_k_dense_replace < 0:
            raise ValueError(f"first_k_dense_replace must be non-negative, got {self.first_k_dense_replace}.")
        if self.first_k_dense_replace > self.num_layers:
            raise ValueError(
                f"first_k_dense_replace ({self.first_k_dense_replace}) cannot exceed num_layers ({self.num_layers})."
            )
        if self.moe_layer_freq <= 0:
            raise ValueError(f"moe_layer_freq must be positive, got {self.moe_layer_freq}.")
        if self.is_mla:
            required_mla = {
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
            }
            missing = tuple(name for name, value in required_mla.items() if value is None)
            if missing:
                raise ValueError(f"MLA configs with kv_lora_rank set must also set {', '.join(missing)}.")
            if self.qk_rope_head_dim % 2:
                raise ValueError(f"qk_rope_head_dim must be even, got {self.qk_rope_head_dim}.")
        if self.is_moe:
            if self.n_group is not None and self.n_routed_experts % self.n_group != 0:
                raise ValueError(f"n_routed_experts ({self.n_routed_experts}) must be divisible by n_group ({self.n_group}).")
            if self.topk_group is not None and self.n_group is not None and self.topk_group > self.n_group:
                raise ValueError(f"topk_group ({self.topk_group}) cannot exceed n_group ({self.n_group}).")

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank is not None

    @property
    def is_moe(self) -> bool:
        return self.n_routed_experts is not None

    @property
    def qk_head_dim(self) -> Optional[int]:
        if self.qk_nope_head_dim is None or self.qk_rope_head_dim is None:
            return None
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` uses the routed MoE feed-forward block."""
        layer_idx = _normalize_count("layer_idx", layer_idx)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}.")
        return self.is_moe and layer_idx >= self.first_k_dense_replace and layer_idx % self.moe_layer_freq == 0

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` uses Kimi Delta Attention."""
        layer_idx = _normalize_count("layer_idx", layer_idx)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}.")
        return layer_idx in self.kda_layers

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
        from transformers import (
            DeepseekV3Config,
            Gemma3TextConfig,
            Glm4MoeLiteConfig,
            GPTNeoXConfig,
            LlamaConfig,
            MistralConfig,
        )

        model_type = getattr(config, "model_type", None)
        if model_type == "kimi_linear":
            # Kimi Linear uses a remote-code config and mixes KDA with NoPE MLA.
            config_type = "kimi_linear"
        elif model_type == "kimi_k2":
            # Kimi-K2 intentionally reuses the DeepSeek-V3 text architecture,
            # but its remote-code config is not a DeepseekV3Config subclass.
            config_type = "kimi_k2"
        elif isinstance(config, LlamaConfig):
            config_type = "llama"
        elif isinstance(config, MistralConfig):
            config_type = "mistral"
        elif isinstance(config, DeepseekV3Config):
            config_type = "deepseek"
        elif isinstance(config, Glm4MoeLiteConfig):
            config_type = "glm4_moe_lite"
        elif isinstance(config, GPTNeoXConfig):
            config_type = "gpt_neox"
        elif isinstance(config, Gemma3TextConfig):
            config_type = "gemma3"
        else:
            raise ValueError(f"Unsupported config class {config.__class__}")

        # Shared params.
        # hidden_size is guaranteed to exist in HF config
        hidden_size = _normalize_count("hidden_size", config.hidden_size)
        intermediate_size = _normalize_count("intermediate_size", config.intermediate_size)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}.")
        factor = math.gcd(intermediate_size, hidden_size)

        # head_dim is usually hidden_size // num_attention_heads, but it can specify a different number
        head_dim = _infer_hf_head_dim(config)
        num_heads = _normalize_count("num_attention_heads", config.num_attention_heads)
        num_layers = _normalize_count("num_hidden_layers", config.num_hidden_layers)
        max_position_embeddings_value = getattr(
            config,
            "max_position_embeddings",
            getattr(config, "model_max_length", None),
        )
        if max_position_embeddings_value is None:
            raise ValueError(f"{config.__class__.__name__} must expose max_position_embeddings or model_max_length.")
        max_position_embeddings = _normalize_count("max_position_embeddings", max_position_embeddings_value)
        vocab_size = _normalize_count("vocab_size", config.vocab_size)
        for name, value in (
            ("num_attention_heads", num_heads),
            ("num_hidden_layers", num_layers),
            ("max_position_embeddings", max_position_embeddings),
            ("vocab_size", vocab_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        intermediate_ratio = (intermediate_size // factor, hidden_size // factor)
        attn_scale = head_dim**-0.5

        # DeepSeek-V3 MLA / MoE fields. None / disabled for every other architecture.
        q_lora_rank = kv_lora_rank = qk_nope_head_dim = qk_rope_head_dim = v_head_dim = None
        n_routed_experts = n_shared_experts = moe_intermediate_size = None
        num_experts_per_tok = n_group = topk_group = None
        routed_scaling_factor, norm_topk_prob, first_k_dense_replace = 1.0, True, 0
        moe_layer_freq = 1
        rope_type, rope_factor = "default", 1.0
        rope_original_max_position_embeddings = None
        rope_beta_fast, rope_beta_slow = 32.0, 1.0
        rope_mscale = rope_mscale_all_dim = rope_attention_factor = None
        rope_truncate = True
        mla_use_nope = False
        kda_layers = ()
        kda_num_heads = kda_head_dim = None
        kda_short_conv_kernel_size = 4

        # Case-by-case.
        if config_type == "llama":
            norm_eps = config.rms_norm_eps
            num_kv_heads = _normalize_count("num_key_value_heads", config.num_key_value_heads)
            if num_kv_heads <= 0:
                raise ValueError(f"num_key_value_heads must be positive, got {num_kv_heads}.")
            rope_theta = _infer_hf_rope_theta(config)
            rope_scale = _infer_hf_rope_scale(config)
            # no impact
            use_parallel_residual, rotary_pct = True, 1.0
            use_bias = False
            sliding_window = None
            sliding_window_pattern = None
        elif config_type == "mistral":
            # Mistral is architecturally Llama (RMSNorm, SwiGLU, GQA, RoPE, no bias).
            norm_eps = config.rms_norm_eps
            num_kv_heads = _normalize_count("num_key_value_heads", config.num_key_value_heads)
            if num_kv_heads <= 0:
                raise ValueError(f"num_key_value_heads must be positive, got {num_kv_heads}.")
            rope_theta = _infer_hf_rope_theta(config)
            rope_scale = _infer_hf_rope_scale(config)
            use_parallel_residual, rotary_pct = True, 1.0
            use_bias = False
            # The Llama decode path does not apply sliding-window masking, so only
            # full-attention Mistral configs (sliding_window=None) can be loaded faithfully.
            if getattr(config, "sliding_window", None) is not None:
                raise NotImplementedError(
                    "Mistral sliding-window attention is not yet supported; "
                    "only configs with sliding_window=None can be loaded."
                )
            sliding_window = None
            sliding_window_pattern = None
        elif config_type == "gpt_neox":
            norm_eps = config.layer_norm_eps
            num_kv_heads = num_heads
            rope_theta = _infer_hf_rope_theta(config, fallback_attr="rotary_emb_base")
            rope_scale = 1.0  # NeoX was born before RoPE scaling
            use_parallel_residual = config.use_parallel_residual
            rotary_pct = _infer_hf_rotary_pct(config)
            use_bias = True
            sliding_window = None
            sliding_window_pattern = None
        elif config_type == "gemma3":
            query_pre_attn_scalar = _normalize_float("query_pre_attn_scalar", config.query_pre_attn_scalar)
            if query_pre_attn_scalar <= 0:
                raise ValueError(f"query_pre_attn_scalar must be positive, got {query_pre_attn_scalar}.")
            attn_scale = query_pre_attn_scalar**-0.5
            norm_eps = config.rms_norm_eps
            num_kv_heads = _normalize_count("num_key_value_heads", config.num_key_value_heads)
            if num_kv_heads <= 0:
                raise ValueError(f"num_key_value_heads must be positive, got {num_kv_heads}.")
            rope_theta = _infer_hf_rope_theta(config, attention_kind="full_attention")
            rope_scale = _infer_hf_rope_scale(config, attention_kind="full_attention")
            # no impact
            use_parallel_residual, rotary_pct = True, 1.0
            use_bias = False
            sliding_window = config.sliding_window
            sliding_window_pattern = config.sliding_window_pattern
        elif config_type in ("deepseek", "glm4_moe_lite", "kimi_k2", "kimi_linear"):
            # DeepSeek-V3, GLM-4.7-Flash, and Kimi-K2 share MLA attention and
            # fine-grained MoE. The latter two only require config adapters.
            if getattr(config, "rope_interleave", True) is False:
                raise NotImplementedError("MLA configs with rope_interleave=False are not supported.")
            if bool(getattr(config, "attention_bias", False)):
                raise NotImplementedError("MLA configs with attention_bias=True are not yet supported.")
            if getattr(config, "hidden_act", "silu") not in ("silu", "swiglu"):
                raise NotImplementedError("MLA/MoE configs currently require a SiLU/SwiGLU activation.")
            scoring_func = getattr(
                config,
                "scoring_func",
                getattr(config, "moe_router_activation_func", "sigmoid"),
            )
            if scoring_func != "sigmoid":
                raise NotImplementedError("MLA/MoE configs currently require sigmoid expert routing.")
            if getattr(config, "topk_method", "noaux_tc") != "noaux_tc":
                raise NotImplementedError("MLA/MoE configs currently require noaux_tc expert selection.")

            mla_rope = _infer_hf_mla_rope_parameters(config)
            rope_type = mla_rope["rope_type"]
            rope_theta = mla_rope["rope_theta"]
            rope_factor = mla_rope["rope_factor"]
            rope_original_max_position_embeddings = mla_rope["rope_original_max_position_embeddings"]
            rope_beta_fast = mla_rope["rope_beta_fast"]
            rope_beta_slow = mla_rope["rope_beta_slow"]
            rope_mscale = mla_rope["rope_mscale"]
            rope_mscale_all_dim = mla_rope["rope_mscale_all_dim"]
            rope_attention_factor = mla_rope["rope_attention_factor"]
            rope_truncate = mla_rope["rope_truncate"]

            if config_type == "kimi_linear":
                mla_use_nope = _normalize_bool("mla_use_nope", getattr(config, "mla_use_nope", False))
                if not mla_use_nope:
                    raise NotImplementedError("Kimi Linear currently requires mla_use_nope=True.")
                linear_attn_config = getattr(config, "linear_attn_config", None)
                if not isinstance(linear_attn_config, Mapping):
                    raise ValueError("Kimi Linear config must include a linear_attn_config mapping.")
                kda_one_based = tuple(linear_attn_config.get("kda_layers", ()))
                full_one_based = tuple(linear_attn_config.get("full_attn_layers", ()))
                kda_layers = tuple(_normalize_count("KDA layer", idx) - 1 for idx in kda_one_based)
                full_layers = tuple(_normalize_count("full-attention layer", idx) - 1 for idx in full_one_based)
                if set(kda_layers) & set(full_layers):
                    raise ValueError("Kimi Linear KDA and full-attention layer lists must not overlap.")
                if set(kda_layers) | set(full_layers) != set(range(num_layers)):
                    raise ValueError("Kimi Linear KDA and full-attention layer lists must partition every model layer.")
                kda_num_heads = _normalize_count("KDA num_heads", linear_attn_config.get("num_heads"))
                kda_head_dim = _normalize_count("KDA head_dim", linear_attn_config.get("head_dim"))
                kda_short_conv_kernel_size = _normalize_count(
                    "KDA short_conv_kernel_size",
                    linear_attn_config.get("short_conv_kernel_size"),
                )

            qk_nope_head_dim = _normalize_count("qk_nope_head_dim", config.qk_nope_head_dim)
            qk_rope_head_dim = _normalize_count("qk_rope_head_dim", config.qk_rope_head_dim)
            v_head_dim = _normalize_count("v_head_dim", config.v_head_dim)
            kv_lora_rank = _normalize_count("kv_lora_rank", config.kv_lora_rank)
            q_lora_rank = None if config.q_lora_rank is None else _normalize_count("q_lora_rank", config.q_lora_rank)
            # MLA query/key head dim is the concatenation of the nope and rope parts.
            head_dim = qk_nope_head_dim + qk_rope_head_dim
            attn_scale = head_dim**-0.5
            if rope_type == "yarn" and rope_mscale_all_dim:
                mscale = _yarn_get_mscale(rope_factor, _normalize_float("rope mscale_all_dim", rope_mscale_all_dim))
                attn_scale *= mscale * mscale
            norm_eps = config.rms_norm_eps
            num_kv_heads = num_heads  # MLA materializes one k/v per query head.
            rope_scale = 1.0
            use_parallel_residual, rotary_pct = True, 1.0
            use_bias = False
            sliding_window = None
            sliding_window_pattern = None
            n_routed_experts = _normalize_count(
                "n_routed_experts",
                getattr(config, "n_routed_experts", getattr(config, "num_experts", None)),
            )
            n_shared_experts = _normalize_count(
                "n_shared_experts",
                getattr(config, "n_shared_experts", getattr(config, "num_shared_experts", None)),
            )
            moe_intermediate_size = _normalize_count("moe_intermediate_size", config.moe_intermediate_size)
            num_experts_per_tok = _normalize_count(
                "num_experts_per_tok",
                getattr(config, "num_experts_per_tok", getattr(config, "num_experts_per_token", None)),
            )
            n_group = _normalize_count(
                "n_group",
                getattr(config, "n_group", getattr(config, "num_expert_group", None)),
            )
            topk_group = _normalize_count("topk_group", config.topk_group)
            routed_scaling_factor = _normalize_float("routed_scaling_factor", config.routed_scaling_factor)
            norm_topk_prob = _normalize_bool(
                "norm_topk_prob",
                getattr(config, "norm_topk_prob", getattr(config, "moe_renormalize", None)),
            )
            first_k_dense_replace, moe_layer_freq = _infer_hf_mlp_schedule(config, num_layers)

        _validate_rope_rotary_dim(head_dim, rotary_pct)

        return cls(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            architecture=config_type,
            intermediate_ratio=intermediate_ratio,
            norm_eps=norm_eps,
            num_kv_heads=num_kv_heads,
            use_rope=not mla_use_nope,
            rope_theta=rope_theta,
            use_parallel_residual=use_parallel_residual,
            rotary_pct=rotary_pct,
            use_bias=use_bias,
            sliding_window=sliding_window,
            sliding_window_pattern=sliding_window_pattern,
            rope_scale=rope_scale,
            rope_type=rope_type,
            rope_factor=rope_factor,
            rope_original_max_position_embeddings=rope_original_max_position_embeddings,
            rope_beta_fast=rope_beta_fast,
            rope_beta_slow=rope_beta_slow,
            rope_mscale=rope_mscale,
            rope_mscale_all_dim=rope_mscale_all_dim,
            rope_attention_factor=rope_attention_factor,
            rope_truncate=rope_truncate,
            attn_scale=attn_scale,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mla_use_nope=mla_use_nope,
            kda_layers=kda_layers,
            kda_num_heads=kda_num_heads,
            kda_head_dim=kda_head_dim,
            kda_short_conv_kernel_size=kda_short_conv_kernel_size,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            norm_topk_prob=norm_topk_prob,
            first_k_dense_replace=first_k_dense_replace,
            moe_layer_freq=moe_layer_freq,
        )
