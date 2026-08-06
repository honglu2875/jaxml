import re
from collections.abc import Mapping
from operator import index
from os import PathLike, fspath
from typing import Literal

import numpy as np

from jaxml.models.deepseek import DeepseekModelWithHead
from jaxml.models.gemma3 import GemmaModelWithHead
from jaxml.models.gpt_neox import GPTNeoXModelWithHead
from jaxml.models.llama import LlamaModelWithHead

from .utils import _resolve_np_dtype, _state_value_to_numpy, torch_to_jax_states


def _normalize_hf_count(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {type(value)}.")
    try:
        normalized = index(value)
    except TypeError as e:
        raise TypeError(f"{name} must be an integer, got {type(value)}.") from e
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _config_head_dim(config):
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return _normalize_hf_count("head_dim", head_dim)

    hidden_size = _normalize_hf_count("hidden_size", config.hidden_size)
    num_attention_heads = _normalize_hf_count("num_attention_heads", config.num_attention_heads)
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads when head_dim is not set.")
    return hidden_size // num_attention_heads


def _hf_model_config(model, architecture_name: str):
    try:
        return model.config
    except AttributeError as e:
        raise ValueError(f"Expected Hugging Face {architecture_name} model to expose a config.") from e


def _hf_model_state_dict(model, architecture_name: str) -> Mapping:
    state_dict_fn = getattr(model, "state_dict", None)
    if not callable(state_dict_fn):
        raise TypeError(f"Expected Hugging Face {architecture_name} model to expose a callable state_dict method.")
    state_dict = state_dict_fn()
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Hugging Face {architecture_name} state_dict must be a mapping, got {type(state_dict)}.")
    return state_dict


def to_llama_jax_params(model, dtype: str = "float16"):
    llama_config = _hf_model_config(model, "Llama")
    return torch_to_jax_states(_hf_model_state_dict(model, "Llama"), head_dim=_config_head_dim(llama_config), dtype=dtype)


def to_mistral_jax_params(model, dtype: str = "float16"):
    # Mistral shares Llama's parameter naming, so the Llama conversion applies unchanged.
    mistral_config = _hf_model_config(model, "Mistral")
    return torch_to_jax_states(_hf_model_state_dict(model, "Mistral"), head_dim=_config_head_dim(mistral_config), dtype=dtype)


def to_gemma_jax_params(model, dtype: str = "bfloat16"):
    gemma_config = _hf_model_config(model, "Gemma")
    return torch_to_jax_states(_hf_model_state_dict(model, "Gemma"), head_dim=_config_head_dim(gemma_config), dtype=dtype)


def to_neox_jax_params(model, dtype: str = "float16"):
    neox_config = _hf_model_config(model, "GPT-NeoX")
    # the substitution rules may be fragile, but we specifically target HF's neox
    _sub = {
        r"embed_in(.*)": r"embed_tokens\1",
        r"final_layer_norm(.*)": r"norm\1",
        r"embed_out(.*)": r"lm_head\1",
        r"attention\.(.*)": r"self_attn.\1",
        r"query_key_value(.*)": r"qkv_proj\1",
        r"dense\.(.*)": r"o_proj.\1",
        r"dense_h_to_4h(.*)": r"up_proj\1",
        r"dense_4h_to_h(.*)": r"down_proj\1",
    }
    state_dict = {}
    state_sources = {}
    for key, value in _hf_model_state_dict(model, "GPT-NeoX").items():
        name = key
        for k, v in _sub.items():
            name = re.sub(k, v, name)
        if name in state_dict:
            raise ValueError(
                "Multiple Hugging Face GPT-NeoX state keys map to the same jaxml destination "
                f"{name!r}: {state_sources[name]!r} and {key!r}."
            )
        state_dict[name] = value
        state_sources[name] = key
    return torch_to_jax_states(state_dict, head_dim=_config_head_dim(neox_config), dtype=dtype)


def to_deepseek_jax_params(model, dtype: str = "bfloat16"):
    """Convert an HF DeepseekV3ForCausalLM into the jaxml DeepseekModelWithHead tree.

    DeepSeek-V3 needs a bespoke converter: the MLA projections reshape into
    (head, head_dim) kernels, and the routed experts are stacked 3D tensors with
    a fused gate/up projection (none of which the generic converter handles).
    """
    from jaxml.config import ModelConfig

    cfg = ModelConfig.from_hf(_hf_model_config(model, "DeepSeek"))
    state_dict = _hf_model_state_dict(model, "DeepSeek")
    np_dtype = _resolve_np_dtype(dtype)

    def g(key):
        return _state_value_to_numpy(state_dict[key], np_dtype, key)

    nh, qk = cfg.num_heads, cfg.qk_head_dim
    nope, v = cfg.qk_nope_head_dim, cfg.v_head_dim

    def attn_params(prefix):
        attn = {}
        if cfg.q_lora_rank is None:
            attn["q_proj"] = {"kernel": g(prefix + "q_proj.weight").T.reshape(-1, nh, qk)}
        else:
            attn["q_a_proj"] = {"kernel": g(prefix + "q_a_proj.weight").T}
            attn["q_a_layernorm"] = {"weight": g(prefix + "q_a_layernorm.weight")}
            attn["q_b_proj"] = {"kernel": g(prefix + "q_b_proj.weight").T.reshape(-1, nh, qk)}
        attn["kv_a_proj_with_mqa"] = {"kernel": g(prefix + "kv_a_proj_with_mqa.weight").T}
        attn["kv_a_layernorm"] = {"weight": g(prefix + "kv_a_layernorm.weight")}
        attn["kv_b_proj"] = {"kernel": g(prefix + "kv_b_proj.weight").T.reshape(-1, nh, nope + v)}
        attn["o_proj"] = {"kernel": g(prefix + "o_proj.weight").T}
        return attn

    def mlp_params(prefix, is_moe):
        if not is_moe:
            return {
                "gate_proj": {"kernel": g(prefix + "gate_proj.weight").T},
                "up_proj": {"kernel": g(prefix + "up_proj.weight").T},
                "down_proj": {"kernel": g(prefix + "down_proj.weight").T},
            }
        return {
            "gate_kernel": g(prefix + "gate.weight").T,
            "e_score_correction_bias": g(prefix + "gate.e_score_correction_bias"),
            "gate_up_kernel": g(prefix + "experts.gate_up_proj").transpose(0, 2, 1),
            "down_kernel": g(prefix + "experts.down_proj").transpose(0, 2, 1),
            "shared_gate_proj": {"kernel": g(prefix + "shared_experts.gate_proj.weight").T},
            "shared_up_proj": {"kernel": g(prefix + "shared_experts.up_proj.weight").T},
            "shared_down_proj": {"kernel": g(prefix + "shared_experts.down_proj.weight").T},
        }

    model_params = {"embed_tokens": {"embedding": g("model.embed_tokens.weight")}}
    for idx in range(cfg.num_layers):
        layer_prefix = f"model.layers.{idx}."
        model_params[f"layers_{idx}"] = {
            "self_attn": attn_params(layer_prefix + "self_attn."),
            "mlp": mlp_params(layer_prefix + "mlp.", idx >= cfg.first_k_dense_replace),
            "input_layernorm": {"weight": g(layer_prefix + "input_layernorm.weight")},
            "post_attention_layernorm": {"weight": g(layer_prefix + "post_attention_layernorm.weight")},
        }
    model_params["norm"] = {"weight": g("model.norm.weight")}

    return {"params": {"model": model_params, "lm_head": {"kernel": g("lm_head.weight").T}}}


HFArchitecture = Literal["auto", "llama", "mistral", "deepseek", "deepseek_v3", "gpt_neox", "neox", "gemma3", "gemma"]


def _normalize_hf_architecture(architecture: str) -> str:
    if not isinstance(architecture, str):
        raise TypeError(f"architecture must be a string, got {type(architecture)}.")
    normalized = architecture.lower().replace("-", "_")
    match normalized:
        case "auto" | "llama" | "mistral" | "neox" | "gemma":
            return normalized
        case "deepseek" | "deepseek_v3":
            return "deepseek"
        case "gpt_neox":
            return "neox"
        case "gemma3" | "gemma_3":
            return "gemma"
        case _:
            raise ValueError(f"Unsupported Hugging Face architecture {architecture!r}.")


def _normalize_hf_model_name(name: str | PathLike[str]) -> str:
    try:
        normalized = fspath(name)
    except TypeError as e:
        raise TypeError(f"name must be a string or path-like object, got {type(name)}.") from e
    if not isinstance(normalized, str):
        raise TypeError(f"name must resolve to a string path, got {type(normalized)}.")
    if not normalized or not normalized.strip():
        raise ValueError("name must be non-empty.")
    return normalized


def _validate_hf_dtype(dtype: str):
    _resolve_np_dtype(dtype)
    return dtype


def _validate_hf_model_type(config, expected_model_types: tuple[str, ...], architecture_name: str):
    model_type = getattr(config, "model_type", None)
    if model_type not in expected_model_types:
        raise ValueError(
            f"Expected Hugging Face {architecture_name} model_type to be one of {expected_model_types}, got {model_type!r}."
        )


def _infer_hf_architecture(name: str, **config_kwargs) -> str:
    try:
        from transformers import AutoConfig
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    config = AutoConfig.from_pretrained(name, **config_kwargs)
    model_type = getattr(config, "model_type", None)
    match model_type:
        case "llama":
            return "llama"
        case "mistral":
            return "mistral"
        case "deepseek_v3":
            return "deepseek"
        case "gpt_neox":
            return "neox"
        case "gemma3":
            return "gemma"
        case "gemma3_text":
            return "gemma"
        case _:
            raise ValueError(f"Unsupported Hugging Face model type {model_type!r}.")


def load_model_from_hf(
    name: str,
    architecture: HFArchitecture = "auto",
    dtype: str = "float32",
    **from_pretrained_kwargs,
):
    """Load a supported Hugging Face causal LM into a jaxml model and parameter tree."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    architecture = _normalize_hf_architecture(architecture)
    if architecture == "auto":
        architecture = _infer_hf_architecture(name, **from_pretrained_kwargs)

    match architecture:
        case "llama":
            return load_llama_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "mistral":
            return load_mistral_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "deepseek" | "deepseek_v3":
            return load_deepseek_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "gpt_neox" | "neox":
            return load_neox_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "gemma3" | "gemma":
            return load_gemma_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case _:
            raise ValueError(f"Unsupported Hugging Face architecture {architecture!r}.")


def load_llama_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[LlamaModelWithHead, dict]:
    """Load Huggingface llama compatible models directly from either local path
    or the hf-hub identifier."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig

    _model = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
    _config = _hf_model_config(_model, "Llama")
    _validate_hf_model_type(_config, ("llama",), "Llama")
    params = to_llama_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_config)
    model = LlamaModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params


def load_mistral_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[LlamaModelWithHead, dict]:
    """Load Huggingface Mistral compatible models. Mistral reuses the Llama model definition."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig

    _model = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
    _config = _hf_model_config(_model, "Mistral")
    _validate_hf_model_type(_config, ("mistral",), "Mistral")
    params = to_mistral_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_config)
    model = LlamaModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params


def load_deepseek_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[DeepseekModelWithHead, dict]:
    """Load a Hugging Face DeepSeek-V3 / R1 model into a jaxml DeepseekModelWithHead."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig

    _model = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
    _config = _hf_model_config(_model, "DeepSeek")
    _validate_hf_model_type(_config, ("deepseek_v3",), "DeepSeek")
    params = to_deepseek_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_config)
    model = DeepseekModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params


def load_neox_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[GPTNeoXModelWithHead, dict]:
    """Load Huggingface gpt-neox compatible models directly from either local path
    or the hf-hub identifier."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig

    _model = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
    _config = _hf_model_config(_model, "GPT-NeoX")
    _validate_hf_model_type(_config, ("gpt_neox",), "GPT-NeoX")
    params = to_neox_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_config)
    model = GPTNeoXModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params


def load_gemma_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[GemmaModelWithHead, dict]:
    """Load Huggingface gemma3 compatible models directly from either local path
    or the hf-hub identifier."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig

    _wrapper = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
    _wrapper_config = _hf_model_config(_wrapper, "Gemma")
    _validate_hf_model_type(_wrapper_config, ("gemma3",), "Gemma")
    try:
        _model = _wrapper.language_model
    except AttributeError as e:
        raise ValueError("Expected Hugging Face Gemma model to expose a language_model module.") from e
    _model_config = _hf_model_config(_model, "Gemma text")
    _validate_hf_model_type(_model_config, ("gemma3_text",), "Gemma text")
    params = to_gemma_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_model_config)
    model = GemmaModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params
