import re
from operator import index
from os import PathLike, fspath
from typing import Literal

import numpy as np

from jaxml.models.gemma3 import GemmaModelWithHead
from jaxml.models.gpt_neox import GPTNeoXModelWithHead
from jaxml.models.llama import LlamaModelWithHead

from .utils import _resolve_np_dtype, torch_to_jax_states


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


def to_llama_jax_params(model, dtype: str = "float16"):
    llama_config = model.config
    return torch_to_jax_states(model.state_dict(), head_dim=_config_head_dim(llama_config), dtype=dtype)


def to_gemma_jax_params(model, dtype: str = "bfloat16"):
    gemma_config = model.config
    return torch_to_jax_states(model.state_dict(), head_dim=_config_head_dim(gemma_config), dtype=dtype)


def to_neox_jax_params(model, dtype: str = "float16"):
    neox_config = model.config
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
    for key, value in model.state_dict().items():
        name = key
        for k, v in _sub.items():
            name = re.sub(k, v, name)
        state_dict[name] = value
    return torch_to_jax_states(state_dict, head_dim=_config_head_dim(neox_config), dtype=dtype)


HFArchitecture = Literal["auto", "llama", "gpt_neox", "neox", "gemma3", "gemma"]


def _normalize_hf_architecture(architecture: str) -> str:
    if not isinstance(architecture, str):
        raise TypeError(f"architecture must be a string, got {type(architecture)}.")
    normalized = architecture.lower().replace("-", "_")
    match normalized:
        case "auto" | "llama" | "neox" | "gemma":
            return normalized
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
    if not normalized:
        raise ValueError("name must be non-empty.")
    return normalized


def _validate_hf_dtype(dtype: str):
    _resolve_np_dtype(dtype)
    return dtype


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
        case "gpt_neox":
            return "neox"
        case "gemma3":
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
    params = to_llama_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_model.config)
    model = LlamaModelWithHead(config)
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
    params = to_neox_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_model.config)
    model = GPTNeoXModelWithHead(config)
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

    _model = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs).language_model
    params = to_gemma_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_model.config)
    model = GemmaModelWithHead(config)
    return model, params
