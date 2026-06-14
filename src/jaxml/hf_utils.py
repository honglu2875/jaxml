import re
from typing import TYPE_CHECKING, Literal

from jaxml.models.gemma3 import GemmaModelWithHead
from jaxml.models.gpt_neox import GPTNeoXModelWithHead
from jaxml.models.llama import LlamaModelWithHead

from .utils import torch_to_jax_states

if TYPE_CHECKING:
    from transformers import GemmaForCausalLM, GPTNeoXForCausalLM, LlamaForCausalLM


def to_llama_jax_params(model, dtype: str = "float16"):
    llama_config = model.config
    return torch_to_jax_states(model.state_dict(), head_dim=llama_config.head_dim, dtype=dtype)


def to_gemma_jax_params(model, dtype: str = "bfloat16"):
    gemma_config = model.config
    return torch_to_jax_states(model.state_dict(), head_dim=gemma_config.head_dim, dtype=dtype)


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
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        name = key
        for k, v in _sub.items():
            name = re.sub(k, v, name)
        if name != key:
            state_dict[name] = state_dict.pop(key)
    return torch_to_jax_states(state_dict, head_dim=neox_config.hidden_size // neox_config.num_attention_heads, dtype=dtype)


HFArchitecture = Literal["auto", "llama", "gpt_neox", "neox", "gemma3", "gemma"]


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
