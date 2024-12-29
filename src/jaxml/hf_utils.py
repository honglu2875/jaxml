import re

from .utils import torch_to_jax_states


def to_llama_jax_params(model, dtype: str = "float16"):
    llama_config = model.config
    return torch_to_jax_states(model.state_dict(), head_dim=llama_config.head_dim, dtype=dtype)


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


def load_llama_from_hf(name: str, dtype: str = "float32") -> tuple["LlamaForCausalLM", dict]:  # noqa: F821
    """Load Huggingface llama compatible models directly from either local path
    or the hf-hub identifier."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig
    from jaxml.models.llama import LlamaModelWithHead

    _model = AutoModelForCausalLM.from_pretrained(name)
    params = to_llama_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_model.config)
    model = LlamaModelWithHead(config)
    return model, params


def load_neox_from_hf(name: str, dtype: str = "float32") -> tuple["GPTNeoXForCausalLM", dict]:  # noqa: F821
    """Load Huggingface llama compatible models directly from either local path
    or the hf-hub identifier."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig
    from jaxml.models.gpt_neox import GPTNeoXModelWithHead

    _model = AutoModelForCausalLM.from_pretrained(name)
    params = to_neox_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_model.config)
    model = GPTNeoXModelWithHead(config)
    return model, params


