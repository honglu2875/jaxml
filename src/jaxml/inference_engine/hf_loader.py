from jaxml.hf_utils import to_gemma_jax_params, to_llama_jax_params, to_neox_jax_params
from jaxml.inference_engine.engine import Engine, InferenceConfig
from jaxml.models.gemma3 import GemmaModelWithHead
from jaxml.models.gpt_neox import GPTNeoXModelWithHead
from jaxml.models.llama import LlamaModelWithHead
from jaxml.config import ModelConfig
from typing import Any


def load_hf(name: str, tp_size: int = 4) -> tuple[Engine, Any]:
    """A convenience function to load a Hugging Face model into an Engine class.

    Return:
        A tuple containing:
            - An Engine instance with the model and parameters loaded.
            - The tokenizer for the model.
    """
    try:
        from transformers import AutoModelForCausalLM, LlamaForCausalLM, GPTNeoXForCausalLM, Gemma3ForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e


    hf_model = AutoModelForCausalLM.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)

    if isinstance(hf_model, LlamaForCausalLM):
        params = to_llama_jax_params(hf_model)
        config = ModelConfig.from_hf(hf_model.config)
        model = LlamaModelWithHead(config)
    elif isinstance(hf_model, Gemma3ForCausalLM):
        params = to_gemma_jax_params(hf_model.language_model)
        config = ModelConfig.from_hf(hf_model.language_model.config)
        model = GemmaModelWithHead(config)
    elif isinstance(hf_model, GPTNeoXForCausalLM):
        params = to_neox_jax_params(hf_model)
        config = ModelConfig.from_hf(hf_model.config)
        model = GPTNeoXModelWithHead(config)
    else:
        raise ValueError(f"Unsupported model type: {type(hf_model)}")

    # Create the Engine instance
    inference_config = InferenceConfig(tp_size=tp_size)
    engine = Engine(model, inference_config, params, dtype="bfloat16")

    return engine, tokenizer

