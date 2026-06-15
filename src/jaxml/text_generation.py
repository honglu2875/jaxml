import math
import numbers
import operator
from collections.abc import Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from jaxml.hf_utils import (
    HFArchitecture,
    _normalize_hf_architecture,
    _normalize_hf_model_name,
    _validate_hf_dtype,
    load_model_from_hf,
)
from jaxml.inference_engine.engine import Engine, InferenceConfig, _normalize_dtype, _normalize_inference_config


def _normalize_count(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {type(value)}.")
    try:
        return operator.index(value)
    except TypeError as e:
        raise TypeError(f"{name} must be an integer, got {type(value)}.") from e


def _normalize_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value)}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return value


def _normalize_bool(name: str, value: bool) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean, got {type(value)}.")
    return bool(value)


def _normalize_optional_kwargs(name: str, value: Optional[dict[str, Any]]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping when set, got {type(value)}.")
    return dict(value)


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 0
    max_new_tokens: int = 100
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    temperature: float = 1.0
    fuse_decoding: bool = False
    include_prompt: bool = True

    def __post_init__(self):
        seed = _normalize_count("seed", self.seed)
        max_new_tokens = _normalize_count("max_new_tokens", self.max_new_tokens)
        top_k = _normalize_count("top_k", self.top_k)
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}.")
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}.")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "max_new_tokens", max_new_tokens)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "top_p", _normalize_float("top_p", self.top_p))
        object.__setattr__(self, "min_p", _normalize_float("min_p", self.min_p))
        object.__setattr__(self, "temperature", _normalize_float("temperature", self.temperature))
        object.__setattr__(self, "fuse_decoding", _normalize_bool("fuse_decoding", self.fuse_decoding))
        object.__setattr__(self, "include_prompt", _normalize_bool("include_prompt", self.include_prompt))


def _normalize_generation_config(generation_config: Optional[GenerationConfig]) -> GenerationConfig:
    if generation_config is None:
        return GenerationConfig()
    if not isinstance(generation_config, GenerationConfig):
        raise TypeError(f"generation_config must be a GenerationConfig when set, got {type(generation_config)}.")
    return generation_config


def _normalize_generation_options(
    generation_config: Optional[GenerationConfig],
    generation_kwargs: dict[str, Any],
) -> dict[str, Any]:
    config = _normalize_generation_config(generation_config)
    values = {
        "seed": config.seed,
        "max_new_tokens": config.max_new_tokens,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "min_p": config.min_p,
        "temperature": config.temperature,
        "fuse_decoding": config.fuse_decoding,
        "include_prompt": config.include_prompt,
    } | generation_kwargs
    config = GenerationConfig(**values)
    return {
        "seed": config.seed,
        "max_new_tokens": config.max_new_tokens,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "min_p": config.min_p,
        "temperature": config.temperature,
        "fuse_decoding": config.fuse_decoding,
        "include_prompt": config.include_prompt,
    }


def _normalize_tokenizer_arrays(input_ids, attention_mask):
    input_ids = jnp.asarray(input_ids)
    if input_ids.ndim != 2:
        raise ValueError(f"tokenizer input_ids must be a 2D array, got shape {input_ids.shape}.")
    if not jnp.issubdtype(input_ids.dtype, jnp.integer):
        raise TypeError(f"tokenizer input_ids must contain integer token ids, got dtype {input_ids.dtype}.")
    if input_ids.shape[0] == 0:
        raise ValueError("tokenizer input_ids must contain at least one batch row.")
    if input_ids.shape[1] == 0:
        raise ValueError("tokenizer input_ids must contain at least one token.")

    if attention_mask is None:
        return input_ids, None

    attention_mask = jnp.asarray(attention_mask)
    if attention_mask.ndim != 2:
        raise ValueError(f"tokenizer attention_mask must be a 2D array, got shape {attention_mask.shape}.")
    if not (jnp.issubdtype(attention_mask.dtype, jnp.bool_) or jnp.issubdtype(attention_mask.dtype, jnp.integer)):
        raise TypeError(f"tokenizer attention_mask must be boolean or integer, got dtype {attention_mask.dtype}.")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"tokenizer attention_mask shape must match input_ids shape; got {attention_mask.shape} and {input_ids.shape}."
        )
    attention_mask = attention_mask.astype(bool)
    if not bool(jnp.all(jnp.any(attention_mask, axis=1))):
        raise ValueError("tokenizer attention_mask must contain at least one valid token per batch row.")
    return input_ids, attention_mask


def _normalize_generated_tokens(tokens) -> np.ndarray:
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError(f"engine.generate must return a 2D token array, got shape {tokens.shape}.")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise TypeError(f"engine.generate must return integer token ids, got dtype {tokens.dtype}.")
    return tokens


def _normalize_decoded_text(decoded, batch_size: int) -> list[str]:
    if isinstance(decoded, (str, bytes)) or not isinstance(decoded, SequenceABC):
        raise TypeError(f"tokenizer.batch_decode must return a sequence of strings, got {type(decoded)}.")
    decoded = list(decoded)
    decoded_batch_size = len(decoded)
    if decoded_batch_size != batch_size:
        raise ValueError(
            f"tokenizer.batch_decode output batch size must match token batch size; got {decoded_batch_size} and {batch_size}."
        )
    if not all(isinstance(text, str) for text in decoded):
        raise TypeError("tokenizer.batch_decode must return a sequence of strings.")
    return decoded


@dataclass
class TextGenerationPipeline:
    engine: Engine
    tokenizer: Any
    default_tokenize_kwargs: dict[str, Any] = field(default_factory=lambda: {"padding": True})
    default_decode_kwargs: dict[str, Any] = field(default_factory=lambda: {"skip_special_tokens": True})

    def __post_init__(self):
        if not callable(getattr(self.engine, "generate", None)):
            raise TypeError("engine must provide a callable generate method.")
        if not callable(self.tokenizer):
            raise TypeError(f"tokenizer must be callable, got {type(self.tokenizer)}.")
        if not callable(getattr(self.tokenizer, "batch_decode", None)):
            raise TypeError("tokenizer must provide a callable batch_decode method.")
        self.default_tokenize_kwargs = _normalize_optional_kwargs("default_tokenize_kwargs", self.default_tokenize_kwargs)
        self.default_decode_kwargs = _normalize_optional_kwargs("default_decode_kwargs", self.default_decode_kwargs)

    @classmethod
    def from_hf(
        cls,
        name: str | PathLike[str],
        architecture: HFArchitecture = "auto",
        model_dtype: str = "float32",
        engine_dtype: Any = jnp.float32,
        inference_config: Optional[InferenceConfig] = None,
        use_tpu: bool = False,
        cache_stride: int = 256,
        tokenizer: Any = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Build a text-generation pipeline from a supported Hugging Face checkpoint."""
        name = _normalize_hf_model_name(name)
        architecture = _normalize_hf_architecture(architecture)
        model_dtype = _validate_hf_dtype(model_dtype)
        engine_dtype = _normalize_dtype("engine_dtype", engine_dtype)
        inference_config = InferenceConfig() if inference_config is None else _normalize_inference_config(inference_config)
        use_tpu = _normalize_bool("use_tpu", use_tpu)
        cache_stride = _normalize_count("cache_stride", cache_stride)
        if cache_stride <= 0:
            raise ValueError(f"cache_stride must be positive, got {cache_stride}.")
        tokenizer_kwargs = _normalize_optional_kwargs("tokenizer_kwargs", tokenizer_kwargs)
        model_kwargs = _normalize_optional_kwargs("model_kwargs", model_kwargs)
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise ImportError("Please install transformers library.") from e
            tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)

        model, params = load_model_from_hf(
            name,
            architecture=architecture,
            dtype=model_dtype,
            **model_kwargs,
        )
        engine = Engine(
            model,
            inference_config,
            params,
            dtype=engine_dtype,
            cache_stride=cache_stride,
        )
        engine.init_params(use_tpu=use_tpu)
        return cls(engine=engine, tokenizer=tokenizer)

    def _encode(self, prompts: str | Sequence[str], tokenize_kwargs: Optional[dict[str, Any]] = None):
        is_single_prompt = isinstance(prompts, str)
        if is_single_prompt:
            prompt_batch = [prompts]
        else:
            if not isinstance(prompts, SequenceABC):
                raise TypeError(f"prompts must be a string or a sequence of strings, got {type(prompts)}.")
            prompt_batch = list(prompts)
        if not prompt_batch:
            raise ValueError("prompts must contain at least one prompt.")
        if not all(isinstance(prompt, str) for prompt in prompt_batch):
            raise TypeError("prompts must be a string or a sequence of strings.")
        kwargs = self.default_tokenize_kwargs | _normalize_optional_kwargs("tokenize_kwargs", tokenize_kwargs)
        encoded = self.tokenizer(prompt_batch, return_tensors="np", **kwargs)
        input_ids = self._get_encoded_field(encoded, "input_ids")
        attention_mask = self._get_encoded_field(encoded, "attention_mask", default=None)
        input_ids, attention_mask = _normalize_tokenizer_arrays(input_ids, attention_mask)
        return is_single_prompt, input_ids, attention_mask

    @staticmethod
    def _get_encoded_field(encoded: Any, name: str, default: Any = ...):
        if isinstance(encoded, dict):
            if default is ...:
                return encoded[name]
            return encoded.get(name, default)
        if default is ...:
            return getattr(encoded, name)
        return getattr(encoded, name, default)

    def generate_tokens(
        self,
        prompts: str | Sequence[str],
        generation_config: Optional[GenerationConfig] = None,
        tokenize_kwargs: Optional[dict[str, Any]] = None,
        **generation_kwargs,
    ) -> np.ndarray:
        generation_kwargs = _normalize_generation_options(generation_config, generation_kwargs)
        _, input_ids, attention_mask = self._encode(prompts, tokenize_kwargs=tokenize_kwargs)
        return self._generate_tokens_from_arrays(
            input_ids,
            attention_mask,
            **generation_kwargs,
        )

    def _generate_tokens_from_arrays(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> np.ndarray:
        kwargs = _normalize_generation_options(generation_config, generation_kwargs)
        input_ids, attention_mask = _normalize_tokenizer_arrays(input_ids, attention_mask)
        tokens = _normalize_generated_tokens(self.engine.generate(input_ids, attention_mask=attention_mask, **kwargs))
        if tokens.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "engine.generate token batch size must match input batch size; "
                f"got {tokens.shape[0]} and {input_ids.shape[0]}."
            )
        return tokens

    def generate_text(
        self,
        prompts: str | Sequence[str],
        generation_config: Optional[GenerationConfig] = None,
        tokenize_kwargs: Optional[dict[str, Any]] = None,
        decode_kwargs: Optional[dict[str, Any]] = None,
        **generation_kwargs,
    ) -> str | list[str]:
        generation_kwargs = _normalize_generation_options(generation_config, generation_kwargs)
        decode_kwargs = _normalize_optional_kwargs("decode_kwargs", decode_kwargs)
        is_single_prompt, input_ids, attention_mask = self._encode(prompts, tokenize_kwargs=tokenize_kwargs)
        tokens = self._generate_tokens_from_arrays(
            input_ids,
            attention_mask,
            **generation_kwargs,
        )
        decoded = _normalize_decoded_text(
            self.tokenizer.batch_decode(tokens, **(self.default_decode_kwargs | decode_kwargs)),
            batch_size=tokens.shape[0],
        )
        return decoded[0] if is_single_prompt else decoded
