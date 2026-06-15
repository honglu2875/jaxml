import operator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..cache import KVCache


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


def _contains_tracer(x: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(x))


def normalize_model_output_flags(
    use_cache: bool,
    output_attentions: bool,
    output_hidden_states: bool,
) -> tuple[bool, bool, bool]:
    return (
        _normalize_bool("use_cache", use_cache),
        _normalize_bool("output_attentions", output_attentions),
        _normalize_bool("output_hidden_states", output_hidden_states),
    )


def validate_attention_hidden_size(config, architecture_name: str):
    expected_hidden_size = config.num_heads * config.head_dim
    if config.hidden_size != expected_hidden_size:
        raise ValueError(
            f"{architecture_name} hidden_size must equal num_heads * head_dim; "
            f"got hidden_size={config.hidden_size}, num_heads={config.num_heads}, head_dim={config.head_dim}."
        )


def validate_mlp_input(module_name: str, x, hidden_size: int) -> jnp.ndarray:
    x = jnp.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"{module_name} input must be a 3D array, got shape {x.shape}.")
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError(f"{module_name} input must contain floating point values, got dtype {x.dtype}.")
    if any(axis_size <= 0 for axis_size in x.shape):
        raise ValueError(f"{module_name} input must not contain empty axes, got shape {x.shape}.")
    if x.shape[-1] != hidden_size:
        raise ValueError(f"{module_name} hidden dimension mismatch: got {x.shape[-1]} and expected {hidden_size}.")
    return x


def slice_last_n_logits_hidden_states(hidden_states: jnp.ndarray, keep_last_n_logits: int) -> jnp.ndarray:
    hidden_states = jnp.asarray(hidden_states)
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be a 3D array, got shape {hidden_states.shape}.")
    if not jnp.issubdtype(hidden_states.dtype, jnp.floating):
        raise TypeError(f"hidden_states must contain floating point values, got dtype {hidden_states.dtype}.")
    if any(axis_size <= 0 for axis_size in hidden_states.shape):
        raise ValueError(f"hidden_states must not contain empty axes, got shape {hidden_states.shape}.")
    keep_last_n_logits = _normalize_count("keep_last_n_logits", keep_last_n_logits)
    if keep_last_n_logits < 0:
        raise ValueError(f"keep_last_n_logits must be non-negative, got {keep_last_n_logits}.")
    if keep_last_n_logits == 0:
        return hidden_states
    return hidden_states[:, -keep_last_n_logits:]


def prepare_model_inputs(input_ids, attention_mask, kv_caches, num_layers: int, vocab_size: int | None = None):
    num_layers = _normalize_count("num_layers", num_layers)
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")
    if vocab_size is not None:
        vocab_size = _normalize_count("vocab_size", vocab_size)
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")

    input_ids = jnp.asarray(input_ids)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be a 2D array, got shape {input_ids.shape}.")
    if not jnp.issubdtype(input_ids.dtype, jnp.integer):
        raise TypeError(f"input_ids must contain integer token ids, got dtype {input_ids.dtype}.")
    if input_ids.shape[0] == 0:
        raise ValueError("input_ids must contain at least one batch row.")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one token.")
    if vocab_size is not None and not _contains_tracer(input_ids) and bool(jnp.any(input_ids >= vocab_size)):
        raise ValueError(f"input_ids token ids must be less than vocab_size={vocab_size}.")

    if kv_caches is not None:
        try:
            kv_caches = tuple(kv_caches)
        except TypeError as e:
            raise TypeError("kv_caches must be a sequence containing one cache per layer.") from e
        if len(kv_caches) != num_layers:
            raise ValueError(f"kv_caches must contain one cache per layer, got {len(kv_caches)} and expected {num_layers}.")
        for idx, kv_cache in enumerate(kv_caches):
            if kv_cache is not None and not isinstance(kv_cache, KVCache):
                raise TypeError(f"kv_caches entries must be KVCache instances or None, got {type(kv_cache)} at index {idx}.")
            if kv_cache is not None:
                kv_cache.validate_state()

    expected_mask_shape = input_ids.shape
    if kv_caches is not None:
        populated_mask_shapes = tuple(
            kv_cache.mask.shape for kv_cache in kv_caches if kv_cache is not None and kv_cache.mask is not None
        )
        if populated_mask_shapes:
            expected_mask_shape = populated_mask_shapes[0]
            if any(mask_shape != expected_mask_shape for mask_shape in populated_mask_shapes):
                raise ValueError("kv_caches populated entries must share attention mask shape.")
            if expected_mask_shape[0] != input_ids.shape[0]:
                raise ValueError(
                    "kv_caches populated entries must share input_ids batch size; "
                    f"got {expected_mask_shape[0]} and {input_ids.shape[0]}."
                )

    if attention_mask is not None:
        attention_mask = jnp.asarray(attention_mask)
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be a 2D array, got shape {attention_mask.shape}.")
        if not (jnp.issubdtype(attention_mask.dtype, jnp.bool_) or jnp.issubdtype(attention_mask.dtype, jnp.integer)):
            raise TypeError(f"attention_mask must be boolean or integer, got dtype {attention_mask.dtype}.")
        if attention_mask.shape != expected_mask_shape:
            raise ValueError(
                "attention_mask shape must match input_ids shape or populated KV cache mask shape; "
                f"got {attention_mask.shape} and expected {expected_mask_shape}."
            )
        attention_mask = attention_mask.astype(bool)
        if not _contains_tracer(attention_mask) and not bool(jnp.all(jnp.any(attention_mask, axis=1))):
            raise ValueError("attention_mask must contain at least one valid token per batch row.")
        has_valid_negative_placeholder = False
        if attention_mask.shape == input_ids.shape and not _contains_tracer((input_ids, attention_mask)):
            has_valid_negative_placeholder = bool(jnp.any(attention_mask & (input_ids < 0)))
        if has_valid_negative_placeholder:
            raise ValueError("attention_mask must not mark negative placeholder input_ids as valid.")

    return input_ids, attention_mask, kv_caches


def should_use_default_attention_mask(kv_caches) -> bool:
    if kv_caches is None:
        return True
    return not any(kv_cache is not None and kv_cache.mask is not None for kv_cache in kv_caches)


def prepare_default_attention_mask(input_ids, kv_caches):
    if not should_use_default_attention_mask(kv_caches):
        return None
    attention_mask = input_ids >= 0
    if not _contains_tracer(attention_mask) and not bool(jnp.all(jnp.any(attention_mask, axis=1))):
        raise ValueError(
            "input_ids must contain at least one non-negative token per batch row when attention_mask is omitted."
        )
    return attention_mask


def cached_sequence_length(kv_caches):
    if kv_caches is None:
        return None
    sequence_length = None
    for idx, kv_cache in enumerate(kv_caches):
        if kv_cache is None or kv_cache.k is None:
            continue
        if sequence_length is None:
            sequence_length = kv_cache.k.shape[1]
        elif kv_cache.k.shape[1] != sequence_length:
            raise ValueError(
                "kv_caches populated entries must share cached sequence length; "
                f"got {sequence_length} and {kv_cache.k.shape[1]} at index {idx}."
            )
    return sequence_length


def prepare_position_ids(position_ids, input_ids, max_position_embeddings: int | None = None):
    if position_ids is None:
        return None
    if max_position_embeddings is not None:
        max_position_embeddings = _normalize_count("max_position_embeddings", max_position_embeddings)
        if max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {max_position_embeddings}.")
    position_ids = jnp.asarray(position_ids)
    if position_ids.ndim != 2:
        raise ValueError(f"position_ids must be a 2D array, got shape {position_ids.shape}.")
    if not jnp.issubdtype(position_ids.dtype, jnp.integer):
        raise TypeError(f"position_ids must contain integer positions, got dtype {position_ids.dtype}.")
    if position_ids.shape[1] != input_ids.shape[1] or position_ids.shape[0] not in (1, input_ids.shape[0]):
        raise ValueError(
            "position_ids shape must be broadcastable to input_ids batch and sequence axes, "
            f"got {position_ids.shape} and {input_ids.shape}."
        )
    if not _contains_tracer(position_ids) and bool(jnp.any(position_ids < 0)):
        raise ValueError("position_ids must contain non-negative positions.")
    position_ids_exceed_max = max_position_embeddings is not None and not _contains_tracer(position_ids)
    if position_ids_exceed_max and bool(jnp.any(position_ids >= max_position_embeddings)):
        raise ValueError(f"position_ids must be less than max_position_embeddings={max_position_embeddings}.")
    return position_ids
