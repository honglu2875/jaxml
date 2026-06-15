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


def slice_last_n_logits_hidden_states(hidden_states: jnp.ndarray, keep_last_n_logits: int) -> jnp.ndarray:
    hidden_states = jnp.asarray(hidden_states)
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be a 3D array, got shape {hidden_states.shape}.")
    if not jnp.issubdtype(hidden_states.dtype, jnp.floating):
        raise TypeError(f"hidden_states must contain floating point values, got dtype {hidden_states.dtype}.")
    keep_last_n_logits = _normalize_count("keep_last_n_logits", keep_last_n_logits)
    if keep_last_n_logits < 0:
        raise ValueError(f"keep_last_n_logits must be non-negative, got {keep_last_n_logits}.")
    if keep_last_n_logits == 0:
        return hidden_states
    return hidden_states[:, -keep_last_n_logits:]


def prepare_model_inputs(input_ids, attention_mask, kv_caches, num_layers: int):
    num_layers = _normalize_count("num_layers", num_layers)
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")

    input_ids = jnp.asarray(input_ids)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be a 2D array, got shape {input_ids.shape}.")
    if not jnp.issubdtype(input_ids.dtype, jnp.integer):
        raise TypeError(f"input_ids must contain integer token ids, got dtype {input_ids.dtype}.")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one token.")

    if attention_mask is not None:
        attention_mask = jnp.asarray(attention_mask)
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be a 2D array, got shape {attention_mask.shape}.")
        if not (jnp.issubdtype(attention_mask.dtype, jnp.bool_) or jnp.issubdtype(attention_mask.dtype, jnp.integer)):
            raise TypeError(f"attention_mask must be boolean or integer, got dtype {attention_mask.dtype}.")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape must match input_ids shape; got {attention_mask.shape} and {input_ids.shape}."
            )
        attention_mask = attention_mask.astype(bool)
        if not _contains_tracer(attention_mask) and not bool(jnp.all(jnp.any(attention_mask, axis=1))):
            raise ValueError("attention_mask must contain at least one valid token per batch row.")

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

    return input_ids, attention_mask, kv_caches
