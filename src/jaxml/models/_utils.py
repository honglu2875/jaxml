import operator

import jax.numpy as jnp


def _normalize_count(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value)}.")
    try:
        return operator.index(value)
    except TypeError as e:
        raise TypeError(f"{name} must be an integer, got {type(value)}.") from e


def slice_last_n_logits_hidden_states(hidden_states: jnp.ndarray, keep_last_n_logits: int) -> jnp.ndarray:
    keep_last_n_logits = _normalize_count("keep_last_n_logits", keep_last_n_logits)
    if keep_last_n_logits < 0:
        raise ValueError(f"keep_last_n_logits must be non-negative, got {keep_last_n_logits}.")
    if keep_last_n_logits == 0:
        return hidden_states
    return hidden_states[:, -keep_last_n_logits:]
