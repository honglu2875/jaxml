from typing import Any

import jax
import jax.numpy as jnp


def contains_tracer(x: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(x))


def validate_finite_values(name: str, values: jnp.ndarray):
    try:
        has_only_finite_values = bool(jnp.all(jnp.isfinite(values)))
    except jax.errors.TracerBoolConversionError:
        return
    if not contains_tracer(values) and not has_only_finite_values:
        raise ValueError(f"{name} must contain only finite values.")
