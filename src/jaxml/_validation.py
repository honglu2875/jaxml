import math
import numbers
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def normalize_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value)}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return value


def contains_tracer(x: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(x))


def validate_finite_values(name: str, values: jnp.ndarray):
    try:
        has_only_finite_values = bool(jnp.all(jnp.isfinite(values)))
    except jax.errors.TracerBoolConversionError:
        return
    if not contains_tracer(values) and not has_only_finite_values:
        raise ValueError(f"{name} must contain only finite values.")
