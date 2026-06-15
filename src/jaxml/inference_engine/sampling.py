import functools
import logging
import math
import numbers
import operator

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

NEG_INF = float("-inf")
logger = logging.getLogger(__name__)


def _validate_logits(logits):
    logits = jnp.asarray(logits)
    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension for the vocabulary axis.")
    if logits.shape[-1] <= 0:
        raise ValueError(f"logits must have a non-empty vocabulary axis, got shape {logits.shape}.")
    if not jnp.issubdtype(logits.dtype, jnp.floating):
        raise TypeError(f"logits must contain floating point values, got dtype {logits.dtype}.")
    if not _contains_tracer(logits):
        if bool(jnp.any(jnp.isnan(logits) | jnp.isposinf(logits))):
            raise ValueError("logits must not contain NaN or positive infinity.")
        if not bool(jnp.all(jnp.any(jnp.isfinite(logits), axis=-1))):
            raise ValueError("logits must contain at least one finite value per vocabulary row.")
    return logits


def _contains_tracer(x) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(x))


@functools.partial(jax.jit, static_argnames=("top_k",))
def top_k_filtering(rng, logits, top_k, *args):
    logits = _validate_logits(logits)
    top_k = _normalize_top_k(top_k)
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.shape[-1])
    values, _ = jax.lax.top_k(logits, top_k)
    cutoff = values[..., -1:]
    return jnp.where(logits >= cutoff, logits, NEG_INF)


def top_p_filtering(rng, logits, top_k, top_p, min_p, *args):
    logits = _validate_logits(logits)
    top_p = _normalize_probability("top_p", top_p)
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        max_logit = jnp.max(logits, axis=-1, keepdims=True)
        return jnp.where(logits >= max_logit, logits, NEG_INF)

    sorted_logits = -jnp.sort(-logits, axis=-1)
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

    # It selects the logit at the cutoff point and mask out everything below.
    # It is equivalent to top_p for most cases except the edge case when cutoff point has many
    #   tokens with the same logits. But the chance is arguably fairly small.
    cutoff_index = jnp.sum(cumulative_probs < top_p, axis=-1, keepdims=True)
    cutoff_logit = jnp.take_along_axis(sorted_logits, cutoff_index, axis=-1)
    logits = jnp.where(logits < cutoff_logit, NEG_INF, logits)
    return logits


def min_p_filtering(rng, logits, top_k, top_p, min_p, *args):
    logits = _validate_logits(logits)
    min_p = _normalize_probability("min_p", min_p)
    mask = jax.nn.softmax(logits, axis=-1) >= min_p
    has_allowed_token = jnp.any(mask, axis=-1, keepdims=True)
    return jnp.where(has_allowed_token, jnp.where(mask, logits, NEG_INF), logits)


def greedy_fn(rng, logits, *args):
    logits = _validate_logits(logits)
    return logits.argmax(-1)


def _clip_value(name: str, value: int | float, min: int | float, max: int | float):
    if value < min:
        logger.warning("Parameter %s is lower than %f (%f), setting it to be %f", name, min, value, min)
        return min
    elif value > max:
        logger.warning("Parameter %s is higher than %f (%f), setting it to be %f", name, max, value, max)
        return max
    return value


def _normalize_top_k(top_k: int) -> int:
    if isinstance(top_k, (bool, np.bool_)):
        raise TypeError(f"top_k must be an integer, got {type(top_k)}.")
    try:
        top_k = operator.index(top_k)
    except TypeError as e:
        raise TypeError(f"top_k must be an integer, got {type(top_k)}.") from e
    return _clip_value("top_k", top_k, 0, float("inf"))


def _normalize_real(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value)}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return value


def _normalize_probability(name: str, value: float):
    if _contains_tracer(value):
        return value
    return _clip_value(name, _normalize_real(name, value), 0.0, 1.0)


def _normalize_sampling_temperature(temp: float):
    if _contains_tracer(temp):
        return temp
    temp = _normalize_real("temp", temp)
    if temp <= 0.0:
        raise ValueError(f"temp must be positive for non-greedy sampling, got {temp}.")
    return temp


def _normalize_bool(name: str, value: bool) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a boolean, got {type(value)}.")


@struct.dataclass
class SamplingParams:
    top_k: int
    top_p: float
    min_p: float
    temp: float


def normalize_sampling_params(top_k: int, top_p: float, min_p: float, temp: float) -> SamplingParams:
    return SamplingParams(
        top_k=_normalize_top_k(top_k),
        top_p=_clip_value("top_p", _normalize_real("top_p", top_p), 0.0, 1.0),
        min_p=_clip_value("min_p", _normalize_real("min_p", min_p), 0.0, 1.0),
        temp=_clip_value("temp", _normalize_real("temp", temp), 0.0, float("inf")),
    )


@struct.dataclass
class SamplingMethod:
    use_top_k: bool
    use_top_p: bool
    use_min_p: bool
    use_greedy: bool

    def __post_init__(self):
        for name in ("use_top_k", "use_top_p", "use_min_p", "use_greedy"):
            object.__setattr__(self, name, _normalize_bool(name, getattr(self, name)))

    @classmethod
    def from_values(cls, top_k: int, top_p: float, min_p: float, temp: float):
        params = normalize_sampling_params(top_k=top_k, top_p=top_p, min_p=min_p, temp=temp)

        use_greedy = params.temp <= 0.0 or params.top_p <= 0.0 or params.top_k == 1
        # top_k = 0 is an indicator of skipping top_k
        use_top_k = params.top_k > 0 and not use_greedy
        use_top_p = params.top_p < 1.0 and not use_greedy
        use_min_p = params.min_p > 0.0 and not use_greedy
        return cls(use_top_k=use_top_k, use_top_p=use_top_p, use_min_p=use_min_p, use_greedy=use_greedy)

    def get_sampling_fn(self):
        if self.use_greedy:
            return greedy_fn

        pipeline = []
        if self.use_min_p:
            pipeline.append(min_p_filtering)
        if self.use_top_k:
            pipeline.append(top_k_filtering)
        if self.use_top_p:
            pipeline.append(top_p_filtering)

        def _sampling_fn(rng, logits, top_k, top_p, min_p, temp):
            logits = _validate_logits(logits)
            temp = _normalize_sampling_temperature(temp)
            for fn in pipeline:
                logits = fn(rng, logits, top_k, top_p, min_p, temp)
            return jax.random.categorical(rng, logits / temp)

        return _sampling_fn
