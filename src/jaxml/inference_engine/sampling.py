import functools
import logging

import jax
import jax.numpy as jnp
from flax import struct

NEG_INF = float("-inf")
logger = logging.getLogger(__name__)


@functools.partial(jax.jit, static_argnames=("top_k",))
def top_k_filtering(rng, logits, top_k, *args):
    # Remove all tokens with a probability less than the last token of the top-k
    # Use jax.lax.top_k to get the top-k values and their indices
    values, indices = jax.lax.top_k(logits, top_k)

    # Create a mask where entries are True if their corresponding indices are in the top-k
    mask = jnp.zeros_like(logits, dtype=bool)
    mask = mask.at[indices].set(True)

    # Apply the mask to the logits, replacing values that are not in the top-k with the filter_value
    filtered_logits = jnp.where(mask, logits, NEG_INF)

    return filtered_logits


def top_p_filtering(rng, logits, top_k, top_p, min_p, *args):
    sorted_logits = -jnp.sort(-logits, axis=-1)
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

    # It selects the logit at the cutoff point and mask out everything below.
    # It is equivalent to top_p for most cases except the edge case when cutoff point has many
    #   tokens with the same logits. But the chance is arguably fairly small.
    cutoff_index = jnp.sum(cumulative_probs < top_p, axis=-1, keepdims=True)
    cutoff_logit = jnp.take_along_axis(logits, cutoff_index, axis=-1)
    logits = jnp.where(logits < cutoff_logit, NEG_INF, logits)
    return logits


def min_p_filtering(rng, logits, top_k, top_p, min_p, *args):
    mask = jax.nn.softmax(logits, axis=-1) >= min_p
    return jax.lax.cond(
        mask.sum() > 0,
        lambda: jnp.where(mask, logits, NEG_INF),
        lambda: logits,
    )


def greedy_fn(rng, logits, *args):
    return logits.argmax(-1)


def _clip_value(name: str, value: int | float, min: int | float, max: int | float):
    if value < min:
        logger.warning("Parameter %s is lower than %f (%f), setting it to be %f", name, min, value, min)
        return min
    elif value > max:
        logger.warning("Parameter %s is higher than %f (%f), setting it to be %f", name, max, value, max)
        return max
    return value


@struct.dataclass
class SamplingMethod:
    use_top_k: bool
    use_top_p: bool
    use_min_p: bool
    use_greedy: bool

    @classmethod
    def from_values(cls, top_k: int, top_p: float, min_p: float, temp: float):
        top_k = _clip_value("top_k", top_k, 0, float("inf"))        
        min_p = _clip_value("min_p", min_p, 0.0, 1.0)
        temp = _clip_value("temp", temp, 0.0, float("inf"))

        use_greedy = temp <= 0.0 or top_p <= 0.0 or top_k == 1
        # top_k = 0 is an indicator of skipping top_k
        use_top_k = top_k > 0 and not use_greedy
        use_top_p = top_p < 1.0 and not use_greedy
        use_min_p = min_p > 0.0 and not use_greedy
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
            for fn in pipeline:
                logits = fn(rng, logits, top_k, top_p, min_p, temp)
            return jax.random.categorical(rng, logits / temp)

        return _sampling_fn
