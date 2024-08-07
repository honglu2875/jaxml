# coding=utf-8
# Copyright 2023 Honglu Fan (https://github.com/honglu2875).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools

import jax
import jax.numpy as jnp
import tqdm
from typing import Optional

from .cache import KVCache


@functools.partial(jax.jit, static_argnames=("top_k", "filter_value"))
def top_k_filtering(logits, top_k=32, filter_value=-float("Inf")):
    # Remove all tokens with a probability less than the last token of the top-k
    if top_k >= logits.shape[-1]:
        return logits  # No need to filter if top_k is greater than or equal to the number of classes

        # Use jax.lax.top_k to get the top-k values and their indices
    values, indices = jax.lax.top_k(logits, top_k)

    # Create a mask where entries are True if their corresponding indices are in the top-k
    mask = jnp.zeros_like(logits, dtype=bool)
    mask = mask.at[indices].set(True)

    # Apply the mask to the logits, replacing values that are not in the top-k with the filter_value
    filtered_logits = jnp.where(mask, logits, filter_value)

    return filtered_logits


@functools.partial(jax.jit, static_argnames=("top_p", "filter_value"))
def top_p_filtering(logits, top_p=0.9, filter_value=-float("Inf")):
    sorted_indices = jnp.argsort(-logits)
    sorted_logits = logits[sorted_indices]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs[:-1] > top_p

    indices_to_remove = sorted_indices[sorted_indices_to_remove + 1]
    logits = logits.at[indices_to_remove].set(filter_value)
    return logits


@functools.partial(jax.jit, static_argnames=("top_k", "top_p", "filter_value"))
def top_k_top_p_filtering(
    logits: jnp.ndarray,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
):
    """
    Args:
        logits: original logits
        top_k: keep only top k tokens with the rest marked as 'filter_value'
        top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            The rest are marked as 'filter_value'.
        filter_value: the value used to replace the filtered entries
    """
    if top_k > 0:
        logits = top_k_filtering(logits, top_k=top_k, filter_value=filter_value)
    if top_p > 0:
        logits = top_p_filtering(logits, top_p=top_p, filter_value=filter_value)

    return logits


@functools.partial(jax.jit, static_argnames=("top_k", "top_p"))
def sample_with_tk_tp(rng, logits, top_k, top_p, temp):
    return jax.random.categorical(rng, top_k_top_p_filtering(logits / temp, top_k=top_k, top_p=top_p))


@jax.jit
def greedy_fn(rng, logits, *args):
    return logits.argmax(-1)


@functools.partial(jax.jit, static_argnames=("length", "axis"))
def _pad_to(x, length, axis=0):
    pad_shape = x.shape[:axis] + (length - x.shape[axis],) + x.shape[axis + 1 :]
    return jnp.concatenate((jnp.zeros(pad_shape), x), axis=axis)


def _loop_fn(cache_and_rng_and_out, i, params, sample_fn, eval_fn, top_k, top_p, temp):
    kv_caches, key, tok = cache_and_rng_and_out
    key, subkey = jax.random.split(key)
    outputs, kv_caches = eval_fn(
        params,
        tok,
        kv_caches=kv_caches,
        use_cache=True,
    )
    out_tk = sample_fn(key, outputs, top_k, top_p, temp)

    return (kv_caches, key, out_tk), out_tk.squeeze(1).T


def generate(
    params,
    eval_fn,
    prompt_tokens: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray],
    kv_caches: list[KVCache],
    do_sample: bool = True,
    seed: int = 0,
    max_new_tokens: int = 100,
    top_k: int = 0,
    top_p: float = 0.0,
    temperature: float = 1.0,
    include_prompt: bool = False,
    show_progress: bool = False
):
    """
    Args:
        params: FrozenDict containing the model parameters
        eval_fn: the evaluation function (usually the `model.apply` or `jax.jit(model.apply)`)
        prompt_tokens: the tokenized prompt
        do_sample: whether to sample the distribution or take the argmax
        seed: random seed
        max_new_tokens: the max generation length
        top_k: top k
        top_p: top p
        temperature: temperature
        include_prompt: whether to include the prompt in the return
    Returns:
        the completed token array (containing the prompt)
    """
    prompt_tokens = jnp.array(prompt_tokens)
    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens[None]

    rng = jax.random.PRNGKey(seed)
    batch_size, prompt_len = prompt_tokens.shape[:2]

    if do_sample and temperature > 1e-6:
        sample_fn = sample_with_tk_tp
    else:
        sample_fn = greedy_fn

    first_generated_logit, kv_caches = eval_fn(
        params,
        prompt_tokens,
        attention_mask,
        kv_caches,
        use_cache=True,
    )
    first_generated_tok = sample_fn(rng, first_generated_logit[:, -1:], top_k, top_p, temperature)

    loop_fn = functools.partial(
        _loop_fn,
        params=params,
        sample_fn=sample_fn,
        eval_fn=eval_fn,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
    )

    # Scan suffers from shape mismatch.
    # Potentially there are ways to still mitigate that (build separate kernels and pad the context window)
    # But I wonder how much gain there still is. Leave it as future todo.
    """
    loop_fn = jax.jit(loop_fn, static_argnames=("sample_fn", "eval_fn", "top_k", "top_p", "temp"))
    generated_toks = jax.lax.scan(
        loop_fn,
        (kv_caches, rng, first_generated_tok),
        jnp.arange(max_new_tokens - 1) + prompt_len,
    )[1].T
    """
    state = (kv_caches, rng, first_generated_tok)
    new_tokens = []
    for i in tqdm.trange(prompt_len, prompt_len + max_new_tokens - 1, disable=not show_progress):
        state, token = loop_fn(state, i)
        new_tokens.append(token)
    generated_toks = jnp.stack(new_tokens, axis=-1)
    

    if include_prompt:
        return jnp.concatenate((first_generated_tok, generated_toks), axis=-1)
    else:
        return jnp.concatenate((prompt_tokens, first_generated_tok, generated_toks), axis=-1)
