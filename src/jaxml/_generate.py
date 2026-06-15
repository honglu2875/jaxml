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
import logging
import operator
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from jaxml.cache import KVCache
from jaxml.inference_engine.sampling import SamplingMethod, normalize_sampling_params
from jaxml.outputs import GenerationOutput
from jaxml.utils import load_if_exists

logger = logging.getLogger(__name__)


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


def _normalize_callable(name: str, value):
    if not callable(value):
        raise TypeError(f"{name} must be callable, got {type(value)}.")
    return value


def _get_sampling_fn(sampling_method: SamplingMethod):
    get_sampling_fn = getattr(sampling_method, "get_sampling_fn", None)
    if not callable(get_sampling_fn):
        raise TypeError("sampling_method must provide a callable get_sampling_fn method.")
    sample_fn = get_sampling_fn()
    if not callable(sample_fn):
        raise TypeError(f"sampling_method.get_sampling_fn() must return a callable, got {type(sample_fn)}.")
    return sample_fn


def _unpack_eval_output(eval_output, input_tokens: jnp.ndarray):
    try:
        logits, kv_caches = eval_output
    except (TypeError, ValueError) as e:
        raise TypeError("eval_fn must return a pair of (logits, kv_caches).") from e

    logits = jnp.asarray(logits)
    expected_prefix = input_tokens.shape
    if logits.ndim != input_tokens.ndim + 1:
        raise ValueError(
            f"eval_fn logits must have shape input_tokens.shape + (vocab_size,), got {logits.shape} "
            f"for input shape {expected_prefix}."
        )
    if logits.shape[:-1] != expected_prefix:
        raise ValueError(
            f"eval_fn logits leading shape must match input_tokens shape; got {logits.shape[:-1]} " f"and {expected_prefix}."
        )
    if logits.shape[-1] <= 0:
        raise ValueError(f"eval_fn logits must have a non-empty vocabulary axis, got shape {logits.shape}.")
    if not jnp.issubdtype(logits.dtype, jnp.floating):
        raise TypeError(f"eval_fn logits must have a floating dtype, got {logits.dtype}.")

    return logits, _validate_kv_caches(kv_caches)


def _validate_sampled_tokens(sampled_tokens, logits: jnp.ndarray):
    sampled_tokens = jnp.asarray(sampled_tokens)
    expected_shape = logits.shape[:-1]
    if sampled_tokens.shape != expected_shape:
        raise ValueError(f"sample_fn must return token ids with shape {expected_shape}, got {sampled_tokens.shape}.")
    if not jnp.issubdtype(sampled_tokens.dtype, jnp.integer):
        raise TypeError(f"sample_fn must return integer token ids, got dtype {sampled_tokens.dtype}.")
    return sampled_tokens


def _validate_kv_caches(kv_caches) -> tuple[KVCache, ...]:
    try:
        kv_caches = tuple(kv_caches)
    except TypeError as e:
        raise TypeError("kv_caches must be a sequence of KVCache instances.") from e
    for idx, kv_cache in enumerate(kv_caches):
        if not isinstance(kv_cache, KVCache):
            raise TypeError(f"kv_caches entries must be KVCache instances, got {type(kv_cache)} at index {idx}.")
    return kv_caches


def _validate_prefilled_kv_caches(kv_caches: tuple[KVCache, ...]):
    if not kv_caches:
        raise ValueError("skip_prefill=True requires at least one prefilled KV cache.")
    for idx, kv_cache in enumerate(kv_caches):
        if kv_cache.k is None or kv_cache.v is None or kv_cache.mask is None or kv_cache.pos_id is None:
            raise ValueError(f"skip_prefill=True requires kv_caches[{idx}] to be prefilled.")


@functools.partial(jax.jit, static_argnames=("length", "axis"))
def _pad_to(x, length, axis=0):
    pad_shape = x.shape[:axis] + (length - x.shape[axis],) + x.shape[axis + 1 :]
    return jnp.concatenate((jnp.zeros(pad_shape), x), axis=axis)


def _loop_fn(cache_and_rng_and_out, i, sample_fn, eval_fn, top_k, top_p, min_p, temp):
    params, kv_caches, key, tok = cache_and_rng_and_out
    key, subkey = jax.random.split(key)
    eval_output = eval_fn(
        params,
        tok,
        kv_caches=kv_caches,
        use_cache=True,
    )
    outputs, kv_caches = _unpack_eval_output(eval_output, tok)
    out_tk = _validate_sampled_tokens(sample_fn(subkey, outputs, top_k, top_p, min_p, temp), outputs)

    return (params, kv_caches, key, out_tk), out_tk.squeeze(1).T


def _loop_fn_no_scan(key, kv_caches, tok, params, sample_fn, eval_fn, top_k, top_p, min_p, temp):
    # Why not using `_loop_fn` for for-loop:
    #   Outer loop is not compiled, and there is no guarantee that `params` is not copied.
    #   It actually copies and causes an OOM on TPU-v4 with Llama2 7B.
    #   The way to fix it is not to return `params`, thus removing the ambiguity.
    key, subkey = jax.random.split(key)
    eval_output = eval_fn(
        params,
        tok,
        kv_caches=kv_caches,
        use_cache=True,
    )
    outputs, kv_caches = _unpack_eval_output(eval_output, tok)
    out_tk = _validate_sampled_tokens(sample_fn(subkey, outputs, top_k, top_p, min_p, temp), outputs)

    return key, kv_caches, out_tk


def generate(
    params,
    eval_fn,
    prompt_tokens: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray],
    kv_caches: tuple[KVCache, ...],
    call_hash: str,
    sampling_method: SamplingMethod,
    seed: int = 0,
    rng: Optional[jnp.ndarray] = None,
    max_new_tokens: int = 100,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    temperature: float = 1.0,
    include_prompt: bool = False,
    fuse_decoding: bool = False,
    skip_prefill: bool = False,
):
    """
    Args:
        params: FrozenDict containing the model parameters
        eval_fn: the evaluation function (usually the `model.apply` or `jax.jit(model.apply)`)
        prompt_tokens: the tokenized prompt
        attention_mask: The attention mask
        kv_caches: The (list of) kv-caches
        call_hash: A hash unique to each AOT-function for decoding
        sampling_method: A dataclass specifying the sampling method
        seed: random seed
        rng: optional PRNG key to continue generation across calls
        max_new_tokens: the max generation length
        top_k: top k (0 will skip top_k sampling)
        top_p: top p
        temperature: temperature
        include_prompt: whether to include the prompt in the return
        fuse_decoding: whether to fuse decoding in compiling
        skip_prefill: True if k and v are prefilled inside kv_caches and we skip prefill
    Returns:
        GenerationOutput consisting of:
            the completed token array (containing the prompt)
            the kv-caches
    """
    seed = _normalize_count("seed", seed)
    max_new_tokens = _normalize_count("max_new_tokens", max_new_tokens)
    include_prompt = _normalize_bool("include_prompt", include_prompt)
    fuse_decoding = _normalize_bool("fuse_decoding", fuse_decoding)
    skip_prefill = _normalize_bool("skip_prefill", skip_prefill)
    sampling_params = normalize_sampling_params(top_k=top_k, top_p=top_p, min_p=min_p, temp=temperature)
    top_k = sampling_params.top_k
    top_p = sampling_params.top_p
    min_p = sampling_params.min_p
    temperature = sampling_params.temp
    if rng is None:
        rng = jax.random.PRNGKey(seed)
    else:
        rng = jnp.asarray(rng)
        if rng.shape != (2,):
            raise ValueError(f"rng must be a PRNG key with shape (2,), got shape {rng.shape}.")
        if not jnp.issubdtype(rng.dtype, jnp.integer):
            raise TypeError(f"rng must contain integer key data, got dtype {rng.dtype}.")

    prompt_tokens = jnp.array(prompt_tokens)
    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens[None]
    elif prompt_tokens.ndim != 2:
        raise ValueError(f"prompt_tokens must be a 1D or 2D array, got shape {prompt_tokens.shape}.")
    if not jnp.issubdtype(prompt_tokens.dtype, jnp.integer):
        raise TypeError(f"prompt_tokens must contain integer token ids, got dtype {prompt_tokens.dtype}.")
    if prompt_tokens.shape[1] == 0:
        raise ValueError("prompt_tokens must contain at least one token.")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}.")
    kv_caches = _validate_kv_caches(kv_caches)
    if skip_prefill:
        _validate_prefilled_kv_caches(kv_caches)
    if attention_mask is not None:
        attention_mask = jnp.asarray(attention_mask)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask[None]
        elif attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be a 1D or 2D array, got shape {attention_mask.shape}.")
        if not (jnp.issubdtype(attention_mask.dtype, jnp.bool_) or jnp.issubdtype(attention_mask.dtype, jnp.integer)):
            raise TypeError(f"attention_mask must be boolean or integer, got dtype {attention_mask.dtype}.")
        if attention_mask.shape != prompt_tokens.shape:
            raise ValueError(
                f"attention_mask shape must match prompt_tokens shape; got {attention_mask.shape} and {prompt_tokens.shape}."
            )
        attention_mask = attention_mask.astype(bool)
        if not bool(jnp.all(jnp.any(attention_mask, axis=1))):
            raise ValueError("attention_mask must contain at least one valid token per batch row.")

    if max_new_tokens == 0:
        if include_prompt:
            tokens = prompt_tokens
        else:
            tokens = jnp.empty((prompt_tokens.shape[0], 0), dtype=prompt_tokens.dtype)
        return GenerationOutput(tokens=tokens, kv_caches=kv_caches, rng=rng)

    eval_fn = _normalize_callable("eval_fn", eval_fn)
    sample_fn = _get_sampling_fn(sampling_method)
    loop_fn_params = dict(
        sample_fn=sample_fn,
        eval_fn=eval_fn,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        temp=temperature,
    )

    if skip_prefill:
        first_generated_tok = prompt_tokens[:, -1:]
    else:
        # Note that top_k value cannot be traced due to the limit of jax.lax.top_k
        @load_if_exists(name="prefill", hash=call_hash)
        def _prefill(params, prompt_tokens, attention_mask, kv_caches, rng, top_p, min_p, temperature):
            eval_output = eval_fn(
                params,
                prompt_tokens,
                attention_mask,
                kv_caches,
                use_cache=True,
            )
            first_generated_logit, kv_caches = _unpack_eval_output(eval_output, prompt_tokens)
            first_generated_tok = sample_fn(rng, first_generated_logit[:, -1:], top_k, top_p, min_p, temperature)
            first_generated_tok = _validate_sampled_tokens(first_generated_tok, first_generated_logit[:, -1:])
            return (
                first_generated_tok,
                kv_caches,
            )

        first_generated_tok, kv_caches = _prefill(
            params, prompt_tokens, attention_mask, kv_caches, rng, top_p, min_p, temperature
        )

    decode_steps = max_new_tokens if skip_prefill else max_new_tokens - 1

    if fuse_decoding:
        loop_fn = functools.partial(_loop_fn, **loop_fn_params)

        @load_if_exists(name="decode", hash=call_hash)
        def _decode(params, kv_caches, rng, first_generated_tok):
            output = jax.lax.scan(
                loop_fn,
                (params, kv_caches, rng, first_generated_tok),
                jnp.arange(decode_steps),
            )
            return output[1].T, output[0][1], output[0][2]  # tokens, kv_caches, rng

        if decode_steps == 0:
            generated_toks = jnp.empty((prompt_tokens.shape[0], 0), dtype=first_generated_tok.dtype)
        else:
            generated_toks, kv_caches, rng = _decode(params, kv_caches, rng, first_generated_tok)
    else:
        # This could potentially turn into token-streaming
        loop_fn = functools.partial(_loop_fn_no_scan, **loop_fn_params)
        loop_fn = load_if_exists(name="decode_one_step", hash=call_hash, log=False)(loop_fn)

        new_tokens = []
        token = first_generated_tok
        for _ in tqdm.trange(decode_steps):
            rng, kv_caches, token = loop_fn(rng, kv_caches, token, params)
            new_tokens.append(token.squeeze(1).T)

        if new_tokens:
            generated_toks = jnp.stack(new_tokens, axis=-1)
        else:
            generated_toks = jnp.empty((prompt_tokens.shape[0], 0), dtype=first_generated_tok.dtype)

    if skip_prefill:
        tokens = generated_toks
    else:
        if include_prompt:
            tokens = jnp.concatenate((prompt_tokens, first_generated_tok, generated_toks), axis=-1)
        else:
            tokens = jnp.concatenate((first_generated_tok, generated_toks), axis=-1)

    return GenerationOutput(tokens=tokens, kv_caches=kv_caches, rng=rng)
