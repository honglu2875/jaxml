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
from typing import Optional

import jax
import jax.numpy as jnp
import tqdm

from jaxml.cache import KVCache
from jaxml.inference_engine.sampling import SamplingMethod
from jaxml.outputs import GenerationOutput
from jaxml.utils import load_if_exists

logger = logging.getLogger(__name__)


@functools.partial(jax.jit, static_argnames=("length", "axis"))
def _pad_to(x, length, axis=0):
    pad_shape = x.shape[:axis] + (length - x.shape[axis],) + x.shape[axis + 1 :]
    return jnp.concatenate((jnp.zeros(pad_shape), x), axis=axis)


def _loop_fn(cache_and_rng_and_out, i, sample_fn, eval_fn, top_k, top_p, min_p, temp):
    params, kv_caches, key, tok = cache_and_rng_and_out
    key, subkey = jax.random.split(key)
    outputs, kv_caches = eval_fn(
        params,
        tok,
        kv_caches=kv_caches,
        use_cache=True,
    )
    out_tk = sample_fn(subkey, outputs, top_k, top_p, min_p, temp)

    return (params, kv_caches, key, out_tk), out_tk.squeeze(1).T


def _loop_fn_no_scan(key, kv_caches, tok, params, sample_fn, eval_fn, top_k, top_p, min_p, temp):
    # Why not using `_loop_fn` for for-loop:
    #   Outer loop is not compiled, and there is no guarantee that `params` is not copied.
    #   It actually copies and causes an OOM on TPU-v4 with Llama2 7B.
    #   The way to fix it is not to return `params`, thus removing the ambiguity.
    key, subkey = jax.random.split(key)
    outputs, kv_caches = eval_fn(
        params,
        tok,
        kv_caches=kv_caches,
        use_cache=True,
    )
    out_tk = sample_fn(subkey, outputs, top_k, top_p, min_p, temp)

    return key, kv_caches, out_tk


def generate(
    params,
    eval_fn,
    prompt_tokens: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray],
    kv_caches: tuple[KVCache],
    call_hash: int,
    sampling_method: SamplingMethod,
    seed: int = 0,
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
    prompt_tokens = jnp.array(prompt_tokens)
    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens[None]

    rng = jax.random.PRNGKey(seed)
    batch_size, prompt_len = prompt_tokens.shape[:2]

    sample_fn = sampling_method.get_sampling_fn()
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
        def _prefill(params, prompt_tokens, attention_mask, kv_caches, top_p, min_p, temperature):
            first_generated_logit, kv_caches = eval_fn(
                params,
                prompt_tokens,
                attention_mask,
                kv_caches,
                use_cache=True,
            )
            return (
                sample_fn(rng, first_generated_logit[:, -1:], top_k, top_p, min_p, temperature),
                kv_caches,
            )

        first_generated_tok, kv_caches = _prefill(params, prompt_tokens, attention_mask, kv_caches, top_p, min_p, temperature)

    if fuse_decoding:
        loop_fn = functools.partial(_loop_fn, **loop_fn_params)

        @load_if_exists(name="decode", hash=call_hash)
        def _decode(params, kv_caches, rng, first_generated_tok):
            output = jax.lax.scan(
                loop_fn,
                (params, kv_caches, rng, first_generated_tok),
                jnp.arange(max_new_tokens - 1),
            )
            return output[1].T, output[0][1]  # tokens, kv_caches

        generated_toks, kv_caches = _decode(params, kv_caches, rng, first_generated_tok)
    else:
        # This could potentially turn into token-streaming
        loop_fn = functools.partial(_loop_fn_no_scan, **loop_fn_params)
        loop_fn = load_if_exists(name="decode_one_step", hash=call_hash, log=False)(loop_fn)

        new_tokens = []
        token = first_generated_tok
        for i in tqdm.trange(max_new_tokens if skip_prefill else max_new_tokens - 1):
            rng, kv_caches, token = loop_fn(rng, kv_caches, token, params)
            new_tokens.append(token.squeeze(1).T)

        generated_toks = jnp.stack(new_tokens, axis=-1)

    if skip_prefill:
        tokens = generated_toks
    else:
        if include_prompt:
            tokens = jnp.concatenate((prompt_tokens, first_generated_tok, generated_toks), axis=-1)
        else:
            tokens = jnp.concatenate((first_generated_tok, generated_toks), axis=-1)

    return GenerationOutput(tokens=tokens, kv_caches=kv_caches)
