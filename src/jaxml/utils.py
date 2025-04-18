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
import hashlib
import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import torch

logger = logging.getLogger(__name__)


_str_to_np_dtype = {
    "float16": np.float16,
    "bfloat16": jnp.bfloat16,
    "float32": np.float32,
    "float64": np.float64,
}

_torch_to_np_dtype = {
    torch.float16: np.float16,
    torch.bfloat16: jnp.bfloat16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    # "bf16": np.float16,
}


class Timeit:
    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()

    def __str__(self):
        if self.start is None:
            return "Not started."
        elif self.end is None:
            return str(time.perf_counter() - self.start)
        else:
            return str(self.end - self.start)

    __repr__ = __str__


@functools.lru_cache()
def _hash(*args) -> str:
    m = hashlib.sha256()
    for s in args:
        m.update(s.encode(encoding="utf-8"))
    return m.hexdigest()


def check_shape(tensor, *shape):
    chex.assert_shape(tensor, shape)


def get_default_pos_ids(mask):
    """Given an attention mask, we infer the default position id.

    We implement right padding:
    Say an attention mask is the following:
        True True  True  False False
        True False False False False
    Then the first sequence is of length 3 and the second is of length 1.
    And by decoding further, we are supposed to fill in the "False" slots.
    """
    pos_ids = (mask * jnp.arange(mask.shape[1])[None]).argmax(1)  # Last position of True at each row
    return pos_ids[:, None]


def torch_to_jax_states(
    input: torch.nn.Module | dict,
    dtype: str | torch.dtype = torch.float16,
    head_dim: int | None = None,
):
    """
    Converts the states of a PyTorch model to JAX states.
    Args:
        input: a torch state dict
        dtype: the dtype
        head_dim: if not None, it will try to reshape the last axis of q, k, v
            weights further into (..., head_dim, num_head).
    """
    _to_np_dtype = _str_to_np_dtype if isinstance(dtype, str) else _torch_to_np_dtype

    if isinstance(input, torch.nn.Module):
        states = input.state_dict()
    elif isinstance(input, dict):
        states = input
    else:
        raise TypeError(f"Expected input to be either a PyTorch module or a dict, got {type(input)}.")

    jax_states = {"params": {}}

    _dense_key_map = {"weight": ("kernel", lambda x: x.T)}
    if head_dim is None:
        _qkv_separate_map = _dense_key_map
        _qkv_fused_map = {"weight": ("kernel", lambda x: x.T.reshape(x.shape[1], 3, -1))}
    else:
        _qkv_separate_map = {
            "weight": ("kernel", lambda x: x.T.reshape(x.shape[1], -1, head_dim)),
            "bias": ("bias", lambda x: x.reshape(-1, head_dim)),
        }
        _qkv_fused_map = {
            "weight": ("kernel", lambda x: x.T.reshape(x.shape[1], -1, 3, head_dim).transpose(0, 2, 1, 3)),
            "bias": ("bias", lambda x: x.reshape(-1, 3, head_dim).transpose(1, 0, 2)),
        }
    _emb_key_map = {"weight": ("embedding", lambda x: x)}
    _exclude_keys = {
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "input_layernorm",
        "norm",
        "q_norm",
        "k_norm",
    }

    for k, v in states.items():
        split = k.split(".")
        for i, s in enumerate(split):
            if s.isdigit():
                split[i - 1] += "_" + s
                split.pop(i)

        if len(split) >= 2 and split[-2] in _exclude_keys:
            _key_map = {}
        else:
            if "embed_tokens" in split:
                _key_map = _emb_key_map
            elif any(k in split for k in ["q_proj", "k_proj", "v_proj"]):
                _key_map = _qkv_separate_map
            elif "qkv_proj" in split:
                _key_map = _qkv_fused_map
            else:
                _key_map = _dense_key_map

        if split[-1] in _key_map:
            split[-1], func = _key_map[split[-1]]
            val = func(v.numpy().astype(_to_np_dtype[dtype]))
        else:
            val = v.numpy().astype(_to_np_dtype[dtype])

        _dict = jax_states["params"]
        for i, l in enumerate(split):
            _dict[l] = _dict.setdefault(l, {} if i < len(split) - 1 else val)
            _dict = _dict[l]

    return jax_states


def pprint_pytree(obj: Any):
    """Pretty print JAX pytree by collapsing tensors into only shape information."""

    def custom_format(leaf):
        if isinstance(leaf, jnp.ndarray):
            return f"<Array shape={leaf.shape} dtype={leaf.dtype}>"
        else:
            return leaf.__repr__()

    # Apply custom_format to every leaf in the pytree
    formatted_tree = jax.tree.map(custom_format, obj)

    # Serialize as JSON for pretty printing
    pretty_json = json.dumps(formatted_tree, indent=2)
    print(pretty_json)


def timeit(logger):
    def _factory(fn):
        name = fn.__name__

        @functools.wraps(fn)
        def _wrapped(*args, log=True, **kwargs):
            start_time = time.perf_counter()
            ret = fn(*args, **kwargs)
            if log:
                logger.info(f"Execution time for {name}: {time.perf_counter() - start_time}")
            return ret

        return _wrapped

    return _factory


@timeit(logger)
def save_compiled_fn(fn, name: str, hash: str = "0", **kwargs) -> int:
    from jax.experimental.serialize_executable import serialize

    path = Path(".jaxml") / f"{name}_{hash}"
    path.mkdir(parents=True, exist_ok=True)
    fn_path = path / "aot"
    spec_path = path / "in_out_spec"
    aot_fn, in_tree, out_tree = serialize(fn)
    with fn_path.open("wb") as f:
        f.write(aot_fn)
    with spec_path.open("wb") as f:
        io_spec_bytes = pickle.dumps((in_tree, out_tree))
        f.write(io_spec_bytes)
    return len(aot_fn) + len(io_spec_bytes)


def compiled_fn_exist(name: str, hash: str = "0"):
    return (Path(".jaxml") / f"{name}_{hash}").exists()


@timeit(logger)
@functools.lru_cache()
def load_compiled_fn(name: str, hash=0):
    from jax.experimental.serialize_executable import deserialize_and_load

    path = Path(".jaxml") / f"{name}_{hash}"
    fn_path = path / "aot"
    spec_path = path / "in_out_spec"
    if not fn_path.exists() or not spec_path.exists():
        raise ValueError(f"Cannot find files from the folder {path}")

    with fn_path.open("rb") as f:
        aot_fn = f.read()
    with spec_path.open("rb") as f:
        in_tree, out_tree = pickle.load(f)
    compiled_fn = deserialize_and_load(
        aot_fn,
        in_tree,
        out_tree,
    )
    return compiled_fn


def load_if_exists(name: str, hash: str, log: bool = True):
    def _decorator(fn: Callable):
        @functools.wraps(fn)
        def _wrapped_fn(*args, **kwargs):
            if compiled_fn_exist(name, hash):
                _cfn = load_compiled_fn(name, hash, log=log)
            else:
                lowered = jax.jit(fn).lower(*args, **kwargs)
                with Timeit() as t:
                    _cfn = lowered.compile()
                if log:
                    logger.info(f"Compiled function '{name}' ({t} seconds).")
                byte_count = save_compiled_fn(_cfn, name, hash=hash, log=log)
                if log:
                    logger.info(f"Cached AOT-compiled function '{name}' ({byte_count} bytes).")

            return _cfn(*args, **kwargs)

        return _wrapped_fn

    return _decorator
