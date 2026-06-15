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
import operator
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import torch

logger = logging.getLogger(__name__)

JAXML_CACHE_DIR_ENV = "JAXML_CACHE_DIR"


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


def _resolve_np_dtype(dtype: str | torch.dtype):
    if isinstance(dtype, str):
        dtype_map = _str_to_np_dtype
    elif isinstance(dtype, torch.dtype):
        dtype_map = _torch_to_np_dtype
    else:
        raise TypeError(f"Expected dtype to be a string or torch.dtype, got {type(dtype)}.")

    try:
        return dtype_map[dtype]
    except KeyError as e:
        supported = ", ".join(str(k) for k in dtype_map)
        raise ValueError(f"Unsupported dtype {dtype!r}. Supported dtypes are: {supported}.") from e


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


def _write_bytes_atomically(path: Path, data: bytes):
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@functools.lru_cache()
def _hash(*args) -> str:
    m = hashlib.sha256()
    for s in args:
        encoded = s.encode(encoding="utf-8")
        m.update(len(encoded).to_bytes(8, byteorder="big"))
        m.update(encoded)
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


def _normalize_state_key(key: str) -> list[str]:
    if not isinstance(key, str):
        raise TypeError(f"State key must be a string, got {type(key)}.")
    if not key:
        raise ValueError("State key must be non-empty.")
    normalized = []
    for segment in key.split("."):
        if not segment:
            raise ValueError(f"State key {key!r} must not contain empty path segments.")
        if segment.isdigit() and normalized:
            normalized[-1] += "_" + segment
        else:
            normalized.append(segment)
    return normalized


def _insert_state_leaf(tree: dict, path: list[str], value: Any, source_key: str):
    cursor = tree
    for segment in path[:-1]:
        existing = cursor.setdefault(segment, {})
        if not isinstance(existing, dict):
            destination = ".".join(path)
            raise ValueError(f"State key {source_key!r} conflicts with existing leaf at destination {destination!r}.")
        cursor = existing

    leaf = path[-1]
    if leaf in cursor:
        destination = ".".join(path)
        raise ValueError(f"Multiple state keys map to destination {destination!r}.")
    cursor[leaf] = value


def _state_value_to_numpy(value: Any, dtype: Any, source_key: str):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"State value for key {source_key!r} must be a torch.Tensor, got {type(value)}.")
    tensor = value.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor.numpy().astype(dtype)


def _normalize_optional_head_dim(head_dim: int | None) -> int | None:
    if head_dim is None:
        return None
    if isinstance(head_dim, (bool, np.bool_)):
        raise TypeError(f"head_dim must be an integer when set, got {type(head_dim)}.")
    try:
        head_dim = operator.index(head_dim)
    except TypeError as e:
        raise TypeError(f"head_dim must be an integer when set, got {type(head_dim)}.") from e
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive when set, got {head_dim}.")
    return head_dim


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
    np_dtype = _resolve_np_dtype(dtype)
    head_dim = _normalize_optional_head_dim(head_dim)

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

        def _reshape_qkv_separate_weight(x):
            if x.shape[0] % head_dim != 0:
                raise ValueError(f"Q/K/V projection output dimension {x.shape[0]} must be divisible by head_dim {head_dim}.")
            return x.T.reshape(x.shape[1], -1, head_dim)

        def _reshape_qkv_separate_bias(x):
            if x.shape[0] % head_dim != 0:
                raise ValueError(f"Q/K/V projection bias length {x.shape[0]} must be divisible by head_dim {head_dim}.")
            return x.reshape(-1, head_dim)

        def _reshape_qkv_fused_weight(x):
            divisor = 3 * head_dim
            if x.shape[0] % divisor != 0:
                raise ValueError(
                    f"Fused QKV projection output dimension {x.shape[0]} must be divisible by 3 * head_dim {divisor}."
                )
            return x.T.reshape(x.shape[1], -1, 3, head_dim).transpose(0, 2, 1, 3)

        def _reshape_qkv_fused_bias(x):
            divisor = 3 * head_dim
            if x.shape[0] % divisor != 0:
                raise ValueError(f"Fused QKV projection bias length {x.shape[0]} must be divisible by 3 * head_dim {divisor}.")
            return x.reshape(-1, 3, head_dim).transpose(1, 0, 2)

        _qkv_separate_map = {
            "weight": ("kernel", _reshape_qkv_separate_weight),
            "bias": ("bias", _reshape_qkv_separate_bias),
        }
        _qkv_fused_map = {
            "weight": ("kernel", _reshape_qkv_fused_weight),
            "bias": ("bias", _reshape_qkv_fused_bias),
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
        split = _normalize_state_key(k)

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
            val = func(_state_value_to_numpy(v, np_dtype, k))
        else:
            val = _state_value_to_numpy(v, np_dtype, k)

        _insert_state_leaf(jax_states["params"], split, val, k)

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
    path = compiled_fn_path(name, hash)

    from jax.experimental.serialize_executable import serialize

    path.mkdir(parents=True, exist_ok=True)
    fn_path = path / "aot"
    spec_path = path / "in_out_spec"
    aot_fn, in_tree, out_tree = serialize(fn)
    _write_bytes_atomically(fn_path, aot_fn)
    io_spec_bytes = pickle.dumps((in_tree, out_tree))
    _write_bytes_atomically(spec_path, io_spec_bytes)
    return len(aot_fn) + len(io_spec_bytes)


def _validate_cache_key_part(part: str, label: str) -> str:
    part = str(part)
    if part in {"", ".", ".."}:
        raise ValueError(f"AOT cache {label} must be a non-empty path segment, got {part!r}.")
    if "/" in part or "\\" in part:
        raise ValueError(f"AOT cache {label} must not contain path separators, got {part!r}.")
    return part


def compiled_fn_path(name: str, hash: str = "0", cache_dir: str | Path | None = None) -> Path:
    name = _validate_cache_key_part(name, "name")
    hash = _validate_cache_key_part(hash, "hash")
    cache_root = Path(cache_dir) if cache_dir is not None else Path(os.environ.get(JAXML_CACHE_DIR_ENV, ".jaxml"))
    return cache_root / f"{name}_{hash}"


def compiled_fn_exist(name: str, hash: str = "0"):
    path = compiled_fn_path(name, hash)
    return (path / "aot").is_file() and (path / "in_out_spec").is_file()


@functools.lru_cache()
def _load_compiled_fn_from_path(path: str):
    from jax.experimental.serialize_executable import deserialize_and_load

    path = Path(path)
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


@timeit(logger)
def load_compiled_fn(name: str, hash=0):
    return _load_compiled_fn_from_path(str(compiled_fn_path(name, hash)))


def _is_stale_compiled_fn_error(error: TypeError) -> bool:
    return "Function compiled with input pytree does not match" in str(error)


def load_if_exists(name: str, hash: str, log: bool = True):
    name = _validate_cache_key_part(name, "name")
    hash = _validate_cache_key_part(hash, "hash")

    def _decorator(fn: Callable):
        @functools.wraps(fn)
        def _wrapped_fn(*args, **kwargs):
            def _compile_and_cache():
                lowered = jax.jit(fn).lower(*args, **kwargs)
                with Timeit() as t:
                    compiled_fn = lowered.compile()
                if log:
                    logger.info(f"Compiled function '{name}' ({t} seconds).")
                byte_count = save_compiled_fn(compiled_fn, name, hash=hash, log=log)
                _load_compiled_fn_from_path.cache_clear()
                if log:
                    logger.info(f"Cached AOT-compiled function '{name}' ({byte_count} bytes).")
                return compiled_fn

            if compiled_fn_exist(name, hash):
                try:
                    _cfn = load_compiled_fn(name, hash, log=log)
                except Exception as e:
                    if log:
                        logger.warning("Failed to load cached AOT function '%s'; recompiling. Error: %s", name, e)
                    _cfn = _compile_and_cache()
            else:
                _cfn = _compile_and_cache()

            try:
                return _cfn(*args, **kwargs)
            except TypeError as e:
                if not _is_stale_compiled_fn_error(e):
                    raise
                if log:
                    logger.warning("Cached AOT function '%s' is stale for current inputs; recompiling.", name)
                return _compile_and_cache()(*args, **kwargs)

        return _wrapped_fn

    return _decorator
