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

import jax
import jax.numpy as jnp
import torch
import numpy as np
from typing import Any


def get_default_pos_ids(shape, mask=None):
    """Given an attention mask, we infer the default position id.
    Assume the sequence axis is 1."""
    bs, seq_len = shape[:2]
    
    if mask is not None:
        check_shape(mask, bs, seq_len)
    else:
        return jnp.broadcast_to(jnp.arange(seq_len), shape[:2])

    pos_ids = (
        jnp.arange(seq_len, dtype=jnp.int32)[None]
        - (1 - mask).sum(1, keepdims=True)
    )
    pos_ids = jnp.where(pos_ids >= 0, pos_ids, 0)
    return pos_ids


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
    _to_np_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        # "bf16": np.float16,
    }

    if isinstance(input, torch.nn.Module):
        states = input.state_dict()
    elif isinstance(input, dict):
        states = input
    else:
        raise TypeError(
            f"Expected input to be either a PyTorch module or a dict, got {type(input)}."
        )

    jax_states = {"params": {}}

    _dense_key_map = {"weight": ("kernel", lambda x: x.T)}
    if head_dim is None:
        _qkv_separate_map = _dense_key_map
    else:
        _qkv_separate_map = {
            "weight": ("kernel", lambda x: x.T.reshape(x.shape[1], -1, head_dim))
        }
    _emb_key_map = {"weight": ("embedding", lambda x: x)}
    _exclude_keys = {"post_attention_layernorm", "input_layernorm", "norm"}

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
        if isinstance()
        else:
            return leaf.__repr__()

    # Apply custom_format to every leaf in the pytree
    formatted_tree = jax.tree.map(custom_format, obj)

    # Serialize as JSON for pretty printing
    pretty_json = json.dumps(formatted_tree, indent=2)
    print(pretty_json)