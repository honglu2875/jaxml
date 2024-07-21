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

import json
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import torch


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
        raise TypeError(f"Expected input to be either a PyTorch module or a dict, got {type(input)}.")

    jax_states = {"params": {}}

    _dense_key_map = {"weight": ("kernel", lambda x: x.T)}
    if head_dim is None:
        _qkv_separate_map = _dense_key_map
    else:
        _qkv_separate_map = {"weight": ("kernel", lambda x: x.T.reshape(x.shape[1], -1, head_dim))}
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
        else:
            return leaf.__repr__()

    # Apply custom_format to every leaf in the pytree
    formatted_tree = jax.tree.map(custom_format, obj)

    # Serialize as JSON for pretty printing
    pretty_json = json.dumps(formatted_tree, indent=2)
    print(pretty_json)


def load_llama_from_hf(name: str) -> tuple["LlamaForCausalLM", dict]:
    """Load Huggingface llama compatible models directly from either local path
    or the hf-hub identifier."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e
    

    from jaxml.config import ModelConfig
    from jaxml.models.llama import LlamaForCausalLM

    _model = AutoModelForCausalLM.from_pretrained(name)
    _state_dict = _model.state_dict()
    params = torch_to_jax_states(_state_dict, head_dim=_model.config.hidden_size // _model.config.num_attention_heads)
    config = ModelConfig.from_hf(_model.config)
    model = LlamaForCausalLM(config)
    return model, params

