#  Copyright 2024 Honglu Fan
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes, flash_attention

from jaxml.hf_utils import to_llama_jax_params, to_neox_jax_params
from jaxml.test_utils.torch_utils import DummyPosEmb


@pytest.mark.parametrize(
    "model_type,with_rope",
    [
        ("llama", False),
        ("llama", True),
        ("neox", False),
        ("neox", True),
    ],
)
def test_attention_with_rope(attention_factory, model_type, with_rope, cos_sin_factory):
    with jax.default_device(jax.devices("cpu")[0]):
        bs, seq_len = 4, 10
        cos_sin = cos_sin_factory(model_type)
        hf, (attn, init_param) = attention_factory(model_type=model_type, with_rope=with_rope)
        if not with_rope:
            hf.rotary_emb = DummyPosEmb()

        match model_type:
            case "llama":
                params = to_llama_jax_params(hf, dtype="float32")
            case "neox":
                params = to_neox_jax_params(hf, dtype="float32")
            case _:
                raise

        key = jax.random.PRNGKey(0)

        hidden_size = attn.config.hidden_size
        hidden = jax.random.uniform(key, (bs, seq_len, hidden_size), dtype=jnp.float32)

        # When RoPE is available, a "cache" field would exist in init_param which is needed for fwd pass
        output = attn.apply({**init_param, "params": params["params"]}, hidden, cos_sin=cos_sin, output_attentions=True)
        out = output.attention_output
        # out_flash = attn.apply({**init_param, "params": params["params"]}, hidden, use_flash=True)
        # out_flash = out_flash.attention_output
        kwargs = {
            "hidden_states": torch.tensor(np.array(hidden)),
            "position_ids": torch.arange(seq_len)[None],
            "attention_mask": torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    fill_value=float("-inf"),
                    dtype=torch.float32,
                ),
                diagonal=1,
            )[None, None].repeat(bs, 1, 1, 1),
            "output_attentions": True,
        }
        #if model_type == "neox":
        rotary_dim = int(attn.config.head_dim * attn.config.rotary_pct)
        if with_rope:
            kwargs["position_embeddings"] = tuple(
                map(lambda x: torch.tensor(np.array(x[None, :seq_len, :rotary_dim])), cos_sin)
            )
        else:
            kwargs["position_embeddings"] = (
                torch.ones((1, seq_len, rotary_dim), dtype=torch.float32),
                torch.zeros((1, seq_len, rotary_dim), dtype=torch.float32),
            )
        with torch.no_grad():
            output2 = hf(**kwargs)
            out2 = output2[0]

        assert out.shape == out2.shape
        #print(np.abs(out - out2.numpy()).max())
        assert np.allclose(out, out2.numpy(), atol=1e-5)


"""
@pytest.mark.parametrize("with_rope", [False, True])
@pytest.mark.parametrize("seq_len", [1, 60, 497])
def test_flash_attention(attention_with_rope_small, attention_small, with_rope, seq_len):
    bs = 4
    if with_rope:
        attn, init_param = attention_with_rope_small
    else:
        attn, init_param = attention_small

    params = init_param

    key = jax.random.PRNGKey(0)

    hidden_size = attn.config.hidden_size
    hidden = jax.random.uniform(key, (bs, seq_len, hidden_size), dtype=jnp.float32)

    out = attn.apply({**init_param, "params": params["params"]}, hidden)
    out = out.attention_output
    out_flash = attn.apply({**init_param, "params": params["params"]}, hidden, use_flash=True)
    out_flash = out_flash.attention_output

    # Eager and JAX FA somehow have a bit of discrepancy, though still tolerable for inference
    assert jnp.allclose(out, out_flash, atol=1e-2)
"""
