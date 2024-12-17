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
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention, BlockSizes
import numpy as np
import pytest
import torch

from jaxml.test_utils.torch_utils import DummyPosEmb
from jaxml.utils import torch_to_jax_states


@pytest.mark.parametrize("with_rope", [False, True])
def test_attention_with_rope(attention_with_rope_small, attention_small, hf_attention_with_rope, with_rope):
    with jax.default_device(jax.devices("cpu")[0]):
        bs, seq_len = 4, 10
        if with_rope:
            attn, init_param = attention_with_rope_small
        else:
            attn, init_param = attention_small
            hf_attention_with_rope.rotary_emb = DummyPosEmb()

        params = torch_to_jax_states(hf_attention_with_rope, head_dim=attn.head_dim, dtype=torch.float32)

        key = jax.random.PRNGKey(0)

        hidden_size = attn.config.hidden_size
        hidden = jax.random.uniform(key, (bs, seq_len, hidden_size), dtype=jnp.float32)

        # When RoPE is available, a "cache" field would exist in init_param which is needed for fwd pass
        out = attn.apply({**init_param, "params": params["params"]}, hidden)
        out = out.attention_output
        #out_flash = attn.apply({**init_param, "params": params["params"]}, hidden, use_flash=True)
        #out_flash = out_flash.attention_output
        with torch.no_grad():
            out2, _, _ = hf_attention_with_rope(
                torch.tensor(np.array(hidden)),
                position_ids=torch.arange(seq_len)[None],
                attention_mask=torch.triu(
                    torch.full(
                        (seq_len, seq_len),
                        fill_value=float("-inf"),
                        dtype=torch.float32,
                    ),
                    diagonal=1,
                )[None, None].repeat(bs, 1, 1, 1),
            )

        assert out.shape == out2.shape
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
