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
import torch

from jaxml.utils import torch_to_jax_states
from jaxml.cache import KVCache


def test_llama_mlp(llama_mlp, hf_llama_mlp):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        mlp, _ = llama_mlp
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (bs, seq_len, 48), dtype=jnp.float32)
        params = torch_to_jax_states(hf_llama_mlp, dtype=torch.float32)
        y = mlp.apply(params, x)
        with torch.no_grad():
            y2 = hf_llama_mlp(torch.tensor(np.array(x))).numpy()

        assert np.allclose(y, y2, atol=1e-5)


def test_llama_decoder(llama_decoder, hf_llama_decoder):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        decoder, init_param = llama_decoder
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (bs, seq_len, 48), dtype=jnp.float32)
        params = torch_to_jax_states(hf_llama_decoder, head_dim=decoder.head_dim, dtype=torch.float32)
        y = decoder.apply({**init_param, "params": params["params"]}, x, output_attentions=True)
        with torch.no_grad():
            y2 = hf_llama_decoder(
                torch.tensor(np.array(x)),
                position_ids=torch.arange(seq_len)[None],
                attention_mask=torch.triu(
                    torch.full(
                        (seq_len, seq_len),
                        fill_value=float("-inf"),
                        dtype=torch.float32,
                    ),
                    diagonal=1,
                )[None, None].repeat(bs, 1, 1, 1),
                output_attentions=True,
            )
        
        assert np.allclose(y.hidden_states, y2[0].numpy(), atol=1e-5)
        assert np.allclose(y.attention_weight, y2[1].numpy(), atol=1e-5)

def test_llama_model(llama_model, hf_llama_model):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        model, init_param = llama_model
        key = jax.random.PRNGKey(0)
        x = jax.random.randint(key, (bs, seq_len), 0, model.config.vocab_size - 1, dtype=jnp.int32)
        params = torch_to_jax_states(hf_llama_model, head_dim=model.head_dim, dtype=torch.float32)
        y = model.apply({**init_param, "params": params["params"]}, x, output_attentions=True, output_hidden_states=True)
        with torch.no_grad():
            y2 = hf_llama_model(
                torch.tensor(np.array(x)),
                output_attentions=True,
                output_hidden_states=True,
            )
        
        assert np.allclose(y.last_hidden_state, y2.last_hidden_state.numpy(), atol=1e-5)
        #print(len(y.attention_weights), [type(t) for t in y2[1]])
        assert all(np.allclose(a, b.numpy(), atol=1e-5) for a, b in zip(y.attention_weights, y2.attentions))
        assert all(np.allclose(a, b.numpy(), atol=1e-5) for a, b in zip(y.hidden_states, y2.hidden_states))

def test_llama_causal_model(llama_causal_model, hf_llama_causal_model):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        model, init_param = llama_causal_model
        key = jax.random.PRNGKey(0)
        x = jax.random.randint(key, (bs, seq_len), 0, model.config.vocab_size - 1, dtype=jnp.int32)
        params = torch_to_jax_states(hf_llama_causal_model, head_dim=model.head_dim, dtype=torch.float32)
        y = model.generate({**init_param, "params": params["params"]}, x, max_new_tokens=10, temperature=0.0)
        with torch.no_grad():
            y2 = hf_llama_causal_model.generate(
                input_ids=torch.tensor(np.array(x)),
                max_new_tokens=10,
                do_sample=False,
            )
        assert np.all(y == y2.numpy())

def test_kv_cache(llama_model):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        model, init_param = llama_model

        key = jax.random.PRNGKey(0)
        x = jax.random.randint(key, (bs, seq_len), 0, model.config.vocab_size - 1, dtype=jnp.int32)

        kv_caches = [KVCache.init(20, None, None, dtype=jnp.float32) for _ in range(model.config.num_layers)]
        y = model.apply(init_param, x, output_attentions=True, output_hidden_states=True, use_cache=True, kv_caches=kv_caches)
        kv_caches = y.kv_caches

        new_caches = tuple(c.rollback(1) for c in kv_caches)
        y2 = model.apply(init_param, x[:, -1:], output_attentions=True, output_hidden_states=True, use_cache=True, kv_caches=new_caches)

        assert jnp.allclose(y.last_hidden_state[:, seq_len - 1 :seq_len], y2.last_hidden_state, atol=1e-6)