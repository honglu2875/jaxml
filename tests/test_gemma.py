import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jaxml.hf_utils import to_gemma_jax_params
from jaxml.inference_engine.engine import Engine, InferenceConfig
from jaxml.utils import torch_to_jax_states


@pytest.mark.parametrize("sliding", (True, False))
def test_gemma_decoder(gemma_decoder, hf_gemma_decoder, hf_gemma_decoder_global, cos_sin_factory, sliding: bool):
    bs, seq_len = 4, 10
    cos_sin_local = cos_sin_factory("llama")
    cos_sin_global = cos_sin_factory("gemma")
    hf_decoder = hf_gemma_decoder if sliding else hf_gemma_decoder_global
    with jax.default_device(jax.devices("cpu")[0]):
        decoder, init_param = gemma_decoder

        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (bs, seq_len, 64), dtype=jnp.float32)

        decoder.use_sliding = sliding
        cos_sin = cos_sin_local if sliding else cos_sin_global
        params = torch_to_jax_states(hf_decoder, head_dim=decoder.head_dim, dtype=torch.float32)
        y = decoder.apply(init_param | {"params": params["params"]}, x, cos_sin, output_attentions=True)
        with torch.no_grad():
            y2 = hf_decoder(
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
                position_embeddings_global=tuple(map(lambda x: torch.tensor(np.array(x[None, :seq_len])), cos_sin_global)),
                position_embeddings_local=tuple(map(lambda x: torch.tensor(np.array(x[None, :seq_len])), cos_sin_local)),
                output_attentions=True,
                cache_position=torch.arange(seq_len)[None],
            )

        print(np.abs(y.hidden_states - y2[0].numpy()).max())
        assert np.allclose(y.hidden_states, y2[0].numpy(), atol=1e-5)
        assert np.allclose(y.attention_weight, y2[1].numpy(), atol=1e-5)


def test_gemma_model(gemma_model, hf_gemma_model):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        model, init_param = gemma_model
        key = jax.random.PRNGKey(0)
        x = jax.random.randint(key, (bs, seq_len), 0, model.config.vocab_size - 1, dtype=jnp.int32)
        params = to_gemma_jax_params(hf_gemma_model, dtype="float32")
        y = model.apply({**init_param, "params": params["params"]}, x, output_attentions=True, output_hidden_states=True)
        with torch.no_grad():
            y2 = hf_gemma_model(
                torch.tensor(np.array(x)),
                output_attentions=True,
                output_hidden_states=True,
            )
        assert np.allclose(y.last_hidden_state, y2.last_hidden_state.numpy(), atol=1e-5)
        assert all(np.allclose(a, b.numpy(), atol=1e-5) for a, b in zip(y.attention_weights, y2.attentions))
        print("attention weight diffs:", [np.abs(a - b.numpy()).max() for a, b in zip(y.attention_weights, y2.attentions)])
        # last layer appears to have a bump in error... strange
        print("hidden state diffs:", [np.abs(a - b.numpy()).max() for a, b in zip(y.hidden_states, y2.hidden_states)])
        assert all(np.allclose(a, b.numpy(), atol=1e-5) for a, b in zip(y.hidden_states, y2.hidden_states))


def test_gemma_completion(gemma_model_with_head, hf_gemma_causal_model):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        model, init_param = gemma_model_with_head
        key = jax.random.PRNGKey(0)
        x = jax.random.randint(key, (bs, seq_len), 0, model.config.vocab_size - 1, dtype=jnp.int32)
        params = to_gemma_jax_params(hf_gemma_causal_model, dtype="float32")

        config = InferenceConfig()
        engine = Engine(model, config, init_param | {"params": params["params"]})
        engine.init_params(use_tpu=False)
        y = engine.generate(x, max_new_tokens=10, temperature=0.0)
        with torch.no_grad():
            y2 = hf_gemma_causal_model.generate(
                input_ids=torch.tensor(np.array(x)),
                max_new_tokens=10,
                do_sample=False,
            )
        print(y, y2)
        assert np.all(y == y2.numpy())
