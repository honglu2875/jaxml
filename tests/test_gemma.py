import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jaxml.config import ModelConfig
from jaxml.hf_utils import to_gemma_jax_params
from jaxml.inference_engine.engine import Engine, InferenceConfig
from jaxml.models.gemma3 import GemmaDecoder
from jaxml.utils import torch_to_jax_states

pytestmark = pytest.mark.milestone


def test_gemma_decoder_rejects_disabled_rope():
    config = ModelConfig(
        hidden_size=64,
        head_dim=8,
        num_heads=4,
        num_layers=1,
        intermediate_ratio=(2, 1),
        max_position_embeddings=16,
        vocab_size=128,
        num_kv_heads=2,
        attn_scale=8**-0.5,
        use_rope=False,
    )
    decoder = GemmaDecoder(config=config, dtype=jnp.float32)
    hidden_states = jnp.ones((1, 2, config.hidden_size), dtype=jnp.float32)

    with pytest.raises(ValueError, match="use_rope=True"):
        decoder.init(jax.random.PRNGKey(0), hidden_states)


@pytest.mark.parametrize("sliding", (True, False))
def test_gemma_decoder(gemma_decoder, hf_gemma_decoder, hf_gemma_decoder_global, cos_sin_factory, sliding: bool):
    bs, seq_len = 4, 10
    cos_sin_local = cos_sin_factory("llama")
    cos_sin_global = cos_sin_factory("gemma")
    hf_decoder = hf_gemma_decoder if sliding else hf_gemma_decoder_global
    with jax.default_device(jax.devices("cpu")[0]):
        decoder, init_param = gemma_decoder

        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (bs, seq_len, decoder.config.hidden_size), dtype=jnp.float32)

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
                position_embeddings=tuple(map(lambda x: torch.tensor(np.array(x[None, :seq_len])), cos_sin)),
                cache_position=torch.arange(seq_len)[None],
            )

        assert np.allclose(y.hidden_states, y2.numpy(), atol=1e-5)


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
        assert all(np.allclose(a, b.numpy(), atol=1e-5) for a, b in zip(y.hidden_states, y2.hidden_states))


def test_gemma_model_sanitizes_default_masked_negative_token_ids(gemma_model):
    with jax.default_device(jax.devices("cpu")[0]):
        model, init_param = gemma_model
        input_ids = jnp.array([[-100, 1, 2, 3], [4, -100, 5, 6]], dtype=jnp.int32)
        attention_mask = input_ids >= 0
        sanitized = jnp.where(attention_mask, input_ids, 0)

        y = model.apply(init_param, input_ids)
        y2 = model.apply(init_param, sanitized, attention_mask=attention_mask)

    assert np.all(np.isfinite(np.array(y.last_hidden_state)))
    assert np.allclose(y.last_hidden_state, y2.last_hidden_state, atol=1e-6)


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
            y2 = torch.tensor(np.array(x))
            for _ in range(10):
                logits = hf_gemma_causal_model(input_ids=y2).logits[:, -1]
                next_tokens = logits.argmax(dim=-1, keepdim=True)
                y2 = torch.cat((y2, next_tokens), dim=-1)
        assert np.all(y == y2.numpy())
