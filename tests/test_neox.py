import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxml.config import ModelConfig
from jaxml.hf_utils import to_neox_jax_params
from jaxml.inference_engine.engine import Engine, InferenceConfig
from jaxml.models.gpt_neox import GPTNeoXModel

NEOX_PARITY_ATOL = 1e-5


def test_neox_model(neox_model, hf_neox_model):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        model, init_param = neox_model
        key = jax.random.PRNGKey(0)
        x = jax.random.randint(key, (bs, seq_len), 0, model.config.vocab_size - 1, dtype=jnp.int32)
        params = to_neox_jax_params(hf_neox_model, dtype=torch.float32)
        y = model.apply({**init_param, "params": params["params"]}, x, output_attentions=True, output_hidden_states=True)
        with torch.no_grad():
            y2 = hf_neox_model(
                torch.tensor(np.array(x)),
                output_attentions=True,
                output_hidden_states=True,
            )
        assert np.allclose(y.last_hidden_state, y2.last_hidden_state.numpy(), atol=NEOX_PARITY_ATOL)
        assert all(np.allclose(a, b.numpy(), atol=NEOX_PARITY_ATOL) for a, b in zip(y.attention_weights, y2.attentions))
        assert all(np.allclose(a, b.numpy(), atol=NEOX_PARITY_ATOL) for a, b in zip(y.hidden_states, y2.hidden_states))


def test_neox_model_sanitizes_default_masked_negative_token_ids(neox_model):
    with jax.default_device(jax.devices("cpu")[0]):
        model, init_param = neox_model
        input_ids = jnp.array([[-100, 1, 2, 3], [4, -100, 5, 6]], dtype=jnp.int32)
        attention_mask = input_ids >= 0
        sanitized = jnp.where(attention_mask, input_ids, 0)

        y = model.apply(init_param, input_ids)
        y2 = model.apply(init_param, sanitized, attention_mask=attention_mask)

    assert np.all(np.isfinite(np.array(y.last_hidden_state)))
    assert np.allclose(y.last_hidden_state, y2.last_hidden_state, atol=1e-6)


def test_neox_final_norm_respects_model_dtype():
    config = ModelConfig(
        hidden_size=4,
        head_dim=2,
        num_heads=2,
        num_layers=1,
        intermediate_ratio=(2, 1),
        max_position_embeddings=8,
        vocab_size=16,
        attn_scale=2**-0.5,
        use_rope=False,
        use_bias=True,
    )
    model = GPTNeoXModel(config, dtype=jnp.bfloat16)
    input_ids = jnp.array([[0, 1]], dtype=jnp.int32)

    params = model.init(jax.random.PRNGKey(0), input_ids)

    assert params["params"]["norm"]["weight"].value.dtype == jnp.bfloat16
    assert params["params"]["norm"]["bias"].value.dtype == jnp.bfloat16


def test_neox_completion(neox_model_with_head, hf_neox_causal_model):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        model, init_param = neox_model_with_head
        key = jax.random.PRNGKey(0)
        x = jax.random.randint(key, (bs, seq_len), 0, model.config.vocab_size - 1, dtype=jnp.int32)
        params = to_neox_jax_params(hf_neox_causal_model, dtype=torch.float32)

        config = InferenceConfig()
        engine = Engine(model, config, {**init_param, "params": params["params"]})
        engine.init_params(use_tpu=False)
        y = engine.generate(x, max_new_tokens=10, temperature=0.0)
        with torch.no_grad():
            y2 = torch.tensor(np.array(x))
            for _ in range(10):
                logits = hf_neox_causal_model(input_ids=y2).logits[:, -1]
                next_tokens = logits.argmax(dim=-1, keepdim=True)
                y2 = torch.cat((y2, next_tokens), dim=-1)
        print(y, y2)
        assert np.all(y == y2.numpy())
