import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.inference_engine.engine import Engine, InferenceConfig


@pytest.mark.parametrize(
    "max_new_tokens,include_prompt,cache_stride,fuse_decoding,expected_length",
    [
        (10, True, 256, False, 20),
        (10, False, 8, False, 10),
        (3, True, 8, True, 13),
        (1, False, 256, False, 1),
        (0, True, 256, False, 10),
        (0, False, 256, False, 0),
    ],
)
def test_engine_generate_token_count_contract(
    llama_model_with_head,
    max_new_tokens,
    include_prompt,
    cache_stride,
    fuse_decoding,
    expected_length,
):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (bs, seq_len), 0, model.config.vocab_size - 1, dtype=jnp.int32)

        engine = Engine(model, InferenceConfig(), params, cache_stride=cache_stride)
        engine.init_params(use_tpu=False)
        output = engine.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            include_prompt=include_prompt,
            fuse_decoding=fuse_decoding,
        )

    assert output.shape == (bs, expected_length)
    if include_prompt:
        assert np.array_equal(np.array(output[:, :seq_len]), np.array(input_ids))


def test_engine_generate_rejects_negative_max_new_tokens(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.ones((1, 4), dtype=jnp.int32)

        with pytest.raises(ValueError, match="max_new_tokens"):
            engine.generate(input_ids, max_new_tokens=-1)
