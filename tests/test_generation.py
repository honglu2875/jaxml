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


def test_engine_generate_accepts_unbatched_prompt_tokens(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.arange(4, dtype=jnp.int32)

        output = engine.generate(input_ids, max_new_tokens=0, include_prompt=True)

    assert output.shape == (1, 4)
    assert np.array_equal(np.array(output[0]), np.array(input_ids))


def test_engine_generate_accepts_unbatched_attention_mask(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.arange(4, dtype=jnp.int32)
        attention_mask = jnp.ones((4,), dtype=bool)

        output = engine.generate(input_ids, attention_mask=attention_mask, max_new_tokens=0, include_prompt=True)

    assert output.shape == (1, 4)


@pytest.mark.parametrize(
    "input_ids,match",
    [
        (jnp.ones((1, 2, 3), dtype=jnp.int32), "prompt_tokens must be a 1D or 2D array"),
        (jnp.ones((1, 0), dtype=jnp.int32), "at least one token"),
    ],
)
def test_engine_generate_rejects_invalid_prompt_tokens(llama_model_with_head, input_ids, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(ValueError, match=match):
            engine.generate(input_ids, max_new_tokens=0)


def test_engine_generate_rejects_mismatched_attention_mask(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.ones((2, 4), dtype=jnp.int32)
        attention_mask = jnp.ones((1, 4), dtype=bool)

        with pytest.raises(ValueError, match="attention_mask shape must match"):
            engine.generate(input_ids, attention_mask=attention_mask, max_new_tokens=0)


@pytest.mark.parametrize(
    "input_ids,attention_mask",
    [
        (jnp.ones((2, 4), dtype=jnp.int32), jnp.array([[True, False, False, False], [False, False, False, False]])),
        (jnp.ones((4,), dtype=jnp.int32), jnp.zeros((4,), dtype=bool)),
    ],
)
def test_engine_generate_rejects_attention_mask_without_valid_tokens(llama_model_with_head, input_ids, attention_mask):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(ValueError, match="at least one valid token"):
            engine.generate(input_ids, attention_mask=attention_mask, max_new_tokens=0)


def test_engine_rejects_non_positive_cache_stride(llama_model_with_head):
    model, params = llama_model_with_head

    with pytest.raises(ValueError, match="cache_stride"):
        Engine(model, InferenceConfig(), params, cache_stride=0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tp_size": 0}, "tp_size"),
        ({"tp_size": -1}, "tp_size"),
        ({"dp_size": 0}, "dp_size"),
        ({"dp_size": -1}, "dp_size"),
    ],
)
def test_inference_config_rejects_non_positive_mesh_sizes(kwargs, match):
    with pytest.raises(ValueError, match=match):
        InferenceConfig(**kwargs)


def test_engine_rejects_mesh_larger_than_available_devices(llama_model_with_head):
    model, params = llama_model_with_head
    config = InferenceConfig(tp_size=jax.device_count() + 1)

    with pytest.raises(ValueError, match="requires .* devices"):
        Engine(model, config, params)
