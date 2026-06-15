import jax
import jax.numpy as jnp
import numpy as np
import pytest


MODEL_FIXTURES = ["llama_model", "neox_model", "gemma_model"]


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
@pytest.mark.parametrize(
    "input_ids,exception,match",
    [
        (jnp.arange(4, dtype=jnp.int32), ValueError, "input_ids must be a 2D array"),
        (jnp.ones((1, 2, 1), dtype=jnp.int32), ValueError, "input_ids must be a 2D array"),
        (jnp.ones((1, 4), dtype=jnp.float32), TypeError, "integer token ids"),
        (jnp.ones((1, 0), dtype=jnp.int32), ValueError, "at least one token"),
    ],
)
def test_models_reject_invalid_input_ids(request, fixture_name, input_ids, exception, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)

        with pytest.raises(exception, match=match):
            model.apply(params, input_ids)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
@pytest.mark.parametrize(
    "attention_mask,exception,match",
    [
        (jnp.ones((4,), dtype=bool), ValueError, "attention_mask must be a 2D array"),
        (jnp.ones((1, 2, 1), dtype=bool), ValueError, "attention_mask must be a 2D array"),
        (jnp.ones((1, 3), dtype=bool), ValueError, "attention_mask shape must match"),
        (jnp.ones((1, 4), dtype=jnp.float32), TypeError, "attention_mask must be boolean or integer"),
    ],
)
def test_models_reject_invalid_attention_mask(request, fixture_name, attention_mask, exception, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(exception, match=match):
            model.apply(params, input_ids, attention_mask=attention_mask)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_canonicalize_integer_attention_mask(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]
        integer_mask = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)
        bool_mask = integer_mask.astype(bool)

        output = model.apply(params, input_ids, attention_mask=integer_mask)
        expected = model.apply(params, input_ids, attention_mask=bool_mask)

    assert np.allclose(output.last_hidden_state, expected.last_hidden_state, atol=1e-6)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_wrong_number_of_kv_caches(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(ValueError, match="one cache per layer"):
            model.apply(params, input_ids, kv_caches=(), use_cache=True)
