import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.cache import KVCache

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
def test_models_reject_attention_mask_without_valid_tokens(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.tile(jnp.arange(4, dtype=jnp.int32)[None], (2, 1))
        attention_mask = jnp.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)

        with pytest.raises(ValueError, match="at least one valid token"):
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


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_non_sequence_kv_caches(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(TypeError, match="kv_caches must be a sequence"):
            model.apply(params, input_ids, kv_caches=KVCache.init(4), use_cache=True)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_invalid_kv_cache_entries(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]
        kv_caches = (None,) * (model.num_layers - 1) + (object(),)

        with pytest.raises(TypeError, match="KVCache instances or None"):
            model.apply(params, input_ids, kv_caches=kv_caches, use_cache=True)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"use_cache": 1}, "use_cache must be a boolean"),
        ({"use_cache": None}, "use_cache must be a boolean"),
        ({"output_attentions": 1}, "output_attentions must be a boolean"),
        ({"output_hidden_states": "true"}, "output_hidden_states must be a boolean"),
    ],
)
def test_models_reject_non_boolean_output_flags(request, fixture_name, kwargs, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(TypeError, match=match):
            model.apply(params, input_ids, **kwargs)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_accept_numpy_boolean_output_flags(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        output = model.apply(
            params,
            input_ids,
            use_cache=np.bool_(True),
            output_attentions=np.bool_(False),
            output_hidden_states=np.bool_(True),
        )

    assert output.kv_caches is not None
    assert output.hidden_states is not None
    assert output.attention_weights is None
