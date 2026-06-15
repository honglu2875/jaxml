import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "fixture_name",
    ["llama_model_with_head", "neox_model_with_head", "gemma_model_with_head"],
)
@pytest.mark.parametrize(
    "keep_last_n_logits,expected_length",
    [
        (0, 4),
        (2, 2),
        (np.int64(1), 1),
    ],
)
def test_model_heads_keep_last_n_logits_shape(request, fixture_name, keep_last_n_logits, expected_length):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        output = model.apply(params, input_ids, keep_last_n_logits=keep_last_n_logits)

    assert output.logits.shape == (1, expected_length, model.config.vocab_size)


@pytest.mark.parametrize(
    "fixture_name",
    ["llama_model_with_head", "neox_model_with_head", "gemma_model_with_head"],
)
@pytest.mark.parametrize(
    "keep_last_n_logits,exception,match",
    [
        (True, TypeError, "keep_last_n_logits must be an integer"),
        (1.5, TypeError, "keep_last_n_logits must be an integer"),
        (-1, ValueError, "keep_last_n_logits must be non-negative"),
    ],
)
def test_model_heads_reject_invalid_keep_last_n_logits(request, fixture_name, keep_last_n_logits, exception, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(exception, match=match):
            model.apply(params, input_ids, keep_last_n_logits=keep_last_n_logits)
