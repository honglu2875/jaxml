import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.models._utils import slice_last_n_logits_hidden_states

pytestmark = pytest.mark.milestone


def test_slice_last_n_logits_hidden_states_rejects_invalid_rank():
    hidden_states = jnp.ones((1, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="3D array"):
        slice_last_n_logits_hidden_states(hidden_states, keep_last_n_logits=1)


def test_slice_last_n_logits_hidden_states_rejects_non_floating_hidden_states():
    hidden_states = jnp.ones((1, 4, 8), dtype=jnp.int32)

    with pytest.raises(TypeError, match="floating point"):
        slice_last_n_logits_hidden_states(hidden_states, keep_last_n_logits=1)


@pytest.mark.parametrize(
    "hidden_states",
    [
        jnp.ones((0, 4, 8), dtype=jnp.float32),
        jnp.ones((1, 0, 8), dtype=jnp.float32),
        jnp.ones((1, 4, 0), dtype=jnp.float32),
    ],
)
def test_slice_last_n_logits_hidden_states_rejects_empty_axes(hidden_states):
    with pytest.raises(ValueError, match="empty axes"):
        slice_last_n_logits_hidden_states(hidden_states, keep_last_n_logits=1)


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, -jnp.inf])
def test_slice_last_n_logits_hidden_states_rejects_non_finite_values(value):
    hidden_states = jnp.ones((1, 4, 8), dtype=jnp.float32).at[0, 0, 0].set(value)

    with pytest.raises(ValueError, match="hidden_states must contain only finite values"):
        slice_last_n_logits_hidden_states(hidden_states, keep_last_n_logits=1)


def test_slice_last_n_logits_hidden_states_accepts_traced_logits_positions():
    hidden_states = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3)
    gather = jax.jit(
        lambda positions: slice_last_n_logits_hidden_states(
            hidden_states,
            keep_last_n_logits=0,
            logits_positions=positions,
        )
    )

    selected = gather(jnp.array([3, 1], dtype=jnp.int32))

    assert selected.shape == (2, 1, 3)
    assert np.array_equal(selected[:, 0], hidden_states[jnp.array([0, 1]), jnp.array([3, 1])])


@pytest.mark.parametrize(
    "logits_positions,exception,match",
    [
        (jnp.array(1, dtype=jnp.int32), ValueError, "1D array"),
        (jnp.ones((2, 1), dtype=jnp.int32), ValueError, "1D array"),
        (jnp.array([1], dtype=jnp.int32), ValueError, "one position per batch row"),
        (jnp.array([1.0, 2.0], dtype=jnp.float32), TypeError, "integer positions"),
        (jnp.array([-1, 2], dtype=jnp.int32), ValueError, "non-negative positions"),
        (jnp.array([1, 4], dtype=jnp.int32), ValueError, "less than sequence length 4"),
    ],
)
def test_slice_last_n_logits_hidden_states_rejects_invalid_logits_positions(logits_positions, exception, match):
    hidden_states = jnp.ones((2, 4, 3), dtype=jnp.float32)

    with pytest.raises(exception, match=match):
        slice_last_n_logits_hidden_states(
            hidden_states,
            keep_last_n_logits=0,
            logits_positions=logits_positions,
        )


def test_slice_last_n_logits_hidden_states_rejects_positions_with_last_n_slice():
    hidden_states = jnp.ones((2, 4, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="keep_last_n_logits must be 0"):
        slice_last_n_logits_hidden_states(
            hidden_states,
            keep_last_n_logits=1,
            logits_positions=jnp.array([1, 2], dtype=jnp.int32),
        )


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
def test_model_heads_logits_positions_match_full_logits_for_right_padded_batch(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.array([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=jnp.int32)
        attention_mask = jnp.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)
        logits_positions = jnp.array([3, 1], dtype=jnp.int32)

        full = model.apply(params, input_ids, attention_mask=attention_mask).logits
        selected = model.apply(
            params,
            input_ids,
            attention_mask=attention_mask,
            logits_positions=logits_positions,
        ).logits

    assert selected.shape == (2, 1, model.config.vocab_size)
    assert np.allclose(selected[:, 0], full[jnp.arange(2), logits_positions])


@pytest.mark.parametrize(
    "fixture_name",
    ["llama_model_with_head", "neox_model_with_head", "gemma_model_with_head"],
)
@pytest.mark.parametrize(
    "keep_last_n_logits,exception,match",
    [
        (True, TypeError, "keep_last_n_logits must be an integer"),
        (np.bool_(True), TypeError, "keep_last_n_logits must be an integer"),
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


@pytest.mark.parametrize(
    "fixture_name",
    ["llama_model_with_head", "neox_model_with_head", "gemma_model_with_head"],
)
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"use_cache": 1}, "use_cache must be a boolean"),
        ({"use_cache": None}, "use_cache must be a boolean"),
        ({"output_attentions": 1}, "output_attentions must be a boolean"),
        ({"output_hidden_states": "true"}, "output_hidden_states must be a boolean"),
    ],
)
def test_model_heads_reject_non_boolean_output_flags(request, fixture_name, kwargs, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(TypeError, match=match):
            model.apply(params, input_ids, **kwargs)


@pytest.mark.parametrize(
    "fixture_name",
    ["llama_model_with_head", "neox_model_with_head", "gemma_model_with_head"],
)
def test_model_heads_accept_numpy_boolean_output_flags(request, fixture_name):
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
