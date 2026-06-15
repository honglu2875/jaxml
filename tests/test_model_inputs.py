import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.cache import KVCache
from jaxml.config import ModelConfig
from jaxml.models._utils import prepare_model_inputs, prepare_position_ids
from jaxml.models.gemma3 import GemmaModel
from jaxml.models.gpt_neox import GPTNeoXModel
from jaxml.models.llama import LlamaModel

MODEL_FIXTURES = ["llama_model", "neox_model", "gemma_model"]


def _small_model_config(**overrides):
    kwargs = {
        "hidden_size": 8,
        "head_dim": 4,
        "num_heads": 2,
        "num_layers": 1,
        "intermediate_ratio": (2, 1),
        "max_position_embeddings": 8,
        "vocab_size": 16,
        "attn_scale": 4**-0.5,
        "num_kv_heads": 2,
    } | overrides
    return ModelConfig(**kwargs)


def test_prepare_model_inputs_accepts_numpy_integer_num_layers():
    input_ids, attention_mask, kv_caches = prepare_model_inputs(
        jnp.ones((1, 2), dtype=jnp.int32),
        None,
        (None,),
        num_layers=np.int64(1),
    )

    assert input_ids.shape == (1, 2)
    assert attention_mask is None
    assert kv_caches == (None,)


@pytest.mark.parametrize("num_layers", [True, np.bool_(True), 1.5])
def test_prepare_model_inputs_rejects_non_integer_num_layers(num_layers):
    with pytest.raises(TypeError, match="num_layers must be an integer"):
        prepare_model_inputs(jnp.ones((1, 2), dtype=jnp.int32), None, None, num_layers=num_layers)


@pytest.mark.parametrize("num_layers", [0, -1])
def test_prepare_model_inputs_rejects_non_positive_num_layers(num_layers):
    with pytest.raises(ValueError, match="num_layers must be positive"):
        prepare_model_inputs(jnp.ones((1, 2), dtype=jnp.int32), None, None, num_layers=num_layers)


def test_prepare_position_ids_accepts_none():
    input_ids = jnp.ones((2, 4), dtype=jnp.int32)

    assert prepare_position_ids(None, input_ids) is None


def test_prepare_position_ids_accepts_broadcast_batch_axis():
    input_ids = jnp.ones((2, 4), dtype=jnp.int32)
    position_ids = jnp.arange(4, dtype=jnp.int32)[None]

    prepared = prepare_position_ids(position_ids, input_ids)

    assert prepared.shape == (1, 4)


@pytest.mark.parametrize(
    "model_cls,config",
    [
        (LlamaModel, _small_model_config(use_rope=True)),
        (GPTNeoXModel, _small_model_config(use_rope=True, use_bias=True)),
        (
            GemmaModel,
            _small_model_config(
                hidden_size=8,
                use_rope=True,
                sliding_window=4,
                sliding_window_pattern=2,
            ),
        ),
    ],
)
def test_models_propagate_dtype_to_rotary_embedding_cache(model_cls, config):
    model = model_cls(config, dtype=jnp.bfloat16)
    input_ids = jnp.array([[0, 1]], dtype=jnp.int32)

    params = model.init(jax.random.PRNGKey(0), input_ids)

    assert params["cache"]["rotary_emb"]["cos_cached"].dtype == jnp.bfloat16
    assert params["cache"]["rotary_emb"]["sin_cached"].dtype == jnp.bfloat16
    if model_cls is GemmaModel:
        assert params["cache"]["rotary_emb_local"]["cos_cached"].dtype == jnp.bfloat16
        assert params["cache"]["rotary_emb_local"]["sin_cached"].dtype == jnp.bfloat16


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
@pytest.mark.parametrize(
    "position_ids,exception,match",
    [
        (jnp.arange(4, dtype=jnp.int32), ValueError, "position_ids must be a 2D array"),
        (jnp.ones((1, 4), dtype=jnp.float32), TypeError, "integer positions"),
        (jnp.ones((1, 3), dtype=jnp.int32), ValueError, "position_ids shape must be broadcastable"),
        (jnp.array([[-1, 0, 1, 2]], dtype=jnp.int32), ValueError, "non-negative positions"),
    ],
)
def test_models_reject_invalid_position_ids(request, fixture_name, position_ids, exception, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(exception, match=match):
            model.apply(params, input_ids, position_ids=position_ids)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_accept_broadcast_position_ids(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]
        position_ids = jnp.arange(4, dtype=jnp.int32)[None]

        output = model.apply(params, input_ids, position_ids=position_ids)

    assert output.last_hidden_state.shape[:2] == input_ids.shape


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
